"""Document delivery Celery task.

Sends R2-hosted PDF links to a lead via GHL when they request brochures,
floor plans, or pricing documents. Ports the Send Building Docs sub-workflow
from n8n (Section 3.7 of the audit).

Design decisions:
- queue="processing": collocated with the main processing pipeline workers
- max_retries=0: fail-open; a missed doc delivery is better than a stuck queue
- asyncio.run() + close_driver() in finally: same pattern as health_check.py
  to prevent "Future attached to a different loop" across Celery invocations
- Fail-open on Neo4j: if doc lookup fails we send a polite fallback message
- Fail-open on GHL: if send fails we log but do not crash the task
- Deduplication: handled upstream via :SENT_DOCUMENT relationships in Neo4j
- Language-aware: sends brochure matching lead's detected language
- Smart filtering: excludes banners, unbranded materials, wrong-language docs
"""

from __future__ import annotations

import asyncio

import structlog

from app.celery_app import celery_app

logger = structlog.get_logger()

# Message templates
_MSG_DOCS_TEMPLATE_ES = (
    "Aquí te comparto la información de {building_name}:\n\n"
    "{doc_lines}\n\n"
    "¿Te gustaría agendar una visita o tienes alguna pregunta?"
)

_MSG_DOCS_TEMPLATE_EN = (
    "Here's the information for {building_name}:\n\n"
    "{doc_lines}\n\n"
    "Would you like to schedule a visit or do you have any questions?"
)

_MSG_NO_DOCS_ES = (
    "Muchas gracias por tu interés. En este momento no tenemos los "
    "documentos disponibles digitalmente, pero con gusto te los "
    "enviamos en breve. ¿Hay algún otro tema en que te podamos ayudar?"
)

_MSG_NO_DOCS_EN = (
    "Thank you for your interest. We don't have the documents available "
    "digitally at the moment, but we'll send them to you shortly. "
    "Is there anything else we can help with?"
)

# Request-type-aware message templates
_MSG_FLOORPLAN_TEMPLATE_ES = (
    "Aqui te comparto los planos de {building_name}:"
)
_MSG_FLOORPLAN_TEMPLATE_EN = (
    "Here are the floor plans for {building_name}:"
)

_MSG_PHOTOS_TEMPLATE_ES = (
    "Aqui tienes las imagenes de {building_name}:"
)
_MSG_PHOTOS_TEMPLATE_EN = (
    "Here are the images for {building_name}:"
)

_MSG_PRICING_TEMPLATE_ES = (
    "Aqui te comparto la lista de precios de {building_name}:"
)
_MSG_PRICING_TEMPLATE_EN = (
    "Here's the pricing information for {building_name}:"
)

_MSG_BROCHURE_TEMPLATE_ES = (
    "Aqui te comparto el brochure de {building_name}:"
)
_MSG_BROCHURE_TEMPLATE_EN = (
    "Here's the brochure for {building_name}:"
)

# Map request_type -> (doc_type filter for Neo4j, ES template, EN template)
_REQUEST_TYPE_MAP: dict[str, tuple[str | None, str, str]] = {
    "floorplan": ("floorplan", _MSG_FLOORPLAN_TEMPLATE_ES, _MSG_FLOORPLAN_TEMPLATE_EN),
    "photos":    ("floorplan", _MSG_PHOTOS_TEMPLATE_ES, _MSG_PHOTOS_TEMPLATE_EN),
    "pricing":   ("price_list", _MSG_PRICING_TEMPLATE_ES, _MSG_PRICING_TEMPLATE_EN),
    "brochure":  ("brochure", _MSG_BROCHURE_TEMPLATE_ES, _MSG_BROCHURE_TEMPLATE_EN),
    "all":       (None, _MSG_DOCS_TEMPLATE_ES, _MSG_DOCS_TEMPLATE_EN),
}

# Document name patterns to EXCLUDE (case-insensitive)
_EXCLUDE_PATTERNS = [
    "banner",
    "rollup",
    "unbranded",
    "commission",
    "agent rate",
    "broker rate",
    "media strategy",
    "estrategia de medios",
]

# Language suffixes in filenames/URLs
_LANG_SUFFIXES = {
    "es": ["-es", "spanish", "espanol"],
    "en": ["-en", "english"],
    "pt": ["-pt", "portuguese", "portugues"],
}


def _filter_docs_for_lead(docs: list[dict], language: str) -> list[dict]:
    """Filter documents to send only relevant ones for this lead.

    Rules:
    1. Exclude banners, unbranded, commission sheets, media strategies
    2. For brochures: pick only the one matching lead's language
    3. Keep all factsheets, price lists, floor plans, renderings
    4. If no language-specific brochure exists, fall back to the default one
    """
    filtered = []
    brochures = []
    non_brochures = []

    for doc in docs:
        name_lower = (doc.get("name") or "").lower()
        url_lower = (doc.get("url") or "").lower()

        # Step 1: Exclude unwanted document types
        if any(pattern in name_lower or pattern in url_lower for pattern in _EXCLUDE_PATTERNS):
            continue

        # Step 2: Categorize as brochure vs non-brochure
        is_brochure = "brochure" in name_lower or "brochure" in url_lower
        if is_brochure:
            brochures.append(doc)
        else:
            non_brochures.append(doc)

    # Step 3: Pick the right brochure for this language
    if brochures:
        lang_suffixes = _LANG_SUFFIXES.get(language, [])

        # Try to find language-specific brochure
        lang_match = None
        default_brochure = None
        for b in brochures:
            name_lower = (b.get("name") or "").lower()
            url_lower = (b.get("url") or "").lower()
            combined = name_lower + " " + url_lower

            # Check if this is the lead's language
            if any(suffix in combined for suffix in lang_suffixes):
                lang_match = b
                break

            # Track the "default" brochure (no language suffix = usually English)
            has_any_lang_suffix = any(
                suffix in combined
                for suffixes in _LANG_SUFFIXES.values()
                for suffix in suffixes
            )
            if not has_any_lang_suffix:
                default_brochure = b

        # Pick: language match > default > first brochure
        chosen = lang_match or default_brochure or brochures[0]
        filtered.append(chosen)

    # Step 4: Add all non-brochure docs (factsheets, prices, floor plans, renderings)
    filtered.extend(non_brochures)

    return filtered


@celery_app.task(
    name="processing.deliver_documents",
    bind=True,
    max_retries=0,
    acks_late=True,
    reject_on_worker_lost=True,
    queue="processing",
)
def deliver_documents(
    self,
    contact_id: str,
    phone: str,
    building_name: str,
    channel: str,
    trace_id: str,
    language: str = "es",
    request_type: str = "all",
) -> dict:
    """Fetch unsent building documents from Neo4j and deliver links via GHL.

    Args:
        contact_id: GHL contact ID used for message delivery.
        phone: Lead phone number used for Neo4j Lead lookup.
        building_name: Building name as returned by Claude (case-insensitive match).
        channel: GHL delivery channel (SMS, WhatsApp, etc.).
        trace_id: Trace ID inherited from the originating process_message call.
        language: Lead's detected language (es, en, pt). Defaults to es.
        request_type: Type of documents requested: 'floorplan', 'photos',
                      'pricing', 'brochure', or 'all'. Defaults to 'all'.

    Steps:
        1. Query Neo4j for documents not yet sent to this lead.
        2. Filter: exclude banners/unbranded, pick language-matched brochure.
        3. If none found: send polite "not available" message.
        4. If found: send as media attachments via GHL (text fallback on error).
        5. Record :SENT_DOCUMENT relationships in Neo4j.
    """

    async def _run() -> dict:
        from app.repositories.base import close_driver, get_driver
        from app.repositories.building_repository import BuildingRepository
        from app.services.ghl_service import send_media_message, send_message

        driver = None
        try:
            driver = await get_driver()
            building_repo = BuildingRepository(driver)

            # Step 1: Fetch unsent documents for this building + lead
            try:
                # Resolve doc_type filter from request_type
                type_config = _REQUEST_TYPE_MAP.get(request_type, _REQUEST_TYPE_MAP["all"])
                doc_type_filter = type_config[0]

                if doc_type_filter:
                    docs = await building_repo.get_unsent_docs_by_name_and_type(
                        phone=phone,
                        building_name=building_name,
                        doc_type=doc_type_filter,
                    )
                else:
                    docs = await building_repo.get_unsent_docs_by_name(
                        phone=phone,
                        building_name=building_name,
                    )
            except Exception:
                logger.exception(
                    "doc_delivery.neo4j_lookup_failed",
                    trace_id=trace_id,
                    contact_id=contact_id,
                    building_name=building_name,
                    request_type=request_type,
                )
                docs = []

            # Step 2: Smart filter — language-match brochure, exclude junk
            if docs:
                docs = _filter_docs_for_lead(docs, language)

            logger.info(
                "doc_delivery.docs_found",
                trace_id=trace_id,
                contact_id=contact_id,
                building_name=building_name,
                count=len(docs),
                language=language,
                request_type=request_type,
            )

            # Step 3: Compose message
            is_english = language.startswith("en")
            type_config = _REQUEST_TYPE_MAP.get(request_type, _REQUEST_TYPE_MAP["all"])
            _, template_es, template_en = type_config

            if not docs:
                message_text = _MSG_NO_DOCS_EN if is_english else _MSG_NO_DOCS_ES
                attachment_urls: list[str] = []
                logger.info(
                    "doc_delivery.no_docs_available",
                    trace_id=trace_id,
                    building_name=building_name,
                    request_type=request_type,
                )
            else:
                # Collect attachment URLs for media message
                attachment_urls = [
                    doc["url"] for doc in docs if doc.get("url")
                ]
                template = template_en if is_english else template_es
                message_text = template.format(building_name=building_name)

                # If no valid URLs, fall back to no-docs message
                if not attachment_urls:
                    message_text = _MSG_NO_DOCS_EN if is_english else _MSG_NO_DOCS_ES
                    docs = []

            # Step 4: Send via GHL — try media attachments first, text fallback
            try:
                if attachment_urls:
                    # Try media message with inline attachments
                    try:
                        await send_media_message(
                            contact_id=contact_id,
                            message=message_text,
                            attachment_urls=attachment_urls,
                            channel=channel,
                        )
                        logger.info(
                            "doc_delivery.media_message_sent",
                            trace_id=trace_id,
                            contact_id=contact_id,
                            channel=channel,
                            docs_count=len(docs),
                            request_type=request_type,
                        )
                    except Exception:
                        # Media send failed — fall back to text-only with URLs
                        logger.warning(
                            "doc_delivery.media_send_failed_falling_back_to_text",
                            trace_id=trace_id,
                            contact_id=contact_id,
                        )
                        doc_lines = "\n".join(
                            f"\U0001f4c4 {doc.get('name', 'Document')}: {doc.get('url', '')}"
                            for doc in docs
                            if doc.get("url")
                        )
                        fallback_template = _MSG_DOCS_TEMPLATE_EN if is_english else _MSG_DOCS_TEMPLATE_ES
                        fallback_text = fallback_template.format(
                            building_name=building_name,
                            doc_lines=doc_lines,
                        )
                        await send_message(
                            contact_id=contact_id,
                            message=fallback_text,
                            channel=channel,
                        )
                        logger.info(
                            "doc_delivery.text_fallback_sent",
                            trace_id=trace_id,
                            contact_id=contact_id,
                        )
                else:
                    # No attachments (no-docs message) — send as plain text
                    await send_message(
                        contact_id=contact_id,
                        message=message_text,
                        channel=channel,
                    )
                    logger.info(
                        "doc_delivery.no_docs_message_sent",
                        trace_id=trace_id,
                        contact_id=contact_id,
                    )
            except Exception:
                logger.exception(
                    "doc_delivery.ghl_send_failed",
                    trace_id=trace_id,
                    contact_id=contact_id,
                    channel=channel,
                )
                return {
                    "status": "ghl_send_failed",
                    "trace_id": trace_id,
                    "docs_found": len(docs),
                }

            # Step 5: Record sent documents in Neo4j (only if GHL succeeded)
            if docs:
                sent_urls = [d["url"] for d in docs if d.get("url")]
                try:
                    await building_repo.record_sent_docs(
                        phone=phone,
                        doc_urls=sent_urls,
                    )
                    logger.info(
                        "doc_delivery.sent_docs_recorded",
                        trace_id=trace_id,
                        phone=phone[-4:] if phone else "",
                        count=len(sent_urls),
                    )
                except Exception:
                    logger.exception(
                        "doc_delivery.record_sent_failed",
                        trace_id=trace_id,
                        phone=phone[-4:] if phone else "",
                    )

            return {
                "status": "delivered" if docs else "no_docs",
                "trace_id": trace_id,
                "building_name": building_name,
                "docs_sent": len(docs),
                "language": language,
                "request_type": request_type,
            }

        finally:
            await close_driver()

    return asyncio.run(_run())
