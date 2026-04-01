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
"""

from __future__ import annotations

import asyncio

import structlog

from app.celery_app import celery_app

logger = structlog.get_logger()

# Message templates (Spanish -- primary language for Vive Polanco leads)
_MSG_DOCS_TEMPLATE = (
    "Aquí te comparto la información de {building_name}:\n\n"
    "{doc_lines}\n\n"
    "¿Te gustaría agendar una visita o tienes alguna pregunta?"
)

_MSG_NO_DOCS = (
    "Muchas gracias por tu interés. En este momento no tenemos los "
    "documentos disponibles digitalmente, pero con gusto te los "
    "enviamos en breve. ¿Hay algún otro tema en que te podamos ayudar?"
)


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
) -> dict:
    """Fetch unsent building documents from Neo4j and deliver links via GHL.

    Args:
        contact_id: GHL contact ID used for message delivery.
        phone: Lead phone number used for Neo4j Lead lookup.
        building_name: Building name as returned by Claude (case-insensitive match).
        channel: GHL delivery channel (SMS, WhatsApp, etc.).
        trace_id: Trace ID inherited from the originating process_message call.

    Steps:
        1. Query Neo4j for documents not yet sent to this lead.
        2. If none found: send polite "not available" message.
        3. If found: format and send PDF links via GHL.
        4. Record :SENT_DOCUMENT relationships in Neo4j.
    """

    async def _run() -> dict:
        from app.repositories.base import close_driver, get_driver
        from app.repositories.building_repository import BuildingRepository
        from app.services.ghl_service import send_message

        driver = None
        try:
            driver = await get_driver()
            building_repo = BuildingRepository(driver)

            # Step 1: Fetch unsent documents for this building + lead
            try:
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
                )
                # Fail-open: send fallback message
                docs = []

            logger.info(
                "doc_delivery.docs_found",
                trace_id=trace_id,
                contact_id=contact_id,
                building_name=building_name,
                count=len(docs),
            )

            # Step 2 or 3: Compose message
            if not docs:
                message_text = _MSG_NO_DOCS
                logger.info(
                    "doc_delivery.no_docs_available",
                    trace_id=trace_id,
                    building_name=building_name,
                )
            else:
                doc_lines = "\n".join(
                    f"📄 {doc.get('name', 'Documento')}: {doc.get('url', '')}"
                    for doc in docs
                    if doc.get("url")
                )
                if not doc_lines:
                    # All docs had empty URLs -- treat as unavailable
                    message_text = _MSG_NO_DOCS
                    docs = []
                else:
                    message_text = _MSG_DOCS_TEMPLATE.format(
                        building_name=building_name,
                        doc_lines=doc_lines,
                    )

            # Send via GHL
            try:
                await send_message(
                    contact_id=contact_id,
                    message=message_text,
                    channel=channel,
                )
                logger.info(
                    "doc_delivery.message_sent",
                    trace_id=trace_id,
                    contact_id=contact_id,
                    channel=channel,
                )
            except Exception:
                logger.exception(
                    "doc_delivery.ghl_send_failed",
                    trace_id=trace_id,
                    contact_id=contact_id,
                    channel=channel,
                )
                # Fail-open: log and return; don't mark docs as sent
                return {
                    "status": "ghl_send_failed",
                    "trace_id": trace_id,
                    "docs_found": len(docs),
                }

            # Step 4: Record sent documents in Neo4j (only if GHL succeeded)
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
                    # Non-fatal: docs were delivered; Neo4j write is best-effort

            return {
                "status": "delivered" if docs else "no_docs",
                "trace_id": trace_id,
                "building_name": building_name,
                "docs_sent": len(docs),
            }

        finally:
            # Always close the driver so the next asyncio.run() gets a fresh
            # driver bound to its own event loop (prevents "Future attached to
            # a different loop" errors in Celery prefork workers).
            await close_driver()

    return asyncio.run(_run())
