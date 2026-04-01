"""Document delivery task -- sends R2-hosted PDF links to leads.

When a lead asks for brochures, floor plans, or pricing, this task
fetches the relevant documents from Neo4j and sends the download links
via GHL message.

Design decisions:
- Fail-open: doc delivery failure should never block the conversation
- Uses asyncio.run() -- safe in prefork Celery workers
- Closes Neo4j driver in finally block to prevent event loop issues
- Records sent documents in Neo4j to avoid duplicate sends
"""

from __future__ import annotations

import asyncio

import structlog

from app.celery_app import celery_app

logger = structlog.get_logger()


def _format_doc_message(building_name: str, docs: list[dict], language: str = "es") -> str:
    """Format document links into a user-friendly message."""
    if language.startswith("en"):
        header = f"Here's the information for {building_name}:"
        footer = "Would you like to schedule a visit or do you have any questions?"
    else:
        header = f"Aqu\u00ed te comparto la informaci\u00f3n de {building_name}:"
        footer = "\u00bfTe gustar\u00eda agendar una visita o tienes alguna pregunta?"

    doc_lines = []
    for doc in docs[:5]:  # Max 5 docs per message
        name = doc.get("name", doc.get("type", "Documento"))
        url = doc.get("r2_url", doc.get("url", ""))
        if url:
            doc_lines.append(f"\U0001f4c4 {name}: {url}")

    if not doc_lines:
        if language.startswith("en"):
            return f"I'm looking for the documents for {building_name}. I'll send them as soon as they're available."
        return f"Estoy buscando los documentos de {building_name}. Te los env\u00edo en cuanto est\u00e9n disponibles."

    return f"{header}\n\n" + "\n".join(doc_lines) + f"\n\n{footer}"


@celery_app.task(
    name="documents.deliver",
    bind=True,
    max_retries=0,
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
) -> dict:
    """Fetch unsent docs from Neo4j and send links via GHL."""

    async def _run() -> dict:
        from app.repositories.base import close_driver, get_driver
        from app.repositories.building_repository import BuildingRepository
        from app.services.ghl_service import send_message

        try:
            driver = await get_driver()
            building_repo = BuildingRepository(driver)

            # Try to find docs for this building
            docs = []
            try:
                docs = await building_repo.get_unsent_docs(contact_id, building_name)
            except Exception:
                logger.exception(
                    "doc_delivery.neo4j_query_failed",
                    trace_id=trace_id,
                    building=building_name,
                )

            # Format and send message
            message = _format_doc_message(building_name, docs, language)

            try:
                await send_message(contact_id, message, channel)
                logger.info(
                    "doc_delivery.sent",
                    trace_id=trace_id,
                    contact_id=contact_id,
                    building=building_name,
                    doc_count=len(docs),
                )
            except Exception:
                logger.exception(
                    "doc_delivery.send_failed",
                    trace_id=trace_id,
                    contact_id=contact_id,
                )
                return {"status": "send_failed"}

            # Record sent docs in Neo4j (non-critical)
            if docs:
                try:
                    await building_repo.record_discussed(contact_id, building_name)
                except Exception:
                    logger.warning(
                        "doc_delivery.record_failed",
                        trace_id=trace_id,
                    )

            return {"status": "delivered", "doc_count": len(docs)}
        finally:
            await close_driver()

    try:
        return asyncio.run(_run())
    except Exception:
        logger.exception(
            "doc_delivery.task_failed",
            trace_id=trace_id,
            contact_id=contact_id,
            building=building_name,
        )
        return {"status": "error"}
