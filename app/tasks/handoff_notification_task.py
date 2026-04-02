"""WhatsApp/SMS handoff notification task.

Sends a formatted notification to the human agent (Fernando) when the AI
hands off a lead. Runs on the default "celery" queue.

Design decisions:
- Fail-open: logs errors but never raises, so a notification failure never
  blocks the main processing pipeline.
- Uses asyncio.run() -- safe in prefork Celery workers (no existing loop).
- Sends via GHL SMS channel to Fernando's internal contact ID.  WhatsApp
  can be toggled later by changing the channel argument.
"""

from __future__ import annotations

import asyncio

import structlog

from app.celery_app import celery_app

logger = structlog.get_logger()

# Fernando Bello's GHL contact ID — DISABLED until correct ID is provided
# NuXxb4oBIbbGuMhDVtqI was Fernando Duran Romero (WRONG person)
# User will create Fernando Bello as a contact and provide the correct ID
FERNANDO_CONTACT_ID = ""  # Empty = notifications silently skip


def _build_message(
    lead_name: str,
    phone: str,
    handoff_reason: str,
    priority: str,
    building: str,
    context_summary: str,
) -> str:
    """Build the Spanish-language handoff notification message."""
    summary_snippet = context_summary[:300] if context_summary else ""
    return (
        "\U0001f514 Nuevo lead requiere atenci\u00f3n\n"
        "\n"
        f"Nombre: {lead_name}\n"
        f"Tel\u00e9fono: {phone}\n"
        f"Raz\u00f3n: {handoff_reason}\n"
        f"Prioridad: {priority}\n"
        f"Edificio: {building}\n"
        "\n"
        "Resumen:\n"
        f"{summary_snippet}"
    )


@celery_app.task(
    name="notifications.send_handoff_notification",
    bind=True,
    max_retries=0,
    acks_late=True,
    reject_on_worker_lost=True,
    queue="celery",
)
def send_handoff_notification(
    self,
    contact_id: str,
    phone: str,
    lead_name: str,
    handoff_reason: str,
    priority: str,
    context_summary: str,
    building: str,
    trace_id: str,
) -> dict:
    """Send a WhatsApp/SMS handoff notification to Fernando."""

    async def _run() -> dict:
        from app.services.ghl_service import send_message

        message = _build_message(
            lead_name=lead_name,
            phone=phone,
            handoff_reason=handoff_reason,
            priority=priority,
            building=building,
            context_summary=context_summary,
        )

        try:
            await send_message(FERNANDO_CONTACT_ID, message, "SMS")
            logger.info(
                "handoff_notification.sent",
                trace_id=trace_id,
                lead_contact_id=contact_id,
                recipient=FERNANDO_CONTACT_ID,
                priority=priority,
                reason=handoff_reason,
            )
            return {"status": "sent"}
        except Exception:
            logger.exception(
                "handoff_notification.failed",
                trace_id=trace_id,
                lead_contact_id=contact_id,
                recipient=FERNANDO_CONTACT_ID,
            )
            return {"status": "failed"}

    return asyncio.run(_run())
