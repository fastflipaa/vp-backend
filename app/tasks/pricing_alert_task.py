"""Pricing alert notification task.

Sends a notification to the team (Fernando/Lorena) when a lead asks about
pricing for a building that doesn't have verified pricing in Neo4j.

This is NOT a handoff — the AI continues qualifying. This just alerts the
team so they can follow up with the developer price list.

Design: mirrors handoff_notification_task.py pattern (fail-open, async, SMS).
"""

from __future__ import annotations

import asyncio

import structlog

from app.celery_app import celery_app

logger = structlog.get_logger()

# Fernando's GHL internal contact ID (same as handoff notification)
FERNANDO_CONTACT_ID = "NuXxb4oBIbbGuMhDVtqI"


def _build_pricing_alert(
    lead_name: str,
    phone: str,
    building: str,
    lead_message: str,
) -> str:
    """Build a concise pricing alert message."""
    msg_preview = lead_message[:150] if lead_message else ""
    return (
        "\U0001f4b0 Pricing solicitado\n"
        "\n"
        f"Lead: {lead_name}\n"
        f"Tel: {phone}\n"
        f"Edificio: {building}\n"
        "\n"
        "El lead pregunto por precios pero este edificio NO tiene pricing "
        "verificado en el sistema. Natalia le dijo que verificaria contigo.\n"
        "\n"
        f"Mensaje del lead: \"{msg_preview}\"\n"
        "\n"
        "Accion: Enviar price list al equipo para cargar al sistema, "
        "o responder al lead directamente con los precios."
    )


@celery_app.task(
    name="notifications.send_pricing_alert",
    bind=True,
    max_retries=0,
    acks_late=True,
    reject_on_worker_lost=True,
    queue="celery",
)
def send_pricing_alert(
    self,
    contact_id: str,
    phone: str,
    lead_name: str,
    building: str,
    lead_message: str,
    trace_id: str,
) -> dict:
    """Send a pricing alert to Fernando — NOT a handoff, just an FYI."""

    async def _run() -> dict:
        from app.services.ghl_service import send_message

        message = _build_pricing_alert(
            lead_name=lead_name,
            phone=phone,
            building=building,
            lead_message=lead_message,
        )

        try:
            await send_message(FERNANDO_CONTACT_ID, message, "SMS")
            logger.info(
                "pricing_alert.sent",
                trace_id=trace_id,
                lead_contact_id=contact_id,
                building=building,
                recipient=FERNANDO_CONTACT_ID,
            )
            return {"status": "sent", "building": building}
        except Exception:
            logger.exception(
                "pricing_alert.failed",
                trace_id=trace_id,
                lead_contact_id=contact_id,
                building=building,
            )
            return {"status": "failed"}

    return asyncio.run(_run())
