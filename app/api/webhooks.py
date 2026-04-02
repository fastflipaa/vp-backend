"""Webhook endpoints for gate processing.

/webhooks/inbound  - Receives GHL webhooks (raw or n8n-forwarded), enqueues gate pipeline
/webhooks/outbound - Detects human agent activity, sets human locks
"""

from __future__ import annotations

import uuid

import structlog
from fastapi import APIRouter, Request
from fastapi.responses import JSONResponse

from app.services.phone_normalizer import normalize_phone

logger = structlog.get_logger()

router = APIRouter()

# Known building tags for auto-detection from contact tags
_BUILDING_TAGS = [
    "armani", "ritz", "thompson", "virreyes", "polanco park", "be grand",
    "punto polanco", "delano", "brickell", "saint regis", "one park", "park hyatt",
]


def _normalize_ghl_payload(raw: dict) -> dict | None:
    """Normalize a raw GHL webhook payload to match InboundPayload schema.

    Handles both raw GHL format and pre-transformed n8n format.
    Returns None if the message should be skipped (outbound, no phone, etc.).
    Mirrors the logic from the n8n 'Extract Message Data' code node.
    """
    body = raw.get("body", raw) if isinstance(raw.get("body"), dict) else raw
    cd = body.get("customData", {})

    # Extract fields with GHL → InboundPayload mapping
    raw_phone = (
        body.get("phone") or body.get("contactPhone")
        or cd.get("Contact Phone") or cd.get("Phone")
        or body.get("from") or ""
    )
    message = (
        cd.get("Message Body") or body.get("body") or body.get("message")
        or body.get("text") or body.get("messageBody") or ""
    )
    # If message is a dict (nested GHL format), extract .body
    if isinstance(message, dict):
        message = message.get("body", "")

    contact_id = body.get("contactId") or body.get("contact_id") or cd.get("Contact ID") or ""
    location_id = (
        body.get("locationId") or body.get("location_id")
        or (body.get("location", {}).get("id") if isinstance(body.get("location"), dict) else "")
        or cd.get("Location ID") or ""
    )
    conversation_id = body.get("conversationId") or body.get("conversation_id") or cd.get("Conversation ID") or ""
    channel = cd.get("Response Channel") or body.get("type") or body.get("messageType") or body.get("channel") or "unknown"
    direction = body.get("direction", "inbound")
    contact_name = (
        body.get("contactName") or body.get("contact_name")
        or body.get("name") or body.get("full_name")
        or cd.get("Contact Name") or ""
    )
    email = body.get("email") or body.get("contactEmail") or ""

    # Tags: may be string, list, or absent
    tags_raw = body.get("tags", [])
    if isinstance(tags_raw, str):
        tags = [t.strip() for t in tags_raw.split(",") if t.strip()]
    elif isinstance(tags_raw, list):
        tags = tags_raw
    else:
        tags = []

    # Auto-trigger detection
    is_auto_trigger = (
        body.get("aiTextStatus") == "auto_trigger"
        or direction == "auto_trigger"
        or message == "NEW_LEAD_NO_MESSAGE"
        or body.get("messageType") == "auto_trigger"
    )

    # Skip outbound (non-auto-trigger)
    is_outbound = direction == "outbound" and not is_auto_trigger
    if is_outbound:
        return None

    # Normalize phone
    phone = normalize_phone(raw_phone)
    # Try nested contact object if phone still empty
    if not phone and isinstance(body.get("contact"), dict):
        phone = normalize_phone(body["contact"].get("phone", ""))
    if not phone and not contact_id:
        return None  # No identifiers at all

    # Building detection from tags
    building_source = body.get("buildingSource") or body.get("building") or "unknown"
    if building_source in ("unknown", None):
        tag_str = ",".join(tags).lower()
        for bt in _BUILDING_TAGS:
            if bt in tag_str:
                building_source = bt
                break

    return {
        "phone": phone or "",
        "message": "" if is_auto_trigger else message,
        "contactId": contact_id,
        "locationId": location_id,
        "conversationId": conversation_id,
        "channel": channel,
        "contactName": contact_name,
        "email": email,
        "buildingSource": building_source,
        "tags": tags,
        "direction": direction,
        "isAutoTrigger": is_auto_trigger,
        "messageType": body.get("messageType", ""),
        "messageId": body.get("messageId") or body.get("id") or "",
    }


@router.post("/webhooks/inbound")
async def shadow_inbound(request: Request) -> JSONResponse:
    """Accept an inbound webhook payload and enqueue it for gate processing.

    Handles both raw GHL webhook format and pre-transformed n8n payloads.
    Returns 200 immediately with a trace_id. The actual gate processing
    happens asynchronously in a Celery task.
    """
    trace_id = str(uuid.uuid4())

    try:
        raw_payload = await request.json()
    except Exception:
        logger.warning("invalid_json_body", trace_id=trace_id)
        return JSONResponse(
            status_code=400,
            content={"status": "error", "detail": "invalid JSON body"},
        )

    # Normalize GHL fields → InboundPayload format
    payload = _normalize_ghl_payload(raw_payload)
    if payload is None:
        logger.info("webhook_skipped", trace_id=trace_id, reason="outbound_or_no_phone")
        return JSONResponse(
            status_code=200,
            content={"status": "skipped", "trace_id": trace_id, "reason": "outbound_or_no_identifiers"},
        )

    logger.info(
        "webhook_received",
        trace_id=trace_id,
        contactId=payload.get("contactId", ""),
        messageId=payload.get("messageId", ""),
        channel=payload.get("channel", ""),
    )

    # Synchronous canary check so n8n knows whether to skip its own processing.
    # Uses config + tags only (no Redis needed for the routing decision).
    from app.services.canary_router import should_route_canary

    tags = payload.get("tags", [])
    is_canary = should_route_canary(tags, None)

    # Lazy import to avoid circular imports (celery_app -> config -> settings)
    from app.tasks.gate_tasks import process_gates_shadow

    process_gates_shadow.delay(payload, trace_id)

    return JSONResponse(
        status_code=200,
        content={"status": "accepted", "trace_id": trace_id, "is_canary": is_canary},
    )


@router.post("/webhooks/outreach")
async def outreach_inbound(request: Request) -> JSONResponse:
    """Alias for /webhooks/inbound — handles AI Outreach (first touch, re-engagement).

    GHL 'AI Outreach Webhook' custom value points here. Same pipeline as inbound.
    """
    return await shadow_inbound(request)


@router.post("/webhooks/call")
async def call_inbound(request: Request) -> JSONResponse:
    """Alias for /webhooks/inbound — handles AI Call webhook events.

    GHL 'AI Call Webhook' custom value points here. Same pipeline as inbound.
    """
    return await shadow_inbound(request)


@router.post("/webhooks/outbound")
async def outbound_hook(request: Request) -> JSONResponse:
    """Process outbound webhook events to detect human agent activity.

    When a human agent sends a message, sets a human lock on the contact
    to prevent AI from responding while the human is active.
    """
    try:
        payload = await request.json()
    except Exception:
        return JSONResponse(
            status_code=400,
            content={"status": "error", "detail": "invalid JSON body"},
        )

    contact_id = payload.get("contactId", "")
    source = payload.get("source", "")
    agent_info = payload.get("agentName", source or "unknown_agent")

    # Detect human agent activity (non-bot, non-system sources)
    is_human_activity = source not in ("", "bot", "system", "automation", "workflow")

    if is_human_activity and contact_id:
        # Use sync Redis for simplicity (lock setting is not latency-critical)
        from app.gates.human_lock import set_human_lock
        from app.dependencies import get_sync_redis

        redis_client = get_sync_redis()
        set_human_lock(contact_id, agent_info, redis_client)
        logger.info(
            "human_lock_set_via_outbound",
            contact_id=contact_id,
            agent_info=agent_info,
        )

    return JSONResponse(
        status_code=200,
        content={"status": "processed"},
    )
