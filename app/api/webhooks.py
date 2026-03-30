"""Webhook endpoints for shadow mode gate processing.

/webhooks/inbound  - Receives forwarded payloads from n8n, enqueues gate pipeline
/webhooks/outbound - Detects human agent activity, sets human locks
"""

from __future__ import annotations

import uuid

import structlog
from fastapi import APIRouter, Request
from fastapi.responses import JSONResponse

from app.dependencies import get_async_redis

logger = structlog.get_logger()

router = APIRouter()


@router.post("/webhooks/inbound")
async def shadow_inbound(request: Request) -> JSONResponse:
    """Accept an inbound webhook payload and enqueue it for gate processing.

    Returns 200 immediately with a trace_id. The actual gate processing
    happens asynchronously in a Celery task.
    """
    trace_id = str(uuid.uuid4())

    try:
        payload = await request.json()
    except Exception:
        logger.warning("invalid_json_body", trace_id=trace_id)
        return JSONResponse(
            status_code=400,
            content={"status": "error", "detail": "invalid JSON body"},
        )

    logger.info(
        "shadow_webhook_received",
        trace_id=trace_id,
        contactId=payload.get("contactId", ""),
        messageId=payload.get("messageId", ""),
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
