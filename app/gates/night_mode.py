"""Night mode gate - blocks proactive messages during CDMX night hours.

Implements SAFE-14 night mode:
- Night window: 10 PM to 8 AM Mexico City time
- Proactive (outbound-initiated) messages are BLOCKED during night hours
- Reactive (inbound-triggered) messages PASS through 24/7
- Blocked messages are queued in a Redis list for morning processing

Uses stdlib zoneinfo for CDMX timezone. Mexico abolished DST in Oct 2022,
so America/Mexico_City is permanent UTC-6.
"""

from __future__ import annotations

import json
from datetime import datetime
from zoneinfo import ZoneInfo

import redis
import structlog

from app.gates.base import GateDecision, GateResult

logger = structlog.get_logger()

CDMX_TZ = ZoneInfo("America/Mexico_City")
NIGHT_START_HOUR = 22  # 10 PM
NIGHT_END_HOUR = 8  # 8 AM
NIGHT_QUEUE_KEY = "night_queue"


def _is_night_hour(hour: int) -> bool:
    """Return True if the given hour falls within the night window."""
    return hour >= NIGHT_START_HOUR or hour < NIGHT_END_HOUR


def _get_cdmx_time(payload: dict) -> datetime:
    """Get the relevant CDMX time for this payload.

    If the payload contains a receivedAt or dateAdded timestamp, use that
    for consistent comparison with n8n's check. Otherwise use current time.
    """
    for field_name in ("receivedAt", "dateAdded"):
        ts = payload.get(field_name)
        if ts and isinstance(ts, str):
            try:
                # Parse ISO format timestamp and convert to CDMX
                dt = datetime.fromisoformat(ts.replace("Z", "+00:00"))
                return dt.astimezone(CDMX_TZ)
            except (ValueError, TypeError):
                pass  # Fall through to datetime.now()

    return datetime.now(tz=CDMX_TZ)


def check_night_mode(
    payload: dict, trace_id: str, redis_client: redis.Redis
) -> GateResult:
    """Check if the message should be blocked due to night hours.

    Proactive messages (non-inbound) are blocked during the CDMX night
    window (10 PM - 8 AM) and queued for morning delivery. Reactive
    messages (direction == "inbound") always pass through.

    Returns PASS during daytime or for reactive messages, BLOCK for
    proactive messages during night hours.
    """
    cdmx_now = _get_cdmx_time(payload)
    hour = cdmx_now.hour
    time_str = cdmx_now.strftime("%H:%M")
    direction = payload.get("direction", "")
    contact_id = payload.get("contactId", "")

    # Daytime - everything passes
    if not _is_night_hour(hour):
        return GateResult(
            gate_name="night_mode",
            decision=GateDecision.PASS,
            reason=f"daytime:{time_str} CDMX",
            duration_ms=0.0,
        )

    # Night + reactive (inbound-triggered) - always allowed
    if direction == "inbound":
        return GateResult(
            gate_name="night_mode",
            decision=GateDecision.PASS,
            reason=f"night_reactive_allowed:{time_str} CDMX",
            duration_ms=0.0,
        )

    # Night + proactive - block and queue for morning
    queue_entry = json.dumps({
        "payload": payload,
        "trace_id": trace_id,
        "queued_at": cdmx_now.isoformat(),
    })
    redis_client.rpush(NIGHT_QUEUE_KEY, queue_entry)

    logger.info(
        "night_mode_blocked",
        contact_id=contact_id,
        cdmx_hour=hour,
        cdmx_time=time_str,
        trace_id=trace_id,
    )

    return GateResult(
        gate_name="night_mode",
        decision=GateDecision.BLOCK,
        reason=f"night_proactive_blocked:{time_str} CDMX, queued for morning",
        duration_ms=0.0,
        metadata={"contact_id": contact_id, "cdmx_hour": hour},
    )
