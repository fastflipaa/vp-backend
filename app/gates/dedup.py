"""Dedup gate - blocks duplicate webhooks via Redis SETNX.

Uses messageId as the dedup key with a 15-minute TTL. The first webhook
with a given messageId passes; subsequent duplicates within the TTL are blocked.
"""

from __future__ import annotations

import redis
import structlog

from app.gates.base import GateDecision, GateResult

logger = structlog.get_logger()

DEDUP_TTL_SECONDS = 900  # 15 minutes


def check_dedup(payload: dict, trace_id: str, redis_client: redis.Redis) -> GateResult:
    """Check if this messageId has been seen within the dedup window.

    Returns PASS for new messages, BLOCK for duplicates.
    Messages without a messageId are passed through (no dedup possible).
    """
    message_id = payload.get("messageId", "")

    if not message_id:
        return GateResult(
            gate_name="dedup",
            decision=GateDecision.PASS,
            reason="no_message_id_skip",
            duration_ms=0.0,
        )

    key = f"dedup:{message_id}"
    was_set = redis_client.set(key, trace_id, nx=True, ex=DEDUP_TTL_SECONDS)

    if was_set:
        return GateResult(
            gate_name="dedup",
            decision=GateDecision.PASS,
            reason="new_message",
            duration_ms=0.0,
        )
    else:
        logger.info(
            "dedup_blocked",
            message_id=message_id,
            trace_id=trace_id,
        )
        return GateResult(
            gate_name="dedup",
            decision=GateDecision.BLOCK,
            reason=f"duplicate_message_id:{message_id}",
            duration_ms=0.0,
            metadata={"messageId": message_id},
        )
