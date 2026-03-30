"""Human lock gate - blocks AI processing when a human agent is active.

Uses Redis EXISTS for O(1) lock detection. Lock auto-expires after 24 hours.
Trigger word patterns on inbound messages can release the lock, allowing
the message through.
"""

from __future__ import annotations

import re

import redis
import structlog

from app.gates.base import GateDecision, GateResult

logger = structlog.get_logger()

HUMAN_LOCK_TTL_SECONDS = 86400  # 24 hours
HUMAN_LOCK_HISTORY_TTL_SECONDS = 172800  # 48 hours -- outlives lock for re-entry context

# Trigger word patterns that release the human lock (case-insensitive)
# Supports both English and Spanish trigger phrases
TRIGGER_WORD_PATTERNS: list[re.Pattern] = [
    re.compile(r"\b(resume|reanudar|continuar)\s*(bot|ai|asistente|chat)\b", re.IGNORECASE),
    re.compile(r"\b(activar|activate)\s*(bot|ai|asistente)\b", re.IGNORECASE),
    re.compile(r"\b(bot\s+on|encender\s+bot)\b", re.IGNORECASE),
]


def check_trigger_word_release(
    message: str, contact_id: str, redis_client: redis.Redis
) -> bool:
    """Check if the message contains a trigger word that releases the human lock.

    Returns True if a trigger word was found and the lock was released.
    """
    if not message or not contact_id:
        return False

    for pattern in TRIGGER_WORD_PATTERNS:
        match = pattern.search(message)
        if match:
            released = release_human_lock(contact_id, redis_client)
            logger.info(
                "trigger_word_lock_released",
                contact_id=contact_id,
                matched_pattern=match.group(0),
                lock_released=released,
            )
            return True

    return False


def check_human_lock(
    payload: dict, trace_id: str, redis_client: redis.Redis
) -> GateResult:
    """Check if a human agent lock is active for this contact.

    Returns PASS if no lock exists, BLOCK if a human agent is handling the contact.
    Inbound messages with trigger words release the lock and pass through.
    """
    contact_id = payload.get("contactId", "")
    message = payload.get("message", "")
    direction = payload.get("direction", "")

    if not contact_id:
        return GateResult(
            gate_name="human_lock",
            decision=GateDecision.PASS,
            reason="no_contact_id_skip",
            duration_ms=0.0,
        )

    # Trigger word release check (SAFE-12): inbound messages can release the lock
    if message and direction == "inbound":
        if check_trigger_word_release(message, contact_id, redis_client):
            return GateResult(
                gate_name="human_lock",
                decision=GateDecision.PASS,
                reason="human_lock_released_by_trigger_word",
                duration_ms=0.0,
                metadata={"contact_id": contact_id},
            )

    key = f"human_lock:{contact_id}"
    lock_exists = redis_client.exists(key)

    if lock_exists:
        lock_info = redis_client.get(key) or "unknown"
        logger.info(
            "human_lock_blocked",
            contact_id=contact_id,
            lock_info=lock_info,
            trace_id=trace_id,
        )
        return GateResult(
            gate_name="human_lock",
            decision=GateDecision.BLOCK,
            reason=f"human_agent_active:{contact_id}",
            duration_ms=0.0,
            metadata={"contact_id": contact_id, "lock_info": lock_info},
        )

    return GateResult(
        gate_name="human_lock",
        decision=GateDecision.PASS,
        reason="no_human_lock",
        duration_ms=0.0,
    )


def set_human_lock(
    contact_id: str, agent_info: str, redis_client: redis.Redis
) -> bool:
    """Set a human agent lock on a contact with 24-hour TTL.

    Returns True on success.
    """
    key = f"human_lock:{contact_id}"
    redis_client.set(key, agent_info, ex=HUMAN_LOCK_TTL_SECONDS)

    # Persist lock history for human re-entry context (48h TTL outlives 24h lock)
    history_key = f"human_lock_history:{contact_id}"
    redis_client.set(history_key, agent_info, ex=HUMAN_LOCK_HISTORY_TTL_SECONDS)

    logger.info(
        "human_lock_set",
        contact_id=contact_id,
        agent_info=agent_info,
        ttl_seconds=HUMAN_LOCK_TTL_SECONDS,
        history_ttl_seconds=HUMAN_LOCK_HISTORY_TTL_SECONDS,
    )
    return True


def release_human_lock(contact_id: str, redis_client: redis.Redis) -> bool:
    """Release the human agent lock on a contact.

    Returns True if a lock was actually deleted, False if none existed.
    """
    key = f"human_lock:{contact_id}"
    deleted = redis_client.delete(key) > 0
    if deleted:
        logger.info("human_lock_released", contact_id=contact_id)
    return deleted
