"""Rate limit gate - sliding window rate limiter using Redis sorted sets.

Implements SAFE-13 dual-cap rate limiting:
1. Daily cap: 15 outbound messages per 24h rolling window (sorted set)
2. Unreplied cap: 3 consecutive outbound messages without reply (simple counter)

Redis sorted sets give a true sliding window (not fixed-window counters).
record_outbound/record_inbound helpers manage counters atomically via pipelines.
"""

from __future__ import annotations

import time

import redis
import structlog

from app.gates.base import GateDecision, GateResult

logger = structlog.get_logger()

UNREPLIED_CAP = 3  # consecutive outbound without reply
DAILY_CAP = 15  # outbound per 24h rolling window
DAILY_WINDOW_SECONDS = 86400  # 24 hours


def check_rate_limit(
    payload: dict, trace_id: str, redis_client: redis.Redis
) -> GateResult:
    """Check rate limits for the contact: daily cap and unreplied cap.

    Daily cap uses a Redis sorted set with timestamp scores for true
    sliding-window enforcement. Unreplied cap uses a simple counter
    that resets on any inbound message from the contact.

    Returns PASS if both limits are within bounds, BLOCK otherwise.
    Messages without a contactId are passed through.
    """
    contact_id = payload.get("contactId", "")

    if not contact_id:
        return GateResult(
            gate_name="rate_limit",
            decision=GateDecision.PASS,
            reason="no_contact_id_skip",
            duration_ms=0.0,
        )

    now = time.time()
    cutoff = now - DAILY_WINDOW_SECONDS
    daily_key = f"rate_limit:daily:{contact_id}"

    # Check 1 - Daily cap (sliding window via sorted set)
    pipe = redis_client.pipeline(transaction=True)
    pipe.zremrangebyscore(daily_key, "-inf", cutoff)
    pipe.zcard(daily_key)
    pipe.expire(daily_key, DAILY_WINDOW_SECONDS)
    results = pipe.execute()

    daily_count = results[1]

    if daily_count >= DAILY_CAP:
        logger.info(
            "rate_limit_daily_cap_blocked",
            contact_id=contact_id,
            daily_count=daily_count,
            daily_cap=DAILY_CAP,
            trace_id=trace_id,
        )
        return GateResult(
            gate_name="rate_limit",
            decision=GateDecision.BLOCK,
            reason=f"daily_cap_exceeded:{daily_count}/{DAILY_CAP}",
            duration_ms=0.0,
            metadata={"contact_id": contact_id, "daily_count": daily_count},
        )

    # Check 2 - Unreplied outbound cap
    unreplied_key = f"rate_limit:unreplied:{contact_id}"
    unreplied_raw = redis_client.get(unreplied_key)
    unreplied_count = int(unreplied_raw) if unreplied_raw else 0

    if unreplied_count >= UNREPLIED_CAP:
        logger.info(
            "rate_limit_unreplied_cap_blocked",
            contact_id=contact_id,
            unreplied_count=unreplied_count,
            unreplied_cap=UNREPLIED_CAP,
            trace_id=trace_id,
        )
        return GateResult(
            gate_name="rate_limit",
            decision=GateDecision.BLOCK,
            reason=f"unreplied_cap_exceeded:{unreplied_count}/{UNREPLIED_CAP}",
            duration_ms=0.0,
            metadata={"contact_id": contact_id, "unreplied_count": unreplied_count},
        )

    # Both checks passed
    return GateResult(
        gate_name="rate_limit",
        decision=GateDecision.PASS,
        reason=f"within_limits:daily={daily_count}/{DAILY_CAP},unreplied={unreplied_count}/{UNREPLIED_CAP}",
        duration_ms=0.0,
    )


def record_outbound(contact_id: str, redis_client: redis.Redis) -> None:
    """Record an outbound message for rate limiting.

    Atomically:
    1. Adds current timestamp to the daily sorted set (sliding window)
    2. Sets TTL on the daily key (24h)
    3. Increments the unreplied counter
    4. Sets TTL on the unreplied key (24h)
    """
    now = time.time()
    timestamp_str = str(now)
    daily_key = f"rate_limit:daily:{contact_id}"
    unreplied_key = f"rate_limit:unreplied:{contact_id}"

    pipe = redis_client.pipeline(transaction=True)
    pipe.zadd(daily_key, {timestamp_str: now})
    pipe.expire(daily_key, DAILY_WINDOW_SECONDS)
    pipe.incr(unreplied_key)
    pipe.expire(unreplied_key, DAILY_WINDOW_SECONDS)
    pipe.execute()

    logger.debug(
        "rate_limit_outbound_recorded",
        contact_id=contact_id,
    )


def record_inbound(contact_id: str, redis_client: redis.Redis) -> None:
    """Record an inbound message - resets the unreplied counter.

    Any inbound message from a contact resets the consecutive-unreplied
    count to zero, as the lead has engaged.
    """
    unreplied_key = f"rate_limit:unreplied:{contact_id}"
    redis_client.delete(unreplied_key)

    logger.debug(
        "rate_limit_inbound_recorded",
        contact_id=contact_id,
    )
