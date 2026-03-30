"""Canary routing logic for Phase 17 migration.

Determines whether a contact should be processed end-to-end by FastAPI
(canary mode) or remain in shadow-only mode (n8n handles processing).

Routing decision: contacts tagged with CANARY_TAG (default 'v3-canary')
AND CANARY_ENABLED=True get full FastAPI processing. All others stay in
shadow comparison mode.

Tracks canary processing stats in Redis for the /canary/status endpoint.
"""

from __future__ import annotations

import time

import structlog

from app.config import settings

logger = structlog.get_logger()


def should_route_canary(tags: list[str], redis_client) -> bool:
    """Check whether a contact should be routed through canary (full FastAPI).

    Returns True only when BOTH conditions are met:
    1. settings.CANARY_ENABLED is True (global kill switch)
    2. settings.CANARY_TAG is present in the contact's tags list

    Args:
        tags: List of GHL tags on the contact.
        redis_client: Redis client (unused currently, reserved for future
            percentage-based rollout or feature flags).

    Returns:
        True if the contact should be processed end-to-end by FastAPI.
    """
    if not settings.CANARY_ENABLED:
        return False
    return settings.CANARY_TAG in tags


def track_canary_result(
    redis_client,
    success: bool = True,
    latency_ms: float = 0,
) -> None:
    """Record a canary processing result in Redis for monitoring.

    Maintains both hourly-bucketed counters (for recent stats) and
    rolling 24h counters (for the /canary/status endpoint).

    Args:
        redis_client: Sync Redis client.
        success: Whether the canary processing succeeded.
        latency_ms: Total pipeline latency in milliseconds.
    """
    try:
        hour_bucket = int(time.time() // 3600)
        ttl_2h = 7200  # 2-hour overlapping window

        pipe = redis_client.pipeline()

        # Hourly bucketed counter
        count_key = f"canary:processed:count:{hour_bucket}"
        pipe.incr(count_key)
        pipe.expire(count_key, ttl_2h)

        if not success:
            err_key = f"canary:errors:count:{hour_bucket}"
            pipe.incr(err_key)
            pipe.expire(err_key, ttl_2h)

        # Latency tracking (keep last 100 per bucket)
        if latency_ms > 0:
            lat_key = f"canary:latencies:{hour_bucket}"
            pipe.rpush(lat_key, str(round(latency_ms, 2)))
            pipe.ltrim(lat_key, -100, -1)
            pipe.expire(lat_key, ttl_2h)

        pipe.execute()

        # Rolling 24h counters
        redis_client.incr("canary:processed:24h")
        # Set TTL only if not already set (avoid resetting on every increment)
        if redis_client.ttl("canary:processed:24h") == -1:
            redis_client.expire("canary:processed:24h", 86400)

        if not success:
            redis_client.incr("canary:errors:24h")
            if redis_client.ttl("canary:errors:24h") == -1:
                redis_client.expire("canary:errors:24h", 86400)

    except Exception:
        logger.exception("canary_tracking_error")


def get_canary_stats(redis_client) -> dict:
    """Retrieve canary processing statistics from Redis.

    Returns a dict with:
    - processed_24h: Total canary-processed messages in last 24h
    - errors_24h: Total canary errors in last 24h
    - error_rate: errors / processed (0.0 if no messages)
    - avg_latency_ms: Average latency from most recent hour bucket
    - shadow_count_24h: Total shadow-mode messages in last 24h

    Args:
        redis_client: Sync Redis client.

    Returns:
        Dict with canary stats. All values default to 0 on Redis errors.
    """
    try:
        processed = int(redis_client.get("canary:processed:24h") or 0)
        errors = int(redis_client.get("canary:errors:24h") or 0)
        shadow_count = int(redis_client.get("shadow:count:24h") or 0)

        # Compute average latency from current hour bucket
        hour_bucket = int(time.time() // 3600)
        lat_key = f"canary:latencies:{hour_bucket}"
        latencies_raw = redis_client.lrange(lat_key, 0, -1)
        if latencies_raw:
            latencies = [float(x) for x in latencies_raw]
            avg_latency = sum(latencies) / len(latencies)
        else:
            avg_latency = 0.0

        error_rate = (errors / processed) if processed > 0 else 0.0

        return {
            "processed_24h": processed,
            "errors_24h": errors,
            "error_rate": round(error_rate, 4),
            "avg_latency_ms": round(avg_latency, 2),
            "shadow_count_24h": shadow_count,
        }
    except Exception:
        logger.exception("canary_stats_error")
        return {
            "processed_24h": 0,
            "errors_24h": 0,
            "error_rate": 0.0,
            "avg_latency_ms": 0.0,
            "shadow_count_24h": 0,
        }
