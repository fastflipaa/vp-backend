"""Out-of-band ground truth tracker for the AI processing pipeline.

Records two independent events:
  * Inbound webhook hits     -- ``monitor:gt:inbound`` (sorted set)
  * Successful AI deliveries -- ``monitor:gt:delivery`` (sorted set)

Both are Redis sorted sets keyed by ``trace_id`` with score = epoch ms.
``scheduled.ground_truth_check`` (running every 10 min) compares the
counts in the last 15-min window. If ``inbound > 0`` and
``delivery == 0`` for 3 consecutive checks -> CRITICAL alert.

REGRESSION CONTEXT
------------------
On 2026-04-05 to 2026-04-07, ``system_health_scan`` reported green for
46 hours while 100% of message processing failed silently. The in-band
scanner only counted errors that processing code explicitly registered
inside try/except blocks. The SM ``InvalidDefinition`` raised *before*
any such try/except, so the in-band counters never incremented and the
scanner saw zero errors. This module exists to make that failure mode
impossible to hide:

  * The webhook receiver records ``inbound`` BEFORE any downstream
    processing -- it cannot be skipped by a downstream raise.
  * ``response_delivery.deliver`` records ``delivery`` ONLY after GHL
    confirms message acceptance -- it cannot be faked by intermediate
    code.
  * Mismatch between the two is the smoking gun for pipeline death.

This is "ground truth" in the sense that it doesn't trust any in-band
metric -- it compares two independently-observed events.
"""

from __future__ import annotations

import time

import redis
import structlog

logger = structlog.get_logger()

# Sorted set keys -- score = epoch ms, member = trace_id
INBOUND_KEY = "monitor:gt:inbound"
DELIVERY_KEY = "monitor:gt:delivery"

# Keep entries for 2 hours (12 x the 10-min check window) so the
# scheduled checker can always find data even after worker restarts.
KEY_TTL_SECONDS = 7200

# Module-level lazy Redis client. Initialized on first use so this
# module can be imported safely without a live Redis connection.
_redis_client: redis.Redis | None = None


def _get_redis() -> redis.Redis | None:
    """Lazy init a Redis client pointing at the cache URL.

    Uses the same Redis DB as ``SystemHealthScanner`` (settings.redis_cache_url
    -> db2 by default). Returns None on connection failure rather than
    raising -- the tracker MUST NEVER break the webhook handler or the
    delivery pipeline.
    """
    global _redis_client
    if _redis_client is None:
        try:
            from app.config import settings

            _redis_client = redis.Redis.from_url(
                settings.redis_cache_url,
                decode_responses=True,
                socket_connect_timeout=2,
            )
        except Exception as e:
            logger.warning("ground_truth_tracker.redis_init_failed", error=str(e))
            return None
    return _redis_client


def record_inbound(trace_id: str) -> None:
    """Record an inbound webhook hit.

    MUST be called from the webhook receiver BEFORE any try/except so
    it always fires regardless of what happens downstream. Failures
    here are silently swallowed -- the tracker is observability, not
    a critical path.
    """
    if not trace_id:
        return
    client = _get_redis()
    if client is None:
        return
    try:
        score = time.time() * 1000
        client.zadd(INBOUND_KEY, {trace_id: score})
        client.expire(INBOUND_KEY, KEY_TTL_SECONDS)
    except Exception:
        # Observability code must NEVER break the webhook handler
        pass


def record_delivery(trace_id: str) -> None:
    """Record a successful AI message delivery to GHL.

    Called from ``ResponseDeliveryService.deliver`` AFTER GHL confirms
    message acceptance. NOT called for fallback deliveries (those are
    a separate degraded path).
    """
    if not trace_id:
        return
    client = _get_redis()
    if client is None:
        return
    try:
        score = time.time() * 1000
        client.zadd(DELIVERY_KEY, {trace_id: score})
        client.expire(DELIVERY_KEY, KEY_TTL_SECONDS)
    except Exception:
        pass


def get_window_counts(window_minutes: int = 15) -> tuple[int, int]:
    """Return ``(inbound_count, delivery_count)`` for the last window.

    Used by ``scheduled.ground_truth_check`` to compare ground truth.
    Returns ``(0, 0)`` on Redis failure -- the caller treats that as
    "no data" rather than as a successful zero count.
    """
    client = _get_redis()
    if client is None:
        return 0, 0
    cutoff_ms = (time.time() - window_minutes * 60) * 1000
    try:
        inbound = client.zcount(INBOUND_KEY, cutoff_ms, "+inf")
        delivery = client.zcount(DELIVERY_KEY, cutoff_ms, "+inf")
        return int(inbound), int(delivery)
    except Exception as e:
        logger.warning("ground_truth_tracker.window_query_failed", error=str(e))
        return 0, 0


def cleanup_old_entries(keep_minutes: int = 120) -> None:
    """Trim entries older than ``keep_minutes`` from both sorted sets.

    Called by the scheduled checker on each run to prevent unbounded
    growth. Failures are swallowed.
    """
    client = _get_redis()
    if client is None:
        return
    cutoff_ms = (time.time() - keep_minutes * 60) * 1000
    try:
        client.zremrangebyscore(INBOUND_KEY, "-inf", cutoff_ms)
        client.zremrangebyscore(DELIVERY_KEY, "-inf", cutoff_ms)
    except Exception:
        pass
