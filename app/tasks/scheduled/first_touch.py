"""First-touch guarantee -- prevents duplicate greetings from race conditions.

Provides two synchronous helper functions (NOT Celery tasks) that are
called by the processing pipeline:

  - ensure_first_touch(phone, trace_id) -> bool
    Redis SETNX guarantee: exactly one greeting per phone per 24 hours.

  - handle_first_touch_failure(contact_id, phone, payload, trace_id)
    Called when GHL delivery fails for a first-touch (GREETING state) message.
    Retries once after 30 seconds, then tags ``manual-review`` on second failure.

These functions use the module-level ``_redis_client`` which is initialized
by the ``worker_process_init`` signal. Each scheduled task module registers
its own init handler for safety.
"""

from __future__ import annotations

import asyncio

import redis
import structlog
from celery import signals

from app.config import settings

logger = structlog.get_logger()

# Per-worker Redis client (initialized on worker_process_init)
_redis_client: redis.Redis | None = None


@signals.worker_process_init.connect
def init_first_touch_worker(**kwargs) -> None:
    """Initialize per-worker Redis connection pool on process startup."""
    global _redis_client
    pool = redis.ConnectionPool.from_url(
        settings.redis_cache_url,
        max_connections=20,
        decode_responses=True,
    )
    _redis_client = redis.Redis(connection_pool=pool)
    logger.info("first_touch_worker_redis_initialized")


def ensure_first_touch(phone: str, trace_id: str) -> bool:
    """Guarantee exactly one first-touch greeting per phone per 24 hours.

    Uses Redis SETNX with 24-hour TTL. Returns True if this is the first
    touch (caller should proceed with greeting). Returns False if a greeting
    was already sent to this phone within 24 hours (caller should skip).

    Args:
        phone: The lead's phone number (used as dedup key).
        trace_id: For structured logging.

    Returns:
        True if first touch was claimed (proceed), False if duplicate (skip).
    """
    key = f"first_touch:{phone}"
    was_set = _redis_client.set(key, trace_id, nx=True, ex=86400)

    if was_set:
        logger.info("first_touch.claimed", phone=phone[-4:], trace_id=trace_id)
        return True
    else:
        logger.info("first_touch.duplicate", phone=phone[-4:], trace_id=trace_id)
        return False


def handle_first_touch_failure(
    contact_id: str,
    phone: str,
    payload: dict,
    trace_id: str,
) -> None:
    """Handle GHL delivery failure for a first-touch greeting message.

    Called when GHL delivery fails for a GREETING state message. Retries
    once after 30 seconds via process_message.apply_async(countdown=30).
    On second failure, tags the contact as ``manual-review`` in GHL.

    Args:
        contact_id: GHL contact ID for tagging.
        phone: Lead's phone number (used as retry counter key).
        payload: Original message payload for retry.
        trace_id: For structured logging and retry tracking.
    """
    key = f"first_touch_retry:{phone}"
    retry_count = _redis_client.incr(key)
    _redis_client.expire(key, 3600)

    if retry_count <= 1:
        # First failure: retry after 30 seconds
        from app.tasks.processing_task import process_message

        process_message.apply_async(
            args=[payload, trace_id],
            countdown=30,
        )
        logger.info(
            "first_touch.retry_scheduled",
            contact_id=contact_id,
            phone=phone[-4:],
            trace_id=trace_id,
            retry_count=retry_count,
        )
    else:
        # Second failure: tag for manual review
        from app.services.ghl_service import add_tag

        try:
            asyncio.run(add_tag(contact_id, "manual-review"))
        except Exception:
            logger.exception(
                "first_touch.manual_review_tag_failed",
                contact_id=contact_id,
            )

        logger.warning(
            "first_touch.manual_review",
            contact_id=contact_id,
            phone=phone[-4:],
            trace_id=trace_id,
            retry_count=retry_count,
        )
