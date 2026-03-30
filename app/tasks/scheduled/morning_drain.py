"""Morning queue drain task -- processes night-queued messages at 09:01 CDMX.

Runs once daily at 09:01 America/Mexico_City via Celery Beat (schedule
configured in Plan 17-03). Pops all entries from the Redis ``night_queue``
list (FIFO order) and routes each through ``process_gates_shadow.delay()``
so that ALL safety gates re-execute at delivery time.

CRITICAL SAFETY: Routes through process_gates_shadow, NOT process_message.
Night-queued items only passed the night_mode gate when they arrived.
At 09:01 drain time, other gates must re-execute because conditions may
have changed overnight:
  - human_lock: A human agent may have taken over the contact
  - rate_limit: The lead's daily send count may already be maxed
  - dedup: An identical message may have arrived via a different path

Uses time.sleep(5) between entries to avoid burst spam (synchronous task).
"""

from __future__ import annotations

import json
import time
import uuid

import redis
import structlog
from celery import signals

from app.celery_app import celery_app
from app.config import settings

logger = structlog.get_logger()

# Per-worker Redis client (initialized on worker_process_init)
_redis_client: redis.Redis | None = None


@signals.worker_process_init.connect
def init_morning_drain_worker(**kwargs) -> None:
    """Initialize per-worker Redis connection pool on process startup."""
    global _redis_client
    pool = redis.ConnectionPool.from_url(
        settings.redis_cache_url,
        max_connections=20,
        decode_responses=True,
    )
    _redis_client = redis.Redis(connection_pool=pool)
    logger.info("morning_drain_worker_redis_initialized")


@celery_app.task(name="scheduled.morning_drain", queue="celery")
def morning_drain() -> dict:
    """Drain the night queue and route each entry through the gate pipeline.

    Pops entries from Redis night_queue list (FIFO -- first queued = first
    drained). Each entry is routed through process_gates_shadow.delay() so
    all safety gates re-execute at delivery time. Includes a 5-second delay
    between entries to avoid burst spam.

    Returns dict with count of drained entries.
    """
    from app.gates.night_mode import NIGHT_QUEUE_KEY
    from app.tasks.gate_tasks import process_gates_shadow

    queue_length = _redis_client.llen(NIGHT_QUEUE_KEY)

    if queue_length == 0:
        logger.info("morning_drain.empty")
        return {"drained": 0}

    logger.info("morning_drain.start", queue_length=queue_length)

    drained_count = 0

    while True:
        raw_entry = _redis_client.lpop(NIGHT_QUEUE_KEY)
        if raw_entry is None:
            break

        try:
            entry = json.loads(raw_entry)
            payload = entry.get("payload", {})
            original_trace_id = entry.get("trace_id", "unknown")

            # Generate new trace_id for morning processing
            new_trace_id = str(uuid.uuid4())

            logger.info(
                "morning_drain.processing",
                original_trace_id=original_trace_id,
                new_trace_id=new_trace_id,
                contact_id=payload.get("contactId", ""),
                queued_at=entry.get("queued_at", ""),
            )

            # Route through full gate pipeline (NOT process_message)
            process_gates_shadow.delay(payload, new_trace_id)
            drained_count += 1

        except json.JSONDecodeError:
            logger.error(
                "morning_drain.invalid_json",
                raw_entry=raw_entry[:200] if raw_entry else "",
            )
            continue
        except Exception:
            logger.exception("morning_drain.entry_error")
            continue

        # 5-second delay between entries to avoid burst spam
        # (synchronous task -- time.sleep is fine here)
        remaining = _redis_client.llen(NIGHT_QUEUE_KEY)
        if remaining > 0:
            time.sleep(5)

    logger.info(
        "morning_drain.complete",
        drained=drained_count,
        original_queue_length=queue_length,
    )

    return {"drained": drained_count}
