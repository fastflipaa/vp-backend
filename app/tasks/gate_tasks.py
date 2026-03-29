"""Celery tasks for running the safety gate pipeline.

Initializes a per-worker Redis connection pool and builds the gate pipeline.
The process_gates_shadow task is the entry point called from the webhook endpoint.
"""

from __future__ import annotations

import redis
import structlog
from celery import signals

from app.celery_app import celery_app
from app.config import settings
from app.gates.broker import check_broker
from app.gates.dedup import check_dedup
from app.gates.exfiltration import check_exfiltration
from app.gates.human_lock import check_human_lock
from app.gates.injection import check_injection
from app.gates.night_mode import check_night_mode
from app.gates.pipeline import SafetyPipeline
from app.gates.rate_limit import check_rate_limit

logger = structlog.get_logger()

_redis_client: redis.Redis | None = None


@signals.worker_process_init.connect
def init_worker_process(**kwargs) -> None:
    """Initialize per-worker Redis connection pool on process startup."""
    global _redis_client
    pool = redis.ConnectionPool.from_url(
        settings.redis_cache_url,
        max_connections=20,
        decode_responses=True,
    )
    _redis_client = redis.Redis(connection_pool=pool)
    logger.info("worker_redis_initialized", redis_url=settings.redis_cache_url)


def _build_pipeline() -> SafetyPipeline:
    """Build the safety gate pipeline with all 7 registered gates.

    Gate order matters: cheapest / highest-block-rate first, external API last.
      1. dedup        -- Redis SETNX (cheapest, blocks most duplicates)
      2. human_lock   -- Redis EXISTS (very fast)
      3. rate_limit   -- Redis sorted set sliding window
      4. night_mode   -- Timezone check + Redis queue
      5. injection    -- Pure regex (no Redis needed)
      6. exfiltration -- Pure regex (no Redis needed)
      7. broker       -- Regex + optional Haiku API call (most expensive, last)
    """
    gates = [
        ("dedup", lambda p, t: check_dedup(p, t, _redis_client)),
        ("human_lock", lambda p, t: check_human_lock(p, t, _redis_client)),
        ("rate_limit", lambda p, t: check_rate_limit(p, t, _redis_client)),
        ("night_mode", lambda p, t: check_night_mode(p, t, _redis_client)),
        ("injection", lambda p, t: check_injection(p, t)),
        ("exfiltration", lambda p, t: check_exfiltration(p, t)),
        ("broker", lambda p, t: check_broker(p, t, _redis_client)),
    ]
    return SafetyPipeline(gates=gates)


@celery_app.task(name="gates.process_shadow")
def process_gates_shadow(payload: dict, trace_id: str) -> dict:
    """Run the safety gate pipeline against an inbound payload (shadow mode).

    This task is enqueued by the /webhooks/inbound endpoint. It processes
    the payload through all registered gates and logs the result. In shadow
    mode, no messages are sent -- results are logged only.
    """
    pipeline = _build_pipeline()
    result = pipeline.run(payload, trace_id)

    logger.info(
        "pipeline_complete",
        trace_id=result.trace_id,
        overall_decision=result.overall_decision.value,
        gate_count=len(result.gate_results),
        total_duration_ms=round(result.total_duration_ms, 2),
        contact_id=result.contact_id,
        message_id=result.message_id,
        short_circuited_at=result.short_circuited_at,
    )

    return {
        "trace_id": result.trace_id,
        "overall_decision": result.overall_decision.value,
        "gate_count": len(result.gate_results),
        "total_duration_ms": round(result.total_duration_ms, 2),
    }
