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
from app.gates.dedup import check_dedup
from app.gates.human_lock import check_human_lock
from app.gates.pipeline import SafetyPipeline

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
    """Build the safety gate pipeline with all registered gates.

    Gate order matters: dedup runs first (cheapest, blocks most),
    then human_lock (Redis EXISTS, very fast).
    """
    gates = [
        ("dedup", lambda p, t: check_dedup(p, t, _redis_client)),
        ("human_lock", lambda p, t: check_human_lock(p, t, _redis_client)),
        # TODO: Add rate_limit, night_mode, injection, exfiltration, broker gates (Plans 15-02, 15-03)
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
