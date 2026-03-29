"""Celery tasks for running the safety gate pipeline.

Initializes a per-worker Redis connection pool and builds the gate pipeline.
The process_gates_shadow task is the entry point called from the webhook endpoint.

Includes block-rate monitoring (Slack alert when >50% blocked in 1h window)
and shadow comparison logic (Python vs n8n gate decisions stored in Neo4j).
"""

from __future__ import annotations

import time

import redis
import structlog
from celery import signals

from app.celery_app import celery_app
from app.config import settings
from app.gates.base import GateDecision
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


def _monitor_block_rate(result_decision: GateDecision, trace_id: str) -> None:
    """Track block rate per 1-hour window. Alert via Slack if >50% blocked.

    Uses Redis INCR with 1h TTL on two counters:
    - block_rate:total:{hour_bucket}   -- every message
    - block_rate:blocked:{hour_bucket} -- only BLOCKed messages

    Fires Slack alert at most once per hour bucket (SETNX guard).
    Only alerts when total >= 10 to avoid noise on small samples.
    """
    try:
        hour_bucket = int(time.time() // 3600)
        total_key = f"block_rate:total:{hour_bucket}"
        blocked_key = f"block_rate:blocked:{hour_bucket}"
        alerted_key = f"block_rate:alerted:{hour_bucket}"

        pipe = _redis_client.pipeline()
        pipe.incr(total_key)
        pipe.expire(total_key, 3600)
        if result_decision == GateDecision.BLOCK:
            pipe.incr(blocked_key)
            pipe.expire(blocked_key, 3600)
        pipe.execute()

        total = int(_redis_client.get(total_key) or 0)
        blocked = int(_redis_client.get(blocked_key) or 0)

        if total >= 10:
            ratio = blocked / total
            if ratio > 0.50:
                # Only alert once per hour bucket
                was_set = _redis_client.set(alerted_key, "1", nx=True, ex=3600)
                if was_set:
                    from app.tasks.alerting_tasks import send_slack_alert

                    send_slack_alert.delay(
                        f"High block rate: {blocked}/{total} ({ratio:.0%}) in last hour"
                    )
                    logger.warning(
                        "high_block_rate_alert",
                        trace_id=trace_id,
                        blocked=blocked,
                        total=total,
                        ratio=round(ratio, 4),
                    )
    except Exception:
        # Monitoring failure must NOT affect gate processing
        logger.exception("block_rate_monitoring_error", trace_id=trace_id)


def _run_shadow_comparison(result, payload: dict, trace_id: str) -> None:
    """Compare Python gate decisions against n8n decisions, store in Neo4j.

    Only runs if the payload contains n8n_gate_decisions (populated when
    n8n Main Router forwards payloads to the shadow endpoint).
    Neo4j storage failure is non-fatal -- logged but doesn't fail the task.
    """
    n8n_decisions = payload.get("n8n_gate_decisions", {})
    if not n8n_decisions:
        return

    from app.shadow.comparator import compare_results
    from app.shadow.neo4j_store import store_comparison

    comparison = compare_results(result, n8n_decisions)

    logger.info(
        "shadow_comparison",
        trace_id=trace_id,
        agreement_rate=round(comparison["agreement_rate"], 4),
        disagreements=comparison["disagreements"],
        known_improvements=comparison["known_improvements"],
    )

    # Store in Neo4j (non-critical -- fail-open)
    driver = None
    try:
        from neo4j import GraphDatabase

        driver = GraphDatabase.driver(
            settings.NEO4J_URI,
            auth=("neo4j", settings.NEO4J_PASSWORD),
        )
        store_comparison(driver, comparison)
    except Exception:
        logger.exception(
            "shadow_comparison_neo4j_error",
            trace_id=trace_id,
        )
    finally:
        if driver is not None:
            try:
                driver.close()
            except Exception:
                pass


@celery_app.task(name="gates.process_shadow")
def process_gates_shadow(payload: dict, trace_id: str) -> dict:
    """Run the safety gate pipeline against an inbound payload (shadow mode).

    This task is enqueued by the /webhooks/inbound endpoint. It processes
    the payload through all registered gates and logs the result. In shadow
    mode, no messages are sent -- results are logged only.

    After pipeline execution:
    - Monitors block rate (Slack alert if >50% blocked in 1h window)
    - Compares Python vs n8n decisions if n8n_gate_decisions present
    - Stores comparison as :ShadowComparison node in Neo4j
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

    # Block-rate monitoring (non-fatal)
    _monitor_block_rate(result.overall_decision, trace_id)

    # Shadow comparison (non-fatal)
    _run_shadow_comparison(result, payload, trace_id)

    return {
        "trace_id": result.trace_id,
        "overall_decision": result.overall_decision.value,
        "gate_count": len(result.gate_results),
        "total_duration_ms": round(result.total_duration_ms, 2),
    }
