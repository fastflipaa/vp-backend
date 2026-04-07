"""Out-of-band ground truth health check -- runs every 10 minutes via Beat.

Compares inbound webhook count vs successful delivery count over the
last 15 minutes via the ``ground_truth_tracker`` sorted sets. Alerts
CRITICAL if pipeline death is detected (inbound > 0, delivery == 0)
for 3 consecutive checks (~30 min).

REGRESSION CONTEXT: see app/services/monitoring/ground_truth_tracker.py
"""

from __future__ import annotations

import asyncio
import time

import redis
import structlog
from celery import signals

from app.celery_app import celery_app
from app.config import settings

logger = structlog.get_logger()

# Per-worker Redis client (pool created in worker_process_init)
_redis_client: redis.Redis | None = None

# Persistent state across check runs (in Redis to survive worker restarts)
CONSECUTIVE_FAILURES_KEY = "monitor:gt:consecutive_failures"
LAST_ALERT_AT_KEY = "monitor:gt:last_alert_at"

# Tunables
WINDOW_MINUTES = 15
ALERT_AFTER_CONSECUTIVE_FAILURES = 3   # 3 x 10-min runs = 30 min
ALERT_COOLDOWN_SECONDS = 1800          # don't re-alert within 30 min
LOW_VOLUME_THRESHOLD = 1               # below this, treat as "no traffic"


@signals.worker_process_init.connect
def _init_ground_truth_redis(**kwargs) -> None:
    """Initialize per-worker Redis connection pool on process startup."""
    global _redis_client
    pool = redis.ConnectionPool.from_url(
        settings.redis_cache_url,
        max_connections=5,
        decode_responses=True,
    )
    _redis_client = redis.Redis(connection_pool=pool)
    logger.info("ground_truth_check_worker_redis_initialized")


@celery_app.task(name="scheduled.ground_truth_check", queue="celery")
def ground_truth_check() -> dict:
    """Compare inbound webhook count vs delivery count; alert on mismatch.

    Returns the metrics dict for logging/diagnostics.
    """

    async def _run() -> dict:
        from app.services.monitoring.alert_manager import AlertLevel, AlertManager
        from app.services.monitoring.ground_truth_tracker import (
            cleanup_old_entries,
            get_window_counts,
        )

        if _redis_client is None:
            logger.warning("ground_truth_check.redis_not_initialized")
            return {"status": "skipped", "reason": "no_redis"}

        # 1. Cleanup old entries (housekeeping)
        cleanup_old_entries(keep_minutes=120)

        # 2. Read window counts -- the actual ground truth comparison
        inbound, delivery = get_window_counts(window_minutes=WINDOW_MINUTES)

        result: dict = {
            "window_minutes": WINDOW_MINUTES,
            "inbound": inbound,
            "delivery": delivery,
            "ratio": (delivery / inbound) if inbound > 0 else None,
            "alerted": False,
            "consecutive_failures": 0,
        }

        # 3. Detect pipeline death
        # Pipeline is "dead" when inbound traffic exists but no deliveries
        # are happening. Below LOW_VOLUME_THRESHOLD we treat as "quiet
        # period" and reset the counter (don't alert on weekends).
        if inbound < LOW_VOLUME_THRESHOLD:
            _redis_client.delete(CONSECUTIVE_FAILURES_KEY)
            logger.info("ground_truth_check.quiet_period", **result)
            return result

        pipeline_dead = (delivery == 0)

        if not pipeline_dead:
            # Healthy: reset consecutive counter
            _redis_client.delete(CONSECUTIVE_FAILURES_KEY)
            logger.info("ground_truth_check.healthy", **result)
            return result

        # Pipeline appears dead -- increment consecutive failure counter
        consecutive = int(_redis_client.incr(CONSECUTIVE_FAILURES_KEY))
        _redis_client.expire(CONSECUTIVE_FAILURES_KEY, 3600)
        result["consecutive_failures"] = consecutive

        if consecutive < ALERT_AFTER_CONSECUTIVE_FAILURES:
            logger.warning("ground_truth_check.failure_counted", **result)
            return result

        # Threshold exceeded -- alert (with cooldown)
        last_alert_raw = _redis_client.get(LAST_ALERT_AT_KEY)
        now = int(time.time())
        if last_alert_raw:
            try:
                cooldown_remaining = ALERT_COOLDOWN_SECONDS - (now - int(last_alert_raw))
            except (TypeError, ValueError):
                cooldown_remaining = 0
            if cooldown_remaining > 0:
                result["cooldown_remaining_s"] = cooldown_remaining
                logger.warning("ground_truth_check.alert_cooldown_active", **result)
                return result

        alert_mgr = AlertManager(_redis_client)
        sent = await alert_mgr.send(
            alert_type="ground_truth_pipeline_dead",
            message=(
                f"GROUND TRUTH ALERT: pipeline appears dead. "
                f"{inbound} inbound webhook(s) in last {WINDOW_MINUTES} min, "
                f"{delivery} successful deliveries. "
                f"This is the {consecutive}th consecutive failed check. "
                f"Investigate vp-worker-live logs for InvalidDefinition / Traceback."
            ),
            level=AlertLevel.CRITICAL,
            entity_id="ground_truth",
            extra={
                "inbound": str(inbound),
                "delivery": str(delivery),
                "consecutive_failures": str(consecutive),
                "window_minutes": str(WINDOW_MINUTES),
            },
        )
        if sent:
            _redis_client.set(LAST_ALERT_AT_KEY, str(now))
            result["alerted"] = True
        logger.error("ground_truth_check.pipeline_dead", **result)
        return result

    return asyncio.run(_run())
