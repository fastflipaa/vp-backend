"""System health scan -- runs every 5 minutes via Celery Beat.

Checks:
1. Circuit breaker states (claude, ghl)
2. Error rate counters (sliding 5-min window in Redis)
3. Dead-letter queue depth
4. JSON parse failure counter
5. State machine transition error counter

Sends Slack alerts for critical issues.

Follows the same pattern as health_check.py:
- Per-worker Redis via worker_process_init
- async _run() wrapped in asyncio.run()
"""

from __future__ import annotations

import asyncio

import redis
import structlog
from celery import signals

from app.celery_app import celery_app
from app.config import settings

logger = structlog.get_logger()

# Per-worker Redis client
_redis_client: redis.Redis | None = None


@signals.worker_process_init.connect
def init_system_health_worker(**kwargs) -> None:
    """Initialize per-worker Redis connection pool on process startup."""
    global _redis_client
    pool = redis.ConnectionPool.from_url(
        settings.redis_cache_url,
        max_connections=20,
        decode_responses=True,
    )
    _redis_client = redis.Redis(connection_pool=pool)
    logger.info("system_health_scan_worker_redis_initialized")


@celery_app.task(name="scheduled.system_health_scan", queue="celery")
def system_health_scan() -> dict:
    """Check system health metrics and alert on issues.

    Returns dict with health metrics and alert count.
    """

    async def _run() -> dict:
        from app.services.monitoring.alert_manager import AlertManager
        from app.services.monitoring.system_health_scanner import SystemHealthScanner

        alert_mgr = AlertManager(_redis_client)
        scanner = SystemHealthScanner(_redis_client, alert_mgr)

        metrics = await scanner.scan()
        alerts_sent = await scanner.alert_on_issues(metrics)

        logger.info(
            "system_health_scan.complete",
            alerts_sent=alerts_sent,
            circuit_breakers=metrics.get("circuit_breakers"),
            dlq_depth=metrics.get("dlq_depth"),
        )

        return {**metrics, "alerts_sent": alerts_sent}

    return asyncio.run(_run())
