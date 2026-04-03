"""Conversation quality scan -- runs every 15 minutes via Celery Beat.

Scans all leads with recent interactions for:
1. Repetitive AI responses (TF-IDF cosine similarity)
2. Ignored user requests (regex intent matching)
3. Sentiment drift (bilingual keyword scoring)
4. Hallucinated numeric claims (cross-check vs Neo4j)
5. Language mismatch (lingua detection)

Self-healing actions are taken automatically for repetition,
negative sentiment, and language mismatch.

Follows the same pattern as health_check.py and stale_reengagement.py:
- Per-worker Redis via worker_process_init
- async _run() wrapped in asyncio.run()
- close_driver() in finally block
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
def init_conversation_scan_worker(**kwargs) -> None:
    """Initialize per-worker Redis connection pool on process startup."""
    global _redis_client
    pool = redis.ConnectionPool.from_url(
        settings.redis_cache_url,
        max_connections=20,
        decode_responses=True,
    )
    _redis_client = redis.Redis(connection_pool=pool)
    logger.info("conversation_scan_worker_redis_initialized")


@celery_app.task(name="scheduled.conversation_quality_scan", queue="celery")
def conversation_quality_scan() -> dict:
    """Scan recent conversations for quality issues.

    Returns dict with issue counts by type.
    """

    async def _run() -> dict:
        from app.repositories.base import close_driver, get_driver
        from app.services.monitoring.alert_manager import AlertManager
        from app.services.monitoring.conversation_scanner import ConversationQualityScanner

        try:
            driver = await get_driver()
            alert_mgr = AlertManager(_redis_client)
            scanner = ConversationQualityScanner(_redis_client, alert_mgr)

            counts = await scanner.scan_all_active_leads(driver, limit=50)

            logger.info(
                "conversation_quality_scan.complete",
                **counts,
            )

            return counts
        finally:
            await close_driver()

    return asyncio.run(_run())
