"""Lesson lifecycle -- runs daily at 3 AM CDMX via Celery Beat.

Steps:
1. Decay old candidate lesson confidence (90d threshold, -0.1)
2. Archive candidates at floor confidence (<=0.1)
3. Auto-promote mature high-confidence candidates (>=0.7, >7d old)
4. Cleanup stale embeddings (180d)
5. Cleanup rejected lessons (30d)

Follows the same pattern as conversation_scorer.py:
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
def init_lesson_lifecycle_worker(**kwargs) -> None:
    """Initialize per-worker Redis connection pool on process startup."""
    global _redis_client
    pool = redis.ConnectionPool.from_url(
        settings.redis_cache_url,
        max_connections=20,
        decode_responses=True,
    )
    _redis_client = redis.Redis(connection_pool=pool)
    logger.info("lesson_lifecycle_worker_redis_initialized")


@celery_app.task(name="scheduled.lesson_lifecycle", queue="celery")
def lesson_lifecycle() -> dict:
    """Run daily lesson lifecycle: decay, archive, promote, cleanup.

    Returns summary dict with counts.
    """

    async def _run() -> dict:
        from app.repositories.base import close_driver, get_driver
        from app.repositories.learning_repository import LearningRepository
        from app.services.monitoring.alert_manager import AlertLevel, AlertManager

        summary = {
            "decayed": 0,
            "archived": 0,
            "promoted": 0,
            "embeddings_cleaned": 0,
            "rejected_cleaned": 0,
        }

        try:
            driver = await get_driver()
            repo = LearningRepository(driver)
            alert_mgr = AlertManager(_redis_client)

            # 1. Decay old candidate confidence
            try:
                summary["decayed"] = await repo.decay_lesson_confidence(90, 0.1)
            except Exception:
                logger.exception("lesson_lifecycle.decay_failed")

            # 2. Archive candidates at floor confidence
            try:
                summary["archived"] = await repo.archive_low_confidence()
            except Exception:
                logger.exception("lesson_lifecycle.archive_failed")

            # 3. Auto-promote mature high-confidence candidates
            promoted_ids: list[str] = []
            try:
                promoted_ids = await repo.auto_promote_candidates(
                    min_confidence=0.7, min_age_days=7
                )
                summary["promoted"] = len(promoted_ids)
            except Exception:
                logger.exception("lesson_lifecycle.promote_failed")

            # 4. Cleanup stale embeddings
            try:
                summary["embeddings_cleaned"] = await repo.cleanup_old_embeddings(180)
            except Exception:
                logger.exception("lesson_lifecycle.embeddings_cleanup_failed")

            # 5. Cleanup rejected lessons
            try:
                summary["rejected_cleaned"] = await repo.cleanup_rejected_lessons(30)
            except Exception:
                logger.exception("lesson_lifecycle.rejected_cleanup_failed")

            # Slack alert if any promotions occurred
            if promoted_ids:
                try:
                    await alert_mgr.send(
                        alert_type="lesson_lifecycle",
                        message=(
                            f"Lesson lifecycle: {len(promoted_ids)} auto-promoted, "
                            f"{summary['decayed']} decayed, {summary['archived']} archived, "
                            f"{summary['embeddings_cleaned']} embeddings cleaned, "
                            f"{summary['rejected_cleaned']} rejected cleaned"
                        ),
                        level=AlertLevel.INFO,
                        entity_id="lesson_lifecycle_daily",
                    )
                except Exception:
                    logger.exception("lesson_lifecycle.alert_failed")

            logger.info("lesson_lifecycle.complete", **summary)
            return summary

        finally:
            await close_driver()

    return asyncio.run(_run())
