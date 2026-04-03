"""Weekly learning effectiveness report -- runs Monday 9 AM CDMX via Celery Beat.

Steps:
1. Generate new lessons from recurring error patterns
2. Build and send weekly effectiveness report via Slack

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
def init_learning_report_worker(**kwargs) -> None:
    """Initialize per-worker Redis connection pool on process startup."""
    global _redis_client
    pool = redis.ConnectionPool.from_url(
        settings.redis_cache_url,
        max_connections=20,
        decode_responses=True,
    )
    _redis_client = redis.Redis(connection_pool=pool)
    logger.info("learning_report_worker_redis_initialized")


@celery_app.task(name="scheduled.learning_report", queue="celery")
def learning_report() -> dict:
    """Generate lessons and send weekly effectiveness report.

    Returns summary dict.
    """

    async def _run() -> dict:
        from app.repositories.base import close_driver, get_driver
        from app.repositories.learning_repository import LearningRepository
        from app.services.monitoring.alert_manager import AlertLevel, AlertManager
        from app.services.monitoring.learning_tracker import LearningEffectivenessTracker
        from app.services.monitoring.lesson_generator import AutoLessonGenerator

        summary = {
            "lessons_generated": 0,
            "report_sent": False,
        }

        try:
            driver = await get_driver()
            repo = LearningRepository(driver)
            alert_mgr = AlertManager(_redis_client)
            generator = AutoLessonGenerator(repo, alert_mgr)
            tracker = LearningEffectivenessTracker(repo)

            # 1. Generate new lessons from error patterns
            try:
                generated = await generator.analyze_and_generate()
                summary["lessons_generated"] = len(generated)
            except Exception:
                logger.exception("learning_report.generation_failed")

            # 2. Build and send weekly report
            try:
                report = await tracker.generate_weekly_report()
                await alert_mgr.send(
                    alert_type="learning_report",
                    message=report,
                    level=AlertLevel.INFO,
                    entity_id="learning_report_weekly",
                )
                summary["report_sent"] = True
            except Exception:
                logger.exception("learning_report.report_failed")

            logger.info("learning_report.complete", **summary)
            return summary

        finally:
            await close_driver()

    return asyncio.run(_run())
