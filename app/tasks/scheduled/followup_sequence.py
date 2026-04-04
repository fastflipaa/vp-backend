"""Hourly follow-up sequence task for leads needing attention.

Runs every hour via Celery Beat (schedule configured in Plan 17-03).
Checks for leads in FOLLOW_UP state whose last interaction was more
than 24 hours ago but less than 30 days (beyond 30 days, the stale
re-engagement COLD tier handles them).

Each follow-up routes through process_message.delay() so it passes
through the full safety pipeline (state resolution, prompt generation,
GHL delivery). The messageType "follow_up" allows the prompt builder
to select the appropriate follow-up prompt template.

5-second async delays between leads to avoid burst spam.
"""

from __future__ import annotations

import asyncio
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
def init_followup_worker(**kwargs) -> None:
    """Initialize per-worker Redis connection pool on process startup."""
    global _redis_client
    pool = redis.ConnectionPool.from_url(
        settings.redis_cache_url,
        max_connections=20,
        decode_responses=True,
    )
    _redis_client = redis.Redis(connection_pool=pool)
    logger.info("followup_worker_redis_initialized")


@celery_app.task(name="scheduled.followup_check", queue="celery")
def followup_check() -> dict:
    """Check for leads needing follow-up and send messages.

    1. Query Neo4j for leads in FOLLOW_UP state with >24h since last interaction
    2. Build synthetic outbound payload with messageType="follow_up"
    3. Route through process_message.delay() for full pipeline processing
    4. Wait 5 seconds between leads to avoid burst spam

    Returns dict with count of follow-ups triggered.
    """

    async def _run() -> dict:
        from app.repositories.base import close_driver, get_driver
        from app.repositories.lead_repository import LeadRepository
        from app.tasks.processing_task import process_message

        try:
            driver = await get_driver()
            lead_repo = LeadRepository(driver)

            # Find leads due for follow-up
            followup_leads = await lead_repo.find_followup_due(limit=20)

            if not followup_leads:
                logger.info("followup_check.no_leads_due")
                return {"followups_triggered": 0}

            logger.info(
                "followup_check.leads_found",
                count=len(followup_leads),
            )

            followups_triggered = 0

            for lead in followup_leads:
                contact_id = lead["contact_id"]
                phone = lead["phone"]
                name = lead.get("name", "")

                trace_id = str(uuid.uuid4())

                # Build synthetic payload for process_message
                synthetic_payload = {
                    "contactId": contact_id,
                    "phone": phone,
                    "message": "",
                    "direction": "outbound",
                    "messageType": "follow_up",
                    "isAutoTrigger": True,
                    "tags": [],
                    "leadName": name,
                }

                # Cross-task dedup — skip if another task recently contacted this lead
                lock_key = f"outbound:recently_contacted:{contact_id}"
                if _redis_client and _redis_client.get(lock_key):
                    logger.info("followup.skipped_recently_contacted", contact_id=contact_id)
                    continue
                if _redis_client:
                    _redis_client.set(lock_key, "1", ex=21600)  # 6h TTL

                # Route through full processing pipeline
                process_message.delay(synthetic_payload, trace_id)
                followups_triggered += 1

                logger.info(
                    "followup_check.lead_queued",
                    contact_id=contact_id,
                    trace_id=trace_id,
                )

                # 5-second delay between leads to avoid burst spam
                if followups_triggered < len(followup_leads):
                    await asyncio.sleep(5)

            logger.info(
                "followup_check.complete",
                followups_triggered=followups_triggered,
            )

            return {"followups_triggered": followups_triggered}
        finally:
            await close_driver()

    return asyncio.run(_run())
