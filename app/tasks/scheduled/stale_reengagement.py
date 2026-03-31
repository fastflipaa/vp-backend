"""Daily stale lead re-engagement task with HOT/WARM/COLD tiers.

Runs once daily at 10:00 CDMX via Celery Beat (schedule configured in
Plan 17-03). Queries Neo4j for leads that have gone quiet across three
inactivity tiers and sends tier-appropriate re-engagement messages through
the full safety pipeline via process_message.delay().

Tier definitions:
  - HOT  (2-5 days):  Active leads who dropped off mid-conversation.
                       States: QUALIFYING, BUILDING_INFO, SCHEDULING. Limit 20.
  - WARM (5-14 days): Leads who went quiet after initial engagement.
                       States: QUALIFYING, BUILDING_INFO, FOLLOW_UP. Limit 15.
  - COLD (14-60 days): Leads who went silent for extended periods.
                        States: QUALIFYING, FOLLOW_UP, NON_RESPONSIVE. Limit 10.

Messages route through process_message.delay() so they pass through the
full processing pipeline (state resolution, prompt generation, GHL delivery).
The messageType field (e.g. "reengagement_hot") allows the prompt builder
to select tier-appropriate prompts.

5-second async delays between leads within each tier to avoid burst spam.
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

# Tier definitions: inactivity windows, eligible states, and batch limits
TIERS = {
    "HOT": {
        "min_days": 2,
        "max_days": 5,
        "states": ["QUALIFYING", "BUILDING_INFO", "SCHEDULING"],
        "limit": 20,
    },
    "WARM": {
        "min_days": 5,
        "max_days": 14,
        "states": ["QUALIFYING", "BUILDING_INFO", "FOLLOW_UP"],
        "limit": 15,
    },
    "COLD": {
        "min_days": 14,
        "max_days": 60,
        "states": ["QUALIFYING", "FOLLOW_UP", "NON_RESPONSIVE"],
        "limit": 10,
    },
}

# Per-worker Redis client (initialized on worker_process_init)
_redis_client: redis.Redis | None = None


@signals.worker_process_init.connect
def init_reengagement_worker(**kwargs) -> None:
    """Initialize per-worker Redis connection pool on process startup."""
    global _redis_client
    pool = redis.ConnectionPool.from_url(
        settings.redis_cache_url,
        max_connections=20,
        decode_responses=True,
    )
    _redis_client = redis.Redis(connection_pool=pool)
    logger.info("reengagement_worker_redis_initialized")


@celery_app.task(name="scheduled.stale_reengagement", queue="celery")
def stale_reengagement() -> dict:
    """Process stale leads across HOT/WARM/COLD tiers.

    For each tier:
    1. Query Neo4j for stale leads matching the tier criteria
    2. Build a synthetic outbound payload with tier-specific messageType
    3. Route through process_message.delay() for full pipeline processing
    4. Wait 5 seconds between leads to avoid burst spam

    Returns dict with counts per tier and total.
    """

    async def _run() -> dict:
        from app.repositories.base import close_driver, get_driver
        from app.repositories.lead_repository import LeadRepository
        from app.tasks.processing_task import process_message

        try:
            driver = await get_driver()
            lead_repo = LeadRepository(driver)

            tier_counts = {}
            total_sent = 0

            for tier_name, tier_config in TIERS.items():
                # Query Neo4j for stale leads in this tier
                stale_leads = await lead_repo.find_stale_leads(
                    min_inactive_days=tier_config["min_days"],
                    max_inactive_days=tier_config["max_days"],
                    states=tier_config["states"],
                    limit=tier_config["limit"],
                )

                if not stale_leads:
                    logger.info(
                        "stale_reengagement.tier_empty",
                        tier=tier_name,
                    )
                    tier_counts[tier_name] = 0
                    continue

                tier_sent = 0

                for lead in stale_leads:
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
                        "messageType": f"reengagement_{tier_name.lower()}",
                        "isAutoTrigger": True,
                        "tags": [],
                        "leadName": name,
                    }

                    # Route through full processing pipeline
                    process_message.delay(synthetic_payload, trace_id)
                    tier_sent += 1

                    logger.info(
                        "stale_reengagement.lead_queued",
                        tier=tier_name,
                        contact_id=contact_id,
                        trace_id=trace_id,
                    )

                    # 5-second delay between leads to avoid burst spam
                    if tier_sent < len(stale_leads):
                        await asyncio.sleep(5)

                tier_counts[tier_name] = tier_sent
                total_sent += tier_sent

                logger.info(
                    "stale_reengagement.tier_complete",
                    tier=tier_name,
                    sent=tier_sent,
                )

            logger.info(
                "stale_reengagement.complete",
                tier_counts=tier_counts,
                total_sent=total_sent,
            )

            return {"tier_counts": tier_counts, "total_sent": total_sent}
        finally:
            await close_driver()

    return asyncio.run(_run())
