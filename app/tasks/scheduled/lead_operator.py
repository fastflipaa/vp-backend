"""Lead Operator Agent — discover orphan leads, sync from GHL, classify, and route.

Schedule: Every 3 hours via Celery Beat (at :30 to offset from other tasks).
Batch: Max 50 orphans per sweep, 5s delay between leads.
Cost: ~$0.05 per sweep (50 leads x ~$0.001 Haiku call each).
Night mode: Leads are synced and classified at any hour, but routing
    (message delivery) is blocked 22:00-08:00 CDMX.

Purpose: 5,314 of 5,413 leads (98%) have NULL current_state in Neo4j because
they were created by the old n8n system. The follow-up and re-engagement tasks
only query leads WITH state, making these orphans invisible to the AI. The Lead
Operator fixes this by pulling their GHL data, classifying their conversation
stage, assigning a state, and routing them through process_message.delay() so
the full pipeline handles them.
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

# States that should NOT be routed (terminal / non-actionable)
_SKIP_ROUTE_STATES = {"CLOSED", "BROKER", "NON_RESPONSIVE"}

# Per-worker Redis client (initialized on worker_process_init)
_redis_client: redis.Redis | None = None


@signals.worker_process_init.connect
def init_operator_worker(**kwargs) -> None:
    """Initialize per-worker Redis connection pool on process startup."""
    global _redis_client
    pool = redis.ConnectionPool.from_url(
        settings.redis_cache_url,
        max_connections=20,
        decode_responses=True,
    )
    _redis_client = redis.Redis(connection_pool=pool)
    logger.info("operator_worker_redis_initialized")


@celery_app.task(name="scheduled.lead_operator_sweep", queue="celery")
def lead_operator_sweep() -> dict:
    """Discover orphan leads, sync GHL data, classify, and route.

    For each orphan lead (NULL current_state in Neo4j):
    1. Sync GHL data (contact, conversations, notes, tags) to Neo4j
    2. Classify via Claude Haiku (state, sentiment, worth_pursuing)
    3. Save state + sentiment to Neo4j
    4. Route worthy leads through process_message.delay()

    Returns dict with counts: orphans_found, synced, classified, routed,
    errored, skipped_not_worth.
    """

    async def _run() -> dict:
        from app.repositories.base import close_driver, get_driver
        from app.repositories.lead_repository import LeadRepository
        from app.services.operator.ghl_sync import GHLSyncService
        from app.services.operator.lead_classifier import LeadClassifier
        from app.services.operator.operator_router import OperatorRouter

        if _redis_client is None:
            logger.error("lead_operator.redis_not_initialized")
            return {"orphans_found": 0, "synced": 0, "classified": 0, "routed": 0, "errored": 0, "skipped_not_worth": 0}

        try:
            driver = await get_driver()
            sync_service = GHLSyncService(driver)
            classifier = LeadClassifier(_redis_client)
            router = OperatorRouter(_redis_client)
            lead_repo = LeadRepository(driver)

            # Query orphan leads from Neo4j
            orphan_leads = await _query_orphan_leads(driver)

            synced_count = 0
            classified_count = 0
            routed_count = 0
            errored_count = 0
            skipped_count = 0

            for idx, lead in enumerate(orphan_leads):
                contact_id = lead["contact_id"]
                phone = lead.get("phone", "")
                name = lead.get("name", "")
                trace_id = str(uuid.uuid4())

                try:
                    # 1. Sync GHL data to Neo4j
                    sync_result = await sync_service.sync_lead(contact_id, phone)
                    synced_count += 1

                    # 2. Classify with Haiku
                    classification = await classifier.classify(sync_result)
                    classified_count += 1

                    # 3. Save state to Neo4j
                    await lead_repo.save_state(
                        contact_id, classification["state"]
                    )

                    # 4. Save sentiment
                    if phone:
                        await lead_repo.save_sentiment(
                            phone, classification["sentiment"], 0.8
                        )

                    # 5. Route if worth pursuing and not in terminal state
                    if (
                        classification["worth_pursuing"]
                        and classification["state"] not in _SKIP_ROUTE_STATES
                    ):
                        result = await router.route_lead(
                            contact_id=contact_id,
                            phone=phone,
                            name=name,
                            classified_state=classification["state"],
                            sentiment=classification["sentiment"],
                            trace_id=trace_id,
                        )
                        if result.get("routed"):
                            routed_count += 1
                        else:
                            skipped_count += 1
                    else:
                        skipped_count += 1

                    logger.info(
                        "operator.lead_processed",
                        contact_id=contact_id,
                        state=classification["state"],
                        sentiment=classification["sentiment"],
                        routed=classification["worth_pursuing"]
                        and classification["state"] not in _SKIP_ROUTE_STATES,
                        trace_id=trace_id,
                    )

                except Exception as e:
                    errored_count += 1
                    logger.error(
                        "operator.lead_error",
                        contact_id=contact_id,
                        error=str(e),
                        error_type=type(e).__name__,
                        trace_id=trace_id,
                    )

                # 5-second delay between leads (skip after last lead)
                if idx < len(orphan_leads) - 1:
                    await asyncio.sleep(5)

            summary = {
                "orphans_found": len(orphan_leads),
                "synced": synced_count,
                "classified": classified_count,
                "routed": routed_count,
                "errored": errored_count,
                "skipped_not_worth": skipped_count,
            }

            logger.info("operator.sweep_complete", **summary)
            return summary

        finally:
            await close_driver()

    return asyncio.run(_run())


async def _query_orphan_leads(driver) -> list[dict]:
    """Query orphan leads (NULL current_state) from Neo4j.

    Returns up to BATCH_LIMIT leads that have no current_state and haven't been
    synced by the operator in the last 7 days.
    """
    query = """
    MATCH (l:Lead)
    WHERE l.current_state IS NULL
    AND l.ghl_contact_id IS NOT NULL
    AND (l.operator_synced_at IS NULL
         OR l.operator_synced_at < datetime() - duration({days: 7}))
    RETURN l.ghl_contact_id AS contact_id,
           l.phone AS phone,
           l.name AS name
    ORDER BY l.createdAt DESC
    LIMIT 50
    """
    async with driver.session() as session:
        result = await session.run(query)
        return [dict(record) async for record in result]
