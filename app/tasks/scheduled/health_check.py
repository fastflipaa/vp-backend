"""Scheduled health check task -- detects leads with missed inbound messages.

Runs every 5 minutes via Celery Beat (schedule configured in Plan 17-03).
Finds leads who sent an inbound message but received no outbound response
within 10 minutes. Each missed lead gets exactly ONE recovery attempt per
24 hours (Redis SETNX dedup). Recovery messages pass through the full
safety pipeline via process_message.delay().

Before sending recovery, a :RecoveryAttempt audit node is created in Neo4j
with the trace_id and reason.
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
def init_health_check_worker(**kwargs) -> None:
    """Initialize per-worker Redis connection pool on process startup."""
    global _redis_client
    pool = redis.ConnectionPool.from_url(
        settings.redis_cache_url,
        max_connections=20,
        decode_responses=True,
    )
    _redis_client = redis.Redis(connection_pool=pool)
    logger.info("health_check_worker_redis_initialized")


@celery_app.task(name="scheduled.health_check", queue="celery")
def health_check() -> dict:
    """Detect leads with missed inbound messages and trigger recovery.

    1. Query Neo4j for leads with unanswered inbound (>10 min, <20 min window)
    2. For each missed lead, check Redis SETNX dedup (1 recovery per 24h)
    3. Create RecoveryAttempt audit node in Neo4j
    4. Route recovery through full pipeline via process_message.delay()

    Returns dict with count of recoveries triggered.
    """

    async def _run() -> dict:
        from app.repositories.base import close_driver, get_driver
        from app.repositories.lead_repository import LeadRepository
        from app.tasks.processing_task import process_message

        try:
            driver = await get_driver()
            lead_repo = LeadRepository(driver)

            # Find leads with unanswered inbound messages
            missed_leads = await lead_repo.find_missed_inbound(
                window_minutes=20,
                response_threshold_minutes=10,
            )

            if not missed_leads:
                logger.info("health_check.no_missed_leads")
                return {"recoveries_triggered": 0}

            logger.info(
                "health_check.missed_leads_found",
                count=len(missed_leads),
            )

            recoveries_triggered = 0

            for lead in missed_leads:
                contact_id = lead["contact_id"]
                phone = lead["phone"]
                name = lead.get("name", "")

                # Redis SETNX dedup: one recovery per contact per 24h
                dedup_key = f"recovery_dedup:{contact_id}"
                was_set = _redis_client.set(dedup_key, "1", nx=True, ex=86400)
                if not was_set:
                    logger.debug(
                        "health_check.already_recovered_today",
                        contact_id=contact_id,
                    )
                    continue

                # Generate trace_id for this recovery
                trace_id = str(uuid.uuid4())

                # Create RecoveryAttempt audit node in Neo4j
                try:
                    await lead_repo.create_recovery_attempt(
                        contact_id=contact_id,
                        trace_id=trace_id,
                        reason="missed_inbound_10min",
                    )
                except Exception:
                    logger.exception(
                        "health_check.recovery_attempt_create_failed",
                        contact_id=contact_id,
                        trace_id=trace_id,
                    )
                    # Continue anyway -- recovery is more important than audit

                # Build synthetic payload for process_message
                synthetic_payload = {
                    "contactId": contact_id,
                    "phone": phone,
                    "message": "",
                    "direction": "outbound",
                    "messageType": "health_check_recovery",
                    "isAutoTrigger": True,
                    "tags": lead.get("tags", []),
                    "leadName": name,
                }

                # Route through full processing pipeline
                process_message.delay(synthetic_payload, trace_id)
                recoveries_triggered += 1

                logger.info(
                    "health_check.recovery_triggered",
                    contact_id=contact_id,
                    trace_id=trace_id,
                )

            logger.info(
                "health_check.complete",
                recoveries_triggered=recoveries_triggered,
                missed_leads_found=len(missed_leads),
            )

            return {"recoveries_triggered": recoveries_triggered}
        finally:
            # Close the cached Neo4j driver so the next asyncio.run() gets a
            # fresh driver bound to its own event loop.  Without this, the
            # second invocation hits "Future attached to a different loop"
            # because the driver's connections are still bound to the first
            # (now-closed) event loop created by asyncio.run().
            await close_driver()

    return asyncio.run(_run())
