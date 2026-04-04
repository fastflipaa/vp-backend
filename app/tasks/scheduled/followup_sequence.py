"""Hourly follow-up sequence task with counter-based 3-attempt lifecycle.

Runs every hour via Celery Beat (schedule configured in Plan 17-03).
Orchestrates the follow-up lifecycle for leads in FOLLOW_UP state:

1. Night mode guard: only runs during 8AM-10PM CDMX.
2. Queries Neo4j for leads in FOLLOW_UP state (20h+ since last interaction).
3. Exhaustion check: leads with followup_count >= 3 transition to
   NON_RESPONSIVE and get the "email-drip-ready" GHL tag.
4. Per-lead jitter: deterministic 20-28h interval per contact_id so
   follow-ups feel natural, not robotic.
5. Cross-task dedup + cooldown respect (Redis).
6. Eligible leads route through process_message.delay() with
   messageType="follow_up" so the FollowUpProcessor generates the
   AI message and increments the counter.

Counter increment ownership: FollowUpProcessor (Plan 19-01) calls
lead_repo.increment_followup_count() after successful AI generation.
This scheduler only READS followup_count to decide actions.
"""

from __future__ import annotations

import asyncio
import uuid
from datetime import datetime, timedelta
from zoneinfo import ZoneInfo

import redis
import structlog
from celery import signals

from app.celery_app import celery_app
from app.config import settings

logger = structlog.get_logger()

CDMX_TZ = ZoneInfo("America/Mexico_City")

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
    """Orchestrate the 3-attempt follow-up lifecycle.

    For each FOLLOW_UP lead:
    - followup_count >= 3: Transition to NON_RESPONSIVE + GHL tag
    - followup_count < 3 + jitter due: Route through pipeline
    - Otherwise: Skip (not due yet, recently contacted, or in cooldown)

    Returns dict with counts of triggered, exhausted, and skipped leads.
    """

    async def _run() -> dict:
        from app.repositories.base import close_driver, get_driver
        from app.repositories.lead_repository import LeadRepository
        from app.services import ghl_service
        from app.tasks.processing_task import process_message

        try:
            # ── Night mode guard ─────────────────────────────────
            now_cdmx = datetime.now(CDMX_TZ)
            if now_cdmx.hour < 8 or now_cdmx.hour >= 22:
                logger.info("followup.night_mode_skip", hour=now_cdmx.hour)
                return {"followups_triggered": 0, "night_mode": True}

            # ── Query eligible leads ─────────────────────────────
            driver = await get_driver()
            lead_repo = LeadRepository(driver)

            followup_leads = await lead_repo.find_followup_due(limit=30)

            if not followup_leads:
                logger.info("followup_check.no_leads_due")
                return {
                    "followups_triggered": 0,
                    "leads_exhausted": 0,
                    "leads_skipped": 0,
                }

            logger.info(
                "followup_check.leads_found",
                count=len(followup_leads),
            )

            followups_triggered = 0
            leads_exhausted = 0
            leads_skipped = 0

            for lead in followup_leads:
                contact_id = lead["contact_id"]
                phone = lead["phone"]
                name = lead.get("name", "")
                followup_count = lead.get("followup_count", 0)

                # ── Exhaustion check: 3 attempts used up ─────────
                if followup_count >= 3:
                    # Transition to NON_RESPONSIVE in Neo4j
                    await lead_repo.save_state(contact_id, "NON_RESPONSIVE")

                    # GHL tagging + note (best-effort — state transition is critical path)
                    try:
                        await ghl_service.add_tag(contact_id, "email-drip-ready")
                    except Exception:
                        logger.warning(
                            "followup.ghl_tag_failed",
                            contact_id=contact_id,
                            exc_info=True,
                        )
                    try:
                        await ghl_service.add_note(
                            contact_id,
                            "[AI] Follow-up sequence exhausted (3 attempts). "
                            "Moved to NON_RESPONSIVE. Tagged for email drip.",
                        )
                    except Exception:
                        logger.warning(
                            "followup.ghl_note_failed",
                            contact_id=contact_id,
                            exc_info=True,
                        )

                    logger.info(
                        "followup.lead_exhausted",
                        contact_id=contact_id,
                        followup_count=followup_count,
                    )
                    leads_exhausted += 1
                    continue

                # ── Per-lead jitter filter ────────────────────────
                # Deterministic jitter: 0-8 extra hours on top of 20h base.
                # Neo4j already filters at 20h minimum, so we only need to
                # check if the ADDITIONAL jitter time has elapsed since the
                # last follow-up.
                followup_data = await lead_repo.get_followup_data(contact_id)
                last_followup_at = followup_data.get("last_followup_at")

                if last_followup_at is not None:
                    # Convert Neo4j datetime to Python datetime if needed
                    if hasattr(last_followup_at, "to_native"):
                        last_followup_at = last_followup_at.to_native()
                    # Make timezone-aware if naive
                    if last_followup_at.tzinfo is None:
                        last_followup_at = last_followup_at.replace(tzinfo=CDMX_TZ)

                    jitter_hours = abs(hash(contact_id)) % 9  # 0-8 hours
                    required_interval = timedelta(hours=20 + jitter_hours)
                    if (now_cdmx - last_followup_at) < required_interval:
                        logger.debug(
                            "followup.jitter_skip",
                            contact_id=contact_id,
                            jitter_hours=jitter_hours,
                        )
                        leads_skipped += 1
                        continue

                # ── Cross-task dedup ──────────────────────────────
                lock_key = f"outbound:recently_contacted:{contact_id}"
                if _redis_client and _redis_client.get(lock_key):
                    logger.info(
                        "followup.skipped_recently_contacted",
                        contact_id=contact_id,
                    )
                    leads_skipped += 1
                    continue
                if _redis_client:
                    _redis_client.set(lock_key, "1", ex=21600)  # 6h TTL

                # ── Per-contact cooldown ──────────────────────────
                cooldown_key = f"outbound:cooldown:{contact_id}"
                if _redis_client and _redis_client.get(cooldown_key):
                    logger.info(
                        "followup.skipped_cooldown",
                        contact_id=contact_id,
                    )
                    leads_skipped += 1
                    continue

                # ── Send follow-up through pipeline ───────────────
                trace_id = str(uuid.uuid4())

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

                process_message.delay(synthetic_payload, trace_id)

                logger.info(
                    "followup.lead_queued",
                    contact_id=contact_id,
                    attempt=followup_count + 1,
                    trace_id=trace_id,
                )
                followups_triggered += 1

                # 5-second delay between leads to avoid burst spam
                await asyncio.sleep(5)

            logger.info(
                "followup_check.complete",
                followups_triggered=followups_triggered,
                leads_exhausted=leads_exhausted,
                leads_skipped=leads_skipped,
            )

            return {
                "followups_triggered": followups_triggered,
                "leads_exhausted": leads_exhausted,
                "leads_skipped": leads_skipped,
            }
        finally:
            await close_driver()

    return asyncio.run(_run())
