"""Daily old-lead batch re-engagement scheduler with age-wave processing.

Runs once daily at 9:15 AM CDMX via Celery Beat. Queries Neo4j for leads
inactive for 30+ days, checks rolling 90-day eligibility, and schedules
outreach messages spread across business hours (9 AM -- 7 PM CDMX) using
Celery ETA to avoid one-morning burst.

Flow per lead:
  1. Eligibility check (rolling 90-day window, 2 attempts max)
  2. Dead classification if at_limit_within_90day_window and not yet exhausted
  3. Cross-task dedup + per-contact cooldown (Redis)
  4. Build synthetic payload routed via process_message with messageType="old_lead_outreach"
  5. Schedule via apply_async(eta=...) spread across 9 AM--7 PM CDMX

Dead-lead handling:
  - Tag "dead-lead" in GHL
  - Add note in GHL
  - Move to dead/lost pipeline stage
  - Mark permanently exhausted in Neo4j
  - No further automated outreach ever

Counter increment ownership: ReEngagementProcessor (Plan 20-02) calls
lead_repo.increment_reengagement_count() after successful AI generation.
This scheduler only READS reengagement_count via check_reengagement_eligible.
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
def init_outreach_worker(**kwargs) -> None:
    """Initialize per-worker Redis connection pool on process startup."""
    global _redis_client
    pool = redis.ConnectionPool.from_url(
        settings.redis_cache_url,
        max_connections=20,
        decode_responses=True,
    )
    _redis_client = redis.Redis(connection_pool=pool)
    logger.info("outreach_worker_redis_initialized")


async def _handle_dead_lead(
    lead_repo,
    contact_id: str,
    name: str,
    trace_id: str,
) -> None:
    """Classify a lead as dead/lost after exhausting re-engagement attempts.

    Per user decision: After 2 failed attempts (no reply), tag as dead/lost
    in GHL, move to dead/lost pipeline stage, mark permanently exhausted
    in Neo4j. No further automated outreach.

    GHL operations are individually try/excepted (Phase 19 pattern) --
    state transition (Neo4j exhaustion) is the critical path.
    """
    from app.services import ghl_service

    # 1. Mark permanently exhausted in Neo4j (critical path)
    await lead_repo.mark_reengagement_exhausted(contact_id)

    # 2. GHL: Tag as dead lead (best-effort)
    try:
        await ghl_service.add_tag(contact_id, "dead-lead")
    except Exception:
        logger.warning(
            "outreach.ghl_dead_tag_failed",
            contact_id=contact_id,
            exc_info=True,
        )

    # 3. GHL: Add note (best-effort)
    try:
        await ghl_service.add_note(
            contact_id,
            "[AI] Re-engagement exhausted (2 attempts, no reply). "
            "Classified as dead/lost lead. No further automated outreach.",
        )
    except Exception:
        logger.warning(
            "outreach.ghl_dead_note_failed",
            contact_id=contact_id,
            exc_info=True,
        )

    # 4. GHL: Move to dead/lost pipeline stage (fire-and-forget via Celery task)
    try:
        from app.tasks.pipeline_sync_task import sync_pipeline_stage

        sync_pipeline_stage.delay(contact_id, "DEAD_LOST", trace_id)
    except Exception:
        logger.warning(
            "outreach.pipeline_sync_failed",
            contact_id=contact_id,
            exc_info=True,
        )

    logger.info(
        "outreach.dead_lead_classified",
        contact_id=contact_id,
        name=name,
        trace_id=trace_id,
    )


@celery_app.task(name="scheduled.old_lead_outreach", queue="celery")
def old_lead_outreach_batch() -> dict:
    """Daily batch re-engagement of old leads with age-wave processing.

    For each candidate:
    - Ineligible + at_limit_within_90day_window -> dead classification
    - Eligible -> dedup/cooldown -> schedule via Celery ETA spread
    - Batch size capped at OUTREACH_BATCH_SIZE (default 75)

    Returns dict with counts of queued, exhausted, and skipped leads.
    """

    async def _run() -> dict:
        from app.repositories.base import close_driver, get_driver
        from app.repositories.lead_repository import LeadRepository
        from app.tasks.processing_task import process_message

        try:
            # -- Night mode guard (9 AM -- 7 PM CDMX per user decision) --
            now_cdmx = datetime.now(CDMX_TZ)
            if now_cdmx.hour < 9 or now_cdmx.hour >= 19:
                logger.info("outreach.night_mode_skip", hour=now_cdmx.hour)
                return {"leads_queued": 0, "night_mode": True}

            # -- Query eligible leads (fetch 2x to account for filtering) --
            driver = await get_driver()
            lead_repo = LeadRepository(driver)

            candidates = await lead_repo.find_old_leads_for_reengagement(
                min_inactive_days=settings.OUTREACH_MIN_INACTIVE_DAYS,
                batch_size=settings.OUTREACH_BATCH_SIZE * 2,
            )

            if not candidates:
                logger.info("outreach.no_candidates")
                return {
                    "leads_queued": 0,
                    "leads_exhausted": 0,
                    "leads_skipped": 0,
                    "candidates_found": 0,
                }

            logger.info(
                "outreach.candidates_found",
                count=len(candidates),
            )

            leads_queued = 0
            leads_exhausted = 0
            leads_skipped = 0

            # Business hours window for ETA spread
            business_start = now_cdmx.replace(
                hour=9, minute=0, second=0, microsecond=0
            )
            business_end = now_cdmx.replace(
                hour=19, minute=0, second=0, microsecond=0
            )
            total_business_seconds = int(
                (business_end - business_start).total_seconds()
            )

            for candidate in candidates:
                contact_id = candidate["contact_id"]
                phone = candidate["phone"]
                name = candidate.get("name", "")
                trace_id = str(uuid.uuid4())

                # -- Rolling 90-day eligibility check --
                eligibility = await lead_repo.check_reengagement_eligible(
                    contact_id
                )

                if not eligibility["eligible"]:
                    reason = eligibility.get("reason", "")

                    # Dead classification: at limit within window and not
                    # already permanently exhausted
                    if reason == "at_limit_within_90day_window":
                        await _handle_dead_lead(
                            lead_repo, contact_id, name, trace_id
                        )
                        leads_exhausted += 1
                    else:
                        leads_skipped += 1

                    continue

                # -- Cross-task dedup (same pattern as followup_sequence) --
                lock_key = f"outbound:recently_contacted:{contact_id}"
                if _redis_client and _redis_client.get(lock_key):
                    logger.debug(
                        "outreach.skipped_recently_contacted",
                        contact_id=contact_id,
                    )
                    leads_skipped += 1
                    continue
                if _redis_client:
                    _redis_client.set(lock_key, "1", ex=21600)  # 6h TTL

                # -- Per-contact cooldown --
                cooldown_key = f"outbound:cooldown:{contact_id}"
                if _redis_client and _redis_client.get(cooldown_key):
                    logger.debug(
                        "outreach.skipped_cooldown",
                        contact_id=contact_id,
                    )
                    leads_skipped += 1
                    continue

                # -- Build synthetic payload --
                synthetic_payload = {
                    "contactId": contact_id,
                    "phone": phone,
                    "message": "",
                    "direction": "outbound",
                    "messageType": "old_lead_outreach",
                    "tags": [],
                    "leadName": name,
                }

                # -- Schedule with business-hours spread via Celery ETA --
                # Deterministic position in the day based on contact_id hash
                position = abs(hash(contact_id)) % total_business_seconds
                send_at = business_start + timedelta(seconds=position)

                # If send_at is in the past (task runs mid-day), send with
                # small staggered delay instead
                if send_at < now_cdmx:
                    send_at = now_cdmx + timedelta(
                        seconds=5 + (leads_queued * 5)
                    )

                process_message.apply_async(
                    args=[synthetic_payload, trace_id],
                    eta=send_at.astimezone(ZoneInfo("UTC")),
                )

                logger.info(
                    "outreach.lead_scheduled",
                    contact_id=contact_id,
                    name=name,
                    send_at=send_at.isoformat(),
                    trace_id=trace_id,
                )
                leads_queued += 1

                # -- Batch size cap --
                if leads_queued >= settings.OUTREACH_BATCH_SIZE:
                    break

            logger.info(
                "outreach.batch_complete",
                leads_queued=leads_queued,
                leads_exhausted=leads_exhausted,
                leads_skipped=leads_skipped,
                candidates_found=len(candidates),
            )

            return {
                "leads_queued": leads_queued,
                "leads_exhausted": leads_exhausted,
                "leads_skipped": leads_skipped,
                "candidates_found": len(candidates),
            }
        finally:
            await close_driver()

    return asyncio.run(_run())
