"""Celery task: sync a lead's GHL pipeline stage after a state transition.

This task is intentionally fail-open.  Pipeline stage management is a
nice-to-have enrichment layer -- it must never block or fail the main
message processing pipeline.

State -> Stage mapping
---------------------
GREETING            -> New Leads in 24hr       (d2e0d93a-d779-4665-ad48-6367da06e2a8)
QUALIFYING          -> Communicating           (35543f46-ac43-4890-9e14-987ce6a2919e)
HANDOFF             -> Communicating           (35543f46-ac43-4890-9e14-987ce6a2919e)
SCHEDULED           -> Engaging and/or Viewing (80eced37-ce1b-4991-9c85-9825a7752ca9)
BROKER              -> Agents and Brokers      (afd2783a-a9ec-4ce3-b837-e925f1156f94)
CLOSED              -> Not Qualified Leads     (57ed155f-d423-4a47-a4df-18be69c74550)
NON_RESPONSIVE      -> Not Qualified Leads     (57ed155f-d423-4a47-a4df-18be69c74550)
"""

from __future__ import annotations

import asyncio

import structlog

from app.celery_app import celery_app

logger = structlog.get_logger()

PIPELINE_ID = "JhXL9OVBp0ApfAfPIHuk"
LOCATION_ID = "ER9V9WFNXLK3NNXnObw9"

_STATE_TO_STAGE: dict[str, str] = {
    "GREETING": "d2e0d93a-d779-4665-ad48-6367da06e2a8",
    "QUALIFYING": "35543f46-ac43-4890-9e14-987ce6a2919e",
    "HANDOFF": "35543f46-ac43-4890-9e14-987ce6a2919e",
    "SCHEDULED": "80eced37-ce1b-4991-9c85-9825a7752ca9",
    "BROKER": "afd2783a-a9ec-4ce3-b837-e925f1156f94",
    "CLOSED": "57ed155f-d423-4a47-a4df-18be69c74550",
    "NON_RESPONSIVE": "57ed155f-d423-4a47-a4df-18be69c74550",
    "DEAD_LOST": "57ed155f-d423-4a47-a4df-18be69c74550",  # Maps to "Not Qualified Leads" -- update if dedicated dead stage exists
}


@celery_app.task(
    name="pipeline.sync_stage",
    bind=True,
    max_retries=0,
    queue="celery",
)
def sync_pipeline_stage(self, contact_id: str, new_state: str, trace_id: str) -> dict:
    """Move the contact's GHL pipeline opportunity to the matching stage."""
    stage_id = _STATE_TO_STAGE.get(new_state.upper() if new_state else "")
    if not stage_id:
        logger.info(
            "pipeline_sync_skipped",
            trace_id=trace_id,
            contact_id=contact_id,
            new_state=new_state,
            reason="no_stage_mapping",
        )
        return {"status": "skipped", "reason": "no_stage_mapping", "state": new_state}

    async def _run() -> dict:
        from app.services.ghl_service import update_opportunity_stage

        await update_opportunity_stage(
            pipeline_id=PIPELINE_ID,
            stage_id=stage_id,
            contact_id=contact_id,
            location_id=LOCATION_ID,
        )
        logger.info(
            "pipeline_sync_complete",
            trace_id=trace_id,
            contact_id=contact_id,
            new_state=new_state,
            stage_id=stage_id,
        )
        return {"status": "synced", "state": new_state, "stage_id": stage_id}

    try:
        return asyncio.run(_run())
    except Exception:
        logger.exception(
            "pipeline_sync_failed",
            trace_id=trace_id,
            contact_id=contact_id,
            new_state=new_state,
        )
        return {"status": "error", "state": new_state}
