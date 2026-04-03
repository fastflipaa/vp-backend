"""Operator Router — routes classified leads through the processing pipeline.

Checks night mode (22:00-08:00 CDMX), tracks return counts in Redis,
escalates to "needs-human-review" GHL tag after 3 returns, and dispatches
via process_message.delay() with a synthetic outbound payload.
"""

from __future__ import annotations

from datetime import datetime
from zoneinfo import ZoneInfo

import redis
import structlog

from app.services import ghl_service

logger = structlog.get_logger()

CDMX_TZ = ZoneInfo("America/Mexico_City")


class OperatorRouter:
    """Routes classified orphan leads through the full processing pipeline."""

    def __init__(self, redis_client: redis.Redis) -> None:
        self._redis = redis_client

    async def route_lead(
        self,
        contact_id: str,
        phone: str,
        name: str,
        classified_state: str,
        sentiment: str,
        trace_id: str,
    ) -> dict:
        """Route a classified lead through process_message.delay().

        Checks return count and night mode before routing.
        Returns dict with: routed (bool), reason (str).
        """
        # 1. Night mode check
        now_cdmx = datetime.now(CDMX_TZ)
        if now_cdmx.hour >= 22 or now_cdmx.hour < 8:
            logger.info(
                "operator.night_mode_block",
                contact_id=contact_id,
                hour=now_cdmx.hour,
            )
            return {"routed": False, "reason": "night_mode"}

        # 2. Return tracking — check if lead has exceeded max returns
        return_key = f"operator:returned:{contact_id}"
        return_count = self._redis.get(return_key)
        if return_count is not None and int(return_count) >= 3:
            try:
                await ghl_service.add_tag(contact_id, "needs-human-review")
            except Exception as e:
                logger.warning(
                    "operator.add_tag_failed",
                    contact_id=contact_id,
                    error=str(e),
                )
            logger.info(
                "operator.max_returns_exceeded",
                contact_id=contact_id,
                return_count=int(return_count),
            )
            return {"routed": False, "reason": "max_returns_exceeded"}

        # 3. Build synthetic payload matching process_message format
        payload = {
            "contactId": contact_id,
            "phone": phone,
            "message": "",
            "direction": "outbound",
            "messageType": f"operator_{classified_state.lower()}",
            "isAutoTrigger": True,
            "tags": [],
            "leadName": name,
        }

        # 4. Dispatch through process_message (import inside to avoid circular)
        from app.tasks.processing_task import process_message

        process_message.delay(payload, trace_id)

        logger.info(
            "operator.lead_routed",
            contact_id=contact_id,
            state=classified_state,
            trace_id=trace_id,
        )

        return {"routed": True, "reason": "dispatched"}

    async def handle_return(self, contact_id: str, reason: str) -> None:
        """Track a lead that returned to the operator after sub-AI failure."""
        key = f"operator:returned:{contact_id}"
        count = self._redis.incr(key)
        if count == 1:
            self._redis.expire(key, 604800)  # 7 days
        logger.info(
            "operator.lead_returned",
            contact_id=contact_id,
            return_count=count,
            reason=reason,
        )
