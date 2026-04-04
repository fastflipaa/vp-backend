"""StateResolver -- resolves lead conversation state from Neo4j with overrides.

Ports logic from n8n Main Router Stage 2 ("Determine State" node).
Loads lead data from Neo4j, applies auto-trigger and messageType overrides,
validates against LeadState enum, and defaults to GREETING for unknown states.

Usage:
    resolver = StateResolver(lead_repo)
    result = await resolver.resolve(phone="+5215512345678")
    # result = {"state": "GREETING", "is_new_lead": True, "lead_data": {}}
"""

from __future__ import annotations

import structlog

from app.repositories.lead_repository import LeadRepository
from app.state_machine.states import LeadState

logger = structlog.get_logger()

# Valid state values from LeadState enum
_VALID_STATES = {s.value for s in LeadState}


class StateResolver:
    """Resolves the current conversation state for a lead.

    Loads lead data from Neo4j and applies override logic matching
    the n8n "Determine State" node behavior:
    - Auto-trigger forces GREETING (restart conversation)
    - messageType overrides: re_engagement -> RE_ENGAGE, etc.
    - Unknown/missing states default to GREETING
    """

    def __init__(self, lead_repo: LeadRepository) -> None:
        self._lead_repo = lead_repo

    async def resolve(
        self,
        phone: str,
        is_auto_trigger: bool = False,
        message_type: str = "",
    ) -> dict:
        """Resolve lead state from Neo4j with override logic.

        Args:
            phone: Lead phone number (E.164 format).
            is_auto_trigger: True if this is an auto-trigger message
                (forces state to GREETING).
            message_type: Override type from webhook payload
                (re_engagement, post_appointment, follow_up,
                old_lead_outreach).

        Returns:
            Dict with keys:
            - state: resolved state string (e.g. "GREETING")
            - is_new_lead: True if lead not found in Neo4j
            - lead_data: dict of lead properties from Neo4j (empty if new)
        """
        # 1. Query Neo4j for lead data
        lead_data = await self._lead_repo.get_lead_by_phone(phone)

        if lead_data is None:
            logger.info(
                "state_resolver.new_lead",
                phone=phone[-4:],
                resolved_state="GREETING",
            )
            return {
                "state": "GREETING",
                "is_new_lead": True,
                "lead_data": {},
            }

        # 2. Extract current state
        current_state = lead_data.get("current_state") or ""
        override_applied = None

        # 3. Apply overrides (matching n8n Determine State node)
        if is_auto_trigger:
            current_state = "GREETING"
            override_applied = "auto_trigger"
        elif message_type == "re_engagement":
            current_state = "RE_ENGAGE"
            override_applied = "re_engagement"
        elif message_type == "post_appointment":
            current_state = "SCHEDULING"
            override_applied = "post_appointment"
        elif message_type == "follow_up":
            current_state = "FOLLOW_UP"
            override_applied = "follow_up"
        elif message_type == "old_lead_outreach":
            current_state = "RE_ENGAGE"
            override_applied = "old_lead_outreach"

        # 4. Default empty state to GREETING
        if not current_state:
            current_state = "GREETING"
            override_applied = "empty_state_default"

        # 5. Validate against LeadState enum
        if current_state not in _VALID_STATES:
            logger.warning(
                "state_resolver.invalid_state",
                phone=phone[-4:],
                invalid_state=current_state,
                defaulting_to="GREETING",
            )
            current_state = "GREETING"
            override_applied = "invalid_state_fallback"

        logger.info(
            "state_resolver.resolved",
            phone=phone[-4:],
            resolved_state=current_state,
            override_applied=override_applied,
        )

        return {
            "state": current_state,
            "is_new_lead": False,
            "lead_data": lead_data,
        }
