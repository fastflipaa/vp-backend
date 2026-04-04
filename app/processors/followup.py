"""FollowUpProcessor -- generates building-specific follow-up messages.

Handles leads in FOLLOW_UP state with 3-variation escalating prompts:
- Attempt 1: Building-specific check-in referencing their interest from Neo4j
- Attempt 2: Cross-sell -- introduce a building they haven't seen, market update
- Attempt 3: Offer human connection with Fernando/Lorena for private tour

The follow-up counter (followup_count) is persisted on the Lead node in Neo4j
and incremented ONLY after successful Claude generation. Failed AI calls do
not increment the counter, preserving the attempt for retry.

Usage:
    processor = FollowUpProcessor(
        claude_service=svc, prompt_builder=pb,
        lead_repo=lr, conversation_repo=cr,
    )
    result = await processor.process(message, lead_data, ...)
"""

from __future__ import annotations

import structlog

from app.processors.base import BaseProcessor, ProcessorResult
from app.prompts.builder import PromptBuilder
from app.repositories.conversation_repository import ConversationRepository
from app.repositories.lead_repository import LeadRepository
from app.services.circuit_breaker import CircuitOpenError
from app.services.claude_service import ClaudeService

logger = structlog.get_logger()

# Prompt variation keys mapped to followup_count
_ATTEMPT_PROMPT_MAP = {
    0: "system_prompt_attempt_1",
    1: "system_prompt_attempt_2",
    2: "system_prompt_attempt_3",
}


class FollowUpProcessor(BaseProcessor):
    """3-variation follow-up processor with escalating message strategy.

    Attempt 1 (count=0): Building-specific check-in
    Attempt 2 (count=1): Cross-sell / new info
    Attempt 3 (count=2): Offer human connection (Fernando/Lorena)

    After 3 attempts the scheduler transitions to NON_RESPONSIVE.
    """

    def __init__(
        self,
        claude_service: ClaudeService,
        prompt_builder: PromptBuilder,
        lead_repo: LeadRepository,
        conversation_repo: ConversationRepository,
        **kwargs,
    ) -> None:
        self._claude = claude_service
        self._prompt_builder = prompt_builder
        self._lead_repo = lead_repo
        self._conv_repo = conversation_repo

    async def process(
        self,
        message: str,
        lead_data: dict,
        enriched_context: dict,
        conversation_context: dict,
        trace_id: str,
    ) -> ProcessorResult:
        """Generate a follow-up message based on the current attempt number."""
        contact_id = lead_data.get("contact_id", "")
        phone = lead_data.get("phone", "")

        # Fetch follow-up data from Neo4j
        followup_data = await self._lead_repo.get_followup_data(contact_id)
        followup_count = followup_data.get("followup_count", 0)

        logger.info(
            "followup.processing",
            trace_id=trace_id,
            contact_id=contact_id,
            followup_count=followup_count,
            phone=phone[-4:] if phone else "",
        )

        # Guard: 3+ attempts should not reach processor (scheduler handles
        # NON_RESPONSIVE transition), but defend in depth
        if followup_count >= 3:
            logger.warning(
                "followup.exhausted",
                trace_id=trace_id,
                contact_id=contact_id,
                followup_count=followup_count,
            )
            return ProcessorResult(
                response_text="",
                new_state=None,
                metadata={"followup_exhausted": True, "followup_count": followup_count},
            )

        # Determine attempt number (1-indexed for display, 0-indexed for lookup)
        attempt_number = followup_count + 1

        # Load followup YAML config
        raw_yaml = self._prompt_builder._load_yaml(
            f"{self._prompt_builder.version}/followup.yaml"
        )

        # Select the correct system prompt variation
        prompt_key = _ATTEMPT_PROMPT_MAP.get(followup_count, "system_prompt_attempt_3")
        system_prompt_template = raw_yaml.get(prompt_key, "")

        if not system_prompt_template:
            logger.error(
                "followup.missing_prompt_variation",
                trace_id=trace_id,
                prompt_key=prompt_key,
            )
            return ProcessorResult(
                response_text="",
                new_state=None,
                metadata={"error": f"missing_prompt:{prompt_key}"},
            )

        # Build context for template rendering
        lead_name = followup_data.get("name", "") or lead_data.get("name", "")
        language = followup_data.get("language", "es") or lead_data.get("language", "es")
        building_names = followup_data.get("building_names", [])
        primary_building = building_names[0] if building_names else ""
        last_ai_message = followup_data.get("last_ai_message", "")

        # For attempt 2 (cross-sell), find a building they haven't seen
        cross_sell_building = ""
        if followup_count == 1 and phone:
            cross_sell_building = await self._find_cross_sell_building(
                phone, building_names, trace_id
            )

        context = {
            "lead_name": lead_name,
            "language": language,
            "building": primary_building,
            "buildings_seen": ", ".join(building_names) if building_names else "",
            "followup_attempt": attempt_number,
            "last_topic": last_ai_message[:200] if last_ai_message else "",
            "cross_sell_building": cross_sell_building,
        }

        # Render system prompt with context
        system_prompt = self._prompt_builder._render_template(
            system_prompt_template, context, f"followup.attempt_{attempt_number}.system"
        )

        # Render user template
        user_template = raw_yaml.get("user_template", "")
        user_message = self._prompt_builder._render_template(
            user_template, context, f"followup.attempt_{attempt_number}.user"
        )

        # Get model config
        model = raw_yaml.get("model", "claude-sonnet-4-20250514")
        max_tokens = raw_yaml.get("max_tokens", 400)

        # Call Claude
        try:
            response_text = await self._claude.generate(
                system_prompt=system_prompt,
                user_message=user_message,
                conversation_history=conversation_context.get("structured_turns"),
                model=model,
                max_tokens=max_tokens,
            )
        except CircuitOpenError:
            logger.warning("followup.circuit_open", trace_id=trace_id)
            return ProcessorResult(
                response_text="",
                new_state=None,
                metadata={"fallback_needed": True},
            )

        # CRITICAL: Increment follow-up counter AFTER successful Claude generation.
        # If generate() raised an exception, we never reach here -- counter stays
        # the same, preserving the attempt for the next cycle's retry.
        new_count = await self._lead_repo.increment_followup_count(contact_id)
        logger.info(
            "followup.count_incremented",
            contact_id=contact_id,
            new_count=new_count,
            attempt=attempt_number,
            trace_id=trace_id,
        )

        # Get prompt version for traceability
        prompt_version = raw_yaml.get("version")

        return ProcessorResult(
            response_text=response_text,
            new_state=None,  # Stay in FOLLOW_UP -- scheduler handles transitions
            metadata={
                "followup_attempt": attempt_number,
                "followup_count_after": new_count,
                "building": primary_building,
            },
            prompt_version=prompt_version,
        )

    async def _find_cross_sell_building(
        self, phone: str, known_buildings: list[str], trace_id: str
    ) -> str:
        """Find a building the lead has NOT seen for cross-sell attempt.

        Uses cross-session memory to identify buildings already discussed,
        then picks one not in that list.
        """
        try:
            memory = await self._lead_repo.get_cross_session_memory(phone)
            interested_buildings = [
                b.get("name", "") for b in memory.get("buildings", [])
            ]

            # All buildings the lead has interacted with
            all_seen = set(known_buildings) | set(interested_buildings)

            # We don't have a full building catalog here, so the best we can do
            # is note what they HAVE seen and let the prompt reference it.
            # The prompt will instruct Claude to suggest something new based on
            # knowledge of the Polanco market.
            if all_seen:
                return f"El lead ya ha visto: {', '.join(all_seen)}. Sugiere algo diferente."
            return ""
        except Exception:
            logger.exception(
                "followup.cross_sell_lookup_failed",
                trace_id=trace_id,
                phone=phone[-4:] if phone else "",
            )
            return ""
