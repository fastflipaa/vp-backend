"""ReEngagementProcessor -- generates old-lead batch re-engagement messages.

Handles leads in RE_ENGAGE state with personalized outreach using
cross-sell building recommendations from Neo4j SIMILAR_TO relationships.

Two attempt variations + no-data re-qualifying nudge:
- Attempt 1: Cross-sell building hook (SIMILAR_TO filtered to active buildings)
- Attempt 2: Different hook (market update, exclusivity, new amenities)
- No-data: Re-qualifying nudge for leads with no building interest

The reengagement counter is persisted on the Lead node in Neo4j
and incremented ONLY after successful Claude generation. Failed AI calls
do not increment the counter, preserving the attempt for retry.

Recommended buildings are tracked to prevent duplicate suggestions across
attempts and 90-day windows.

Usage:
    processor = ReEngagementProcessor(
        claude_service=svc, prompt_builder=pb,
        lead_repo=lr, conversation_repo=cr,
        building_repo=br,
    )
    result = await processor.process(message, lead_data, ...)
"""

from __future__ import annotations

import json

import structlog

from app.processors.base import BaseProcessor, ProcessorResult
from app.prompts.builder import PromptBuilder
from app.repositories.building_repository import BuildingRepository
from app.repositories.conversation_repository import ConversationRepository
from app.repositories.lead_repository import LeadRepository
from app.services.circuit_breaker import CircuitOpenError
from app.services.claude_service import ClaudeService

logger = structlog.get_logger()

# Prompt variation keys mapped to reengagement_count
_ATTEMPT_PROMPT_MAP = {
    0: "system_prompt_attempt_1",
    1: "system_prompt_attempt_2",
}


class ReEngagementProcessor(BaseProcessor):
    """Old-lead re-engagement processor with cross-sell building selection.

    Attempt 1 (count=0): Cross-sell building recommendation from SIMILAR_TO
    Attempt 2 (count=1): Different hook -- market update, exclusivity angle

    After 2 attempts the scheduler marks leads as dead/lost.
    """

    def __init__(
        self,
        claude_service: ClaudeService,
        prompt_builder: PromptBuilder,
        lead_repo: LeadRepository,
        conversation_repo: ConversationRepository,
        building_repo: BuildingRepository = None,
        **kwargs,
    ) -> None:
        self._claude = claude_service
        self._prompt_builder = prompt_builder
        self._lead_repo = lead_repo
        self._conv_repo = conversation_repo
        self._building_repo = building_repo

    async def process(
        self,
        message: str,
        lead_data: dict,
        enriched_context: dict,
        conversation_context: dict,
        trace_id: str,
    ) -> ProcessorResult:
        """Generate a re-engagement message based on the current attempt number."""
        contact_id = lead_data.get("contact_id", "")

        # 1. Fetch re-engagement data from Neo4j
        reengagement_data = await self._lead_repo.get_reengagement_data(contact_id)
        reengagement_count = reengagement_data.get("reengagement_count", 0)
        building_names = reengagement_data.get("building_names", [])
        building_ids = reengagement_data.get("building_ids", [])
        recommended_buildings = reengagement_data.get("recommended_buildings", [])
        language = reengagement_data.get("language", "es") or lead_data.get("language", "es")
        name = reengagement_data.get("name", "") or lead_data.get("name", "")
        last_ai_message = reengagement_data.get("last_ai_message", "")

        logger.info(
            "reengagement.processing",
            trace_id=trace_id,
            contact_id=contact_id,
            reengagement_count=reengagement_count,
            building_count=len(building_names),
        )

        # 2. Guard: 2+ attempts should not reach processor (scheduler handles
        # exhaustion), but defend in depth
        if reengagement_count >= 2:
            logger.warning(
                "reengagement.exhausted",
                trace_id=trace_id,
                contact_id=contact_id,
                reengagement_count=reengagement_count,
            )
            return ProcessorResult(
                response_text="",
                new_state=None,
                metadata={"reengagement_exhausted": True, "reengagement_count": reengagement_count},
            )

        # 3. Determine attempt number (1-indexed for display)
        attempt_number = reengagement_count + 1

        # 4. Cross-sell building selection (OUTREACH-02 requirement)
        cross_sell_results = await self._select_cross_sell_buildings(
            building_ids, recommended_buildings
        )
        has_cross_sell = len(cross_sell_results) > 0

        # Format cross-sell context -- name and city only, NO pricing per user decision
        cross_sell_context = ""
        if has_cross_sell:
            cross_sell_parts = []
            for b in cross_sell_results:
                b_name = b.get("name", "")
                b_city = b.get("city", "")
                if b_city:
                    cross_sell_parts.append(f"{b_name} ({b_city})")
                else:
                    cross_sell_parts.append(b_name)
            cross_sell_context = ", ".join(cross_sell_parts)

        # 5. Determine prompt variation
        if has_cross_sell:
            # Cross-sell data available -- use attempt-specific prompt
            prompt_key = _ATTEMPT_PROMPT_MAP.get(reengagement_count, "system_prompt_attempt_2")
        elif not building_names:
            # No building interest AND no SIMILAR_TO data -- re-qualifying nudge
            prompt_key = "system_prompt_no_data"
        else:
            # Known building interest but no SIMILAR_TO data -- use attempt prompt
            # without cross_sell_building context (Claude references original interest)
            prompt_key = _ATTEMPT_PROMPT_MAP.get(reengagement_count, "system_prompt_attempt_2")

        # 6. Load YAML and render prompt
        raw_yaml = self._prompt_builder._load_yaml(
            f"{self._prompt_builder.version}/reengagement_outreach.yaml"
        )

        system_prompt_template = raw_yaml.get(prompt_key, "")
        if not system_prompt_template:
            logger.error(
                "reengagement.missing_prompt_variation",
                trace_id=trace_id,
                prompt_key=prompt_key,
            )
            return ProcessorResult(
                response_text="",
                new_state=None,
                metadata={"error": f"missing_prompt:{prompt_key}"},
            )

        primary_building = building_names[0] if building_names else ""

        context = {
            "lead_name": name,
            "language": language or "es",
            "building": primary_building,
            "buildings_seen": ", ".join(building_names) if building_names else "",
            "cross_sell_buildings": cross_sell_context,
            "reengagement_attempt": attempt_number,
            "last_topic": last_ai_message[:200] if last_ai_message else "",
        }

        # Render system prompt with context
        system_prompt = self._prompt_builder._render_template(
            system_prompt_template, context, f"reengagement.attempt_{attempt_number}.system"
        )

        # Render user template
        user_template = raw_yaml.get("user_template", "")
        user_message = self._prompt_builder._render_template(
            user_template, context, f"reengagement.attempt_{attempt_number}.user"
        )

        # Get model config
        model = raw_yaml.get("model", "claude-sonnet-4-20250514")
        max_tokens = raw_yaml.get("max_tokens", 400)

        # 7. Call Claude -- same pattern as FollowUpProcessor
        try:
            response_text = await self._claude.generate(
                system_prompt=system_prompt,
                user_message=user_message,
                conversation_history=conversation_context.get("structured_turns"),
                model=model,
                max_tokens=max_tokens,
            )
        except CircuitOpenError:
            logger.warning("reengagement.circuit_open", trace_id=trace_id)
            return ProcessorResult(
                response_text="",
                new_state=None,
                metadata={"fallback_needed": True},
            )

        # 8. Post-generation: increment counter AFTER successful Claude generation
        new_count = await self._lead_repo.increment_reengagement_count(contact_id)
        logger.info(
            "reengagement.count_incremented",
            contact_id=contact_id,
            new_count=new_count,
            attempt=attempt_number,
            trace_id=trace_id,
        )

        # 9. Record recommended buildings to prevent duplicate suggestions
        if has_cross_sell:
            new_recommendations = [b.get("name", "") for b in cross_sell_results if b.get("name")]
            updated_list = list(set(recommended_buildings + new_recommendations))
            await self._lead_repo.record_recommended_buildings(
                contact_id, json.dumps(updated_list)
            )
            logger.info(
                "reengagement.buildings_recorded",
                contact_id=contact_id,
                new_count=len(new_recommendations),
                total=len(updated_list),
                trace_id=trace_id,
            )

        # Get prompt version for traceability
        prompt_version = raw_yaml.get("version")

        return ProcessorResult(
            response_text=response_text,
            new_state=None,  # Stay in RE_ENGAGE -- scheduler handles transitions
            metadata={
                "reengagement_attempt": attempt_number,
                "reengagement_count_after": new_count,
                "building": primary_building,
                "has_cross_sell": has_cross_sell,
                "cross_sell_buildings": [b.get("name", "") for b in cross_sell_results] if has_cross_sell else [],
                "prompt_key": prompt_key,
            },
            prompt_version=prompt_version,
        )

    async def _select_cross_sell_buildings(
        self,
        building_ids: list[str],
        already_recommended: list[str],
    ) -> list[dict]:
        """Select cross-sell buildings from SIMILAR_TO relationships.

        Per user decision (LOCKED): Highest relevance SIMILAR_TO filtered
        to currently active/marketed buildings. The post-filter here is the
        authoritative active gate -- do NOT rely on the repository query
        to filter active-only (it currently does not).

        Returns up to 3 buildings ordered by similarity DESC, excluding
        buildings already recommended to this lead.
        """
        if not building_ids or self._building_repo is None:
            return []

        try:
            results = await self._building_repo.get_similar_buildings(building_ids)

            # CRITICAL -- active-only filter (per user LOCKED decision):
            # get_similar_buildings does NOT include WHERE rec.status = 'active'
            # in its Cypher query. Apply unconditional post-filter here.
            # If the method is later updated to filter natively, this is a no-op.
            results = [b for b in results if b.get("status") == "active"]

            # Dedup: filter out buildings already recommended to this lead
            results = [b for b in results if b.get("name") not in already_recommended]

            # Return top 3 (already ordered by similarity DESC from the repo)
            return results[:3]

        except Exception:
            logger.exception(
                "reengagement.cross_sell_lookup_failed",
                building_ids=building_ids,
            )
            return []
