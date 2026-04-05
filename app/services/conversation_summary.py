"""Conversation summary service — Claude Haiku-powered lead summaries.

Generates structured summaries (journey, preferences, personality) for
leads with extensive conversation histories. Summaries are stored as
:ConversationSummary nodes in Neo4j and injected into prompts for
personalised re-engagement, follow-up, and qualifying.

Uses Claude Haiku (claude-haiku-4-5-20251001) for cost efficiency.
All operations fail-open: errors are logged and the pipeline continues.
"""

from __future__ import annotations

import structlog

from app.config import settings
from app.repositories.lead_repository import LeadRepository
from app.services.claude_service import ClaudeService

logger = structlog.get_logger()

SUMMARY_SYSTEM_PROMPT = """\
You are analyzing a real estate lead's conversation history. \
Generate a structured summary with EXACTLY three sections:

JOURNEY: 2-3 sentences about what buildings they explored, what they asked \
about, and where the conversation left off. Include specific building names \
and topics discussed.

PREFERENCES: Budget range, preferred zones/neighborhoods, investment vs living \
purpose, bedroom count, amenity priorities, and any other property preferences \
mentioned. List as key-value pairs.

PERSONALITY: Communication style (formal/casual), responsiveness (fast/slow \
replier), language preference (es/en), decision-making style \
(analytical/emotional/impulsive), and any notable behavioral patterns. \
List as key-value pairs.

Output format (strict):
JOURNEY: <text>
PREFERENCES: <text>
PERSONALITY: <text>"""


class ConversationSummaryService:
    """Generates and retrieves conversation summaries for leads.

    Entry point is ``maybe_generate_summary`` which checks thresholds
    and refresh intervals before calling Claude Haiku.
    """

    def __init__(
        self,
        lead_repo: LeadRepository,
        claude_service: ClaudeService,
    ) -> None:
        self._lead_repo = lead_repo
        self._claude = claude_service

    async def maybe_generate_summary(
        self, contact_id: str
    ) -> dict | None:
        """Generate a summary if the lead qualifies and one is needed.

        Returns the summary dict on success, or None if:
        - Not enough interactions (below SUMMARY_INTERACTION_THRESHOLD)
        - Summary is already current (no refresh needed)
        - An error occurred (fail-open)
        """
        try:
            refresh_info = await self._lead_repo.should_refresh_summary(
                contact_id, settings.SUMMARY_REFRESH_INTERVAL
            )

            # Not enough data to generate a meaningful summary
            if refresh_info["total_interactions"] < settings.SUMMARY_INTERACTION_THRESHOLD:
                logger.debug(
                    "summary_skipped_low_interactions",
                    contact_id=contact_id,
                    total=refresh_info["total_interactions"],
                    threshold=settings.SUMMARY_INTERACTION_THRESHOLD,
                )
                return None

            # Summary exists and is still current
            if not refresh_info["needs_refresh"]:
                logger.debug(
                    "summary_skipped_current",
                    contact_id=contact_id,
                    last_count=refresh_info["last_count"],
                )
                return None

            return await self._generate_summary(contact_id)

        except Exception:
            logger.exception(
                "summary_generation_failed",
                contact_id=contact_id,
            )
            return None

    async def _generate_summary(self, contact_id: str) -> dict | None:
        """Fetch conversation data, call Claude Haiku, and persist summary."""
        conversation = await self._lead_repo.get_conversation_for_summary(
            contact_id
        )
        lead = conversation["lead"]
        interactions = conversation["interactions"]
        buildings = conversation["buildings"]
        total = conversation["total_interactions"]

        if not interactions:
            logger.warning(
                "summary_no_interactions",
                contact_id=contact_id,
            )
            return None

        # Format conversation text for Claude
        user_message = self._format_conversation(
            lead, interactions, buildings, total
        )

        # Call Claude Haiku
        response = await self._claude.generate(
            system_prompt=SUMMARY_SYSTEM_PROMPT,
            user_message=user_message,
            model="claude-haiku-4-5-20251001",
            max_tokens=500,
        )

        # Parse the three sections
        summary_data = self._parse_summary(response)
        summary_data["interaction_count"] = total

        # Persist to Neo4j
        await self._lead_repo.save_conversation_summary(
            contact_id, response, summary_data
        )

        logger.info(
            "summary_generated",
            contact_id=contact_id,
            interaction_count=total,
            summary_length=len(response),
        )

        return summary_data

    async def get_summary_for_prompt(self, contact_id: str) -> str:
        """Return a formatted summary string ready for prompt injection.

        Returns an empty string if no summary exists, so callers can
        safely concatenate without extra checks.
        """
        try:
            summary = await self._lead_repo.get_conversation_summary(
                contact_id
            )
            if not summary:
                return ""

            return (
                "HISTORIAL DEL LEAD:\n"
                f"Trayectoria: {summary.get('journey', '')}\n"
                f"Preferencias: {summary.get('preferences', '')}\n"
                f"Perfil: {summary.get('personality', '')}"
            )

        except Exception:
            logger.exception(
                "summary_for_prompt_failed",
                contact_id=contact_id,
            )
            return ""

    # --- Private helpers ---

    @staticmethod
    def _format_conversation(
        lead: dict,
        interactions: list[dict],
        buildings: list[str],
        total: int,
    ) -> str:
        """Format conversation data into structured text for Claude."""
        lines = []

        # Lead header
        name = lead.get("name") or "Unknown"
        language = lead.get("language") or "es"
        state = lead.get("current_state") or "Unknown"
        lines.append(f"Lead: {name}, Language: {language}, State: {state}")

        if buildings:
            lines.append(f"Buildings of interest: {', '.join(buildings)}")

        budget_min = lead.get("budgetMin")
        budget_max = lead.get("budgetMax")
        timeline = lead.get("timeline")
        if budget_min or budget_max:
            lines.append(f"Budget: {budget_min or '?'}-{budget_max or '?'}")
        if timeline:
            lines.append(f"Timeline: {timeline}")

        lines.append("")
        lines.append(f"Conversation ({total} messages):")

        # Interactions in chronological order
        for interaction in interactions:
            role = interaction.get("role", "unknown")
            content = interaction.get("content", "")
            timestamp = interaction.get("created_at", "")
            # Truncate very long messages to avoid blowing up tokens
            if len(content) > 500:
                content = content[:497] + "..."
            lines.append(f"[{role}]: {content} ({timestamp})")

        return "\n".join(lines)

    @staticmethod
    def _parse_summary(response: str) -> dict[str, str]:
        """Parse Claude's response into journey, preferences, personality."""
        result = {"journey": "", "preferences": "", "personality": ""}

        # Split on section markers
        text = response.strip()

        # Find each section
        journey_idx = text.find("JOURNEY:")
        preferences_idx = text.find("PREFERENCES:")
        personality_idx = text.find("PERSONALITY:")

        if journey_idx != -1:
            start = journey_idx + len("JOURNEY:")
            end = preferences_idx if preferences_idx != -1 else len(text)
            result["journey"] = text[start:end].strip()

        if preferences_idx != -1:
            start = preferences_idx + len("PREFERENCES:")
            end = personality_idx if personality_idx != -1 else len(text)
            result["preferences"] = text[start:end].strip()

        if personality_idx != -1:
            start = personality_idx + len("PERSONALITY:")
            result["personality"] = text[start:].strip()

        return result
