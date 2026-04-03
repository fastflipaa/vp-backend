"""Lead Classifier — uses Claude Haiku to classify orphan leads.

Analyzes synced GHL data (contact info, conversation history, notes,
building interest) and assigns a conversation state + sentiment.

Short-circuits for obvious cases to save API costs:
- Zero messages + no buildings -> GREETING (skip Claude call)
- All outbound messages with no inbound -> NON_RESPONSIVE (skip Claude call)

Cost: ~$0.001 per lead for Haiku classification.
"""

from __future__ import annotations

import json
import re

import redis
import structlog

from app.services.claude_service import ClaudeService
from app.services.circuit_breaker import CircuitOpenError

logger = structlog.get_logger()

VALID_STATES = {
    "GREETING",
    "QUALIFYING",
    "SCHEDULING",
    "FOLLOW_UP",
    "NON_RESPONSIVE",
    "BROKER",
    "CLOSED",
}

SYSTEM_PROMPT = """You classify leads for Vive Polanco, a luxury real estate company in Mexico City.

States (pick one):
GREETING = No real conversation yet
QUALIFYING = Lead showed interest, needs budget/timeline/preferences
SCHEDULING = Qualified, ready for viewing appointment
FOLLOW_UP = Was engaged but went quiet
NON_RESPONSIVE = Contacted multiple times, no reply
BROKER = Real estate broker, not end buyer
CLOSED = Said not interested or clearly not a prospect

Output ONLY this JSON, nothing else:
{"state":"STATE","sentiment":"positive|neutral|negative","worth_pursuing":true,"reason":"brief"}"""


class LeadClassifier:
    """Classifies orphan leads using Claude Haiku based on synced GHL data."""

    def __init__(self, redis_client: redis.Redis) -> None:
        self._claude = ClaudeService(redis_client)

    async def classify(self, sync_result: dict) -> dict:
        """Classify an orphan lead based on synced GHL data.

        Args:
            sync_result: The dict returned by GHLSyncService.sync_lead(),
                         containing contact, messages_summary, notes_count,
                         buildings_matched, message_count.

        Returns:
            Dict with: state (str), sentiment (str), worth_pursuing (bool), reason (str).
        """
        contact = sync_result.get("contact", {})
        message_count = sync_result.get("message_count", 0)
        messages_summary = sync_result.get("messages_summary", [])
        notes_count = sync_result.get("notes_count", 0)
        buildings_matched = sync_result.get("buildings_matched", [])
        name = contact.get("name", "Unknown")

        # Short-circuit 1: No messages and no buildings -> GREETING
        if message_count == 0 and not buildings_matched:
            logger.info(
                "operator.classify_shortcircuit",
                name=name,
                reason="no_messages_no_buildings",
                state="GREETING",
            )
            return {
                "state": "GREETING",
                "sentiment": "neutral",
                "worth_pursuing": True,
                "reason": "no_conversation_history",
            }

        # Short-circuit 2: All outbound, no inbound -> NON_RESPONSIVE
        if messages_summary and all(
            m.get("role") == "assistant" for m in messages_summary
        ):
            logger.info(
                "operator.classify_shortcircuit",
                name=name,
                reason="all_outbound",
                state="NON_RESPONSIVE",
            )
            return {
                "state": "NON_RESPONSIVE",
                "sentiment": "neutral",
                "worth_pursuing": False,
                "reason": "all_outbound_no_response",
            }

        # Build user message for Claude
        tags_str = ", ".join(contact.get("tags", [])) or "none"
        buildings_str = ", ".join(buildings_matched) or "none detected"

        conversation_lines = []
        if messages_summary:
            for msg in messages_summary:
                role = msg.get("role", "unknown")
                content = msg.get("content", "")[:200]
                conversation_lines.append(f"[{role}]: {content}")
        else:
            conversation_lines.append("No conversation history available")

        user_message = (
            f"Contact: {name}, {contact.get('email', 'no email')}\n"
            f"Tags: {tags_str}\n"
            f"Buildings interested: {buildings_str}\n"
            f"Total messages: {message_count}\n"
            f"Notes: {notes_count} notes on file\n\n"
            f"Recent conversation (last 5 messages, newest first):\n"
            + "\n".join(conversation_lines)
        )

        # Call Claude Haiku for classification
        try:
            response_text = await self._claude.generate(
                system_prompt=SYSTEM_PROMPT,
                user_message=user_message,
                model="claude-haiku-4-5-20251001",
                max_tokens=256,
            )

            # Strip markdown code blocks if present
            cleaned = response_text.strip()
            code_block = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", cleaned, re.DOTALL)
            if code_block:
                cleaned = code_block.group(1)
            # Also try to extract raw JSON object
            elif not cleaned.startswith("{"):
                json_match = re.search(r"\{[^{}]*\}", cleaned)
                if json_match:
                    cleaned = json_match.group(0)

            result = json.loads(cleaned)

            state = result.get("state", "GREETING")
            if state not in VALID_STATES:
                logger.warning(
                    "operator.classify_invalid_state",
                    name=name,
                    raw_state=state,
                    defaulting_to="GREETING",
                )
                state = "GREETING"

            sentiment = result.get("sentiment", "neutral")
            if sentiment not in ("positive", "neutral", "negative"):
                sentiment = "neutral"

            worth_pursuing = bool(result.get("worth_pursuing", True))
            reason = result.get("reason", "")

            classification = {
                "state": state,
                "sentiment": sentiment,
                "worth_pursuing": worth_pursuing,
                "reason": reason,
            }

        except (json.JSONDecodeError, KeyError, TypeError) as e:
            logger.warning(
                "operator.classify_parse_error",
                name=name,
                error=str(e),
            )
            classification = {
                "state": "GREETING",
                "sentiment": "neutral",
                "worth_pursuing": True,
                "reason": "classification_parse_error",
            }

        except CircuitOpenError:
            logger.warning("operator.classify_circuit_open", name=name)
            classification = {
                "state": "GREETING",
                "sentiment": "neutral",
                "worth_pursuing": True,
                "reason": "circuit_open",
            }

        logger.info(
            "operator.classified",
            name=name,
            state=classification["state"],
            sentiment=classification["sentiment"],
            worth_pursuing=classification["worth_pursuing"],
        )

        return classification
