"""Human Agent Detector — checks GHL conversation for human activity.

Replicates n8n Main Router Stage 3 (Human Agent Detection): fetches the
last 10 messages from a GHL conversation and checks for human agent
activity within a configurable time window (default 6 hours).

Detection signals:
- Messages from non-bot sources (not in bot/system/automation/workflow/api)
- Trigger words indicating human handoff (natalia, te paso, pass you back, etc.)
- Spam limiter: 3+ consecutive outbound without inbound reply

Also extracts:
- ghlConversationContext: last 5 messages as text for Claude prompt injection
- mostRecentBuilding: building name mentions in recent messages
- spamLimitReached: True if consecutive outbound exceeds threshold

Usage:
    detector = HumanAgentDetector()
    is_active, reason = await detector.is_human_active("contact_abc", "conv_123")
    context = detector.get_conversation_context()
"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone

import structlog

from app.services import ghl_service

logger = structlog.get_logger()

# Trigger words indicating a human agent has taken over (EN + ES)
TRIGGER_WORDS = [
    "natalia",
    "te paso",
    "pass you back",
    "te comunico",
    "mi compañera",
    "mi colega",
]

# Sources that are NOT human agents
BOT_SOURCES = {"bot", "system", "automation", "workflow", "api"}

# Known building names for context extraction
# Order matters: longer/more-specific names first to avoid partial matches
# (e.g., "one park" before "one", "888 brickell" before "888")
BUILDING_NAMES = [
    "one park tower",
    "one park",
    "888 brickell",
    "brickell",
    "ritz-carlton",
    "ritz carlton",
    "ritz",
    "armani",
    "park hyatt",
    "park",
    "thompson",
    "oscar",
    "en-vogue",
    "the gallery",
    "glass",
    "nobu",
    "888",
]

# Max consecutive outbound messages before spam limit triggers
SPAM_LIMIT_THRESHOLD = 3


class HumanAgentDetector:
    """Detects human agent activity in GHL conversations.

    Call ``is_human_active()`` first, then ``get_conversation_context()``
    to retrieve cached context (avoids a second GHL API call).
    """

    def __init__(self, agent_window_hours: int = 6) -> None:
        self.agent_window_hours = agent_window_hours
        self._last_messages: list[dict] = []

    async def is_human_active(
        self, contact_id: str, conversation_id: str = ""
    ) -> tuple[bool, str]:
        """Check if a human agent is active in the conversation.

        Args:
            contact_id: GHL contact ID.
            conversation_id: GHL conversation ID (optional; will search if
                not provided).

        Returns:
            Tuple of (is_active, reason). Reasons:
            - ``"trigger_word"`` — trigger word detected in recent messages
            - ``"agent_active"`` — non-bot message in the time window
            - ``"no_conversation"`` — no conversation found for contact
            - ``"clear"`` — no human activity detected
        """
        # Get conversation_id if not provided
        if not conversation_id:
            try:
                conv = await ghl_service.search_conversations(contact_id)
                if conv:
                    conversation_id = conv.get("id", "")
            except Exception as e:
                logger.warning(
                    "human_detector.search_failed",
                    contact_id=contact_id,
                    error=str(e),
                )

        if not conversation_id:
            self._last_messages = []
            logger.debug(
                "human_detector.no_conversation", contact_id=contact_id
            )
            return False, "no_conversation"

        # Fetch last 10 messages
        try:
            self._last_messages = await ghl_service.get_conversation_messages(
                conversation_id, limit=10
            )
        except Exception as e:
            logger.warning(
                "human_detector.fetch_failed",
                conversation_id=conversation_id,
                error=str(e),
            )
            self._last_messages = []
            return False, "clear"

        if not self._last_messages:
            return False, "clear"

        # Time window boundary
        cutoff = datetime.now(timezone.utc) - timedelta(
            hours=self.agent_window_hours
        )

        for msg in self._last_messages:
            msg_body = msg.get("body", "") or ""
            msg_body_lower = msg_body.lower()
            msg_source = (msg.get("source", "") or "").lower()
            msg_direction = (msg.get("direction", "") or "").lower()

            # Parse message timestamp
            msg_date = self._parse_timestamp(msg.get("dateAdded", ""))
            if msg_date and msg_date < cutoff:
                continue  # Outside the detection window

            # Only check outbound messages from non-bot sources (human agents)
            if (
                msg_direction == "outbound"
                and msg_source not in BOT_SOURCES
                and msg_source != ""
            ):
                # Check for trigger words in human agent messages
                for trigger in TRIGGER_WORDS:
                    if trigger in msg_body_lower:
                        logger.info(
                            "human_detector.trigger_word",
                            contact_id=contact_id,
                            trigger=trigger,
                            source=msg_source,
                        )
                        return True, "trigger_word"

                # Human agent sent a message within the window
                logger.info(
                    "human_detector.agent_active",
                    contact_id=contact_id,
                    source=msg_source,
                )
                return True, "agent_active"

        logger.debug("human_detector.clear", contact_id=contact_id)
        return False, "clear"

    def get_conversation_context(self) -> dict:
        """Return cached conversation context from the last ``is_human_active()`` call.

        Returns:
            Dict with keys:
            - ``ghlConversationContext``: last 5 messages as readable text
            - ``mostRecentBuilding``: detected building name or None
            - ``spamLimitReached``: True if 3+ consecutive outbound without reply
        """
        if not self._last_messages:
            return {
                "ghlConversationContext": "",
                "mostRecentBuilding": None,
                "spamLimitReached": False,
            }

        # Build conversation context (last 5 messages)
        context_lines = []
        for msg in self._last_messages[:5]:
            direction = msg.get("direction", "unknown")
            body = msg.get("body", "") or ""
            context_lines.append(f"[{direction}] {body}")
        ghl_context = "\n".join(context_lines)

        # Detect building mentions in recent messages
        most_recent_building = self._detect_building(self._last_messages)

        # Check spam limit (consecutive outbound without inbound reply)
        spam_reached = self._check_spam_limit(self._last_messages)

        return {
            "ghlConversationContext": ghl_context,
            "mostRecentBuilding": most_recent_building,
            "spamLimitReached": spam_reached,
        }

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _parse_timestamp(ts: str) -> datetime | None:
        """Parse a GHL timestamp string to a timezone-aware datetime."""
        if not ts:
            return None
        try:
            # GHL uses ISO 8601 format
            # Handle both Z suffix and +00:00
            ts = ts.replace("Z", "+00:00")
            return datetime.fromisoformat(ts)
        except (ValueError, TypeError):
            return None

    @staticmethod
    def _detect_building(messages: list[dict]) -> str | None:
        """Detect the most recently mentioned building name in messages."""
        for msg in messages:
            body = (msg.get("body", "") or "").lower()
            for building in BUILDING_NAMES:
                if building in body:
                    return building
        return None

    @staticmethod
    def _check_spam_limit(messages: list[dict]) -> bool:
        """Check if the last N messages are all outbound (spam limit).

        Returns True if SPAM_LIMIT_THRESHOLD or more consecutive
        outbound messages appear at the start of the message list
        (most recent first) without an inbound reply.
        """
        consecutive_outbound = 0
        for msg in messages:
            if (msg.get("direction", "") or "").lower() == "outbound":
                consecutive_outbound += 1
            else:
                break  # Hit an inbound message, stop counting

        return consecutive_outbound >= SPAM_LIMIT_THRESHOLD
