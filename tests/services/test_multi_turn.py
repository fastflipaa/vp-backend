"""Tests for multi-turn conversation support.

Tests HumanAgentDetector.get_structured_turns() and ClaudeService.generate()
conversation_history parameter assembly.
"""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from app.services.human_agent_detector import HumanAgentDetector

pytestmark = pytest.mark.asyncio


# ---------------------------------------------------------------------------
# get_structured_turns tests
# ---------------------------------------------------------------------------

class TestGetStructuredTurns:
    def _make_detector(self, messages: list[dict]) -> HumanAgentDetector:
        """Helper to create detector with pre-set _last_messages."""
        d = HumanAgentDetector()
        d._last_messages = messages
        return d

    async def test_maps_inbound_to_user_role(self):
        """Inbound messages map to role='user'."""
        d = self._make_detector([
            {"direction": "inbound", "body": "Hola", "source": ""},
        ])

        turns = d.get_structured_turns()

        assert turns == [{"role": "user", "content": "Hola"}]

    async def test_maps_outbound_to_assistant_role(self):
        """Outbound bot messages map to role='assistant'."""
        d = self._make_detector([
            {"direction": "outbound", "body": "Bienvenido", "source": "bot"},
        ])

        turns = d.get_structured_turns()

        assert turns == [{"role": "assistant", "content": "Bienvenido"}]

    async def test_filters_empty_body(self):
        """Messages with empty or None body are filtered out."""
        d = self._make_detector([
            {"direction": "inbound", "body": "", "source": ""},
            {"direction": "inbound", "body": None, "source": ""},
            {"direction": "inbound", "body": "Real message", "source": ""},
        ])

        turns = d.get_structured_turns()

        assert len(turns) == 1
        assert turns[0]["content"] == "Real message"

    async def test_filters_human_agent_messages(self):
        """Outbound from non-bot source (human agent) are filtered out."""
        d = self._make_detector([
            {"direction": "outbound", "body": "Human response", "source": "sales_team"},
        ])

        turns = d.get_structured_turns()

        assert turns == []

    async def test_keeps_bot_source_messages(self):
        """Outbound from bot/system/automation/workflow/api kept as assistant."""
        messages = []
        for source in ["bot", "system", "automation", "workflow", "api"]:
            messages.append({"direction": "outbound", "body": f"From {source}", "source": source})

        d = self._make_detector(messages)
        turns = d.get_structured_turns()

        # After reverse + merge (all consecutive assistant), should be 1 merged turn
        assert len(turns) >= 1
        for turn in turns:
            assert turn["role"] == "assistant"

    async def test_reverses_to_chronological(self):
        """GHL newest-first order reversed to oldest-first for Claude."""
        d = self._make_detector([
            {"direction": "outbound", "body": "Response", "source": "bot"},
            {"direction": "inbound", "body": "Question", "source": ""},
        ])

        turns = d.get_structured_turns()

        # After reverse: Question (user) then Response (assistant)
        assert turns[0]["role"] == "user"
        assert turns[0]["content"] == "Question"
        assert turns[1]["role"] == "assistant"
        assert turns[1]["content"] == "Response"

    async def test_caps_at_10_messages(self):
        """Output capped at 10 messages maximum."""
        messages = []
        for i in range(15):
            direction = "inbound" if i % 2 == 0 else "outbound"
            source = "" if direction == "inbound" else "bot"
            messages.append({"direction": direction, "body": f"Message {i}", "source": source})

        d = self._make_detector(messages)
        turns = d.get_structured_turns()

        # After reverse and cap, should be <= 10
        assert len(turns) <= 10

    async def test_truncates_body_at_500_chars(self):
        """Message body truncated to 500 characters."""
        long_body = "A" * 600
        d = self._make_detector([
            {"direction": "inbound", "body": long_body, "source": ""},
        ])

        turns = d.get_structured_turns()

        assert len(turns[0]["content"]) == 500

    async def test_merges_consecutive_same_role(self):
        """Consecutive same-role messages merged with newline separator."""
        d = self._make_detector([
            # GHL newest-first -- after reverse these become chronological
            {"direction": "inbound", "body": "Third", "source": ""},
            {"direction": "inbound", "body": "Second", "source": ""},
            {"direction": "inbound", "body": "First", "source": ""},
        ])

        turns = d.get_structured_turns()

        # All 3 inbound reversed to First, Second, Third -- then merged
        assert len(turns) == 1
        assert turns[0]["role"] == "user"
        assert "First" in turns[0]["content"]
        assert "Third" in turns[0]["content"]

    async def test_empty_messages_returns_empty(self):
        """Empty _last_messages returns empty list."""
        d = self._make_detector([])

        turns = d.get_structured_turns()

        assert turns == []


class TestHumanAgentDetectorHelpers:
    """Test internal helper methods of HumanAgentDetector."""

    def test_parse_timestamp_valid_iso(self):
        """_parse_timestamp parses ISO 8601 string."""
        d = HumanAgentDetector()
        result = d._parse_timestamp("2026-01-01T12:00:00Z")
        assert result is not None
        assert result.year == 2026

    def test_parse_timestamp_empty_returns_none(self):
        """_parse_timestamp returns None for empty string."""
        d = HumanAgentDetector()
        assert d._parse_timestamp("") is None

    def test_parse_timestamp_invalid_returns_none(self):
        """_parse_timestamp returns None for invalid string."""
        d = HumanAgentDetector()
        assert d._parse_timestamp("not-a-date") is None

    def test_detect_building(self):
        """_detect_building finds building name in messages."""
        d = HumanAgentDetector()
        messages = [
            {"body": "Me interesa el edificio Thompson en Polanco"},
        ]
        result = d._detect_building(messages)
        assert result == "thompson"

    def test_detect_building_no_match(self):
        """_detect_building returns None when no building found."""
        d = HumanAgentDetector()
        messages = [
            {"body": "Hola, quiero informacion"},
        ]
        result = d._detect_building(messages)
        assert result is None

    def test_check_spam_limit_reached(self):
        """_check_spam_limit returns True for 3+ consecutive outbound."""
        d = HumanAgentDetector()
        messages = [
            {"direction": "outbound"},
            {"direction": "outbound"},
            {"direction": "outbound"},
            {"direction": "inbound"},
        ]
        assert d._check_spam_limit(messages) is True

    def test_check_spam_limit_not_reached(self):
        """_check_spam_limit returns False for < 3 consecutive outbound."""
        d = HumanAgentDetector()
        messages = [
            {"direction": "outbound"},
            {"direction": "inbound"},
            {"direction": "outbound"},
        ]
        assert d._check_spam_limit(messages) is False

    def test_get_conversation_context_empty(self):
        """get_conversation_context with no messages returns defaults."""
        d = HumanAgentDetector()
        d._last_messages = []
        result = d.get_conversation_context()
        assert result["ghlConversationContext"] == ""
        assert result["mostRecentBuilding"] is None
        assert result["spamLimitReached"] is False

    def test_get_conversation_context_with_messages(self):
        """get_conversation_context with messages returns context string."""
        d = HumanAgentDetector()
        d._last_messages = [
            {"direction": "inbound", "body": "Hola, me interesa thompson"},
            {"direction": "outbound", "body": "Bienvenido!"},
        ]
        result = d.get_conversation_context()
        assert "inbound" in result["ghlConversationContext"]
        assert result["mostRecentBuilding"] == "thompson"
        assert result["spamLimitReached"] is False


# ---------------------------------------------------------------------------
# ClaudeService.generate() conversation_history tests
# ---------------------------------------------------------------------------

class TestClaudeServiceConversationHistory:
    async def test_generate_without_history(self, redis_client):
        """generate() without history sends single user message."""
        from app.services.claude_service import ClaudeService

        mock_client = MagicMock()
        mock_response = MagicMock()
        mock_response.content = [MagicMock(text="response")]
        mock_response.usage = MagicMock(input_tokens=10, output_tokens=5)
        mock_response.stop_reason = "end_turn"
        mock_client.messages.create = AsyncMock(return_value=mock_response)

        with patch("app.services.claude_service.get_claude_client", return_value=mock_client):
            service = ClaudeService(redis_client)
            result = await service.generate("system prompt", "hello")

        assert result == "response"
        call_kwargs = mock_client.messages.create.call_args.kwargs
        assert call_kwargs["messages"] == [{"role": "user", "content": "hello"}]

    async def test_generate_with_history(self, redis_client):
        """generate() with history prepends history to messages."""
        from app.services.claude_service import ClaudeService

        mock_client = MagicMock()
        mock_response = MagicMock()
        mock_response.content = [MagicMock(text="response")]
        mock_response.usage = MagicMock(input_tokens=10, output_tokens=5)
        mock_response.stop_reason = "end_turn"
        mock_client.messages.create = AsyncMock(return_value=mock_response)

        history = [
            {"role": "user", "content": "prev_q"},
            {"role": "assistant", "content": "prev_a"},
        ]

        with patch("app.services.claude_service.get_claude_client", return_value=mock_client):
            service = ClaudeService(redis_client)
            result = await service.generate("system", "hello", conversation_history=history)

        call_kwargs = mock_client.messages.create.call_args.kwargs
        messages = call_kwargs["messages"]
        assert len(messages) == 3  # 2 history + 1 current
        assert messages[0]["role"] == "user"
        assert messages[2]["content"] == "hello"

    async def test_history_caps_at_20_messages(self, redis_client):
        """Messages capped at 20 items total."""
        from app.services.claude_service import ClaudeService

        mock_client = MagicMock()
        mock_response = MagicMock()
        mock_response.content = [MagicMock(text="response")]
        mock_response.usage = MagicMock(input_tokens=10, output_tokens=5)
        mock_response.stop_reason = "end_turn"
        mock_client.messages.create = AsyncMock(return_value=mock_response)

        # 22 history messages + 1 user = 23 total
        history = []
        for i in range(22):
            role = "user" if i % 2 == 0 else "assistant"
            history.append({"role": role, "content": f"msg {i}"})

        with patch("app.services.claude_service.get_claude_client", return_value=mock_client):
            service = ClaudeService(redis_client)
            await service.generate("system", "hello", conversation_history=history)

        call_kwargs = mock_client.messages.create.call_args.kwargs
        messages = call_kwargs["messages"]
        assert len(messages) <= 20

    async def test_history_ensures_first_message_is_user(self, redis_client):
        """First message in final list has role='user' (Claude API requirement)."""
        from app.services.claude_service import ClaudeService

        mock_client = MagicMock()
        mock_response = MagicMock()
        mock_response.content = [MagicMock(text="response")]
        mock_response.usage = MagicMock(input_tokens=10, output_tokens=5)
        mock_response.stop_reason = "end_turn"
        mock_client.messages.create = AsyncMock(return_value=mock_response)

        # History starts with assistant (should be stripped)
        history = [
            {"role": "assistant", "content": "old response"},
            {"role": "user", "content": "question"},
            {"role": "assistant", "content": "answer"},
        ]

        with patch("app.services.claude_service.get_claude_client", return_value=mock_client):
            service = ClaudeService(redis_client)
            await service.generate("system", "hello", conversation_history=history)

        call_kwargs = mock_client.messages.create.call_args.kwargs
        messages = call_kwargs["messages"]
        assert messages[0]["role"] == "user"

    async def test_learning_context_appended_to_system_prompt(self, redis_client):
        """Learning context is appended to system prompt."""
        from app.services.claude_service import ClaudeService

        mock_client = MagicMock()
        mock_response = MagicMock()
        mock_response.content = [MagicMock(text="response")]
        mock_response.usage = MagicMock(input_tokens=10, output_tokens=5)
        mock_response.stop_reason = "end_turn"
        mock_client.messages.create = AsyncMock(return_value=mock_response)

        with patch("app.services.claude_service.get_claude_client", return_value=mock_client):
            service = ClaudeService(redis_client)
            service.learning_context = "LEARNING CONTEXT: test lesson"
            await service.generate("Base system prompt", "hello")

        call_kwargs = mock_client.messages.create.call_args.kwargs
        system_prompt = call_kwargs["system"]
        assert "LEARNING CONTEXT: test lesson" in system_prompt
        assert "Base system prompt" in system_prompt


class TestClaudeServiceClassify:
    """Test ClaudeService.classify() method."""

    async def test_classify_returns_classification(self, redis_client):
        """classify() returns classification dict from valid JSON response."""
        from app.services.claude_service import ClaudeService

        mock_client = MagicMock()
        mock_response = MagicMock()
        mock_response.content = [MagicMock(text='{"classification": "potential_lead", "confidence": 0.95, "reason": "interested"}')]
        mock_response.usage = MagicMock(input_tokens=10, output_tokens=5)
        mock_response.stop_reason = "end_turn"
        mock_client.messages.create = AsyncMock(return_value=mock_response)

        with patch("app.services.claude_service.get_claude_client", return_value=mock_client):
            service = ClaudeService(redis_client)
            result = await service.classify("Me interesa un depto", "+5215512345678")

        assert result["classification"] == "potential_lead"
        assert result["confidence"] == 0.95

    async def test_classify_json_parse_error_returns_ambiguous(self, redis_client):
        """classify() returns ambiguous on JSON parse error."""
        from app.services.claude_service import ClaudeService

        mock_client = MagicMock()
        mock_response = MagicMock()
        mock_response.content = [MagicMock(text="not valid json at all")]
        mock_response.usage = MagicMock(input_tokens=10, output_tokens=5)
        mock_response.stop_reason = "end_turn"
        mock_client.messages.create = AsyncMock(return_value=mock_response)

        with patch("app.services.claude_service.get_claude_client", return_value=mock_client):
            service = ClaudeService(redis_client)
            result = await service.classify("Hola", "+5215512345678")

        assert result["classification"] == "ambiguous"
        assert result["reason"] == "parse_error"

    async def test_classify_circuit_open_returns_ambiguous(self, redis_client):
        """classify() returns ambiguous when circuit breaker is open."""
        from app.services.claude_service import ClaudeService

        service = ClaudeService(redis_client)
        # Open the circuit by recording enough failures
        for _ in range(3):
            service.circuit_breaker.record_failure()

        result = await service.classify("Hola", "+5215512345678")

        assert result["classification"] == "ambiguous"
        assert result["reason"] == "circuit_open"

    async def test_is_circuit_open(self, redis_client):
        """is_circuit_open() returns correct state."""
        from app.services.claude_service import ClaudeService

        service = ClaudeService(redis_client)
        assert service.is_circuit_open() is False

        for _ in range(3):
            service.circuit_breaker.record_failure()

        assert service.is_circuit_open() is True
