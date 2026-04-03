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
