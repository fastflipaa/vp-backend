"""StateResolver tests -- resolves lead state from Neo4j with overrides.

Mocks LeadRepository since tests run without Neo4j. Tests cover:
new lead -> GREETING, existing lead -> stored state, auto-trigger override,
messageType overrides (RE_ENGAGE, FOLLOW_UP), and invalid state fallback.
"""

from __future__ import annotations

from unittest.mock import AsyncMock

import pytest

from app.services.state_resolver import StateResolver


@pytest.fixture()
def mock_lead_repo():
    """Mock LeadRepository with async get_lead_by_phone."""
    repo = AsyncMock()
    repo.get_lead_by_phone = AsyncMock(return_value=None)
    return repo


@pytest.fixture()
def resolver(mock_lead_repo) -> StateResolver:
    """StateResolver with mocked LeadRepository."""
    return StateResolver(mock_lead_repo)


class TestNewLead:
    """New leads (not found in Neo4j) default to GREETING."""

    @pytest.mark.asyncio
    async def test_new_lead_returns_greeting(self, resolver: StateResolver):
        """Lead not in Neo4j -> GREETING with is_new_lead=True."""
        result = await resolver.resolve(phone="+5215512345678")
        assert result["state"] == "GREETING"
        assert result["is_new_lead"] is True
        assert result["lead_data"] == {}


class TestExistingLead:
    """Existing leads use their stored state from Neo4j."""

    @pytest.mark.asyncio
    async def test_existing_lead_returns_stored_state(self, resolver: StateResolver, mock_lead_repo):
        """Lead with current_state in Neo4j uses that state."""
        mock_lead_repo.get_lead_by_phone.return_value = {
            "current_state": "QUALIFYING",
            "name": "Carlos",
            "phone": "+5215512345678",
        }
        result = await resolver.resolve(phone="+5215512345678")
        assert result["state"] == "QUALIFYING"
        assert result["is_new_lead"] is False
        assert result["lead_data"]["name"] == "Carlos"

    @pytest.mark.asyncio
    async def test_empty_state_defaults_to_greeting(self, resolver: StateResolver, mock_lead_repo):
        """Lead with empty current_state defaults to GREETING."""
        mock_lead_repo.get_lead_by_phone.return_value = {
            "current_state": "",
            "name": "Ana",
        }
        result = await resolver.resolve(phone="+5215512345678")
        assert result["state"] == "GREETING"
        assert result["is_new_lead"] is False

    @pytest.mark.asyncio
    async def test_none_state_defaults_to_greeting(self, resolver: StateResolver, mock_lead_repo):
        """Lead with None current_state defaults to GREETING."""
        mock_lead_repo.get_lead_by_phone.return_value = {
            "current_state": None,
            "name": "Pedro",
        }
        result = await resolver.resolve(phone="+5215512345678")
        assert result["state"] == "GREETING"


class TestOverrides:
    """Message type and auto-trigger overrides."""

    @pytest.mark.asyncio
    async def test_auto_trigger_forces_greeting(self, resolver: StateResolver, mock_lead_repo):
        """is_auto_trigger=True forces state to GREETING regardless of stored state."""
        mock_lead_repo.get_lead_by_phone.return_value = {
            "current_state": "SCHEDULING",
        }
        result = await resolver.resolve(phone="+5215512345678", is_auto_trigger=True)
        assert result["state"] == "GREETING"

    @pytest.mark.asyncio
    async def test_re_engagement_override(self, resolver: StateResolver, mock_lead_repo):
        """message_type='re_engagement' overrides to RE_ENGAGE."""
        mock_lead_repo.get_lead_by_phone.return_value = {
            "current_state": "QUALIFYING",
        }
        result = await resolver.resolve(phone="+5215512345678", message_type="re_engagement")
        assert result["state"] == "RE_ENGAGE"

    @pytest.mark.asyncio
    async def test_follow_up_override(self, resolver: StateResolver, mock_lead_repo):
        """message_type='follow_up' overrides to FOLLOW_UP."""
        mock_lead_repo.get_lead_by_phone.return_value = {
            "current_state": "QUALIFYING",
        }
        result = await resolver.resolve(phone="+5215512345678", message_type="follow_up")
        assert result["state"] == "FOLLOW_UP"

    @pytest.mark.asyncio
    async def test_post_appointment_override(self, resolver: StateResolver, mock_lead_repo):
        """message_type='post_appointment' overrides to SCHEDULING."""
        mock_lead_repo.get_lead_by_phone.return_value = {
            "current_state": "HANDOFF",
        }
        result = await resolver.resolve(phone="+5215512345678", message_type="post_appointment")
        assert result["state"] == "SCHEDULING"


class TestInvalidStateFallback:
    """Invalid states in Neo4j fall back to GREETING."""

    @pytest.mark.asyncio
    async def test_invalid_state_falls_back_to_greeting(self, resolver: StateResolver, mock_lead_repo):
        """Unknown state string in Neo4j falls back to GREETING."""
        mock_lead_repo.get_lead_by_phone.return_value = {
            "current_state": "TOTALLY_INVALID_STATE",
        }
        result = await resolver.resolve(phone="+5215512345678")
        assert result["state"] == "GREETING"

    @pytest.mark.asyncio
    async def test_valid_states_pass_through(self, resolver: StateResolver, mock_lead_repo):
        """All valid LeadState values pass validation."""
        for valid_state in ["GREETING", "QUALIFYING", "SCHEDULING", "HANDOFF", "CLOSED",
                            "BROKER", "QUALIFIED", "RE_ENGAGE", "FOLLOW_UP",
                            "NON_RESPONSIVE", "RECOVERY", "BUILDING_INFO"]:
            mock_lead_repo.get_lead_by_phone.return_value = {
                "current_state": valid_state,
            }
            result = await resolver.resolve(phone="+5215512345678")
            assert result["state"] == valid_state, (
                f"State {valid_state} should pass validation"
            )
