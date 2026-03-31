"""Tests for LeadState enum -- all 12 values present with correct string values."""

from __future__ import annotations

import pytest

from app.state_machine.states import LeadState


# All expected enum members and their string values
EXPECTED_STATES = {
    "GREETING": "GREETING",
    "QUALIFYING": "QUALIFYING",
    "BUILDING_INFO": "BUILDING_INFO",
    "SCHEDULING": "SCHEDULING",
    "HANDOFF": "HANDOFF",
    "CLOSED": "CLOSED",
    "BROKER": "BROKER",
    "QUALIFIED": "QUALIFIED",
    "RE_ENGAGE": "RE_ENGAGE",
    "FOLLOW_UP": "FOLLOW_UP",
    "NON_RESPONSIVE": "NON_RESPONSIVE",
    "RECOVERY": "RECOVERY",
}


class TestLeadStateEnum:
    """Validate LeadState enum has all expected members with correct values."""

    def test_enum_has_exactly_12_members(self):
        """LeadState should have exactly 12 members."""
        assert len(LeadState) == 12

    @pytest.mark.parametrize(
        "name,expected_value",
        list(EXPECTED_STATES.items()),
        ids=list(EXPECTED_STATES.keys()),
    )
    def test_enum_member_exists_with_correct_value(self, name: str, expected_value: str):
        """Each enum member has the expected string value (UPPERCASE, matches Neo4j)."""
        member = LeadState[name]
        assert member.value == expected_value

    def test_enum_is_strenum(self):
        """LeadState inherits from StrEnum for direct string comparison."""
        assert LeadState.GREETING == "GREETING"
        assert isinstance(LeadState.GREETING, str)

    def test_no_unexpected_members(self):
        """All enum members are in the expected set (no extras crept in)."""
        actual_names = {m.name for m in LeadState}
        expected_names = set(EXPECTED_STATES.keys())
        assert actual_names == expected_names, (
            f"Unexpected members: {actual_names - expected_names}, "
            f"Missing members: {expected_names - actual_names}"
        )

    def test_values_are_uppercase_python_names(self):
        """All values use UPPERCASE Python attribute names (not Title Case)."""
        for member in LeadState:
            assert member.value == member.value.upper(), (
                f"{member.name} value '{member.value}' is not uppercase"
            )
            assert member.value == member.name, (
                f"{member.name} value '{member.value}' does not match attribute name"
            )
