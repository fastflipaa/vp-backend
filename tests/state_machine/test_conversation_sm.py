"""Parametrized transition tests for ConversationSM.

Covers every valid transition (~30 cases), invalid transitions from
terminal/final states, and state restoration via from_persisted_state().

Guard conditions are tested by passing the correct kwargs to satisfy
the guard functions discovered via source inspection:
  - has_name_and_interest(lead_name, interested)
  - is_qualified(budget, timeline)
  - wants_appointment(appointment_requested)
  - has_appointment(appointment_set)
  - is_unresponsive(no_response_count >= 3)
  - is_broker(classification in ("broker", "advertiser"))
"""

from __future__ import annotations

import warnings

import pytest
from statemachine.exceptions import TransitionNotAllowed

from app.state_machine.conversation_sm import ConversationModel, ConversationSM

# Suppress python-statemachine deprecation warning for current_state
warnings.filterwarnings("ignore", message=".*current_state.*deprecated.*")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_sm(start_state: str) -> ConversationSM:
    """Create a ConversationSM positioned at start_state."""
    return ConversationSM.from_persisted_state(start_state, "test_contact_001")


# ---------------------------------------------------------------------------
# Valid transitions -- parametrized
# ---------------------------------------------------------------------------

# Each tuple: (start_state, event, kwargs, expected_end_state)
VALID_TRANSITIONS = [
    # --- advance event (8 transitions with guards) ---
    ("GREETING", "advance", {"lead_name": "Juan", "interested": True}, "QUALIFYING"),
    ("QUALIFYING", "advance", {"budget": 500000, "timeline": "6 months"}, "QUALIFIED"),
    ("QUALIFYING", "advance", {"appointment_requested": True}, "SCHEDULING"),
    ("SCHEDULING", "advance", {"appointment_set": True}, "HANDOFF"),
    ("SCHEDULING", "advance", {"no_response_count": 3}, "NON_RESPONSIVE"),
    ("QUALIFIED", "advance", {"appointment_set": True}, "HANDOFF"),
    ("NON_RESPONSIVE", "advance", {"no_response_count": 3}, "CLOSED"),
    ("RECOVERY", "advance", {}, "GREETING"),
    # --- escalate event (6 transitions, no guards) ---
    ("GREETING", "escalate", {}, "HANDOFF"),
    ("QUALIFYING", "escalate", {}, "HANDOFF"),
    ("SCHEDULING", "escalate", {}, "HANDOFF"),
    ("QUALIFIED", "escalate", {}, "HANDOFF"),
    ("NON_RESPONSIVE", "escalate", {}, "HANDOFF"),
    ("RECOVERY", "escalate", {}, "HANDOFF"),
    # --- classify_broker event (2 transitions with guards) ---
    ("GREETING", "classify_broker", {"classification": "broker"}, "BROKER"),
    ("QUALIFYING", "classify_broker", {"classification": "broker"}, "BROKER"),
    # Also test "advertiser" classification
    ("GREETING", "classify_broker", {"classification": "advertiser"}, "BROKER"),
    ("QUALIFYING", "classify_broker", {"classification": "advertiser"}, "BROKER"),
    # --- close event (6 transitions, no guards) ---
    ("GREETING", "close", {}, "CLOSED"),
    ("QUALIFYING", "close", {}, "CLOSED"),
    ("SCHEDULING", "close", {}, "CLOSED"),
    ("QUALIFIED", "close", {}, "CLOSED"),
    ("NON_RESPONSIVE", "close", {}, "CLOSED"),
    ("RECOVERY", "close", {}, "CLOSED"),
    # --- recover event (5 transitions, no guards) ---
    ("GREETING", "recover", {}, "RECOVERY"),
    ("QUALIFYING", "recover", {}, "RECOVERY"),
    ("SCHEDULING", "recover", {}, "RECOVERY"),
    ("QUALIFIED", "recover", {}, "RECOVERY"),
    ("NON_RESPONSIVE", "recover", {}, "RECOVERY"),
]


@pytest.mark.parametrize(
    "start_state,event,kwargs,expected_end",
    VALID_TRANSITIONS,
    ids=[
        f"{s}-{e}->{exp}"
        for s, e, _, exp in VALID_TRANSITIONS
    ],
)
def test_valid_transition(start_state: str, event: str, kwargs: dict, expected_end: str):
    """Every valid transition moves the SM to the expected target state."""
    sm = _make_sm(start_state)
    sm.send(event, **kwargs)
    assert sm.model.state == expected_end


# ---------------------------------------------------------------------------
# Invalid transitions from terminal/final states
# ---------------------------------------------------------------------------

# Terminal states cannot advance, escalate, close, recover, or classify_broker
INVALID_TRANSITIONS = [
    ("CLOSED", "advance"),
    ("CLOSED", "escalate"),
    ("CLOSED", "close"),
    ("CLOSED", "recover"),
    ("BROKER", "advance"),
    ("BROKER", "escalate"),
    ("BROKER", "close"),
    ("BROKER", "recover"),
    ("HANDOFF", "advance"),
    ("HANDOFF", "escalate"),
    ("HANDOFF", "close"),
    ("HANDOFF", "recover"),
]


@pytest.mark.parametrize(
    "start_state,event",
    INVALID_TRANSITIONS,
    ids=[f"{s}-{e}-INVALID" for s, e in INVALID_TRANSITIONS],
)
def test_invalid_transition_raises(start_state: str, event: str):
    """Transitions from terminal states raise TransitionNotAllowed."""
    sm = _make_sm(start_state)
    with pytest.raises(TransitionNotAllowed):
        sm.send(event)


# ---------------------------------------------------------------------------
# Guard failures (valid source state, but guard condition not met)
# ---------------------------------------------------------------------------


class TestGuardFailures:
    """Guards that return False prevent transitions."""

    def test_advance_greeting_without_name(self):
        """GREETING->QUALIFYING requires lead_name AND interested."""
        sm = _make_sm("GREETING")
        with pytest.raises(TransitionNotAllowed):
            sm.send("advance", lead_name="", interested=True)

    def test_advance_greeting_without_interest(self):
        """GREETING->QUALIFYING requires interested=True."""
        sm = _make_sm("GREETING")
        with pytest.raises(TransitionNotAllowed):
            sm.send("advance", lead_name="Juan", interested=False)

    def test_advance_qualifying_no_budget(self):
        """QUALIFYING->QUALIFIED requires budget > 0."""
        sm = _make_sm("QUALIFYING")
        with pytest.raises(TransitionNotAllowed):
            sm.send("advance", budget=0, timeline="6 months")

    def test_advance_scheduling_no_appointment(self):
        """SCHEDULING->HANDOFF requires appointment_set=True."""
        sm = _make_sm("SCHEDULING")
        with pytest.raises(TransitionNotAllowed):
            sm.send("advance", appointment_set=False, no_response_count=0)

    def test_classify_broker_wrong_classification(self):
        """classify_broker guard requires classification in (broker, advertiser)."""
        sm = _make_sm("GREETING")
        with pytest.raises(TransitionNotAllowed):
            sm.send("classify_broker", classification="normal")

    def test_advance_non_responsive_low_count(self):
        """NON_RESPONSIVE->CLOSED requires no_response_count >= 3."""
        sm = _make_sm("NON_RESPONSIVE")
        with pytest.raises(TransitionNotAllowed):
            sm.send("advance", no_response_count=2)


# ---------------------------------------------------------------------------
# State restoration
# ---------------------------------------------------------------------------


class TestStateRestoration:
    """ConversationSM.from_persisted_state restores SM to correct position."""

    @pytest.mark.parametrize(
        "state",
        ["GREETING", "QUALIFYING", "SCHEDULING", "QUALIFIED", "NON_RESPONSIVE", "RECOVERY"],
    )
    def test_restoration_to_active_states(self, state: str):
        """SM can be restored to any active (non-final) state."""
        sm = _make_sm(state)
        assert sm.model.state == state
        assert sm.current_state.id == state

    @pytest.mark.parametrize("state", ["HANDOFF", "CLOSED", "BROKER"])
    def test_restoration_to_final_states(self, state: str):
        """SM can be restored to final states (no further transitions allowed)."""
        sm = _make_sm(state)
        assert sm.model.state == state
        assert sm.current_state.id == state

    def test_unknown_state_falls_back_to_greeting(self):
        """Unknown state strings default to GREETING."""
        sm = _make_sm("NONEXISTENT_STATE")
        assert sm.model.state == "GREETING"

    def test_empty_state_falls_back_to_greeting(self):
        """Empty string state defaults to GREETING."""
        sm = _make_sm("")
        assert sm.model.state == "GREETING"

    def test_contact_id_preserved(self):
        """Contact ID is stored on the model for logging."""
        sm = _make_sm("QUALIFYING")
        assert sm.model.contact_id == "test_contact_001"


# ---------------------------------------------------------------------------
# Model state update hook
# ---------------------------------------------------------------------------


class TestConversationModel:
    """ConversationModel tracks state changes from SM transitions."""

    def test_model_state_updates_after_transition(self):
        """model.state reflects the latest SM state after transition."""
        sm = _make_sm("GREETING")
        assert sm.model.state == "GREETING"
        sm.send("advance", lead_name="Juan", interested=True)
        assert sm.model.state == "QUALIFYING"

    def test_model_on_state_change(self):
        """on_state_change updates the model state attribute."""
        model = ConversationModel(state="GREETING", contact_id="c1")
        model.on_state_change("QUALIFYING")
        assert model.state == "QUALIFYING"

    def test_model_default_state(self):
        """Default model state is GREETING."""
        model = ConversationModel()
        assert model.state == "GREETING"
        assert model.contact_id == ""
