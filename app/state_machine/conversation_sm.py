"""Conversation state machine using python-statemachine 3.0.

``ConversationSM`` routes conversations through the LEVITAS lifecycle:
GREETING -> QUALIFYING -> SCHEDULING -> HANDOFF, with guard conditions
controlling each transition.

**Thread safety:** SM instances are NOT thread-safe. Create one per Celery
task invocation -- never share across concurrent tasks.

**State persistence:** State is restored from Neo4j via
``from_persisted_state()`` at the start of each task and persisted back
after every transition through the ``after_transition`` hook which calls
``model.on_state_change()``.

**State values:** The SM stores state values using the Python attribute
name (e.g. ``"GREETING"``, ``"QUALIFYING"``), which matches the
``LeadState`` enum values stored in Neo4j.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field

from statemachine import State, StateMachine

logger = logging.getLogger(__name__)


@dataclass
class ConversationModel:
    """Domain model bound to the state machine via ``model=``.

    The ``state`` attribute is read by the SM on construction (to determine
    initial state) and updated by ``on_state_change`` after every transition.
    """

    state: str = "GREETING"
    contact_id: str = ""

    def on_state_change(self, new_state: str) -> None:
        """Called by ``ConversationSM.after_transition``.

        Updates the in-memory state so the caller can persist it to Neo4j.
        """
        self.state = new_state


class ConversationSM(StateMachine):
    """LEVITAS conversation state machine.

    States correspond to ``LeadState`` enum values. Transitions are
    driven by the ``advance`` event with guard conditions that inspect
    keyword arguments passed from the processing pipeline.
    """

    # --- States ---
    GREETING = State("Greeting", initial=True)
    QUALIFYING = State("Qualifying")
    SCHEDULING = State("Scheduling")
    HANDOFF = State("Handoff", final=True)
    CLOSED = State("Closed", final=True)
    BROKER = State("Broker", final=True)
    QUALIFIED = State("Qualified")
    NON_RESPONSIVE = State("NonResponsive")
    RECOVERY = State("Recovery")
    FOLLOW_UP = State("FollowUp")
    RE_ENGAGE = State("ReEngage")

    # --- Transitions ---

    # Primary conversation flow
    advance = (
        GREETING.to(QUALIFYING, cond="has_name_and_interest")
        | QUALIFYING.to(QUALIFIED, cond="is_qualified")
        | QUALIFYING.to(SCHEDULING, cond="wants_appointment")
        | SCHEDULING.to(HANDOFF, cond="has_appointment")
        | SCHEDULING.to(NON_RESPONSIVE, cond="is_unresponsive")
        | QUALIFIED.to(HANDOFF, cond="has_appointment")
        | NON_RESPONSIVE.to(CLOSED, cond="is_unresponsive")
        | RECOVERY.to(GREETING)
        # Follow-up: transition to NON_RESPONSIVE after 3 unanswered attempts
        | FOLLOW_UP.to(NON_RESPONSIVE, cond="is_unresponsive")
        # Follow-up: lead replies -> re-enter qualifying
        | FOLLOW_UP.to(QUALIFYING, cond="has_reply")
        # Re-engage: lead replies -> re-enter qualifying
        | RE_ENGAGE.to(QUALIFYING, cond="has_reply")
    )

    # Explicit escalation from any active state
    escalate = (
        GREETING.to(HANDOFF)
        | QUALIFYING.to(HANDOFF)
        | SCHEDULING.to(HANDOFF)
        | QUALIFIED.to(HANDOFF)
        | NON_RESPONSIVE.to(HANDOFF)
        | RECOVERY.to(HANDOFF)
        | FOLLOW_UP.to(HANDOFF)
        | RE_ENGAGE.to(HANDOFF)
    )

    # Broker classification -- only from early states
    classify_broker = (
        GREETING.to(BROKER, cond="is_broker")
        | QUALIFYING.to(BROKER, cond="is_broker")
    )

    # Close conversation from any active state
    close = (
        GREETING.to(CLOSED)
        | QUALIFYING.to(CLOSED)
        | SCHEDULING.to(CLOSED)
        | QUALIFIED.to(CLOSED)
        | NON_RESPONSIVE.to(CLOSED)
        | RECOVERY.to(CLOSED)
        | FOLLOW_UP.to(CLOSED)
        | RE_ENGAGE.to(CLOSED)
    )

    # Error recovery -- any active state can enter recovery
    recover = (
        GREETING.to(RECOVERY)
        | QUALIFYING.to(RECOVERY)
        | SCHEDULING.to(RECOVERY)
        | QUALIFIED.to(RECOVERY)
        | NON_RESPONSIVE.to(RECOVERY)
        | FOLLOW_UP.to(RECOVERY)
        | RE_ENGAGE.to(RECOVERY)
    )

    # --- Guard conditions ---

    def has_name_and_interest(self, lead_name: str = "", interested: bool = False, **kwargs) -> bool:
        """Lead expressed interest in a property (name is optional for transition)."""
        return bool(interested)

    def is_qualified(self, budget: float = 0, timeline: str = "", **kwargs) -> bool:
        """Lead has provided budget and timeline -- qualification complete."""
        return budget > 0 and bool(timeline)

    def wants_appointment(self, appointment_requested: bool = False, **kwargs) -> bool:
        """Lead has requested to schedule an appointment."""
        return bool(appointment_requested)

    def needs_escalation(self, escalation_flag: bool = False, frustration_detected: bool = False, **kwargs) -> bool:
        """Lead needs to be escalated to a human agent."""
        return bool(escalation_flag) or bool(frustration_detected)

    def has_appointment(self, appointment_set: bool = False, **kwargs) -> bool:
        """Appointment has been confirmed via Cal.com."""
        return bool(appointment_set)

    def is_unresponsive(self, no_response_count: int = 0, **kwargs) -> bool:
        """Lead has not responded to 3+ consecutive messages."""
        return no_response_count >= 3

    def is_broker(self, classification: str = "", **kwargs) -> bool:
        """Lead classified as broker or advertiser by detection gate."""
        return classification in ("broker", "advertiser")

    def has_reply(self, has_reply: bool = False, **kwargs) -> bool:
        """Lead replied during follow-up or re-engagement sequence."""
        return bool(has_reply)

    # --- Hooks ---

    def after_transition(self, event: str, source: State, target: State) -> None:
        """Persist state change via the domain model after every transition.

        The external caller (Celery task) is responsible for writing
        ``model.state`` back to Neo4j after the SM fires this hook.
        """
        logger.info(
            "state_transition",
            extra={
                "event": event,
                "source": source.id,
                "target": target.id,
                "contact_id": getattr(self.model, "contact_id", ""),
            },
        )
        if hasattr(self.model, "on_state_change"):
            self.model.on_state_change(target.id)

    # --- Factory ---

    @classmethod
    def from_persisted_state(cls, state_str: str, contact_id: str) -> "ConversationSM":
        """Restore a state machine to a previously persisted state.

        Args:
            state_str: The state value from Neo4j (e.g. ``"QUALIFYING"``).
                Must match a State's ``id`` (the Python attribute name).
                Falls back to ``"GREETING"`` if the value is unknown.
            contact_id: GHL contact ID for the lead.

        Returns:
            A ``ConversationSM`` instance positioned at the given state.
        """
        # Validate state_str against known states
        valid_states = {s.id for s in cls.states}
        if state_str not in valid_states:
            logger.warning(
                "unknown_state_restoring_to_greeting",
                extra={"state_str": state_str, "contact_id": contact_id},
            )
            state_str = "GREETING"

        model = ConversationModel(state=state_str, contact_id=contact_id)
        sm = cls(model=model, start_value=state_str)
        return sm
