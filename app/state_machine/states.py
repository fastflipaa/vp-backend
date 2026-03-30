"""Lead lifecycle states from the n8n workflow audit.

Each state maps to a stage in the LEVITAS conversation flow. The string
values match the state identifiers stored in Neo4j ``Lead.current_state``.

States are grouped by lifecycle phase:

Active conversation:
    GREETING, QUALIFYING, BUILDING_INFO, SCHEDULING

Terminal / handoff:
    HANDOFF, CLOSED, BROKER, QUALIFIED

Re-engagement:
    RE_ENGAGE, FOLLOW_UP, NON_RESPONSIVE

Error recovery:
    RECOVERY
"""

from enum import StrEnum


class LeadState(StrEnum):
    """All conversation states tracked across the LEVITAS lifecycle."""

    # --- Active conversation ---
    GREETING = "GREETING"
    """Initial state for new leads or leads with unknown state."""

    QUALIFYING = "QUALIFYING"
    """5-step qualification funnel: name, interest, budget, timeline, email."""

    BUILDING_INFO = "BUILDING_INFO"
    """Building details discussion -- currently alias to QUALIFYING in n8n."""

    SCHEDULING = "SCHEDULING"
    """Appointment scheduling via Cal.com."""

    # --- Terminal / handoff ---
    HANDOFF = "HANDOFF"
    """Human agent handoff (Fernando / Lorena)."""

    CLOSED = "CLOSED"
    """Conversation ended -- no further automated contact."""

    BROKER = "BROKER"
    """Classified as broker or advertiser -- terminal, no response."""

    QUALIFIED = "QUALIFIED"
    """Fully qualified, ready for handoff to human agent."""

    # --- Re-engagement ---
    RE_ENGAGE = "RE_ENGAGE"
    """Proactive re-engagement -- Phase 17 scope."""

    FOLLOW_UP = "FOLLOW_UP"
    """Follow-up sequence after initial conversation."""

    NON_RESPONSIVE = "NON_RESPONSIVE"
    """3+ unanswered messages -- awaiting re-engage or close."""

    # --- Error recovery ---
    RECOVERY = "RECOVERY"
    """Error recovery path -- restarts to GREETING."""
