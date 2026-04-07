"""ProcessorRouter tests -- pure state-to-processor mapping.

Verifies that active states map to correct processor classes and
terminal/deferred states return None.
"""

from __future__ import annotations

import pytest

from app.services.processor_router import PROCESSOR_MAP, ProcessorRouter


@pytest.fixture()
def router() -> ProcessorRouter:
    return ProcessorRouter()


class TestProcessorRouter:
    """ProcessorRouter.get_processor() maps states to processor classes."""

    def test_greeting_maps_to_greeting_processor(self, router: ProcessorRouter):
        """GREETING -> GreetingProcessor."""
        processor = router.get_processor("GREETING")
        assert processor is not None
        assert processor.__name__ == "GreetingProcessor"

    def test_qualifying_maps_to_qualifying_processor(self, router: ProcessorRouter):
        """QUALIFYING -> QualifyingProcessor."""
        processor = router.get_processor("QUALIFYING")
        assert processor is not None
        assert processor.__name__ == "QualifyingProcessor"

    def test_scheduling_maps_to_scheduling_processor(self, router: ProcessorRouter):
        """SCHEDULING -> SchedulingProcessor."""
        processor = router.get_processor("SCHEDULING")
        assert processor is not None
        assert processor.__name__ == "SchedulingProcessor"

    def test_handoff_maps_to_handoff_processor(self, router: ProcessorRouter):
        """HANDOFF -> HandoffProcessor."""
        processor = router.get_processor("HANDOFF")
        assert processor is not None
        assert processor.__name__ == "HandoffProcessor"

    def test_qualified_maps_to_handoff_processor(self, router: ProcessorRouter):
        """QUALIFIED -> HandoffProcessor (qualified leads go to handoff)."""
        processor = router.get_processor("QUALIFIED")
        assert processor is not None
        assert processor.__name__ == "HandoffProcessor"

    def test_building_info_maps_to_qualifying_processor(self, router: ProcessorRouter):
        """BUILDING_INFO -> QualifyingProcessor (merged in n8n)."""
        processor = router.get_processor("BUILDING_INFO")
        assert processor is not None
        assert processor.__name__ == "QualifyingProcessor"

    @pytest.mark.parametrize(
        "terminal_state",
        # Truly terminal/no-processor states. FOLLOW_UP and RE_ENGAGE were
        # added as real processors in Phases 19 and 20 -- see
        # test_followup_maps_to_followup_processor and
        # test_reengage_maps_to_reengagement_processor below.
        ["BROKER", "CLOSED", "NON_RESPONSIVE", "RECOVERY"],
    )
    def test_terminal_and_deferred_states_return_none(self, router: ProcessorRouter, terminal_state: str):
        """Truly terminal states have no processor (returns None)."""
        assert router.get_processor(terminal_state) is None

    def test_followup_maps_to_followup_processor(self, router: ProcessorRouter):
        """FOLLOW_UP -> FollowUpProcessor (Phase 19)."""
        processor = router.get_processor("FOLLOW_UP")
        assert processor is not None
        assert processor.__name__ == "FollowUpProcessor"

    def test_reengage_maps_to_reengagement_processor(self, router: ProcessorRouter):
        """RE_ENGAGE -> ReEngagementProcessor (Phase 20)."""
        processor = router.get_processor("RE_ENGAGE")
        assert processor is not None
        assert processor.__name__ == "ReEngagementProcessor"

    def test_unknown_state_returns_none(self, router: ProcessorRouter):
        """Unknown state string returns None."""
        assert router.get_processor("COMPLETELY_UNKNOWN") is None

    def test_processor_map_has_required_entries(self):
        """PROCESSOR_MAP has all the active-state entries.

        The exact count is no longer enforced because new processors may be
        added with each phase. The test verifies the REQUIRED set is mapped.
        """
        required = {
            "GREETING", "QUALIFYING", "SCHEDULING", "QUALIFIED", "HANDOFF",
            "BUILDING_INFO",  # merged into qualifying
            "FOLLOW_UP",      # Phase 19
            "RE_ENGAGE",      # Phase 20
        }
        missing = required - set(PROCESSOR_MAP.keys())
        assert not missing, f"PROCESSOR_MAP missing required entries: {missing}"
