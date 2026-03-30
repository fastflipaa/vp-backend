"""ProcessorRouter -- dispatches messages to state-specific processors.

Maps conversation states to processor classes. Terminal states (BROKER,
CLOSED) and deferred states (RE_ENGAGE, FOLLOW_UP) return None,
indicating no AI response should be generated.

Usage:
    router = ProcessorRouter()
    processor_class = router.get_processor("GREETING")
    if processor_class:
        processor = processor_class(claude_service=..., prompt_builder=...)
        result = await processor.process(...)
"""

from __future__ import annotations

import structlog

from app.processors.greeting import GreetingProcessor
from app.processors.handoff import HandoffProcessor
from app.processors.qualifying import QualifyingProcessor
from app.processors.scheduling import SchedulingProcessor

logger = structlog.get_logger()

# State -> Processor class mapping
# Terminal states (BROKER, CLOSED) and deferred states (RE_ENGAGE, FOLLOW_UP)
# are NOT mapped -- get_processor returns None for them.
PROCESSOR_MAP: dict[str, type] = {
    "GREETING": GreetingProcessor,
    "QUALIFYING": QualifyingProcessor,
    "SCHEDULING": SchedulingProcessor,
    "HANDOFF": HandoffProcessor,
    "QUALIFIED": HandoffProcessor,  # Qualified leads go to handoff
    # BUILDING_INFO routes to QualifyingProcessor (stub in n8n, merged into qualifying)
    "BUILDING_INFO": QualifyingProcessor,
}


class ProcessorRouter:
    """Routes messages to the correct state processor.

    Returns None for states that don't process AI responses in Phase 16:
    - RE_ENGAGE: Phase 17 proactive re-engagement
    - FOLLOW_UP: Phase 17 follow-up sequences
    - BROKER: Terminal, no response
    - CLOSED: Terminal, no response
    - NON_RESPONSIVE: Awaiting re-engage or close
    - RECOVERY: Error recovery (restarts to GREETING via SM)
    """

    def get_processor(self, state: str) -> type | None:
        """Return the processor class for the given state, or None.

        Args:
            state: Conversation state string (e.g. "GREETING", "QUALIFYING").

        Returns:
            Processor class (not instance) or None if state has no processor.
        """
        processor_class = PROCESSOR_MAP.get(state)
        if processor_class is None:
            logger.debug(
                "processor_router.no_processor",
                state=state,
                msg="State has no processor in Phase 16",
            )
        return processor_class
