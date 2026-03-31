"""Base processor interface and result dataclass.

All state processors inherit from BaseProcessor and return ProcessorResult.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field


@dataclass
class ProcessorResult:
    """Result returned by every state processor.

    Attributes:
        response_text: The AI-generated response to send to the lead.
            Empty string means no response (e.g., fallback needed).
        new_state: Target state for SM transition, or None for no change.
        metadata: Additional data from the processor (building_source,
            handoff_reason, etc.). Also passed to SM.advance() as kwargs.
        should_handoff: True if the processor detected escalation need.
        sub_state_update: Dict of qualifying sub-state updates to save
            to Neo4j (e.g., {"sub_state": "QUAL_BUDGET"}).
    """

    response_text: str
    new_state: str | None = None
    metadata: dict = field(default_factory=dict)
    should_handoff: bool = False
    sub_state_update: dict | None = None
    prompt_version: str | None = None


class BaseProcessor(ABC):
    """Abstract base class for all state processors.

    Processors receive a message + context and return a ProcessorResult
    containing the AI response and any state/metadata changes.
    """

    @abstractmethod
    async def process(
        self,
        message: str,
        lead_data: dict,
        enriched_context: dict,
        conversation_context: dict,
        trace_id: str,
    ) -> ProcessorResult:
        """Process a message in the current conversation state.

        Args:
            message: The inbound message text from the lead.
            lead_data: Lead properties from Neo4j + pipeline additions
                (phone, contact_id, channel, is_new_lead, etc.).
            enriched_context: CRM enrichment from GHL (tags, notes,
                messages, customFields).
            conversation_context: Cached conversation context from
                HumanAgentDetector (ghlConversationContext,
                mostRecentBuilding, spamLimitReached).
            trace_id: Pipeline trace ID for logging correlation.

        Returns:
            ProcessorResult with response, state transition, and metadata.
        """
        ...
