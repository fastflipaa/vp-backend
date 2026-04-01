"""HandoffProcessor -- generates context summary for human agent handoff.

Ports logic from n8n Handoff Handler (7 nodes -- Section 3.3 of the audit):
- Detects handoff reason (explicit request, frustration, high value)
- Generates context summary for human agent via Claude
- Determines priority (urgent, high, normal)
- Returns language-specific transition message

Usage:
    processor = HandoffProcessor(
        claude_service=svc, prompt_builder=pb,
        lead_repo=lr, conversation_repo=cr,
    )
    result = await processor.process(message, lead_data, ...)
"""

from __future__ import annotations

import structlog

from app.processors.base import BaseProcessor, ProcessorResult
from app.prompts.builder import PromptBuilder
from app.repositories.conversation_repository import ConversationRepository
from app.repositories.lead_repository import LeadRepository
from app.services.circuit_breaker import CircuitOpenError
from app.services.claude_service import ClaudeService

logger = structlog.get_logger()

# Handoff reason detection keywords
_EXPLICIT_REQUEST_KEYWORDS = [
    "quiero hablar con una persona",
    "quiero hablar con alguien",
    "pasame con un humano",
    "hablar con un asesor",
    "hablar con fernando",
    "hablar con lorena",
    "want to talk to a person",
    "speak to someone",
    "talk to an agent",
    "connect me with",
]

_FRUSTRATION_KEYWORDS = [
    "frustrado", "frustrada", "harto", "harta", "molesto", "molesta",
    "frustrated", "annoyed", "angry", "upset",
]


def _detect_handoff_reason(
    message: str, lead_data: dict, sentiment: str = ""
) -> str:
    """Detect why the lead is being handed off.

    Returns: explicit_request, detected_frustration, high_value_opportunity,
    or processor_escalation (default).
    """
    lower = message.lower()

    # Explicit request
    for kw in _EXPLICIT_REQUEST_KEYWORDS:
        if kw in lower:
            return "explicit_request"

    # Frustration detection
    for kw in _FRUSTRATION_KEYWORDS:
        if kw in lower:
            return "detected_frustration"

    if sentiment.lower() in ("frustrated", "negative"):
        return "detected_frustration"

    # High-value opportunity
    budget_max = lead_data.get("budgetMax")
    if budget_max and isinstance(budget_max, (int, float)):
        if budget_max >= 10_000_000:  # 10M MXN
            return "high_value_opportunity"

    return "processor_escalation"


def _determine_priority(reason: str, lead_data: dict) -> str:
    """Determine handoff priority based on reason and lead data.

    Returns: urgent, high, or normal.
    """
    budget_max = lead_data.get("budgetMax")
    is_high_value = (
        budget_max and isinstance(budget_max, (int, float))
        and budget_max >= 10_000_000
    )

    if reason == "explicit_request" and is_high_value:
        return "urgent"
    if reason == "explicit_request":
        return "high"
    if reason == "detected_frustration":
        return "high"
    if is_high_value:
        return "high"
    return "normal"


class HandoffProcessor(BaseProcessor):
    """Generates handoff context summary and transition message.

    Calls Claude to generate a concise summary for the human agent,
    then returns a language-appropriate transition message for the lead.
    Sets state to HANDOFF.
    """

    def __init__(
        self,
        claude_service: ClaudeService,
        prompt_builder: PromptBuilder,
        lead_repo: LeadRepository,
        conversation_repo: ConversationRepository,
    ) -> None:
        self._claude = claude_service
        self._prompt_builder = prompt_builder
        self._lead_repo = lead_repo
        self._conv_repo = conversation_repo

    async def process(
        self,
        message: str,
        lead_data: dict,
        enriched_context: dict,
        conversation_context: dict,
        trace_id: str,
    ) -> ProcessorResult:
        """Generate handoff context summary and transition message."""
        language = lead_data.get("language", "es")

        # Detect handoff reason
        sentiment = lead_data.get("sentiment_current", "")
        reason = _detect_handoff_reason(message, lead_data, sentiment)
        priority = _determine_priority(reason, lead_data)

        # Generate context summary via Claude
        context_summary = ""
        config = self._prompt_builder.get_config("handoff")
        try:
            context = {
                "lead_name": lead_data.get("name", ""),
                "phone": lead_data.get("phone", ""),
                "channel": lead_data.get("channel", ""),
                "building": conversation_context.get("mostRecentBuilding", ""),
                "interest_type": lead_data.get("interestType", "unknown"),
                "budget_min": lead_data.get("budgetMin", "?"),
                "budget_max": lead_data.get("budgetMax", "?"),
                "budget_currency": "MXN",
                "timeline": lead_data.get("timeline", "unknown"),
                "handoff_reason": reason,
                "message": message,
            }
            system_prompt, user_message = self._prompt_builder.render(
                "handoff", context
            )
            context_summary = await self._claude.generate(
                system_prompt=system_prompt,
                user_message=user_message,
                model=config.get("model", "claude-sonnet-4-20250514"),
                max_tokens=config.get("max_tokens", 200),
            )
        except CircuitOpenError:
            logger.warning("handoff.circuit_open", trace_id=trace_id)
            context_summary = (
                f"Lead {lead_data.get('name', 'unknown')} needs handoff. "
                f"Reason: {reason}. Priority: {priority}."
            )
        except Exception:
            logger.exception("handoff.summary_failed", trace_id=trace_id)
            context_summary = (
                f"Lead {lead_data.get('name', 'unknown')} needs handoff. "
                f"Reason: {reason}."
            )

        # Language-specific transition message
        if language.lower().startswith("es"):
            transition_message = (
                "Entiendo, te paso con un asesor que puede ayudarte mejor. "
                "Te va a contactar en breve."
            )
        else:
            transition_message = (
                "I understand, let me connect you with an advisor who can "
                "help you better. They'll reach out shortly."
            )

        logger.info(
            "handoff.processed",
            trace_id=trace_id,
            reason=reason,
            priority=priority,
            summary_length=len(context_summary),
        )

        # Enqueue WhatsApp notification to Fernando (fail-open)
        try:
            from app.tasks.handoff_notification_task import send_handoff_notification

            send_handoff_notification.delay(
                lead_data.get("contact_id", ""),
                lead_data.get("phone", ""),
                lead_data.get("name", ""),
                reason,
                priority,
                context_summary,
                conversation_context.get("mostRecentBuilding", "unknown"),
                trace_id,
            )
        except Exception:
            logger.exception("handoff.notification_enqueue_failed", trace_id=trace_id)

        return ProcessorResult(
            response_text=transition_message,
            new_state="HANDOFF",
            should_handoff=True,
            metadata={
                "handoff_reason": reason,
                "priority": priority,
                "context_summary": context_summary,
                "escalation_flag": True,
            },
        )
