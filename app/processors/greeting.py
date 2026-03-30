"""GreetingProcessor -- handles new and returning lead greetings.

Ports logic from n8n Greeting Handler (Section 3.1 of the audit):
- New leads: first-touch greeting with building-specific context
- Returning leads: personalized re-greeting using cross-session memory
- Building source detection from message + tags + formName
- Cadence detection (SPEED/EXPLORER/ANALYST) delegated to Claude

Usage:
    processor = GreetingProcessor(
        claude_service=svc, prompt_builder=pb,
        lead_repo=lr, conversation_repo=cr,
    )
    result = await processor.process(message, lead_data, ...)
"""

from __future__ import annotations

import json
import re

import structlog

from app.processors.base import BaseProcessor, ProcessorResult
from app.prompts.builder import PromptBuilder
from app.repositories.conversation_repository import ConversationRepository
from app.repositories.lead_repository import LeadRepository
from app.services.circuit_breaker import CircuitOpenError
from app.services.claude_service import ClaudeService

logger = structlog.get_logger()

# Building name patterns for source detection
_BUILDING_PATTERNS = {
    "armani": re.compile(r"armani", re.IGNORECASE),
    "park": re.compile(r"park\s*hyatt|parkhyatt", re.IGNORECASE),
    "one": re.compile(r"\bone\b", re.IGNORECASE),
    "oscar": re.compile(r"oscar", re.IGNORECASE),
    "en-vogue": re.compile(r"en[\s-]?vogue", re.IGNORECASE),
    "the gallery": re.compile(r"gallery", re.IGNORECASE),
    "glass": re.compile(r"glass", re.IGNORECASE),
    "ritz": re.compile(r"ritz", re.IGNORECASE),
}


def _detect_building_source(
    message: str, tags: list[str], form_name: str
) -> str:
    """Detect building source from message text, tags, and form name.

    Checks message first, then tags, then formName. Returns the building
    key (e.g., "armani", "park") or "none" if no match.
    """
    # Check message
    for name, pattern in _BUILDING_PATTERNS.items():
        if pattern.search(message):
            return name

    # Check tags
    tags_lower = " ".join(t.lower() for t in tags) if tags else ""
    for name, pattern in _BUILDING_PATTERNS.items():
        if pattern.search(tags_lower):
            return name

    # Check form name
    if form_name:
        for name, pattern in _BUILDING_PATTERNS.items():
            if pattern.search(form_name):
                return name

    return "none"


class GreetingProcessor(BaseProcessor):
    """Processes greeting state for new and returning leads.

    For new leads: builds first-touch greeting with building context.
    For returning leads: fetches cross-session memory and personalizes.
    Transitions to QUALIFYING after greeting.
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
        """Generate a greeting response for the lead.

        New leads get a first-touch greeting. Returning leads get a
        personalized re-greeting using cross-session memory from Neo4j.
        """
        phone = lead_data.get("phone", "")
        is_new_lead = lead_data.get("is_new_lead", not lead_data.get("current_state"))
        is_auto_trigger = lead_data.get("is_auto_trigger", False)

        # Detect building source
        building_source = _detect_building_source(
            message,
            enriched_context.get("tags", []),
            enriched_context.get("formName", ""),
        )

        # Fetch cross-session memory for returning leads
        memory = None
        if not is_new_lead and phone:
            try:
                memory = await self._lead_repo.get_cross_session_memory(phone)
            except Exception:
                logger.exception(
                    "greeting.memory_fetch_failed",
                    trace_id=trace_id,
                    phone=phone[-4:],
                )

        # Load prompt config for model/token settings
        config = self._prompt_builder.get_config("greeting")

        # Determine which system prompt to use
        raw_yaml = self._prompt_builder._load_yaml(
            f"{self._prompt_builder.version}/greeting.yaml"
        )

        if not is_new_lead and memory and memory.get("recent_interactions"):
            # Returning lead -- use returning_system_prompt
            system_template = raw_yaml.get(
                "returning_system_prompt", raw_yaml.get("system_prompt", "")
            )
            max_tokens = raw_yaml.get(
                "returning_max_tokens", config.get("max_tokens", 200)
            )
        else:
            # New lead -- use main system_prompt
            system_template = raw_yaml.get("system_prompt", "")
            max_tokens = config.get("max_tokens", 200)

        # Build context dict
        context = {
            "lead_name": lead_data.get("name", ""),
            "building": building_source,
            "language": lead_data.get("language", "es"),
            "channel": lead_data.get("channel", "whatsapp"),
            "message": message,
            "is_auto_trigger": is_auto_trigger,
            "ghl_conversation_context": conversation_context.get(
                "ghlConversationContext", ""
            ),
            "crm_context": json.dumps(enriched_context) if enriched_context else "",
            "cross_session_memory": json.dumps(memory) if memory else "",
            "cadence": (
                lead_data.get("cadence", "explorer")
                if not is_new_lead
                else "explorer"
            ),
        }

        # Render templates
        system_prompt = self._prompt_builder._render_template(
            system_template, context, "greeting.system_prompt"
        )
        user_template = raw_yaml.get("user_template", "")
        user_message = self._prompt_builder._render_template(
            user_template, context, "greeting.user_template"
        )

        # Call Claude
        try:
            response = await self._claude.generate(
                system_prompt=system_prompt,
                user_message=user_message,
                model=config.get("model", "claude-sonnet-4-20250514"),
                max_tokens=max_tokens,
            )
        except CircuitOpenError:
            logger.warning("greeting.circuit_open", trace_id=trace_id)
            return ProcessorResult(
                response_text="",
                new_state=None,
                metadata={"fallback_needed": True, "building_source": building_source},
            )

        logger.info(
            "greeting.processed",
            trace_id=trace_id,
            building_source=building_source,
            is_new_lead=is_new_lead,
            response_length=len(response),
        )

        return ProcessorResult(
            response_text=response,
            new_state="QUALIFYING",
            metadata={
                "building_source": building_source,
                "interested": True,
                "lead_name": lead_data.get("name", ""),
            },
        )
