"""SchedulingProcessor -- handles appointment scheduling intent.

Ports logic from n8n Scheduling Handler (8 nodes -- Section 3.5):
- Detects scheduling intent and day preferences (EN/ES)
- Generates slot suggestions via Claude
- Cal.com API integration is STUBBED for Phase 16

NOTE: Cal.com real integration may come in Phase 17 or 16.5.
The current implementation uses placeholder available slots.

Usage:
    processor = SchedulingProcessor(
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

# Day preference patterns (EN + ES)
_DAY_PATTERNS = {
    "monday": re.compile(r"\b(lunes|monday|mon)\b", re.IGNORECASE),
    "tuesday": re.compile(r"\b(martes|tuesday|tue|tues)\b", re.IGNORECASE),
    "wednesday": re.compile(r"\b(mi[eé]rcoles|wednesday|wed)\b", re.IGNORECASE),
    "thursday": re.compile(r"\b(jueves|thursday|thu|thurs)\b", re.IGNORECASE),
    "friday": re.compile(r"\b(viernes|friday|fri)\b", re.IGNORECASE),
    "saturday": re.compile(r"\b(s[aá]bado|saturday|sat)\b", re.IGNORECASE),
    "sunday": re.compile(r"\b(domingo|sunday|sun)\b", re.IGNORECASE),
    "tomorrow": re.compile(r"\b(ma[nñ]ana|tomorrow)\b", re.IGNORECASE),
    "today": re.compile(r"\b(hoy|today)\b", re.IGNORECASE),
}

# Time preference patterns
_TIME_PATTERNS = {
    "morning": re.compile(r"\b(ma[nñ]ana|morning|am|temprano|early)\b", re.IGNORECASE),
    "afternoon": re.compile(r"\b(tarde|afternoon|pm|mediod[ií]a)\b", re.IGNORECASE),
    "evening": re.compile(r"\b(noche|evening|night)\b", re.IGNORECASE),
}

# TODO (Phase 17): Replace with real Cal.com API integration
_STUB_AVAILABLE_SLOTS = (
    "Lunes 10:00 AM, Lunes 3:00 PM, "
    "Martes 11:00 AM, Martes 4:00 PM, "
    "Miercoles 10:00 AM, Miercoles 2:00 PM, "
    "Jueves 11:00 AM, Jueves 3:00 PM, "
    "Viernes 10:00 AM, Viernes 1:00 PM"
)


def _detect_day_preference(message: str) -> str:
    """Extract day preference from message text."""
    for day, pattern in _DAY_PATTERNS.items():
        if pattern.search(message):
            return day
    return "none"


def _detect_time_preference(message: str) -> str:
    """Extract time preference from message text."""
    for period, pattern in _TIME_PATTERNS.items():
        if pattern.search(message):
            return period
    return "none"


def _strip_markdown_fences(text: str) -> str:
    """Remove markdown code fences (```json ... ```) from Claude output."""
    stripped = re.sub(r"```(?:json)?\s*\n?", "", text).strip()
    return stripped.rstrip("`").strip()


def _extract_outermost_json(text: str) -> str | None:
    """Extract the outermost JSON object from text, handling nested braces."""
    start = text.find("{")
    if start == -1:
        return None
    depth = 0
    in_string = False
    escape_next = False
    for i in range(start, len(text)):
        ch = text[i]
        if escape_next:
            escape_next = False
            continue
        if ch == "\\":
            escape_next = True
            continue
        if ch == '"' and not escape_next:
            in_string = not in_string
            continue
        if in_string:
            continue
        if ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0:
                return text[start : i + 1]
    return None


def _parse_scheduling_response(response_text: str) -> dict:
    """Parse Claude's scheduling JSON response.

    Handles:
    - Clean JSON
    - JSON wrapped in markdown code fences (```json ... ```)
    - JSON with nested objects (suggestedSlots array)
    """
    # Step 1: Strip markdown fences if present
    cleaned = _strip_markdown_fences(response_text)

    # Step 2: Try direct parse
    try:
        return json.loads(cleaned.strip())
    except json.JSONDecodeError:
        pass

    # Step 3: Extract outermost JSON object (handles nested braces)
    json_str = _extract_outermost_json(cleaned)
    if json_str:
        try:
            return json.loads(json_str)
        except json.JSONDecodeError:
            pass

    logger.warning(
        "scheduling.json_parse_fallback",
        raw_length=len(response_text),
        preview=response_text[:200],
    )
    return {"response": response_text, "needsMoreInfo": True}


class SchedulingProcessor(BaseProcessor):
    """Handles appointment scheduling with day/time preference detection.

    Detects scheduling intent and day/time preferences from the lead's
    message (bilingual ES/EN). Calls Claude with scheduling prompt +
    available slots to generate slot suggestions.

    NOTE: Cal.com API integration is STUBBED. Available slots are
    placeholder text. Real integration is planned for Phase 17 or 16.5.
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
        """Process scheduling intent and generate slot suggestions."""
        # Detect preferences
        preferred_day = _detect_day_preference(message)
        preferred_time = _detect_time_preference(message)
        building = conversation_context.get("mostRecentBuilding", "")
        language = lead_data.get("language", "es")

        # TODO (Phase 17): Fetch real available slots from Cal.com API
        # For now, use stub slots
        logger.warning(
            "scheduling.cal_com_stubbed",
            trace_id=trace_id,
            msg="Cal.com API integration is STUBBED -- using placeholder slots",
        )
        available_slots = _STUB_AVAILABLE_SLOTS

        # Build prompt context
        context = {
            "lead_name": lead_data.get("name", ""),
            "building": building,
            "preferred_day": preferred_day,
            "preferred_time": preferred_time,
            "message": message,
            "available_slots": available_slots,
            "language": language,
        }

        # Render scheduling prompt
        config = self._prompt_builder.get_config("scheduling")
        try:
            system_prompt, user_message = self._prompt_builder.render(
                "scheduling", context
            )
        except Exception:
            logger.exception("scheduling.prompt_render_failed", trace_id=trace_id)
            return ProcessorResult(
                response_text="",
                new_state=None,
                metadata={"fallback_needed": True},
            )

        # Call Claude
        try:
            response_text = await self._claude.generate(
                system_prompt=system_prompt,
                user_message=user_message,
                conversation_history=conversation_context.get("structured_turns"),
                model=config.get("model", "claude-sonnet-4-20250514"),
                max_tokens=config.get("max_tokens", 300),
            )
        except CircuitOpenError:
            logger.warning("scheduling.circuit_open", trace_id=trace_id)
            return ProcessorResult(
                response_text="",
                new_state=None,
                metadata={"fallback_needed": True},
            )

        # Parse response
        parsed = _parse_scheduling_response(response_text)
        response_msg = parsed.get("response", response_text)
        needs_more_info = parsed.get("needsMoreInfo", False)

        logger.info(
            "scheduling.processed",
            trace_id=trace_id,
            preferred_day=preferred_day,
            preferred_time=preferred_time,
            building=building,
            needs_more_info=needs_more_info,
        )

        return ProcessorResult(
            response_text=response_msg,
            new_state=None,  # Stay in SCHEDULING until appointment confirmed
            metadata={
                "scheduling_building": building,
                "preferred_day": preferred_day,
                "preferred_time": preferred_time,
                "needs_more_info": needs_more_info,
                "suggested_slots": parsed.get("suggestedSlots", []),
            },
        )
