"""QualifyingProcessor -- manages 5-step qualification sub-state machine.

Ports logic from n8n Qualifying Handler (58 nodes -- Section 3.2):
- 5 sub-states: QUAL_INTEREST, QUAL_BUDGET, QUAL_TIMELINE, QUAL_CONTACT, QUAL_BUILDING_MATCH
- Auto-advance: skips sub-states whose data already exists
- GraphRAG integration for building match (via BuildingRepository)
- Escalation detection: frustration, high-value budget, explicit request
- Cadence detection: SPEED/EXPLORER/ANALYST from message patterns
- Sentiment extraction from Claude response metadata

Usage:
    processor = QualifyingProcessor(
        claude_service=svc, prompt_builder=pb,
        lead_repo=lr, conversation_repo=cr, building_repo=br,
    )
    result = await processor.process(message, lead_data, ...)
"""

from __future__ import annotations

import json
import re

import structlog

from app.processors.base import BaseProcessor, ProcessorResult
from app.prompts.builder import PromptBuilder
from app.repositories.building_repository import BuildingRepository
from app.repositories.conversation_repository import ConversationRepository
from app.repositories.lead_repository import LeadRepository
from app.services.circuit_breaker import CircuitOpenError
from app.services.claude_service import ClaudeService

logger = structlog.get_logger()

# Sub-state progression order
SUB_STATE_ORDER = [
    "QUAL_INTEREST",
    "QUAL_BUDGET",
    "QUAL_TIMELINE",
    "QUAL_CONTACT",
    "QUAL_BUILDING_MATCH",
]

# Fields that indicate a sub-state's data has been collected
SUB_STATE_DATA_KEYS = {
    "QUAL_INTEREST": "interestType",
    "QUAL_BUDGET": "budgetMin",
    "QUAL_TIMELINE": "timeline",
    "QUAL_CONTACT": "email",
    "QUAL_BUILDING_MATCH": None,  # No auto-advance for building match
}

# Budget threshold for high-value escalation (MXN)
HIGH_VALUE_THRESHOLD_MXN = 10_000_000

# Escalation keywords (ES + EN)
ESCALATION_KEYWORDS = [
    "quiero hablar con una persona",
    "quiero hablar con alguien",
    "pasame con un humano",
    "hablar con un asesor",
    "want to talk to a person",
    "speak to someone",
    "speak to a human",
    "talk to an agent",
    "frustrated",
    "frustrado",
    "harto",
    "molesto",
]

# Email validation pattern
_EMAIL_RE = re.compile(r"^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$")


def _enqueue_doc_delivery(
    contact_id: str,
    phone: str,
    building_name: str,
    channel: str,
    trace_id: str,
    language: str = "es",
) -> None:
    """Fire-and-forget: enqueue a doc delivery task for this lead.

    Import is deferred to avoid circular imports at module load time.
    Logs a warning and continues if enqueueing fails (fail-open).
    """
    try:
        from app.tasks.doc_delivery_task import deliver_documents
        deliver_documents.delay(contact_id, phone, building_name, channel, trace_id, language)
    except Exception:
        logger.exception(
            "qualifying.doc_delivery_enqueue_failed",
            trace_id=trace_id,
            contact_id=contact_id,
            building_name=building_name,
        )


def _maybe_send_pricing_alert(
    response_msg: str,
    lead_data: dict,
    building_context: str,
    message: str,
    trace_id: str,
) -> None:
    """Send a pricing alert to the team if Claude deferred pricing.

    Detects pricing deferral by checking:
    1. Building context contains "NOT VERIFIED" (unverified buildings were shown)
    2. Claude's response mentions Fernando/consultar/verificar (pricing deferral)

    This is NOT a handoff — just a notification so the team can provide pricing.
    Fail-open: logs and continues if anything goes wrong.
    """
    has_unverified = "NOT VERIFIED" in building_context
    deferral_keywords = ["fernando", "consultar", "verificar", "check with", "get back"]
    response_lower = response_msg.lower()
    deferred_pricing = any(kw in response_lower for kw in deferral_keywords)

    if not (has_unverified and deferred_pricing):
        return

    try:
        from app.tasks.pricing_alert_task import send_pricing_alert

        # Extract building name from response or context
        building = "Unknown"
        for line in building_context.split("\n"):
            if line.startswith("BUILDING: "):
                candidate = line.replace("BUILDING: ", "").strip()
                if candidate.lower() in response_lower:
                    building = candidate
                    break
                building = candidate  # fallback to last seen

        send_pricing_alert.delay(
            contact_id=lead_data.get("contact_id", ""),
            phone=lead_data.get("phone", ""),
            lead_name=lead_data.get("name", "Unknown"),
            building=building,
            lead_message=message,
            trace_id=trace_id,
        )
        logger.info(
            "qualifying.pricing_alert_enqueued",
            trace_id=trace_id,
            building=building,
        )
    except Exception:
        logger.exception(
            "qualifying.pricing_alert_enqueue_failed",
            trace_id=trace_id,
        )


def _detect_cadence(message: str) -> str:
    """Detect lead cadence from message patterns.

    - SPEED: Short messages (<20 words), direct questions
    - ANALYST: Questions about ROI, investment, legal
    - EXPLORER: Everything else (default)
    """
    word_count = len(message.split())
    lower = message.lower()

    if word_count < 20:
        # Check for direct price/size/availability questions
        speed_indicators = [
            "cuanto", "precio", "cuesta", "disponib", "metros",
            "how much", "price", "available", "size", "sqft",
        ]
        if any(ind in lower for ind in speed_indicators):
            return "speed"

    # Check for analyst indicators
    analyst_indicators = [
        "roi", "inversion", "plusvalia", "rendimiento", "legal",
        "investment", "return", "appreciation", "yield", "process",
        "comparar", "compare",
    ]
    if any(ind in lower for ind in analyst_indicators):
        return "analyst"

    return "explorer"


def _check_escalation(
    message: str, lead_data: dict, sentiment: str
) -> tuple[bool, str]:
    """Check if the lead should be escalated to a human agent.

    Returns (should_escalate, reason).
    """
    lower = message.lower()

    # Explicit request
    for keyword in ESCALATION_KEYWORDS:
        if keyword in lower:
            return True, "explicit_request"

    # Frustration sentiment
    if sentiment.lower() in ("frustrated", "negative"):
        return True, "detected_frustration"

    # High-value budget
    budget_max = lead_data.get("budgetMax")
    if budget_max and isinstance(budget_max, (int, float)):
        if budget_max >= HIGH_VALUE_THRESHOLD_MXN:
            return True, "high_value_opportunity"

    return False, ""


def _extract_email_from_message(message: str) -> str | None:
    """Extract an email address from a message, if present."""
    words = message.split()
    for word in words:
        cleaned = word.strip(".,;:!?()[]<>")
        if _EMAIL_RE.match(cleaned):
            return cleaned
    return None


def _parse_claude_json(response_text: str) -> dict:
    """Parse Claude's JSON response, handling embedded text.

    Claude sometimes returns text + JSON. We extract the JSON block.
    """
    # Try direct parse first
    try:
        return json.loads(response_text.strip())
    except json.JSONDecodeError:
        pass

    # Look for JSON block in response
    json_match = re.search(r"\{[^{}]*\}", response_text, re.DOTALL)
    if json_match:
        try:
            return json.loads(json_match.group())
        except json.JSONDecodeError:
            pass

    return {}


def _format_building_context(buildings: list[dict]) -> str:
    """Format building data for Claude's context, respecting pricing_verified flag.

    When pricing_verified is False or price data is 0/NULL, injects an explicit
    instruction for Claude to defer pricing questions to Fernando/Lorena.
    This prevents hallucination of prices — a legal liability risk.
    """
    if not buildings:
        return ""

    lines = []
    for b in buildings:
        name = b.get("name", "Unknown")
        verified = b.get("pricing_verified", False)
        price_min = b.get("price_min_usd") or 0
        price_max = b.get("price_max_usd") or 0
        has_real_pricing = verified and price_min > 0 and price_max > 0

        lines.append(f"BUILDING: {name}")
        lines.append(f"  City: {b.get('city', '?')}, {b.get('country', '?')}")
        lines.append(f"  Status: {b.get('status', '?')}")

        if has_real_pricing:
            lines.append(f"  Price Range: ${price_min:,.0f} - ${price_max:,.0f} USD (VERIFIED)")
            if b.get("total_floors"):
                lines.append(f"  Floors: {b['total_floors']}")
            if b.get("total_units"):
                lines.append(f"  Units: {b['total_units']}")
            if b.get("views"):
                lines.append(f"  Views: {b['views']}")
            if b.get("key_features"):
                lines.append(f"  Features: {b['key_features']}")
            if b.get("completion_date"):
                lines.append(f"  Completion: {b['completion_date']}")
            # Include unit types if available
            units = b.get("units", [])
            if units:
                for u in units:
                    uname = u.get("name", "?")
                    beds = u.get("bedrooms", "?")
                    pf = u.get("price_from") or 0
                    pt = u.get("price_to") or 0
                    if pf > 0 and pt > 0:
                        lines.append(f"    Unit: {uname} ({beds} bed) — ${pf:,.0f}-${pt:,.0f}")
                    elif pf > 0:
                        lines.append(f"    Unit: {uname} ({beds} bed) — from ${pf:,.0f}")
                    else:
                        lines.append(f"    Unit: {uname} ({beds} bed)")
        else:
            lines.append(
                f"  PRICING: NOT VERIFIED — You do NOT have pricing for {name}. "
                f"ONCE (first time only): mention you will check with Fernando. "
                f"AFTER THAT: if the GHL conversation already shows you said you would check with Fernando, "
                f"DO NOT say it again. Just continue qualifying naturally — ask about their preferences, "
                f"timeline, what matters most to them. Repeating 'voy a consultar con Fernando' every message sounds robotic. "
                f"Keep next_action as \"continue_qualifying\". "
                f"ONLY set \"handoff_fernando\" if the lead explicitly asks to speak with a human."
            )
            if b.get("description_es"):
                lines.append(f"  Description: {b['description_es']}")

        lines.append("")

    return "\n".join(lines)


class QualifyingProcessor(BaseProcessor):
    """5-step qualification processor with auto-advance and escalation.

    Sub-states: QUAL_INTEREST -> QUAL_BUDGET -> QUAL_TIMELINE ->
    QUAL_CONTACT -> QUAL_BUILDING_MATCH

    Auto-advances past sub-states whose data already exists in lead_data.
    Detects escalation triggers and routes to HANDOFF when needed.
    """

    def __init__(
        self,
        claude_service: ClaudeService,
        prompt_builder: PromptBuilder,
        lead_repo: LeadRepository,
        conversation_repo: ConversationRepository,
        building_repo: BuildingRepository | None = None,
    ) -> None:
        self._claude = claude_service
        self._prompt_builder = prompt_builder
        self._lead_repo = lead_repo
        self._conv_repo = conversation_repo
        self._building_repo = building_repo

    async def process(
        self,
        message: str,
        lead_data: dict,
        enriched_context: dict,
        conversation_context: dict,
        trace_id: str,
    ) -> ProcessorResult:
        """Process a qualifying message through the sub-state machine."""
        phone = lead_data.get("phone", "")
        contact_id = lead_data.get("contact_id", "")
        current_sub = lead_data.get("sub_state", "QUAL_INTEREST")

        # Auto-advance: skip sub-states whose data already exists
        current_sub = self._auto_advance(current_sub, lead_data)

        logger.info(
            "qualifying.processing",
            trace_id=trace_id,
            sub_state=current_sub,
            phone=phone[-4:] if phone else "",
        )

        # Route to sub-state handler
        if current_sub == "QUAL_CONTACT":
            return await self._handle_contact(
                message, lead_data, conversation_context, trace_id
            )

        if current_sub == "QUAL_BUILDING_MATCH":
            return await self._handle_building_match(
                message, lead_data, enriched_context, conversation_context, trace_id
            )

        # QUAL_INTEREST, QUAL_BUDGET, QUAL_TIMELINE use Claude
        return await self._handle_claude_sub_state(
            current_sub, message, lead_data, enriched_context,
            conversation_context, trace_id,
        )

    def _auto_advance(self, current_sub: str, lead_data: dict) -> str:
        """Skip sub-states whose data already exists in lead_data."""
        idx = SUB_STATE_ORDER.index(current_sub) if current_sub in SUB_STATE_ORDER else 0

        while idx < len(SUB_STATE_ORDER) - 1:
            sub = SUB_STATE_ORDER[idx]
            data_key = SUB_STATE_DATA_KEYS.get(sub)
            if data_key is None:
                break  # No auto-advance for this sub-state
            if not lead_data.get(data_key):
                break  # Data not yet collected
            logger.debug(
                "qualifying.auto_advance",
                skipping=sub,
                data_key=data_key,
            )
            idx += 1

        return SUB_STATE_ORDER[idx]

    async def _handle_claude_sub_state(
        self,
        sub_state: str,
        message: str,
        lead_data: dict,
        enriched_context: dict,
        conversation_context: dict,
        trace_id: str,
    ) -> ProcessorResult:
        """Handle QUAL_INTEREST, QUAL_BUDGET, or QUAL_TIMELINE via Claude."""
        # Detect cadence
        cadence = _detect_cadence(message)

        # Build dynamic blocks for the prompt
        known_items = self._build_known_items(lead_data)

        # Determine if broker-aware
        tags = enriched_context.get("tags", [])
        broker_aware = any(
            "broker" in (t or "").lower() or "ambiguous" in (t or "").lower()
            for t in tags
        )

        # Fetch building data with pricing_verified context
        building_context = ""
        phone = lead_data.get("phone", "")
        if self._building_repo and phone:
            try:
                buildings = await self._building_repo.get_buildings_for_lead(phone)
                if buildings:
                    building_context = _format_building_context(buildings)
            except Exception:
                logger.exception(
                    "qualifying.building_context_failed", trace_id=trace_id
                )

        # Load qualifying YAML config
        raw_yaml = self._prompt_builder._load_yaml(
            f"{self._prompt_builder.version}/qualifying.yaml"
        )

        # Get sub-state-specific prompt
        sub_state_key = sub_state  # e.g., "QUAL_INTEREST"
        sub_config = raw_yaml.get("sub_states", {}).get(sub_state_key, {})
        system_template = sub_config.get("system_prompt", "")
        max_tokens = sub_config.get("max_tokens", 500)

        if not system_template:
            # Fallback to main system_prompt
            system_template = raw_yaml.get("system_prompt", "")

        # Build context for template rendering
        context = {
            "lead_name": lead_data.get("name", ""),
            "building": conversation_context.get("mostRecentBuilding", ""),
            "language": lead_data.get("language", "es"),
            "message": message,
            "sub_state": sub_state,
            "cadence": cadence,
            "known_items": known_items,
            "broker_aware": broker_aware,
            "building_context": building_context,
            "ghl_conversation_context": conversation_context.get(
                "ghlConversationContext", ""
            ),
            "most_recent_building": conversation_context.get(
                "mostRecentBuilding", ""
            ),
        }

        # Render the known_data_block, broker_aware_block, cadence_block
        known_data_block = self._prompt_builder._render_template(
            raw_yaml.get("known_data_block", ""), context, "qualifying.known_data_block"
        )
        broker_aware_block = self._prompt_builder._render_template(
            raw_yaml.get("broker_aware_block", ""), context, "qualifying.broker_aware_block"
        )
        cadence_block = self._prompt_builder._render_template(
            raw_yaml.get("cadence_block", ""), context, "qualifying.cadence_block"
        )

        # Inject rendered blocks into the system template context
        context["known_data_block"] = known_data_block
        context["broker_aware_block"] = broker_aware_block
        context["cadence_block"] = cadence_block

        # Render system prompt with blocks injected
        system_prompt = self._prompt_builder._render_template(
            system_template, context, f"qualifying.{sub_state}.system"
        )

        # Render user template
        user_template = raw_yaml.get("user_template", "")
        user_message = self._prompt_builder._render_template(
            user_template, context, f"qualifying.{sub_state}.user"
        )

        # Call Claude
        config = self._prompt_builder.get_config("qualifying")
        try:
            response_text = await self._claude.generate(
                system_prompt=system_prompt,
                user_message=user_message,
                model=config.get("model", "claude-sonnet-4-20250514"),
                max_tokens=max_tokens,
            )
        except CircuitOpenError:
            logger.warning("qualifying.circuit_open", trace_id=trace_id)
            return ProcessorResult(
                response_text="",
                new_state=None,
                metadata={"fallback_needed": True},
            )

        # Parse Claude response (expects JSON with response + extracted data)
        parsed = _parse_claude_json(response_text)
        response_msg = parsed.get("response", response_text)
        sentiment = parsed.get("sentiment", "neutral")
        sentiment_confidence = float(parsed.get("sentiment_confidence", 0.5))

        # Check for escalation
        should_escalate, escalation_reason = _check_escalation(
            message, lead_data, sentiment
        )
        if should_escalate or parsed.get("escalate"):
            reason = escalation_reason or parsed.get("escalation_reason", "processor_detected")
            return ProcessorResult(
                response_text=response_msg,
                new_state="HANDOFF",
                should_handoff=True,
                metadata={
                    "escalation_reason": reason,
                    "sentiment": sentiment,
                },
            )

        # Check next_action from Claude
        next_action = parsed.get("next_action", "continue_qualifying")
        if next_action == "handoff_fernando":
            return ProcessorResult(
                response_text=response_msg,
                new_state="HANDOFF",
                should_handoff=True,
                metadata={"escalation_reason": "handoff_fernando"},
            )
        if next_action == "schedule_appointment":
            return ProcessorResult(
                response_text=response_msg,
                new_state="SCHEDULING",
                metadata={"appointment_requested": True},
            )
        if next_action == "send_docs":
            building_name = (
                parsed.get("building_name")
                or conversation_context.get("mostRecentBuilding", "")
            )
            _enqueue_doc_delivery(
                contact_id=lead_data.get("contact_id", ""),
                phone=lead_data.get("phone", ""),
                building_name=building_name,
                channel=lead_data.get("channel", "SMS"),
                trace_id=trace_id,
                language=lead_data.get("language", "es"),
            )
            logger.info(
                "qualifying.doc_delivery_enqueued",
                trace_id=trace_id,
                building_name=building_name,
            )

        # Pricing alert: if Claude deferred pricing to Fernando, notify the team
        _maybe_send_pricing_alert(
            response_msg=response_msg,
            lead_data=lead_data,
            building_context=building_context,
            message=message,
            trace_id=trace_id,
        )

        # Build qualification data update
        qual_update = {"sub_state": self._next_sub_state(sub_state)}
        if sub_state == "QUAL_INTEREST" and parsed.get("interest_type"):
            qual_update["interestType"] = parsed["interest_type"]
        elif sub_state == "QUAL_BUDGET":
            if parsed.get("budget_min"):
                qual_update["budgetMin"] = parsed["budget_min"]
            if parsed.get("budget_max"):
                qual_update["budgetMax"] = parsed["budget_max"]
        elif sub_state == "QUAL_TIMELINE" and parsed.get("timeline"):
            qual_update["timeline"] = parsed["timeline"]

        # Save sentiment
        phone = lead_data.get("phone", "")
        if phone and sentiment:
            try:
                await self._lead_repo.save_sentiment(
                    phone, sentiment, sentiment_confidence
                )
            except Exception:
                logger.exception("qualifying.sentiment_save_failed", trace_id=trace_id)

        logger.info(
            "qualifying.sub_state_complete",
            trace_id=trace_id,
            sub_state=sub_state,
            next_sub=qual_update.get("sub_state"),
            sentiment=sentiment,
        )

        return ProcessorResult(
            response_text=response_msg,
            new_state=None,  # Stay in QUALIFYING
            sub_state_update=qual_update,
            metadata={
                "cadence": cadence,
                "sentiment": sentiment,
            },
        )

    async def _handle_contact(
        self,
        message: str,
        lead_data: dict,
        conversation_context: dict,
        trace_id: str,
    ) -> ProcessorResult:
        """Handle QUAL_CONTACT: validate email/phone inline (no Claude call).

        Extracts email from message if present. Phone is already known
        from the inbound payload.
        """
        email = _extract_email_from_message(message)
        qual_update: dict = {"sub_state": "QUAL_BUILDING_MATCH"}

        if email:
            qual_update["email"] = email
            response = (
                "Perfecto, lo tengo. Dejame armar las mejores opciones para ti."
                if lead_data.get("language", "es").startswith("es")
                else "Perfect, got it. Let me put together the best options for you."
            )
        else:
            # No email found -- skip and move on (don't block qualifying)
            response = (
                "Perfecto. Dejame ver las opciones que mejor te van."
                if lead_data.get("language", "es").startswith("es")
                else "Perfect. Let me look at the best options for you."
            )

        logger.info(
            "qualifying.contact_validated",
            trace_id=trace_id,
            email_found=email is not None,
        )

        return ProcessorResult(
            response_text=response,
            new_state=None,  # Stay in QUALIFYING
            sub_state_update=qual_update,
            metadata={"email_extracted": email},
        )

    async def _handle_building_match(
        self,
        message: str,
        lead_data: dict,
        enriched_context: dict,
        conversation_context: dict,
        trace_id: str,
    ) -> ProcessorResult:
        """Handle QUAL_BUILDING_MATCH: GraphRAG + Claude for building recommendation."""
        phone = lead_data.get("phone", "")
        cadence = _detect_cadence(message)
        graphrag_context = ""

        # Fetch building data + GraphRAG recommendations
        building_context = ""
        if self._building_repo and phone:
            try:
                buildings = await self._building_repo.get_buildings_for_lead(phone)
                if buildings:
                    # Format building data with pricing_verified checks
                    building_context = _format_building_context(buildings)
                    building_ids = [
                        b["building_id"] for b in buildings if b.get("building_id")
                    ]
                    if building_ids:
                        budget_max = lead_data.get("budgetMax")
                        similar = await self._building_repo.get_similar_buildings(
                            building_ids,
                            budget_max=float(budget_max) if budget_max else None,
                        )
                        if similar:
                            graphrag_context = json.dumps(similar, default=str)
            except Exception:
                logger.exception(
                    "qualifying.graphrag_failed", trace_id=trace_id
                )

        # Build context
        known_items = self._build_known_items(lead_data)
        qual_context = json.dumps(
            {
                k: lead_data.get(k)
                for k in ("interestType", "budgetMin", "budgetMax", "timeline", "email")
                if lead_data.get(k)
            }
        )

        # Load sub-state prompt
        raw_yaml = self._prompt_builder._load_yaml(
            f"{self._prompt_builder.version}/qualifying.yaml"
        )
        sub_config = raw_yaml.get("sub_states", {}).get("QUAL_BUILDING_MATCH", {})
        system_template = sub_config.get("system_prompt", "")
        max_tokens = sub_config.get("max_tokens", 600)

        # Build lead_preferences from known data
        lead_preferences = ""
        if lead_data.get("interestType"):
            lead_preferences += f"Interest: {lead_data['interestType']}. "
        if lead_data.get("budgetMin") or lead_data.get("budgetMax"):
            lead_preferences += f"Budget: {lead_data.get('budgetMin', '?')}-{lead_data.get('budgetMax', '?')}. "
        if lead_data.get("timeline"):
            lead_preferences += f"Timeline: {lead_data['timeline']}. "

        context = {
            "lead_name": lead_data.get("name", ""),
            "building": conversation_context.get("mostRecentBuilding", ""),
            "language": lead_data.get("language", "es"),
            "message": message,
            "cadence": cadence,
            "graphrag_context": graphrag_context,
            "building_context": building_context,
            "lead_preferences": lead_preferences,
            "qual_context": qual_context,
            "known_items": known_items,
            "broker_aware": False,
            "ghl_conversation_context": conversation_context.get(
                "ghlConversationContext", ""
            ),
            "most_recent_building": conversation_context.get(
                "mostRecentBuilding", ""
            ),
        }

        # Render dynamic blocks
        known_data_block = self._prompt_builder._render_template(
            raw_yaml.get("known_data_block", ""), context, "qualifying.known_data_block"
        )
        cadence_block = self._prompt_builder._render_template(
            raw_yaml.get("cadence_block", ""), context, "qualifying.cadence_block"
        )
        context["known_data_block"] = known_data_block
        context["broker_aware_block"] = ""
        context["cadence_block"] = cadence_block

        system_prompt = self._prompt_builder._render_template(
            system_template, context, "qualifying.building_match.system"
        )
        user_template = raw_yaml.get("user_template", "")
        user_message = self._prompt_builder._render_template(
            user_template, context, "qualifying.building_match.user"
        )

        config = self._prompt_builder.get_config("qualifying")
        try:
            response_text = await self._claude.generate(
                system_prompt=system_prompt,
                user_message=user_message,
                model=config.get("model", "claude-sonnet-4-20250514"),
                max_tokens=max_tokens,
            )
        except CircuitOpenError:
            logger.warning("qualifying.building_match.circuit_open", trace_id=trace_id)
            return ProcessorResult(
                response_text="",
                new_state=None,
                metadata={"fallback_needed": True},
            )

        parsed = _parse_claude_json(response_text)
        response_msg = parsed.get("response", response_text)
        sentiment = parsed.get("sentiment", "neutral")
        sentiment_confidence = float(parsed.get("sentiment_confidence", 0.5))

        # Check escalation
        should_escalate, escalation_reason = _check_escalation(
            message, lead_data, sentiment
        )
        if should_escalate or parsed.get("escalate"):
            reason = escalation_reason or parsed.get("escalation_reason", "building_match_escalation")
            return ProcessorResult(
                response_text=response_msg,
                new_state="HANDOFF",
                should_handoff=True,
                metadata={"escalation_reason": reason},
            )

        # Check next_action
        next_action = parsed.get("next_action", "continue_qualifying")
        if next_action == "handoff_fernando":
            return ProcessorResult(
                response_text=response_msg,
                new_state="HANDOFF",
                should_handoff=True,
                metadata={"escalation_reason": "handoff_fernando"},
            )
        if next_action == "schedule_appointment":
            return ProcessorResult(
                response_text=response_msg,
                new_state="SCHEDULING",
                metadata={"appointment_requested": True},
            )
        if next_action == "send_docs":
            building_name = (
                parsed.get("building_name")
                or conversation_context.get("mostRecentBuilding", "")
            )
            _enqueue_doc_delivery(
                contact_id=lead_data.get("contact_id", ""),
                phone=lead_data.get("phone", ""),
                building_name=building_name,
                channel=lead_data.get("channel", "SMS"),
                trace_id=trace_id,
                language=lead_data.get("language", "es"),
            )
            logger.info(
                "qualifying.building_match.doc_delivery_enqueued",
                trace_id=trace_id,
                building_name=building_name,
            )

        # Pricing alert for building match
        _maybe_send_pricing_alert(
            response_msg=response_msg,
            lead_data=lead_data,
            building_context=building_context,
            message=message,
            trace_id=trace_id,
        )

        # Record discussed building
        matched_building = parsed.get("matched_building")
        if matched_building and self._building_repo and phone:
            try:
                await self._building_repo.record_discussed(phone, matched_building)
            except Exception:
                logger.exception(
                    "qualifying.record_discussed_failed", trace_id=trace_id
                )

        # Save sentiment
        if phone and sentiment:
            try:
                await self._lead_repo.save_sentiment(
                    phone, sentiment, sentiment_confidence
                )
            except Exception:
                logger.exception(
                    "qualifying.sentiment_save_failed", trace_id=trace_id
                )

        logger.info(
            "qualifying.building_match_complete",
            trace_id=trace_id,
            matched_building=matched_building,
            cadence=cadence,
        )

        return ProcessorResult(
            response_text=response_msg,
            new_state=None,  # Stay in QUALIFYING until handoff/scheduling
            metadata={
                "matched_building": matched_building,
                "cadence": cadence,
                "sentiment": sentiment,
            },
        )

    def _next_sub_state(self, current: str) -> str:
        """Return the next sub-state in the progression."""
        try:
            idx = SUB_STATE_ORDER.index(current)
            if idx < len(SUB_STATE_ORDER) - 1:
                return SUB_STATE_ORDER[idx + 1]
        except ValueError:
            pass
        return current  # Stay at current if at end or unknown

    @staticmethod
    def _build_known_items(lead_data: dict) -> list[str]:
        """Build a list of known data items for the knownDataBlock."""
        items = []
        if lead_data.get("name"):
            items.append(f"Nombre: {lead_data['name']}")
        if lead_data.get("interestType"):
            items.append(f"Interes: {lead_data['interestType']}")
        if lead_data.get("budgetMin") or lead_data.get("budgetMax"):
            items.append(
                f"Presupuesto: {lead_data.get('budgetMin', '?')}-{lead_data.get('budgetMax', '?')}"
            )
        if lead_data.get("timeline"):
            items.append(f"Timeline: {lead_data['timeline']}")
        if lead_data.get("email"):
            items.append(f"Email: {lead_data['email']}")
        return items
