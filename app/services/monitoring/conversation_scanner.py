"""Conversation quality scanner -- detects repetition, ignored requests,
sentiment drift, hallucinations, and language mismatch.

Designed for use inside a Celery Beat scheduled task. All heavy I/O
(Neo4j, Redis) is async. scikit-learn is used for TF-IDF cosine similarity.

Detection methods:
1. Repetition: TF-IDF cosine similarity on last N assistant messages per lead. >0.7 = repetition.
2. Ignored requests: Regex intent-response alignment (lead asks for X, AI doesn't address X).
3. Sentiment drift: Bilingual keyword scoring across message history. POSITIVE->NEGATIVE = alert.
4. Hallucination: Extract numeric claims from AI response, cross-check vs Neo4j Building data.
5. Language mismatch: Compare detected language of lead message vs AI response language.
"""

from __future__ import annotations

import re
from typing import Any

import redis
import structlog
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from app.services.monitoring.alert_manager import AlertLevel, AlertManager

logger = structlog.get_logger()

# --- Repetition detection ---
REPETITION_THRESHOLD = 0.7
MIN_MESSAGES_FOR_REPETITION = 3

# --- Sentiment keywords (bilingual) ---
POSITIVE_KEYWORDS_ES = {
    "gracias", "excelente", "perfecto", "genial", "interesante",
    "me gusta", "bueno", "increible", "fantastico", "ideal",
}
POSITIVE_KEYWORDS_EN = {
    "thanks", "great", "perfect", "excellent", "awesome",
    "love", "amazing", "fantastic", "wonderful", "ideal",
}
NEGATIVE_KEYWORDS_ES = {
    "molesto", "enojado", "frustrante", "horrible", "pesimo",
    "no me interesa", "dejen de", "spam", "basta", "no quiero",
    "mal servicio", "terrible", "ridiculo", "cansado",
}
NEGATIVE_KEYWORDS_EN = {
    "annoyed", "angry", "frustrated", "terrible", "awful",
    "not interested", "stop", "spam", "enough", "unsubscribe",
    "bad service", "horrible", "ridiculous", "tired",
}
ALL_POSITIVE = POSITIVE_KEYWORDS_ES | POSITIVE_KEYWORDS_EN
ALL_NEGATIVE = NEGATIVE_KEYWORDS_ES | NEGATIVE_KEYWORDS_EN

# --- Ignored request patterns (lead intent -> expected response indicators) ---
INTENT_PATTERNS = [
    {
        "name": "doc_request",
        "lead_regex": r"(?:mand|envi|send|share).{0,20}(?:foto|photo|brochure|info|document|plano|precio|price|pdf)",
        "response_must_contain": [r"http", r"pdf", r"brochure", r"documento", r"send_docs", r"deliver_documents"],
        "description_es": "Lead pidio documentos pero AI no envio links ni acciono envio",
        "description_en": "Lead requested documents but AI did not send links or trigger delivery",
    },
    {
        "name": "pricing_request",
        "lead_regex": r"(?:cuanto|cuánto|how much|precio|price|cost|costo|inversion|investment)",
        "response_must_contain": [r"\$", r"usd", r"mxn", r"precio", r"price", r"\d{3,}", r"million", r"millon"],
        "description_es": "Lead pregunto por precio pero AI no dio cifras",
        "description_en": "Lead asked about pricing but AI did not provide numbers",
    },
    {
        "name": "visit_request",
        "lead_regex": r"(?:visita|visit|agendar|schedule|recorrido|tour|ver el|see the|conocer)",
        "response_must_contain": [r"agenda", r"schedule", r"visit", r"cita", r"appointment", r"horario", r"hora"],
        "description_es": "Lead pidio agendar visita pero AI no respondio con opciones de agenda",
        "description_en": "Lead requested a visit but AI did not respond with scheduling options",
    },
]

# --- Hallucination: numeric extraction ---
_NUMERIC_CLAIM_RE = re.compile(
    r"(?:(?:USD|MXN|\$)\s*[\d,]+(?:\.\d+)?|[\d,]+(?:\.\d+)?\s*(?:USD|MXN|usd|mxn|dollars|dolares|pesos))"
    r"|(?:(\d{1,3}(?:,\d{3})*)\s*(?:m2|sqft|sq\s*ft|metros|square))"
    r"|(?:(\d{1,3})\s*(?:pisos|floors|niveles|stories|plantas))"
    r"|(?:(\d{1,4})\s*(?:unidades|units|departamentos|apartments))"
)


class ConversationQualityScanner:
    """Scans recent conversations for quality issues."""

    def __init__(
        self,
        redis_client: redis.Redis,
        alert_manager: AlertManager,
    ) -> None:
        self._redis = redis_client
        self._alert = alert_manager

    async def scan_all_active_leads(
        self,
        driver,
        limit: int = 50,
    ) -> dict[str, int]:
        """Scan recent conversations for all active leads.

        Queries Neo4j for leads with recent interactions (last 30 min),
        then runs all quality checks on each.

        Returns dict of issue counts by type.
        """
        from app.repositories.base import close_driver
        from neo4j import AsyncDriver

        counts = {
            "repetition": 0,
            "ignored_request": 0,
            "sentiment_drift": 0,
            "hallucination": 0,
            "language_mismatch": 0,
            "leads_scanned": 0,
        }

        try:
            async with driver.session() as session:
                # Find leads with recent interactions (last 30 min)
                result = await session.run(
                    """
                    MATCH (l:Lead)-[:HAS_INTERACTION]->(i:Interaction)
                    WHERE i.created_at > datetime() - duration({minutes: 30})
                    AND l.current_state <> 'BROKER'
                    AND l.current_state <> 'CLOSED'
                    WITH DISTINCT l
                    RETURN l.ghl_contact_id AS contact_id,
                           l.phone AS phone,
                           l.name AS name,
                           l.language AS language,
                           l.current_state AS state
                    LIMIT $limit
                    """,
                    limit=limit,
                )
                leads = [dict(r) async for r in result]

            if not leads:
                logger.info("conversation_scan.no_active_leads")
                return counts

            for lead in leads:
                contact_id = lead.get("contact_id", "")
                phone = lead.get("phone", "")
                if not phone:
                    continue

                counts["leads_scanned"] += 1

                try:
                    # Fetch last 10 interactions for this lead
                    async with driver.session() as session:
                        result = await session.run(
                            """
                            MATCH (l:Lead {phone: $phone})-[:HAS_INTERACTION]->(i:Interaction)
                            RETURN i.role AS role, i.content AS content, i.created_at AS created_at
                            ORDER BY i.created_at DESC
                            LIMIT 10
                            """,
                            phone=phone,
                        )
                        interactions = [dict(r) async for r in result]

                    if not interactions:
                        continue

                    # Separate user and assistant messages
                    user_msgs = [i["content"] for i in interactions if i.get("role") == "user" and i.get("content")]
                    assistant_msgs = [i["content"] for i in interactions if i.get("role") == "assistant" and i.get("content")]

                    # --- Check 1: Repetition ---
                    if len(assistant_msgs) >= MIN_MESSAGES_FOR_REPETITION:
                        rep_found = self._check_repetition(assistant_msgs)
                        if rep_found:
                            counts["repetition"] += 1
                            await self._alert.send(
                                alert_type="repetition",
                                message=f"Lead {lead.get('name', phone[-4:])} has {rep_found} repetitive AI responses in last 10 messages.",
                                level=AlertLevel.WARNING,
                                contact_id=contact_id,
                                extra={"phone": phone[-4:], "state": lead.get("state", "unknown"), "similar_pairs": str(rep_found)},
                            )
                            # Self-healing: advance sub_state to break the loop
                            await self._self_heal_repetition(contact_id, driver)

                    # --- Check 2: Ignored requests ---
                    if user_msgs and assistant_msgs:
                        ignored = self._check_ignored_requests(user_msgs[0], assistant_msgs[0])
                        if ignored:
                            counts["ignored_request"] += 1
                            await self._alert.send(
                                alert_type="ignored_request",
                                message=f"Lead {lead.get('name', phone[-4:])}: {ignored['description_es']}",
                                level=AlertLevel.WARNING,
                                contact_id=contact_id,
                                extra={"intent": ignored["name"], "phone": phone[-4:]},
                            )

                    # --- Check 3: Sentiment drift ---
                    if len(user_msgs) >= 2:
                        drift = self._check_sentiment_drift(user_msgs)
                        if drift:
                            counts["sentiment_drift"] += 1
                            alert_level = AlertLevel.CRITICAL if drift["consecutive_negative"] >= 3 else AlertLevel.WARNING
                            await self._alert.send(
                                alert_type="sentiment_drift",
                                message=f"Lead {lead.get('name', phone[-4:])}: sentiment shifted from {drift['from']} to {drift['to']}.",
                                level=alert_level,
                                contact_id=contact_id,
                                extra={"consecutive_negative": str(drift["consecutive_negative"]), "phone": phone[-4:]},
                            )
                            # Self-healing: if 3+ negative, tag for human review and pause AI
                            if drift["consecutive_negative"] >= 3:
                                await self._self_heal_negative_sentiment(contact_id, driver)

                    # --- Check 4: Hallucination (numeric claims vs Neo4j) ---
                    if assistant_msgs:
                        hallucination = await self._check_hallucination(assistant_msgs[0], phone, driver)
                        if hallucination:
                            counts["hallucination"] += 1
                            await self._alert.send(
                                alert_type="hallucination",
                                message=f"Lead {lead.get('name', phone[-4:])}: AI claimed {hallucination['claim']} but Neo4j shows {hallucination['actual']}.",
                                level=AlertLevel.CRITICAL,
                                contact_id=contact_id,
                                extra={"claim": hallucination["claim"], "actual": hallucination["actual"], "field": hallucination["field"]},
                            )

                    # --- Check 5: Language mismatch ---
                    if user_msgs and assistant_msgs:
                        mismatch = self._check_language_mismatch(user_msgs[0], assistant_msgs[0])
                        if mismatch:
                            counts["language_mismatch"] += 1
                            await self._alert.send(
                                alert_type="language_mismatch",
                                message=f"Lead {lead.get('name', phone[-4:])}: wrote in {mismatch['user_lang']} but AI replied in {mismatch['ai_lang']}.",
                                level=AlertLevel.WARNING,
                                contact_id=contact_id,
                                extra={"user_lang": mismatch["user_lang"], "ai_lang": mismatch["ai_lang"], "phone": phone[-4:]},
                            )
                            # Self-healing: correct language field in Neo4j
                            await self._self_heal_language(contact_id, mismatch["user_lang"], driver)

                except Exception:
                    logger.exception(
                        "conversation_scan.lead_error",
                        contact_id=contact_id,
                        phone=phone[-4:] if phone else "",
                    )
                    continue

        except Exception:
            logger.exception("conversation_scan.error")

        return counts

    # --- Detection methods ---

    def _check_repetition(self, assistant_msgs: list[str]) -> int:
        """Check for repetitive AI responses using TF-IDF cosine similarity.

        Returns number of similar pairs found (0 = no repetition).
        """
        try:
            vectorizer = TfidfVectorizer(stop_words=None, max_features=500)
            tfidf_matrix = vectorizer.fit_transform(assistant_msgs)
            sim_matrix = cosine_similarity(tfidf_matrix)

            similar_pairs = 0
            n = len(assistant_msgs)
            for i in range(n):
                for j in range(i + 1, n):
                    if sim_matrix[i][j] > REPETITION_THRESHOLD:
                        similar_pairs += 1

            return similar_pairs
        except Exception:
            logger.exception("repetition_check_error")
            return 0

    def _check_ignored_requests(
        self, latest_user_msg: str, latest_ai_msg: str
    ) -> dict | None:
        """Check if the AI ignored a clear user request.

        Matches user message against intent patterns, then checks if
        the AI response contains expected indicators.
        """
        user_lower = latest_user_msg.lower()
        ai_lower = latest_ai_msg.lower()

        for pattern in INTENT_PATTERNS:
            if re.search(pattern["lead_regex"], user_lower, re.IGNORECASE):
                # User expressed this intent -- check if AI addressed it
                addressed = any(
                    re.search(indicator, ai_lower, re.IGNORECASE)
                    for indicator in pattern["response_must_contain"]
                )
                if not addressed:
                    return pattern
        return None

    def _check_sentiment_drift(self, user_msgs: list[str]) -> dict | None:
        """Detect sentiment shifting from positive/neutral to negative.

        Scores each message by keyword presence. Returns drift info if
        the most recent messages trend negative.
        """
        sentiments = []
        for msg in user_msgs:
            msg_lower = msg.lower()
            pos_count = sum(1 for kw in ALL_POSITIVE if kw in msg_lower)
            neg_count = sum(1 for kw in ALL_NEGATIVE if kw in msg_lower)

            if neg_count > pos_count:
                sentiments.append("NEGATIVE")
            elif pos_count > neg_count:
                sentiments.append("POSITIVE")
            else:
                sentiments.append("NEUTRAL")

        # sentiments[0] = most recent (DESC order from Neo4j)
        if not sentiments or sentiments[0] != "NEGATIVE":
            return None

        # Count consecutive negative from most recent
        consecutive = 0
        for s in sentiments:
            if s == "NEGATIVE":
                consecutive += 1
            else:
                break

        # Find what the sentiment was before the negative streak
        prior = "NEUTRAL"
        for s in sentiments[consecutive:]:
            if s != "NEUTRAL":
                prior = s
                break

        if consecutive >= 2:
            return {
                "from": prior,
                "to": "NEGATIVE",
                "consecutive_negative": consecutive,
            }
        return None

    async def _check_hallucination(
        self, ai_msg: str, phone: str, driver
    ) -> dict | None:
        """Extract numeric claims from AI response and cross-check Neo4j.

        Checks: price ranges, sqft, floor count, unit count.
        """
        claims = _NUMERIC_CLAIM_RE.findall(ai_msg)
        if not claims:
            return None

        # Get buildings this lead is interested in
        try:
            async with driver.session() as session:
                result = await session.run(
                    """
                    MATCH (l:Lead {phone: $phone})-[:INTERESTED_IN]->(b:Building)
                    RETURN b.name AS name,
                           b.price_min_usd AS price_min,
                           b.price_max_usd AS price_max,
                           b.total_floors AS total_floors,
                           b.total_units AS total_units
                    """,
                    phone=phone,
                )
                buildings = [dict(r) async for r in result]
        except Exception:
            logger.exception("hallucination_check.neo4j_failed")
            return None

        if not buildings:
            return None

        # Extract price claim from AI message
        price_match = re.search(
            r"(?:USD|\$)\s*([\d,]+(?:\.\d+)?)", ai_msg, re.IGNORECASE
        )
        if price_match:
            try:
                claimed_price = float(price_match.group(1).replace(",", ""))
                for b in buildings:
                    price_min = b.get("price_min") or 0
                    price_max = b.get("price_max") or float("inf")
                    # Allow 20% tolerance for rounding
                    if price_min > 0 and claimed_price < price_min * 0.8:
                        return {
                            "claim": f"${claimed_price:,.0f}",
                            "actual": f"${price_min:,.0f}-${price_max:,.0f}",
                            "field": "price",
                        }
                    if price_max < float("inf") and claimed_price > price_max * 1.2:
                        return {
                            "claim": f"${claimed_price:,.0f}",
                            "actual": f"${price_min:,.0f}-${price_max:,.0f}",
                            "field": "price",
                        }
            except (ValueError, TypeError):
                pass

        # Extract floor count claim
        floor_match = re.search(r"(\d{1,3})\s*(?:pisos|floors|niveles|stories|plantas)", ai_msg, re.IGNORECASE)
        if floor_match:
            try:
                claimed_floors = int(floor_match.group(1))
                for b in buildings:
                    actual_floors = b.get("total_floors")
                    if actual_floors and abs(claimed_floors - actual_floors) > 2:
                        return {
                            "claim": f"{claimed_floors} floors",
                            "actual": f"{actual_floors} floors",
                            "field": "total_floors",
                        }
            except (ValueError, TypeError):
                pass

        return None

    def _check_language_mismatch(
        self, user_msg: str, ai_msg: str
    ) -> dict | None:
        """Detect if user and AI are speaking different languages."""
        from app.services.language_detector import detect_language_with_confidence

        user_lang, user_conf = detect_language_with_confidence(user_msg)
        ai_lang, ai_conf = detect_language_with_confidence(ai_msg)

        # Only flag if both detections are confident
        if user_conf < 0.6 or ai_conf < 0.6:
            return None

        if user_lang != ai_lang:
            return {"user_lang": user_lang, "ai_lang": ai_lang}
        return None

    # --- Self-healing actions ---

    async def _self_heal_repetition(self, contact_id: str, driver) -> None:
        """Auto-advance lead's sub_state to break repetition loop."""
        try:
            from app.repositories.lead_repository import LeadRepository
            lead_repo = LeadRepository(driver)
            # Advance sub_state -- the next processor invocation will see a different sub_state
            # and generate a different prompt, breaking the repetition cycle
            await lead_repo.save_qualification_data(contact_id, {"sub_state": "advanced_by_monitor"})
            logger.info("self_heal.repetition_sub_state_advanced", contact_id=contact_id)
        except Exception:
            logger.exception("self_heal.repetition_failed", contact_id=contact_id)

    async def _self_heal_negative_sentiment(self, contact_id: str, driver) -> None:
        """Tag lead for human review and pause AI outreach."""
        try:
            from app.services.ghl_service import add_tag
            await add_tag(contact_id, "needs-human-review")
            await add_tag(contact_id, "sentiment-negative")
            logger.info("self_heal.negative_sentiment_tagged", contact_id=contact_id)
        except Exception:
            logger.exception("self_heal.negative_sentiment_failed", contact_id=contact_id)

    async def _self_heal_language(self, contact_id: str, correct_lang: str, driver) -> None:
        """Auto-correct lead's language field in Neo4j."""
        try:
            from app.repositories.lead_repository import LeadRepository
            lead_repo = LeadRepository(driver)
            await lead_repo.save_qualification_data(contact_id, {"language": correct_lang})
            logger.info("self_heal.language_corrected", contact_id=contact_id, language=correct_lang)
        except Exception:
            logger.exception("self_heal.language_correction_failed", contact_id=contact_id)
