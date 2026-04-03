"""Error classifier -- deterministic rule-based classification of conversation errors.

Reuses detection logic from ConversationQualityScanner (DRY). Classifies
errors into 5 types: repetition, missed_intent, tone_mismatch,
hallucination, wrong_information.

After classification, persists errors to Neo4j via LearningRepository and
auto-generates candidate lessons when error patterns reach frequency >= 3.
"""

from __future__ import annotations

import re
from typing import Any

import structlog
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from app.services.monitoring.alert_manager import AlertLevel, AlertManager
from app.services.monitoring.conversation_scanner import (
    ALL_NEGATIVE,
    ALL_POSITIVE,
    INTENT_PATTERNS,
    MIN_MESSAGES_FOR_REPETITION,
    REPETITION_THRESHOLD,
    _NUMERIC_CLAIM_RE,
)

logger = structlog.get_logger()


class ErrorClassifier:
    """Deterministic error classification pipeline for conversations."""

    def __init__(
        self,
        repo,  # LearningRepository -- lazy import avoidance
        alert_manager: AlertManager,
    ) -> None:
        self._repo = repo
        self._alert = alert_manager

    async def classify(
        self,
        lead_data: dict[str, Any],
        messages: list[dict[str, Any]],
        buildings: list[dict[str, Any]],
    ) -> list[dict[str, str]]:
        """Classify errors in a conversation.

        Args:
            lead_data: {contact_id, phone, name, state, language}
            messages: [{role, content, created_at}] most recent first
            buildings: Building dicts from Neo4j

        Returns:
            List of {type, details, severity} error dicts.
        """
        errors: list[dict[str, str]] = []

        assistant_msgs = [
            m["content"]
            for m in messages
            if m.get("role") == "assistant" and m.get("content")
        ]
        user_msgs = [
            m["content"]
            for m in messages
            if m.get("role") == "user" and m.get("content")
        ]

        # 1. Repetition check
        if len(assistant_msgs) >= MIN_MESSAGES_FOR_REPETITION:
            similar_pairs = self._check_repetition(assistant_msgs)
            if similar_pairs > 0:
                errors.append({
                    "type": "repetition",
                    "details": f"{similar_pairs} similar response pairs",
                    "severity": "warning",
                })

        # 2. Ignored request check
        if user_msgs and assistant_msgs:
            ignored = self._check_ignored_request(user_msgs[0], assistant_msgs[0])
            if ignored:
                errors.append({
                    "type": "missed_intent",
                    "details": ignored["description_en"],
                    "severity": "warning",
                })

        # 3. Sentiment / tone mismatch check
        if user_msgs and assistant_msgs:
            tone_mismatch = self._check_tone_mismatch(user_msgs[0], assistant_msgs[0])
            if tone_mismatch:
                errors.append({
                    "type": "tone_mismatch",
                    "details": "AI responded positively to negative user sentiment",
                    "severity": "warning",
                })

        # 4. Hallucination check
        if assistant_msgs and buildings:
            hallucination = self._check_hallucination(assistant_msgs[0], buildings)
            if hallucination:
                errors.append({
                    "type": "hallucination",
                    "details": f"Claimed {hallucination['claim']} but actual is {hallucination['actual']}",
                    "severity": "critical",
                })

        # Persist errors and update patterns
        contact_id = lead_data.get("contact_id")
        phone = lead_data.get("phone", "")
        conversation_id = f"{contact_id}:{messages[0].get('created_at', 'unknown')}" if messages else contact_id

        for error in errors:
            try:
                await self._repo.create_agent_error(
                    conversation_id=conversation_id,
                    error_type=error["type"],
                    details=error["details"],
                    severity=error["severity"],
                    contact_id=contact_id,
                )
                frequency = await self._repo.increment_error_pattern(error["type"])
                if frequency >= 3:
                    await self._auto_generate_lesson(error["type"], frequency)
            except Exception:
                logger.exception(
                    "error_classifier.persist_failed",
                    error_type=error["type"],
                )

        return errors

    # --- Detection methods (reuse scanner logic, DRY) ---

    @staticmethod
    def _check_repetition(assistant_msgs: list[str]) -> int:
        """TF-IDF cosine similarity check for repetitive responses."""
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
            logger.exception("error_classifier.repetition_check_failed")
            return 0

    @staticmethod
    def _check_ignored_request(
        latest_user_msg: str, latest_ai_msg: str
    ) -> dict | None:
        """Check if AI ignored a clear user request via intent pattern matching."""
        user_lower = latest_user_msg.lower()
        ai_lower = latest_ai_msg.lower()

        for pattern in INTENT_PATTERNS:
            if re.search(pattern["lead_regex"], user_lower, re.IGNORECASE):
                addressed = any(
                    re.search(indicator, ai_lower, re.IGNORECASE)
                    for indicator in pattern["response_must_contain"]
                )
                if not addressed:
                    return pattern
        return None

    @staticmethod
    def _check_tone_mismatch(user_msg: str, ai_msg: str) -> bool:
        """Check if AI responded positively to negative user sentiment."""
        user_lower = user_msg.lower()
        ai_lower = ai_msg.lower()

        user_neg = sum(1 for kw in ALL_NEGATIVE if kw in user_lower)
        user_pos = sum(1 for kw in ALL_POSITIVE if kw in user_lower)
        ai_neg = sum(1 for kw in ALL_NEGATIVE if kw in ai_lower)
        ai_pos = sum(1 for kw in ALL_POSITIVE if kw in ai_lower)

        # User is negative AND AI is positive = tone-deaf
        return user_neg > user_pos and ai_pos > ai_neg

    @staticmethod
    def _check_hallucination(
        ai_msg: str, buildings: list[dict[str, Any]]
    ) -> dict | None:
        """Extract numeric claims from AI and cross-check against building data."""
        if not buildings:
            return None

        # Price claim check
        price_match = re.search(
            r"(?:USD|\$)\s*([\d,]+(?:\.\d+)?)", ai_msg, re.IGNORECASE
        )
        if price_match:
            try:
                claimed_price = float(price_match.group(1).replace(",", ""))
                for b in buildings:
                    price_min = b.get("price_min_usd") or b.get("price_min") or 0
                    price_max = b.get("price_max_usd") or b.get("price_max") or float("inf")
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

        # Floor count claim check
        floor_match = re.search(
            r"(\d{1,3})\s*(?:pisos|floors|niveles|stories|plantas)",
            ai_msg,
            re.IGNORECASE,
        )
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

    async def _auto_generate_lesson(self, error_type: str, frequency: int) -> None:
        """Auto-generate a candidate lesson when error pattern reaches threshold."""
        severity = "critical" if error_type == "hallucination" else "warning"
        rule = f"Pattern detected: {error_type} occurred {frequency} times"
        why = f"Automated detection of recurring {error_type} errors"

        try:
            await self._repo.create_lesson_learned(
                rule=rule,
                why=why,
                severity=severity,
                confidence=0.5,
            )
            await self._alert.send(
                alert_type="lesson_candidate",
                message=(
                    f"Auto-generated lesson candidate for recurring {error_type} "
                    f"(frequency={frequency}). Review in Neo4j."
                ),
                level=AlertLevel.INFO,
            )
            logger.info(
                "error_classifier.lesson_auto_generated",
                error_type=error_type,
                frequency=frequency,
            )
        except Exception:
            logger.exception(
                "error_classifier.lesson_generation_failed",
                error_type=error_type,
            )
