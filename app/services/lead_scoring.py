"""Lead scoring service -- weighted 1-100 score computation.

Computes a composite lead score using four weighted factors:
  - Budget (0.30): Explicit budget data from Neo4j Lead properties
  - Engagement (0.25): Interaction volume, reply rate, quality signals
  - Timeline (0.25): Urgency keywords and explicit timeline data
  - Response Speed (0.20): Average reply latency from last 5 user replies

Missing data is handled via weight redistribution: factors without data
have their weight distributed proportionally among available factors,
so leads are never penalised for incomplete information.

Tier thresholds (configurable via settings):
  - Hot: score >= 80
  - Warm: score >= 50
  - Cold: score < 50
"""

from __future__ import annotations

import re
from typing import Any

import structlog

from app.config import settings
from app.repositories.lead_repository import LeadRepository

logger = structlog.get_logger()

# --- Weight constants ---
WEIGHT_BUDGET = 0.30
WEIGHT_ENGAGEMENT = 0.25
WEIGHT_TIMELINE = 0.25
WEIGHT_RESPONSE_SPEED = 0.20

# --- Keyword patterns (case-insensitive) ---
_URGENT_KEYWORDS = re.compile(
    r"(este mes|esta semana|urgente|this month|asap|inmediato|inmediata|lo antes posible|cuanto antes)",
    re.IGNORECASE,
)
_SCHEDULING_FINANCING_KEYWORDS = re.compile(
    r"(visita|visit|financiamiento|financing|credito|hipoteca|mortgage|cita|agendar|appointment)",
    re.IGNORECASE,
)


class LeadScoringService:
    """Computes and persists 1-100 lead scores with tier classification."""

    def __init__(self, lead_repo: LeadRepository) -> None:
        self._lead_repo = lead_repo

    async def compute_score(
        self,
        contact_id: str,
        enriched_context: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Compute lead score from Neo4j signals, persist, and return result.

        Args:
            contact_id: GHL contact ID.
            enriched_context: Optional GHL custom field data (bedroom count,
                budget from form submissions, etc.).

        Returns:
            Dict with score, tier, factors breakdown, previous_score,
            and crossed_hot flag.
        """
        if not settings.LEAD_SCORE_ENABLED:
            logger.info("lead_scoring.disabled", contact_id=contact_id)
            return {
                "score": 0,
                "tier": "cold",
                "factors": {},
                "previous_score": None,
                "crossed_hot": False,
            }

        signals = await self._lead_repo.get_scoring_signals(contact_id)

        # Compute individual factor scores (None = missing data)
        budget_score = self._compute_budget(signals, enriched_context)
        engagement_score = self._compute_engagement(signals)
        timeline_score = self._compute_timeline(signals)
        speed_score = self._compute_response_speed(signals)

        # Build factors dict
        factors: dict[str, dict[str, Any]] = {
            "budget": {
                "raw_score": budget_score,
                "weight": WEIGHT_BUDGET,
                "adjusted_weight": 0.0,
                "source": self._budget_source(signals, enriched_context, budget_score),
            },
            "engagement": {
                "raw_score": engagement_score,
                "weight": WEIGHT_ENGAGEMENT,
                "adjusted_weight": 0.0,
                "source": "interaction_count" if engagement_score is not None else "missing",
            },
            "timeline": {
                "raw_score": timeline_score,
                "weight": WEIGHT_TIMELINE,
                "adjusted_weight": 0.0,
                "source": self._timeline_source(signals, timeline_score),
            },
            "response_speed": {
                "raw_score": speed_score,
                "weight": WEIGHT_RESPONSE_SPEED,
                "adjusted_weight": 0.0,
                "source": "reply_times" if speed_score is not None else "missing",
            },
        }

        # Weight redistribution for missing factors
        available = {k: v for k, v in factors.items() if v["raw_score"] is not None}

        if not available:
            # No data at all -- neutral low score
            final_score = 25
            tier = "cold"
            for f in factors.values():
                f["adjusted_weight"] = 0.0
        else:
            total_available_weight = sum(v["weight"] for v in available.values())
            weighted_sum = 0.0
            for key, fdata in available.items():
                adjusted = fdata["weight"] / total_available_weight
                fdata["adjusted_weight"] = round(adjusted, 4)
                weighted_sum += fdata["raw_score"] * adjusted

            final_score = max(1, min(100, round(weighted_sum)))

            # Determine tier
            if final_score >= settings.LEAD_SCORE_HOT_THRESHOLD:
                tier = "hot"
            elif final_score >= settings.LEAD_SCORE_WARM_THRESHOLD:
                tier = "warm"
            else:
                tier = "cold"

        # Add summary fields to factors dict
        factors["final_score"] = final_score  # type: ignore[assignment]
        factors["tier"] = tier  # type: ignore[assignment]

        # Hot-lead crossing detection
        previous_score = signals.get("current_score")
        crossed_hot = (
            final_score >= settings.LEAD_SCORE_HOT_THRESHOLD
            and (previous_score is None or previous_score < settings.LEAD_SCORE_HOT_THRESHOLD)
        )

        # Persist to Neo4j
        await self._lead_repo.save_lead_score(contact_id, final_score, tier, factors)

        logger.info(
            "lead_score.computed",
            contact_id=contact_id,
            score=final_score,
            tier=tier,
            previous_score=previous_score,
            crossed_hot=crossed_hot,
            budget=budget_score,
            engagement=engagement_score,
            timeline=timeline_score,
            response_speed=speed_score,
        )

        return {
            "score": final_score,
            "tier": tier,
            "factors": factors,
            "previous_score": previous_score,
            "crossed_hot": crossed_hot,
        }

    # --- Factor computations ---

    @staticmethod
    def _compute_budget(
        signals: dict[str, Any],
        enriched_context: dict[str, Any] | None,
    ) -> int | None:
        """Budget factor: 0-100 based on explicit budget data.

        Priority: Neo4j budget properties > GHL custom fields > keyword inference.
        For Polanco luxury context: higher budgets score higher.
        """
        budget_min = signals.get("budgetMin")
        budget_max = signals.get("budgetMax")

        # Primary: explicit budget on Lead node
        if budget_max is not None or budget_min is not None:
            # Use budgetMax as primary indicator; fall back to budgetMin
            budget_val = budget_max or budget_min
            try:
                budget_num = float(budget_val)
            except (TypeError, ValueError):
                # Budget exists but is not numeric -- still a signal
                return 40

            # Polanco luxury thresholds (MXN)
            if budget_num > 10_000_000:
                return 90
            if budget_num > 5_000_000:
                return 70
            if budget_num > 3_000_000:
                return 50
            return 40

        # Secondary: GHL custom fields (bedroom count, form budget)
        if enriched_context:
            bedroom_count = enriched_context.get("bedroom_count")
            form_budget = enriched_context.get("budget")
            if bedroom_count or form_budget:
                # Secondary source: lower confidence, score 40-70
                if form_budget:
                    try:
                        fb = float(form_budget)
                        if fb > 10_000_000:
                            return 70
                        if fb > 5_000_000:
                            return 60
                        return 50
                    except (TypeError, ValueError):
                        return 50
                return 40  # bedroom count exists but no explicit budget

        # Tertiary: quality interactions mentioning pricing
        quality = signals.get("quality_interactions", 0)
        if quality and quality > 0:
            # Inferred interest -- lower score range
            if quality >= 3:
                return 40
            return 20

        return None  # Missing data

    @staticmethod
    def _compute_engagement(signals: dict[str, Any]) -> int | None:
        """Engagement factor: interaction volume + reply rate + quality bonus."""
        total = signals.get("total_interactions", 0)

        if total == 0:
            return 0

        # Base score from interaction count
        if total >= 16:
            base = 80
        elif total >= 8:
            base = 60
        elif total >= 4:
            base = 40
        elif total >= 1:
            base = 20
        else:
            base = 0

        # Reply rate bonus: user_interactions / assistant_interactions * 20
        user_count = signals.get("user_interactions", 0)
        assistant_count = max(signals.get("assistant_interactions", 0), 1)
        reply_rate_bonus = min((user_count / assistant_count) * 20, 20)

        # Quality bonus: quality_interactions / user_interactions * 20
        quality_count = signals.get("quality_interactions", 0)
        user_for_quality = max(user_count, 1)
        quality_bonus = min((quality_count / user_for_quality) * 20, 20)

        return min(round(base + reply_rate_bonus + quality_bonus), 100)

    @staticmethod
    def _compute_timeline(signals: dict[str, Any]) -> int | None:
        """Timeline factor: urgency keywords and explicit timeline data."""
        timeline = signals.get("timeline")

        if timeline:
            timeline_str = str(timeline).strip()
            if _URGENT_KEYWORDS.search(timeline_str):
                return 90
            # Timeline field exists with any value -- moderate urgency
            return 60

        # Behavioral: quality interactions include scheduling/financing keywords
        quality = signals.get("quality_interactions", 0)
        if quality and quality > 0:
            return 50

        return None  # Missing data

    @staticmethod
    def _compute_response_speed(signals: dict[str, Any]) -> int | None:
        """Response speed factor: average of recent reply time gaps in minutes."""
        reply_times = signals.get("recent_reply_times", [])

        if not reply_times:
            return None  # Missing data

        avg_minutes = sum(reply_times) / len(reply_times)

        if avg_minutes < 5:
            return 100
        if avg_minutes < 15:
            return 80
        if avg_minutes < 60:
            return 60
        if avg_minutes < 240:
            return 40
        if avg_minutes < 1440:
            return 20
        return 10

    # --- Source label helpers ---

    @staticmethod
    def _budget_source(
        signals: dict[str, Any],
        enriched_context: dict[str, Any] | None,
        score: int | None,
    ) -> str:
        if score is None:
            return "missing"
        if signals.get("budgetMax") is not None or signals.get("budgetMin") is not None:
            return "neo4j_property"
        if enriched_context and (enriched_context.get("bedroom_count") or enriched_context.get("budget")):
            return "ghl_custom_field"
        return "keyword_inference"

    @staticmethod
    def _timeline_source(signals: dict[str, Any], score: int | None) -> str:
        if score is None:
            return "missing"
        if signals.get("timeline"):
            return "neo4j_property"
        return "behavioral_inference"
