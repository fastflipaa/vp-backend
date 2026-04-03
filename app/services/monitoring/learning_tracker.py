"""Track learning system effectiveness via before/after outcome metrics.

Compares average conversation scores (last 7 days vs prior 7 days)
to measure whether lessons are improving agent quality.
"""

from __future__ import annotations

import structlog

from app.repositories.learning_repository import LearningRepository

logger = structlog.get_logger()


class LearningEffectivenessTracker:
    """Compute and report on learning system effectiveness."""

    def __init__(self, repo: LearningRepository) -> None:
        self._repo = repo

    async def compute_metrics(self) -> dict:
        """Compare last 7 days vs prior 7 days conversation outcome averages.

        Returns dict with current_7d, previous_7d, deltas, and injection_count_7d.
        Derives prior period from weighted subtraction of 14d and 7d averages.
        """
        try:
            current = await self._repo.get_outcome_averages(days=7)
        except Exception:
            logger.exception("learning_tracker.current_period_failed")
            return {}

        try:
            all_14d = await self._repo.get_outcome_averages(days=14)
        except Exception:
            logger.exception("learning_tracker.total_period_failed")
            all_14d = {"repetition": 0.0, "sentiment": 0.0, "intent_alignment": 0.0, "count": 0}

        # Compute prior period averages by weighted subtraction
        curr_count = current.get("count", 0)
        total_count = all_14d.get("count", 0)
        prior_count = total_count - curr_count

        previous: dict = {"count": prior_count}
        for key in ("repetition", "sentiment", "intent_alignment"):
            if prior_count > 0 and total_count > 0:
                total_sum = all_14d.get(key, 0.0) * total_count
                curr_sum = current.get(key, 0.0) * curr_count
                previous[key] = round((total_sum - curr_sum) / prior_count, 4)
            else:
                previous[key] = 0.0

        deltas: dict = {}
        for key in ("repetition", "sentiment", "intent_alignment"):
            deltas[key] = round(current.get(key, 0.0) - previous.get(key, 0.0), 4)

        injection_count = 0
        try:
            injection_count = await self._repo.get_lesson_injection_count(days=7)
        except Exception:
            logger.warning("learning_tracker.injection_count_failed")

        return {
            "current_7d": current,
            "previous_7d": previous,
            "deltas": deltas,
            "injection_count_7d": injection_count,
        }

    async def generate_weekly_report(self) -> str:
        """Format metrics as a Slack-friendly message string."""
        metrics = await self.compute_metrics()
        if not metrics:
            return "Learning Effectiveness Report: insufficient data"

        current = metrics.get("current_7d", {})
        previous = metrics.get("previous_7d", {})
        deltas = metrics.get("deltas", {})
        injections = metrics.get("injection_count_7d", 0)

        def arrow(val: float, key: str) -> str:
            # For repetition, LOWER is better (negative delta = improvement)
            # For sentiment and intent_alignment, HIGHER is better
            if key == "repetition":
                return "v (improved)" if val < 0 else "^ (worsened)" if val > 0 else "="
            else:
                return "^ (improved)" if val > 0 else "v (worsened)" if val < 0 else "="

        lines = [
            ":bar_chart: *Weekly Learning Effectiveness Report*",
            "",
            f"*Conversations scored (7d):* {current.get('count', 0)} | Prior 7d: {previous.get('count', 0)}",
            f"*Lesson injections (7d):* {injections}",
            "",
            "*Score Averages (current vs prior):*",
            f"  Repetition: {current.get('repetition', 0):.3f} vs {previous.get('repetition', 0):.3f} {arrow(deltas.get('repetition', 0), 'repetition')}",
            f"  Sentiment: {current.get('sentiment', 0):.3f} vs {previous.get('sentiment', 0):.3f} {arrow(deltas.get('sentiment', 0), 'sentiment')}",
            f"  Intent Alignment: {current.get('intent_alignment', 0):.3f} vs {previous.get('intent_alignment', 0):.3f} {arrow(deltas.get('intent_alignment', 0), 'intent_alignment')}",
        ]

        return "\n".join(lines)
