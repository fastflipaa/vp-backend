"""Auto-generate candidate lessons from recurring error patterns.

Uses deterministic TEMPLATE rules (no LLM). Checks semantic dedup
before creating to avoid duplicate lessons.
"""

from __future__ import annotations

from typing import Any

import structlog

from app.repositories.learning_repository import LearningRepository
from app.services.monitoring.alert_manager import AlertLevel, AlertManager

logger = structlog.get_logger()

LESSON_TEMPLATES: dict[str, dict[str, str]] = {
    "repetition": {
        "rule": "Avoid repeating similar responses. Rephrase key information differently each time.",
        "why": "Leads disengage when they receive identical or near-identical responses",
        "severity": "warning",
    },
    "hallucination": {
        "rule": "Double-check all numeric claims against building data before responding.",
        "why": "Incorrect prices, sizes, or unit counts destroy credibility",
        "severity": "critical",
    },
    "missed_intent": {
        "rule": "Prioritize document/media delivery when a lead requests photos, plans, or pricing.",
        "why": "Ignoring explicit requests frustrates leads and delays sales",
        "severity": "warning",
    },
    "tone_mismatch": {
        "rule": "Match the lead's emotional tone before providing information.",
        "why": "Tone-deaf responses to frustrated leads cause disengagement",
        "severity": "warning",
    },
    "wrong_information": {
        "rule": "Verify building details against the knowledge base before stating facts.",
        "why": "Incorrect building info leads to lost trust and potential legal issues",
        "severity": "critical",
    },
}


class AutoLessonGenerator:
    """Generate candidate lessons from error patterns using templates."""

    def __init__(self, repo: LearningRepository, alert_manager: AlertManager) -> None:
        self._repo = repo
        self._alert = alert_manager

    async def analyze_and_generate(self) -> list[dict[str, Any]]:
        """Scan error patterns with frequency >= 3, generate lessons for unmatched ones.

        Returns list of generated lesson dicts with {id, pattern_name, rule}.
        """
        generated: list[dict[str, Any]] = []

        try:
            patterns = await self._repo.get_error_patterns(min_frequency=3)
        except Exception:
            logger.exception("lesson_generator.get_patterns_failed")
            return generated

        for pattern in patterns:
            pattern_name = pattern.get("pattern_name", "")
            if not pattern_name:
                continue

            # Skip patterns without a template
            template = LESSON_TEMPLATES.get(pattern_name)
            if not template:
                logger.debug("lesson_generator.no_template", pattern_name=pattern_name)
                continue

            # Semantic dedup: check if a lesson already exists for this pattern
            try:
                exists = await self._repo.check_lesson_exists_for_pattern(pattern_name)
                if exists:
                    logger.debug("lesson_generator.already_exists", pattern_name=pattern_name)
                    continue
            except Exception:
                logger.exception("lesson_generator.dedup_check_failed", pattern_name=pattern_name)
                continue

            # Generate candidate lesson from template
            try:
                lesson_id = await self._repo.create_lesson_learned(
                    rule=template["rule"],
                    why=template["why"],
                    severity=template["severity"],
                    confidence=0.5,
                )
                generated.append({
                    "id": lesson_id,
                    "pattern_name": pattern_name,
                    "rule": template["rule"],
                })
                logger.info(
                    "lesson_generator.lesson_created",
                    lesson_id=lesson_id,
                    pattern_name=pattern_name,
                )

                # Slack alert for new candidate
                await self._alert.send(
                    alert_type="lesson_candidate",
                    message=(
                        f"New lesson candidate from pattern '{pattern_name}' "
                        f"(freq={pattern.get('frequency', '?')}): {template['rule'][:100]}"
                    ),
                    level=AlertLevel.INFO,
                    entity_id=f"lesson_candidate_{pattern_name}",
                )
            except Exception:
                logger.exception("lesson_generator.create_failed", pattern_name=pattern_name)

        logger.info("lesson_generator.complete", generated=len(generated))
        return generated
