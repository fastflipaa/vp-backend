"""Lesson injection pipeline -- retrieves relevant lessons from Neo4j for AI prompts.

3-stage pipeline:
1. Vector similarity: find similar past conversations and their linked lessons
2. Context-matched: lookup lessons by building/state with confidence threshold
3. Building error history: caution warnings from recent errors

Each stage is independently try/excepted. A failure in any stage does NOT
prevent the others from running. Total failure returns empty string.

Max 3 items total, deduped by lesson ID. Only approved/evergreen lessons.
"""

from __future__ import annotations

from typing import Any

import structlog

from app.repositories.learning_repository import LearningRepository
from app.services.monitoring.embedding_service import (
    EmbeddingCircuitOpen,
    EmbeddingService,
)

logger = structlog.get_logger()


class LessonInjector:
    """Retrieves and formats learning context for AI prompt injection."""

    def __init__(
        self,
        learning_repo: LearningRepository,
        embedding_service: EmbeddingService,
    ) -> None:
        self._learning_repo = learning_repo
        self._embedding_service = embedding_service

    async def get_learning_context(
        self,
        contact_id: str,
        building_id: str | None,
        state: str,
        current_message: str,
    ) -> str:
        """Build learning context string for AI prompt injection.

        Returns a multi-line string block to append to the system prompt,
        or empty string if nothing relevant found. Never raises.
        """
        seen_ids: set[str] = set()
        items: list[str] = []

        # -- Stage 1: Vector similarity --
        vector_matches = 0
        try:
            vector = await self._embedding_service.embed_text(current_message)
            similar = await self._learning_repo.find_similar_conversations(
                vector, threshold=0.7, limit=3
            )
            conv_ids = [s["conversation_id"] for s in similar]
            if conv_ids:
                lessons = await self._learning_repo.get_lessons_for_conversation_errors(
                    conv_ids
                )
                for lesson in lessons:
                    lid = lesson["id"]
                    if lid in seen_ids:
                        continue
                    seen_ids.add(lid)
                    rule = str(lesson.get("rule", ""))[:180]
                    error_type = lesson.get("error_type", "unknown")
                    items.append(
                        f"[SIMILAR ERROR] Past error: {error_type}. Lesson: {rule}"
                    )
                    vector_matches += 1
        except (EmbeddingCircuitOpen, Exception):
            logger.warning("lesson_injector.vector_stage_failed")

        # -- Stage 2: Context-matched lesson lookup --
        context_matches = 0
        try:
            context_lessons = await self._learning_repo.get_lessons_for_context(
                building_id=building_id,
                state=state,
                min_confidence=0.7,
                limit=3,
            )
            for lesson in context_lessons:
                lid = lesson["id"]
                if lid in seen_ids:
                    continue
                seen_ids.add(lid)
                rule = str(lesson.get("rule", ""))[:180]
                why = str(lesson.get("why", ""))[:80]
                confidence = lesson.get("confidence", 0.0)
                items.append(
                    f"[LESSON] {rule} -- Why: {why} (confidence: {confidence:.0%})"
                )
                context_matches += 1
        except Exception:
            logger.warning("lesson_injector.context_stage_failed")

        # -- Stage 3: Building error history (caution warnings) --
        cautions = 0
        caution_items: list[str] = []
        if building_id:
            try:
                errors = await self._learning_repo.get_building_error_history(
                    building_id, days=30
                )
                # Group by error type and count
                type_counts: dict[str, int] = {}
                for err in errors:
                    etype = err.get("type", "unknown")
                    type_counts[etype] = type_counts.get(etype, 0) + 1

                for etype, count in type_counts.items():
                    caution_items.append(
                        f"[BUILDING CAUTION] Building {building_id}: "
                        f"{count} {etype} error(s) in last 30 days"
                    )
                    cautions += 1
            except Exception:
                logger.warning("lesson_injector.building_stage_failed")

        # -- Dedup + cap at 3 total --
        # Lessons from Stage 1+2 first, then Stage 3 cautions
        all_items = items + caution_items
        capped = all_items[:3]

        if not capped:
            logger.info(
                "lesson_injector.context_built",
                lesson_count=0,
                vector_matches=vector_matches,
                context_matches=context_matches,
                cautions=cautions,
            )
            return ""

        logger.info(
            "lesson_injector.context_built",
            lesson_count=len(capped),
            vector_matches=vector_matches,
            context_matches=context_matches,
            cautions=cautions,
        )

        lines = ["LEARNING CONTEXT (from past interactions):"]
        for i, item in enumerate(capped, 1):
            lines.append(f"{i}. {item}")

        return "\n".join(lines)
