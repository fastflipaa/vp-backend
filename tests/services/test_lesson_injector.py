"""Tests for LessonInjector -- 3-stage learning context pipeline.

Unit tests with mocked LearningRepository and EmbeddingService.
Covers: full pipeline, dedup, cap at 3, graceful degradation, format.
"""

from __future__ import annotations

from unittest.mock import AsyncMock

import pytest

from app.services.monitoring.embedding_service import EmbeddingCircuitOpen
from app.services.monitoring.lesson_injector import LessonInjector

pytestmark = pytest.mark.asyncio


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture()
def mock_repo():
    """Mocked LearningRepository with default return values."""
    repo = AsyncMock()
    repo.find_similar_conversations.return_value = [{"conversation_id": "conv-1"}]
    repo.get_lessons_for_conversation_errors.return_value = [
        {"id": "L1", "rule": "Avoid repeating", "error_type": "repetition"},
    ]
    repo.get_lessons_for_context.return_value = [
        {"id": "L2", "rule": "Check prices", "why": "accuracy", "confidence": 0.85},
    ]
    repo.get_building_error_history.return_value = [{"type": "hallucination"}]
    return repo


@pytest.fixture()
def mock_embedding():
    """Mocked EmbeddingService."""
    emb = AsyncMock()
    emb.embed_text.return_value = [0.1] * 768
    return emb


@pytest.fixture()
def injector(mock_repo, mock_embedding):
    """LessonInjector with mocked dependencies."""
    return LessonInjector(mock_repo, mock_embedding)


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

class TestLessonInjector:
    async def test_full_pipeline_returns_learning_context(self, injector):
        """Full pipeline returns context with SIMILAR ERROR, LESSON, BUILDING CAUTION."""
        result = await injector.get_learning_context("contact-1", "bldg-1", "QUALIFYING", "Hola")

        assert "LEARNING CONTEXT (from past interactions):" in result
        assert "[SIMILAR ERROR]" in result
        assert "[LESSON]" in result
        assert "[BUILDING CAUTION]" in result

    async def test_dedup_by_lesson_id(self, injector, mock_repo):
        """Same lesson ID from Stage 1 and Stage 2 appears only once."""
        # Both stages return lesson with same ID "L1"
        mock_repo.get_lessons_for_conversation_errors.return_value = [
            {"id": "L1", "rule": "Avoid repeating", "error_type": "repetition"},
        ]
        mock_repo.get_lessons_for_context.return_value = [
            {"id": "L1", "rule": "Avoid repeating", "why": "accuracy", "confidence": 0.85},
        ]

        result = await injector.get_learning_context("contact-1", "bldg-1", "QUALIFYING", "Hola")

        # L1 should appear only once (deduped by seen_ids)
        lines = [ln for ln in result.split("\n") if ln.strip().startswith(("1.", "2.", "3.", "4."))]
        l1_count = sum(1 for ln in lines if "Avoid repeating" in ln)
        assert l1_count == 1

    async def test_cap_at_3_items(self, injector, mock_repo):
        """Output has at most 3 numbered items."""
        mock_repo.get_lessons_for_conversation_errors.return_value = [
            {"id": "A", "rule": "Rule A", "error_type": "rep"},
            {"id": "B", "rule": "Rule B", "error_type": "rep"},
        ]
        mock_repo.get_lessons_for_context.return_value = [
            {"id": "C", "rule": "Rule C", "why": "test", "confidence": 0.9},
            {"id": "D", "rule": "Rule D", "why": "test", "confidence": 0.9},
        ]
        mock_repo.get_building_error_history.return_value = [
            {"type": "hallucination"},
            {"type": "repetition"},
        ]

        result = await injector.get_learning_context("contact-1", "bldg-1", "QUALIFYING", "Hola")

        lines = result.strip().split("\n")
        numbered = [ln for ln in lines if ln.strip() and ln.strip()[0].isdigit() and "." in ln[:3]]
        assert len(numbered) == 3
        assert not any(ln.strip().startswith("4.") for ln in lines)

    async def test_graceful_degradation_embedding_failure(self, injector, mock_embedding):
        """Pipeline continues with Stage 2/3 when embedding fails."""
        mock_embedding.embed_text.side_effect = EmbeddingCircuitOpen("test")

        result = await injector.get_learning_context("contact-1", "bldg-1", "QUALIFYING", "Hola")

        # Stage 2 and 3 should still produce output
        assert result != ""
        assert "[LESSON]" in result or "[BUILDING CAUTION]" in result

    async def test_all_stages_fail_returns_empty(self, injector, mock_repo, mock_embedding):
        """Empty string returned when all 3 stages fail."""
        mock_embedding.embed_text.side_effect = Exception("fail")
        mock_repo.get_lessons_for_context.side_effect = Exception("fail")
        mock_repo.get_building_error_history.side_effect = Exception("fail")

        result = await injector.get_learning_context("contact-1", "bldg-1", "QUALIFYING", "Hola")

        assert result == ""

    async def test_empty_when_no_results(self, injector, mock_repo, mock_embedding):
        """Empty string returned when all stages return empty lists."""
        mock_repo.find_similar_conversations.return_value = []
        mock_repo.get_lessons_for_conversation_errors.return_value = []
        mock_repo.get_lessons_for_context.return_value = []
        mock_repo.get_building_error_history.return_value = []

        result = await injector.get_learning_context("contact-1", "bldg-1", "QUALIFYING", "Hola")

        assert result == ""

    async def test_no_building_cautions_without_building_id(self, injector, mock_repo):
        """get_building_error_history NOT called when building_id is None."""
        await injector.get_learning_context("contact-1", None, "QUALIFYING", "Hola")

        mock_repo.get_building_error_history.assert_not_called()

    async def test_output_format(self, injector):
        """Each item line follows format: N. [TAG] content."""
        result = await injector.get_learning_context("contact-1", "bldg-1", "QUALIFYING", "Hola")

        lines = result.strip().split("\n")
        # Skip header line
        item_lines = [ln for ln in lines[1:] if ln.strip()]
        for line in item_lines:
            # Should match: "N. [TAG] ..."
            assert line.strip()[0].isdigit(), f"Expected numbered line: {line}"
            assert "[" in line and "]" in line, f"Expected [TAG] in line: {line}"
