"""Tests for AutoLessonGenerator -- template-based lesson generation.

Unit tests with mocked LearningRepository and AlertManager.
Covers: 5 templates, semantic dedup, Slack alerts, error handling.
"""

from __future__ import annotations

from unittest.mock import AsyncMock

import pytest

from app.services.monitoring.alert_manager import AlertLevel
from app.services.monitoring.lesson_generator import AutoLessonGenerator, LESSON_TEMPLATES

pytestmark = pytest.mark.asyncio


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture()
def mock_repo():
    """Mocked LearningRepository."""
    repo = AsyncMock()
    repo.get_error_patterns.return_value = [
        {"pattern_name": "repetition", "frequency": 5},
        {"pattern_name": "hallucination", "frequency": 3},
    ]
    repo.check_lesson_exists_for_pattern.return_value = False
    repo.create_lesson_learned.return_value = "new-lesson-id"
    return repo


@pytest.fixture()
def mock_alert():
    """Mocked AlertManager."""
    alert = AsyncMock()
    alert.send.return_value = True
    return alert


@pytest.fixture()
def generator(mock_repo, mock_alert):
    """AutoLessonGenerator with mocked dependencies."""
    return AutoLessonGenerator(mock_repo, mock_alert)


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

class TestAutoLessonGenerator:
    async def test_generates_lessons_for_matching_patterns(self, generator, mock_repo):
        """Generates lessons for patterns with templates (repetition + hallucination)."""
        result = await generator.analyze_and_generate()

        assert len(result) == 2
        assert mock_repo.create_lesson_learned.call_count == 2

    async def test_skips_patterns_without_template(self, generator, mock_repo):
        """Patterns without a template in LESSON_TEMPLATES are skipped."""
        mock_repo.get_error_patterns.return_value = [
            {"pattern_name": "repetition", "frequency": 5},
            {"pattern_name": "hallucination", "frequency": 3},
            {"pattern_name": "unknown_type", "frequency": 5},
        ]

        result = await generator.analyze_and_generate()

        # Only repetition and hallucination have templates
        assert len(result) == 2

    async def test_semantic_dedup_skips_existing(self, generator, mock_repo):
        """Skips patterns that already have lessons (semantic dedup)."""
        # Return True for "repetition" (already exists), False for "hallucination"
        async def dedup_side_effect(pattern_name):
            return pattern_name == "repetition"

        mock_repo.check_lesson_exists_for_pattern.side_effect = dedup_side_effect

        result = await generator.analyze_and_generate()

        assert len(result) == 1
        assert result[0]["pattern_name"] == "hallucination"

    async def test_uses_correct_template_content(self, generator, mock_repo):
        """create_lesson_learned called with exact template rule text."""
        await generator.analyze_and_generate()

        # Check that at least one call used the repetition template
        calls = mock_repo.create_lesson_learned.call_args_list
        rules = [c.kwargs.get("rule", c.args[0] if c.args else "") for c in calls]
        assert LESSON_TEMPLATES["repetition"]["rule"] in rules

    async def test_sends_slack_alert_per_candidate(self, generator, mock_alert):
        """Slack alert sent for each generated lesson."""
        await generator.analyze_and_generate()

        assert mock_alert.send.call_count == 2
        for call in mock_alert.send.call_args_list:
            assert call.kwargs.get("alert_type") == "lesson_candidate"
            assert call.kwargs.get("level") == AlertLevel.INFO

    async def test_returns_empty_when_no_patterns(self, generator, mock_repo):
        """Returns empty list when no error patterns exist."""
        mock_repo.get_error_patterns.return_value = []

        result = await generator.analyze_and_generate()

        assert result == []

    async def test_handles_repo_failure_gracefully(self, generator, mock_repo):
        """Returns empty list when get_error_patterns raises."""
        mock_repo.get_error_patterns.side_effect = Exception("DB down")

        result = await generator.analyze_and_generate()

        assert result == []

    async def test_handles_create_failure_continues(self, generator, mock_repo):
        """Continues to next pattern when create_lesson_learned fails."""
        call_count = 0

        async def create_side_effect(**kwargs):
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                raise Exception("Write failed")
            return "lesson-id"

        mock_repo.create_lesson_learned.side_effect = create_side_effect

        result = await generator.analyze_and_generate()

        # First pattern fails, second succeeds
        assert len(result) == 1
