"""Tests for LearningEffectivenessTracker -- before/after outcome metrics.

Unit tests with mocked LearningRepository.
Covers: metrics computation, weighted subtraction, report format, arrows.
"""

from __future__ import annotations

from unittest.mock import AsyncMock

import pytest

from app.services.monitoring.learning_tracker import LearningEffectivenessTracker

pytestmark = pytest.mark.asyncio


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture()
def mock_repo():
    """Mocked LearningRepository."""
    return AsyncMock()


@pytest.fixture()
def tracker(mock_repo):
    """LearningEffectivenessTracker with mocked repo."""
    return LearningEffectivenessTracker(mock_repo)


# ---------------------------------------------------------------------------
# Compute metrics tests
# ---------------------------------------------------------------------------

class TestComputeMetrics:
    async def test_compute_metrics_with_data(self, tracker, mock_repo):
        """Metrics computed correctly with weighted subtraction."""
        mock_repo.get_outcome_averages.side_effect = [
            # First call: days=7 (current)
            {"repetition": 0.3, "sentiment": 0.7, "intent_alignment": 0.9, "count": 10},
            # Second call: days=14 (all)
            {"repetition": 0.35, "sentiment": 0.65, "intent_alignment": 0.85, "count": 20},
        ]
        mock_repo.get_lesson_injection_count.return_value = 5

        result = await tracker.compute_metrics()

        assert result["current_7d"]["count"] == 10
        assert result["injection_count_7d"] == 5
        # prior_count = 20 - 10 = 10
        assert result["previous_7d"]["count"] == 10
        # prior repetition = (0.35*20 - 0.3*10) / 10 = (7 - 3) / 10 = 0.4
        assert abs(result["previous_7d"]["repetition"] - 0.4) < 0.01
        # Deltas
        assert "deltas" in result

    async def test_compute_metrics_empty_data(self, tracker, mock_repo):
        """No division by zero with zero counts."""
        mock_repo.get_outcome_averages.return_value = {
            "repetition": 0, "sentiment": 0, "intent_alignment": 0, "count": 0,
        }
        mock_repo.get_lesson_injection_count.return_value = 0

        result = await tracker.compute_metrics()

        assert result["previous_7d"]["repetition"] == 0.0
        assert result["previous_7d"]["sentiment"] == 0.0
        assert result["previous_7d"]["intent_alignment"] == 0.0

    async def test_compute_metrics_current_period_failure(self, tracker, mock_repo):
        """Returns empty dict when current period query fails."""
        mock_repo.get_outcome_averages.side_effect = Exception("DB down")

        result = await tracker.compute_metrics()

        assert result == {}

    async def test_injection_count_failure_returns_zero(self, tracker, mock_repo):
        """injection_count_7d is 0 when get_lesson_injection_count fails."""
        mock_repo.get_outcome_averages.side_effect = [
            {"repetition": 0.3, "sentiment": 0.7, "intent_alignment": 0.9, "count": 10},
            {"repetition": 0.35, "sentiment": 0.65, "intent_alignment": 0.85, "count": 20},
        ]
        mock_repo.get_lesson_injection_count.side_effect = Exception("fail")

        result = await tracker.compute_metrics()

        assert result["injection_count_7d"] == 0


# ---------------------------------------------------------------------------
# Weekly report tests
# ---------------------------------------------------------------------------

class TestWeeklyReport:
    async def test_report_contains_expected_sections(self, tracker, mock_repo):
        """Report contains expected section headers and metrics."""
        mock_repo.get_outcome_averages.side_effect = [
            {"repetition": 0.3, "sentiment": 0.7, "intent_alignment": 0.9, "count": 10},
            {"repetition": 0.35, "sentiment": 0.65, "intent_alignment": 0.85, "count": 20},
        ]
        mock_repo.get_lesson_injection_count.return_value = 5

        report = await tracker.generate_weekly_report()

        assert "Weekly Learning Effectiveness Report" in report
        assert "Conversations scored" in report
        assert "Lesson injections" in report
        assert "Repetition:" in report
        assert "Sentiment:" in report
        assert "Intent Alignment:" in report

    async def test_report_shows_improvement_arrows(self, tracker, mock_repo):
        """Report shows (improved) when metrics improve."""
        # Set up so deltas show improvement:
        # repetition: current 0.2 vs prior 0.4 -> delta = -0.2 (improved)
        # sentiment: current 0.8 vs prior 0.6 -> delta = +0.2 (improved)
        mock_repo.get_outcome_averages.side_effect = [
            {"repetition": 0.2, "sentiment": 0.8, "intent_alignment": 0.9, "count": 10},
            {"repetition": 0.3, "sentiment": 0.7, "intent_alignment": 0.85, "count": 20},
        ]
        mock_repo.get_lesson_injection_count.return_value = 3

        report = await tracker.generate_weekly_report()

        assert "(improved)" in report

    async def test_report_shows_worsened(self, tracker, mock_repo):
        """Report shows (worsened) when metrics decline."""
        # repetition: current 0.5 vs prior 0.2 -> delta = +0.3 (worsened)
        # sentiment: current 0.4 vs prior 0.8 -> delta = -0.4 (worsened)
        mock_repo.get_outcome_averages.side_effect = [
            {"repetition": 0.5, "sentiment": 0.4, "intent_alignment": 0.5, "count": 10},
            {"repetition": 0.35, "sentiment": 0.6, "intent_alignment": 0.7, "count": 20},
        ]
        mock_repo.get_lesson_injection_count.return_value = 1

        report = await tracker.generate_weekly_report()

        assert "(worsened)" in report

    async def test_insufficient_data_report(self, tracker, mock_repo):
        """Report returns 'insufficient data' when compute_metrics is empty."""
        mock_repo.get_outcome_averages.side_effect = Exception("DB down")

        report = await tracker.generate_weekly_report()

        assert "insufficient data" in report
