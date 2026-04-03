"""Tests for learning schema constants and monitoring imports.

Verifies schema constants are correctly defined and task modules are importable.
Quick coverage boost for modules that are otherwise hard to test without full
infrastructure.
"""

from __future__ import annotations


class TestLearningSchemaConstants:
    def test_schema_statements_has_expected_count(self):
        """SCHEMA_STATEMENTS has 8 statements (5 constraints + 3 indexes)."""
        from app.migrations.learning_schema import SCHEMA_STATEMENTS
        assert len(SCHEMA_STATEMENTS) == 8

    def test_vector_index_contains_768(self):
        """VECTOR_INDEX_STATEMENT specifies 768 dimensions."""
        from app.migrations.learning_schema import VECTOR_INDEX_STATEMENT
        assert "768" in VECTOR_INDEX_STATEMENT
        assert "cosine" in VECTOR_INDEX_STATEMENT

    def test_migration_key_defined(self):
        """MIGRATION_KEY is a meaningful string."""
        from app.migrations.learning_schema import MIGRATION_KEY
        assert "learning_schema" in MIGRATION_KEY


class TestScheduledTaskImports:
    """Verify scheduled task modules are importable (catches import errors)."""

    def test_conversation_quality_scan_importable(self):
        from app.tasks.scheduled import conversation_quality_scan
        assert hasattr(conversation_quality_scan, "conversation_quality_scan")

    def test_system_health_scan_importable(self):
        from app.tasks.scheduled import system_health_scan
        assert hasattr(system_health_scan, "system_health_scan")

    def test_learning_report_importable(self):
        from app.tasks.scheduled import learning_report
        assert hasattr(learning_report, "learning_report")

    def test_lesson_lifecycle_importable(self):
        from app.tasks.scheduled import lesson_lifecycle
        assert hasattr(lesson_lifecycle, "lesson_lifecycle")

    def test_conversation_scorer_importable(self):
        from app.tasks.scheduled import conversation_scorer
        assert hasattr(conversation_scorer, "conversation_scorer")


class TestLessonTemplates:
    """Verify LESSON_TEMPLATES cover all expected error types."""

    def test_templates_have_5_entries(self):
        from app.services.monitoring.lesson_generator import LESSON_TEMPLATES
        assert len(LESSON_TEMPLATES) == 5

    def test_all_templates_have_required_keys(self):
        from app.services.monitoring.lesson_generator import LESSON_TEMPLATES
        for name, template in LESSON_TEMPLATES.items():
            assert "rule" in template, f"{name} missing 'rule'"
            assert "why" in template, f"{name} missing 'why'"
            assert "severity" in template, f"{name} missing 'severity'"

    def test_template_names_match_error_types(self):
        from app.services.monitoring.lesson_generator import LESSON_TEMPLATES
        expected = {"repetition", "hallucination", "missed_intent", "tone_mismatch", "wrong_information"}
        assert set(LESSON_TEMPLATES.keys()) == expected


class TestConversationScannerConstants:
    """Verify scanner constants are defined correctly."""

    def test_repetition_threshold(self):
        from app.services.monitoring.conversation_scanner import REPETITION_THRESHOLD
        assert 0 < REPETITION_THRESHOLD < 1

    def test_min_messages_for_repetition(self):
        from app.services.monitoring.conversation_scanner import MIN_MESSAGES_FOR_REPETITION
        assert MIN_MESSAGES_FOR_REPETITION >= 2

    def test_intent_patterns_defined(self):
        from app.services.monitoring.conversation_scanner import INTENT_PATTERNS
        assert len(INTENT_PATTERNS) >= 3  # doc_request, pricing_request, visit_request

    def test_sentiment_keywords_bilingual(self):
        from app.services.monitoring.conversation_scanner import ALL_POSITIVE, ALL_NEGATIVE
        assert len(ALL_POSITIVE) >= 10  # ES + EN
        assert len(ALL_NEGATIVE) >= 10  # ES + EN
