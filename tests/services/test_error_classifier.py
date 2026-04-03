"""Tests for ErrorClassifier -- deterministic rule-based error classification.

Unit tests with mocked LearningRepository and AlertManager.
Covers 4 detection checks: repetition, missed_intent, tone_mismatch,
hallucination, plus auto-lesson generation and error persistence.
"""

from __future__ import annotations

from unittest.mock import AsyncMock

import pytest

from app.services.monitoring.error_classifier import ErrorClassifier

pytestmark = pytest.mark.asyncio


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture()
def mock_repo():
    """Mocked LearningRepository with default return values."""
    repo = AsyncMock()
    repo.create_agent_error.return_value = "err-id"
    repo.increment_error_pattern.return_value = 1
    repo.create_lesson_learned.return_value = "lesson-id"
    return repo


@pytest.fixture()
def mock_alert():
    """Mocked AlertManager."""
    alert = AsyncMock()
    alert.send.return_value = True
    return alert


@pytest.fixture()
def classifier(mock_repo, mock_alert):
    """ErrorClassifier with mocked dependencies."""
    return ErrorClassifier(mock_repo, mock_alert)


def _lead_data(contact_id="cid-1", phone="+5215512345678"):
    return {"contact_id": contact_id, "phone": phone}


# ---------------------------------------------------------------------------
# Repetition check
# ---------------------------------------------------------------------------

class TestRepetitionCheck:
    async def test_detects_similar_responses(self, classifier, mock_repo):
        """Repetition detected with 3 near-identical assistant messages."""
        messages = [
            {"role": "assistant", "content": "Bienvenido a Polanco, tenemos los mejores departamentos disponibles para usted con precios competitivos", "created_at": "2026-01-01"},
            {"role": "assistant", "content": "Bienvenido a Polanco, tenemos los mejores departamentos disponibles para usted con precios competitivos hoy", "created_at": "2026-01-01"},
            {"role": "assistant", "content": "Bienvenido a Polanco, tenemos los mejores departamentos disponibles para usted con precios competitivos ahora", "created_at": "2026-01-01"},
        ]

        result = await classifier.classify(_lead_data(), messages, buildings=[])

        error_types = [e["type"] for e in result]
        assert "repetition" in error_types

    async def test_no_repetition_with_diverse_responses(self, classifier, mock_repo):
        """No repetition with diverse assistant messages."""
        messages = [
            {"role": "assistant", "content": "El edificio cuenta con 30 pisos y vista al parque Lincoln", "created_at": "2026-01-01"},
            {"role": "assistant", "content": "Los precios van desde $500,000 USD hasta $2,000,000 USD", "created_at": "2026-01-01"},
            {"role": "assistant", "content": "Le puedo agendar una visita para el jueves a las 4pm", "created_at": "2026-01-01"},
        ]

        result = await classifier.classify(_lead_data(), messages, buildings=[])

        error_types = [e["type"] for e in result]
        assert "repetition" not in error_types

    async def test_ignores_when_fewer_than_min_messages(self, classifier, mock_repo):
        """No repetition check when fewer than MIN_MESSAGES_FOR_REPETITION."""
        messages = [
            {"role": "assistant", "content": "Bienvenido a Polanco, tenemos departamentos increibles", "created_at": "2026-01-01"},
            {"role": "assistant", "content": "Bienvenido a Polanco, tenemos departamentos increibles", "created_at": "2026-01-01"},
        ]

        result = await classifier.classify(_lead_data(), messages, buildings=[])

        error_types = [e["type"] for e in result]
        assert "repetition" not in error_types


# ---------------------------------------------------------------------------
# Ignored request check
# ---------------------------------------------------------------------------

class TestIgnoredRequestCheck:
    async def test_detects_ignored_doc_request(self, classifier, mock_repo):
        """missed_intent detected when AI ignores document request."""
        messages = [
            {"role": "user", "content": "mandame fotos del departamento", "created_at": "2026-01-01"},
            {"role": "assistant", "content": "Es un placer saludarle, como puedo ayudarle", "created_at": "2026-01-01"},
        ]

        result = await classifier.classify(_lead_data(), messages, buildings=[])

        error_types = [e["type"] for e in result]
        assert "missed_intent" in error_types

    async def test_no_error_when_request_addressed(self, classifier, mock_repo):
        """No missed_intent when AI provides the requested document link."""
        messages = [
            {"role": "user", "content": "mandame fotos del departamento", "created_at": "2026-01-01"},
            {"role": "assistant", "content": "Aqui le envio el brochure: https://example.com/brochure.pdf", "created_at": "2026-01-01"},
        ]

        result = await classifier.classify(_lead_data(), messages, buildings=[])

        error_types = [e["type"] for e in result]
        assert "missed_intent" not in error_types

    async def test_detects_ignored_pricing_request(self, classifier, mock_repo):
        """missed_intent detected when AI ignores pricing question."""
        messages = [
            {"role": "user", "content": "cuanto cuesta el departamento?", "created_at": "2026-01-01"},
            {"role": "assistant", "content": "Es un hermoso edificio con excelentes amenidades", "created_at": "2026-01-01"},
        ]

        result = await classifier.classify(_lead_data(), messages, buildings=[])

        error_types = [e["type"] for e in result]
        assert "missed_intent" in error_types


# ---------------------------------------------------------------------------
# Tone mismatch check
# ---------------------------------------------------------------------------

class TestToneMismatch:
    async def test_detects_positive_response_to_negative_user(self, classifier, mock_repo):
        """tone_mismatch detected when AI is positive while user is negative."""
        messages = [
            {"role": "user", "content": "Es frustrante y horrible su servicio, estoy muy molesto", "created_at": "2026-01-01"},
            {"role": "assistant", "content": "Excelente! Todo es perfecto y genial aqui", "created_at": "2026-01-01"},
        ]

        result = await classifier.classify(_lead_data(), messages, buildings=[])

        error_types = [e["type"] for e in result]
        assert "tone_mismatch" in error_types

    async def test_no_mismatch_when_both_neutral(self, classifier, mock_repo):
        """No tone_mismatch when both messages are neutral."""
        messages = [
            {"role": "user", "content": "Me gustaria informacion del departamento 401", "created_at": "2026-01-01"},
            {"role": "assistant", "content": "El departamento 401 tiene 120 metros cuadrados", "created_at": "2026-01-01"},
        ]

        result = await classifier.classify(_lead_data(), messages, buildings=[])

        error_types = [e["type"] for e in result]
        assert "tone_mismatch" not in error_types

    async def test_no_mismatch_when_ai_matches_tone(self, classifier, mock_repo):
        """No tone_mismatch when AI also uses negative/sympathetic tone."""
        messages = [
            {"role": "user", "content": "Es frustrante, horrible servicio", "created_at": "2026-01-01"},
            {"role": "assistant", "content": "Lamento mucho la frustrante experiencia, es terrible", "created_at": "2026-01-01"},
        ]

        result = await classifier.classify(_lead_data(), messages, buildings=[])

        error_types = [e["type"] for e in result]
        assert "tone_mismatch" not in error_types


# ---------------------------------------------------------------------------
# Hallucination check
# ---------------------------------------------------------------------------

class TestHallucination:
    async def test_detects_price_below_range(self, classifier, mock_repo):
        """hallucination detected when AI claims price below building range."""
        messages = [
            {"role": "assistant", "content": "El precio es de $100,000 USD", "created_at": "2026-01-01"},
        ]
        buildings = [{"price_min_usd": 500000, "price_max_usd": 1000000}]

        result = await classifier.classify(_lead_data(), messages, buildings)

        error_types = [e["type"] for e in result]
        assert "hallucination" in error_types

    async def test_detects_price_above_range(self, classifier, mock_repo):
        """hallucination detected when AI claims price above building range."""
        messages = [
            {"role": "assistant", "content": "El precio es de $5,000,000 USD", "created_at": "2026-01-01"},
        ]
        buildings = [{"price_min_usd": 500000, "price_max_usd": 1000000}]

        result = await classifier.classify(_lead_data(), messages, buildings)

        error_types = [e["type"] for e in result]
        assert "hallucination" in error_types

    async def test_no_hallucination_within_tolerance(self, classifier, mock_repo):
        """No hallucination when price is within 20% tolerance."""
        messages = [
            {"role": "assistant", "content": "El precio es de $550,000 USD", "created_at": "2026-01-01"},
        ]
        buildings = [{"price_min_usd": 500000, "price_max_usd": 1000000}]

        result = await classifier.classify(_lead_data(), messages, buildings)

        error_types = [e["type"] for e in result]
        assert "hallucination" not in error_types

    async def test_detects_floor_count_mismatch(self, classifier, mock_repo):
        """hallucination detected when AI claims wrong floor count."""
        messages = [
            {"role": "assistant", "content": "El edificio tiene 50 pisos con vista panoramica", "created_at": "2026-01-01"},
        ]
        buildings = [{"total_floors": 25}]

        result = await classifier.classify(_lead_data(), messages, buildings)

        error_types = [e["type"] for e in result]
        assert "hallucination" in error_types

    async def test_no_hallucination_without_buildings(self, classifier, mock_repo):
        """No hallucination check when buildings list is empty."""
        messages = [
            {"role": "assistant", "content": "El precio es de $100,000 USD", "created_at": "2026-01-01"},
        ]

        result = await classifier.classify(_lead_data(), messages, buildings=[])

        error_types = [e["type"] for e in result]
        assert "hallucination" not in error_types


# ---------------------------------------------------------------------------
# Auto-lesson generation
# ---------------------------------------------------------------------------

class TestAutoLessonGeneration:
    async def test_auto_generates_lesson_at_frequency_3(self, classifier, mock_repo, mock_alert):
        """Auto-generates lesson when error frequency reaches 3."""
        mock_repo.increment_error_pattern.return_value = 3

        messages = [
            {"role": "assistant", "content": "Bienvenido a Polanco, tenemos los mejores departamentos disponibles para usted con precios competitivos", "created_at": "2026-01-01"},
            {"role": "assistant", "content": "Bienvenido a Polanco, tenemos los mejores departamentos disponibles para usted con precios competitivos hoy", "created_at": "2026-01-01"},
            {"role": "assistant", "content": "Bienvenido a Polanco, tenemos los mejores departamentos disponibles para usted con precios competitivos ahora", "created_at": "2026-01-01"},
        ]

        await classifier.classify(_lead_data(), messages, buildings=[])

        assert mock_repo.create_lesson_learned.called
        assert mock_alert.send.called

    async def test_no_lesson_below_frequency_3(self, classifier, mock_repo, mock_alert):
        """No lesson generated when frequency is below 3."""
        mock_repo.increment_error_pattern.return_value = 2

        messages = [
            {"role": "assistant", "content": "Bienvenido a Polanco, tenemos los mejores departamentos disponibles para usted con precios competitivos", "created_at": "2026-01-01"},
            {"role": "assistant", "content": "Bienvenido a Polanco, tenemos los mejores departamentos disponibles para usted con precios competitivos hoy", "created_at": "2026-01-01"},
            {"role": "assistant", "content": "Bienvenido a Polanco, tenemos los mejores departamentos disponibles para usted con precios competitivos ahora", "created_at": "2026-01-01"},
        ]

        await classifier.classify(_lead_data(), messages, buildings=[])

        assert not mock_repo.create_lesson_learned.called

    async def test_persist_errors_to_repo(self, classifier, mock_repo):
        """Errors are persisted to the repository."""
        messages = [
            {"role": "user", "content": "mandame fotos del departamento", "created_at": "2026-01-01"},
            {"role": "assistant", "content": "Es un placer saludarle", "created_at": "2026-01-01"},
        ]

        await classifier.classify(_lead_data(), messages, buildings=[])

        assert mock_repo.create_agent_error.called
        call_args = mock_repo.create_agent_error.call_args
        assert call_args.kwargs.get("error_type") == "missed_intent" or \
               (call_args.args and call_args.args[1] == "missed_intent") or \
               (len(call_args.kwargs) > 0 and "missed_intent" in str(call_args))
