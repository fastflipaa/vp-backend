"""Tests for the broker detection gate (SAFE-17).

Uses fakeredis for cache operations. Patches _tag_and_note to avoid
GHL + Neo4j calls. Mock Anthropic client for Haiku classification tests.
"""

from __future__ import annotations

import json
from unittest.mock import MagicMock, patch

from app.gates.base import GateDecision
from app.gates.broker import BROKER_CACHE_TTL, check_broker


class TestBroker:
    """Broker gate: two-tier detection (regex + Haiku) with Redis cache."""

    @patch("app.gates.broker._tag_and_note")
    def test_normal_lead_passes(self, mock_tag, redis_client, make_inbound_payload):
        """Regular lead message should PASS."""
        payload = make_inbound_payload(
            message="Hola, busco un departamento de 3 recamaras en Polanco"
        )
        result = check_broker(payload, "trace-1", redis_client)

        assert result.decision == GateDecision.PASS
        assert result.reason == "not_broker"

    @patch("app.gates.broker._tag_and_note")
    def test_regex_broker_blocked(self, mock_tag, redis_client, make_inbound_payload):
        """'Soy agente inmobiliario' regex pattern should BLOCK."""
        payload = make_inbound_payload(
            message="Soy agente inmobiliario y tengo clientes interesados",
            contactId="contact-broker-regex",
        )
        result = check_broker(payload, "trace-1", redis_client)

        assert result.decision == GateDecision.BLOCK
        assert "broker_regex_match" in result.reason
        mock_tag.assert_called_once()

    @patch("app.gates.broker._tag_and_note")
    def test_cached_broker_result(self, mock_tag, redis_client, make_inbound_payload):
        """Second call for same contact should use Redis cache (no Anthropic call)."""
        contact_id = "contact-cached"

        # Pre-populate cache with broker classification
        cache_key = f"broker_cache:{contact_id}"
        cached_data = {
            "classification": "broker",
            "confidence": 0.95,
            "method": "regex",
        }
        redis_client.set(cache_key, json.dumps(cached_data), ex=BROKER_CACHE_TTL)

        # Second call should use cache
        payload = make_inbound_payload(
            contactId=contact_id,
            message="Tengo otra propiedad para ofrecer",
        )
        mock_client = MagicMock()
        result = check_broker(payload, "trace-1", redis_client, mock_client)

        assert result.decision == GateDecision.BLOCK
        assert "cached_classification" in result.reason
        # Anthropic should NOT have been called
        mock_client.messages.create.assert_not_called()

    @patch("app.gates.broker._tag_and_note")
    def test_haiku_fallback_classifies(
        self, mock_tag, redis_client, make_inbound_payload, mock_anthropic
    ):
        """Ambiguous message with Haiku returning broker -> BLOCK."""
        # Configure mock to return broker classification with high confidence
        mock_response = MagicMock()
        mock_response.content = [
            MagicMock(
                text='{"classification": "broker", "confidence": 0.92, "reason": "industry jargon"}'
            )
        ]
        mock_anthropic.messages.create.return_value = mock_response

        payload = make_inbound_payload(
            message="Tenemos varios clientes buscando en esa zona",
            contactId="contact-haiku-broker",
        )
        result = check_broker(payload, "trace-1", redis_client, mock_anthropic)

        assert result.decision == GateDecision.BLOCK
        assert "haiku_classification" in result.reason
        mock_anthropic.messages.create.assert_called_once()

    @patch("app.gates.broker._tag_and_note")
    def test_haiku_failure_fails_open(
        self, mock_tag, redis_client, make_inbound_payload
    ):
        """Mock Anthropic raises exception -> PASS (fail-open)."""
        mock_client = MagicMock()
        mock_client.messages.create.side_effect = Exception("API timeout")

        payload = make_inbound_payload(
            message="Me interesa colaborar en proyectos inmobiliarios",
            contactId="contact-haiku-fail",
        )
        result = check_broker(payload, "trace-1", redis_client, mock_client)

        assert result.decision == GateDecision.PASS
        assert result.reason == "not_broker"

    @patch("app.gates.broker._tag_and_note")
    def test_cache_ttl(self, mock_tag, redis_client, make_inbound_payload):
        """Cached broker result should have 7-day TTL in Redis."""
        contact_id = "contact-ttl-check"
        payload = make_inbound_payload(
            message="Soy corredor inmobiliario certificado",
            contactId=contact_id,
        )
        check_broker(payload, "trace-1", redis_client)

        cache_key = f"broker_cache:{contact_id}"
        ttl = redis_client.ttl(cache_key)
        assert 0 < ttl <= BROKER_CACHE_TTL

    @patch("app.gates.broker._tag_and_note")
    def test_advertiser_regex_blocked(
        self, mock_tag, redis_client, make_inbound_payload
    ):
        """Advertiser pattern should BLOCK."""
        payload = make_inbound_payload(
            message="Somos empresa de mantenimiento y limpieza",
            contactId="contact-advertiser",
        )
        result = check_broker(payload, "trace-1", redis_client)

        assert result.decision == GateDecision.BLOCK
        assert "advertiser_regex_match" in result.reason

    def test_empty_message_passes(self, redis_client, make_inbound_payload):
        """Empty message should PASS."""
        payload = make_inbound_payload(message="")
        result = check_broker(payload, "trace-1", redis_client)

        assert result.decision == GateDecision.PASS
        assert result.reason == "empty_message"
