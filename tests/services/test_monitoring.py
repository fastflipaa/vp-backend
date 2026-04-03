"""Tests for monitoring services -- ConversationQualityScanner, SystemHealthScanner, AlertManager.

Unit tests with fakeredis and mocked dependencies.
Covers: repetition detection, ignored request, sentiment drift,
circuit breaker state, error rates, DLQ depth, alert dedup.
"""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from app.services.monitoring.alert_manager import AlertLevel, AlertManager
from app.services.monitoring.conversation_scanner import ConversationQualityScanner
from app.services.monitoring.system_health_scanner import SystemHealthScanner

pytestmark = pytest.mark.asyncio


# ---------------------------------------------------------------------------
# ConversationQualityScanner detection methods
# ---------------------------------------------------------------------------

class TestConversationQualityScannerDetection:
    async def test_repetition_detects_similar_messages(self, redis_client):
        """_check_repetition returns >0 for near-identical messages."""
        scanner = ConversationQualityScanner(
            redis_client=redis_client,
            alert_manager=AsyncMock(),
        )

        result = scanner._check_repetition([
            "Bienvenido a Polanco, tenemos los mejores departamentos disponibles para usted con precios competitivos",
            "Bienvenido a Polanco, tenemos los mejores departamentos disponibles para usted con precios competitivos hoy",
            "Bienvenido a Polanco, tenemos los mejores departamentos disponibles para usted con precios competitivos ahora",
        ])

        assert result > 0

    async def test_repetition_returns_zero_for_diverse(self, redis_client):
        """_check_repetition returns 0 for diverse messages."""
        scanner = ConversationQualityScanner(
            redis_client=redis_client,
            alert_manager=AsyncMock(),
        )

        result = scanner._check_repetition([
            "El edificio tiene 30 pisos con hermosa vista al parque Lincoln",
            "Los precios van desde $500,000 USD hasta $2,000,000 USD para las suites",
            "Le puedo agendar una visita para el jueves a las 4pm en nuestras oficinas",
        ])

        assert result == 0

    async def test_ignored_request_detects_unaddressed_doc_request(self, redis_client):
        """_check_ignored_requests detects unaddressed document request."""
        scanner = ConversationQualityScanner(
            redis_client=redis_client,
            alert_manager=AsyncMock(),
        )

        result = scanner._check_ignored_requests(
            "mandame fotos del depto por favor",
            "Que gusto saludarle, como puedo ayudarle",
        )

        assert result is not None
        assert result["name"] == "doc_request"

    async def test_ignored_request_returns_none_when_addressed(self, redis_client):
        """_check_ignored_requests returns None when request is addressed."""
        scanner = ConversationQualityScanner(
            redis_client=redis_client,
            alert_manager=AsyncMock(),
        )

        result = scanner._check_ignored_requests(
            "mandame fotos del depto",
            "Aqui le comparto el brochure https://link.com/brochure.pdf",
        )

        assert result is None

    async def test_sentiment_drift_detects_negative_shift(self, redis_client):
        """_check_sentiment_drift detects shift to negative sentiment."""
        scanner = ConversationQualityScanner(
            redis_client=redis_client,
            alert_manager=AsyncMock(),
        )

        # Messages in newest-first order (as returned by Neo4j)
        result = scanner._check_sentiment_drift([
            "horrible servicio, estoy muy molesto y frustrante",
            "es terrible y frustrante su atencion",
            "gracias por responder, todo bien",
        ])

        assert result is not None
        assert result["to"] == "NEGATIVE"
        assert result["consecutive_negative"] >= 2

    async def test_sentiment_drift_returns_none_if_not_negative(self, redis_client):
        """_check_sentiment_drift returns None when sentiment is positive."""
        scanner = ConversationQualityScanner(
            redis_client=redis_client,
            alert_manager=AsyncMock(),
        )

        result = scanner._check_sentiment_drift([
            "gracias, excelente servicio",
            "perfecto, genial atencion",
            "increible, fantastico todo",
        ])

        assert result is None


# ---------------------------------------------------------------------------
# SystemHealthScanner
# ---------------------------------------------------------------------------

class TestSystemHealthScanner:
    async def test_check_circuit_breakers(self, redis_client):
        """_check_circuit_breakers reads circuit states from Redis."""
        redis_client.set("circuit:claude", "OPEN")

        scanner = SystemHealthScanner(redis_client, AsyncMock())
        result = scanner._check_circuit_breakers()

        assert result["claude"] == "OPEN"
        assert result["ghl"] == "CLOSED"

    async def test_check_error_rates(self, redis_client):
        """_check_error_rates reads error counters."""
        redis_client.set("monitor:errors:processing_task", "7")

        scanner = SystemHealthScanner(redis_client, AsyncMock())
        result = scanner._check_error_rates()

        assert result["processing_task"] == 7

    async def test_check_dlq_depth(self, redis_client):
        """_check_dlq_depth returns correct list length."""
        redis_client.rpush("dlq:failed_messages", "msg1", "msg2")

        scanner = SystemHealthScanner(redis_client, AsyncMock())
        result = scanner._check_dlq_depth()

        assert result == 2

    async def test_check_json_failures(self, redis_client):
        """_check_json_failures reads JSON parse counter."""
        redis_client.set("monitor:errors:json_parse", "4")

        scanner = SystemHealthScanner(redis_client, AsyncMock())
        result = scanner._check_json_failures()

        assert result == 4

    async def test_check_sm_errors(self, redis_client):
        """_check_sm_errors reads state machine error counter."""
        redis_client.set("monitor:errors:sm_transition", "2")

        scanner = SystemHealthScanner(redis_client, AsyncMock())
        result = scanner._check_sm_errors()

        assert result == 2

    async def test_scan_returns_all_metrics(self, redis_client):
        """scan() returns dict with all expected metric keys."""
        redis_client.set("circuit:claude", "CLOSED")
        redis_client.set("monitor:errors:processing_task", "3")

        scanner = SystemHealthScanner(redis_client, AsyncMock())
        metrics = await scanner.scan()

        assert "circuit_breakers" in metrics
        assert "error_counts" in metrics
        assert "dlq_depth" in metrics
        assert "json_parse_failures" in metrics
        assert "state_machine_errors" in metrics

    async def test_alert_on_critical_error_rate(self, redis_client):
        """alert_on_issues sends CRITICAL alert for high error rate."""
        mock_alert = AsyncMock()
        mock_alert.send.return_value = True

        scanner = SystemHealthScanner(redis_client, mock_alert)
        metrics = {
            "circuit_breakers": {},
            "error_counts": {"processing_task": 16},
            "dlq_depth": 0,
            "json_parse_failures": 0,
            "state_machine_errors": 0,
        }

        await scanner.alert_on_issues(metrics)

        assert mock_alert.send.called
        # Check that at least one call was CRITICAL level
        critical_calls = [
            c for c in mock_alert.send.call_args_list
            if c.kwargs.get("level") == AlertLevel.CRITICAL
        ]
        assert len(critical_calls) >= 1

    async def test_increment_error_static(self, redis_client):
        """increment_error static method increments Redis counter."""
        SystemHealthScanner.increment_error(redis_client, "processing_task")

        val = redis_client.get("monitor:errors:processing_task")
        assert val == "1"

    async def test_alert_on_dlq_warning(self, redis_client):
        """alert_on_issues sends WARNING alert for DLQ depth >= 3."""
        mock_alert = AsyncMock()
        mock_alert.send.return_value = True

        scanner = SystemHealthScanner(redis_client, mock_alert)
        metrics = {
            "circuit_breakers": {},
            "error_counts": {"processing_task": 0},
            "dlq_depth": 5,
            "json_parse_failures": 0,
            "state_machine_errors": 0,
        }

        count = await scanner.alert_on_issues(metrics)

        assert count >= 1
        warning_calls = [
            c for c in mock_alert.send.call_args_list
            if c.kwargs.get("level") == AlertLevel.WARNING
        ]
        assert len(warning_calls) >= 1

    async def test_alert_on_dlq_critical(self, redis_client):
        """alert_on_issues sends CRITICAL alert for DLQ depth >= 10."""
        mock_alert = AsyncMock()
        mock_alert.send.return_value = True

        scanner = SystemHealthScanner(redis_client, mock_alert)
        metrics = {
            "circuit_breakers": {},
            "error_counts": {"processing_task": 0},
            "dlq_depth": 12,
            "json_parse_failures": 0,
            "state_machine_errors": 0,
        }

        count = await scanner.alert_on_issues(metrics)

        assert count >= 1
        critical_calls = [
            c for c in mock_alert.send.call_args_list
            if c.kwargs.get("level") == AlertLevel.CRITICAL
        ]
        assert len(critical_calls) >= 1

    async def test_alert_on_json_parse_failures(self, redis_client):
        """alert_on_issues sends alert for 3+ JSON parse failures."""
        mock_alert = AsyncMock()
        mock_alert.send.return_value = True

        scanner = SystemHealthScanner(redis_client, mock_alert)
        metrics = {
            "circuit_breakers": {},
            "error_counts": {"processing_task": 0},
            "dlq_depth": 0,
            "json_parse_failures": 5,
            "state_machine_errors": 0,
        }

        count = await scanner.alert_on_issues(metrics)

        assert count >= 1

    async def test_alert_on_sm_errors(self, redis_client):
        """alert_on_issues sends alert for 3+ state machine errors."""
        mock_alert = AsyncMock()
        mock_alert.send.return_value = True

        scanner = SystemHealthScanner(redis_client, mock_alert)
        metrics = {
            "circuit_breakers": {},
            "error_counts": {"processing_task": 0},
            "dlq_depth": 0,
            "json_parse_failures": 0,
            "state_machine_errors": 4,
        }

        count = await scanner.alert_on_issues(metrics)

        assert count >= 1

    async def test_alert_on_circuit_breaker_open(self, redis_client):
        """alert_on_issues sends CRITICAL for open circuit breaker."""
        mock_alert = AsyncMock()
        mock_alert.send.return_value = True

        scanner = SystemHealthScanner(redis_client, mock_alert)
        metrics = {
            "circuit_breakers": {"claude": "OPEN"},
            "error_counts": {"processing_task": 0},
            "dlq_depth": 0,
            "json_parse_failures": 0,
            "state_machine_errors": 0,
        }

        count = await scanner.alert_on_issues(metrics)

        assert count >= 1

    async def test_alert_on_elevated_error_rate(self, redis_client):
        """alert_on_issues sends WARNING for elevated (not critical) error rate."""
        mock_alert = AsyncMock()
        mock_alert.send.return_value = True

        scanner = SystemHealthScanner(redis_client, mock_alert)
        metrics = {
            "circuit_breakers": {},
            "error_counts": {"processing_task": 6},
            "dlq_depth": 0,
            "json_parse_failures": 0,
            "state_machine_errors": 0,
        }

        count = await scanner.alert_on_issues(metrics)

        assert count >= 1
        warning_calls = [
            c for c in mock_alert.send.call_args_list
            if c.kwargs.get("level") == AlertLevel.WARNING
        ]
        assert len(warning_calls) >= 1

    async def test_no_alerts_when_healthy(self, redis_client):
        """alert_on_issues sends no alerts when all metrics are healthy."""
        mock_alert = AsyncMock()
        mock_alert.send.return_value = True

        scanner = SystemHealthScanner(redis_client, mock_alert)
        metrics = {
            "circuit_breakers": {"claude": "CLOSED", "ghl": "CLOSED"},
            "error_counts": {"processing_task": 1},
            "dlq_depth": 0,
            "json_parse_failures": 0,
            "state_machine_errors": 0,
        }

        count = await scanner.alert_on_issues(metrics)

        assert count == 0


# ---------------------------------------------------------------------------
# AlertManager
# ---------------------------------------------------------------------------

class TestAlertManager:
    async def test_send_returns_true_first_time(self, redis_client):
        """First send returns True (SETNX succeeds)."""
        mgr = AlertManager(redis_client)

        result = await mgr.send(
            alert_type="test",
            message="hello",
            level=AlertLevel.WARNING,
        )

        assert result is True

    async def test_dedup_returns_false_second_time(self, redis_client):
        """Second send with same type+entity returns False (deduplicated)."""
        mgr = AlertManager(redis_client)

        first = await mgr.send(
            alert_type="test",
            message="hello",
            level=AlertLevel.WARNING,
            entity_id="same-entity",
        )
        second = await mgr.send(
            alert_type="test",
            message="hello again",
            level=AlertLevel.WARNING,
            entity_id="same-entity",
        )

        assert first is True
        assert second is False

    async def test_dedup_uses_entity_id(self, redis_client):
        """Different entity_ids are not deduplicated."""
        mgr = AlertManager(redis_client)

        first = await mgr.send(
            alert_type="test",
            message="hello",
            level=AlertLevel.WARNING,
            entity_id="A",
        )
        second = await mgr.send(
            alert_type="test",
            message="hello",
            level=AlertLevel.WARNING,
            entity_id="B",
        )

        assert first is True
        assert second is True

    async def test_slack_skipped_when_no_webhook(self, redis_client):
        """Slack skipped when SLACK_WEBHOOK_URL is empty, no crash."""
        mgr = AlertManager(redis_client)

        with patch("app.services.monitoring.alert_manager.httpx.post") as mock_post:
            result = await mgr.send(
                alert_type="test",
                message="hello",
                level=AlertLevel.INFO,
            )

        assert result is True
        mock_post.assert_not_called()

    async def test_ghl_note_on_warning_with_contact(self, redis_client):
        """GHL note added on WARNING level with contact_id."""
        mgr = AlertManager(redis_client)

        with patch("app.services.monitoring.alert_manager.settings") as mock_settings:
            mock_settings.SLACK_WEBHOOK_URL = ""
            with patch("app.services.monitoring.alert_manager.AlertManager._add_ghl_note", new_callable=AsyncMock) as mock_note:
                result = await mgr.send(
                    alert_type="test",
                    message="hello",
                    level=AlertLevel.WARNING,
                    contact_id="cid-1",
                )

        assert result is True
        mock_note.assert_called_once()

    async def test_no_ghl_on_info_level(self, redis_client):
        """No GHL note on INFO level even with contact_id."""
        mgr = AlertManager(redis_client)

        with patch("app.services.monitoring.alert_manager.settings") as mock_settings:
            mock_settings.SLACK_WEBHOOK_URL = ""
            with patch("app.services.monitoring.alert_manager.AlertManager._add_ghl_note", new_callable=AsyncMock) as mock_note:
                result = await mgr.send(
                    alert_type="test",
                    message="hello",
                    level=AlertLevel.INFO,
                    contact_id="cid-1",
                )

        assert result is True
        mock_note.assert_not_called()

    async def test_ghl_tag_on_critical(self, redis_client):
        """GHL tag added on CRITICAL level with contact_id."""
        mgr = AlertManager(redis_client)

        with patch("app.services.monitoring.alert_manager.settings") as mock_settings:
            mock_settings.SLACK_WEBHOOK_URL = ""
            with patch("app.services.monitoring.alert_manager.AlertManager._add_ghl_tag", new_callable=AsyncMock) as mock_tag, \
                 patch("app.services.monitoring.alert_manager.AlertManager._add_ghl_note", new_callable=AsyncMock):
                result = await mgr.send(
                    alert_type="test",
                    message="hello",
                    level=AlertLevel.CRITICAL,
                    contact_id="cid-1",
                )

        assert result is True
        mock_tag.assert_called_once_with("cid-1", "needs-human-review")
