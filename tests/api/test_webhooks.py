"""Tests for webhook API endpoints using FastAPI TestClient.

Mocks Celery tasks (process_gates_shadow.delay) to prevent actual execution.
Includes trace_id propagation test (ROADMAP SC-1) verifying UUID in structured logs.

NOTE: Webhook handlers use lazy imports inside function bodies. Mocks must
target the source modules (app.tasks.gate_tasks, app.services.canary_router,
app.gates.human_lock, app.dependencies) so the lazy ``from X import Y``
picks up the mock at import time.
"""

from __future__ import annotations

import logging
import re
import uuid
from unittest.mock import MagicMock, patch

import pytest
from fastapi.testclient import TestClient

from app.main import app


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture()
def valid_payload():
    """A valid inbound webhook payload."""
    return {
        "messageId": str(uuid.uuid4()),
        "contactId": f"contact_{uuid.uuid4().hex[:8]}",
        "phone": "+5215512345678",
        "message": "Hola, me interesa un departamento",
        "direction": "inbound",
        "channel": "whatsapp",
        "source": "ghl",
        "tags": [],
    }


@pytest.fixture()
def mock_gate_task():
    """Mock the Celery gate task at its source module."""
    mock_task = MagicMock()
    mock_task.delay = MagicMock()
    with patch("app.tasks.gate_tasks.process_gates_shadow", mock_task):
        yield mock_task


@pytest.fixture()
def mock_canary():
    """Mock canary routing to always return False (shadow mode)."""
    with patch("app.services.canary_router.should_route_canary", return_value=False) as m:
        yield m


@pytest.fixture()
def inbound_client(mock_gate_task, mock_canary):
    """TestClient with gate task and canary mocked for inbound tests."""
    client = TestClient(app)
    client._mock_task = mock_gate_task
    client._mock_canary = mock_canary
    return client


# ---------------------------------------------------------------------------
# Inbound webhook tests
# ---------------------------------------------------------------------------

class TestInboundWebhook:
    def test_inbound_webhook_returns_200(self, inbound_client, valid_payload):
        """POST valid payload to /webhooks/inbound returns 200."""
        response = inbound_client.post("/webhooks/inbound", json=valid_payload)

        assert response.status_code == 200
        body = response.json()
        assert body["status"] == "accepted"
        assert "trace_id" in body

    def test_inbound_webhook_enqueues_task(self, inbound_client, valid_payload):
        """POST valid payload enqueues Celery gate task."""
        response = inbound_client.post("/webhooks/inbound", json=valid_payload)

        assert response.status_code == 200
        # Verify Celery task was called
        inbound_client._mock_task.delay.assert_called_once()
        call_args = inbound_client._mock_task.delay.call_args
        assert call_args[0][0] == valid_payload  # First arg is payload
        # Second arg is trace_id (UUID string)
        trace_id = call_args[0][1]
        assert re.match(r"^[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}$", trace_id)

    def test_inbound_webhook_returns_trace_id(self, inbound_client, valid_payload):
        """Response contains a valid UUID trace_id."""
        response = inbound_client.post("/webhooks/inbound", json=valid_payload)

        body = response.json()
        trace_id = body["trace_id"]
        assert re.match(
            r"^[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}$",
            trace_id,
        )

    def test_invalid_json_returns_400(self, inbound_client):
        """POST with invalid JSON body returns 400."""
        response = inbound_client.post(
            "/webhooks/inbound",
            content=b"not valid json{{{",
            headers={"Content-Type": "application/json"},
        )

        assert response.status_code == 400
        body = response.json()
        assert body["status"] == "error"

    def test_duplicate_returns_200(self, inbound_client, valid_payload):
        """POST same messageId twice both return 200 (dedup happens in gate, not endpoint)."""
        r1 = inbound_client.post("/webhooks/inbound", json=valid_payload)
        r2 = inbound_client.post("/webhooks/inbound", json=valid_payload)

        assert r1.status_code == 200
        assert r2.status_code == 200
        # Different trace_ids for each request
        assert r1.json()["trace_id"] != r2.json()["trace_id"]

    def test_trace_id_in_logs(self, inbound_client, valid_payload, caplog):
        """ROADMAP SC-1: POST inbound payload produces log entry with trace_id UUID.

        Verifies end-to-end trace_id propagation through structured logging.
        Uses caplog to capture log output and checks for UUID-format trace_id.
        """
        with caplog.at_level(logging.INFO):
            response = inbound_client.post("/webhooks/inbound", json=valid_payload)

        assert response.status_code == 200
        returned_trace_id = response.json()["trace_id"]

        # Verify the trace_id is a valid UUID format
        assert re.match(
            r"^[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}$",
            returned_trace_id,
        ), f"trace_id is not valid UUID: {returned_trace_id}"

        # The webhook handler logs "shadow_webhook_received" with trace_id.
        # structlog routes through stdlib logging; caplog captures the record.
        # Check caplog records for the trace_id in the message or record attrs.
        found_trace = False
        for record in caplog.records:
            record_text = record.getMessage()
            # structlog may embed trace_id in the rendered message text
            if returned_trace_id in str(record_text):
                found_trace = True
                break
            # structlog may store the event as a dict in record.msg
            if hasattr(record, "msg") and isinstance(record.msg, dict):
                if record.msg.get("trace_id") == returned_trace_id:
                    found_trace = True
                    break

        assert returned_trace_id is not None, "No trace_id returned in response"
        assert found_trace, f"trace_id {returned_trace_id} not found in log output (ROADMAP SC-1)"


# ---------------------------------------------------------------------------
# Outbound webhook tests
# ---------------------------------------------------------------------------

class TestOutboundWebhook:
    def test_outbound_webhook_human_lock(self):
        """POST outbound with human source sets human lock."""
        mock_lock = MagicMock()
        mock_redis = MagicMock()

        with patch("app.gates.human_lock.set_human_lock", mock_lock), \
             patch("app.dependencies.get_sync_redis", return_value=mock_redis):
            client = TestClient(app)

            payload = {
                "contactId": "contact_abc123",
                "source": "human_agent",
                "agentName": "Fernando",
            }
            response = client.post("/webhooks/outbound", json=payload)

            assert response.status_code == 200
            mock_lock.assert_called_once_with(
                "contact_abc123", "Fernando", mock_redis
            )

    def test_outbound_webhook_ignores_bot_source(self):
        """POST outbound with bot/system source does NOT set human lock."""
        mock_lock = MagicMock()
        mock_redis = MagicMock()

        with patch("app.gates.human_lock.set_human_lock", mock_lock), \
             patch("app.dependencies.get_sync_redis", return_value=mock_redis):
            client = TestClient(app)

            payload = {
                "contactId": "contact_abc123",
                "source": "bot",
            }
            response = client.post("/webhooks/outbound", json=payload)

            assert response.status_code == 200
            mock_lock.assert_not_called()

    def test_outbound_invalid_json_returns_400(self):
        """POST with invalid JSON to outbound returns 400."""
        client = TestClient(app)

        response = client.post(
            "/webhooks/outbound",
            content=b"{{invalid",
            headers={"Content-Type": "application/json"},
        )

        assert response.status_code == 400
