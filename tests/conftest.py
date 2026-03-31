"""Root test configuration -- env vars, fakeredis fixtures, payload factories.

CRITICAL: The ``set_test_env`` fixture is session-scoped and autouse. It sets
all required env vars via ``os.environ.setdefault()`` BEFORE any ``app.*``
module is imported, preventing the ``Settings`` singleton from crashing on
missing environment variables.
"""

from __future__ import annotations

import os
import uuid

import fakeredis
import pytest


# ---------------------------------------------------------------------------
# Environment -- MUST run before any app import
# ---------------------------------------------------------------------------


@pytest.fixture(scope="session", autouse=True)
def set_test_env():
    """Populate all required environment variables for the Settings singleton.

    Uses ``setdefault`` so a real env var (e.g., in CI) is never overwritten.
    Every field that has NO default in ``app.config.Settings`` must appear here.
    """
    defaults = {
        "ANTHROPIC_API_KEY": "test-key-not-real",
        "NEO4J_URI": "bolt://localhost:7687",
        "NEO4J_PASSWORD": "test-password",
        "REDIS_URL": "redis://localhost:6379/0",
        "GHL_TOKEN": "test-ghl-token",
        "GHL_LOCATION_ID": "test-location-id",
        # Optional but explicit for deterministic tests
        "SLACK_WEBHOOK_URL": "",
        "CANARY_ENABLED": "false",
        "CANARY_TAG": "v3-canary",
    }
    for key, value in defaults.items():
        os.environ.setdefault(key, value)


# ---------------------------------------------------------------------------
# fakeredis -- real Redis commands, in-process
# ---------------------------------------------------------------------------


@pytest.fixture(scope="session")
def redis_server():
    """Session-scoped FakeServer shared across all tests."""
    return fakeredis.FakeServer()


@pytest.fixture()
def redis_client(redis_server):
    """Per-test FakeRedis client. Flushed after each test for isolation."""
    client = fakeredis.FakeRedis(server=redis_server, decode_responses=True)
    yield client
    client.flushall()


# ---------------------------------------------------------------------------
# Payload factories
# ---------------------------------------------------------------------------


@pytest.fixture()
def make_inbound_payload():
    """Factory fixture that builds valid inbound webhook payloads.

    Returns a callable accepting keyword overrides for any field.
    Default values mirror a typical GHL inbound WhatsApp message.
    """

    def _factory(**overrides) -> dict:
        base = {
            "messageId": str(uuid.uuid4()),
            "contactId": f"contact_{uuid.uuid4().hex[:8]}",
            "phone": "+5215512345678",
            "message": "Hola, me interesa un departamento en Polanco",
            "direction": "inbound",
            "channel": "whatsapp",
            "source": "ghl",
            "contactName": "Test Lead",
            "email": "",
            "locationId": "test-location-id",
            "conversationId": f"conv_{uuid.uuid4().hex[:8]}",
            "tags": [],
            "isAutoTrigger": False,
            "messageType": "",
            "n8n_gate_decisions": {},
        }
        base.update(overrides)
        return base

    return _factory
