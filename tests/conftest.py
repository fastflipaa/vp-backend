"""Root test configuration -- env vars, fakeredis fixtures, payload factories.

CRITICAL: Environment variables are set in ``pytest_configure()`` (a pytest
hook that runs BEFORE test collection). This prevents the ``Settings``
singleton from crashing when test modules import ``app.config.settings``
at collection time.
"""

from __future__ import annotations

import os
import uuid

import fakeredis
import pytest


# ---------------------------------------------------------------------------
# Environment -- MUST run BEFORE collection (not inside a fixture)
# ---------------------------------------------------------------------------

def pytest_configure(config):
    """Set all required env vars before any app module is imported.

    ``pytest_configure`` runs before test collection, which is when
    ``from app.gates.injection import check_injection`` triggers
    ``from app.config import settings`` -> ``Settings()`` instantiation.
    Uses ``setdefault`` so real env vars (e.g., in CI) are never overwritten.
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
