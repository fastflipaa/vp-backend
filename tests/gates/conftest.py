"""Gate-specific fixtures reusing root redis_client and payload factory."""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock

import pytest


@pytest.fixture()
def base_payload(make_inbound_payload):
    """Standard inbound payload with predictable test values."""
    return make_inbound_payload(
        messageId="msg-test-001",
        contactId="contact-test-001",
        phone="+5215512345678",
        message="Hola, me interesa un departamento en Polanco",
        direction="inbound",
    )


@pytest.fixture()
def mock_anthropic():
    """Mock Anthropic client for broker gate Haiku classification.

    Returns a MagicMock whose ``messages.create()`` returns a configurable
    classification JSON response.  The default response classifies the
    message as ``buyer_lead`` with high confidence (PASS).
    """
    client = MagicMock()

    # Default: buyer_lead classification (PASS)
    default_response = MagicMock()
    default_response.content = [
        MagicMock(
            text='{"classification": "buyer_lead", "confidence": 0.95, "reason": "genuine interest"}'
        )
    ]
    client.messages.create.return_value = default_response

    return client
