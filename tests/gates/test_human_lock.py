"""Tests for the human lock gate -- blocks AI when human agent is active.

Uses fakeredis for real Redis command execution.
"""

from __future__ import annotations

from app.gates.base import GateDecision
from app.gates.human_lock import (
    HUMAN_LOCK_TTL_SECONDS,
    check_human_lock,
    set_human_lock,
)


class TestHumanLock:
    """Human lock gate: Redis-based lock detection with trigger word release."""

    def test_no_lock_passes(self, redis_client, make_inbound_payload):
        """No lock key in Redis -> PASS."""
        payload = make_inbound_payload(contactId="contact-unlocked")
        result = check_human_lock(payload, "trace-1", redis_client)

        assert result.decision == GateDecision.PASS
        assert result.reason == "no_human_lock"

    def test_locked_contact_blocked(self, redis_client, make_inbound_payload):
        """Active human lock on contact -> BLOCK."""
        contact_id = "contact-locked"
        set_human_lock(contact_id, "agent-fernando", redis_client)

        payload = make_inbound_payload(contactId=contact_id)
        result = check_human_lock(payload, "trace-1", redis_client)

        assert result.decision == GateDecision.BLOCK
        assert "human_agent_active" in result.reason

    def test_trigger_word_releases_lock(self, redis_client, make_inbound_payload):
        """Inbound message with trigger word should release lock and PASS."""
        contact_id = "contact-trigger"
        set_human_lock(contact_id, "agent-lorena", redis_client)

        # Trigger phrase: "resume bot" matches TRIGGER_WORD_PATTERNS
        payload = make_inbound_payload(
            contactId=contact_id,
            message="Please resume bot",
            direction="inbound",
        )
        result = check_human_lock(payload, "trace-1", redis_client)

        assert result.decision == GateDecision.PASS
        assert "trigger_word" in result.reason
        # Lock should be deleted
        assert not redis_client.exists(f"human_lock:{contact_id}")

    def test_lock_set_on_outbound(self, redis_client):
        """set_human_lock should create the lock key + history key."""
        contact_id = "contact-outbound"
        set_human_lock(contact_id, "agent-phill", redis_client)

        assert redis_client.exists(f"human_lock:{contact_id}")
        assert redis_client.get(f"human_lock:{contact_id}") == "agent-phill"
        assert redis_client.exists(f"human_lock_history:{contact_id}")

    def test_lock_has_ttl(self, redis_client):
        """Human lock key should have 24h TTL."""
        contact_id = "contact-ttl"
        set_human_lock(contact_id, "agent-test", redis_client)

        ttl = redis_client.ttl(f"human_lock:{contact_id}")
        assert 0 < ttl <= HUMAN_LOCK_TTL_SECONDS

    def test_different_contact_not_affected(self, redis_client, make_inbound_payload):
        """Lock on contact_A should not block contact_B."""
        set_human_lock("contact-A", "agent-1", redis_client)

        payload_b = make_inbound_payload(contactId="contact-B")
        result = check_human_lock(payload_b, "trace-1", redis_client)

        assert result.decision == GateDecision.PASS

    def test_spanish_trigger_word(self, redis_client, make_inbound_payload):
        """Spanish trigger phrase 'activar bot' should release the lock."""
        contact_id = "contact-es-trigger"
        set_human_lock(contact_id, "agente", redis_client)

        payload = make_inbound_payload(
            contactId=contact_id,
            message="Por favor activar bot",
            direction="inbound",
        )
        result = check_human_lock(payload, "trace-1", redis_client)

        assert result.decision == GateDecision.PASS
        assert "trigger_word" in result.reason
