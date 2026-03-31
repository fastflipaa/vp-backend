"""Tests for the dedup gate -- blocks duplicate webhooks via Redis SETNX.

Uses fakeredis for real Redis command execution (no mocking).
"""

from __future__ import annotations

from app.gates.base import GateDecision
from app.gates.dedup import DEDUP_TTL_SECONDS, check_dedup


class TestDedup:
    """Dedup gate: messageId-based duplicate detection with 15-min TTL."""

    def test_new_message_passes(self, redis_client, make_inbound_payload):
        """A fresh messageId should pass through (first seen)."""
        payload = make_inbound_payload(messageId="msg-fresh-001")
        result = check_dedup(payload, "trace-1", redis_client)

        assert result.decision == GateDecision.PASS
        assert result.gate_name == "dedup"
        assert result.reason == "new_message"

    def test_duplicate_message_blocked(self, redis_client, make_inbound_payload):
        """Same messageId sent twice: second call should be BLOCKED."""
        payload = make_inbound_payload(messageId="msg-dup-001")

        first = check_dedup(payload, "trace-1", redis_client)
        second = check_dedup(payload, "trace-2", redis_client)

        assert first.decision == GateDecision.PASS
        assert second.decision == GateDecision.BLOCK
        assert "duplicate_message_id" in second.reason

    def test_no_message_id_passes(self, redis_client, make_inbound_payload):
        """Payload without messageId should pass (no dedup possible)."""
        payload = make_inbound_payload(messageId="")
        result = check_dedup(payload, "trace-1", redis_client)

        assert result.decision == GateDecision.PASS
        assert result.reason == "no_message_id_skip"

    def test_different_message_ids_both_pass(self, redis_client, make_inbound_payload):
        """Two different messageIds should both pass."""
        payload_a = make_inbound_payload(messageId="msg-a")
        payload_b = make_inbound_payload(messageId="msg-b")

        result_a = check_dedup(payload_a, "trace-1", redis_client)
        result_b = check_dedup(payload_b, "trace-2", redis_client)

        assert result_a.decision == GateDecision.PASS
        assert result_b.decision == GateDecision.PASS

    def test_dedup_key_has_ttl(self, redis_client, make_inbound_payload):
        """After dedup, the Redis key should have a TTL <= DEDUP_TTL_SECONDS."""
        payload = make_inbound_payload(messageId="msg-ttl-check")
        check_dedup(payload, "trace-1", redis_client)

        key = "dedup:msg-ttl-check"
        ttl = redis_client.ttl(key)
        assert 0 < ttl <= DEDUP_TTL_SECONDS

    def test_missing_message_id_key_passes(self, redis_client):
        """Payload dict without messageId key at all should pass."""
        payload = {"contactId": "c1", "message": "hello"}
        result = check_dedup(payload, "trace-1", redis_client)

        assert result.decision == GateDecision.PASS
        assert result.reason == "no_message_id_skip"
