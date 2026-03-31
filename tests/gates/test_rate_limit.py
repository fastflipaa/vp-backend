"""Tests for the rate limit gate -- sliding window + unreplied cap.

Uses fakeredis for real Redis sorted set operations.
"""

from __future__ import annotations

import time

from app.gates.base import GateDecision
from app.gates.rate_limit import (
    DAILY_CAP,
    UNREPLIED_CAP,
    check_rate_limit,
    record_inbound,
    record_outbound,
)


class TestRateLimit:
    """Rate limit gate: dual-cap (daily sliding window + unreplied counter)."""

    def test_under_limit_passes(self, redis_client, make_inbound_payload):
        """First message for a contact should pass (no counters yet)."""
        payload = make_inbound_payload(contactId="contact-fresh")
        result = check_rate_limit(payload, "trace-1", redis_client)

        assert result.decision == GateDecision.PASS
        assert "within_limits" in result.reason

    def test_unreplied_cap_blocks(self, redis_client, make_inbound_payload):
        """3 unreplied outbound messages should block the 4th attempt."""
        contact_id = "contact-unreplied"

        # Record UNREPLIED_CAP outbound messages
        for _ in range(UNREPLIED_CAP):
            record_outbound(contact_id, redis_client)

        payload = make_inbound_payload(contactId=contact_id)
        result = check_rate_limit(payload, "trace-1", redis_client)

        assert result.decision == GateDecision.BLOCK
        assert "unreplied_cap_exceeded" in result.reason

    def test_daily_cap_blocks(self, redis_client, make_inbound_payload):
        """15 outbound messages in 24h should block the next attempt."""
        contact_id = "contact-daily"

        # Record DAILY_CAP outbound messages (also resets unreplied each cycle)
        for i in range(DAILY_CAP):
            record_outbound(contact_id, redis_client)
            # Reset unreplied counter so we don't hit unreplied cap first
            record_inbound(contact_id, redis_client)

        # Now record enough outbound to NOT hit unreplied cap but to exceed daily
        # At this point daily_count == DAILY_CAP (15 entries in sorted set)
        payload = make_inbound_payload(contactId=contact_id)
        result = check_rate_limit(payload, "trace-1", redis_client)

        assert result.decision == GateDecision.BLOCK
        assert "daily_cap_exceeded" in result.reason

    def test_sliding_window_expiry(self, redis_client, make_inbound_payload):
        """Old messages outside the 24h window should not count toward daily cap."""
        contact_id = "contact-window"
        daily_key = f"rate_limit:daily:{contact_id}"

        # Manually add old entries to the sorted set (timestamps from 25h ago)
        old_timestamp = time.time() - 90000  # 25 hours ago
        for i in range(DAILY_CAP):
            ts = old_timestamp + i
            redis_client.zadd(daily_key, {str(ts): ts})

        payload = make_inbound_payload(contactId=contact_id)
        result = check_rate_limit(payload, "trace-1", redis_client)

        # Old entries should be pruned by ZREMRANGEBYSCORE, so we're under limit
        assert result.decision == GateDecision.PASS

    def test_inbound_resets_unreplied(self, redis_client, make_inbound_payload):
        """An inbound message should reset the unreplied counter to 0."""
        contact_id = "contact-reset"

        # Send some outbound (but not enough to hit cap)
        record_outbound(contact_id, redis_client)
        record_outbound(contact_id, redis_client)

        # Inbound resets
        record_inbound(contact_id, redis_client)

        # Now unreplied is 0, so should pass
        payload = make_inbound_payload(contactId=contact_id)
        result = check_rate_limit(payload, "trace-1", redis_client)

        assert result.decision == GateDecision.PASS

        # Verify the unreplied key is deleted
        unreplied_key = f"rate_limit:unreplied:{contact_id}"
        assert redis_client.get(unreplied_key) is None

    def test_no_contact_id_passes(self, redis_client, make_inbound_payload):
        """Payload without contactId should pass (no rate limiting possible)."""
        payload = make_inbound_payload(contactId="")
        result = check_rate_limit(payload, "trace-1", redis_client)

        assert result.decision == GateDecision.PASS
        assert result.reason == "no_contact_id_skip"
