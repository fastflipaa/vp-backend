"""Tests for the night mode gate -- blocks proactive messages during CDMX night.

Uses freezegun for time control since the gate calls datetime.now(tz=CDMX_TZ)
as a fallback when the payload lacks receivedAt/dateAdded timestamps.
We set receivedAt in the payload for deterministic control.
"""

from __future__ import annotations

import json

from app.gates.base import GateDecision
from app.gates.night_mode import NIGHT_QUEUE_KEY, check_night_mode


class TestNightMode:
    """Night mode gate: 10PM-8AM CDMX block for proactive messages."""

    def test_daytime_proactive_passes(self, redis_client, make_inbound_payload):
        """2 PM CDMX outbound -> PASS (daytime)."""
        # 2 PM CDMX = 20:00 UTC (CDMX is UTC-6)
        payload = make_inbound_payload(
            direction="outbound",
            receivedAt="2026-03-31T20:00:00Z",  # 14:00 CDMX
        )
        result = check_night_mode(payload, "trace-1", redis_client)

        assert result.decision == GateDecision.PASS
        assert "daytime" in result.reason

    def test_night_proactive_blocked(self, redis_client, make_inbound_payload):
        """11 PM CDMX outbound -> BLOCK (night)."""
        # 11 PM CDMX = 05:00 UTC next day
        payload = make_inbound_payload(
            direction="outbound",
            receivedAt="2026-04-01T05:00:00Z",  # 23:00 CDMX
        )
        result = check_night_mode(payload, "trace-1", redis_client)

        assert result.decision == GateDecision.BLOCK
        assert "night_proactive_blocked" in result.reason

    def test_night_reactive_passes(self, redis_client, make_inbound_payload):
        """11 PM CDMX inbound -> PASS (reactive, 24/7)."""
        # 11 PM CDMX = 05:00 UTC next day
        payload = make_inbound_payload(
            direction="inbound",
            receivedAt="2026-04-01T05:00:00Z",  # 23:00 CDMX
        )
        result = check_night_mode(payload, "trace-1", redis_client)

        assert result.decision == GateDecision.PASS
        assert "night_reactive_allowed" in result.reason

    def test_night_queue_entry_created(self, redis_client, make_inbound_payload):
        """Night-blocked proactive message should be queued in Redis."""
        payload = make_inbound_payload(
            direction="outbound",
            contactId="contact-night-queue",
            receivedAt="2026-04-01T05:00:00Z",  # 23:00 CDMX
        )
        check_night_mode(payload, "trace-1", redis_client)

        # Verify queue entry exists
        queue_length = redis_client.llen(NIGHT_QUEUE_KEY)
        assert queue_length >= 1

        # Verify the queued entry is valid JSON with expected fields
        raw = redis_client.lindex(NIGHT_QUEUE_KEY, -1)
        entry = json.loads(raw)
        assert "payload" in entry
        assert "trace_id" in entry
        assert "queued_at" in entry

    def test_boundary_8am_passes(self, redis_client, make_inbound_payload):
        """Exactly 8:00 AM CDMX outbound -> PASS (boundary, daytime starts)."""
        # 8 AM CDMX = 14:00 UTC
        payload = make_inbound_payload(
            direction="outbound",
            receivedAt="2026-03-31T14:00:00Z",  # 08:00 CDMX
        )
        result = check_night_mode(payload, "trace-1", redis_client)

        assert result.decision == GateDecision.PASS

    def test_boundary_10pm_blocks(self, redis_client, make_inbound_payload):
        """Exactly 10:00 PM CDMX outbound -> BLOCK (boundary, night starts)."""
        # 10 PM CDMX = 04:00 UTC next day
        payload = make_inbound_payload(
            direction="outbound",
            receivedAt="2026-04-01T04:00:00Z",  # 22:00 CDMX
        )
        result = check_night_mode(payload, "trace-1", redis_client)

        assert result.decision == GateDecision.BLOCK

    def test_early_morning_blocks(self, redis_client, make_inbound_payload):
        """3 AM CDMX outbound -> BLOCK (still within night window)."""
        # 3 AM CDMX = 09:00 UTC
        payload = make_inbound_payload(
            direction="outbound",
            receivedAt="2026-03-31T09:00:00Z",  # 03:00 CDMX
        )
        result = check_night_mode(payload, "trace-1", redis_client)

        assert result.decision == GateDecision.BLOCK
