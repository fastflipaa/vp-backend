"""Tests for ground_truth_tracker.

Verifies the out-of-band ground truth recording works correctly:
  * record_inbound and record_delivery both ZADD with epoch ms scores
  * get_window_counts respects the window cutoff
  * Old entries get cleaned up by cleanup_old_entries
  * Tracker is fail-safe (silent on Redis errors, never raises)

These tests use the fakeredis fixture from conftest. The tracker module
maintains its own lazy singleton client; we monkeypatch it to point at
the test fakeredis instance.
"""

from __future__ import annotations

import time

import pytest

from app.services.monitoring import ground_truth_tracker as gt


@pytest.fixture(autouse=True)
def patch_tracker_redis(redis_client, monkeypatch):
    """Replace the tracker's lazy redis client with the test fakeredis client."""
    monkeypatch.setattr(gt, "_redis_client", redis_client)
    yield
    # Reset after test
    monkeypatch.setattr(gt, "_redis_client", None)


class TestRecordInbound:
    def test_records_trace_id(self, redis_client):
        gt.record_inbound("trace_001")
        # Should be in the sorted set
        assert redis_client.zcard(gt.INBOUND_KEY) == 1
        # The member should be our trace id
        members = redis_client.zrange(gt.INBOUND_KEY, 0, -1)
        assert "trace_001" in members

    def test_score_is_epoch_ms(self, redis_client):
        before = time.time() * 1000
        gt.record_inbound("trace_002")
        after = time.time() * 1000
        score = redis_client.zscore(gt.INBOUND_KEY, "trace_002")
        assert score is not None
        assert before <= float(score) <= after

    def test_empty_trace_id_is_noop(self, redis_client):
        gt.record_inbound("")
        assert redis_client.zcard(gt.INBOUND_KEY) == 0

    def test_multiple_records_accumulate(self, redis_client):
        for i in range(5):
            gt.record_inbound(f"trace_{i}")
        assert redis_client.zcard(gt.INBOUND_KEY) == 5


class TestRecordDelivery:
    def test_records_trace_id(self, redis_client):
        gt.record_delivery("trace_001")
        assert redis_client.zcard(gt.DELIVERY_KEY) == 1

    def test_separate_from_inbound(self, redis_client):
        gt.record_inbound("trace_001")
        gt.record_delivery("trace_002")
        assert redis_client.zcard(gt.INBOUND_KEY) == 1
        assert redis_client.zcard(gt.DELIVERY_KEY) == 1
        assert "trace_001" in redis_client.zrange(gt.INBOUND_KEY, 0, -1)
        assert "trace_002" in redis_client.zrange(gt.DELIVERY_KEY, 0, -1)


class TestGetWindowCounts:
    def test_zero_zero_when_empty(self, redis_client):
        inbound, delivery = gt.get_window_counts(window_minutes=15)
        assert inbound == 0
        assert delivery == 0

    def test_counts_recent_entries(self, redis_client):
        for i in range(3):
            gt.record_inbound(f"in_{i}")
        for i in range(2):
            gt.record_delivery(f"out_{i}")
        inbound, delivery = gt.get_window_counts(window_minutes=15)
        assert inbound == 3
        assert delivery == 2

    def test_excludes_entries_older_than_window(self, redis_client):
        # Insert old entries directly with epoch ms scores far in the past
        old_score = (time.time() - 3600) * 1000  # 1 hour ago
        redis_client.zadd(gt.INBOUND_KEY, {"old_1": old_score})
        redis_client.zadd(gt.DELIVERY_KEY, {"old_2": old_score})
        # And recent ones
        gt.record_inbound("recent_1")
        gt.record_delivery("recent_2")

        inbound, delivery = gt.get_window_counts(window_minutes=15)
        assert inbound == 1
        assert delivery == 1


class TestCleanupOldEntries:
    def test_removes_entries_older_than_keep_minutes(self, redis_client):
        old_score = (time.time() - 7200) * 1000  # 2 hours ago
        redis_client.zadd(gt.INBOUND_KEY, {"ancient": old_score})
        gt.record_inbound("recent")

        assert redis_client.zcard(gt.INBOUND_KEY) == 2
        gt.cleanup_old_entries(keep_minutes=60)
        assert redis_client.zcard(gt.INBOUND_KEY) == 1
        members = redis_client.zrange(gt.INBOUND_KEY, 0, -1)
        assert "recent" in members
        assert "ancient" not in members


class TestFailSafe:
    def test_record_inbound_swallows_redis_errors(self, monkeypatch):
        """Tracker MUST NEVER break the webhook handler."""
        class BrokenRedis:
            def zadd(self, *a, **k):
                raise RuntimeError("redis exploded")
            def expire(self, *a, **k):
                raise RuntimeError("nope")

        monkeypatch.setattr(gt, "_redis_client", BrokenRedis())
        # Should not raise
        gt.record_inbound("trace_001")

    def test_record_delivery_swallows_redis_errors(self, monkeypatch):
        class BrokenRedis:
            def zadd(self, *a, **k):
                raise RuntimeError("redis exploded")
            def expire(self, *a, **k):
                raise RuntimeError("nope")

        monkeypatch.setattr(gt, "_redis_client", BrokenRedis())
        gt.record_delivery("trace_001")  # should not raise

    def test_get_window_counts_returns_zero_zero_on_error(self, monkeypatch):
        class BrokenRedis:
            def zcount(self, *a, **k):
                raise RuntimeError("redis exploded")

        monkeypatch.setattr(gt, "_redis_client", BrokenRedis())
        inbound, delivery = gt.get_window_counts(window_minutes=15)
        assert inbound == 0
        assert delivery == 0
