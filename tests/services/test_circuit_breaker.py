"""Circuit breaker tests -- uses fakeredis from root conftest.

Tests RedisCircuitBreaker with real Redis commands via fakeredis:
record_failure threshold, TTL-based reset, record_success, is_open.
"""

from __future__ import annotations

import pytest

from app.services.circuit_breaker import (
    CIRCUIT_TTL,
    CircuitOpenError,
    FAILURE_THRESHOLD,
    RedisCircuitBreaker,
)


@pytest.fixture()
def breaker(redis_client) -> RedisCircuitBreaker:
    """Circuit breaker with fakeredis client, default threshold=3, ttl=60."""
    return RedisCircuitBreaker("test_service", redis_client)


class TestCircuitBreakerState:
    """Circuit breaker state transitions: CLOSED -> OPEN -> CLOSED."""

    def test_initial_state_is_closed(self, breaker: RedisCircuitBreaker):
        """New circuit breaker starts in CLOSED state."""
        assert breaker.get_state() == "CLOSED"
        assert breaker.is_open() is False

    def test_failures_below_threshold_stay_closed(self, breaker: RedisCircuitBreaker):
        """Recording fewer failures than threshold keeps circuit CLOSED."""
        breaker.record_failure()
        breaker.record_failure()
        assert breaker.get_state() == "CLOSED"
        assert breaker.is_open() is False

    def test_failures_at_threshold_open_circuit(self, breaker: RedisCircuitBreaker):
        """Recording failures at threshold opens the circuit."""
        for _ in range(FAILURE_THRESHOLD):
            breaker.record_failure()
        assert breaker.get_state() == "OPEN"
        assert breaker.is_open() is True

    def test_record_success_resets_to_closed(self, breaker: RedisCircuitBreaker):
        """record_success() clears both state key and failure counter."""
        # Record some failures
        breaker.record_failure()
        breaker.record_failure()
        # Success resets everything
        breaker.record_success()
        assert breaker.get_state() == "CLOSED"
        # Verify failures counter was also reset
        assert breaker.redis.get(breaker._failures_key) is None

    def test_record_success_after_open_resets(self, breaker: RedisCircuitBreaker):
        """record_success() resets circuit even from OPEN state."""
        for _ in range(FAILURE_THRESHOLD):
            breaker.record_failure()
        assert breaker.is_open() is True
        breaker.record_success()
        assert breaker.is_open() is False
        assert breaker.get_state() == "CLOSED"


class TestCircuitBreakerTTL:
    """TTL-based auto-reset from OPEN to CLOSED."""

    def test_open_state_has_ttl(self, breaker: RedisCircuitBreaker, redis_client):
        """OPEN state key has TTL set for auto-expiry."""
        for _ in range(FAILURE_THRESHOLD):
            breaker.record_failure()
        ttl = redis_client.ttl(breaker._state_key)
        assert ttl > 0
        assert ttl <= CIRCUIT_TTL

    def test_failures_counter_cleared_on_open(self, breaker: RedisCircuitBreaker, redis_client):
        """Failures counter is deleted when circuit opens (reset for next cycle)."""
        for _ in range(FAILURE_THRESHOLD):
            breaker.record_failure()
        # After opening, failures key should be deleted
        assert redis_client.get(breaker._failures_key) is None


class TestCircuitBreakerCall:
    """Async call() method with circuit breaker protection."""

    @pytest.mark.asyncio
    async def test_call_succeeds_when_closed(self, breaker: RedisCircuitBreaker):
        """call() executes coroutine normally when circuit is CLOSED."""
        async def good_coro():
            return "success"

        result = await breaker.call(good_coro())
        assert result == "success"

    @pytest.mark.asyncio
    async def test_call_raises_circuit_open_error(self, breaker: RedisCircuitBreaker):
        """call() raises CircuitOpenError when circuit is OPEN."""
        for _ in range(FAILURE_THRESHOLD):
            breaker.record_failure()

        async def any_coro():
            return "should not reach"

        with pytest.raises(CircuitOpenError, match="test_service"):
            await breaker.call(any_coro())

    @pytest.mark.asyncio
    async def test_call_records_failure_on_exception(self, breaker: RedisCircuitBreaker, redis_client):
        """call() records failure when the coroutine raises."""
        async def bad_coro():
            raise ValueError("simulated error")

        with pytest.raises(ValueError, match="simulated error"):
            await breaker.call(bad_coro())

        # One failure should be recorded
        failures = redis_client.get(breaker._failures_key)
        assert failures == "1"

    @pytest.mark.asyncio
    async def test_call_records_success_on_completion(self, breaker: RedisCircuitBreaker, redis_client):
        """call() records success when coroutine succeeds (resets counters)."""
        # Add a failure first
        breaker.record_failure()
        assert redis_client.get(breaker._failures_key) == "1"

        async def good_coro():
            return "ok"

        await breaker.call(good_coro())
        # Success should reset the failure counter
        assert redis_client.get(breaker._failures_key) is None


class TestCircuitBreakerConfig:
    """Custom configuration for per-service circuit breakers."""

    def test_custom_threshold(self, redis_client):
        """Custom failure_threshold is respected."""
        breaker = RedisCircuitBreaker("custom", redis_client, failure_threshold=5)
        for _ in range(4):
            breaker.record_failure()
        assert breaker.is_open() is False
        breaker.record_failure()  # 5th failure
        assert breaker.is_open() is True

    def test_key_namespacing(self, redis_client):
        """Different service names use separate Redis keys."""
        b1 = RedisCircuitBreaker("claude", redis_client)
        b2 = RedisCircuitBreaker("ghl", redis_client)
        assert b1._state_key == "circuit:claude"
        assert b2._state_key == "circuit:ghl"
        assert b1._failures_key == "circuit:claude:failures"
        assert b2._failures_key == "circuit:ghl:failures"

    def test_default_constants(self):
        """Default constants match expected values."""
        assert FAILURE_THRESHOLD == 3
        assert CIRCUIT_TTL == 60
