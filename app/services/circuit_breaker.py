"""Redis-backed circuit breaker for external service calls.

Manual implementation — pybreaker is incompatible with decode_responses=True
Redis pools. State is stored in Redis keys with TTL-based auto-reset.

States:
    CLOSED  — normal operation, requests pass through
    OPEN    — failures exceeded threshold, requests blocked for TTL seconds
    HALF_OPEN — not explicitly used; OPEN expires via Redis TTL -> CLOSED

Usage:
    breaker = RedisCircuitBreaker("claude", redis_client)
    result = await breaker.call(some_async_coroutine)

Per-service circuit breakers use SEPARATE key namespaces:
    circuit:claude   — Anthropic API calls
    circuit:ghl      — GHL API calls
    circuit:neo4j    — Neo4j calls (future)
"""

from __future__ import annotations

import redis
import structlog

logger = structlog.get_logger()

# Defaults
CIRCUIT_TTL = 60          # OPEN state duration in seconds (auto-resets via Redis TTL)
FAILURE_THRESHOLD = 3     # Consecutive failures before tripping to OPEN


class CircuitOpenError(Exception):
    """Raised when a circuit breaker is in OPEN state."""

    pass


class RedisCircuitBreaker:
    """Redis-backed circuit breaker with CLOSED/OPEN states.

    OPEN state auto-expires via Redis key TTL (no HALF_OPEN polling needed).
    All Redis operations are SYNC because the existing Redis pool in the
    codebase is sync (redis.Redis, not redis.asyncio). The call() method
    is async (awaits the protected coroutine) but uses sync Redis for
    state checks.
    """

    def __init__(
        self,
        name: str,
        redis_client: redis.Redis,
        failure_threshold: int = FAILURE_THRESHOLD,
        ttl: int = CIRCUIT_TTL,
    ) -> None:
        self.name = name
        self.redis = redis_client
        self.failure_threshold = failure_threshold
        self.ttl = ttl
        # Key names: circuit:{name} (state), circuit:{name}:failures (counter)
        self._state_key = f"circuit:{name}"
        self._failures_key = f"circuit:{name}:failures"

    def get_state(self) -> str:
        """Return current circuit state: CLOSED, OPEN, or HALF_OPEN.

        If the Redis key does not exist, the circuit is CLOSED.
        """
        state = self.redis.get(self._state_key)
        if state is None:
            return "CLOSED"
        return str(state)

    def is_open(self) -> bool:
        """Return True if the circuit is OPEN (blocking requests)."""
        return self.get_state() == "OPEN"

    def record_failure(self) -> None:
        """Record a failure. If threshold exceeded, trip to OPEN.

        Uses INCR + EXPIRE on the failures counter. When failures reach
        the threshold, sets the state key to OPEN with TTL and clears
        the failures counter.
        """
        failures = self.redis.incr(self._failures_key)
        self.redis.expire(self._failures_key, self.ttl)

        if failures >= self.failure_threshold:
            self.redis.set(self._state_key, "OPEN", ex=self.ttl)
            self.redis.delete(self._failures_key)
            logger.warning(
                "circuit_breaker.opened",
                name=self.name,
                failures=failures,
                threshold=self.failure_threshold,
                ttl=self.ttl,
            )

    def record_success(self) -> None:
        """Record a success — reset circuit to CLOSED.

        Deletes both the state key and the failures counter.
        """
        self.redis.delete(self._state_key)
        self.redis.delete(self._failures_key)
        logger.debug("circuit_breaker.success", name=self.name)

    async def call(self, coro):
        """Execute an async coroutine with circuit breaker protection.

        Args:
            coro: An awaitable coroutine to execute.

        Returns:
            The result of the coroutine.

        Raises:
            CircuitOpenError: If the circuit is OPEN.
            Exception: Re-raises any exception from the coroutine after
                recording the failure.
        """
        if self.is_open():
            logger.warning("circuit_breaker.blocked", name=self.name)
            raise CircuitOpenError(f"Circuit {self.name} is OPEN")

        try:
            result = await coro
        except Exception:
            self.record_failure()
            raise
        else:
            self.record_success()
            return result
