"""Embedding service -- OpenAI text-embedding-3-small via httpx with circuit breaker.

Uses a Redis-backed circuit breaker pattern:
- CLOSED: normal operation
- OPEN: 3+ failures in 5 min -> refuse requests for 60s
- HALF_OPEN: after cool-down, allow one request to test recovery

Does NOT use the openai SDK -- direct httpx POST for minimal dependencies.
"""

from __future__ import annotations

import time

import httpx
import redis
import structlog

from app.config import settings

logger = structlog.get_logger()

# Circuit breaker Redis keys
_CB_FAILURES_KEY = "circuit:embedding:failures"
_CB_OPEN_UNTIL_KEY = "circuit:embedding:open_until"
_CB_FAILURE_THRESHOLD = 3
_CB_FAILURE_WINDOW = 300  # 5 minutes
_CB_OPEN_DURATION = 60  # seconds


class EmbeddingCircuitOpen(Exception):
    """Raised when embedding service circuit breaker is open."""
    pass


class EmbeddingService:
    """OpenAI embedding service with Redis circuit breaker."""

    def __init__(self) -> None:
        self._client = httpx.AsyncClient(timeout=30.0)
        self._api_key = settings.OPENAI_API_KEY
        self._redis: redis.Redis | None = None

    def _get_redis(self) -> redis.Redis:
        """Lazy Redis connection for circuit breaker state."""
        if self._redis is None:
            self._redis = redis.Redis.from_url(
                settings.redis_cache_url,
                decode_responses=True,
            )
        return self._redis

    def _is_circuit_open(self) -> bool:
        """Check if the circuit breaker is currently open."""
        try:
            r = self._get_redis()
            open_until = r.get(_CB_OPEN_UNTIL_KEY)
            if open_until and float(open_until) > time.time():
                return True
            return False
        except Exception:
            # Redis failure should not block embeddings
            logger.warning("embedding.circuit_breaker_check_failed")
            return False

    def _record_failure(self) -> None:
        """Record a failure and open circuit if threshold reached."""
        try:
            r = self._get_redis()
            pipe = r.pipeline()
            pipe.incr(_CB_FAILURES_KEY)
            pipe.expire(_CB_FAILURES_KEY, _CB_FAILURE_WINDOW)
            results = pipe.execute()
            failure_count = results[0]

            if failure_count >= _CB_FAILURE_THRESHOLD:
                r.set(
                    _CB_OPEN_UNTIL_KEY,
                    str(time.time() + _CB_OPEN_DURATION),
                    ex=_CB_OPEN_DURATION + 10,
                )
                logger.warning(
                    "embedding.circuit_opened",
                    failures=failure_count,
                    open_for=_CB_OPEN_DURATION,
                )
        except Exception:
            logger.warning("embedding.circuit_breaker_record_failed")

    def _reset_failures(self) -> None:
        """Reset failure counter on success (used in half-open recovery)."""
        try:
            r = self._get_redis()
            r.delete(_CB_FAILURES_KEY, _CB_OPEN_UNTIL_KEY)
        except Exception:
            pass

    async def embed_text(self, text: str) -> list[float]:
        """Embed a single text string. Returns 768-dim float vector."""
        results = await self.embed_texts([text])
        return results[0]

    async def embed_texts(self, texts: list[str]) -> list[list[float]]:
        """Embed multiple texts in a single API call.

        Returns list of 768-dim float vectors, one per input text.

        Raises:
            EmbeddingCircuitOpen: If the circuit breaker is open.
            httpx.HTTPStatusError: On API errors (after recording failure).
        """
        if self._is_circuit_open():
            raise EmbeddingCircuitOpen(
                "Embedding service circuit breaker is open. "
                "Retrying in up to 60 seconds."
            )

        if not self._api_key:
            logger.warning("embedding.no_api_key", reason="OPENAI_API_KEY not set")
            raise EmbeddingCircuitOpen("OPENAI_API_KEY not configured")

        try:
            response = await self._client.post(
                "https://api.openai.com/v1/embeddings",
                headers={
                    "Authorization": f"Bearer {self._api_key}",
                    "Content-Type": "application/json",
                },
                json={
                    "model": "text-embedding-3-small",
                    "input": texts,
                    "dimensions": 768,
                },
            )
            response.raise_for_status()
            data = response.json()

            vectors = [item["embedding"] for item in data["data"]]
            self._reset_failures()

            logger.debug(
                "embedding.success",
                count=len(texts),
                dims=len(vectors[0]) if vectors else 0,
            )
            return vectors

        except (httpx.HTTPStatusError, httpx.RequestError) as exc:
            self._record_failure()
            logger.error(
                "embedding.api_error",
                error=str(exc),
                text_count=len(texts),
            )
            raise
