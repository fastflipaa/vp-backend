"""Tests for EmbeddingService -- httpx-based OpenAI embeddings with circuit breaker.

Uses ``httpx.MockTransport`` to intercept the HTTP calls made inside
``EmbeddingService.embed_texts``. The service constructs a fresh
``httpx.AsyncClient(timeout=30.0)`` per call (via async context manager),
so we patch ``httpx.AsyncClient`` at the embedding_service module level
to inject a transport that returns scripted responses.

NO real OpenAI API calls.
"""

from __future__ import annotations

from typing import Callable

import httpx
import pytest

from app.services.monitoring.embedding_service import (
    EmbeddingCircuitOpen,
    EmbeddingService,
    _CB_FAILURES_KEY,
    _CB_FAILURE_THRESHOLD,
    _CB_OPEN_UNTIL_KEY,
)

pytestmark = pytest.mark.asyncio


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture()
def embedding_svc(redis_client):
    """EmbeddingService with fakeredis and test API key injected."""
    svc = EmbeddingService()
    svc._redis = redis_client
    svc._api_key = "test-key"
    return svc


@pytest.fixture()
def mock_httpx(monkeypatch):
    """Patch ``httpx.AsyncClient`` in embedding_service so all .post() calls
    return scripted responses from a list.

    Returns a callable that takes a handler ``(request) -> httpx.Response``
    and installs it as the mock transport. Each test uses this to script
    its expected HTTP response.
    """

    original_async_client = httpx.AsyncClient

    def install(handler: Callable[[httpx.Request], httpx.Response]):
        transport = httpx.MockTransport(handler)

        def patched_async_client(*args, **kwargs):
            kwargs["transport"] = transport
            return original_async_client(*args, **kwargs)

        monkeypatch.setattr(
            "app.services.monitoring.embedding_service.httpx.AsyncClient",
            patched_async_client,
        )

    return install


def _ok_response(vectors: list[list[float]]) -> Callable[[httpx.Request], httpx.Response]:
    """Return a MockTransport handler that emits a 200 with the given vectors."""

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            json={"data": [{"embedding": v} for v in vectors]},
        )

    return handler


def _error_response(status: int) -> Callable[[httpx.Request], httpx.Response]:
    """Return a MockTransport handler that emits the given HTTP status."""

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(status)

    return handler


def _connection_error() -> Callable[[httpx.Request], httpx.Response]:
    """Return a handler that raises a connection error."""

    def handler(request: httpx.Request) -> httpx.Response:
        raise httpx.RequestError("Connection failed", request=request)

    return handler


# ---------------------------------------------------------------------------
# embed_text / embed_texts
# ---------------------------------------------------------------------------


class TestEmbedText:
    async def test_embed_text_returns_vector(self, embedding_svc, mock_httpx):
        """embed_text returns a 768-dim float vector."""
        mock_httpx(_ok_response([[0.1] * 768]))
        result = await embedding_svc.embed_text("hello")
        assert isinstance(result, list)
        assert len(result) == 768

    async def test_embed_texts_batch(self, embedding_svc, mock_httpx):
        """embed_texts returns one vector per input text."""
        mock_httpx(_ok_response([[0.1] * 768, [0.2] * 768]))
        result = await embedding_svc.embed_texts(["hello", "world"])
        assert len(result) == 2
        assert len(result[0]) == 768
        assert len(result[1]) == 768


# ---------------------------------------------------------------------------
# Circuit breaker tests
# ---------------------------------------------------------------------------


class TestCircuitBreaker:
    async def test_circuit_opens_after_3_failures(self, embedding_svc, mock_httpx, redis_client):
        """Circuit breaker opens after _CB_FAILURE_THRESHOLD failures."""
        mock_httpx(_connection_error())

        for _ in range(_CB_FAILURE_THRESHOLD):
            with pytest.raises(httpx.RequestError):
                await embedding_svc.embed_texts(["test"])

        # Circuit should now be open -- next call raises EmbeddingCircuitOpen
        with pytest.raises(EmbeddingCircuitOpen):
            await embedding_svc.embed_texts(["test"])

    async def test_circuit_recovers_after_success(self, embedding_svc, mock_httpx, redis_client):
        """Circuit breaker recovers after a successful call."""
        import time

        # Pre-load failure state, then clear the open_until to simulate half-open
        redis_client.set(_CB_FAILURES_KEY, str(_CB_FAILURE_THRESHOLD))
        redis_client.set(_CB_OPEN_UNTIL_KEY, str(time.time() + 60))
        redis_client.delete(_CB_OPEN_UNTIL_KEY)

        mock_httpx(_ok_response([[0.1] * 768]))
        result = await embedding_svc.embed_texts(["hello"])

        assert len(result) == 1
        # Failures should be reset on success
        assert redis_client.get(_CB_FAILURES_KEY) is None

    async def test_no_api_key_raises_circuit_open(self, redis_client):
        """EmbeddingCircuitOpen raised when OPENAI_API_KEY is empty."""
        svc = EmbeddingService()
        svc._redis = redis_client
        svc._api_key = ""

        with pytest.raises(EmbeddingCircuitOpen, match="not configured"):
            await svc.embed_texts(["hello"])

    async def test_records_failure_on_http_error(self, embedding_svc, mock_httpx, redis_client):
        """HTTP 500 response records a failure in Redis."""
        mock_httpx(_error_response(500))

        with pytest.raises(httpx.HTTPStatusError):
            await embedding_svc.embed_texts(["test"])

        # Failure should be recorded
        failures = redis_client.get(_CB_FAILURES_KEY)
        assert failures is not None
        assert int(failures) >= 1
