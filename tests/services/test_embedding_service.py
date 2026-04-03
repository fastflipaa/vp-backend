"""Tests for EmbeddingService -- httpx-based OpenAI embeddings with circuit breaker.

Unit tests with mocked httpx.AsyncClient and fakeredis.
NO real OpenAI API calls.
"""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, patch

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


# ---------------------------------------------------------------------------
# embed_text / embed_texts
# ---------------------------------------------------------------------------

class TestEmbedText:
    async def test_embed_text_returns_vector(self, embedding_svc):
        """embed_text returns a 768-dim float vector."""
        mock_response = httpx.Response(
            200,
            json={"data": [{"embedding": [0.1] * 768}]},
            request=httpx.Request("POST", "https://api.openai.com/v1/embeddings"),
        )

        with patch.object(embedding_svc._client, "post", new_callable=AsyncMock, return_value=mock_response):
            result = await embedding_svc.embed_text("hello")

        assert isinstance(result, list)
        assert len(result) == 768

    async def test_embed_texts_batch(self, embedding_svc):
        """embed_texts returns one vector per input text."""
        mock_response = httpx.Response(
            200,
            json={"data": [{"embedding": [0.1] * 768}, {"embedding": [0.2] * 768}]},
            request=httpx.Request("POST", "https://api.openai.com/v1/embeddings"),
        )

        with patch.object(embedding_svc._client, "post", new_callable=AsyncMock, return_value=mock_response):
            result = await embedding_svc.embed_texts(["hello", "world"])

        assert len(result) == 2
        assert len(result[0]) == 768
        assert len(result[1]) == 768


# ---------------------------------------------------------------------------
# Circuit breaker tests
# ---------------------------------------------------------------------------

class TestCircuitBreaker:
    async def test_circuit_opens_after_3_failures(self, embedding_svc, redis_client):
        """Circuit breaker opens after _CB_FAILURE_THRESHOLD failures."""
        with patch.object(
            embedding_svc._client,
            "post",
            new_callable=AsyncMock,
            side_effect=httpx.RequestError("Connection failed", request=httpx.Request("POST", "https://api.openai.com")),
        ):
            for _ in range(_CB_FAILURE_THRESHOLD):
                with pytest.raises(httpx.RequestError):
                    await embedding_svc.embed_texts(["test"])

            # Circuit should now be open -- next call raises EmbeddingCircuitOpen
            with pytest.raises(EmbeddingCircuitOpen):
                await embedding_svc.embed_texts(["test"])

    async def test_circuit_recovers_after_success(self, embedding_svc, redis_client):
        """Circuit breaker recovers after a successful call."""
        # Open the circuit manually
        import time
        redis_client.set(_CB_FAILURES_KEY, str(_CB_FAILURE_THRESHOLD))
        redis_client.set(_CB_OPEN_UNTIL_KEY, str(time.time() + 60))

        # Manually clear the open_until to simulate half-open
        redis_client.delete(_CB_OPEN_UNTIL_KEY)

        mock_response = httpx.Response(
            200,
            json={"data": [{"embedding": [0.1] * 768}]},
            request=httpx.Request("POST", "https://api.openai.com/v1/embeddings"),
        )

        with patch.object(embedding_svc._client, "post", new_callable=AsyncMock, return_value=mock_response):
            result = await embedding_svc.embed_texts(["hello"])

        assert len(result) == 1
        # Failures should be reset
        assert redis_client.get(_CB_FAILURES_KEY) is None

    async def test_no_api_key_raises_circuit_open(self, redis_client):
        """EmbeddingCircuitOpen raised when OPENAI_API_KEY is empty."""
        svc = EmbeddingService()
        svc._redis = redis_client
        svc._api_key = ""

        with pytest.raises(EmbeddingCircuitOpen, match="not configured"):
            await svc.embed_texts(["hello"])

    async def test_records_failure_on_http_error(self, embedding_svc, redis_client):
        """HTTP 500 response records a failure in Redis."""
        mock_response = httpx.Response(
            500,
            request=httpx.Request("POST", "https://api.openai.com/v1/embeddings"),
        )

        with patch.object(embedding_svc._client, "post", new_callable=AsyncMock, return_value=mock_response):
            with pytest.raises(httpx.HTTPStatusError):
                await embedding_svc.embed_texts(["test"])

        # Failure should be recorded
        failures = redis_client.get(_CB_FAILURES_KEY)
        assert failures is not None
        assert int(failures) >= 1
