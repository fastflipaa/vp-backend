"""Tests for health, readiness, and canary status endpoints.

Uses FastAPI TestClient with mocked external dependencies.
The readiness endpoint imports redis.asyncio and AsyncGraphDatabase inside
the function body, so mocks target those source modules.
"""

from __future__ import annotations

from unittest.mock import MagicMock, AsyncMock, patch

import pytest
from fastapi.testclient import TestClient

from app.main import app


# ---------------------------------------------------------------------------
# Health endpoint tests
# ---------------------------------------------------------------------------

class TestHealthEndpoint:
    def test_health_returns_200(self):
        """GET /health returns 200 with healthy status."""
        client = TestClient(app)

        response = client.get("/health")

        assert response.status_code == 200
        body = response.json()
        assert body["status"] == "healthy"

    def test_health_no_dependencies(self):
        """GET /health does not call any external services (pure liveness probe)."""
        client = TestClient(app)
        response = client.get("/health")

        assert response.status_code == 200


# ---------------------------------------------------------------------------
# Readiness endpoint tests
# ---------------------------------------------------------------------------

class TestReadyEndpoint:
    @staticmethod
    def _make_neo4j_driver_mock():
        """Create a properly structured async mock for Neo4j driver.

        The ready endpoint uses:
            driver = AsyncGraphDatabase.driver(...)
            async with driver.session() as session:
                await session.run("RETURN 1")
            await driver.close()

        driver.session() returns a sync context-manager-like object whose
        __aenter__/__aexit__ are awaitable.
        """
        mock_session = AsyncMock()
        mock_session.run = AsyncMock()

        mock_ctx = AsyncMock()
        mock_ctx.__aenter__ = AsyncMock(return_value=mock_session)
        mock_ctx.__aexit__ = AsyncMock(return_value=False)

        mock_driver = MagicMock()  # sync object -- .session() returns sync
        mock_driver.session.return_value = mock_ctx
        mock_driver.close = AsyncMock()
        return mock_driver

    def test_ready_all_ok(self):
        """GET /ready returns 200 when Redis and Neo4j are reachable."""
        mock_redis_instance = AsyncMock()
        mock_redis_instance.ping = AsyncMock()
        mock_redis_instance.aclose = AsyncMock()

        mock_driver = self._make_neo4j_driver_mock()

        with patch("redis.asyncio.from_url", return_value=mock_redis_instance), \
             patch("neo4j.AsyncGraphDatabase.driver", return_value=mock_driver):

            client = TestClient(app)
            response = client.get("/ready")

        assert response.status_code == 200
        body = response.json()
        assert body["status"] == "ready"
        assert body["checks"]["redis"] == "ok"
        assert body["checks"]["neo4j"] == "ok"

    def test_ready_redis_down(self):
        """GET /ready returns 503 when Redis is unreachable."""
        mock_driver = self._make_neo4j_driver_mock()

        with patch("redis.asyncio.from_url", side_effect=ConnectionError("Redis refused")), \
             patch("neo4j.AsyncGraphDatabase.driver", return_value=mock_driver):

            client = TestClient(app)
            response = client.get("/ready")

        assert response.status_code == 503
        body = response.json()
        assert body["status"] == "not_ready"
        assert "error" in body["checks"]["redis"]

    def test_ready_neo4j_down(self):
        """GET /ready returns 503 when Neo4j is unreachable."""
        mock_redis_instance = AsyncMock()
        mock_redis_instance.ping = AsyncMock()
        mock_redis_instance.aclose = AsyncMock()

        with patch("redis.asyncio.from_url", return_value=mock_redis_instance), \
             patch("neo4j.AsyncGraphDatabase.driver", side_effect=Exception("Neo4j connection refused")):

            client = TestClient(app)
            response = client.get("/ready")

        assert response.status_code == 503
        body = response.json()
        assert body["status"] == "not_ready"
        assert "error" in body["checks"]["neo4j"]


# ---------------------------------------------------------------------------
# Canary status endpoint tests
# ---------------------------------------------------------------------------

class TestCanaryStatus:
    def test_canary_status_returns_200(self):
        """GET /canary/status returns 200 with canary config."""
        with patch("app.api.canary.get_sync_redis") as mock_get_redis, \
             patch("app.api.canary.get_canary_stats") as mock_stats:
            mock_get_redis.return_value = MagicMock()
            mock_stats.return_value = {
                "processed_24h": 42,
                "errors_24h": 1,
                "error_rate": 0.0238,
                "avg_latency_ms": 123.5,
                "shadow_count_24h": 200,
            }

            client = TestClient(app)
            response = client.get("/canary/status")

            assert response.status_code == 200
            body = response.json()
            assert "canary_enabled" in body
            assert "canary_tag" in body
            assert body["canary_processed_24h"] == 42
            assert body["canary_errors_24h"] == 1
            assert body["canary_avg_latency_ms"] == 123.5

    def test_canary_status_redis_error_fallback(self):
        """GET /canary/status returns zeros when Redis fails."""
        with patch("app.api.canary.get_sync_redis") as mock_get_redis:
            mock_get_redis.side_effect = ConnectionError("Redis down")

            client = TestClient(app)
            response = client.get("/canary/status")

            assert response.status_code == 200
            body = response.json()
            assert body["canary_processed_24h"] == 0
            assert body["canary_errors_24h"] == 0
