"""Shared dependencies for FastAPI and Celery workers.

Provides Redis connection pools (sync for Celery, async for FastAPI).
Uses db2 (cache) for all gate operations (dedup, locks, rate limits).
"""

from __future__ import annotations

import redis
import redis.asyncio as aioredis

from app.config import settings

# --- Sync Redis (for Celery workers) ---

_sync_pool: redis.ConnectionPool | None = None


def get_sync_redis() -> redis.Redis:
    """Return a sync Redis client backed by a shared connection pool.

    Uses db2 (cache) for gate operations. Pool is created once at module
    level and reused across all calls.
    """
    global _sync_pool
    if _sync_pool is None:
        _sync_pool = redis.ConnectionPool.from_url(
            settings.redis_cache_url,
            max_connections=20,
            decode_responses=True,
        )
    return redis.Redis(connection_pool=_sync_pool)


# --- Async Redis (for FastAPI endpoints) ---

_async_pool: aioredis.ConnectionPool | None = None


def get_async_redis() -> aioredis.Redis:
    """Return an async Redis client backed by a shared connection pool.

    Uses db2 (cache) for gate operations.
    """
    global _async_pool
    if _async_pool is None:
        _async_pool = aioredis.ConnectionPool.from_url(
            settings.redis_cache_url,
            max_connections=20,
            decode_responses=True,
        )
    return aioredis.Redis(connection_pool=_async_pool)
