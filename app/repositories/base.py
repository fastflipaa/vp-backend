"""Async Neo4j driver singleton with connection pooling.

Provides a module-level async driver factory. The driver is created once
on first call and reused for the lifetime of the process.

IMPORTANT: This module uses ONLY async Neo4j APIs. The sync driver in
``app/shadow/neo4j_store.py`` is a separate concern for shadow mode.
Do NOT mix sync and async drivers in the same codebase path.
"""

from __future__ import annotations

import structlog
from neo4j import AsyncGraphDatabase, AsyncDriver

from app.config import settings

logger = structlog.get_logger()

_driver: AsyncDriver | None = None


async def get_driver() -> AsyncDriver:
    """Return the module-level async Neo4j driver singleton.

    Creates the driver on first call with:
    - 200 max connections (sized for peak Celery concurrency)
    - 1h max connection lifetime (matches Neo4j server default)
    - 30s acquisition timeout (fail fast if pool exhausted)
    """
    global _driver
    if _driver is None:
        _driver = AsyncGraphDatabase.driver(
            settings.NEO4J_URI,
            auth=("neo4j", settings.NEO4J_PASSWORD),
            max_connection_pool_size=200,
            max_connection_lifetime=3600,
            connection_acquisition_timeout=30.0,
        )
        logger.info(
            "neo4j_async_driver_created",
            uri=settings.NEO4J_URI,
            pool_size=200,
        )
    return _driver


async def close_driver() -> None:
    """Close the async Neo4j driver and release all connections.

    Call this during application shutdown (e.g. FastAPI lifespan).
    """
    global _driver
    if _driver is not None:
        await _driver.close()
        _driver = None
        logger.info("neo4j_async_driver_closed")
