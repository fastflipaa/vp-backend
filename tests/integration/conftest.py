"""Integration test conftest -- exposes Neo4j testcontainer fixtures.

Re-defines the ``neo4j_container`` and ``neo4j_driver`` fixtures here
so they are available to tests under ``tests/integration/``. The same
fixtures live in ``tests/repositories/conftest.py`` for the repository
tests; pytest fixture discovery is scoped per directory tree, so we
need to either move them up to ``tests/conftest.py`` (which would
spread testcontainers dependency to ALL tests) or duplicate them
here. Duplication keeps the slow/testcontainers dependency contained.
"""

from __future__ import annotations

import pytest
import pytest_asyncio

# Guard import so tests are collectable even when Docker/testcontainers unavailable
try:
    from testcontainers.neo4j import Neo4jContainer

    _HAS_TESTCONTAINERS = True
except Exception:
    _HAS_TESTCONTAINERS = False


@pytest.fixture(scope="session")
def neo4j_container():
    """Start a Neo4j 5-community container for the entire test session.

    Yields the container object. Automatically stopped on session teardown.
    Skipped if Docker is not available (e.g. running tests outside CI).
    """
    if not _HAS_TESTCONTAINERS:
        pytest.skip("testcontainers not installed or Docker unavailable")

    try:
        container = Neo4jContainer("neo4j:5-community")
        container.start()
    except Exception as exc:
        pytest.skip(f"Cannot start Neo4j testcontainer (Docker not running?): {exc}")

    yield container

    container.stop()


@pytest_asyncio.fixture()
async def neo4j_driver(neo4j_container):
    """Per-test async Neo4j driver pointing at the testcontainer.

    After yield, runs ``MATCH (n) DETACH DELETE n`` to reset the DB for
    the next test, then closes the driver.
    """
    from neo4j import AsyncGraphDatabase

    uri = neo4j_container.get_connection_url()
    driver = AsyncGraphDatabase.driver(
        uri,
        auth=(neo4j_container.username, neo4j_container.password),
    )

    yield driver

    # Cleanup: remove ALL nodes and relationships
    async with driver.session() as session:
        await session.run("MATCH (n) DETACH DELETE n")

    await driver.close()
