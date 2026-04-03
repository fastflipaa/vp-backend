"""Vive Polanco Backend - FastAPI application with health endpoints.

Imports settings at module level for fail-fast behavior:
if any required env var is missing, the process crashes immediately
with a clear pydantic ValidationError.
"""

from fastapi import FastAPI
from fastapi.responses import JSONResponse
from app.config import settings  # noqa: F401 -- fail-fast import
from app.logging_config import configure_logging
from app.api.webhooks import router as webhooks_router
from app.api.canary import router as canary_router
from app.routes.lessons import router as lessons_router

# Initialize structlog JSON logging before anything else
configure_logging()

app = FastAPI(title="Vive Polanco Backend", docs_url=None, redoc_url=None)
app.include_router(webhooks_router)
app.include_router(canary_router)
app.include_router(lessons_router)


@app.on_event("startup")
async def startup_learning_schema():
    """Run learning schema migration on first startup."""
    try:
        from app.repositories.base import get_driver
        from app.migrations.learning_schema import run_learning_schema_migration

        driver = await get_driver()
        await run_learning_schema_migration(driver)
    except Exception:
        import structlog

        structlog.get_logger().exception("learning_schema_migration.startup_failed")


@app.get("/health")
async def health():
    """Liveness probe -- no external dependencies."""
    return {"status": "healthy"}


@app.get("/ready")
async def ready():
    """Readiness probe -- checks Redis and Neo4j reachability."""
    checks = {}
    all_ok = True

    # Redis check
    try:
        import redis.asyncio as aioredis

        r = aioredis.from_url(settings.REDIS_URL, socket_connect_timeout=2)
        await r.ping()
        await r.aclose()
        checks["redis"] = "ok"
    except Exception as e:
        checks["redis"] = f"error: {e}"
        all_ok = False

    # Neo4j check
    try:
        from neo4j import AsyncGraphDatabase

        driver = AsyncGraphDatabase.driver(
            settings.NEO4J_URI,
            auth=("neo4j", settings.NEO4J_PASSWORD),
        )
        async with driver.session() as session:
            await session.run("RETURN 1")
        await driver.close()
        checks["neo4j"] = "ok"
    except Exception as e:
        checks["neo4j"] = f"error: {e}"
        all_ok = False

    status_code = 200 if all_ok else 503
    return JSONResponse(
        status_code=status_code,
        content={"status": "ready" if all_ok else "not_ready", "checks": checks},
    )
