"""Learning schema migration -- creates constraints, indexes, and vector index
for the self-learning subsystem.

Runs once per deployment (guarded by Redis SETNX key). Idempotent Cypher
statements (IF NOT EXISTS) make it safe to re-run if the Redis key is lost.

Node types created:
- :ConversationOutcome   (id UNIQUE)
- :AgentError            (id UNIQUE, type INDEX)
- :LessonLearned         (id UNIQUE, confidence INDEX)
- :PromptVersion         (id UNIQUE)
- :ErrorPattern          (pattern_name UNIQUE)
- :ConversationEmbedding (model INDEX, vector VECTOR INDEX 768-dim cosine)
- :LeadSatisfaction      (no unique constraint -- each measurement distinct)
"""

from __future__ import annotations

import redis
import structlog
from neo4j import AsyncDriver

from app.config import settings

logger = structlog.get_logger()

MIGRATION_KEY = "migration:learning_schema:v1"

SCHEMA_STATEMENTS: list[str] = [
    # Unique constraints
    "CREATE CONSTRAINT conversation_outcome_id IF NOT EXISTS FOR (n:ConversationOutcome) REQUIRE n.id IS UNIQUE",
    "CREATE CONSTRAINT agent_error_id IF NOT EXISTS FOR (n:AgentError) REQUIRE n.id IS UNIQUE",
    "CREATE CONSTRAINT lesson_learned_id IF NOT EXISTS FOR (n:LessonLearned) REQUIRE n.id IS UNIQUE",
    "CREATE CONSTRAINT prompt_version_id IF NOT EXISTS FOR (n:PromptVersion) REQUIRE n.id IS UNIQUE",
    "CREATE CONSTRAINT error_pattern_name IF NOT EXISTS FOR (n:ErrorPattern) REQUIRE n.pattern_name IS UNIQUE",
    # Property indexes
    "CREATE INDEX agent_error_type IF NOT EXISTS FOR (n:AgentError) ON (n.type)",
    "CREATE INDEX lesson_confidence IF NOT EXISTS FOR (n:LessonLearned) ON (n.confidence)",
    "CREATE INDEX conversation_embedding_model IF NOT EXISTS FOR (n:ConversationEmbedding) ON (n.model)",
]

VECTOR_INDEX_STATEMENT = (
    "CREATE VECTOR INDEX conversation_embedding_vector IF NOT EXISTS "
    "FOR (n:ConversationEmbedding) ON (n.vector) "
    "OPTIONS {indexConfig: {`vector.dimensions`: 768, `vector.similarity_function`: 'cosine'}}"
)


async def run_learning_schema_migration(driver: AsyncDriver) -> None:
    """Create learning-subsystem schema constraints and indexes.

    Uses a Redis SETNX guard so only the first startup after deployment
    actually runs the DDL. Subsequent startups skip immediately.
    """
    r: redis.Redis | None = None
    try:
        r = redis.Redis.from_url(settings.redis_cache_url, decode_responses=True)
        was_set = r.set(MIGRATION_KEY, "1", nx=True)
        if not was_set:
            logger.info("learning_schema_migration.skipped", reason="already_run")
            return
    except Exception:
        logger.warning(
            "learning_schema_migration.redis_guard_failed",
            reason="proceeding_anyway",
        )
    finally:
        if r is not None:
            r.close()

    try:
        async with driver.session() as session:
            for stmt in SCHEMA_STATEMENTS:
                await session.run(stmt)
            # Vector index uses different syntax
            await session.run(VECTOR_INDEX_STATEMENT)

        logger.info(
            "learning_schema_migration.complete",
            constraints=5,
            indexes=3,
            vector_indexes=1,
        )
    except Exception:
        logger.exception("learning_schema_migration.error")
