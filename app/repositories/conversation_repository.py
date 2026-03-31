"""Conversation repository for async Neo4j operations.

Handles interaction logging (inbound + assistant messages), recent
interaction retrieval, and pipeline trace logging.

Each method creates a fresh ``session()`` per operation.
"""

from __future__ import annotations

import structlog
from neo4j import AsyncDriver

logger = structlog.get_logger()


class ConversationRepository:
    """Async repository for :Interaction and :Trace node operations."""

    def __init__(self, driver: AsyncDriver) -> None:
        self._driver = driver

    # --- Interaction logging ---

    async def log_interaction(
        self,
        phone: str,
        role: str,
        content: str,
        channel: str,
        trace_id: str,
        prompt_version: str | None = None,
    ) -> None:
        """Log an interaction (inbound or assistant) linked to a Lead.

        Args:
            phone: Lead phone number.
            role: ``"user"`` for inbound, ``"assistant"`` for AI response.
            content: Message text.
            channel: Delivery channel (e.g. ``"sms"``, ``"whatsapp"``).
            trace_id: Pipeline trace ID for correlation.
            prompt_version: Optional prompt file version (e.g. ``"qualifying_es.yaml"``).
        """
        async with self._driver.session() as session:
            await session.execute_write(
                self._log_interaction_tx, phone, role, content, channel, trace_id, prompt_version
            )
            logger.info(
                "interaction_logged",
                phone=phone[-4:],
                role=role,
                channel=channel,
                trace_id=trace_id,
                prompt_version=prompt_version,
            )

    @staticmethod
    async def _log_interaction_tx(
        tx, phone: str, role: str, content: str, channel: str, trace_id: str,
        prompt_version: str | None = None,
    ) -> None:
        # Build property map -- prompt_version is optional for backward compat
        props = {
            "phone": phone,
            "role": role,
            "content": content,
            "channel": channel,
            "trace_id": trace_id,
        }
        if prompt_version is not None:
            # Include prompt_version in the Interaction node
            await tx.run(
                """
                MATCH (l:Lead {phone: $phone})
                CREATE (i:Interaction {
                    role: $role,
                    content: $content,
                    channel: $channel,
                    trace_id: $trace_id,
                    prompt_version: $prompt_version,
                    created_at: datetime()
                })
                MERGE (l)-[:HAS_INTERACTION]->(i)
                """,
                **props,
                prompt_version=prompt_version,
            )
        else:
            await tx.run(
                """
                MATCH (l:Lead {phone: $phone})
                CREATE (i:Interaction {
                    role: $role,
                    content: $content,
                    channel: $channel,
                    trace_id: $trace_id,
                    created_at: datetime()
                })
                MERGE (l)-[:HAS_INTERACTION]->(i)
                """,
                **props,
            )

    # --- Interaction retrieval ---

    async def get_recent_interactions(
        self, phone: str, limit: int = 5
    ) -> list[dict]:
        """Return the last N interactions for a lead, newest first.

        Each dict contains: role, content, created_at.
        """
        async with self._driver.session() as session:
            result = await session.execute_read(
                self._get_recent_interactions_tx, phone, limit
            )
            logger.debug(
                "recent_interactions_read",
                phone=phone[-4:],
                count=len(result),
            )
            return result

    @staticmethod
    async def _get_recent_interactions_tx(
        tx, phone: str, limit: int
    ) -> list[dict]:
        result = await tx.run(
            """
            MATCH (l:Lead {phone: $phone})-[:HAS_INTERACTION]->(i:Interaction)
            RETURN i.role AS role, i.content AS content, i.created_at AS created_at
            ORDER BY i.created_at DESC
            LIMIT $limit
            """,
            phone=phone,
            limit=limit,
        )
        return [dict(r) async for r in result]

    # --- Trace logging ---

    async def log_trace(
        self,
        trace_id: str,
        phone: str,
        state: str,
        duration_ms: float,
    ) -> None:
        """Log a pipeline trace node for observability.

        Each trace represents one full processing pipeline execution
        (gate check -> state machine -> Claude -> GHL delivery).
        """
        async with self._driver.session() as session:
            await session.execute_write(
                self._log_trace_tx, trace_id, phone, state, duration_ms
            )
            logger.info(
                "trace_logged",
                trace_id=trace_id,
                phone=phone[-4:],
                state=state,
                duration_ms=duration_ms,
            )

    @staticmethod
    async def _log_trace_tx(
        tx, trace_id: str, phone: str, state: str, duration_ms: float
    ) -> None:
        await tx.run(
            """
            CREATE (t:Trace {
                trace_id: $tid,
                phone: $phone,
                state: $state,
                duration_ms: $dur,
                created_at: datetime()
            })
            """,
            tid=trace_id,
            phone=phone,
            state=state,
            dur=duration_ms,
        )
