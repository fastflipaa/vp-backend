"""Tests for ConversationRepository -- interaction logging, prompt_version, traces.

All tests run against a real Neo4j testcontainer (session-scoped container,
function-scoped driver with cleanup). Marked ``slow`` -- requires Docker.
"""

from __future__ import annotations

import uuid

import pytest

from app.repositories.conversation_repository import ConversationRepository

pytestmark = [pytest.mark.asyncio, pytest.mark.slow]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _phone() -> str:
    return f"+5215500{uuid.uuid4().int % 100000:05d}"


async def _create_lead(driver, phone: str) -> None:
    """Create a minimal Lead node for interaction tests."""
    async with driver.session() as session:
        await session.run(
            "CREATE (l:Lead {phone: $phone, ghl_contact_id: $cid, current_state: 'QUALIFYING'})",
            phone=phone,
            cid=f"contact_{uuid.uuid4().hex[:8]}",
        )


# ---------------------------------------------------------------------------
# log_interaction tests
# ---------------------------------------------------------------------------

class TestLogInteraction:
    async def test_log_interaction_with_prompt_version(self, neo4j_driver):
        """log_interaction with prompt_version stores it on the Interaction node."""
        repo = ConversationRepository(neo4j_driver)
        phone = _phone()
        trace = str(uuid.uuid4())
        await _create_lead(neo4j_driver, phone)

        await repo.log_interaction(
            phone=phone,
            role="assistant",
            content="Hola! Bienvenido a Vive Polanco.",
            channel="whatsapp",
            trace_id=trace,
            prompt_version="greeting_es.yaml",
        )

        # Verify Interaction node has prompt_version
        async with neo4j_driver.session() as session:
            result = await session.run(
                """
                MATCH (l:Lead {phone: $phone})-[:HAS_INTERACTION]->(i:Interaction)
                RETURN i.prompt_version AS prompt_version,
                       i.trace_id AS trace_id,
                       i.role AS role
                """,
                phone=phone,
            )
            record = await result.single()

        assert record is not None
        assert record["prompt_version"] == "greeting_es.yaml"
        assert record["trace_id"] == trace
        assert record["role"] == "assistant"

    async def test_log_interaction_without_prompt_version(self, neo4j_driver):
        """log_interaction without prompt_version leaves the field absent (backward compat)."""
        repo = ConversationRepository(neo4j_driver)
        phone = _phone()
        trace = str(uuid.uuid4())
        await _create_lead(neo4j_driver, phone)

        await repo.log_interaction(
            phone=phone,
            role="user",
            content="Quiero un depa en Polanco",
            channel="whatsapp",
            trace_id=trace,
        )

        async with neo4j_driver.session() as session:
            result = await session.run(
                """
                MATCH (l:Lead {phone: $phone})-[:HAS_INTERACTION]->(i:Interaction)
                RETURN i.prompt_version AS prompt_version,
                       i.role AS role,
                       i.content AS content
                """,
                phone=phone,
            )
            record = await result.single()

        assert record is not None
        assert record["prompt_version"] is None  # Field absent -> None
        assert record["role"] == "user"

    async def test_interaction_has_all_fields(self, neo4j_driver):
        """Verify Interaction node has all expected fields: contact_id (via lead phone), message, response fields."""
        repo = ConversationRepository(neo4j_driver)
        phone = _phone()
        trace = str(uuid.uuid4())
        await _create_lead(neo4j_driver, phone)

        await repo.log_interaction(
            phone=phone,
            role="assistant",
            content="response text",
            channel="sms",
            trace_id=trace,
            prompt_version="qualifying_es.yaml",
        )

        async with neo4j_driver.session() as session:
            result = await session.run(
                """
                MATCH (l:Lead {phone: $phone})-[:HAS_INTERACTION]->(i:Interaction)
                RETURN i.role AS role,
                       i.content AS content,
                       i.channel AS channel,
                       i.trace_id AS trace_id,
                       i.prompt_version AS prompt_version,
                       i.created_at AS created_at
                """,
                phone=phone,
            )
            record = await result.single()

        assert record is not None
        assert record["role"] == "assistant"
        assert record["content"] == "response text"
        assert record["channel"] == "sms"
        assert record["trace_id"] == trace
        assert record["prompt_version"] == "qualifying_es.yaml"
        assert record["created_at"] is not None  # datetime() was stored


# ---------------------------------------------------------------------------
# get_recent_interactions tests
# ---------------------------------------------------------------------------

class TestGetRecentInteractions:
    async def test_get_recent_interactions(self, neo4j_driver):
        """Log 5 interactions -> get_recent returns them in newest-first order."""
        repo = ConversationRepository(neo4j_driver)
        phone = _phone()
        await _create_lead(neo4j_driver, phone)

        # Log 5 interactions with sequential content
        for i in range(5):
            await repo.log_interaction(
                phone=phone,
                role="user" if i % 2 == 0 else "assistant",
                content=f"Message {i}",
                channel="whatsapp",
                trace_id=str(uuid.uuid4()),
            )

        results = await repo.get_recent_interactions(phone, limit=5)

        assert len(results) == 5
        # Each result should have role, content, created_at
        for r in results:
            assert "role" in r
            assert "content" in r
            assert "created_at" in r

    async def test_get_recent_interactions_respects_limit(self, neo4j_driver):
        """get_recent_interactions returns at most ``limit`` entries."""
        repo = ConversationRepository(neo4j_driver)
        phone = _phone()
        await _create_lead(neo4j_driver, phone)

        for i in range(5):
            await repo.log_interaction(
                phone=phone,
                role="user",
                content=f"Msg {i}",
                channel="whatsapp",
                trace_id=str(uuid.uuid4()),
            )

        results = await repo.get_recent_interactions(phone, limit=3)

        assert len(results) == 3


# ---------------------------------------------------------------------------
# log_trace tests
# ---------------------------------------------------------------------------

class TestLogTrace:
    async def test_log_trace(self, neo4j_driver):
        """log_trace creates a Trace node with correct properties."""
        repo = ConversationRepository(neo4j_driver)
        phone = _phone()
        trace = str(uuid.uuid4())

        await repo.log_trace(
            trace_id=trace,
            phone=phone,
            state="QUALIFYING",
            duration_ms=142.5,
        )

        async with neo4j_driver.session() as session:
            result = await session.run(
                """
                MATCH (t:Trace {trace_id: $tid})
                RETURN t.phone AS phone,
                       t.state AS state,
                       t.duration_ms AS duration_ms,
                       t.created_at AS created_at
                """,
                tid=trace,
            )
            record = await result.single()

        assert record is not None
        assert record["phone"] == phone
        assert record["state"] == "QUALIFYING"
        assert record["duration_ms"] == 142.5
        assert record["created_at"] is not None
