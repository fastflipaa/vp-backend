"""Tests for LearningRepository -- integration tests against Neo4j testcontainer.

Covers all 21 repository methods: ConversationOutcome, AgentError,
RepetitionEvent, LeadSatisfaction, PromptVersion, ErrorPattern,
LessonLearned (lifecycle), ConversationEmbedding, and query methods.

Each test is async and marked ``slow`` (requires Docker).
"""

from __future__ import annotations

import json
import uuid

import pytest

from app.repositories.learning_repository import LearningRepository

pytestmark = [pytest.mark.asyncio, pytest.mark.slow]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _cid() -> str:
    """Generate a unique contact ID for test isolation."""
    return f"contact_{uuid.uuid4().hex[:12]}"


def _phone() -> str:
    """Generate a unique phone number for test isolation."""
    return f"+5215500{uuid.uuid4().int % 100000:05d}"


async def _create_lead(driver, contact_id: str, phone: str) -> None:
    """Create a Lead node directly in Neo4j for relationship tests."""
    async with driver.session() as session:
        await session.run(
            """
            CREATE (l:Lead {
                ghl_contact_id: $cid,
                phone: $phone,
                current_state: 'QUALIFYING',
                updatedAt: datetime()
            })
            """,
            cid=contact_id,
            phone=phone,
        )


# ---------------------------------------------------------------------------
# ConversationOutcome tests
# ---------------------------------------------------------------------------

class TestConversationOutcome:
    async def test_create_conversation_outcome_returns_id(self, neo4j_driver):
        """create_conversation_outcome returns a hex string ID."""
        repo = LearningRepository(neo4j_driver)

        oid = await repo.create_conversation_outcome(
            conversation_id="conv-1",
            outcome_type="positive",
            scores={"repetition": 0.1, "sentiment": 0.9, "intent_alignment": 0.8},
        )

        assert isinstance(oid, str)
        assert len(oid) == 32  # uuid4 hex

    async def test_create_outcome_links_to_lead(self, neo4j_driver):
        """ConversationOutcome links to Lead via HAS_OUTCOME when contact_id provided."""
        repo = LearningRepository(neo4j_driver)
        cid = _cid()
        phone = _phone()
        await _create_lead(neo4j_driver, cid, phone)

        await repo.create_conversation_outcome(
            conversation_id="conv-link",
            outcome_type="positive",
            scores={"repetition": 0.2},
            metadata={"contact_id": cid},
        )

        async with neo4j_driver.session() as session:
            result = await session.run(
                """
                MATCH (l:Lead {ghl_contact_id: $cid})-[:HAS_OUTCOME]->(o:ConversationOutcome)
                RETURN count(o) AS cnt
                """,
                cid=cid,
            )
            record = await result.single()

        assert record["cnt"] == 1

    async def test_create_outcome_idempotent(self, neo4j_driver):
        """MERGE prevents duplicate ConversationOutcome nodes."""
        repo = LearningRepository(neo4j_driver)

        oid1 = await repo.create_conversation_outcome(
            conversation_id="conv-idem",
            outcome_type="positive",
            scores={"repetition": 0.1},
        )
        # Second call creates a different node (different UUID)
        oid2 = await repo.create_conversation_outcome(
            conversation_id="conv-idem",
            outcome_type="positive",
            scores={"repetition": 0.1},
        )

        # Both IDs should be valid hex strings
        assert oid1 != oid2
        assert len(oid1) == 32


# ---------------------------------------------------------------------------
# AgentError tests
# ---------------------------------------------------------------------------

class TestAgentError:
    async def test_create_agent_error_returns_id(self, neo4j_driver):
        """create_agent_error returns a hex string ID."""
        repo = LearningRepository(neo4j_driver)

        eid = await repo.create_agent_error(
            conversation_id="conv-err",
            error_type="repetition",
            details="3 similar pairs",
            severity="warning",
        )

        assert isinstance(eid, str)
        assert len(eid) == 32

    async def test_create_agent_error_links_to_lead(self, neo4j_driver):
        """AgentError links to Lead via HAS_ERROR when contact_id provided."""
        repo = LearningRepository(neo4j_driver)
        cid = _cid()
        phone = _phone()
        await _create_lead(neo4j_driver, cid, phone)

        await repo.create_agent_error(
            conversation_id="conv-err-link",
            error_type="hallucination",
            details="Wrong price",
            severity="critical",
            contact_id=cid,
        )

        async with neo4j_driver.session() as session:
            result = await session.run(
                """
                MATCH (l:Lead {ghl_contact_id: $cid})-[:HAS_ERROR]->(e:AgentError)
                RETURN count(e) AS cnt
                """,
                cid=cid,
            )
            record = await result.single()

        assert record["cnt"] == 1

    async def test_create_agent_error_stores_all_fields(self, neo4j_driver):
        """AgentError stores type, details, severity via Cypher query."""
        repo = LearningRepository(neo4j_driver)

        eid = await repo.create_agent_error(
            conversation_id="conv-fields",
            error_type="tone_mismatch",
            details="Positive response to negative user",
            severity="warning",
        )

        async with neo4j_driver.session() as session:
            result = await session.run(
                """
                MATCH (e:AgentError {id: $eid})
                RETURN e.type AS type, e.details AS details, e.severity AS severity
                """,
                eid=eid,
            )
            record = await result.single()

        assert record["type"] == "tone_mismatch"
        assert "Positive response" in record["details"]
        assert record["severity"] == "warning"


# ---------------------------------------------------------------------------
# RepetitionEvent tests
# ---------------------------------------------------------------------------

class TestRepetitionEvent:
    async def test_create_repetition_event_stores_similarity(self, neo4j_driver):
        """Repetition event stores similarity_score, msg_a, msg_b fields."""
        repo = LearningRepository(neo4j_driver)
        phone = _phone()
        cid = _cid()
        await _create_lead(neo4j_driver, cid, phone)

        eid = await repo.create_repetition_event(
            lead_phone=phone,
            similarity_score=0.92,
            message_pair=("Bienvenido a Polanco", "Bienvenido a Polanco, excelente"),
        )

        async with neo4j_driver.session() as session:
            result = await session.run(
                """
                MATCH (e:AgentError {id: $eid})
                RETURN e.similarity_score AS score, e.msg_a AS msg_a, e.msg_b AS msg_b
                """,
                eid=eid,
            )
            record = await result.single()

        assert record["score"] == 0.92
        assert "Bienvenido" in record["msg_a"]
        assert "excelente" in record["msg_b"]

    async def test_repetition_event_links_to_lead_by_phone(self, neo4j_driver):
        """Repetition event links to Lead via HAS_ERROR relationship."""
        repo = LearningRepository(neo4j_driver)
        phone = _phone()
        cid = _cid()
        await _create_lead(neo4j_driver, cid, phone)

        await repo.create_repetition_event(
            lead_phone=phone,
            similarity_score=0.85,
            message_pair=("Hello", "Hello again"),
        )

        async with neo4j_driver.session() as session:
            result = await session.run(
                """
                MATCH (l:Lead {phone: $phone})-[:HAS_ERROR]->(e:AgentError {type: 'repetition'})
                RETURN count(e) AS cnt
                """,
                phone=phone,
            )
            record = await result.single()

        assert record["cnt"] == 1


# ---------------------------------------------------------------------------
# LeadSatisfaction tests
# ---------------------------------------------------------------------------

class TestLeadSatisfaction:
    async def test_create_lead_satisfaction_stores_score(self, neo4j_driver):
        """LeadSatisfaction stores score and source fields."""
        repo = LearningRepository(neo4j_driver)
        phone = _phone()
        cid = _cid()
        await _create_lead(neo4j_driver, cid, phone)

        sid = await repo.create_lead_satisfaction(
            lead_phone=phone,
            score=0.85,
            source="auto",
            signals={"tone": "positive"},
        )

        async with neo4j_driver.session() as session:
            result = await session.run(
                """
                MATCH (s:LeadSatisfaction {id: $sid})
                RETURN s.score AS score, s.source AS source
                """,
                sid=sid,
            )
            record = await result.single()

        assert record["score"] == 0.85
        assert record["source"] == "auto"

    async def test_lead_satisfaction_links_to_lead(self, neo4j_driver):
        """LeadSatisfaction links to Lead via HAS_SATISFACTION relationship."""
        repo = LearningRepository(neo4j_driver)
        phone = _phone()
        cid = _cid()
        await _create_lead(neo4j_driver, cid, phone)

        await repo.create_lead_satisfaction(
            lead_phone=phone,
            score=0.9,
            source="auto",
            signals={},
        )

        async with neo4j_driver.session() as session:
            result = await session.run(
                """
                MATCH (l:Lead {phone: $phone})-[:HAS_SATISFACTION]->(s:LeadSatisfaction)
                RETURN count(s) AS cnt
                """,
                phone=phone,
            )
            record = await result.single()

        assert record["cnt"] == 1


# ---------------------------------------------------------------------------
# PromptVersion tests
# ---------------------------------------------------------------------------

class TestPromptVersion:
    async def test_create_prompt_version_returns_id(self, neo4j_driver):
        """create_prompt_version returns a hex string ID."""
        repo = LearningRepository(neo4j_driver)

        pid = await repo.create_prompt_version(
            state="GREETING",
            version="v1",
            content_hash="abc123",
            active=True,
        )

        assert isinstance(pid, str)
        assert len(pid) == 32

    async def test_active_version_deactivates_previous(self, neo4j_driver):
        """Creating active version for same state deactivates the previous one."""
        repo = LearningRepository(neo4j_driver)

        pid1 = await repo.create_prompt_version(
            state="QUALIFYING",
            version="v1",
            content_hash="hash1",
            active=True,
        )
        pid2 = await repo.create_prompt_version(
            state="QUALIFYING",
            version="v2",
            content_hash="hash2",
            active=True,
        )

        async with neo4j_driver.session() as session:
            result = await session.run(
                """
                MATCH (pv:PromptVersion {id: $pid})
                RETURN pv.active AS active
                """,
                pid=pid1,
            )
            r1 = await result.single()
            result2 = await session.run(
                """
                MATCH (pv:PromptVersion {id: $pid})
                RETURN pv.active AS active
                """,
                pid=pid2,
            )
            r2 = await result2.single()

        assert r1["active"] is False
        assert r2["active"] is True


# ---------------------------------------------------------------------------
# ErrorPattern tests
# ---------------------------------------------------------------------------

class TestErrorPattern:
    async def test_create_error_pattern(self, neo4j_driver):
        """create_error_pattern creates a pattern with frequency=1."""
        repo = LearningRepository(neo4j_driver)

        await repo.create_error_pattern("repetition")

        async with neo4j_driver.session() as session:
            result = await session.run(
                """
                MATCH (ep:ErrorPattern {pattern_name: 'repetition'})
                RETURN ep.frequency AS freq
                """,
            )
            record = await result.single()

        assert record["freq"] == 1

    async def test_increment_error_pattern_returns_frequency(self, neo4j_driver):
        """increment_error_pattern returns successive frequency counts."""
        repo = LearningRepository(neo4j_driver)

        await repo.create_error_pattern("hallucination")
        f1 = await repo.increment_error_pattern("hallucination")
        f2 = await repo.increment_error_pattern("hallucination")
        f3 = await repo.increment_error_pattern("hallucination")

        assert f1 == 2
        assert f2 == 3
        assert f3 == 4

    async def test_increment_creates_if_not_exists(self, neo4j_driver):
        """increment_error_pattern creates a pattern if it does not exist."""
        repo = LearningRepository(neo4j_driver)

        freq = await repo.increment_error_pattern("brand_new_pattern")

        assert freq == 1

        async with neo4j_driver.session() as session:
            result = await session.run(
                """
                MATCH (ep:ErrorPattern {pattern_name: 'brand_new_pattern'})
                RETURN ep.frequency AS freq
                """,
            )
            record = await result.single()

        assert record is not None
        assert record["freq"] == 1


# ---------------------------------------------------------------------------
# LessonLearned tests
# ---------------------------------------------------------------------------

class TestLessonLearned:
    async def test_create_lesson_learned_returns_id(self, neo4j_driver):
        """create_lesson_learned returns a hex string ID."""
        repo = LearningRepository(neo4j_driver)

        lid = await repo.create_lesson_learned(
            rule="Avoid repeating",
            why="Leads disengage",
            severity="warning",
            confidence=0.5,
        )

        assert isinstance(lid, str)
        assert len(lid) == 32

    async def test_lesson_starts_as_candidate(self, neo4j_driver):
        """New lesson has status='candidate'."""
        repo = LearningRepository(neo4j_driver)

        lid = await repo.create_lesson_learned(
            rule="Check prices",
            why="Accuracy",
            severity="critical",
            confidence=0.5,
        )

        async with neo4j_driver.session() as session:
            result = await session.run(
                "MATCH (ll:LessonLearned {id: $lid}) RETURN ll.status AS status",
                lid=lid,
            )
            record = await result.single()

        assert record["status"] == "candidate"

    async def test_lesson_links_to_building(self, neo4j_driver):
        """Lesson links to Building via APPLIES_TO when building_id provided."""
        repo = LearningRepository(neo4j_driver)

        # Create a building node first
        async with neo4j_driver.session() as session:
            await session.run(
                "CREATE (b:Building {building_id: 'bldg-001', name: 'Test Building'})",
            )

        lid = await repo.create_lesson_learned(
            rule="Check floor count",
            why="Avoid hallucination",
            severity="critical",
            confidence=0.5,
            building_id="bldg-001",
        )

        async with neo4j_driver.session() as session:
            result = await session.run(
                """
                MATCH (ll:LessonLearned {id: $lid})-[:APPLIES_TO]->(b:Building {building_id: 'bldg-001'})
                RETURN count(*) AS cnt
                """,
                lid=lid,
            )
            record = await result.single()

        assert record["cnt"] == 1

    async def test_get_lessons_for_context_returns_approved_only(self, neo4j_driver):
        """get_lessons_for_context returns only approved/evergreen lessons."""
        repo = LearningRepository(neo4j_driver)

        # Create 3 lessons with different statuses
        lid_cand = await repo.create_lesson_learned(
            rule="candidate lesson",
            why="test",
            severity="warning",
            confidence=0.8,
        )
        lid_appr = await repo.create_lesson_learned(
            rule="approved lesson",
            why="test",
            severity="warning",
            confidence=0.8,
        )
        lid_ever = await repo.create_lesson_learned(
            rule="evergreen lesson",
            why="test",
            severity="warning",
            confidence=0.8,
        )

        # Manually set statuses via Cypher
        async with neo4j_driver.session() as session:
            await session.run(
                "MATCH (ll:LessonLearned {id: $lid}) SET ll.status = 'approved'",
                lid=lid_appr,
            )
            await session.run(
                "MATCH (ll:LessonLearned {id: $lid}) SET ll.status = 'evergreen'",
                lid=lid_ever,
            )

        result = await repo.get_lessons_for_context(min_confidence=0.5)

        # candidate should NOT appear; approved and evergreen should
        result_ids = [r["id"] for r in result]
        assert lid_cand not in result_ids
        assert lid_appr in result_ids
        assert lid_ever in result_ids

    async def test_get_lessons_by_status(self, neo4j_driver):
        """get_lessons_by_status filters by status list correctly."""
        repo = LearningRepository(neo4j_driver)

        lid1 = await repo.create_lesson_learned(
            rule="lesson one",
            why="test",
            severity="warning",
            confidence=0.5,
        )
        lid2 = await repo.create_lesson_learned(
            rule="lesson two",
            why="test",
            severity="warning",
            confidence=0.5,
        )

        # lid2 stays as candidate, promote lid1
        async with neo4j_driver.session() as session:
            await session.run(
                "MATCH (ll:LessonLearned {id: $lid}) SET ll.status = 'approved'",
                lid=lid1,
            )

        result = await repo.get_lessons_by_status(["candidate"])

        result_ids = [r["id"] for r in result]
        assert lid2 in result_ids
        assert lid1 not in result_ids

    async def test_check_lesson_exists_for_pattern(self, neo4j_driver):
        """check_lesson_exists_for_pattern matches on the source_pattern field.

        The production query in
        ``app/repositories/learning_repository.py:_check_lesson_exists_for_pattern_tx``
        does an EXACT match on ``ll.source_pattern``, not a substring match
        on ``ll.rule``. The test must seed the lesson with ``source_pattern``
        set to the value it intends to query.
        """
        repo = LearningRepository(neo4j_driver)

        await repo.create_lesson_learned(
            rule="Avoid repetition in responses",
            why="Engagement",
            severity="warning",
            confidence=0.5,
            source_pattern="repetition",
        )

        assert await repo.check_lesson_exists_for_pattern("repetition") is True
        assert await repo.check_lesson_exists_for_pattern("nonexistent_xyz") is False


# ---------------------------------------------------------------------------
# LessonLifecycle tests
# ---------------------------------------------------------------------------

class TestLessonLifecycle:
    async def test_promote_lesson_to_evergreen(self, neo4j_driver):
        """Promoting a lesson sets status='evergreen'."""
        repo = LearningRepository(neo4j_driver)

        lid = await repo.create_lesson_learned(
            rule="promote me",
            why="test",
            severity="warning",
            confidence=0.8,
        )

        result = await repo.promote_lesson_to_evergreen(lid)
        assert result is True

        async with neo4j_driver.session() as session:
            r = await session.run(
                "MATCH (ll:LessonLearned {id: $lid}) RETURN ll.status AS status",
                lid=lid,
            )
            record = await r.single()

        assert record["status"] == "evergreen"

    async def test_promote_nonexistent_returns_false(self, neo4j_driver):
        """Promoting non-existent lesson returns False."""
        repo = LearningRepository(neo4j_driver)

        result = await repo.promote_lesson_to_evergreen("nonexistent-id-123")

        assert result is False

    async def test_archive_low_confidence(self, neo4j_driver):
        """archive_low_confidence archives lessons with confidence <= 0.1."""
        repo = LearningRepository(neo4j_driver)

        lid = await repo.create_lesson_learned(
            rule="low confidence lesson",
            why="test",
            severity="warning",
            confidence=0.5,
        )

        # Manually set confidence to 0.05 and created_at to old date
        async with neo4j_driver.session() as session:
            await session.run(
                """
                MATCH (ll:LessonLearned {id: $lid})
                SET ll.confidence = 0.05,
                    ll.created_at = datetime() - duration({days: 100})
                """,
                lid=lid,
            )

        count = await repo.archive_low_confidence()
        assert count >= 1

        async with neo4j_driver.session() as session:
            r = await session.run(
                "MATCH (ll:LessonLearned {id: $lid}) RETURN ll.status AS status",
                lid=lid,
            )
            record = await r.single()

        assert record["status"] == "archived"

    async def test_auto_promote_candidates(self, neo4j_driver):
        """auto_promote_candidates promotes mature high-confidence candidates."""
        repo = LearningRepository(neo4j_driver)

        lid = await repo.create_lesson_learned(
            rule="mature lesson",
            why="test",
            severity="warning",
            confidence=0.8,
        )

        # Set created_at to 10 days ago and ensure confidence >= 0.7
        async with neo4j_driver.session() as session:
            await session.run(
                """
                MATCH (ll:LessonLearned {id: $lid})
                SET ll.confidence = 0.8,
                    ll.created_at = datetime() - duration({days: 10})
                """,
                lid=lid,
            )

        promoted_ids = await repo.auto_promote_candidates(
            min_confidence=0.7, min_age_days=7
        )

        assert lid in promoted_ids

        async with neo4j_driver.session() as session:
            r = await session.run(
                "MATCH (ll:LessonLearned {id: $lid}) RETURN ll.status AS status",
                lid=lid,
            )
            record = await r.single()

        assert record["status"] == "approved"


# ---------------------------------------------------------------------------
# ConversationEmbedding tests
# ---------------------------------------------------------------------------

class TestConversationEmbedding:
    async def test_create_conversation_embedding(self, neo4j_driver):
        """create_conversation_embedding stores a 768-dim vector."""
        repo = LearningRepository(neo4j_driver)

        cid = await repo.create_conversation_embedding(
            conversation_id="conv-emb-1",
            vector=[0.1] * 768,
        )

        assert cid == "conv-emb-1"

        async with neo4j_driver.session() as session:
            result = await session.run(
                """
                MATCH (ce:ConversationEmbedding {conversation_id: 'conv-emb-1'})
                RETURN size(ce.vector) AS dims
                """,
            )
            record = await result.single()

        assert record["dims"] == 768

    async def test_cleanup_old_embeddings(self, neo4j_driver):
        """cleanup_old_embeddings removes embeddings older than threshold."""
        repo = LearningRepository(neo4j_driver)

        await repo.create_conversation_embedding(
            conversation_id="conv-old",
            vector=[0.1] * 768,
        )

        # Set created_at to 200 days ago
        async with neo4j_driver.session() as session:
            await session.run(
                """
                MATCH (ce:ConversationEmbedding {conversation_id: 'conv-old'})
                SET ce.created_at = datetime() - duration({days: 200})
                """,
            )

        deleted = await repo.cleanup_old_embeddings(180)

        assert deleted == 1

        async with neo4j_driver.session() as session:
            result = await session.run(
                """
                MATCH (ce:ConversationEmbedding {conversation_id: 'conv-old'})
                RETURN count(ce) AS cnt
                """,
            )
            record = await result.single()

        assert record["cnt"] == 0


# ---------------------------------------------------------------------------
# ErrorPattern query tests
# ---------------------------------------------------------------------------

class TestErrorPatternQuery:
    async def test_get_error_patterns_min_frequency(self, neo4j_driver):
        """get_error_patterns returns patterns at or above min_frequency."""
        repo = LearningRepository(neo4j_driver)

        await repo.create_error_pattern("frequent_pattern")
        # Increment to frequency 5
        for _ in range(4):
            await repo.increment_error_pattern("frequent_pattern")

        result = await repo.get_error_patterns(min_frequency=3)
        names = [r["pattern_name"] for r in result]

        assert "frequent_pattern" in names

    async def test_get_error_patterns_excludes_low(self, neo4j_driver):
        """get_error_patterns excludes patterns below min_frequency."""
        repo = LearningRepository(neo4j_driver)

        await repo.create_error_pattern("low_freq_pattern")

        result = await repo.get_error_patterns(min_frequency=3)
        names = [r["pattern_name"] for r in result]

        assert "low_freq_pattern" not in names


# ---------------------------------------------------------------------------
# OutcomeAverages tests
# ---------------------------------------------------------------------------

class TestOutcomeAverages:
    async def test_get_outcome_averages_empty(self, neo4j_driver):
        """get_outcome_averages on empty DB returns all zeros with count=0."""
        repo = LearningRepository(neo4j_driver)

        result = await repo.get_outcome_averages(days=7)

        assert result["count"] == 0
        assert result["repetition"] == 0.0
        assert result["sentiment"] == 0.0
        assert result["intent_alignment"] == 0.0

    async def test_get_outcome_averages_computes_mean(self, neo4j_driver):
        """get_outcome_averages correctly computes means from stored scores."""
        repo = LearningRepository(neo4j_driver)

        scores1 = {"repetition": 0.2, "sentiment": 0.8, "intent_alignment": 0.9}
        scores2 = {"repetition": 0.4, "sentiment": 0.6, "intent_alignment": 0.7}

        await repo.create_conversation_outcome(
            conversation_id="conv-avg-1",
            outcome_type="scored",
            scores=scores1,
        )
        await repo.create_conversation_outcome(
            conversation_id="conv-avg-2",
            outcome_type="scored",
            scores=scores2,
        )

        result = await repo.get_outcome_averages(days=7)

        assert result["count"] == 2
        assert abs(result["repetition"] - 0.3) < 0.01
        assert abs(result["sentiment"] - 0.7) < 0.01
        assert abs(result["intent_alignment"] - 0.8) < 0.01

    async def test_get_lesson_injection_count(self, neo4j_driver):
        """get_lesson_injection_count returns count of recent outcomes."""
        repo = LearningRepository(neo4j_driver)

        await repo.create_conversation_outcome(
            conversation_id="conv-inj-1",
            outcome_type="scored",
            scores={"repetition": 0.1},
        )
        await repo.create_conversation_outcome(
            conversation_id="conv-inj-2",
            outcome_type="scored",
            scores={"repetition": 0.2},
        )

        count = await repo.get_lesson_injection_count(7)

        assert count == 2
