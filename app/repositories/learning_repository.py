"""Learning repository for async Neo4j operations.

Handles CRUD for all self-learning node types: ConversationOutcome,
AgentError, LessonLearned, PromptVersion, ErrorPattern,
ConversationEmbedding, and LeadSatisfaction.

All writes use MERGE for idempotency. NEVER modifies Building or Lead
core data -- only creates learning nodes and relationships FROM
leads/conversations TO learning nodes.

Each method creates a fresh ``session()`` per operation.
"""

from __future__ import annotations

import json
import uuid
from typing import Any

import structlog
from neo4j import AsyncDriver

logger = structlog.get_logger()


class LearningRepository:
    """Async repository for self-learning node operations in Neo4j."""

    def __init__(self, driver: AsyncDriver) -> None:
        self._driver = driver

    @staticmethod
    def _new_id() -> str:
        return uuid.uuid4().hex

    # ------------------------------------------------------------------
    # ConversationOutcome
    # ------------------------------------------------------------------

    async def create_conversation_outcome(
        self,
        conversation_id: str,
        outcome_type: str,
        scores: dict,
        metadata: dict | None = None,
    ) -> str:
        """Create a ConversationOutcome node and optionally link to Lead."""
        oid = self._new_id()
        contact_id = (metadata or {}).get("contact_id")
        async with self._driver.session() as session:
            await session.execute_write(
                self._create_conversation_outcome_tx,
                oid,
                conversation_id,
                outcome_type,
                json.dumps(scores),
                json.dumps(metadata) if metadata else "{}",
                contact_id,
            )
        logger.debug(
            "learning.conversation_outcome_created",
            id=oid,
            conversation_id=conversation_id,
        )
        return oid

    @staticmethod
    async def _create_conversation_outcome_tx(
        tx,
        oid: str,
        conversation_id: str,
        outcome_type: str,
        scores_json: str,
        metadata_json: str,
        contact_id: str | None,
    ) -> None:
        await tx.run(
            """
            MERGE (o:ConversationOutcome {id: $oid})
            SET o.conversation_id = $cid,
                o.outcome_type = $otype,
                o.scores = $scores,
                o.metadata = $meta,
                o.created_at = datetime()
            """,
            oid=oid,
            cid=conversation_id,
            otype=outcome_type,
            scores=scores_json,
            meta=metadata_json,
        )
        if contact_id:
            await tx.run(
                """
                MATCH (l:Lead {ghl_contact_id: $contact_id})
                MATCH (o:ConversationOutcome {id: $oid})
                MERGE (l)-[:HAS_OUTCOME]->(o)
                """,
                contact_id=contact_id,
                oid=oid,
            )

    # ------------------------------------------------------------------
    # AgentError
    # ------------------------------------------------------------------

    async def create_agent_error(
        self,
        conversation_id: str,
        error_type: str,
        details: str,
        severity: str,
        contact_id: str | None = None,
    ) -> str:
        """Create an AgentError node and optionally link to Lead."""
        eid = self._new_id()
        async with self._driver.session() as session:
            await session.execute_write(
                self._create_agent_error_tx,
                eid,
                conversation_id,
                error_type,
                details,
                severity,
                contact_id,
            )
        logger.debug(
            "learning.agent_error_created",
            id=eid,
            type=error_type,
            severity=severity,
        )
        return eid

    @staticmethod
    async def _create_agent_error_tx(
        tx,
        eid: str,
        conversation_id: str,
        error_type: str,
        details: str,
        severity: str,
        contact_id: str | None,
    ) -> None:
        await tx.run(
            """
            MERGE (e:AgentError {id: $eid})
            SET e.conversation_id = $cid,
                e.type = $etype,
                e.details = $details,
                e.severity = $severity,
                e.created_at = datetime()
            """,
            eid=eid,
            cid=conversation_id,
            etype=error_type,
            details=details,
            severity=severity,
        )
        if contact_id:
            await tx.run(
                """
                MATCH (l:Lead {ghl_contact_id: $contact_id})
                MATCH (e:AgentError {id: $eid})
                MERGE (l)-[:HAS_ERROR]->(e)
                """,
                contact_id=contact_id,
                eid=eid,
            )

    # ------------------------------------------------------------------
    # Repetition event (specialized AgentError)
    # ------------------------------------------------------------------

    async def create_repetition_event(
        self,
        lead_phone: str,
        similarity_score: float,
        message_pair: tuple[str, str],
    ) -> str:
        """Create a repetition-type AgentError linked to Lead by phone."""
        eid = self._new_id()
        async with self._driver.session() as session:
            await session.execute_write(
                self._create_repetition_event_tx,
                eid,
                lead_phone,
                similarity_score,
                message_pair[0][:200],
                message_pair[1][:200],
            )
        logger.debug(
            "learning.repetition_event_created",
            id=eid,
            phone=lead_phone[-4:],
            score=similarity_score,
        )
        return eid

    @staticmethod
    async def _create_repetition_event_tx(
        tx,
        eid: str,
        lead_phone: str,
        similarity_score: float,
        msg_a: str,
        msg_b: str,
    ) -> None:
        await tx.run(
            """
            MERGE (e:AgentError {id: $eid})
            SET e.type = 'repetition',
                e.similarity_score = $score,
                e.msg_a = $msg_a,
                e.msg_b = $msg_b,
                e.created_at = datetime()
            """,
            eid=eid,
            score=similarity_score,
            msg_a=msg_a,
            msg_b=msg_b,
        )
        await tx.run(
            """
            MATCH (l:Lead {phone: $phone})
            MATCH (e:AgentError {id: $eid})
            MERGE (l)-[:HAS_ERROR]->(e)
            """,
            phone=lead_phone,
            eid=eid,
        )

    # ------------------------------------------------------------------
    # LeadSatisfaction
    # ------------------------------------------------------------------

    async def create_lead_satisfaction(
        self,
        lead_phone: str,
        score: float,
        source: str,
        signals: dict,
    ) -> str:
        """Create a LeadSatisfaction measurement linked to Lead by phone."""
        sid = self._new_id()
        async with self._driver.session() as session:
            await session.execute_write(
                self._create_lead_satisfaction_tx,
                sid,
                lead_phone,
                score,
                source,
                json.dumps(signals),
            )
        logger.debug(
            "learning.lead_satisfaction_created",
            id=sid,
            phone=lead_phone[-4:],
            score=score,
        )
        return sid

    @staticmethod
    async def _create_lead_satisfaction_tx(
        tx,
        sid: str,
        lead_phone: str,
        score: float,
        source: str,
        signals_json: str,
    ) -> None:
        await tx.run(
            """
            MERGE (s:LeadSatisfaction {id: $sid})
            SET s.lead_phone = $phone,
                s.score = $score,
                s.source = $source,
                s.signals = $signals,
                s.created_at = datetime()
            """,
            sid=sid,
            phone=lead_phone,
            score=score,
            source=source,
            signals=signals_json,
        )
        await tx.run(
            """
            MATCH (l:Lead {phone: $phone})
            MATCH (s:LeadSatisfaction {id: $sid})
            MERGE (l)-[:HAS_SATISFACTION]->(s)
            """,
            phone=lead_phone,
            sid=sid,
        )

    # ------------------------------------------------------------------
    # PromptVersion
    # ------------------------------------------------------------------

    async def create_prompt_version(
        self,
        state: str,
        version: str,
        content_hash: str,
        active: bool = True,
    ) -> str:
        """Create a PromptVersion node. If active, deactivate others for this state."""
        pid = self._new_id()
        async with self._driver.session() as session:
            await session.execute_write(
                self._create_prompt_version_tx,
                pid,
                state,
                version,
                content_hash,
                active,
            )
        logger.debug(
            "learning.prompt_version_created",
            id=pid,
            state=state,
            version=version,
            active=active,
        )
        return pid

    @staticmethod
    async def _create_prompt_version_tx(
        tx,
        pid: str,
        state: str,
        version: str,
        content_hash: str,
        active: bool,
    ) -> None:
        if active:
            # Deactivate all other versions for this state first
            await tx.run(
                """
                MATCH (pv:PromptVersion {state: $state})
                WHERE pv.active = true
                SET pv.active = false
                """,
                state=state,
            )
        await tx.run(
            """
            MERGE (pv:PromptVersion {id: $pid})
            SET pv.state = $state,
                pv.version = $version,
                pv.content_hash = $hash,
                pv.active = $active,
                pv.created_at = datetime()
            """,
            pid=pid,
            state=state,
            version=version,
            hash=content_hash,
            active=active,
        )

    # ------------------------------------------------------------------
    # LessonLearned
    # ------------------------------------------------------------------

    async def create_lesson_learned(
        self,
        rule: str,
        why: str,
        severity: str,
        confidence: float,
        building_id: str | None = None,
    ) -> str:
        """Create a LessonLearned node (status=candidate). Optionally link to Building."""
        lid = self._new_id()
        async with self._driver.session() as session:
            await session.execute_write(
                self._create_lesson_learned_tx,
                lid,
                rule,
                why,
                severity,
                confidence,
                building_id,
            )
        logger.debug(
            "learning.lesson_learned_created",
            id=lid,
            rule=rule[:80],
            confidence=confidence,
        )
        return lid

    @staticmethod
    async def _create_lesson_learned_tx(
        tx,
        lid: str,
        rule: str,
        why: str,
        severity: str,
        confidence: float,
        building_id: str | None,
    ) -> None:
        await tx.run(
            """
            MERGE (ll:LessonLearned {id: $lid})
            SET ll.rule = $rule,
                ll.why = $why,
                ll.severity = $severity,
                ll.confidence = $confidence,
                ll.status = 'candidate',
                ll.created_at = datetime()
            """,
            lid=lid,
            rule=rule,
            why=why,
            severity=severity,
            confidence=confidence,
        )
        if building_id:
            await tx.run(
                """
                MATCH (b:Building {building_id: $bid})
                MATCH (ll:LessonLearned {id: $lid})
                MERGE (ll)-[:APPLIES_TO]->(b)
                """,
                bid=building_id,
                lid=lid,
            )

    # ------------------------------------------------------------------
    # ErrorPattern
    # ------------------------------------------------------------------

    async def create_error_pattern(
        self,
        pattern_name: str,
        frequency: int = 1,
    ) -> str:
        """Create or get an ErrorPattern node. Does NOT increment on match."""
        async with self._driver.session() as session:
            await session.execute_write(
                self._create_error_pattern_tx, pattern_name
            )
        logger.debug("learning.error_pattern_created", pattern_name=pattern_name)
        return pattern_name

    @staticmethod
    async def _create_error_pattern_tx(tx, pattern_name: str) -> None:
        await tx.run(
            """
            MERGE (ep:ErrorPattern {pattern_name: $name})
            ON CREATE SET ep.frequency = 1,
                          ep.first_seen = datetime(),
                          ep.last_seen = datetime()
            """,
            name=pattern_name,
        )

    async def increment_error_pattern(self, pattern_name: str) -> int:
        """Increment frequency for an ErrorPattern. Creates if not exists."""
        async with self._driver.session() as session:
            result = await session.execute_write(
                self._increment_error_pattern_tx, pattern_name
            )
        logger.debug(
            "learning.error_pattern_incremented",
            pattern_name=pattern_name,
            frequency=result,
        )
        return result

    @staticmethod
    async def _increment_error_pattern_tx(tx, pattern_name: str) -> int:
        result = await tx.run(
            """
            MERGE (ep:ErrorPattern {pattern_name: $name})
            ON CREATE SET ep.frequency = 1,
                          ep.first_seen = datetime(),
                          ep.last_seen = datetime()
            ON MATCH SET ep.frequency = ep.frequency + 1,
                         ep.last_seen = datetime()
            RETURN ep.frequency AS frequency
            """,
            name=pattern_name,
        )
        record = await result.single()
        return record["frequency"] if record else 1

    # ------------------------------------------------------------------
    # ConversationEmbedding
    # ------------------------------------------------------------------

    async def create_conversation_embedding(
        self,
        conversation_id: str,
        vector: list[float],
        model: str = "text-embedding-3-small",
    ) -> str:
        """Create or update a ConversationEmbedding node (one per conversation)."""
        async with self._driver.session() as session:
            await session.execute_write(
                self._create_conversation_embedding_tx,
                conversation_id,
                vector,
                model,
            )
        logger.debug(
            "learning.conversation_embedding_created",
            conversation_id=conversation_id,
            model=model,
            dims=len(vector),
        )
        return conversation_id

    @staticmethod
    async def _create_conversation_embedding_tx(
        tx,
        conversation_id: str,
        vector: list[float],
        model: str,
    ) -> None:
        await tx.run(
            """
            MERGE (ce:ConversationEmbedding {conversation_id: $cid})
            SET ce.vector = $vector,
                ce.model = $model,
                ce.created_at = datetime()
            """,
            cid=conversation_id,
            vector=vector,
            model=model,
        )

    # ------------------------------------------------------------------
    # Vector similarity search
    # ------------------------------------------------------------------

    async def find_similar_conversations(
        self,
        vector: list[float],
        threshold: float = 0.7,
        limit: int = 5,
    ) -> list[dict[str, Any]]:
        """Find conversations with similar embeddings via vector index."""
        async with self._driver.session() as session:
            result = await session.execute_read(
                self._find_similar_conversations_tx, vector, threshold, limit
            )
        return result

    @staticmethod
    async def _find_similar_conversations_tx(
        tx,
        vector: list[float],
        threshold: float,
        limit: int,
    ) -> list[dict[str, Any]]:
        result = await tx.run(
            """
            CALL db.index.vector.queryNodes('conversation_embedding_vector', $limit, $vector)
            YIELD node, score
            WHERE score >= $threshold
            RETURN node.conversation_id AS conversation_id, score
            """,
            vector=vector,
            threshold=threshold,
            limit=limit,
        )
        return [dict(r) async for r in result]

    # ------------------------------------------------------------------
    # Lesson queries
    # ------------------------------------------------------------------

    async def get_lessons_for_context(
        self,
        building_id: str | None = None,
        state: str | None = None,
        topic: str | None = None,
        min_confidence: float = 0.7,
        limit: int = 3,
    ) -> list[dict[str, Any]]:
        """Get approved/evergreen lessons, optionally filtered by building, state, topic."""
        async with self._driver.session() as session:
            result = await session.execute_read(
                self._get_lessons_for_context_tx,
                building_id,
                state,
                topic,
                min_confidence,
                limit,
            )
        return result

    @staticmethod
    async def _get_lessons_for_context_tx(
        tx,
        building_id: str | None,
        state: str | None,
        topic: str | None,
        min_confidence: float,
        limit: int,
    ) -> list[dict[str, Any]]:
        where_clauses = [
            "ll.confidence >= $min_conf",
            "ll.status IN ['approved', 'evergreen']",
        ]
        params: dict[str, Any] = {"min_conf": min_confidence, "lim": limit}

        if building_id:
            where_clauses.append(
                "EXISTS { MATCH (ll)-[:APPLIES_TO]->(b:Building {building_id: $bid}) }"
            )
            params["bid"] = building_id
        if state:
            where_clauses.append("ll.rule CONTAINS $state")
            params["state"] = state
        if topic:
            where_clauses.append("(ll.rule CONTAINS $topic OR ll.why CONTAINS $topic)")
            params["topic"] = topic

        query = (
            "MATCH (ll:LessonLearned) "
            f"WHERE {' AND '.join(where_clauses)} "
            "RETURN ll.id AS id, ll.rule AS rule, ll.why AS why, "
            "ll.severity AS severity, ll.confidence AS confidence, ll.status AS status "
            "ORDER BY ll.confidence DESC LIMIT $lim"
        )
        result = await tx.run(query, **params)
        return [dict(r) async for r in result]

    # ------------------------------------------------------------------
    # ErrorPattern queries
    # ------------------------------------------------------------------

    async def get_error_patterns(self, min_frequency: int = 3) -> list[dict[str, Any]]:
        """Get error patterns that have occurred at least min_frequency times."""
        async with self._driver.session() as session:
            result = await session.execute_read(
                self._get_error_patterns_tx, min_frequency
            )
        return result

    @staticmethod
    async def _get_error_patterns_tx(
        tx, min_frequency: int
    ) -> list[dict[str, Any]]:
        result = await tx.run(
            """
            MATCH (ep:ErrorPattern)
            WHERE ep.frequency >= $min_freq
            RETURN ep.pattern_name AS pattern_name,
                   ep.frequency AS frequency,
                   ep.first_seen AS first_seen,
                   ep.last_seen AS last_seen
            ORDER BY ep.frequency DESC
            """,
            min_freq=min_frequency,
        )
        return [dict(r) async for r in result]

    # ------------------------------------------------------------------
    # Building error history
    # ------------------------------------------------------------------

    async def get_building_error_history(
        self,
        building_id: str,
        days: int = 30,
    ) -> list[dict[str, Any]]:
        """Get AgentError nodes linked to a building in the last N days."""
        async with self._driver.session() as session:
            result = await session.execute_read(
                self._get_building_error_history_tx, building_id, days
            )
        return result

    @staticmethod
    async def _get_building_error_history_tx(
        tx,
        building_id: str,
        days: int,
    ) -> list[dict[str, Any]]:
        result = await tx.run(
            """
            MATCH (e:AgentError)-[:RELATES_TO|HAS_ERROR]-(l:Lead)-[:INTERESTED_IN]->(b:Building {building_id: $bid})
            WHERE e.created_at >= datetime() - duration({days: $days})
            RETURN e.type AS type, e.details AS details,
                   e.severity AS severity, e.created_at AS created_at
            ORDER BY e.created_at DESC
            LIMIT 10
            """,
            bid=building_id,
            days=days,
        )
        return [dict(r) async for r in result]

    # ------------------------------------------------------------------
    # Lesson-error links for conversations
    # ------------------------------------------------------------------

    async def get_lessons_for_conversation_errors(
        self,
        conversation_ids: list[str],
    ) -> list[dict[str, Any]]:
        """Get LessonLearned nodes linked via FIXED_BY from AgentErrors on these conversations."""
        if not conversation_ids:
            return []
        async with self._driver.session() as session:
            result = await session.execute_read(
                self._get_lessons_for_conversation_errors_tx, conversation_ids
            )
        return result

    @staticmethod
    async def _get_lessons_for_conversation_errors_tx(
        tx,
        conversation_ids: list[str],
    ) -> list[dict[str, Any]]:
        result = await tx.run(
            """
            MATCH (e:AgentError)-[:FIXED_BY]->(ll:LessonLearned)
            WHERE e.conversation_id IN $cids
              AND ll.status IN ['approved', 'evergreen']
            RETURN DISTINCT ll.id AS id, ll.rule AS rule, ll.why AS why,
                   ll.confidence AS confidence,
                   e.type AS error_type, e.details AS error_details
            LIMIT 5
            """,
            cids=conversation_ids,
        )
        return [dict(r) async for r in result]

    # ------------------------------------------------------------------
    # Lesson queries by status (admin endpoints)
    # ------------------------------------------------------------------

    async def get_lessons_by_status(
        self,
        statuses: list[str],
        limit: int = 50,
    ) -> list[dict[str, Any]]:
        """Get LessonLearned nodes filtered by status list."""
        async with self._driver.session() as session:
            result = await session.execute_read(
                self._get_lessons_by_status_tx, statuses, limit
            )
        return result

    @staticmethod
    async def _get_lessons_by_status_tx(
        tx,
        statuses: list[str],
        limit: int,
    ) -> list[dict[str, Any]]:
        result = await tx.run(
            """
            MATCH (ll:LessonLearned)
            WHERE ll.status IN $statuses
            RETURN ll.id AS id, ll.rule AS rule, ll.why AS why,
                   ll.severity AS severity, ll.confidence AS confidence,
                   ll.status AS status, ll.created_at AS created_at
            ORDER BY ll.created_at DESC
            LIMIT $lim
            """,
            statuses=statuses,
            lim=limit,
        )
        return [dict(r) async for r in result]

    # ------------------------------------------------------------------
    # Lesson lifecycle
    # ------------------------------------------------------------------

    async def decay_lesson_confidence(
        self,
        days_threshold: int = 90,
        decay_rate: float = 0.1,
    ) -> int:
        """Decay confidence of old candidate lessons. Returns count affected."""
        async with self._driver.session() as session:
            result = await session.execute_write(
                self._decay_lesson_confidence_tx, days_threshold, decay_rate
            )
        logger.info("learning.lesson_confidence_decayed", affected=result)
        return result

    @staticmethod
    async def _decay_lesson_confidence_tx(
        tx, days_threshold: int, decay_rate: float
    ) -> int:
        result = await tx.run(
            """
            MATCH (ll:LessonLearned)
            WHERE ll.created_at < datetime() - duration({days: $days})
              AND ll.status = 'candidate'
              AND ll.confidence > 0.1
            SET ll.confidence = CASE
                WHEN ll.confidence - $decay > 0.1 THEN ll.confidence - $decay
                ELSE 0.1
            END
            RETURN count(ll) AS affected
            """,
            days=days_threshold,
            decay=decay_rate,
        )
        record = await result.single()
        return record["affected"] if record else 0

    async def promote_lesson_to_evergreen(self, lesson_id: str) -> bool:
        """Promote a lesson to evergreen status. Returns True if found and updated."""
        async with self._driver.session() as session:
            result = await session.execute_write(
                self._promote_lesson_to_evergreen_tx, lesson_id
            )
        if result:
            logger.info("learning.lesson_promoted_to_evergreen", lesson_id=lesson_id)
        return result

    @staticmethod
    async def _promote_lesson_to_evergreen_tx(tx, lesson_id: str) -> bool:
        result = await tx.run(
            """
            MATCH (ll:LessonLearned {id: $lid})
            SET ll.status = 'evergreen'
            RETURN ll.id AS id
            """,
            lid=lesson_id,
        )
        record = await result.single()
        return record is not None

    # ------------------------------------------------------------------
    # Lesson existence check (semantic dedup)
    # ------------------------------------------------------------------

    async def check_lesson_exists_for_pattern(self, pattern_name: str) -> bool:
        """Check if any active LessonLearned has a rule containing the pattern name."""
        async with self._driver.session() as session:
            result = await session.execute_read(
                self._check_lesson_exists_for_pattern_tx, pattern_name
            )
        return result

    @staticmethod
    async def _check_lesson_exists_for_pattern_tx(tx, pattern_name: str) -> bool:
        result = await tx.run(
            """
            MATCH (ll:LessonLearned)
            WHERE ll.rule CONTAINS $pattern_name
              AND ll.status IN ['candidate', 'approved', 'evergreen']
            RETURN count(ll) > 0 AS exists
            """,
            pattern_name=pattern_name,
        )
        record = await result.single()
        return record["exists"] if record else False

    # ------------------------------------------------------------------
    # Outcome averages for effectiveness tracking
    # ------------------------------------------------------------------

    async def get_outcome_averages(self, days: int) -> dict:
        """Return AVG scores for ConversationOutcome nodes created in last N days.

        Returns {"repetition": float, "sentiment": float, "intent_alignment": float, "count": int}.
        """
        async with self._driver.session() as session:
            result = await session.execute_read(
                self._get_outcome_averages_tx, days
            )
        return result

    @staticmethod
    async def _get_outcome_averages_tx(tx, days: int) -> dict:
        result = await tx.run(
            """
            MATCH (o:ConversationOutcome)
            WHERE o.created_at >= datetime() - duration({days: $days})
            RETURN o.scores AS scores
            """,
            days=days,
        )
        records = [r async for r in result]

        if not records:
            return {"repetition": 0.0, "sentiment": 0.0, "intent_alignment": 0.0, "count": 0}

        totals = {"repetition": 0.0, "sentiment": 0.0, "intent_alignment": 0.0}
        count = 0
        for record in records:
            scores_raw = record.get("scores", "{}")
            try:
                scores = json.loads(scores_raw) if isinstance(scores_raw, str) else scores_raw
            except (json.JSONDecodeError, TypeError):
                continue
            if not isinstance(scores, dict):
                continue
            count += 1
            for key in totals:
                totals[key] += float(scores.get(key, 0.0))

        if count == 0:
            return {"repetition": 0.0, "sentiment": 0.0, "intent_alignment": 0.0, "count": 0}

        return {
            "repetition": round(totals["repetition"] / count, 4),
            "sentiment": round(totals["sentiment"] / count, 4),
            "intent_alignment": round(totals["intent_alignment"] / count, 4),
            "count": count,
        }

    # ------------------------------------------------------------------
    # Lesson injection count (proxy)
    # ------------------------------------------------------------------

    async def get_lesson_injection_count(self, days: int) -> int:
        """Count ConversationOutcome nodes from last N days (proxy for injection opportunities)."""
        async with self._driver.session() as session:
            result = await session.execute_read(
                self._get_lesson_injection_count_tx, days
            )
        return result

    @staticmethod
    async def _get_lesson_injection_count_tx(tx, days: int) -> int:
        result = await tx.run(
            """
            MATCH (o:ConversationOutcome)
            WHERE o.created_at >= datetime() - duration({days: $days})
            RETURN count(o) AS cnt
            """,
            days=days,
        )
        record = await result.single()
        return record["cnt"] if record else 0

    # ------------------------------------------------------------------
    # Archive low confidence candidates
    # ------------------------------------------------------------------

    async def archive_low_confidence(self) -> int:
        """Archive candidate lessons with confidence <= 0.1. Returns count affected."""
        async with self._driver.session() as session:
            result = await session.execute_write(
                self._archive_low_confidence_tx
            )
        logger.info("learning.low_confidence_archived", affected=result)
        return result

    @staticmethod
    async def _archive_low_confidence_tx(tx) -> int:
        result = await tx.run(
            """
            MATCH (ll:LessonLearned)
            WHERE ll.confidence <= 0.1 AND ll.status = 'candidate'
            SET ll.status = 'archived'
            RETURN count(ll) AS affected
            """,
        )
        record = await result.single()
        return record["affected"] if record else 0

    # ------------------------------------------------------------------
    # Auto-promote mature candidates
    # ------------------------------------------------------------------

    async def auto_promote_candidates(
        self,
        min_confidence: float = 0.7,
        min_age_days: int = 7,
    ) -> list[str]:
        """Promote mature high-confidence candidates to approved. Returns list of promoted IDs."""
        async with self._driver.session() as session:
            result = await session.execute_write(
                self._auto_promote_candidates_tx, min_confidence, min_age_days
            )
        if result:
            logger.info("learning.candidates_auto_promoted", count=len(result))
        return result

    @staticmethod
    async def _auto_promote_candidates_tx(
        tx, min_confidence: float, min_age_days: int
    ) -> list[str]:
        result = await tx.run(
            """
            MATCH (ll:LessonLearned)
            WHERE ll.status = 'candidate'
              AND ll.confidence >= $min_conf
              AND ll.created_at < datetime() - duration({days: $min_days})
            SET ll.status = 'approved'
            RETURN ll.id AS id
            """,
            min_conf=min_confidence,
            min_days=min_age_days,
        )
        return [r["id"] async for r in result]

    # ------------------------------------------------------------------
    # Cleanup stale embeddings
    # ------------------------------------------------------------------

    async def cleanup_old_embeddings(self, days: int = 180) -> int:
        """Delete ConversationEmbedding nodes older than N days. Returns count deleted."""
        async with self._driver.session() as session:
            result = await session.execute_write(
                self._cleanup_old_embeddings_tx, days
            )
        logger.info("learning.old_embeddings_cleaned", deleted=result)
        return result

    @staticmethod
    async def _cleanup_old_embeddings_tx(tx, days: int) -> int:
        # Two-step: count first, then delete
        result = await tx.run(
            """
            MATCH (ce:ConversationEmbedding)
            WHERE ce.created_at < datetime() - duration({days: $days})
            WITH collect(ce) AS nodes, count(ce) AS cnt
            UNWIND nodes AS ce
            DETACH DELETE ce
            RETURN cnt
            """,
            days=days,
        )
        record = await result.single()
        return record["cnt"] if record else 0

    # ------------------------------------------------------------------
    # Cleanup rejected lessons
    # ------------------------------------------------------------------

    async def cleanup_rejected_lessons(self, days: int = 30) -> int:
        """Delete rejected LessonLearned nodes older than N days. Returns count deleted."""
        async with self._driver.session() as session:
            result = await session.execute_write(
                self._cleanup_rejected_lessons_tx, days
            )
        logger.info("learning.rejected_lessons_cleaned", deleted=result)
        return result

    @staticmethod
    async def _cleanup_rejected_lessons_tx(tx, days: int) -> int:
        result = await tx.run(
            """
            MATCH (ll:LessonLearned)
            WHERE ll.status = 'rejected'
              AND ll.created_at < datetime() - duration({days: $days})
            WITH collect(ll) AS nodes, count(ll) AS cnt
            UNWIND nodes AS ll
            DETACH DELETE ll
            RETURN cnt
            """,
            days=days,
        )
        record = await result.single()
        return record["cnt"] if record else 0
