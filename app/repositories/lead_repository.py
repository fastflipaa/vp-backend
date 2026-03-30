"""Lead repository for async Neo4j operations.

Provides read/write access to :Lead nodes including state management,
qualification data, sentiment tracking, and cross-session memory.

Each method creates a fresh ``session()`` per operation -- sessions are
NEVER reused across coroutines to avoid transaction conflicts.
"""

from __future__ import annotations

from typing import Any

import structlog
from neo4j import AsyncDriver

logger = structlog.get_logger()


class LeadRepository:
    """Async repository for :Lead node operations in Neo4j."""

    def __init__(self, driver: AsyncDriver) -> None:
        self._driver = driver

    # --- State management ---

    async def get_state(self, contact_id: str) -> str:
        """Get the current conversation state for a lead.

        Returns ``"GREETING"`` if the lead does not exist or has no state.
        """
        async with self._driver.session() as session:
            result = await session.execute_read(self._get_state_tx, contact_id)
            state = result or "GREETING"
            logger.debug("lead_state_read", contact_id=contact_id, state=state)
            return state

    @staticmethod
    async def _get_state_tx(tx, contact_id: str) -> str | None:
        result = await tx.run(
            "MATCH (l:Lead {ghl_contact_id: $cid}) RETURN l.current_state AS state",
            cid=contact_id,
        )
        record = await result.single()
        return record["state"] if record else None

    async def save_state(self, contact_id: str, state: str) -> None:
        """Persist the conversation state for a lead.

        Uses MERGE so the Lead node is created if it does not exist yet.
        """
        async with self._driver.session() as session:
            await session.execute_write(self._save_state_tx, contact_id, state)
            logger.info("lead_state_saved", contact_id=contact_id, state=state)

    @staticmethod
    async def _save_state_tx(tx, contact_id: str, state: str) -> None:
        await tx.run(
            """
            MERGE (l:Lead {ghl_contact_id: $cid})
            SET l.current_state = $state, l.updatedAt = datetime()
            """,
            cid=contact_id,
            state=state,
        )

    # --- Phone-based lookup ---

    async def get_lead_by_phone(self, phone: str) -> dict[str, Any] | None:
        """Comprehensive lead lookup by phone number.

        Returns the same fields that n8n Main Router Stage 2 reads:
        state, language, contact ID, name, qualification sub-state,
        interest type, budget, timeline, email, source, cadence, and
        night greeting status.
        """
        async with self._driver.session() as session:
            result = await session.execute_read(self._get_lead_by_phone_tx, phone)
            if result:
                logger.debug("lead_found_by_phone", phone=phone[-4:])
            else:
                logger.debug("lead_not_found_by_phone", phone=phone[-4:])
            return result

    @staticmethod
    async def _get_lead_by_phone_tx(tx, phone: str) -> dict[str, Any] | None:
        result = await tx.run(
            """
            MATCH (l:Lead {phone: $phone})
            RETURN l.current_state AS current_state,
                   l.language AS language,
                   l.ghl_contact_id AS ghl_contact_id,
                   l.name AS name,
                   l.sub_state AS sub_state,
                   l.interestType AS interestType,
                   l.budgetMin AS budgetMin,
                   l.budgetMax AS budgetMax,
                   l.timeline AS timeline,
                   l.email AS email,
                   l.source AS source,
                   l.cadence AS cadence,
                   l.nightGreetingSent AS nightGreetingSent
            """,
            phone=phone,
        )
        record = await result.single()
        return dict(record) if record else None

    # --- Qualification data ---

    async def save_qualification_data(self, contact_id: str, data: dict[str, Any]) -> None:
        """Save qualification data for a lead.

        Accepts a dict with optional keys: sub_state, interestType,
        budgetMin, budgetMax, timeline, email, language, name.
        Only non-None values are written.
        """
        # Filter to only non-None values
        fields = {
            k: v
            for k, v in data.items()
            if v is not None
            and k in ("sub_state", "interestType", "budgetMin", "budgetMax",
                       "timeline", "email", "language", "name")
        }
        if not fields:
            return

        # Build dynamic SET clause
        set_parts = [f"l.{key} = ${key}" for key in fields]
        set_parts.append("l.updatedAt = datetime()")
        set_clause = ", ".join(set_parts)

        query = f"""
        MERGE (l:Lead {{ghl_contact_id: $cid}})
        SET {set_clause}
        """

        async with self._driver.session() as session:
            await session.execute_write(
                self._run_parameterized_tx, query, {"cid": contact_id, **fields}
            )
            logger.info(
                "qualification_data_saved",
                contact_id=contact_id,
                fields=list(fields.keys()),
            )

    # --- Sentiment ---

    async def save_sentiment(
        self, phone: str, sentiment: str, confidence: float
    ) -> None:
        """Save sentiment analysis result for a lead."""
        async with self._driver.session() as session:
            await session.execute_write(
                self._save_sentiment_tx, phone, sentiment, confidence
            )
            logger.info(
                "sentiment_saved",
                phone=phone[-4:],
                sentiment=sentiment,
                confidence=confidence,
            )

    @staticmethod
    async def _save_sentiment_tx(
        tx, phone: str, sentiment: str, confidence: float
    ) -> None:
        await tx.run(
            """
            MATCH (l:Lead {phone: $phone})
            SET l.sentiment_current = $sentiment,
                l.sentiment_confidence = $confidence,
                l.sentiment_updated_at = datetime()
            """,
            phone=phone,
            sentiment=sentiment,
            confidence=confidence,
        )

    # --- Cross-session memory ---

    async def get_cross_session_memory(self, phone: str) -> dict[str, Any]:
        """Fetch comprehensive cross-session memory for a lead.

        Matches the n8n Greeting Handler (Section 3.1) data assembly:
        lead preferences, interested buildings, docs sent, last 5
        interactions, and total interaction count.
        """
        async with self._driver.session() as session:
            result = await session.execute_read(
                self._get_cross_session_memory_tx, phone
            )
            logger.debug("cross_session_memory_read", phone=phone[-4:])
            return result

    @staticmethod
    async def _get_cross_session_memory_tx(tx, phone: str) -> dict[str, Any]:
        # Lead basics + preferences
        lead_result = await tx.run(
            """
            MATCH (l:Lead {phone: $phone})
            RETURN l.name AS name,
                   l.language AS language,
                   l.interestType AS interestType,
                   l.budgetMin AS budgetMin,
                   l.budgetMax AS budgetMax,
                   l.timeline AS timeline,
                   l.current_state AS current_state,
                   l.sub_state AS sub_state,
                   l.sentiment_current AS sentiment
            """,
            phone=phone,
        )
        lead_record = await lead_result.single()
        lead = dict(lead_record) if lead_record else {}

        # Interested buildings
        buildings_result = await tx.run(
            """
            MATCH (l:Lead {phone: $phone})-[:INTERESTED_IN]->(b:Building)
            RETURN b.name AS name, b.building_id AS building_id,
                   b.price_range AS price_range
            """,
            phone=phone,
        )
        buildings = [dict(r) async for r in buildings_result]

        # Documents sent
        docs_result = await tx.run(
            """
            MATCH (l:Lead {phone: $phone})-[:SENT_DOCUMENT]->(d:Document)
            RETURN d.name AS name, d.type AS type
            """,
            phone=phone,
        )
        docs_sent = [dict(r) async for r in docs_result]

        # Last 5 interactions
        interactions_result = await tx.run(
            """
            MATCH (l:Lead {phone: $phone})-[:HAS_INTERACTION]->(i:Interaction)
            RETURN i.role AS role, i.content AS content, i.created_at AS created_at
            ORDER BY i.created_at DESC
            LIMIT 5
            """,
            phone=phone,
        )
        interactions = [dict(r) async for r in interactions_result]

        # Total interaction count
        count_result = await tx.run(
            """
            MATCH (l:Lead {phone: $phone})-[:HAS_INTERACTION]->(i:Interaction)
            RETURN count(i) AS total
            """,
            phone=phone,
        )
        count_record = await count_result.single()
        total_interactions = count_record["total"] if count_record else 0

        return {
            "lead": lead,
            "buildings": buildings,
            "docs_sent": docs_sent,
            "recent_interactions": interactions,
            "total_interactions": total_interactions,
        }

    # --- Health check queries ---

    async def find_missed_inbound(
        self,
        window_minutes: int = 20,
        response_threshold_minutes: int = 10,
    ) -> list[dict[str, Any]]:
        """Find leads with unanswered inbound messages.

        Returns leads who sent an inbound message within the last
        ``window_minutes`` but received no outbound response within
        ``response_threshold_minutes`` of that inbound message.

        Excludes BROKER and CLOSED leads (terminal states).
        """
        async with self._driver.session() as session:
            results = await session.execute_read(
                self._find_missed_inbound_tx,
                window_minutes,
                response_threshold_minutes,
            )
            logger.info(
                "missed_inbound_query",
                window_minutes=window_minutes,
                threshold_minutes=response_threshold_minutes,
                found=len(results),
            )
            return results

    @staticmethod
    async def _find_missed_inbound_tx(
        tx, window_minutes: int, response_threshold_minutes: int
    ) -> list[dict[str, Any]]:
        result = await tx.run(
            """
            MATCH (l:Lead)-[:HAS_INTERACTION]->(i:Interaction {role: 'user'})
            WHERE i.created_at > datetime() - duration({minutes: $window})
            AND NOT EXISTS {
              MATCH (l)-[:HAS_INTERACTION]->(r:Interaction {role: 'assistant'})
              WHERE r.created_at > i.created_at
              AND r.created_at < i.created_at + duration({minutes: $threshold})
            }
            AND l.current_state <> 'BROKER'
            AND l.current_state <> 'CLOSED'
            RETURN DISTINCT l.ghl_contact_id AS contact_id,
                   l.phone AS phone,
                   l.name AS name,
                   l.current_state AS state
            """,
            window=window_minutes,
            threshold=response_threshold_minutes,
        )
        return [dict(record) async for record in result]

    async def create_recovery_attempt(
        self,
        contact_id: str,
        trace_id: str,
        reason: str,
    ) -> None:
        """Create a :RecoveryAttempt audit node linked to a lead.

        Records that a recovery message was attempted for this lead,
        including the trace_id for full traceability and the reason
        (e.g. ``missed_inbound_10min``).
        """
        async with self._driver.session() as session:
            await session.execute_write(
                self._create_recovery_attempt_tx,
                contact_id,
                trace_id,
                reason,
            )
            logger.info(
                "recovery_attempt_created",
                contact_id=contact_id,
                trace_id=trace_id,
                reason=reason,
            )

    @staticmethod
    async def _create_recovery_attempt_tx(
        tx, contact_id: str, trace_id: str, reason: str
    ) -> None:
        await tx.run(
            """
            MERGE (l:Lead {ghl_contact_id: $cid})
            CREATE (l)-[:HAS_RECOVERY]->(r:RecoveryAttempt {
                trace_id: $tid,
                reason: $reason,
                created_at: datetime()
            })
            """,
            cid=contact_id,
            tid=trace_id,
            reason=reason,
        )

    # --- Utility ---

    @staticmethod
    async def _run_parameterized_tx(tx, query: str, params: dict) -> None:
        """Execute a parameterized write query."""
        await tx.run(query, **params)
