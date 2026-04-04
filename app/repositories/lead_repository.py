"""Lead repository for async Neo4j operations.

Provides read/write access to :Lead nodes including state management,
qualification data, sentiment tracking, and cross-session memory.

Each method creates a fresh ``session()`` per operation -- sessions are
NEVER reused across coroutines to avoid transaction conflicts.
"""

from __future__ import annotations

import json
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
                   b.price_min_usd AS price_min_usd,
                   b.price_max_usd AS price_max_usd,
                   b.pricing_verified AS pricing_verified
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

    # --- Stale lead queries ---

    async def find_stale_leads(
        self,
        min_inactive_days: int,
        max_inactive_days: int,
        states: list[str],
        limit: int = 20,
    ) -> list[dict[str, Any]]:
        """Find leads inactive for a given window in specified states.

        Returns leads whose ``updatedAt`` falls between ``min_inactive_days``
        and ``max_inactive_days`` ago, in one of the given ``states``.
        Excludes leads who received a :RecoveryAttempt within the last 24 hours
        to avoid over-contacting.

        Used by the stale re-engagement task (HOT/WARM/COLD tiers).
        """
        async with self._driver.session() as session:
            results = await session.execute_read(
                self._find_stale_leads_tx,
                min_inactive_days,
                max_inactive_days,
                states,
                limit,
            )
            logger.info(
                "stale_leads_query",
                min_days=min_inactive_days,
                max_days=max_inactive_days,
                states=states,
                found=len(results),
            )
            return results

    @staticmethod
    async def _find_stale_leads_tx(
        tx,
        min_inactive_days: int,
        max_inactive_days: int,
        states: list[str],
        limit: int,
    ) -> list[dict[str, Any]]:
        result = await tx.run(
            """
            MATCH (l:Lead)
            WHERE l.current_state IN $states
            AND l.updatedAt < datetime() - duration({days: $min_days})
            AND l.updatedAt > datetime() - duration({days: $max_days})
            AND NOT EXISTS {
              MATCH (l)-[:HAS_RECOVERY]->(r:RecoveryAttempt)
              WHERE r.created_at > datetime() - duration({days: 1})
            }
            RETURN l.ghl_contact_id AS contact_id,
                   l.phone AS phone,
                   l.name AS name,
                   l.current_state AS state,
                   l.language AS language
            LIMIT $limit
            """,
            states=states,
            min_days=min_inactive_days,
            max_days=max_inactive_days,
            limit=limit,
        )
        return [dict(record) async for record in result]

    async def find_followup_due(self, limit: int = 20) -> list[dict[str, Any]]:
        """Find leads in FOLLOW_UP state needing their next follow-up nudge.

        Queries only FOLLOW_UP state leads (not QUALIFYING -- those are
        handled by the stale re-engagement task). Uses a 20-hour minimum
        window for compatibility with per-lead jitter (20-28h).

        Returns followup_count so the scheduler can do exhaustion checks
        without a separate query.
        """
        async with self._driver.session() as session:
            results = await session.execute_read(
                self._find_followup_due_tx,
                limit,
            )
            logger.info("followup_due_query", found=len(results))
            return results

    @staticmethod
    async def _find_followup_due_tx(
        tx, limit: int
    ) -> list[dict[str, Any]]:
        result = await tx.run(
            """
            MATCH (l:Lead)-[:HAS_INTERACTION]->(i:Interaction)
            WHERE l.current_state = 'FOLLOW_UP'
            WITH l, max(i.created_at) AS last_interaction
            WHERE last_interaction < datetime() - duration({hours: 20})
            AND last_interaction > datetime() - duration({days: 7})
            AND NOT EXISTS {
              MATCH (l)-[:HAS_RECOVERY]->(r:RecoveryAttempt)
              WHERE r.created_at > datetime() - duration({days: 1})
            }
            RETURN l.ghl_contact_id AS contact_id,
                   l.phone AS phone,
                   l.name AS name,
                   l.language AS language,
                   coalesce(l.followup_count, 0) AS followup_count
            LIMIT $limit
            """,
            limit=limit,
        )
        return [dict(record) async for record in result]

    # --- Follow-up counter ---

    async def increment_followup_count(self, contact_id: str) -> int:
        """Increment the follow-up counter on a Lead node.

        Uses MERGE so the Lead node is created if it does not exist.
        Also updates ``last_followup_at`` for timing queries.

        Returns the new followup_count after increment.
        """
        async with self._driver.session() as session:
            new_count = await session.execute_write(
                self._increment_followup_count_tx, contact_id
            )
            logger.info(
                "followup_count_incremented",
                contact_id=contact_id,
                new_count=new_count,
            )
            return new_count

    @staticmethod
    async def _increment_followup_count_tx(tx, contact_id: str) -> int:
        result = await tx.run(
            """
            MERGE (l:Lead {ghl_contact_id: $cid})
            SET l.followup_count = coalesce(l.followup_count, 0) + 1,
                l.last_followup_at = datetime(),
                l.updatedAt = datetime()
            RETURN l.followup_count AS followup_count
            """,
            cid=contact_id,
        )
        record = await result.single()
        return record["followup_count"] if record else 1

    async def reset_followup_count(self, contact_id: str) -> None:
        """Reset the follow-up counter to 0 (e.g. when a lead replies).

        Mid-sequence reply resets the counter so the lead earns 3 new
        follow-up attempts if they go silent again.
        """
        async with self._driver.session() as session:
            await session.execute_write(
                self._reset_followup_count_tx, contact_id
            )
            logger.info("followup_count_reset", contact_id=contact_id)

    @staticmethod
    async def _reset_followup_count_tx(tx, contact_id: str) -> None:
        await tx.run(
            """
            MATCH (l:Lead {ghl_contact_id: $cid})
            SET l.followup_count = 0,
                l.last_followup_at = null,
                l.updatedAt = datetime()
            """,
            cid=contact_id,
        )

    async def get_followup_data(self, contact_id: str) -> dict[str, Any]:
        """Fetch follow-up data for a lead: count, buildings, language, last AI message.

        Returns a dict with followup_count (defaults to 0), building_names,
        language, name, phone, last_followup_at, and last_ai_message.
        """
        async with self._driver.session() as session:
            result = await session.execute_read(
                self._get_followup_data_tx, contact_id
            )
            logger.debug("followup_data_read", contact_id=contact_id)
            return result

    @staticmethod
    async def _get_followup_data_tx(tx, contact_id: str) -> dict[str, Any]:
        result = await tx.run(
            """
            MATCH (l:Lead {ghl_contact_id: $cid})
            OPTIONAL MATCH (l)-[:INTERESTED_IN]->(b:Building)
            WITH l, collect(b.name) AS building_names
            OPTIONAL MATCH (l)-[:HAS_INTERACTION]->(i:Interaction {role: 'assistant'})
            WITH l, building_names, i
            ORDER BY i.created_at DESC
            LIMIT 1
            RETURN coalesce(l.followup_count, 0) AS followup_count,
                   l.last_followup_at AS last_followup_at,
                   l.language AS language,
                   l.name AS name,
                   l.phone AS phone,
                   building_names,
                   i.content AS last_ai_message
            """,
            cid=contact_id,
        )
        record = await result.single()
        if not record:
            return {
                "followup_count": 0,
                "last_followup_at": None,
                "language": "es",
                "name": "",
                "phone": "",
                "building_names": [],
                "last_ai_message": "",
            }
        return {
            "followup_count": record["followup_count"] or 0,
            "last_followup_at": record["last_followup_at"],
            "language": record["language"] or "es",
            "name": record["name"] or "",
            "phone": record["phone"] or "",
            "building_names": record["building_names"] or [],
            "last_ai_message": record["last_ai_message"] or "",
        }

    # --- Retroactive tagging ---

    async def find_non_responsive_for_tagging(
        self, limit: int = 100
    ) -> list[dict[str, Any]]:
        """Find NON_RESPONSIVE leads eligible for email-drip-ready tagging.

        Returns leads in NON_RESPONSIVE state from the last 60 days with
        valid GHL contact IDs. Used for retroactive tagging of leads from
        the pre-drip period.

        Leads older than 60 days are Phase 20 (old lead re-engagement).
        """
        async with self._driver.session() as session:
            results = await session.execute_read(
                self._find_non_responsive_for_tagging_tx, limit
            )
            logger.info(
                "non_responsive_for_tagging_query",
                found=len(results),
                limit=limit,
            )
            return results

    @staticmethod
    async def _find_non_responsive_for_tagging_tx(
        tx, limit: int
    ) -> list[dict[str, Any]]:
        result = await tx.run(
            """
            MATCH (l:Lead)
            WHERE l.current_state = 'NON_RESPONSIVE'
            AND l.ghl_contact_id IS NOT NULL
            AND l.updatedAt > datetime() - duration({days: 60})
            RETURN l.ghl_contact_id AS contact_id,
                   l.name AS name,
                   l.updatedAt AS updated_at
            LIMIT $limit
            """,
            limit=limit,
        )
        return [dict(record) async for record in result]

    # --- Reengagement (Phase 20) ---

    async def find_old_leads_for_reengagement(
        self,
        min_inactive_days: int,
        batch_size: int,
    ) -> list[dict[str, Any]]:
        """Find leads inactive for N+ days eligible for batch re-engagement.

        Returns leads ordered by age-wave priority (oldest first), with
        leads having building interest prioritised over those without.

        Hard exclusions: BROKER, CLOSED, HANDOFF, RECOVERY, SCHEDULED states
        and permanently exhausted leads (reengagement_exhausted=true).

        Note: GHL DND field is NOT synced to Neo4j Lead nodes. Leads tagged
        DND in GHL but never re-engaged will not be excluded here. See
        Phase 20 SUMMARY Known Gaps for details.
        """
        async with self._driver.session() as session:
            results = await session.execute_read(
                self._find_old_leads_for_reengagement_tx,
                min_inactive_days,
                batch_size,
            )
            logger.info(
                "old_leads_for_reengagement_query",
                min_inactive_days=min_inactive_days,
                batch_size=batch_size,
                found=len(results),
            )
            return results

    @staticmethod
    async def _find_old_leads_for_reengagement_tx(
        tx, min_inactive_days: int, batch_size: int
    ) -> list[dict[str, Any]]:
        result = await tx.run(
            """
            MATCH (l:Lead)
            WHERE l.updatedAt < datetime() - duration({days: $min_days})
            AND l.current_state NOT IN ['BROKER', 'CLOSED', 'HANDOFF', 'RECOVERY', 'SCHEDULED']
            AND coalesce(l.reengagement_exhausted, false) = false
            OPTIONAL MATCH (l)-[:INTERESTED_IN]->(b:Building)
            WITH l, count(b) AS building_count
            ORDER BY
              CASE WHEN building_count > 0 THEN 0 ELSE 1 END ASC,
              l.updatedAt ASC
            LIMIT $batch_size
            RETURN l.ghl_contact_id AS contact_id,
                   l.phone AS phone,
                   l.name AS name,
                   l.language AS language,
                   coalesce(l.reengagement_count, 0) AS reengagement_count,
                   l.last_reengagement_at AS last_reengagement_at,
                   building_count
            """,
            min_days=min_inactive_days,
            batch_size=batch_size,
        )
        return [dict(record) async for record in result]

    async def increment_reengagement_count(self, contact_id: str) -> int:
        """Increment the reengagement counter on a Lead node.

        Uses MERGE so the Lead node is created if it does not exist.
        Also updates ``last_reengagement_at`` for rolling-window queries.

        Returns the new reengagement_count after increment.
        """
        async with self._driver.session() as session:
            new_count = await session.execute_write(
                self._increment_reengagement_count_tx, contact_id
            )
            logger.info(
                "reengagement_count_incremented",
                contact_id=contact_id,
                new_count=new_count,
            )
            return new_count

    @staticmethod
    async def _increment_reengagement_count_tx(tx, contact_id: str) -> int:
        result = await tx.run(
            """
            MERGE (l:Lead {ghl_contact_id: $cid})
            SET l.reengagement_count = coalesce(l.reengagement_count, 0) + 1,
                l.last_reengagement_at = datetime(),
                l.updatedAt = datetime()
            RETURN l.reengagement_count AS reengagement_count
            """,
            cid=contact_id,
        )
        record = await result.single()
        return record["reengagement_count"] if record else 1

    async def check_reengagement_eligible(
        self, contact_id: str
    ) -> dict[str, Any]:
        """Check if a lead is eligible for re-engagement outreach.

        Rolling 90-day window logic:
        - reengagement_count < 2: eligible
        - reengagement_count >= 2 AND within 90 days: NOT eligible (spam limit)
        - reengagement_count >= 2 AND outside 90 days: RESET counter, eligible
        - reengagement_exhausted = true: permanently NOT eligible (dead lead)

        Returns dict with ``eligible`` (bool) and ``reason`` (str).
        Uses execute_write since it may reset the counter.
        """
        async with self._driver.session() as session:
            result = await session.execute_write(
                self._check_reengagement_eligible_tx, contact_id
            )
            logger.debug(
                "reengagement_eligibility_checked",
                contact_id=contact_id,
                eligible=result["eligible"],
                reason=result["reason"],
            )
            return result

    @staticmethod
    async def _check_reengagement_eligible_tx(
        tx, contact_id: str
    ) -> dict[str, Any]:
        # Read current state
        result = await tx.run(
            """
            MATCH (l:Lead {ghl_contact_id: $cid})
            RETURN coalesce(l.reengagement_count, 0) AS reengagement_count,
                   l.last_reengagement_at AS last_reengagement_at,
                   coalesce(l.reengagement_exhausted, false) AS reengagement_exhausted
            """,
            cid=contact_id,
        )
        record = await result.single()
        if not record:
            return {"eligible": True, "reason": "lead_not_found_defaults_eligible"}

        if record["reengagement_exhausted"]:
            return {"eligible": False, "reason": "permanently_exhausted"}

        count = record["reengagement_count"]
        last_at = record["last_reengagement_at"]

        if count < 2:
            return {"eligible": True, "reason": "under_limit"}

        # count >= 2: check if outside 90-day window
        if last_at is None:
            # Has count but no timestamp -- treat as eligible with reset
            await tx.run(
                """
                MATCH (l:Lead {ghl_contact_id: $cid})
                SET l.reengagement_count = 0, l.updatedAt = datetime()
                """,
                cid=contact_id,
            )
            return {"eligible": True, "reason": "counter_reset_no_timestamp"}

        # Check 90-day window using Neo4j datetime comparison
        window_result = await tx.run(
            """
            MATCH (l:Lead {ghl_contact_id: $cid})
            RETURN l.last_reengagement_at <= datetime() - duration({days: 90}) AS outside_window
            """,
            cid=contact_id,
        )
        window_record = await window_result.single()

        if window_record and window_record["outside_window"]:
            # Outside 90-day window: reset counter, eligible for new window
            await tx.run(
                """
                MATCH (l:Lead {ghl_contact_id: $cid})
                SET l.reengagement_count = 0,
                    l.last_reengagement_at = null,
                    l.updatedAt = datetime()
                """,
                cid=contact_id,
            )
            return {"eligible": True, "reason": "counter_reset_90day_window_expired"}

        return {"eligible": False, "reason": "at_limit_within_90day_window"}

    async def get_reengagement_data(
        self, contact_id: str
    ) -> dict[str, Any]:
        """Fetch re-engagement data for a lead: count, buildings, recommended history.

        Returns a dict with reengagement_count, last_reengagement_at,
        building_names, building_ids, recommended_buildings (parsed list),
        language, name, phone, and last_ai_message.
        """
        async with self._driver.session() as session:
            result = await session.execute_read(
                self._get_reengagement_data_tx, contact_id
            )
            logger.debug("reengagement_data_read", contact_id=contact_id)
            return result

    @staticmethod
    async def _get_reengagement_data_tx(
        tx, contact_id: str
    ) -> dict[str, Any]:
        result = await tx.run(
            """
            MATCH (l:Lead {ghl_contact_id: $cid})
            OPTIONAL MATCH (l)-[:INTERESTED_IN]->(b:Building)
            WITH l, collect(b.name) AS building_names, collect(b.building_id) AS building_ids
            OPTIONAL MATCH (l)-[:HAS_INTERACTION]->(i:Interaction {role: 'assistant'})
            WITH l, building_names, building_ids, i
            ORDER BY i.created_at DESC
            LIMIT 1
            RETURN coalesce(l.reengagement_count, 0) AS reengagement_count,
                   l.last_reengagement_at AS last_reengagement_at,
                   l.language AS language,
                   l.name AS name,
                   l.phone AS phone,
                   building_names,
                   building_ids,
                   coalesce(l.recommended_buildings, '[]') AS recommended_buildings_json,
                   i.content AS last_ai_message
            """,
            cid=contact_id,
        )
        record = await result.single()
        if not record:
            return {
                "reengagement_count": 0,
                "last_reengagement_at": None,
                "language": "es",
                "name": "",
                "phone": "",
                "building_names": [],
                "building_ids": [],
                "recommended_buildings": [],
                "last_ai_message": "",
            }

        # Parse recommended_buildings JSON safely
        raw_json = record["recommended_buildings_json"] or "[]"
        try:
            recommended = json.loads(raw_json)
        except (json.JSONDecodeError, TypeError):
            recommended = []

        return {
            "reengagement_count": record["reengagement_count"] or 0,
            "last_reengagement_at": record["last_reengagement_at"],
            "language": record["language"] or "es",
            "name": record["name"] or "",
            "phone": record["phone"] or "",
            "building_names": record["building_names"] or [],
            "building_ids": record["building_ids"] or [],
            "recommended_buildings": recommended,
            "last_ai_message": record["last_ai_message"] or "",
        }

    async def record_recommended_buildings(
        self, contact_id: str, building_names: list[str]
    ) -> None:
        """Update the recommended_buildings JSON array on a Lead node.

        The caller (processor) merges existing + new lists and passes the
        full list. This method serialises to JSON and writes.
        """
        buildings_json = json.dumps(building_names, ensure_ascii=False)
        async with self._driver.session() as session:
            await session.execute_write(
                self._record_recommended_buildings_tx,
                contact_id,
                buildings_json,
            )
            logger.info(
                "recommended_buildings_recorded",
                contact_id=contact_id,
                count=len(building_names),
            )

    @staticmethod
    async def _record_recommended_buildings_tx(
        tx, contact_id: str, buildings_json: str
    ) -> None:
        await tx.run(
            """
            MATCH (l:Lead {ghl_contact_id: $cid})
            SET l.recommended_buildings = $buildings_json,
                l.updatedAt = datetime()
            """,
            cid=contact_id,
            buildings_json=buildings_json,
        )

    async def mark_reengagement_exhausted(self, contact_id: str) -> None:
        """Mark a lead as permanently exhausted from re-engagement.

        Sets ``reengagement_exhausted = true`` on the Lead node. After dead
        classification, no further automated outreach is attempted.
        """
        async with self._driver.session() as session:
            await session.execute_write(
                self._mark_reengagement_exhausted_tx, contact_id
            )
            logger.info(
                "reengagement_exhausted_marked",
                contact_id=contact_id,
            )

    @staticmethod
    async def _mark_reengagement_exhausted_tx(
        tx, contact_id: str
    ) -> None:
        await tx.run(
            """
            MATCH (l:Lead {ghl_contact_id: $cid})
            SET l.reengagement_exhausted = true,
                l.updatedAt = datetime()
            """,
            cid=contact_id,
        )

    # --- Utility ---

    @staticmethod
    async def _run_parameterized_tx(tx, query: str, params: dict) -> None:
        """Execute a parameterized write query."""
        await tx.run(query, **params)
