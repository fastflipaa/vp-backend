"""Tests for LeadRepository -- all methods tested against real Neo4j testcontainer.

Each test is async and marked ``slow`` (requires Docker).
The ``neo4j_driver`` fixture provides a per-test driver with automatic cleanup.
"""

from __future__ import annotations

import uuid

import pytest

from app.repositories.lead_repository import LeadRepository

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


async def _create_lead(driver, contact_id: str, phone: str, state: str = "GREETING") -> None:
    """Helper to create a Lead node directly in Neo4j."""
    async with driver.session() as session:
        await session.run(
            """
            CREATE (l:Lead {
                ghl_contact_id: $cid,
                phone: $phone,
                current_state: $state,
                updatedAt: datetime()
            })
            """,
            cid=contact_id,
            phone=phone,
            state=state,
        )


async def _create_lead_with_interaction(
    driver, contact_id: str, phone: str, role: str = "user", minutes_ago: int = 5
) -> None:
    """Create a lead with a single interaction at a given time offset."""
    async with driver.session() as session:
        await session.run(
            """
            CREATE (l:Lead {
                ghl_contact_id: $cid,
                phone: $phone,
                current_state: 'QUALIFYING',
                updatedAt: datetime() - duration({minutes: $mins})
            })
            CREATE (i:Interaction {
                role: $role,
                content: 'test message',
                created_at: datetime() - duration({minutes: $mins})
            })
            MERGE (l)-[:HAS_INTERACTION]->(i)
            """,
            cid=contact_id,
            phone=phone,
            role=role,
            mins=minutes_ago,
        )


# ---------------------------------------------------------------------------
# State management tests
# ---------------------------------------------------------------------------

class TestSaveAndGetState:
    async def test_save_and_get_state(self, neo4j_driver):
        """save_state -> get_state returns the saved state."""
        repo = LeadRepository(neo4j_driver)
        cid = _cid()

        await repo.save_state(cid, "QUALIFYING")
        state = await repo.get_state(cid)

        assert state == "QUALIFYING"

    async def test_new_lead_defaults_to_greeting(self, neo4j_driver):
        """get_state for nonexistent contact returns GREETING."""
        repo = LeadRepository(neo4j_driver)
        cid = _cid()

        state = await repo.get_state(cid)

        assert state == "GREETING"

    async def test_state_updates_overwrite(self, neo4j_driver):
        """Multiple save_state calls overwrite the previous state."""
        repo = LeadRepository(neo4j_driver)
        cid = _cid()

        await repo.save_state(cid, "QUALIFYING")
        await repo.save_state(cid, "SCHEDULING")
        state = await repo.get_state(cid)

        assert state == "SCHEDULING"


# ---------------------------------------------------------------------------
# Qualification data tests
# ---------------------------------------------------------------------------

class TestSaveQualificationData:
    async def test_save_qualification_data(self, neo4j_driver):
        """save_qualification_data stores budget/timeline on the Lead node."""
        repo = LeadRepository(neo4j_driver)
        cid = _cid()

        # Create lead first
        await repo.save_state(cid, "QUALIFYING")

        await repo.save_qualification_data(cid, {
            "budgetMin": 5000000,
            "budgetMax": 10000000,
            "timeline": "3 months",
            "interestType": "rent",
        })

        # Verify via direct query
        async with neo4j_driver.session() as session:
            result = await session.run(
                "MATCH (l:Lead {ghl_contact_id: $cid}) RETURN l",
                cid=cid,
            )
            record = await result.single()
            lead = dict(record["l"])

        assert lead["budgetMin"] == 5000000
        assert lead["budgetMax"] == 10000000
        assert lead["timeline"] == "3 months"
        assert lead["interestType"] == "rent"

    async def test_save_qualification_ignores_unknown_fields(self, neo4j_driver):
        """Fields not in the allowed list are silently ignored."""
        repo = LeadRepository(neo4j_driver)
        cid = _cid()

        await repo.save_state(cid, "QUALIFYING")
        await repo.save_qualification_data(cid, {
            "budgetMin": 1000,
            "secret_field": "should_not_be_saved",
        })

        async with neo4j_driver.session() as session:
            result = await session.run(
                "MATCH (l:Lead {ghl_contact_id: $cid}) RETURN l",
                cid=cid,
            )
            record = await result.single()
            lead = dict(record["l"])

        assert lead["budgetMin"] == 1000
        assert "secret_field" not in lead


# ---------------------------------------------------------------------------
# Sentiment tests
# ---------------------------------------------------------------------------

class TestSaveSentiment:
    async def test_save_sentiment(self, neo4j_driver):
        """save_sentiment stores score on the Lead node (matched by phone)."""
        repo = LeadRepository(neo4j_driver)
        cid = _cid()
        phone = _phone()

        await _create_lead(neo4j_driver, cid, phone, "QUALIFYING")

        await repo.save_sentiment(phone, "positive", 0.92)

        async with neo4j_driver.session() as session:
            result = await session.run(
                "MATCH (l:Lead {phone: $phone}) RETURN l",
                phone=phone,
            )
            record = await result.single()
            lead = dict(record["l"])

        assert lead["sentiment_current"] == "positive"
        assert lead["sentiment_confidence"] == 0.92


# ---------------------------------------------------------------------------
# Phone-based lookup tests
# ---------------------------------------------------------------------------

class TestGetLeadByPhone:
    async def test_get_lead_by_phone(self, neo4j_driver):
        """get_lead_by_phone returns the lead with all expected fields."""
        repo = LeadRepository(neo4j_driver)
        cid = _cid()
        phone = _phone()

        await _create_lead(neo4j_driver, cid, phone, "QUALIFYING")

        result = await repo.get_lead_by_phone(phone)

        assert result is not None
        assert result["ghl_contact_id"] == cid
        assert result["current_state"] == "QUALIFYING"

    async def test_get_lead_by_phone_not_found(self, neo4j_driver):
        """get_lead_by_phone returns None for nonexistent phone."""
        repo = LeadRepository(neo4j_driver)

        result = await repo.get_lead_by_phone("+0000000000")

        assert result is None


# ---------------------------------------------------------------------------
# Health check query tests
# ---------------------------------------------------------------------------

class TestFindMissedInbound:
    async def test_find_missed_inbound(self, neo4j_driver):
        """Lead with recent inbound but no outbound response appears in results."""
        repo = LeadRepository(neo4j_driver)
        cid = _cid()
        phone = _phone()

        # Create lead with inbound message 5 min ago, no outbound
        await _create_lead_with_interaction(neo4j_driver, cid, phone, role="user", minutes_ago=5)

        results = await repo.find_missed_inbound(window_minutes=20, response_threshold_minutes=10)

        contact_ids = [r["contact_id"] for r in results]
        assert cid in contact_ids

    async def test_find_missed_inbound_excludes_terminal_states(self, neo4j_driver):
        """BROKER/CLOSED leads are excluded from missed inbound results."""
        repo = LeadRepository(neo4j_driver)
        cid = _cid()
        phone = _phone()

        # Create BROKER lead with unanswered inbound
        async with neo4j_driver.session() as session:
            await session.run(
                """
                CREATE (l:Lead {
                    ghl_contact_id: $cid, phone: $phone,
                    current_state: 'BROKER',
                    updatedAt: datetime()
                })
                CREATE (i:Interaction {role: 'user', content: 'hi', created_at: datetime() - duration({minutes: 5})})
                MERGE (l)-[:HAS_INTERACTION]->(i)
                """,
                cid=cid, phone=phone,
            )

        results = await repo.find_missed_inbound(window_minutes=20, response_threshold_minutes=10)
        contact_ids = [r["contact_id"] for r in results]
        assert cid not in contact_ids


# ---------------------------------------------------------------------------
# Recovery attempt tests
# ---------------------------------------------------------------------------

class TestCreateRecoveryAttempt:
    async def test_create_recovery_attempt(self, neo4j_driver):
        """create_recovery_attempt creates a RecoveryAttempt node linked to the lead."""
        repo = LeadRepository(neo4j_driver)
        cid = _cid()
        trace = str(uuid.uuid4())

        await repo.save_state(cid, "QUALIFYING")
        await repo.create_recovery_attempt(cid, trace, "missed_inbound_10min")

        async with neo4j_driver.session() as session:
            result = await session.run(
                """
                MATCH (l:Lead {ghl_contact_id: $cid})-[:HAS_RECOVERY]->(r:RecoveryAttempt)
                RETURN r.trace_id AS trace_id, r.reason AS reason
                """,
                cid=cid,
            )
            record = await result.single()

        assert record is not None
        assert record["trace_id"] == trace
        assert record["reason"] == "missed_inbound_10min"


# ---------------------------------------------------------------------------
# Stale leads tests
# ---------------------------------------------------------------------------

class TestFindStaleLeads:
    async def test_find_stale_leads(self, neo4j_driver):
        """Leads with old updatedAt in matching states appear in results."""
        repo = LeadRepository(neo4j_driver)
        cid = _cid()
        phone = _phone()

        # Create a lead updated 5 days ago in QUALIFYING state
        async with neo4j_driver.session() as session:
            await session.run(
                """
                CREATE (l:Lead {
                    ghl_contact_id: $cid,
                    phone: $phone,
                    current_state: 'QUALIFYING',
                    language: 'es',
                    updatedAt: datetime() - duration({days: 5})
                })
                """,
                cid=cid, phone=phone,
            )

        results = await repo.find_stale_leads(
            min_inactive_days=3,
            max_inactive_days=10,
            states=["QUALIFYING", "GREETING"],
        )

        contact_ids = [r["contact_id"] for r in results]
        assert cid in contact_ids

    async def test_find_stale_leads_excludes_recent_recovery(self, neo4j_driver):
        """Leads with a recovery attempt within 24h are excluded."""
        repo = LeadRepository(neo4j_driver)
        cid = _cid()
        phone = _phone()

        # Create stale lead + recent recovery
        async with neo4j_driver.session() as session:
            await session.run(
                """
                CREATE (l:Lead {
                    ghl_contact_id: $cid,
                    phone: $phone,
                    current_state: 'QUALIFYING',
                    updatedAt: datetime() - duration({days: 5})
                })
                CREATE (r:RecoveryAttempt {
                    trace_id: 'test',
                    reason: 'test',
                    created_at: datetime() - duration({hours: 2})
                })
                MERGE (l)-[:HAS_RECOVERY]->(r)
                """,
                cid=cid, phone=phone,
            )

        results = await repo.find_stale_leads(
            min_inactive_days=3,
            max_inactive_days=10,
            states=["QUALIFYING"],
        )

        contact_ids = [r["contact_id"] for r in results]
        assert cid not in contact_ids
