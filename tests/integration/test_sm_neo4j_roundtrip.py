"""End-to-end integration test: ConversationSM + Neo4j round-trip.

Uses the ``neo4j_container`` testcontainers fixture from
``tests/repositories/conftest.py``. Tests are marked ``slow`` so they
run in the ``integration-tests`` job of the CI deploy gate, separate
from the fast unit tests.

Why this exists
---------------
The Apr 5-7 outage was a pure SM validation failure -- the python-
statemachine library raised ``InvalidDefinition`` on every
instantiation of ``ConversationSM`` because of an outgoing transition
from a final state. The existing parametrized SM tests in
``tests/state_machine/`` would have caught it, but they were never
run before the deploy.

Most of the unit tests in this codebase mock Neo4j. Mocked tests
return clean dict-shaped data; real Neo4j returns ``DateTime`` objects
that crash ``json.dumps``, ``int()``, and ``float()`` calls. This
file exercises the SM and the conversation repository against a
*real* Neo4j container so seam-level bugs (DateTime serialization,
relationship name typos, label mismatches) surface in CI rather than
in production.
"""

from __future__ import annotations

import pytest
import pytest_asyncio

pytestmark = pytest.mark.slow


# ---------------------------------------------------------------------------
# SM instantiation against the real Python runtime
# (does not actually need Neo4j -- just here to live alongside the integration
# tests so the import-time validation runs in the same Python version that
# CI uses for the integration job)
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "state",
    [
        "GREETING", "QUALIFYING", "SCHEDULING", "QUALIFIED",
        "NON_RESPONSIVE", "RECOVERY", "FOLLOW_UP", "RE_ENGAGE",
        "HANDOFF", "CLOSED", "BROKER",
    ],
)
def test_sm_instantiates_for_every_state_in_integration_runtime(state):
    """python-statemachine validation must pass for every state.

    Catches the Apr 5-7 outage failure mode at the integration-test
    runtime (which uses the same Python version as production).
    """
    from app.state_machine.conversation_sm import ConversationSM

    sm = ConversationSM.from_persisted_state(state, f"smoke_{state}")
    assert sm.model.state == state


# ---------------------------------------------------------------------------
# SM <-> Neo4j round-trip
# ---------------------------------------------------------------------------


@pytest_asyncio.fixture()
async def seeded_lead(neo4j_driver):
    """Insert a sample Lead node and return its identifiers."""
    cid = "test_contact_001"
    phone = "+15555550000"
    async with neo4j_driver.session() as session:
        await session.run(
            "CREATE (l:Lead {ghl_contact_id: $cid, current_state: 'GREETING', phone: $phone, full_name: 'Test Lead'})",
            cid=cid, phone=phone,
        )
    return {"contact_id": cid, "phone": phone}


@pytest.mark.asyncio
async def test_sm_state_persists_back_to_neo4j(neo4j_driver, seeded_lead):
    """Round-trip: load Lead -> SM transition -> persist -> reload."""
    from app.state_machine.conversation_sm import ConversationSM

    cid = seeded_lead["contact_id"]

    # 1. Load current state from Neo4j
    async with neo4j_driver.session() as session:
        result = await session.run(
            "MATCH (l:Lead {ghl_contact_id: $cid}) RETURN l.current_state AS state",
            cid=cid,
        )
        row = await result.single()
        assert row["state"] == "GREETING"

    # 2. Restore SM and transition GREETING -> QUALIFYING
    sm = ConversationSM.from_persisted_state("GREETING", cid)
    sm.send("advance", lead_name="Test", interested=True)
    assert sm.model.state == "QUALIFYING"

    # 3. Persist new state
    async with neo4j_driver.session() as session:
        await session.run(
            "MATCH (l:Lead {ghl_contact_id: $cid}) SET l.current_state = $state",
            cid=cid, state=sm.model.state,
        )

    # 4. Reload and verify
    async with neo4j_driver.session() as session:
        result = await session.run(
            "MATCH (l:Lead {ghl_contact_id: $cid}) RETURN l.current_state AS state",
            cid=cid,
        )
        row = await result.single()
        assert row["state"] == "QUALIFYING"


@pytest.mark.asyncio
async def test_sm_handles_terminal_state_round_trip(neo4j_driver, seeded_lead):
    """A Lead in CLOSED state restores cleanly without raising.

    This is the exact failure mode from the Apr 5-7 outage:
    ``ConversationSM.from_persisted_state('CLOSED', cid)`` blew up
    because of the impossible ``CLOSED.to(RE_ENGAGE)`` transition.
    """
    from app.state_machine.conversation_sm import ConversationSM

    cid = seeded_lead["contact_id"]

    # Set lead to CLOSED in Neo4j
    async with neo4j_driver.session() as session:
        await session.run(
            "MATCH (l:Lead {ghl_contact_id: $cid}) SET l.current_state = 'CLOSED'",
            cid=cid,
        )

    # Restore SM -- this is what process_message does, and what
    # raised InvalidDefinition for 46 hours
    sm = ConversationSM.from_persisted_state("CLOSED", cid)
    assert sm.model.state == "CLOSED"
    # CLOSED is a final state -- transitions out of it should be rejected
    from statemachine.exceptions import TransitionNotAllowed
    with pytest.raises(TransitionNotAllowed):
        sm.send("advance")
