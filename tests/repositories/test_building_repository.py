"""Tests for BuildingRepository -- building lookups, GraphRAG similarity, documents.

All tests run against a real Neo4j testcontainer. Marked ``slow``.
"""

from __future__ import annotations

import uuid

import pytest

from app.repositories.building_repository import BuildingRepository

pytestmark = [pytest.mark.asyncio, pytest.mark.slow]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _phone() -> str:
    return f"+5215500{uuid.uuid4().int % 100000:05d}"


def _bid() -> str:
    return f"bld_{uuid.uuid4().hex[:8]}"


async def _create_lead_with_buildings(
    driver, phone: str, building_ids: list[str]
) -> None:
    """Create a Lead with INTERESTED_IN relationships to buildings."""
    async with driver.session() as session:
        await session.run(
            "CREATE (l:Lead {phone: $phone, ghl_contact_id: $cid})",
            phone=phone,
            cid=f"contact_{uuid.uuid4().hex[:8]}",
        )
        for bid in building_ids:
            await session.run(
                """
                MERGE (b:Building {building_id: $bid})
                SET b.name = 'Tower ' + $bid,
                    b.location = 'Polanco',
                    b.price_range = '5M-10M',
                    b.status = 'available'
                WITH b
                MATCH (l:Lead {phone: $phone})
                MERGE (l)-[:INTERESTED_IN]->(b)
                """,
                bid=bid,
                phone=phone,
            )


async def _create_building_with_docs(
    driver, building_id: str, doc_names: list[str]
) -> None:
    """Create a Building with HAS_DOCUMENT relationships to Documents."""
    async with driver.session() as session:
        await session.run(
            """
            MERGE (b:Building {building_id: $bid})
            SET b.name = 'Tower ' + $bid
            """,
            bid=building_id,
        )
        for name in doc_names:
            await session.run(
                """
                MATCH (b:Building {building_id: $bid})
                CREATE (d:Document {
                    name: $name,
                    url: 'https://docs.example.com/' + $name,
                    type: 'brochure'
                })
                MERGE (b)-[:HAS_DOCUMENT]->(d)
                """,
                bid=building_id,
                name=name,
            )


async def _create_similar_buildings(driver, source_id: str, target_ids: list[str]) -> None:
    """Create SIMILAR_TO relationships between a source building and targets."""
    async with driver.session() as session:
        for i, tid in enumerate(target_ids):
            await session.run(
                """
                MERGE (s:Building {building_id: $sid})
                SET s.name = 'Tower ' + $sid
                MERGE (t:Building {building_id: $tid})
                SET t.name = 'Tower ' + $tid,
                    t.price_range = '6M-12M',
                    t.price_min = $price_min
                MERGE (s)-[:SIMILAR_TO {score: $score}]->(t)
                """,
                sid=source_id,
                tid=tid,
                score=0.9 - (i * 0.1),  # Decreasing similarity
                price_min=5000000 + (i * 1000000),
            )


# ---------------------------------------------------------------------------
# get_buildings_for_lead tests
# ---------------------------------------------------------------------------

class TestGetBuildingsForLead:
    async def test_get_buildings_for_lead(self, neo4j_driver):
        """Returns all buildings a lead has expressed interest in."""
        repo = BuildingRepository(neo4j_driver)
        phone = _phone()
        bid1, bid2 = _bid(), _bid()

        await _create_lead_with_buildings(neo4j_driver, phone, [bid1, bid2])

        results = await repo.get_buildings_for_lead(phone)

        assert len(results) == 2
        returned_ids = {r["building_id"] for r in results}
        assert bid1 in returned_ids
        assert bid2 in returned_ids
        # Verify all expected fields are present
        for r in results:
            assert "name" in r
            assert "location" in r
            assert "price_range" in r
            assert "status" in r

    async def test_get_buildings_for_lead_empty(self, neo4j_driver):
        """Returns empty list for lead with no building interests."""
        repo = BuildingRepository(neo4j_driver)
        phone = _phone()

        # Create lead with no buildings
        async with neo4j_driver.session() as session:
            await session.run(
                "CREATE (l:Lead {phone: $phone})",
                phone=phone,
            )

        results = await repo.get_buildings_for_lead(phone)
        assert results == []


# ---------------------------------------------------------------------------
# get_similar_buildings tests
# ---------------------------------------------------------------------------

class TestGetSimilarBuildings:
    async def test_get_similar_buildings(self, neo4j_driver):
        """Returns buildings connected via SIMILAR_TO, excluding known ones."""
        repo = BuildingRepository(neo4j_driver)
        known_id = _bid()
        similar1, similar2 = _bid(), _bid()

        await _create_similar_buildings(neo4j_driver, known_id, [similar1, similar2])

        results = await repo.get_similar_buildings([known_id])

        returned_ids = {r["building_id"] for r in results}
        assert similar1 in returned_ids
        assert similar2 in returned_ids
        assert known_id not in returned_ids  # Known building excluded

    async def test_get_similar_buildings_with_budget_filter(self, neo4j_driver):
        """Budget filter excludes buildings above the max price."""
        repo = BuildingRepository(neo4j_driver)
        known_id = _bid()
        cheap, expensive = _bid(), _bid()

        # cheap has price_min=5M, expensive has price_min=6M
        await _create_similar_buildings(neo4j_driver, known_id, [cheap, expensive])

        # Set budget below expensive building's price_min
        results = await repo.get_similar_buildings([known_id], budget_max=5500000)

        returned_ids = {r["building_id"] for r in results}
        assert cheap in returned_ids
        # expensive (price_min=6M) should be filtered out
        assert expensive not in returned_ids


# ---------------------------------------------------------------------------
# get_unsent_docs tests
# ---------------------------------------------------------------------------

class TestGetUnsentDocs:
    async def test_get_unsent_docs(self, neo4j_driver):
        """Returns documents for a building NOT yet sent to the lead."""
        repo = BuildingRepository(neo4j_driver)
        phone = _phone()
        bid = _bid()

        # Create building with 2 docs
        await _create_building_with_docs(neo4j_driver, bid, ["brochure.pdf", "floorplan.pdf"])

        # Create lead (no SENT_DOCUMENT yet)
        async with neo4j_driver.session() as session:
            await session.run(
                "CREATE (l:Lead {phone: $phone})",
                phone=phone,
            )

        results = await repo.get_unsent_docs(phone, bid)

        assert len(results) == 2
        doc_names = {r["name"] for r in results}
        assert "brochure.pdf" in doc_names
        assert "floorplan.pdf" in doc_names

    async def test_get_unsent_docs_excludes_sent(self, neo4j_driver):
        """Documents already sent to the lead are excluded."""
        repo = BuildingRepository(neo4j_driver)
        phone = _phone()
        bid = _bid()

        await _create_building_with_docs(neo4j_driver, bid, ["brochure.pdf", "floorplan.pdf"])

        # Create lead and mark brochure as sent
        async with neo4j_driver.session() as session:
            await session.run(
                "CREATE (l:Lead {phone: $phone})",
                phone=phone,
            )
            await session.run(
                """
                MATCH (l:Lead {phone: $phone}),
                      (d:Document {name: 'brochure.pdf'})
                MERGE (l)-[:SENT_DOCUMENT]->(d)
                """,
                phone=phone,
            )

        results = await repo.get_unsent_docs(phone, bid)

        assert len(results) == 1
        assert results[0]["name"] == "floorplan.pdf"


# ---------------------------------------------------------------------------
# record_discussed tests
# ---------------------------------------------------------------------------

class TestRecordDiscussed:
    async def test_record_discussed(self, neo4j_driver):
        """record_discussed creates a DISCUSSED relationship."""
        repo = BuildingRepository(neo4j_driver)
        phone = _phone()
        bid = _bid()

        # Create lead and building
        async with neo4j_driver.session() as session:
            await session.run(
                "CREATE (l:Lead {phone: $phone})",
                phone=phone,
            )
            await session.run(
                "CREATE (b:Building {building_id: $bid})",
                bid=bid,
            )

        await repo.record_discussed(phone, bid)

        # Verify relationship exists
        async with neo4j_driver.session() as session:
            result = await session.run(
                """
                MATCH (l:Lead {phone: $phone})-[:DISCUSSED]->(b:Building {building_id: $bid})
                RETURN count(*) AS cnt
                """,
                phone=phone,
                bid=bid,
            )
            record = await result.single()

        assert record["cnt"] == 1

    async def test_record_discussed_idempotent(self, neo4j_driver):
        """Recording discussed twice creates only one relationship (MERGE)."""
        repo = BuildingRepository(neo4j_driver)
        phone = _phone()
        bid = _bid()

        async with neo4j_driver.session() as session:
            await session.run(
                "CREATE (l:Lead {phone: $phone})",
                phone=phone,
            )
            await session.run(
                "CREATE (b:Building {building_id: $bid})",
                bid=bid,
            )

        await repo.record_discussed(phone, bid)
        await repo.record_discussed(phone, bid)  # Second call

        async with neo4j_driver.session() as session:
            result = await session.run(
                """
                MATCH (l:Lead {phone: $phone})-[r:DISCUSSED]->(b:Building {building_id: $bid})
                RETURN count(r) AS cnt
                """,
                phone=phone,
                bid=bid,
            )
            record = await result.single()

        assert record["cnt"] == 1
