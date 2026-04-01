"""Building repository for async Neo4j operations.

Handles property data lookups, GraphRAG similarity queries (via
:SIMILAR_TO relationships), unsent document checks, and discussion
tracking.

Each method creates a fresh ``session()`` per operation.
"""

from __future__ import annotations

from typing import Any

import structlog
from neo4j import AsyncDriver

logger = structlog.get_logger()


class BuildingRepository:
    """Async repository for :Building and :Document node operations."""

    def __init__(self, driver: AsyncDriver) -> None:
        self._driver = driver

    # --- Building lookups ---

    async def get_buildings_for_lead(self, phone: str) -> list[dict[str, Any]]:
        """Get all buildings a lead has expressed interest in.

        Returns list of dicts with: name, building_id, location,
        price_range, status.
        """
        async with self._driver.session() as session:
            result = await session.execute_read(
                self._get_buildings_for_lead_tx, phone
            )
            logger.debug(
                "buildings_for_lead_read",
                phone=phone[-4:],
                count=len(result),
            )
            return result

    @staticmethod
    async def _get_buildings_for_lead_tx(tx, phone: str) -> list[dict[str, Any]]:
        result = await tx.run(
            """
            MATCH (l:Lead {phone: $phone})-[:INTERESTED_IN]->(b:Building)
            RETURN b.name AS name,
                   b.building_id AS building_id,
                   b.location AS location,
                   b.price_range AS price_range,
                   b.status AS status
            """,
            phone=phone,
        )
        return [dict(r) async for r in result]

    # --- GraphRAG similarity ---

    async def get_similar_buildings(
        self,
        building_ids: list[str],
        budget_max: float | None = None,
    ) -> list[dict[str, Any]]:
        """Find similar buildings via :SIMILAR_TO relationships.

        Uses the GraphRAG pattern from the n8n Re-engagement Handler
        (Section 3.4 of the audit). Returns up to 5 recommendations
        that the lead has NOT already seen, ordered by similarity score.

        Args:
            building_ids: IDs of buildings the lead already knows about.
            budget_max: Optional maximum budget filter.
        """
        async with self._driver.session() as session:
            result = await session.execute_read(
                self._get_similar_buildings_tx, building_ids, budget_max
            )
            logger.debug(
                "similar_buildings_found",
                known_ids=building_ids,
                found=len(result),
            )
            return result

    @staticmethod
    async def _get_similar_buildings_tx(
        tx, building_ids: list[str], budget_max: float | None
    ) -> list[dict[str, Any]]:
        # Build query with optional budget filter
        budget_clause = ""
        params: dict[str, Any] = {"known_ids": building_ids}

        if budget_max is not None:
            budget_clause = "AND rec.price_min <= $budget"
            params["budget"] = budget_max

        query = f"""
        MATCH (known:Building)-[sim:SIMILAR_TO]->(rec:Building)
        WHERE known.building_id IN $known_ids
          AND NOT rec.building_id IN $known_ids
          {budget_clause}
        RETURN rec.name AS name,
               rec.building_id AS building_id,
               rec.price_range AS price_range,
               sim.score AS similarity
        ORDER BY similarity DESC
        LIMIT 5
        """

        result = await tx.run(query, **params)
        return [dict(r) async for r in result]

    # --- Document tracking ---

    async def get_unsent_docs(
        self, phone: str, building_id: str
    ) -> list[dict[str, Any]]:
        """Find documents for a building that have NOT been sent to the lead.

        Matches the n8n Send Building Docs sub-workflow (Section 3.7).
        """
        async with self._driver.session() as session:
            result = await session.execute_read(
                self._get_unsent_docs_tx, phone, building_id
            )
            logger.debug(
                "unsent_docs_found",
                phone=phone[-4:],
                building_id=building_id,
                count=len(result),
            )
            return result

    @staticmethod
    async def _get_unsent_docs_tx(
        tx, phone: str, building_id: str
    ) -> list[dict[str, Any]]:
        result = await tx.run(
            """
            MATCH (b:Building {building_id: $bid})-[:HAS_DOCUMENT]->(d:Document)
            WHERE NOT EXISTS {
                MATCH (l:Lead {phone: $phone})-[:SENT_DOCUMENT]->(d)
            }
            RETURN d.name AS name, d.url AS url, d.type AS type
            """,
            bid=building_id,
            phone=phone,
        )
        return [dict(r) async for r in result]

    async def get_unsent_docs_by_name(
        self, phone: str, building_name: str
    ) -> list[dict[str, Any]]:
        """Find unsent documents for a building looked up by name.

        Variant of get_unsent_docs that resolves building by name instead
        of building_id. Used by the doc delivery task which receives a
        building name from the Claude qualifying response.
        """
        async with self._driver.session() as session:
            result = await session.execute_read(
                self._get_unsent_docs_by_name_tx, phone, building_name
            )
            logger.debug(
                "unsent_docs_by_name_found",
                phone=phone[-4:],
                building_name=building_name,
                count=len(result),
            )
            return result

    @staticmethod
    async def _get_unsent_docs_by_name_tx(
        tx, phone: str, building_name: str
    ) -> list[dict[str, Any]]:
        result = await tx.run(
            """
            MATCH (b:Building)-[:HAS_DOCUMENT]->(d:Document)
            WHERE toLower(b.name) = toLower($bname)
            AND NOT EXISTS {
                MATCH (l:Lead {phone: $phone})-[:SENT_DOCUMENT]->(d)
            }
            RETURN d.name AS name, d.url AS url, d.type AS type, b.building_id AS building_id
            """,
            bname=building_name,
            phone=phone,
        )
        return [dict(r) async for r in result]

    async def record_sent_docs(
        self, phone: str, doc_urls: list[str]
    ) -> None:
        """Create :SENT_DOCUMENT relationships between Lead and Documents.

        Uses doc url as the lookup key since building_id is not always
        available at delivery time.
        """
        if not doc_urls:
            return
        async with self._driver.session() as session:
            await session.execute_write(
                self._record_sent_docs_tx, phone, doc_urls
            )
            logger.info(
                "sent_docs_recorded",
                phone=phone[-4:],
                count=len(doc_urls),
            )

    @staticmethod
    async def _record_sent_docs_tx(
        tx, phone: str, doc_urls: list[str]
    ) -> None:
        await tx.run(
            """
            MATCH (l:Lead {phone: $phone})
            MATCH (d:Document) WHERE d.url IN $urls
            MERGE (l)-[:SENT_DOCUMENT]->(d)
            """,
            phone=phone,
            urls=doc_urls,
        )

    # --- Discussion tracking ---

    async def record_discussed(self, phone: str, building_id: str) -> None:
        """Record that a building was discussed with a lead.

        Creates a :DISCUSSED relationship between the Lead and Building.
        """
        async with self._driver.session() as session:
            await session.execute_write(
                self._record_discussed_tx, phone, building_id
            )
            logger.info(
                "building_discussed_recorded",
                phone=phone[-4:],
                building_id=building_id,
            )

    @staticmethod
    async def _record_discussed_tx(
        tx, phone: str, building_id: str
    ) -> None:
        await tx.run(
            """
            MATCH (l:Lead {phone: $phone}), (b:Building {building_id: $bid})
            MERGE (l)-[:DISCUSSED]->(b)
            """,
            phone=phone,
            bid=building_id,
        )
