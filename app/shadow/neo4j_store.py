"""Neo4j storage for shadow comparison results.

Stores each comparison as a :ShadowComparison node (separate label from
production :Lead/:Interaction data to avoid pollution).
"""

from __future__ import annotations

import json

import structlog

logger = structlog.get_logger()


def store_comparison(driver, comparison: dict) -> None:
    """Store a shadow comparison result as a :ShadowComparison node in Neo4j.

    Args:
        driver: An already-created neo4j.Driver instance.
        comparison: The comparison dict from comparator.compare_results().
    """
    query = """
    CREATE (sc:ShadowComparison {
        trace_id: $trace_id,
        contact_id: $contact_id,
        message_id: $message_id,
        agreement_rate: $agreement_rate,
        agreements: $agreements,
        disagreements: $disagreements,
        known_improvements: $known_improvements,
        comparisons: $comparisons_json,
        created_at: datetime()
    })
    """

    with driver.session() as session:
        session.run(
            query,
            trace_id=comparison["trace_id"],
            contact_id=comparison["contact_id"],
            message_id=comparison["message_id"],
            agreement_rate=comparison["agreement_rate"],
            agreements=comparison["agreements"],
            disagreements=comparison["disagreements"],
            known_improvements=comparison["known_improvements"],
            comparisons_json=json.dumps(comparison["comparisons"]),
        )

    logger.info(
        "shadow_comparison_stored",
        trace_id=comparison["trace_id"],
        agreement_rate=comparison["agreement_rate"],
    )
