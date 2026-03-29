"""Celery tasks for shadow mode reporting.

Provides a daily summary of shadow comparison results, querying
:ShadowComparison nodes from Neo4j for the last 24 hours.
"""

from __future__ import annotations

import structlog
from neo4j import GraphDatabase

from app.celery_app import celery_app
from app.config import settings

logger = structlog.get_logger()


@celery_app.task(name="reporting.daily_shadow_summary")
def daily_shadow_summary() -> dict:
    """Generate daily shadow comparison summary from Neo4j.

    Queries all :ShadowComparison nodes created in the last 24 hours
    and produces aggregate statistics: agreement rate, total disagreements,
    known improvements, and a sample of disagreement trace IDs.

    Runs via Celery Beat at 8:00 AM CDMX daily.
    """
    driver = GraphDatabase.driver(
        settings.NEO4J_URI,
        auth=("neo4j", settings.NEO4J_PASSWORD),
    )

    try:
        with driver.session() as session:
            result = session.run("""
                MATCH (sc:ShadowComparison)
                WHERE sc.created_at > datetime() - duration('PT24H')
                RETURN
                    count(sc) AS total_comparisons,
                    avg(sc.agreement_rate) AS avg_agreement_rate,
                    sum(sc.disagreements) AS total_disagreements,
                    sum(sc.known_improvements) AS total_known_improvements,
                    collect(CASE WHEN sc.disagreements > 0 THEN sc.trace_id ELSE null END)[..10] AS sample_disagreement_ids
            """)
            record = result.single()

        total_comparisons = record["total_comparisons"] if record else 0
        avg_agreement_rate = record["avg_agreement_rate"] if record else 0.0
        total_disagreements = record["total_disagreements"] if record else 0
        total_known_improvements = record["total_known_improvements"] if record else 0
        sample_disagreement_ids = record["sample_disagreement_ids"] if record else []

        # Filter null values from sample IDs (CASE returns null for non-disagreements)
        sample_disagreement_ids = [
            tid for tid in sample_disagreement_ids if tid is not None
        ]

        report = {
            "period": "last_24h",
            "total_comparisons": total_comparisons,
            "avg_agreement_rate": round(avg_agreement_rate, 4) if avg_agreement_rate else 0.0,
            "total_disagreements": total_disagreements,
            "total_known_improvements": total_known_improvements,
            "sample_disagreement_ids": sample_disagreement_ids,
        }

        logger.info(
            "daily_shadow_report",
            **report,
        )

        return report

    finally:
        driver.close()
