"""Conversation scorer -- runs every 30 minutes via Celery Beat.

For each lead with recent interactions:
1. Scores conversation quality (repetition, sentiment, intent alignment)
2. Creates ConversationOutcome nodes in Neo4j
3. Runs ErrorClassifier to detect and persist errors
4. Generates conversation embeddings via OpenAI
5. Infers lead satisfaction from behavioral signals

Follows the same pattern as conversation_quality_scan.py:
- Per-worker Redis via worker_process_init
- async _run() wrapped in asyncio.run()
- close_driver() in finally block
"""

from __future__ import annotations

import asyncio
from datetime import datetime, timezone

import redis
import structlog
from celery import signals
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from app.celery_app import celery_app
from app.config import settings
from app.services.monitoring.conversation_scanner import (
    ALL_NEGATIVE,
    ALL_POSITIVE,
    INTENT_PATTERNS,
    MIN_MESSAGES_FOR_REPETITION,
    REPETITION_THRESHOLD,
)

logger = structlog.get_logger()

# Per-worker Redis client
_redis_client: redis.Redis | None = None


@signals.worker_process_init.connect
def init_conversation_scorer_worker(**kwargs) -> None:
    """Initialize per-worker Redis connection pool on process startup."""
    global _redis_client
    pool = redis.ConnectionPool.from_url(
        settings.redis_cache_url,
        max_connections=20,
        decode_responses=True,
    )
    _redis_client = redis.Redis(connection_pool=pool)
    logger.info("conversation_scorer_worker_redis_initialized")


@celery_app.task(name="scheduled.conversation_scorer", queue="celery")
def conversation_scorer() -> dict:
    """Score recent conversations and classify errors.

    Returns summary dict with counts.
    """

    async def _run() -> dict:
        import re

        from app.repositories.base import close_driver, get_driver
        from app.repositories.learning_repository import LearningRepository
        from app.services.monitoring.alert_manager import AlertManager
        from app.services.monitoring.embedding_service import (
            EmbeddingCircuitOpen,
            EmbeddingService,
        )
        from app.services.monitoring.error_classifier import ErrorClassifier

        summary = {
            "leads_scored": 0,
            "outcomes_created": 0,
            "errors_classified": 0,
            "embeddings_created": 0,
            "satisfaction_inferred": 0,
        }

        try:
            driver = await get_driver()
            learning_repo = LearningRepository(driver)
            alert_mgr = AlertManager(_redis_client)
            classifier = ErrorClassifier(learning_repo, alert_mgr)
            embedding_svc = EmbeddingService()

            # Query active leads with recent interactions (last 30 min)
            async with driver.session() as session:
                result = await session.run(
                    """
                    MATCH (l:Lead)-[:HAS_INTERACTION]->(i:Interaction)
                    WHERE i.created_at > datetime() - duration({minutes: 30})
                    AND l.current_state <> 'BROKER'
                    AND l.current_state <> 'CLOSED'
                    WITH DISTINCT l
                    RETURN l.ghl_contact_id AS contact_id,
                           l.phone AS phone,
                           l.name AS name,
                           l.language AS language,
                           l.current_state AS state
                    LIMIT 50
                    """,
                )
                leads = [dict(r) async for r in result]

            if not leads:
                logger.info("conversation_scorer.no_active_leads")
                return summary

            for lead in leads:
                contact_id = lead.get("contact_id", "")
                phone = lead.get("phone", "")
                state = lead.get("state", "unknown")
                if not phone:
                    continue

                summary["leads_scored"] += 1

                # Fetch last 10 interactions
                async with driver.session() as session:
                    result = await session.run(
                        """
                        MATCH (l:Lead {phone: $phone})-[:HAS_INTERACTION]->(i:Interaction)
                        RETURN i.role AS role, i.content AS content,
                               i.created_at AS created_at
                        ORDER BY i.created_at DESC
                        LIMIT 10
                        """,
                        phone=phone,
                    )
                    interactions = [dict(r) async for r in result]

                if not interactions:
                    continue

                # Fetch interested buildings
                async with driver.session() as session:
                    result = await session.run(
                        """
                        MATCH (l:Lead {phone: $phone})-[:INTERESTED_IN]->(b:Building)
                        RETURN b.name AS name,
                               b.building_id AS building_id,
                               b.price_min_usd AS price_min_usd,
                               b.price_max_usd AS price_max_usd,
                               b.total_floors AS total_floors,
                               b.total_units AS total_units
                        """,
                        phone=phone,
                    )
                    buildings = [dict(r) async for r in result]

                # Separate messages
                user_msgs = [
                    i["content"]
                    for i in interactions
                    if i.get("role") == "user" and i.get("content")
                ]
                assistant_msgs = [
                    i["content"]
                    for i in interactions
                    if i.get("role") == "assistant" and i.get("content")
                ]

                # --- Scoring ---

                # Repetition score
                rep_score = 0.0
                if len(assistant_msgs) >= MIN_MESSAGES_FOR_REPETITION:
                    try:
                        vectorizer = TfidfVectorizer(stop_words=None, max_features=500)
                        tfidf_matrix = vectorizer.fit_transform(assistant_msgs)
                        sim_matrix = cosine_similarity(tfidf_matrix)
                        max_sim = 0.0
                        n = len(assistant_msgs)
                        for i in range(n):
                            for j in range(i + 1, n):
                                if sim_matrix[i][j] > max_sim:
                                    max_sim = sim_matrix[i][j]
                        rep_score = max_sim
                    except Exception:
                        pass

                # Sentiment score
                sent_score = 0.5  # neutral default
                if user_msgs:
                    scores = []
                    for msg in user_msgs:
                        msg_lower = msg.lower()
                        pos = sum(1 for kw in ALL_POSITIVE if kw in msg_lower)
                        neg = sum(1 for kw in ALL_NEGATIVE if kw in msg_lower)
                        if pos + neg > 0:
                            scores.append(pos / (pos + neg))
                        else:
                            scores.append(0.5)
                    sent_score = sum(scores) / len(scores)

                # Intent alignment score
                intent_score = 1.0
                if user_msgs and assistant_msgs:
                    checked = 0
                    ignored = 0
                    for umsg in user_msgs:
                        u_lower = umsg.lower()
                        for pattern in INTENT_PATTERNS:
                            if re.search(pattern["lead_regex"], u_lower, re.IGNORECASE):
                                checked += 1
                                # Check latest assistant msg for response
                                a_lower = assistant_msgs[0].lower()
                                addressed = any(
                                    re.search(ind, a_lower, re.IGNORECASE)
                                    for ind in pattern["response_must_contain"]
                                )
                                if not addressed:
                                    ignored += 1
                    if checked > 0:
                        intent_score = 1.0 - (ignored / checked)

                # Create ConversationOutcome
                latest_time = str(interactions[0].get("created_at", "unknown"))
                conversation_id = f"{contact_id}:{latest_time}"

                try:
                    await learning_repo.create_conversation_outcome(
                        conversation_id=conversation_id,
                        outcome_type="auto_scored",
                        scores={
                            "repetition": round(rep_score, 3),
                            "sentiment": round(sent_score, 3),
                            "intent_alignment": round(intent_score, 3),
                        },
                        metadata={
                            "contact_id": contact_id,
                            "phone": phone[-4:],
                            "state": state,
                        },
                    )
                    summary["outcomes_created"] += 1
                except Exception:
                    logger.exception(
                        "conversation_scorer.outcome_failed",
                        contact_id=contact_id,
                    )

                # Run ErrorClassifier
                try:
                    lead_data = {
                        "contact_id": contact_id,
                        "phone": phone,
                        "name": lead.get("name", ""),
                        "state": state,
                        "language": lead.get("language", ""),
                    }
                    errors = await classifier.classify(
                        lead_data, interactions, buildings
                    )
                    summary["errors_classified"] += len(errors)
                except Exception:
                    logger.exception(
                        "conversation_scorer.classify_failed",
                        contact_id=contact_id,
                    )

                # Create Embedding
                try:
                    # Concatenate last 5 messages (user + assistant)
                    combined_msgs = []
                    for ix in interactions[:5]:
                        role = ix.get("role", "unknown")
                        content = ix.get("content", "")
                        if content:
                            combined_msgs.append(f"{role}: {content}")
                    combined_text = "\n".join(combined_msgs)

                    if combined_text.strip():
                        vector = await embedding_svc.embed_text(combined_text)
                        await learning_repo.create_conversation_embedding(
                            conversation_id=conversation_id,
                            vector=vector,
                        )
                        summary["embeddings_created"] += 1
                except EmbeddingCircuitOpen:
                    logger.warning(
                        "conversation_scorer.embedding_circuit_open",
                        contact_id=contact_id,
                    )
                except Exception:
                    logger.exception(
                        "conversation_scorer.embedding_failed",
                        contact_id=contact_id,
                    )

                # Infer LeadSatisfaction
                try:
                    signals_data = {}

                    # Quick replies signal (user responded within 5 min of AI)
                    quick_reply_count = 0
                    total_response_pairs = 0
                    for idx in range(len(interactions) - 1):
                        curr = interactions[idx]
                        prev = interactions[idx + 1]
                        if (
                            curr.get("role") == "user"
                            and prev.get("role") == "assistant"
                            and curr.get("created_at")
                            and prev.get("created_at")
                        ):
                            total_response_pairs += 1
                            try:
                                curr_dt = curr["created_at"]
                                prev_dt = prev["created_at"]
                                # Neo4j datetime objects support comparison
                                if hasattr(curr_dt, "to_native"):
                                    curr_dt = curr_dt.to_native()
                                if hasattr(prev_dt, "to_native"):
                                    prev_dt = prev_dt.to_native()
                                diff = (curr_dt - prev_dt).total_seconds()
                                if diff <= 300:  # 5 minutes
                                    quick_reply_count += 1
                            except Exception:
                                pass

                    quick_reply_ratio = (
                        quick_reply_count / total_response_pairs
                        if total_response_pairs > 0
                        else 0.5
                    )
                    signals_data["quick_replies"] = round(quick_reply_ratio, 3)

                    # Ghosting signal (last user msg > 24h old, no reply after last assistant msg)
                    ghosting = False
                    if interactions:
                        last_interaction = interactions[0]
                        if last_interaction.get("role") == "assistant":
                            last_time = last_interaction.get("created_at")
                            if last_time:
                                try:
                                    if hasattr(last_time, "to_native"):
                                        last_time = last_time.to_native()
                                    age_hours = (
                                        datetime.now(timezone.utc) - last_time.replace(tzinfo=timezone.utc)
                                    ).total_seconds() / 3600
                                    if age_hours > 24:
                                        ghosting = True
                                except Exception:
                                    pass
                    signals_data["ghosting"] = ghosting

                    # Positive/negative keywords
                    signals_data["positive_keywords"] = sum(
                        1
                        for msg in user_msgs
                        for kw in ALL_POSITIVE
                        if kw in msg.lower()
                    )
                    signals_data["negative_keywords"] = sum(
                        1
                        for msg in user_msgs
                        for kw in ALL_NEGATIVE
                        if kw in msg.lower()
                    )

                    # Engagement (messages per interaction session)
                    engagement = min(len(user_msgs) / 5.0, 1.0) if user_msgs else 0.0
                    signals_data["engagement"] = round(engagement, 3)

                    # Weighted satisfaction score
                    ghost_penalty = 0.0 if ghosting else 1.0
                    sat_score = (
                        quick_reply_ratio * 0.3
                        + sent_score * 0.5
                        + engagement * 0.2
                    ) * (0.5 if ghosting else 1.0)
                    sat_score = max(0.0, min(1.0, sat_score))

                    await learning_repo.create_lead_satisfaction(
                        lead_phone=phone,
                        score=round(sat_score, 3),
                        source="inferred",
                        signals=signals_data,
                    )
                    summary["satisfaction_inferred"] += 1
                except Exception:
                    logger.exception(
                        "conversation_scorer.satisfaction_failed",
                        contact_id=contact_id,
                    )

            logger.info("conversation_scorer.complete", **summary)
            return summary

        finally:
            await close_driver()

    return asyncio.run(_run())
