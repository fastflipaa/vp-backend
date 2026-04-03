"""Dry-run test for the self-learning system.

Simulates a fake lead conversation, then runs the full learning pipeline:
1. Creates a fake ConversationOutcome
2. Runs ErrorClassifier against fake messages
3. Creates an embedding via OpenAI
4. Checks LessonInjector retrieval
5. Verifies admin endpoints

Run from inside the worker container:
  docker exec <worker> python3 /app/scripts/dry_run_learning.py
"""

import asyncio
import json
import sys
import time

# ── Setup ──
async def main():
    from app.config import settings
    from app.repositories.base import get_driver, close_driver
    from app.repositories.learning_repository import LearningRepository
    from app.services.monitoring.error_classifier import ErrorClassifier
    from app.services.monitoring.embedding_service import EmbeddingService, EmbeddingCircuitOpen
    from app.services.monitoring.lesson_injector import LessonInjector
    from app.services.monitoring.lesson_generator import AutoLessonGenerator
    from app.services.monitoring.alert_manager import AlertManager
    import redis
    import structlog

    logger = structlog.get_logger()
    results = {}

    print("=" * 60)
    print("SELF-LEARNING SYSTEM DRY RUN")
    print("=" * 60)

    # ── Step 0: Connect ──
    driver = await get_driver()
    redis_client = redis.Redis.from_url(settings.redis_cache_url, decode_responses=True)
    learning_repo = LearningRepository(driver)
    alert_mgr = AlertManager(redis_client)
    embedding_svc = EmbeddingService()
    classifier = ErrorClassifier(learning_repo, alert_mgr)
    injector = LessonInjector(learning_repo, embedding_svc)

    print("\n[1/7] Testing ConversationOutcome creation...")
    try:
        outcome_id = await learning_repo.create_conversation_outcome(
            conversation_id="dry-run-test-001",
            outcome_type="auto_scored",
            scores={"repetition": 0.2, "sentiment": 0.8, "intent_alignment": 0.9},
            metadata={"contact_id": "dry-run-test", "state": "QUALIFYING"},
        )
        print(f"  ✓ ConversationOutcome created: {outcome_id}")
        results["outcome"] = "PASS"
    except Exception as e:
        print(f"  ✗ FAILED: {e}")
        results["outcome"] = f"FAIL: {e}"

    print("\n[2/7] Testing AgentError creation...")
    try:
        error_id = await learning_repo.create_agent_error(
            conversation_id="dry-run-test-001",
            error_type="hallucination",
            details="Dry-run test: AI claimed price was $300K but actual is $490K",
            severity="critical",
        )
        print(f"  ✓ AgentError created: {error_id}")
        results["error"] = "PASS"
    except Exception as e:
        print(f"  ✗ FAILED: {e}")
        results["error"] = f"FAIL: {e}"

    print("\n[3/7] Testing ErrorPattern increment...")
    try:
        freq = await learning_repo.increment_error_pattern("hallucination")
        print(f"  ✓ ErrorPattern 'hallucination' frequency: {freq}")
        results["pattern"] = "PASS"
    except Exception as e:
        print(f"  ✗ FAILED: {e}")
        results["pattern"] = f"FAIL: {e}"

    print("\n[4/7] Testing OpenAI Embedding...")
    try:
        vector = await embedding_svc.embed_text(
            "Hola, me interesan los departamentos de lujo en Polanco con vista al parque"
        )
        print(f"  ✓ Embedding generated: {len(vector)} dimensions, first 5: {vector[:5]}")
        results["embedding"] = "PASS"

        # Store embedding
        emb_id = await learning_repo.create_conversation_embedding(
            conversation_id="dry-run-test-001",
            vector=vector,
        )
        print(f"  ✓ Embedding stored in Neo4j: {emb_id}")
        results["embedding_store"] = "PASS"
    except EmbeddingCircuitOpen as e:
        print(f"  ⚠ Circuit open (expected if no API key): {e}")
        results["embedding"] = f"CIRCUIT_OPEN: {e}"
    except Exception as e:
        print(f"  ✗ FAILED: {e}")
        results["embedding"] = f"FAIL: {e}"

    print("\n[5/7] Testing ErrorClassifier...")
    try:
        fake_messages = [
            {"role": "user", "content": "Cuánto cuestan los departamentos en Park Hyatt?", "created_at": "2026-04-03T10:00:00"},
            {"role": "assistant", "content": "Los precios arrancan desde $300,000 USD", "created_at": "2026-04-03T10:00:05"},
            {"role": "user", "content": "Mándame los planos por favor", "created_at": "2026-04-03T10:01:00"},
            {"role": "assistant", "content": "Park Hyatt es un excelente proyecto con muchas amenidades.", "created_at": "2026-04-03T10:01:05"},
        ]
        fake_buildings = [{"building_id": "park-hyatt", "name": "Park Hyatt Residences", "price_from_usd": 490000}]
        fake_lead = {"contact_id": "dry-run-test", "phone": "+5215500000000", "name": "Test Lead", "state": "QUALIFYING", "language": "es"}

        errors = await classifier.classify(fake_lead, fake_messages, fake_buildings)
        print(f"  ✓ Errors detected: {len(errors)}")
        for err in errors:
            print(f"    - {err.get('type')}: {err.get('details', '')[:80]}")
        results["classifier"] = f"PASS ({len(errors)} errors)"
    except Exception as e:
        print(f"  ✗ FAILED: {e}")
        results["classifier"] = f"FAIL: {e}"

    print("\n[6/7] Testing LessonInjector retrieval...")
    try:
        context = await injector.get_learning_context(
            contact_id="dry-run-test",
            building_id="park-hyatt",
            state="QUALIFYING",
            current_message="Cuánto cuestan los departamentos?",
        )
        if context:
            print(f"  ✓ Learning context generated ({len(context)} chars):")
            print(f"    {context[:200]}...")
        else:
            print("  ✓ No lessons to inject yet (expected — no approved lessons exist)")
        results["injector"] = "PASS"
    except Exception as e:
        print(f"  ✗ FAILED: {e}")
        results["injector"] = f"FAIL: {e}"

    print("\n[7/7] Testing LessonLearned creation (manual candidate)...")
    try:
        lesson_id = await learning_repo.create_lesson_learned(
            rule="Double-check all price claims against building data before responding",
            why="Dry-run test: AI incorrectly stated Park Hyatt prices",
            severity="critical",
            confidence=0.5,
            building_id="park-hyatt",
        )
        print(f"  ✓ LessonLearned candidate created: {lesson_id}")
        results["lesson"] = "PASS"
    except Exception as e:
        print(f"  ✗ FAILED: {e}")
        results["lesson"] = f"FAIL: {e}"

    # ── Cleanup ──
    print("\n" + "=" * 60)
    print("CLEANING UP dry-run test data from Neo4j...")
    try:
        async with driver.session() as session:
            await session.run(
                "MATCH (n) WHERE n.conversation_id = 'dry-run-test-001' "
                "OR (n:LessonLearned AND n.why CONTAINS 'Dry-run test') "
                "OR (n:ErrorPattern AND n.pattern_name = 'hallucination' AND n.frequency <= 1) "
                "DETACH DELETE n"
            )
        print("  ✓ Test data cleaned up")
    except Exception as e:
        print(f"  ⚠ Cleanup issue (non-fatal): {e}")

    await close_driver()

    # ── Summary ──
    print("\n" + "=" * 60)
    print("DRY RUN RESULTS")
    print("=" * 60)
    passed = sum(1 for v in results.values() if v.startswith("PASS"))
    total = len(results)
    for k, v in results.items():
        status = "✓" if v.startswith("PASS") else "✗" if v.startswith("FAIL") else "⚠"
        print(f"  {status} {k}: {v}")
    print(f"\n  {passed}/{total} passed")
    print("=" * 60)

    return passed == total


if __name__ == "__main__":
    success = asyncio.run(main())
    sys.exit(0 if success else 1)
