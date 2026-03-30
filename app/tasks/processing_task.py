"""Monolithic Celery task for the full message processing pipeline.

Orchestrates the complete flow from gate output to GHL delivery:
1. Resolve lead state from Neo4j
2. Check human agent activity (GHL)
3. Classify message (Claude Haiku -- if needed for new leads)
4. Enrich from GHL (contact + notes + emails)
5. Route to state processor
6. Generate AI response (Claude Sonnet -- via processor)
7. Run state machine transition
8. Persist new state + conversation to Neo4j
9. Deliver response via GHL

Design decisions:
- Monolithic (NOT chain): simpler fallback responses, shared state across stages
- max_retries=0: no Celery-level retries -- all retries are internal (tenacity for GHL, max_retries=3 for Claude)
- queue="processing": separate from "celery" queue used by gates, allows independent scaling
- acks_late=True + reject_on_worker_lost=True: crash recovery -- message returns to queue
- asyncio.run(): safe in prefork pool (no existing event loop). NOT safe in gevent/eventlet.
- Every stage has try/except with fail-open behavior
- CircuitOpenError triggers fallback response path

TODO (Phase 17): After canary validation, update gate_tasks.py to call:
    process_message.delay(payload, trace_id)
when overall_decision == PASS. Currently in shadow mode -- gates log only.
"""

from __future__ import annotations

import asyncio
import json
import time

import redis
import structlog
from celery import signals

from app.celery_app import celery_app
from app.config import settings

logger = structlog.get_logger()

# Per-worker Redis client (initialized on worker_process_init)
_redis_client: redis.Redis | None = None


@signals.worker_process_init.connect
def init_processing_worker(**kwargs) -> None:
    """Initialize per-worker Redis for circuit breakers + DLQ."""
    global _redis_client
    pool = redis.ConnectionPool.from_url(
        settings.redis_cache_url,
        max_connections=20,
        decode_responses=True,
    )
    _redis_client = redis.Redis(connection_pool=pool)
    logger.info("processing_worker_redis_initialized")


@celery_app.task(
    name="processing.process_message",
    bind=True,
    max_retries=0,
    acks_late=True,
    reject_on_worker_lost=True,
    queue="processing",
)
def process_message(self, payload: dict, trace_id: str) -> dict:
    """Full message processing pipeline.

    Stages:
    1. Resolve lead state from Neo4j
    2. Check human agent activity (GHL)
    3. Classify message (Claude Haiku -- if needed)
    4. Enrich from GHL (contact + notes + emails)
    5. Route to state processor
    6. Generate AI response (Claude Sonnet -- via processor)
    7. Run state machine transition
    8. Persist new state + conversation to Neo4j
    9. Deliver response via GHL

    On Claude failure at any stage: send fallback via GHL.
    On GHL delivery failure: write to DLQ for retry.
    On Neo4j failure: log error but still attempt delivery (fail-open).
    """
    start_time = time.time()

    async def _run():
        # Import all dependencies inside the task to avoid circular imports
        from app.api.schemas import InboundPayload
        from app.processors.base import ProcessorResult
        from app.repositories.base import get_driver
        from app.repositories.building_repository import BuildingRepository
        from app.repositories.conversation_repository import ConversationRepository
        from app.repositories.lead_repository import LeadRepository
        from app.services.circuit_breaker import CircuitOpenError
        from app.services.claude_service import ClaudeService
        from app.services.ghl_enrichment import GHLEnrichmentService
        from app.services.human_agent_detector import HumanAgentDetector
        from app.services.processor_router import ProcessorRouter
        from app.services.response_delivery import ResponseDeliveryService
        from app.services.state_resolver import StateResolver
        from app.prompts.builder import PromptBuilder
        from app.state_machine.conversation_sm import ConversationSM

        # ── Parse and validate payload ──
        try:
            inbound = InboundPayload(**payload)
        except Exception as e:
            logger.error("invalid_payload", trace_id=trace_id, error=str(e))
            return {"status": "error", "reason": "invalid_payload"}

        phone = inbound.phone
        contact_id = inbound.contactId
        message = inbound.message

        if not phone and not contact_id:
            logger.warning("missing_identifiers", trace_id=trace_id)
            return {"status": "error", "reason": "no_phone_or_contact_id"}

        # ── Initialize services ──
        driver = await get_driver()
        lead_repo = LeadRepository(driver)
        conv_repo = ConversationRepository(driver)
        building_repo = BuildingRepository(driver)
        state_resolver = StateResolver(lead_repo)
        processor_router = ProcessorRouter()
        claude_service = ClaudeService(_redis_client)
        enrichment_service = GHLEnrichmentService()
        human_detector = HumanAgentDetector()
        delivery_service = ResponseDeliveryService(_redis_client)
        prompt_builder = PromptBuilder("v1")

        # ── Stage 1: Resolve State ──
        try:
            resolved = await state_resolver.resolve(
                phone=phone,
                is_auto_trigger=inbound.isAutoTrigger,
                message_type=inbound.messageType,
            )
            current_state = resolved["state"]
            lead_data = resolved["lead_data"]
            is_new_lead = resolved["is_new_lead"]
        except Exception:
            logger.exception("state_resolve_failed", trace_id=trace_id)
            current_state = "GREETING"
            lead_data = {}
            is_new_lead = True

        logger.info(
            "state_resolved",
            trace_id=trace_id,
            state=current_state,
            is_new=is_new_lead,
        )

        # ── Stage 2: Human Agent Check ──
        conversation_ctx: dict = {}
        try:
            is_human_active, human_reason = await human_detector.is_human_active(
                contact_id=contact_id,
                conversation_id=inbound.conversationId,
            )
            if is_human_active:
                logger.info(
                    "human_agent_active",
                    trace_id=trace_id,
                    reason=human_reason,
                )
                return {"status": "skipped", "reason": f"human_active:{human_reason}"}
            # After is_human_active, the detector caches the fetched messages
            conversation_ctx = human_detector.get_conversation_context()
        except Exception:
            logger.exception("human_check_failed", trace_id=trace_id)
            # Fail-open: proceed without human check

        # ── Stage 3: Classification (if needed) ──
        classification = None
        if current_state == "GREETING" and is_new_lead:
            try:
                classification = await claude_service.classify(
                    message=message,
                    phone=phone,
                    contact_context=json.dumps(lead_data) if lead_data else "",
                )
                if classification.get("classification") in ("broker", "advertiser"):
                    confidence = classification.get("confidence", 0)
                    if confidence >= 0.8:
                        logger.info(
                            "non_lead_detected",
                            trace_id=trace_id,
                            classification=classification,
                        )
                        # Set broker state in Neo4j
                        try:
                            await lead_repo.save_state(contact_id, "BROKER")
                        except Exception:
                            logger.exception(
                                "broker_state_save_failed", trace_id=trace_id
                            )
                        return {
                            "status": "classified_non_lead",
                            "classification": classification,
                        }
            except CircuitOpenError:
                logger.warning("classification_circuit_open", trace_id=trace_id)
                # Proceed without classification (fail-open)
            except Exception:
                logger.exception("classification_failed", trace_id=trace_id)

        # ── Stage 4: GHL Enrichment ──
        enriched_context: dict = {}
        try:
            enriched_context = await enrichment_service.enrich(
                contact_id=contact_id,
                conversation_id=inbound.conversationId,
            )
        except Exception:
            logger.exception("enrichment_failed", trace_id=trace_id)
            # Fail-open: proceed without enrichment

        # ── Stage 5: Route to Processor ──
        processor_class = processor_router.get_processor(current_state)
        if processor_class is None:
            logger.warning(
                "no_processor_for_state",
                trace_id=trace_id,
                state=current_state,
            )
            return {"status": "skipped", "reason": f"no_processor:{current_state}"}

        # Instantiate processor with dependencies
        processor_kwargs = {
            "claude_service": claude_service,
            "prompt_builder": prompt_builder,
            "lead_repo": lead_repo,
            "conversation_repo": conv_repo,
        }
        if current_state in ("QUALIFYING", "BUILDING_INFO"):
            processor_kwargs["building_repo"] = building_repo

        processor = processor_class(**processor_kwargs)

        # Merge lead_data with additional context
        lead_data_with_context = {
            **lead_data,
            "phone": phone,
            "contact_id": contact_id,
            "channel": inbound.channel,
            "is_new_lead": is_new_lead,
            "is_auto_trigger": inbound.isAutoTrigger,
        }

        # ── Stage 6: Process (Claude call happens inside processor) ──
        result: ProcessorResult
        try:
            result = await processor.process(
                message=message,
                lead_data=lead_data_with_context,
                enriched_context=enriched_context,
                conversation_context=conversation_ctx,
                trace_id=trace_id,
            )
        except CircuitOpenError:
            logger.warning(
                "claude_circuit_open_in_processor", trace_id=trace_id
            )
            # Send fallback response
            await delivery_service.deliver_fallback(
                contact_id=contact_id,
                phone=phone,
                language=lead_data.get("language", "es"),
                channel=inbound.channel,
                trace_id=trace_id,
                conversation_repo=conv_repo,
            )
            return {"status": "fallback_sent", "reason": "claude_circuit_open"}
        except Exception:
            logger.exception(
                "processor_failed",
                trace_id=trace_id,
                state=current_state,
            )
            # Send fallback response
            await delivery_service.deliver_fallback(
                contact_id=contact_id,
                phone=phone,
                language=lead_data.get("language", "es"),
                channel=inbound.channel,
                trace_id=trace_id,
                conversation_repo=conv_repo,
            )
            return {"status": "fallback_sent", "reason": "processor_error"}

        # Check if processor flagged fallback needed (circuit open inside processor)
        if result.metadata.get("fallback_needed"):
            await delivery_service.deliver_fallback(
                contact_id=contact_id,
                phone=phone,
                language=lead_data.get("language", "es"),
                channel=inbound.channel,
                trace_id=trace_id,
                conversation_repo=conv_repo,
            )
            return {"status": "fallback_sent", "reason": "processor_fallback"}

        # ── Stage 7: State Machine Transition ──
        if result.new_state:
            try:
                sm = ConversationSM.from_persisted_state(current_state, contact_id)
                # Attempt transition via the SM with processor metadata
                sm.advance(**result.metadata)
                new_state = sm.model.state  # Updated by after_transition hook
                await lead_repo.save_state(contact_id, new_state)
                logger.info(
                    "state_transitioned",
                    trace_id=trace_id,
                    from_state=current_state,
                    to_state=new_state,
                )
            except Exception:
                # SM transition failed -- save the processor's suggested state directly
                logger.exception(
                    "sm_transition_failed",
                    trace_id=trace_id,
                    suggested_state=result.new_state,
                )
                try:
                    await lead_repo.save_state(contact_id, result.new_state)
                except Exception:
                    logger.exception(
                        "direct_state_save_failed", trace_id=trace_id
                    )

        # Save sub-state updates (qualifying progress)
        if result.sub_state_update:
            try:
                await lead_repo.save_qualification_data(
                    contact_id, result.sub_state_update
                )
            except Exception:
                logger.exception("sub_state_save_failed", trace_id=trace_id)

        # ── Stage 8: Deliver Response ──
        delivery_result: dict = {"status": "no_response"}
        if result.response_text:
            try:
                delivery_result = await delivery_service.deliver(
                    contact_id=contact_id,
                    phone=phone,
                    response_text=result.response_text,
                    channel=inbound.channel,
                    trace_id=trace_id,
                    inbound_message=message,
                    conversation_repo=conv_repo,
                    lead_phone=phone,
                    lead_email=lead_data.get("email", ""),
                )
            except Exception:
                logger.exception("delivery_failed", trace_id=trace_id)
                delivery_result = {"status": "delivery_error"}

        duration_ms = (time.time() - start_time) * 1000
        logger.info(
            "pipeline_complete",
            trace_id=trace_id,
            state=current_state,
            new_state=result.new_state,
            duration_ms=round(duration_ms, 2),
            delivery_status=delivery_result.get("status"),
        )

        return {
            "status": "processed",
            "trace_id": trace_id,
            "state": current_state,
            "new_state": result.new_state,
            "delivery": delivery_result,
            "duration_ms": round(duration_ms, 2),
        }

    # Run the async pipeline in asyncio
    # Safe in prefork Celery workers (no existing event loop)
    return asyncio.run(_run())
