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
    0. Idempotency + cooldown guards (prevent duplicates on worker restart)
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

    # ── Stage 0: Idempotency Guard ──
    # Prevents re-delivered tasks (from worker crash + acks_late) from
    # sending duplicate messages. Uses Celery task ID as Redis key.
    task_id = self.request.id
    if _redis_client and task_id:
        idempotency_key = f"task:completed:{task_id}"
        if _redis_client.get(idempotency_key):
            logger.info(
                "task_already_processed",
                task_id=task_id,
                trace_id=trace_id,
            )
            return {"status": "skipped", "reason": "already_processed"}

    # ── Stage 0.5: Per-Contact Outbound Cooldown ──
    # For auto-triggered messages (scheduled tasks), enforce a 1-hour
    # cooldown per contact to prevent ANY code path from spamming.
    contact_id_for_cooldown = payload.get("contactId", "")
    is_auto = payload.get("isAutoTrigger", False)
    if is_auto and contact_id_for_cooldown and _redis_client:
        cooldown_key = f"outbound:cooldown:{contact_id_for_cooldown}"
        if _redis_client.get(cooldown_key):
            logger.info(
                "contact_cooldown_active",
                contact_id=contact_id_for_cooldown,
                trace_id=trace_id,
            )
            return {"status": "skipped", "reason": "contact_cooldown"}
        # Set cooldown BEFORE processing to prevent race conditions
        _redis_client.set(cooldown_key, "1", ex=3600)  # 1h cooldown

    async def _run():
        # Import all dependencies inside the task to avoid circular imports
        from app.api.schemas import InboundPayload
        from app.processors.base import ProcessorResult
        from app.repositories.base import close_driver, get_driver
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
        direction = inbound.direction

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

        # ── Stage 1.5: Language Detection ──
        from app.services.language_detector import detect_language

        detected_lang = "es"
        if message and direction == "inbound":
            try:
                detected_lang = detect_language(message)
                # Store detected language on Lead node
                if contact_id:
                    await lead_repo.save_qualification_data(
                        contact_id, {"language": detected_lang}
                    )
                logger.info(
                    "language_detected",
                    trace_id=trace_id,
                    language=detected_lang,
                )
            except Exception:
                logger.exception("language_detection_failed", trace_id=trace_id)
                detected_lang = (
                    lead_data.get("language", "es") if lead_data else "es"
                )
        else:
            # For outbound/proactive messages, use stored language from lead data
            detected_lang = (
                lead_data.get("language", "es") if lead_data else "es"
            )

        # ── Stage 1.8: Follow-Up Re-Entry Check ──
        # Handles two cases:
        # A) NON_RESPONSIVE lead replies (email drip re-entry) -> QUALIFYING + counter reset
        # B) FOLLOW_UP lead replies (mid-sequence) -> QUALIFYING + counter reset
        drip_reentry_context = {}
        if current_state == "NON_RESPONSIVE" and inbound.direction == "inbound" and inbound.message:
            # Lead is replying after being in email drip -- re-enter qualifying
            try:
                await lead_repo.save_state(contact_id, "QUALIFYING")
                await lead_repo.reset_followup_count(contact_id)
                current_state = "QUALIFYING"
                is_new_lead = False

                drip_reentry_context = {
                    "is_drip_reentry": True,
                    "returning_lead": True,
                }

                logger.info(
                    "drip_reentry.state_reset",
                    trace_id=trace_id,
                    contact_id=contact_id,
                    from_state="NON_RESPONSIVE",
                    to_state="QUALIFYING",
                )
            except Exception:
                logger.exception("drip_reentry.reset_failed", trace_id=trace_id)
        elif current_state == "FOLLOW_UP" and inbound.direction == "inbound" and inbound.message:
            # Lead replied during follow-up sequence -- reset counter, continue in QUALIFYING
            try:
                await lead_repo.reset_followup_count(contact_id)
                await lead_repo.save_state(contact_id, "QUALIFYING")
                current_state = "QUALIFYING"

                logger.info(
                    "followup_midsequence_reply",
                    trace_id=trace_id,
                    contact_id=contact_id,
                )
            except Exception:
                logger.exception("followup_reset_failed", trace_id=trace_id)
        elif current_state == "RE_ENGAGE" and inbound.direction == "inbound" and inbound.message:
            # Lead replied to old-lead outreach -- re-enter qualifying with re-engagement context
            try:
                await lead_repo.save_state(contact_id, "QUALIFYING")
                # Do NOT reset reengagement_count -- it tracks historical attempts
                current_state = "QUALIFYING"
                is_new_lead = False

                drip_reentry_context = {
                    "is_reengagement_reentry": True,
                    "returning_lead": True,
                }

                logger.info(
                    "reengagement_reentry.state_reset",
                    trace_id=trace_id,
                    contact_id=contact_id,
                    from_state="RE_ENGAGE",
                    to_state="QUALIFYING",
                )
            except Exception:
                logger.exception("reengagement_reentry.reset_failed", trace_id=trace_id)

        # ── Stage 2: Human Agent Check ──
        conversation_ctx: dict = {}
        is_human_active = False
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
            # Add structured conversation turns for Claude multi-turn API
            conversation_ctx["structured_turns"] = human_detector.get_structured_turns()
        except Exception:
            logger.exception("human_check_failed", trace_id=trace_id)
            # Fail-open: proceed without human check

        # ── Stage 2.5: Human Re-Entry Check ──
        human_reentry_ctx = {}
        if not is_human_active and contact_id:
            try:
                from app.services.human_reentry import build_human_reentry_context

                human_reentry_ctx = await build_human_reentry_context(
                    contact_id=contact_id,
                    conversation_id=inbound.conversationId,
                    redis_client=_redis_client,
                    claude_service=claude_service,
                )
                if human_reentry_ctx.get("had_human_interaction"):
                    logger.info(
                        "human_reentry_context_built",
                        trace_id=trace_id,
                        agent=human_reentry_ctx.get("agent_name"),
                        messages=human_reentry_ctx.get("messages_during_lock"),
                    )
            except Exception:
                logger.exception("human_reentry_check_failed", trace_id=trace_id)

        # ── Stage 3: Classification (if needed) ──
        classification = None
        if current_state == "GREETING" and is_new_lead:
            try:
                classification = await claude_service.classify(
                    message=message,
                    phone=phone,
                    contact_context=json.dumps(lead_data, default=str) if lead_data else "",
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

        # ── Stage 4.5: Resolve Prompt Version ──
        prompt_version = prompt_builder.get_version(current_state.lower())
        if prompt_version:
            logger.debug(
                "prompt_version_resolved",
                trace_id=trace_id,
                state=current_state,
                prompt_version=prompt_version,
            )

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
        if current_state in ("QUALIFYING", "BUILDING_INFO", "RE_ENGAGE"):
            processor_kwargs["building_repo"] = building_repo

        processor = processor_class(**processor_kwargs)

        # ── Stage 5.5: Lesson Injection (GraphRAG) ──
        if settings.LEARNING_INJECTION_ENABLED:
            try:
                from app.repositories.learning_repository import LearningRepository
                from app.services.monitoring.lesson_injector import LessonInjector
                from app.services.monitoring.embedding_service import EmbeddingService

                learning_repo = LearningRepository(driver)
                embedding_svc = EmbeddingService()
                lesson_injector = LessonInjector(learning_repo, embedding_svc)

                # Determine building_id from lead_data
                _building_id = lead_data.get("building_source")
                if _building_id == "none":
                    _building_id = None

                learning_context = await lesson_injector.get_learning_context(
                    contact_id=contact_id or "",
                    building_id=_building_id,
                    state=current_state,
                    current_message=message or "",
                )

                if learning_context:
                    claude_service.learning_context = learning_context
                    logger.info(
                        "lesson_injection_applied",
                        trace_id=trace_id,
                        context_length=len(learning_context),
                    )
            except Exception:
                logger.exception("lesson_injection_failed", trace_id=trace_id)
                # Fail-open: continue without lessons

        # ── Stage 5.6: Drip Re-Entry Context Injection ──
        if drip_reentry_context.get("is_drip_reentry"):
            drip_text = (
                "CONTEXTO IMPORTANTE: Este lead respondio a un email de seguimiento despues de no responder "
                "a multiples mensajes previos. Reconoce su regreso de forma natural y calida. "
                "No menciones el email directamente -- simplemente retoma la conversacion como si estuvieras "
                "encantada de escuchar de ellos de nuevo. Pregunta como puedes ayudarles."
            )
            if detected_lang == "en":
                drip_text = (
                    "IMPORTANT CONTEXT: This lead replied to a follow-up email after not responding "
                    "to multiple previous messages. Acknowledge their return naturally and warmly. "
                    "Don't mention the email directly -- simply resume the conversation as if you're "
                    "delighted to hear from them again. Ask how you can help."
                )
            claude_service.learning_context = (
                (claude_service.learning_context + "\n\n" + drip_text)
                if claude_service.learning_context
                else drip_text
            )
            logger.info(
                "drip_reentry_context_injected",
                trace_id=trace_id,
                language=detected_lang,
            )

        # -- Stage 5.6b: Re-Engagement Re-Entry Context Injection --
        if drip_reentry_context.get("is_reengagement_reentry"):
            reeng_text = (
                "CONTEXTO IMPORTANTE: Este lead respondio a un mensaje de re-engagement despues de mucho "
                "tiempo sin contacto. Es un lead que habia quedado frio pero mostro interes de nuevo. "
                "Reconoce su regreso con entusiasmo genuino. No menciones que es un re-contacto -- "
                "simplemente retoma naturalmente. Pregunta en que le puedes ayudar y en que tipo de "
                "propiedad esta interesado ahora."
            )
            if detected_lang == "en":
                reeng_text = (
                    "IMPORTANT CONTEXT: This lead replied to a re-engagement message after a long period "
                    "of no contact. They had gone cold but showed interest again. Acknowledge their return "
                    "with genuine enthusiasm. Don't mention re-engagement -- simply resume naturally. "
                    "Ask how you can help and what type of property they're interested in now."
                )
            claude_service.learning_context = (
                (claude_service.learning_context + "\n\n" + reeng_text)
                if claude_service.learning_context
                else reeng_text
            )
            logger.info(
                "reengagement_reentry_context_injected",
                trace_id=trace_id,
                language=detected_lang,
            )

        # -- Stage 5.7: Conversation Summary + Score Context Injection --
        if settings.LEAD_SCORE_ENABLED and contact_id:
            try:
                from app.services.conversation_summary import ConversationSummaryService

                _summary_service = ConversationSummaryService(lead_repo, claude_service)
                summary_text = await _summary_service.get_summary_for_prompt(contact_id)

                if summary_text:
                    claude_service.learning_context = (
                        (claude_service.learning_context + "\n\n" + summary_text)
                        if claude_service.learning_context
                        else summary_text
                    )
                    logger.info(
                        "summary_context_injected",
                        trace_id=trace_id,
                        summary_length=len(summary_text),
                    )

                # Score context injection -- adapt AI behavior based on lead score
                lead_score = lead_data.get("lead_score") if lead_data else None
                if lead_score is None:
                    # Try reading from Neo4j (lead_data might not have it yet)
                    try:
                        scoring_signals = await lead_repo.get_scoring_signals(contact_id)
                        lead_score = scoring_signals.get("current_score")
                    except Exception:
                        pass

                if lead_score is not None:
                    if lead_score >= settings.LEAD_SCORE_HOT_THRESHOLD:
                        score_context = (
                            "PRIORIDAD ALTA: Este lead tiene un score de compra alto ({score}/100). "
                            "Es un prospecto caliente -- ofrece atencion premium, opciones adicionales, "
                            "y facilita la conexion con Fernando lo antes posible. Muestra urgencia amable."
                        ).format(score=lead_score)
                    elif lead_score >= settings.LEAD_SCORE_WARM_THRESHOLD:
                        score_context = (
                            "LEAD TIBIO: Score de compra {score}/100. Buen nivel de interes. "
                            "Manten tono informativo y proactivo. Ofrece informacion sin presionar."
                        ).format(score=lead_score)
                    else:
                        score_context = (
                            "LEAD FRIO: Score de compra {score}/100. Contacto temprano o bajo interes. "
                            "Se amable pero eficiente. No insistas demasiado -- responde lo que preguntan."
                        ).format(score=lead_score)

                    claude_service.learning_context = (
                        (claude_service.learning_context + "\n\n" + score_context)
                        if claude_service.learning_context
                        else score_context
                    )
                    logger.debug(
                        "score_context_injected",
                        trace_id=trace_id,
                        score=lead_score,
                    )
            except Exception:
                logger.exception("summary_score_injection_failed", trace_id=trace_id)
                # Fail-open

        # Merge lead_data with additional context
        lead_data_with_context = {
            **lead_data,
            "phone": phone,
            "contact_id": contact_id,
            "channel": inbound.channel,
            "is_new_lead": is_new_lead,
            "is_auto_trigger": inbound.isAutoTrigger,
            "language": detected_lang,
            "human_reentry": (
                human_reentry_ctx
                if human_reentry_ctx.get("had_human_interaction")
                else None
            ),
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
                prompt_version=prompt_version,
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
                prompt_version=prompt_version,
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
                prompt_version=prompt_version,
            )
            return {"status": "fallback_sent", "reason": "processor_fallback"}

        # Reset learning context after processor completes
        claude_service.learning_context = ""

        # ── Stage 7: State Machine Transition ──
        if result.new_state:
            try:
                sm = ConversationSM.from_persisted_state(current_state, contact_id)
                # Use escalate event for handoffs, advance for normal transitions
                if result.should_handoff or result.new_state == "HANDOFF":
                    sm.escalate()
                else:
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

        # ── Stage 7.5: Pipeline Stage Sync (fire-and-forget) ──
        transitioned_state = None
        if result.new_state:
            # Capture whichever state was actually saved
            try:
                transitioned_state = new_state  # from SM transition
            except NameError:
                transitioned_state = result.new_state  # from direct save fallback

        if transitioned_state and contact_id:
            try:
                from app.tasks.pipeline_sync_task import sync_pipeline_stage
                sync_pipeline_stage.delay(contact_id, transitioned_state, trace_id)
            except Exception:
                logger.exception("pipeline_sync_enqueue_failed", trace_id=trace_id)

        # ── Stage 8.5: Handoff Notification (if applicable) ──
        if result.should_handoff and contact_id:
            try:
                from app.tasks.handoff_notification_task import send_handoff_notification
                send_handoff_notification.delay(
                    contact_id,
                    phone,
                    lead_data.get("name", ""),
                    result.metadata.get("handoff_reason", "unknown"),
                    result.metadata.get("priority", "normal"),
                    result.metadata.get("context_summary", ""),
                    lead_data.get("building_source", "unknown"),
                    trace_id,
                )
            except Exception:
                logger.exception("handoff_notification_enqueue_failed", trace_id=trace_id)

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
                    prompt_version=prompt_version,
                )
            except Exception:
                logger.exception("delivery_failed", trace_id=trace_id)
                delivery_result = {"status": "delivery_error"}

        # ── Stage 9: Write GHL Note (CRM sync) ──
        if result.response_text and contact_id:
            try:
                from app.services import ghl_service
                note_body = (
                    f"[AI - {current_state}→{result.new_state or current_state}] "
                    f"Lead: {message[:100] if message else '(auto)'}\n"
                    f"AI: {result.response_text[:200]}"
                )
                await ghl_service.add_note(contact_id, note_body)
            except Exception:
                logger.warning("ghl_note_write_failed", trace_id=trace_id)

        # -- Stage 8.7: Lead Score Computation + GHL Sync --
        if settings.LEAD_SCORE_ENABLED and contact_id and inbound.direction == "inbound":
            try:
                from app.services.lead_scoring import LeadScoringService
                from app.services.monitoring.alert_manager import AlertManager, AlertLevel
                from app.services.ghl_service import update_contact, add_tag, remove_tag

                lead_scoring_service = LeadScoringService(lead_repo)
                score_result = await lead_scoring_service.compute_score(
                    contact_id, enriched_context=enriched_context
                )

                if score_result:
                    score = score_result["score"]
                    tier = score_result["tier"]
                    previous_score = score_result.get("previous_score")
                    crossed_hot = score_result.get("crossed_hot", False)

                    # GHL custom field sync (write numeric score)
                    try:
                        await update_contact(contact_id, {
                            "customFields": [
                                {"key": "lead_score", "value": str(score)}
                            ]
                        })
                    except Exception:
                        logger.warning("lead_score.ghl_custom_field_failed", trace_id=trace_id)

                    # GHL tier tag sync (mutually exclusive)
                    tier_tags = {"hot": "hot-lead", "warm": "warm-lead", "cold": "cold-lead"}
                    new_tag = tier_tags[tier]
                    # Remove all other tier tags, then add the correct one
                    for t_tier, t_tag in tier_tags.items():
                        if t_tier != tier:
                            try:
                                await remove_tag(contact_id, t_tag)
                            except Exception:
                                pass  # Tag might not exist -- that's fine
                    try:
                        await add_tag(contact_id, new_tag)
                    except Exception:
                        logger.warning("lead_score.ghl_tag_failed", trace_id=trace_id, tag=new_tag)

                    # Slack hot-lead alert (when score crosses 80+ threshold)
                    if crossed_hot:
                        try:
                            alert_mgr = AlertManager(_redis_client)
                            building_info = lead_data.get("building_source", "unknown")
                            factors = score_result.get("factors", {})
                            # Build actionable context per user requirement
                            trigger_info = []
                            for factor_name, factor_data in factors.items():
                                if isinstance(factor_data, dict) and factor_data.get("raw_score"):
                                    trigger_info.append(f"{factor_name}: {factor_data['raw_score']}/100")

                            alert_message = (
                                f"*{lead_data.get('name', 'Unknown')}* just hit score *{score}* (was {previous_score or 'unscored'})\n"
                                f"Building: {building_info}\n"
                                f"Last message: _{message[:100] if message else '(auto)'}_ \n"
                                f"Factor breakdown: {', '.join(trigger_info)}"
                            )
                            await alert_mgr.send(
                                alert_type="hot_lead_alert",
                                message=alert_message,
                                level=AlertLevel.WARNING,
                                contact_id=contact_id,
                                entity_id=f"hot_lead:{contact_id}",
                                extra={
                                    "Score": str(score),
                                    "Tier": tier,
                                    "Building": building_info,
                                },
                            )
                        except Exception:
                            logger.warning("lead_score.slack_alert_failed", trace_id=trace_id)

                    logger.info(
                        "lead_score.computed",
                        trace_id=trace_id,
                        contact_id=contact_id,
                        score=score,
                        tier=tier,
                        previous_score=previous_score,
                        crossed_hot=crossed_hot,
                    )
            except Exception:
                logger.exception("lead_score.stage_failed", trace_id=trace_id)
                # Fail-open: scoring failure must NOT block the pipeline

        # -- Stage 8.8: Conversation Summary Generation --
        if settings.LEAD_SCORE_ENABLED and contact_id and inbound.direction == "inbound":
            try:
                from app.services.conversation_summary import ConversationSummaryService

                summary_service = ConversationSummaryService(lead_repo, claude_service)
                summary_result = await summary_service.maybe_generate_summary(contact_id)
                if summary_result:
                    logger.info(
                        "conversation_summary.generated",
                        trace_id=trace_id,
                        contact_id=contact_id,
                        summary_length=len(summary_result.get("summary", "")),
                    )
            except Exception:
                logger.exception("conversation_summary.stage_failed", trace_id=trace_id)
                # Fail-open: summary failure must NOT block the pipeline

        duration_ms = (time.time() - start_time) * 1000
        logger.info(
            "pipeline_complete",
            trace_id=trace_id,
            state=current_state,
            new_state=result.new_state,
            duration_ms=round(duration_ms, 2),
            delivery_status=delivery_result.get("status"),
        )

        # Mark task as completed for idempotency (2h TTL)
        if _redis_client and task_id:
            try:
                _redis_client.set(f"task:completed:{task_id}", "1", ex=7200)
            except Exception:
                logger.warning("idempotency_mark_failed", task_id=task_id)

        return {
            "status": "processed",
            "trace_id": trace_id,
            "state": current_state,
            "new_state": result.new_state,
            "delivery": delivery_result,
            "duration_ms": round(duration_ms, 2),
        }

    async def _run_with_cleanup():
        """Wrapper that ensures Neo4j driver is closed after each task.

        Without this, the cached driver's connections stay bound to the
        event loop created by this asyncio.run(). The NEXT task's
        asyncio.run() creates a new loop, but the old driver connections
        are still bound to the dead loop → "Future attached to a different
        loop" error on every subsequent message in the same worker process.
        """
        try:
            return await _run()
        finally:
            from app.repositories.base import close_driver
            await close_driver()

    # Run the async pipeline in asyncio
    # Safe in prefork Celery workers (no existing event loop)
    return asyncio.run(_run_with_cleanup())
