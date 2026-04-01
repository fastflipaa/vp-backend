"""Response Delivery Service — sends AI responses via GHL + logs to Neo4j.

Replicates n8n Main Router Stage 9 (Response Delivery):
1. Clean response through PII filter
2. Format for channel (WhatsApp/SMS: plain text max 1600 chars; Email: HTML)
3. Check GHL circuit breaker — if open, write to DLQ
4. Send via GHL with circuit breaker protection
5. Log both inbound + assistant interactions to Neo4j
6. Create pipeline trace node

Also provides fallback delivery for when the Claude circuit is open.

Usage:
    service = ResponseDeliveryService(redis_client)
    result = await service.deliver(contact_id, phone, response, channel, trace_id)
"""

from __future__ import annotations

import json
import re
import time

import redis
import structlog

from app.prompts.builder import PromptBuilder
from app.repositories.conversation_repository import ConversationRepository
from app.services import ghl_service
from app.services.circuit_breaker import CircuitOpenError, RedisCircuitBreaker
from app.services.pii_filter import PIIFilter

logger = structlog.get_logger()

# Max message length for SMS/WhatsApp channels
MAX_MESSAGE_LENGTH = 1600


class ResponseDeliveryService:
    """Delivers AI responses via GHL with circuit breaker + DLQ fallback.

    Uses the GHL circuit breaker to protect against cascading failures.
    Failed deliveries are written to a Redis dead-letter queue (DLQ)
    for later retry.
    """

    def __init__(self, redis_client: redis.Redis) -> None:
        self._redis = redis_client
        self.circuit_breaker = RedisCircuitBreaker("ghl", redis_client)
        self.prompt_builder = PromptBuilder("v1")

    async def deliver(
        self,
        contact_id: str,
        phone: str,
        response_text: str,
        channel: str,
        trace_id: str,
        inbound_message: str = "",
        conversation_repo: ConversationRepository | None = None,
        lead_phone: str = "",
        lead_email: str = "",
        prompt_version: str | None = None,
    ) -> dict:
        """Full delivery pipeline: clean -> format -> send -> log.

        Args:
            contact_id: GHL contact ID.
            phone: Lead phone (E.164).
            response_text: Raw AI response text.
            channel: Delivery channel (SMS, WhatsApp, Email, etc.).
            trace_id: Pipeline trace ID.
            inbound_message: Original inbound message (for logging).
            conversation_repo: Neo4j conversation repository (optional).
            lead_phone: Lead's phone for PII filter (defaults to phone).
            lead_email: Lead's email for PII filter.
            prompt_version: Prompt YAML version for A/B tracking on Interaction nodes.

        Returns:
            Dict with delivery status and metadata.
        """
        start = time.monotonic()
        lead_phone = lead_phone or phone

        # 0. Strip trailing JSON metadata from Claude response
        response_text = ResponseDeliveryService._strip_json_metadata(response_text)

        # 1. Clean response through PII filter
        cleaned = PIIFilter.clean(
            response_text, lead_phone=lead_phone, lead_email=lead_email
        )

        # 2. Format for channel
        formatted = self._format_for_channel(cleaned, channel)

        # 3. Check circuit breaker
        if self.circuit_breaker.is_open():
            self._write_to_dlq(
                {"contact_id": contact_id, "message": formatted, "channel": channel},
                trace_id,
                "circuit_open",
            )
            logger.warning(
                "delivery.circuit_open",
                contact_id=contact_id,
                trace_id=trace_id,
            )
            return {"status": "circuit_open", "dlq": True}

        # 4. Send via GHL
        try:
            await ghl_service.send_message(contact_id, formatted, channel)
            self.circuit_breaker.record_success()
        except Exception as e:
            self.circuit_breaker.record_failure()
            self._write_to_dlq(
                {"contact_id": contact_id, "message": formatted, "channel": channel},
                trace_id,
                str(e),
            )
            logger.error(
                "delivery.send_failed",
                contact_id=contact_id,
                trace_id=trace_id,
                error=str(e),
            )
            return {"status": "failed", "dlq": True, "error": str(e)}

        duration_ms = int((time.monotonic() - start) * 1000)

        # 5-7. Log to Neo4j (if repo provided)
        if conversation_repo:
            try:
                # Log inbound interaction (skip if empty)
                if inbound_message:
                    await conversation_repo.log_interaction(
                        phone, "user", inbound_message, channel, trace_id
                    )

                # Log assistant interaction (with prompt version for A/B tracking)
                await conversation_repo.log_interaction(
                    phone, "assistant", cleaned, channel, trace_id,
                    prompt_version=prompt_version,
                )

                # Log trace
                await conversation_repo.log_trace(
                    trace_id, phone, "delivered", duration_ms
                )
            except Exception as e:
                # Neo4j logging failure should NOT fail delivery
                logger.warning(
                    "delivery.neo4j_log_failed",
                    trace_id=trace_id,
                    error=str(e),
                )

        logger.info(
            "delivery.success",
            contact_id=contact_id,
            trace_id=trace_id,
            channel=channel,
            message_length=len(cleaned),
            duration_ms=duration_ms,
        )

        return {
            "status": "delivered",
            "message_length": len(cleaned),
            "channel": channel,
        }

    async def deliver_fallback(
        self,
        contact_id: str,
        phone: str,
        language: str,
        channel: str,
        trace_id: str,
        conversation_repo: ConversationRepository | None = None,
        prompt_version: str | None = None,
    ) -> dict:
        """Send fallback message when Claude circuit is open.

        Args:
            contact_id: GHL contact ID.
            phone: Lead phone (E.164).
            language: Language code ("es" or "en").
            channel: Delivery channel.
            trace_id: Pipeline trace ID.
            conversation_repo: Neo4j conversation repository (optional).
            prompt_version: Prompt YAML version for A/B tracking on Interaction nodes.

        Returns:
            Dict with fallback delivery status.
        """
        # Load fallback text
        fallback_es, fallback_en = self.prompt_builder.render_fallback({})
        fallback_text = fallback_es if language.lower().startswith("es") else fallback_en

        # Send via GHL (best-effort even if GHL circuit is questionable)
        try:
            await ghl_service.send_message(contact_id, fallback_text, channel)
        except Exception as e:
            logger.error(
                "delivery.fallback_failed",
                contact_id=contact_id,
                trace_id=trace_id,
                error=str(e),
            )
            return {"status": "fallback_failed", "error": str(e)}

        # Log as assistant interaction with fallback metadata
        if conversation_repo:
            try:
                await conversation_repo.log_interaction(
                    phone,
                    "assistant",
                    f"[FALLBACK] {fallback_text}",
                    channel,
                    trace_id,
                    prompt_version=prompt_version,
                )
            except Exception as e:
                logger.warning(
                    "delivery.fallback_log_failed",
                    trace_id=trace_id,
                    error=str(e),
                )

        logger.info(
            "delivery.fallback_sent",
            contact_id=contact_id,
            trace_id=trace_id,
            language=language,
        )

        return {"status": "fallback_delivered", "language": language}

    def _write_to_dlq(
        self, payload: dict, trace_id: str, error: str
    ) -> None:
        """Write a failed delivery to the Redis dead-letter queue.

        Caps at 1000 entries (LTRIM) to prevent unbounded growth.
        """
        dlq_entry = json.dumps(
            {
                "payload": payload,
                "trace_id": trace_id,
                "error": str(error),
                "failed_at": time.time(),
            }
        )
        try:
            self._redis.lpush("dlq:failed_messages", dlq_entry)
            self._redis.ltrim("dlq:failed_messages", 0, 999)
            logger.warning(
                "delivery.dlq_written",
                trace_id=trace_id,
                error=error,
            )
        except Exception as e:
            # Even DLQ write failure should be logged but not raised
            logger.error(
                "delivery.dlq_write_failed",
                trace_id=trace_id,
                error=str(e),
            )

    @staticmethod
    def _strip_json_metadata(text: str) -> str:
        """Strip ALL JSON blocks from Claude responses.

        LEVITAS prompts instruct Claude to append hidden JSON like:
        {"cadence": "explorer", "language": "es", ...}
        This must be removed before sending to the lead.
        Catches any JSON object at the end OR anywhere in the text.
        """
        # First: strip any JSON block at the end of the message
        stripped = re.sub(r'\s*\{[^{}]*\}\s*$', "", text).strip()

        # Second: strip any remaining JSON blocks that look like metadata
        # (contains known metadata keys anywhere in the text)
        stripped = re.sub(
            r'\{[^{}]*"(?:cadence|language|sentiment|interest_type|budget_min|'
            r'matched_building|timeline|building_mentioned|confidence|response|'
            r'next_action|tone_applied|escalate|escalation_reason|sentiment_confidence'
            r')"[^{}]*\}',
            "",
            stripped,
        ).strip()

        return stripped if stripped else text

    @staticmethod
    def _format_for_channel(message: str, channel: str) -> str:
        """Format message for the delivery channel.

        - SMS/WhatsApp: plain text, truncated to MAX_MESSAGE_LENGTH
        - Email: wrapped in basic HTML template
        """
        channel_lower = channel.lower()

        if channel_lower in ("email",):
            # Wrap in basic HTML for email delivery
            return (
                f"<html><body>"
                f"<p>{message}</p>"
                f"</body></html>"
            )

        # SMS / WhatsApp / default: plain text, max length
        if len(message) > MAX_MESSAGE_LENGTH:
            # Truncate at word boundary
            truncated = message[:MAX_MESSAGE_LENGTH]
            last_space = truncated.rfind(" ")
            if last_space > MAX_MESSAGE_LENGTH * 0.8:
                truncated = truncated[:last_space]
            return truncated + "..."

        return message
