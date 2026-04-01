"""ClaudeService — async Claude API wrapper with circuit breaker protection.

Wraps AsyncAnthropic with:
- Module-level singleton client (max_retries=3 built-in)
- Redis-backed circuit breaker (opens after 3 failures in 60s)
- Separate handling for transient vs config errors
- Classification using Haiku model for cost efficiency
- Structured logging with structlog

Usage:
    service = ClaudeService(redis_client)
    response = await service.generate(system_prompt, user_message)
    classification = await service.classify(message, phone)
"""

from __future__ import annotations

import json
import time

import redis
import structlog
from anthropic import (
    APIConnectionError,
    APIStatusError,
    AsyncAnthropic,
    AuthenticationError,
    BadRequestError,
    RateLimitError,
)

from app.config import settings
from app.prompts.builder import PromptBuilder
from app.services.circuit_breaker import CircuitOpenError, RedisCircuitBreaker

logger = structlog.get_logger()

# --- Module-level singleton ---

_claude_client: AsyncAnthropic | None = None


def get_claude_client() -> AsyncAnthropic:
    """Return the singleton AsyncAnthropic client.

    Created once with max_retries=3 for built-in exponential backoff
    on transient errors.
    """
    global _claude_client
    if _claude_client is None:
        _claude_client = AsyncAnthropic(
            api_key=settings.ANTHROPIC_API_KEY,
            timeout=30.0,
            max_retries=3,  # Built-in retry with exponential backoff
        )
    return _claude_client


class ClaudeService:
    """Async Claude API service with circuit breaker protection.

    The circuit breaker opens after 3 transient failures within 60 seconds,
    returning CircuitOpenError so callers can use fallback responses.

    Auth and prompt errors (AuthenticationError, BadRequestError) do NOT
    trip the breaker since they indicate config/code bugs, not transient
    infrastructure issues.
    """

    def __init__(self, redis_client: redis.Redis) -> None:
        self.circuit_breaker = RedisCircuitBreaker("claude", redis_client)
        self.prompt_builder = PromptBuilder("v1")

    async def generate(
        self,
        system_prompt: str,
        user_message: str,
        model: str = "claude-sonnet-4-20250514",
        max_tokens: int = 1024,
    ) -> str:
        """Generate a response from Claude.

        Args:
            system_prompt: The system prompt string.
            user_message: The user message string.
            model: Claude model ID (default: Sonnet).
            max_tokens: Max tokens in response.

        Returns:
            The generated text response.

        Raises:
            CircuitOpenError: If the circuit breaker is open.
            RateLimitError: If rate limited (after built-in retries).
            APIConnectionError: If connection fails.
            APIStatusError: If API returns 5xx.
            AuthenticationError: If API key is invalid.
            BadRequestError: If the request is malformed.
        """
        if self.circuit_breaker.is_open():
            logger.warning(
                "claude.circuit_open",
                model=model,
                msg="Circuit breaker is OPEN — use fallback",
            )
            raise CircuitOpenError("Circuit claude is OPEN")

        start = time.monotonic()
        try:
            client = get_claude_client()
            message = await client.messages.create(
                model=model,
                max_tokens=max_tokens,
                system=system_prompt,
                messages=[{"role": "user", "content": user_message}],
            )

            duration_ms = int((time.monotonic() - start) * 1000)
            self.circuit_breaker.record_success()

            logger.info(
                "claude.generate",
                model=model,
                input_tokens=message.usage.input_tokens,
                output_tokens=message.usage.output_tokens,
                duration_ms=duration_ms,
                stop_reason=message.stop_reason,
            )

            return message.content[0].text

        except (RateLimitError, APIConnectionError) as e:
            duration_ms = int((time.monotonic() - start) * 1000)
            self.circuit_breaker.record_failure()
            logger.error(
                "claude.transient_error",
                model=model,
                error_type=type(e).__name__,
                duration_ms=duration_ms,
            )
            raise

        except APIStatusError as e:
            duration_ms = int((time.monotonic() - start) * 1000)
            # Only trip breaker for server errors (5xx)
            if e.status_code >= 500:
                self.circuit_breaker.record_failure()
            logger.error(
                "claude.api_error",
                model=model,
                status_code=e.status_code,
                error_type=type(e).__name__,
                duration_ms=duration_ms,
            )
            raise

        except (AuthenticationError, BadRequestError) as e:
            # Config/prompt bugs — do NOT trip circuit breaker
            duration_ms = int((time.monotonic() - start) * 1000)
            logger.error(
                "claude.config_error",
                model=model,
                error_type=type(e).__name__,
                duration_ms=duration_ms,
                msg="Auth/prompt error — NOT incrementing circuit breaker",
            )
            raise

    async def classify(
        self,
        message: str,
        phone: str,
        contact_context: str = "",
    ) -> dict:
        """Classify a message using Haiku model (fast + cheap).

        Returns a dict with classification, confidence, and reason.
        Falls back to ambiguous classification if JSON parsing fails.

        Args:
            message: The raw message text to classify.
            phone: The sender's phone number.
            contact_context: Optional CRM context string.

        Returns:
            Dict with keys: classification, confidence, reason.
        """
        # Load classification prompt config
        config = self.prompt_builder.get_config("classification")
        system_prompt, user_msg = self.prompt_builder.render(
            "classification",
            {
                "message": message,
                "contact_name": phone,
                "tags": "",
                "recent_notes": contact_context,
            },
        )

        try:
            response_text = await self.generate(
                system_prompt=system_prompt,
                user_message=user_msg,
                model=config.get("model", "claude-haiku-4-5-20251001"),
                max_tokens=config.get("max_tokens", 100),
            )

            # Parse JSON response
            result = json.loads(response_text.strip())
            # Validate expected keys
            return {
                "classification": result.get("classification", "ambiguous"),
                "confidence": float(result.get("confidence", 0.5)),
                "reason": result.get("reason", ""),
            }

        except (json.JSONDecodeError, KeyError, TypeError) as e:
            logger.warning(
                "claude.classify_parse_error",
                error=str(e),
                phone=phone,
            )
            return {
                "classification": "ambiguous",
                "confidence": 0.5,
                "reason": "parse_error",
            }

        except CircuitOpenError:
            logger.warning("claude.classify_circuit_open", phone=phone)
            return {
                "classification": "ambiguous",
                "confidence": 0.5,
                "reason": "circuit_open",
            }

    def is_circuit_open(self) -> bool:
        """Expose circuit state for pipeline pre-check.

        Allows the pipeline to check before attempting a Claude call,
        enabling early fallback without the overhead of preparing
        prompts and context.
        """
        return self.circuit_breaker.is_open()
