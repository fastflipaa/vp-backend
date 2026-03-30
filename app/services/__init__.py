"""Service layer — Claude API integration + circuit breaker.

Provides ClaudeService for AI response generation with Redis-backed
circuit breaker protection. CircuitOpenError is raised when the breaker
is open, signaling the caller to use fallback responses.
"""

from app.services.circuit_breaker import CircuitOpenError, RedisCircuitBreaker
from app.services.claude_service import ClaudeService, get_claude_client

__all__ = [
    "CircuitOpenError",
    "ClaudeService",
    "RedisCircuitBreaker",
    "get_claude_client",
]
