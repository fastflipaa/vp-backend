"""Service layer — Claude API, GHL integration, circuit breaker, PII filtering.

Provides ClaudeService for AI response generation, GHL services for CRM
communication, enrichment, human agent detection, response delivery,
PII filtering, phone normalization, and Redis-backed circuit breaker
protection. CircuitOpenError is raised when a breaker is open, signaling
the caller to use fallback responses.
"""

from app.services.circuit_breaker import CircuitOpenError, RedisCircuitBreaker
from app.services.claude_service import ClaudeService, get_claude_client
from app.services.ghl_enrichment import GHLEnrichmentService
from app.services.human_agent_detector import HumanAgentDetector
from app.services.phone_normalizer import normalize_phone
from app.services.pii_filter import PIIFilter
from app.services.response_delivery import ResponseDeliveryService

__all__ = [
    "CircuitOpenError",
    "ClaudeService",
    "GHLEnrichmentService",
    "HumanAgentDetector",
    "PIIFilter",
    "RedisCircuitBreaker",
    "ResponseDeliveryService",
    "get_claude_client",
    "normalize_phone",
]
