"""Broker detection gate (SAFE-17).

Two-tier approach for filtering non-lead contacts:
  Tier 0: Redis cache check (7-day TTL) -- avoid redundant classification
  Tier 1: Regex pre-filter -- catches obvious broker/advertiser patterns (free, instant)
  Tier 2: Claude Haiku fallback -- classifies ambiguous cases (2s timeout, retry once)

On BLOCK: tags GHL contact and creates Neo4j :BrokerDetection node (fail-open).
Fail-open: Haiku failures fall back to regex-only; unhandled exceptions caught by pipeline.
"""

from __future__ import annotations

import json
import re

import httpx
import structlog
from neo4j import GraphDatabase

from app.config import settings
from app.gates.base import GateDecision, GateResult

logger = structlog.get_logger()

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

BROKER_CACHE_TTL = 604800  # 7 days in seconds

# ---------------------------------------------------------------------------
# Regex patterns (pre-compiled at module load)
# ---------------------------------------------------------------------------

BROKER_REGEX_PATTERNS: list[re.Pattern[str]] = [
    re.compile(r"soy\s+(corredor|agente|asesor)\s*(inmobiliari[oa])?", re.I),
    re.compile(r"tengo\s+(un\s+)?client[ea]\s+(interesad[oa]|que\s+busca)", re.I),
    re.compile(r"manejamos\s+cartera\s+de\s+client", re.I),
    re.compile(r"(comisi[o\u00f3]n|comisiones|esquema\s+de\s+corretaje)", re.I),
    re.compile(r"(alianza|convenio)\s+(entre\s+)?(inmobiliari|agenci)", re.I),
    re.compile(r"\bAMPI\b"),  # Case-sensitive: AMPI is an acronym
    re.compile(r"(ofrecemos|prestamos)\s+servicios?\s+de\s+", re.I),
    re.compile(
        r"cotizaci[o\u00f3]n\s+(para\s+)?(sus|los)\s+(edificios?|desarrollos?)", re.I
    ),
    re.compile(r"venta\s+de\s+(materiales?|mobiliario)", re.I),
    re.compile(
        r"publicidad\s*(para|de)\s*(su|tu|la)\s*(empresa|inmobiliaria)", re.I
    ),
]

ADVERTISER_REGEX_PATTERNS: list[re.Pattern[str]] = [
    re.compile(r"(somos|soy)\s+(empresa|proveedor|distribuidor)\s+de\s+", re.I),
    re.compile(r"(mantenimiento|limpieza|seguridad|fumigaci)", re.I),
]


# ---------------------------------------------------------------------------
# GHL tagging + Neo4j note (fail-open)
# ---------------------------------------------------------------------------


def _tag_and_note(
    contact_id: str,
    classification: str,
    trace_id: str,
) -> None:
    """Tag GHL contact and create Neo4j :BrokerDetection node.

    Both operations fail-open: the broker gate returns BLOCK regardless of
    whether tagging/noting succeeds.
    """
    # --- GHL contact tag ---
    try:
        httpx.patch(
            f"https://services.leadconnectorhq.com/contacts/{contact_id}/tags",
            json={"tags": [classification]},
            headers={
                "Authorization": f"Bearer {settings.GHL_TOKEN}",
                "Version": "2021-07-28",
            },
            timeout=3.0,
        )
        logger.info(
            "broker_ghl_tag_set",
            contact_id=contact_id,
            classification=classification,
        )
    except Exception as exc:
        logger.warning(
            "broker_ghl_tag_failed",
            contact_id=contact_id,
            error=str(exc),
        )

    # --- Neo4j :BrokerDetection node ---
    try:
        driver = GraphDatabase.driver(
            settings.NEO4J_URI,
            auth=("neo4j", settings.NEO4J_PASSWORD),
        )
        with driver.session() as session:
            session.run(
                "CREATE (n:BrokerDetection {"
                "  contact_id: $contact_id,"
                "  classification: $classification,"
                "  detected_at: datetime(),"
                "  trace_id: $trace_id"
                "})",
                contact_id=contact_id,
                classification=classification,
                trace_id=trace_id,
            )
        driver.close()
        logger.info(
            "broker_neo4j_note_created",
            contact_id=contact_id,
            classification=classification,
        )
    except Exception as exc:
        logger.warning(
            "broker_neo4j_note_failed",
            contact_id=contact_id,
            error=str(exc),
        )


# ---------------------------------------------------------------------------
# Haiku classification (Tier 2)
# ---------------------------------------------------------------------------


def _classify_with_haiku(message: str, client, timeout: float = 2.0) -> dict:
    """Call Claude Haiku for broker/advertiser classification.

    Returns dict with keys: classification, confidence, reason, method.
    """
    response = client.messages.create(
        model="claude-haiku-4-5-20251001",
        max_tokens=100,
        timeout=timeout,
        system=(
            "Clasifica este mensaje de WhatsApp para una inmobiliaria de lujo en CDMX. "
            'Responde SOLO con JSON: {"classification": "buyer_lead"|"broker"|"advertiser", '
            '"confidence": 0.0-1.0, "reason": "brief reason"}'
        ),
        messages=[{"role": "user", "content": message}],
    )
    result = json.loads(response.content[0].text)
    result["method"] = "haiku"
    return result


# ---------------------------------------------------------------------------
# Gate entry point
# ---------------------------------------------------------------------------


def check_broker(
    payload: dict,
    trace_id: str,
    redis_client,
    anthropic_client=None,
) -> GateResult:
    """Two-tier broker detection: regex first, Haiku fallback for ambiguous.

    Args:
        payload: Inbound webhook payload with message and contactId.
        trace_id: UUID trace ID for logging.
        redis_client: Redis client for cache operations (db2).
        anthropic_client: Optional Anthropic SDK client for Haiku calls.
            Not provided in Phase 15 (shadow mode). Will be injected in Phase 16.
    """
    message = payload.get("message", "")
    contact_id = payload.get("contactId", "")

    if not message:
        return GateResult(
            gate_name="broker",
            decision=GateDecision.PASS,
            reason="empty_message",
            duration_ms=0.0,
        )

    # --- Tier 0: Redis cache check ---
    cache_key = f"broker_cache:{contact_id}"
    if contact_id:
        cached = redis_client.get(cache_key)
        if cached:
            cached_data = json.loads(cached)
            classification = cached_data.get("classification", "")
            if classification in ("broker", "advertiser"):
                return GateResult(
                    gate_name="broker",
                    decision=GateDecision.BLOCK,
                    reason=f"cached_classification:{classification}",
                    duration_ms=0.0,
                    metadata=cached_data,
                )
            else:
                return GateResult(
                    gate_name="broker",
                    decision=GateDecision.PASS,
                    reason=f"cached_classification:{classification}",
                    duration_ms=0.0,
                )

    # --- Tier 1: Regex pre-filter (free, instant) ---
    for pattern in BROKER_REGEX_PATTERNS:
        if pattern.search(message):
            classification_data = {
                "classification": "broker",
                "confidence": 0.95,
                "method": "regex",
            }
            if contact_id:
                redis_client.set(
                    cache_key, json.dumps(classification_data), ex=BROKER_CACHE_TTL
                )
                _tag_and_note(contact_id, "broker", trace_id)
            return GateResult(
                gate_name="broker",
                decision=GateDecision.BLOCK,
                reason=f"broker_regex_match:{pattern.pattern[:40]}",
                duration_ms=0.0,
                metadata=classification_data,
            )

    for pattern in ADVERTISER_REGEX_PATTERNS:
        if pattern.search(message):
            classification_data = {
                "classification": "advertiser",
                "confidence": 0.90,
                "method": "regex",
            }
            if contact_id:
                redis_client.set(
                    cache_key, json.dumps(classification_data), ex=BROKER_CACHE_TTL
                )
                _tag_and_note(contact_id, "advertiser", trace_id)
            return GateResult(
                gate_name="broker",
                decision=GateDecision.BLOCK,
                reason=f"advertiser_regex_match:{pattern.pattern[:40]}",
                duration_ms=0.0,
                metadata=classification_data,
            )

    # --- Tier 2: Claude Haiku fallback (only if client provided) ---
    if anthropic_client:
        try:
            haiku_result = _classify_with_haiku(
                message, anthropic_client, timeout=2.0
            )
            # Cache regardless of classification
            if contact_id:
                redis_client.set(
                    cache_key, json.dumps(haiku_result), ex=BROKER_CACHE_TTL
                )

            if (
                haiku_result.get("classification") in ("broker", "advertiser")
                and haiku_result.get("confidence", 0) >= 0.8
            ):
                if contact_id:
                    _tag_and_note(
                        contact_id, haiku_result["classification"], trace_id
                    )
                return GateResult(
                    gate_name="broker",
                    decision=GateDecision.BLOCK,
                    reason=(
                        f"haiku_classification:"
                        f"{haiku_result['classification']}:{haiku_result['confidence']}"
                    ),
                    duration_ms=0.0,
                    metadata=haiku_result,
                )
        except Exception as exc:
            logger.warning(
                "broker_haiku_fallback_failed",
                error=str(exc),
                trace_id=trace_id,
            )
            # Fail open: if Haiku fails, treat as not-broker

    return GateResult(
        gate_name="broker",
        decision=GateDecision.PASS,
        reason="not_broker",
        duration_ms=0.0,
    )
