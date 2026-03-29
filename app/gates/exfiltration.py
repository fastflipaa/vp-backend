"""Data exfiltration filter (SAFE-16).

Pre-compiled regex patterns ported from the n8n Security Gate with Spanish
real estate negative lookaheads to avoid blocking legitimate property queries.

Scans INBOUND messages for attempts to extract lead lists, PII, internal data,
API keys, and admin URLs. Response-side PII scanning will be added in Phase 16
when Claude responses exist.

Pure regex -- does NOT need Redis. Accepts (payload, trace_id) only.
"""

from __future__ import annotations

import re

from app.gates.base import GateDecision, GateResult

# ---------------------------------------------------------------------------
# Pre-compiled exfiltration patterns at module load
# ---------------------------------------------------------------------------

EXFILTRATION_PATTERNS: list[re.Pattern[str]] = [
    # --- Ported from n8n Security Gate (with Spanish real estate negative lookaheads) ---
    re.compile(
        r"list\s+(all\s+)?(other\s+)?(leads?|clients?|customers?|contacts?)"
        r"(?!\s*(amenidades?|servicios?|edificios?|departamentos?))",
        re.I,
    ),
    re.compile(
        r"show\s+(me\s+)?(all\s+)?(leads?|clients?|customers?|contacts?|users?)"
        r"(?!\s*(departamento|unidad|precio|plano|edificio|propiedad))",
        re.I,
    ),
    re.compile(
        r"what\s+(other\s+)?(people|clients?|leads?|contacts?)\s+(are|have)", re.I
    ),
    re.compile(r"database\s+(schema|structure|tables?|queries?)", re.I),
    re.compile(r"api\s+key", re.I),
    re.compile(r"commission\s+(structure|rates?|percentages?)", re.I),
    re.compile(r"internal\s+(pricing|margins?|costs?)", re.I),
    # --- Additional PII extraction patterns (SAFE-16) ---
    re.compile(
        r"(give|show|tell|send)\s+(me\s+)?(all\s+)?(phone\s+numbers?|emails?|addresses?)"
        r"\s+(of|from|for)\s+(your|all|other)",
        re.I,
    ),
    re.compile(r"export\s+(all\s+)?(data|contacts?|leads?|clients?)", re.I),
    re.compile(r"(admin|backend|internal)\s+(panel|dashboard|system|url|login)", re.I),
]


def check_exfiltration(payload: dict, trace_id: str) -> GateResult:
    """Data exfiltration filter. Blocks attempts to extract internal/PII data.

    Fail-open: unhandled exceptions are caught by the SafetyPipeline wrapper.
    """
    message = payload.get("message", "")

    if not message:
        return GateResult(
            gate_name="exfiltration",
            decision=GateDecision.PASS,
            reason="empty_message",
            duration_ms=0.0,
        )

    for pattern in EXFILTRATION_PATTERNS:
        if pattern.search(message):
            return GateResult(
                gate_name="exfiltration",
                decision=GateDecision.BLOCK,
                reason=f"data_exfiltration_detected:pattern={pattern.pattern[:50]}",
                duration_ms=0.0,
                metadata={
                    "type": "DATA_EXFILTRATION",
                    "snippet": message[:100],
                    "severity": "HIGH",
                },
            )

    return GateResult(
        gate_name="exfiltration",
        decision=GateDecision.PASS,
        reason="clean",
        duration_ms=0.0,
    )
