"""Prompt injection filter (SAFE-15).

Pre-compiled regex patterns ported from the n8n Security Gate (production-proven),
enhanced with OWASP 2025/2026 additions for invisible Unicode, homoglyph
normalization, chat-markup injection, and encoding-evasion attacks.

Fail-open: fuzzy matches and invisible char density produce flagged PASS, not BLOCK.
On BLOCK: logs a :SecurityEvent node to Neo4j (fail-open if Neo4j unreachable).
"""

from __future__ import annotations

import re
import unicodedata

import structlog
from neo4j import GraphDatabase

from app.config import settings
from app.gates.base import GateDecision, GateResult

logger = structlog.get_logger()

# ---------------------------------------------------------------------------
# Pre-compiled patterns at module load
# ---------------------------------------------------------------------------

INJECTION_PATTERNS: list[re.Pattern[str]] = [
    # === Ported from n8n Security Gate (working in production) ===
    re.compile(
        r"ignore\s+(all\s+)?(previous|prior|above|system)\s+(instructions?|prompts?|rules?|context)",
        re.I,
    ),
    re.compile(
        r"forget\s+(everything|all|your|the)\s*(instructions?|role|context|rules?)?",
        re.I,
    ),
    re.compile(r"you\s+are\s+now\s+(a|an)\s+", re.I),
    re.compile(r"act\s+as\s+(if\s+you\s+(are|were)|a|an)\s+", re.I),
    re.compile(r"new\s+(system\s+)?(prompt|instructions?|role|persona)", re.I),
    re.compile(
        r"reveal\s+(your\s+)?(system\s+)?(prompt|instructions?|api\s+key|password|credentials?)",
        re.I,
    ),
    # Negative lookahead for Spanish real estate terms to avoid false positives
    re.compile(
        r"show\s+(me\s+)?(your\s+)?(system\s+prompt|instructions?|code|database)"
        r"(?!\s*(departamento|unidad|precio|plano|edificio|propiedad|amenidad))",
        re.I,
    ),
    re.compile(
        r"what\s+are\s+your\s+(actual|real|true|hidden)\s+(instructions?|rules?|prompts?)",
        re.I,
    ),
    re.compile(r"\bdeveloper\s+mode\b", re.I),
    re.compile(r"\bjailbreak\b", re.I),
    re.compile(r"\bDAN\b"),  # Case-sensitive intentionally
    re.compile(r"override\s+(safety|security|guidelines?|restrictions?)", re.I),
    # === OWASP 2025/2026 additions ===
    re.compile(r"system\s*:\s*", re.I),  # Delimiter injection
    re.compile(
        r"<\|?(system|user|assistant|im_start|im_end)\|?>", re.I
    ),  # Chat markup injection
    re.compile(
        r"```\s*(system|prompt|instructions?)", re.I
    ),  # Markdown code block injection
    re.compile(
        r"(base64|rot13|hex)\s*(decode|encode|convert)", re.I
    ),  # Encoding evasion
    re.compile(
        r"translate\s+(this|the\s+following)\s+(from|to)\s+.{0,20}(instructions?|prompt)",
        re.I,
    ),  # Translation attack
    re.compile(
        r"repeat\s+(after\s+me|back|the\s+following|everything)", re.I
    ),  # Parrot attack
    re.compile(
        r"(imagine|pretend|suppose|let'?s\s+say)\s+(you\s+)?(are|were|have)\s+no\s+(rules?|restrictions?|guidelines?)",
        re.I,
    ),  # Hypothetical bypass
]

# Fuzzy obfuscation patterns (ported from n8n Security Gate)
FUZZY_PATTERNS: list[str] = [
    "ignroe",
    "bpyass",
    "ovverride",
    "sytesm prmopt",
    "prvious instrucions",
]

# Invisible Unicode characters that could hide injected text
INVISIBLE_CHARS = re.compile(
    r"[\u200b\u200c\u200d\u200e\u200f"  # Zero-width spaces, joiners, direction marks
    r"\u2060\u2061\u2062\u2063\u2064"  # Word joiner, invisible math operators
    r"\ufeff"  # BOM / zero-width no-break space
    r"\u00ad"  # Soft hyphen
    r"\u034f"  # Combining grapheme joiner
    r"\u061c"  # Arabic letter mark
    r"\u115f\u1160"  # Hangul filler characters
    r"\u17b4\u17b5"  # Khmer vowel inherent chars
    r"\u180e"  # Mongolian vowel separator
    r"\uffa0"  # Halfwidth Hangul filler
    r"]"
)


# ---------------------------------------------------------------------------
# Unicode normalization
# ---------------------------------------------------------------------------


def normalize_text(text: str) -> str:
    """Normalize Unicode via NFKD decomposition and strip invisible characters.

    Catches homoglyph attacks (e.g. Cyrillic lookalikes) and hidden text
    injected via zero-width characters.
    """
    normalized = unicodedata.normalize("NFKD", text)
    normalized = INVISIBLE_CHARS.sub("", normalized)
    return normalized


# ---------------------------------------------------------------------------
# Neo4j security event logging (fail-open)
# ---------------------------------------------------------------------------


def _log_security_event(
    trace_id: str,
    contact_id: str,
    event_type: str,
    snippet: str,
    pattern_matched: str,
) -> None:
    """Create a :SecurityEvent node in Neo4j. Fail-open on any error."""
    try:
        driver = GraphDatabase.driver(
            settings.NEO4J_URI,
            auth=("neo4j", settings.NEO4J_PASSWORD),
        )
        with driver.session() as session:
            session.run(
                "CREATE (se:SecurityEvent {"
                "  trace_id: $trace_id,"
                "  contact_id: $contact_id,"
                "  event_type: $event_type,"
                "  snippet: $snippet,"
                "  pattern: $pattern,"
                "  detected_at: datetime()"
                "})",
                trace_id=trace_id,
                contact_id=contact_id,
                event_type=event_type,
                snippet=snippet,
                pattern=pattern_matched,
            )
        driver.close()
        logger.info(
            "security_event_logged",
            trace_id=trace_id,
            event_type=event_type,
        )
    except Exception as exc:
        logger.warning(
            "security_event_logging_failed",
            trace_id=trace_id,
            error=str(exc),
        )


# ---------------------------------------------------------------------------
# Gate entry point
# ---------------------------------------------------------------------------


def check_injection(payload: dict, trace_id: str) -> GateResult:
    """Prompt injection filter with OWASP 2025/2026 patterns.

    Pure regex -- does NOT need Redis. Accepts (payload, trace_id) only.
    On BLOCK: logs :SecurityEvent to Neo4j (fail-open).
    """
    message = payload.get("message", "")
    contact_id = payload.get("contactId", "")

    if not message:
        return GateResult(
            gate_name="injection",
            decision=GateDecision.PASS,
            reason="empty_message",
            duration_ms=0.0,
        )

    # --- Check raw message against all injection patterns ---
    for pattern in INJECTION_PATTERNS:
        if pattern.search(message):
            _log_security_event(
                trace_id=trace_id,
                contact_id=contact_id,
                event_type="PROMPT_INJECTION",
                snippet=message[:100],
                pattern_matched=pattern.pattern[:80],
            )
            return GateResult(
                gate_name="injection",
                decision=GateDecision.BLOCK,
                reason=f"prompt_injection_detected:pattern={pattern.pattern[:50]}",
                duration_ms=0.0,
                metadata={
                    "type": "PROMPT_INJECTION",
                    "snippet": message[:100],
                    "severity": "HIGH",
                },
            )

    # --- Check normalized message (homoglyph + invisible char attacks) ---
    normalized = normalize_text(message)
    if normalized != message:
        for pattern in INJECTION_PATTERNS:
            if pattern.search(normalized):
                _log_security_event(
                    trace_id=trace_id,
                    contact_id=contact_id,
                    event_type="PROMPT_INJECTION_OBFUSCATED",
                    snippet=message[:100],
                    pattern_matched=pattern.pattern[:80],
                )
                return GateResult(
                    gate_name="injection",
                    decision=GateDecision.BLOCK,
                    reason="prompt_injection_detected:obfuscated_unicode",
                    duration_ms=0.0,
                    metadata={
                        "type": "PROMPT_INJECTION_OBFUSCATED",
                        "snippet": message[:100],
                        "severity": "HIGH",
                    },
                )

    # --- Fuzzy obfuscation (typo-based evasion) -- flagged PASS ---
    lower_alpha = message.lower().replace(" ", "")
    for fuzzy in FUZZY_PATTERNS:
        if fuzzy.replace(" ", "") in lower_alpha:
            return GateResult(
                gate_name="injection",
                decision=GateDecision.PASS,
                reason=f"injection_fuzzy_flagged:{fuzzy}",
                duration_ms=0.0,
                metadata={
                    "type": "PROMPT_INJECTION_FUZZY",
                    "severity": "MEDIUM",
                    "flagged": True,
                },
            )

    # --- Invisible char density check -- flagged PASS ---
    invisible_count = len(INVISIBLE_CHARS.findall(message))
    if invisible_count > 3:
        return GateResult(
            gate_name="injection",
            decision=GateDecision.PASS,
            reason=f"suspicious_invisible_chars:count={invisible_count}",
            duration_ms=0.0,
            metadata={
                "type": "SUSPICIOUS_UNICODE",
                "invisible_count": invisible_count,
                "flagged": True,
            },
        )

    return GateResult(
        gate_name="injection",
        decision=GateDecision.PASS,
        reason="clean",
        duration_ms=0.0,
    )
