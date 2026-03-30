"""Language detection service using lingua-language-detector.

Provides singleton-based ES/EN binary language detection optimized for
the Vive Polanco use case: Mexican luxury real estate where the primary
market speaks Spanish, with a subset of English-speaking international
buyers.

Singleton pattern (NOT worker_process_init): this is a service module
used across the application, not a Celery task. The detector is lazy-
initialized on first call and cached for the process lifetime.

lingua-language-detector uses Rust bindings with FST models -- fast
and memory-efficient for binary (2-language) detection.

Default behavior: Spanish ("es") when text is too short, ambiguous,
or undetectable. The primary market is Mexican -- defaulting to Spanish
is the safest choice for any edge case.
"""

from __future__ import annotations

import structlog
from lingua import Language, LanguageDetectorBuilder

logger = structlog.get_logger()

# Module-level singleton -- lazy-initialized on first call
_detector = None


def get_detector():
    """Return the cached lingua LanguageDetector singleton.

    Builds with ONLY Spanish and English for binary detection.
    Two-language mode is faster and more accurate than all-language mode.
    """
    global _detector
    if _detector is None:
        _detector = (
            LanguageDetectorBuilder
            .from_languages(Language.SPANISH, Language.ENGLISH)
            .build()
        )
        logger.info("language_detector_initialized", languages=["es", "en"])
    return _detector


def detect_language(text: str) -> str:
    """Detect whether text is Spanish or English.

    Returns:
        "es" for Spanish (or ambiguous/short/empty text).
        "en" for English.

    Default is "es" (Spanish) for:
        - Empty or None text
        - Text shorter than 3 characters (too short for reliable detection)
        - Ambiguous results (detector returns None)
    """
    if not text or len(text.strip()) < 3:
        logger.debug("language_detect_too_short", text_length=len(text) if text else 0)
        return "es"

    result = get_detector().detect_language_of(text)

    if result is Language.ENGLISH:
        logger.debug("language_detected", language="en", text_preview=text[:50])
        return "en"
    elif result is Language.SPANISH:
        logger.debug("language_detected", language="es", text_preview=text[:50])
        return "es"
    else:
        # None = ambiguous -- default to Spanish (primary market)
        logger.debug("language_detect_ambiguous", text_preview=text[:50])
        return "es"


def detect_language_with_confidence(text: str) -> tuple[str, float]:
    """Detect language with confidence score.

    Returns:
        Tuple of (language_code, confidence) where:
        - language_code is "es" or "en"
        - confidence is 0.0 to 1.0 (rounded to 4 decimal places)

    For empty/short text, returns ("es", 0.0).
    For ambiguous text, returns ("es", 0.0).
    """
    if not text or len(text.strip()) < 3:
        return ("es", 0.0)

    confidence_values = get_detector().compute_language_confidence_values(text)

    if not confidence_values:
        logger.debug("language_confidence_empty", text_preview=text[:50])
        return ("es", 0.0)

    # confidence_values is a list of (Language, confidence) tuples
    # sorted by confidence descending
    top_language, top_confidence = confidence_values[0]

    if top_language is Language.ENGLISH:
        code = "en"
    elif top_language is Language.SPANISH:
        code = "es"
    else:
        code = "es"

    rounded_confidence = round(top_confidence, 4)

    logger.debug(
        "language_confidence_detected",
        language=code,
        confidence=rounded_confidence,
        text_preview=text[:50],
    )

    return (code, rounded_confidence)
