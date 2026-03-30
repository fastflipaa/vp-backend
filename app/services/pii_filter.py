"""PII filter for cleaning AI responses before delivery.

Ports the n8n "PII Output Filter" logic from Greeting and Qualifying
handlers. Strips foreign phone numbers and email addresses from AI
responses while preserving the lead's own contact information.

Usage:
    cleaned = PIIFilter.clean(response, lead_phone="+525512345678")
"""

from __future__ import annotations

import re

import structlog

from app.services.phone_normalizer import normalize_phone

logger = structlog.get_logger()

# Phone: any phone-like pattern (10+ digits with optional separators and prefix)
PHONE_PATTERN = re.compile(r"[\+]?[\d\s\-\(\)]{10,}")

# Email: standard email regex
EMAIL_PATTERN = re.compile(r"[a-zA-Z0-9._%+\-]+@[a-zA-Z0-9.\-]+\.[a-zA-Z]{2,}")


class PIIFilter:
    """Strips foreign PII from AI responses.

    Removes phone numbers and emails that do NOT belong to the lead.
    Logs counts of removed items (never the actual PII).
    """

    @staticmethod
    def clean(
        response: str, lead_phone: str = "", lead_email: str = ""
    ) -> str:
        """Clean PII from an AI response.

        Args:
            response: Raw AI response text.
            lead_phone: The lead's own phone (E.164). Will NOT be stripped.
            lead_email: The lead's own email. Will NOT be stripped.

        Returns:
            Cleaned response with foreign PII replaced by placeholders.
        """
        if not response:
            return response

        phones_removed = 0
        emails_removed = 0

        # Normalize lead phone for comparison
        lead_phone_normalized = normalize_phone(lead_phone) if lead_phone else ""

        # Replace foreign phone numbers
        def replace_phone(match: re.Match) -> str:
            nonlocal phones_removed
            matched = match.group(0)
            # Normalize the matched phone for comparison
            normalized = normalize_phone(matched)
            if normalized and lead_phone_normalized and normalized == lead_phone_normalized:
                return matched  # Keep lead's own phone
            phones_removed += 1
            return "[numero eliminado]"

        cleaned = PHONE_PATTERN.sub(replace_phone, response)

        # Replace foreign email addresses
        def replace_email(match: re.Match) -> str:
            nonlocal emails_removed
            matched = match.group(0)
            if lead_email and matched.lower() == lead_email.lower():
                return matched  # Keep lead's own email
            emails_removed += 1
            return "[email eliminado]"

        cleaned = EMAIL_PATTERN.sub(replace_email, cleaned)

        total_removed = phones_removed + emails_removed
        if total_removed > 0:
            logger.info(
                "pii_filter.cleaned",
                phones_removed=phones_removed,
                emails_removed=emails_removed,
                total_removed=total_removed,
            )

        return cleaned
