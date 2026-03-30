"""Phone normalization matching n8n Extract Message Data logic exactly.

Normalizes raw phone strings to E.164 format with Mexico (+52) and US (+1)
handling. Rejects known agent phone numbers to prevent AI self-conversation.

Rules (from n8n Main Router Stage 0 - Extract Message Data):
    - 10 digits -> +52XXXXXXXXXX (Mexican mobile)
    - 11 digits starting with 1 -> +1XXXXXXXXXX (US/Canada)
    - 12 digits starting with 52 -> +52XXXXXXXXXX (MX with country code)
    - 13 digits starting with 521 -> +52XXXXXXXXXX (MX mobile with country+1)
    - Already has + -> pass through
    - Known agent numbers -> return None (reject)
"""

from __future__ import annotations

import re

# Known agent numbers to reject (from n8n audit)
AGENT_NUMBERS = {"+522292656397", "+2292656397", "2292656397"}


def normalize_phone(raw_phone: str) -> str | None:
    """Normalize phone to E.164 format. Returns None for agent numbers.

    Args:
        raw_phone: Raw phone string from GHL webhook.

    Returns:
        Normalized E.164 phone string, or None if the phone is a known
        agent number or empty.
    """
    if not raw_phone:
        return None

    digits = re.sub(r"[^0-9+]", "", raw_phone.strip())

    if digits in AGENT_NUMBERS or f"+{digits}" in AGENT_NUMBERS:
        return None

    if digits.startswith("+"):
        return digits if digits not in AGENT_NUMBERS else None

    if len(digits) == 10:
        return f"+52{digits}"
    elif len(digits) == 11 and digits.startswith("1"):
        return f"+{digits}"
    elif len(digits) == 12 and digits.startswith("52"):
        return f"+{digits}"
    elif len(digits) == 13 and digits.startswith("521"):
        # MX mobile with country code + 1
        return f"+52{digits[3:]}"
    else:
        return f"+{digits}" if digits else None
