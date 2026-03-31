"""Phone normalizer tests -- pure logic, no external dependencies.

Tests exact n8n port rules: 10-digit MX, 11-digit US, 12-digit MX+52,
13-digit MX+521, agent number rejection, and edge cases.
"""

from __future__ import annotations

import pytest

from app.services.phone_normalizer import AGENT_NUMBERS, normalize_phone


class TestNormalizePhone:
    """normalize_phone() E.164 normalization matching n8n Extract rules."""

    def test_10_digit_mx_mobile(self):
        """10 digits -> +52XXXXXXXXXX (Mexican mobile)."""
        assert normalize_phone("5512345678") == "+525512345678"

    def test_11_digit_us_canada(self):
        """11 digits starting with 1 -> +1XXXXXXXXXX (US/Canada)."""
        assert normalize_phone("12125551234") == "+12125551234"

    def test_12_digit_mx_with_country_code(self):
        """12 digits starting with 52 -> +52XXXXXXXXXX."""
        assert normalize_phone("525512345678") == "+525512345678"

    def test_13_digit_mx_mobile_with_country_plus_1(self):
        """13 digits starting with 521 -> +52XXXXXXXXXX (strips the 1)."""
        assert normalize_phone("5215512345678") == "+525512345678"

    def test_already_has_plus_passthrough(self):
        """Numbers already starting with + pass through."""
        assert normalize_phone("+525512345678") == "+525512345678"

    def test_agent_number_rejection_raw(self):
        """Known agent number (raw digits) returns None."""
        assert normalize_phone("2292656397") is None

    def test_agent_number_rejection_with_plus(self):
        """Known agent number with + prefix returns None."""
        assert normalize_phone("+522292656397") is None

    def test_agent_number_rejection_with_52(self):
        """Known agent number with 52 prefix returns None."""
        assert normalize_phone("+2292656397") is None

    def test_empty_string_returns_none(self):
        """Empty string returns None."""
        assert normalize_phone("") is None

    def test_none_like_empty_returns_none(self):
        """Whitespace-only returns None."""
        assert normalize_phone("   ") is None

    def test_strips_non_digits(self):
        """Non-digit characters (spaces, dashes, parens) are stripped."""
        assert normalize_phone("(55) 1234-5678") == "+525512345678"

    def test_short_number_gets_plus_prefix(self):
        """Numbers not matching any pattern get + prefix."""
        result = normalize_phone("12345")
        assert result == "+12345"

    def test_agent_numbers_set_has_known_values(self):
        """AGENT_NUMBERS constant has the known agent phone variants."""
        assert "+522292656397" in AGENT_NUMBERS
        assert "+2292656397" in AGENT_NUMBERS
        assert "2292656397" in AGENT_NUMBERS
