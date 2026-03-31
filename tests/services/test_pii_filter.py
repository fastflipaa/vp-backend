"""PII filter tests -- pure regex, no external dependencies.

Tests phone number stripping, email stripping, and preservation of
the lead's own contact information.
"""

from __future__ import annotations

from app.services.pii_filter import PIIFilter


class TestPIIFilterClean:
    """PIIFilter.clean() strips foreign PII while preserving lead's own."""

    def test_clean_text_passes_through(self):
        """Text without PII passes through unchanged."""
        text = "Hola, me interesa el departamento en Polanco"
        result = PIIFilter.clean(text)
        assert result == text

    def test_empty_string_returns_empty(self):
        """Empty input returns empty output."""
        assert PIIFilter.clean("") == ""

    def test_phone_number_detected_and_stripped(self):
        """Foreign phone numbers are replaced with placeholder."""
        text = "Llama a este numero: 5551234567 para info"
        result = PIIFilter.clean(text)
        assert "5551234567" not in result
        assert "[numero eliminado]" in result

    def test_email_detected_and_stripped(self):
        """Foreign email addresses are replaced with placeholder."""
        text = "Contacta a ventas@inmobiliaria.com para mas info"
        result = PIIFilter.clean(text)
        assert "ventas@inmobiliaria.com" not in result
        assert "[email eliminado]" in result

    def test_lead_own_phone_preserved(self):
        """Lead's own phone number is NOT stripped."""
        text = "Tu numero es +5215512345678, te contactamos pronto"
        result = PIIFilter.clean(text, lead_phone="+5215512345678")
        assert "+5215512345678" in result

    def test_lead_own_email_preserved(self):
        """Lead's own email is NOT stripped (case-insensitive)."""
        text = "Tu email carlos@test.com esta registrado"
        result = PIIFilter.clean(text, lead_email="Carlos@Test.com")
        assert "carlos@test.com" in result

    def test_foreign_phone_stripped_but_own_preserved(self):
        """Mixed text: foreign phone stripped, lead's phone kept."""
        text = "Tu numero +5215512345678 y llama a 5559876543"
        result = PIIFilter.clean(text, lead_phone="+5215512345678")
        assert "+5215512345678" in result
        assert "5559876543" not in result
        assert "[numero eliminado]" in result

    def test_multiple_emails_stripped(self):
        """Multiple foreign emails are all stripped."""
        text = "info@fake.com y admin@fake.com tienen datos"
        result = PIIFilter.clean(text)
        assert "info@fake.com" not in result
        assert "admin@fake.com" not in result
        assert result.count("[email eliminado]") == 2

    def test_credit_card_like_number_stripped(self):
        """Long numeric sequences (credit-card-like) are caught by phone pattern."""
        text = "Mi tarjeta 4111 1111 1111 1111 no la compartas"
        result = PIIFilter.clean(text)
        # Phone regex catches 10+ digit sequences with separators
        assert "4111 1111 1111 1111" not in result
