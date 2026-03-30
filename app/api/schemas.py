"""Pydantic schemas for webhook API endpoints."""

from __future__ import annotations

from pydantic import BaseModel, field_validator

from app.services.phone_normalizer import normalize_phone


class WebhookPayload(BaseModel):
    """Inbound webhook payload from n8n shadow forwarding."""

    contactId: str = ""
    messageId: str = ""
    message: str = ""
    phone: str = ""
    direction: str = "inbound"
    channel: str = ""
    source: str = ""
    n8n_gate_decisions: dict[str, str] = {}


class InboundPayload(BaseModel):
    """Full inbound webhook payload with all fields from n8n Extract Message Data.

    Auto-normalizes the phone field to E.164 format on construction.
    Extends WebhookPayload with additional CRM fields needed by the
    core processing pipeline.
    """

    contactId: str = ""
    messageId: str = ""
    message: str = ""
    phone: str = ""
    direction: str = "inbound"
    channel: str = ""
    source: str = ""
    contactName: str = ""
    email: str = ""
    locationId: str = ""
    conversationId: str = ""
    tags: list[str] = []
    isAutoTrigger: bool = False
    messageType: str = ""  # re_engagement, post_appointment, etc.
    n8n_gate_decisions: dict[str, str] = {}

    @field_validator("phone", mode="before")
    @classmethod
    def normalize_phone_field(cls, v: str) -> str:
        """Normalize phone to E.164 format via phone_normalizer."""
        if not v:
            return ""
        normalized = normalize_phone(str(v))
        return normalized or ""


class WebhookResponse(BaseModel):
    """Standard response from webhook endpoints."""

    status: str
    trace_id: str
