"""Pydantic schemas for webhook API endpoints."""

from __future__ import annotations

from pydantic import BaseModel


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


class WebhookResponse(BaseModel):
    """Standard response from webhook endpoints."""

    status: str
    trace_id: str
