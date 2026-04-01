"""GHL Enrichment Service — fetches contact + notes + messages from GHL.

Replicates n8n Main Router Stage 4 (CRM Enrichment): fetches contact data,
recent notes, and recent conversation messages from GoHighLevel, assembling
them into an ``enrichedContext`` dict that is injected into Claude prompts.

Fail-open design: if any GHL API call fails, the error is logged and the
enrichment returns partial context. Enrichment failure NEVER blocks the
pipeline.

Usage:
    svc = GHLEnrichmentService()
    context = await svc.enrich("contact_abc", "conv_123")
"""

from __future__ import annotations

import structlog

from app.services import ghl_service

logger = structlog.get_logger()


class GHLEnrichmentService:
    """Assembles enrichedContext from 3 GHL API endpoints.

    Fetches:
    1. Contact data (tags, custom fields, source, name, email, phone)
    2. Recent notes (last 5)
    3. Recent conversation messages (last 10, if conversation_id provided)
    """

    @staticmethod
    def _extract_custom_field(custom_fields, field_name: str) -> str:
        """Extract a value from GHL customFields (list or dict format)."""
        if isinstance(custom_fields, dict):
            return custom_fields.get(field_name, "")
        if isinstance(custom_fields, list):
            for f in custom_fields:
                if isinstance(f, dict) and f.get("id", "") == field_name:
                    return f.get("value", "")
        return ""

    async def enrich(
        self, contact_id: str, conversation_id: str = ""
    ) -> dict:
        """Fetch and assemble CRM enrichment context.

        Args:
            contact_id: GHL contact ID.
            conversation_id: GHL conversation ID (optional).

        Returns:
            Dict with keys: tags, customFields, source, name, email,
            recentNotes, recentMessages, formName. Partial data returned
            on individual endpoint failures (fail-open).
        """
        contact: dict = {}
        notes: list[dict] = []
        messages: list[dict] = []

        # 1. Fetch contact
        try:
            raw = await ghl_service.get_contact(contact_id)
            contact = raw.get("contact", raw) if isinstance(raw, dict) else {}
        except Exception as e:
            logger.warning(
                "ghl_enrichment.contact_failed",
                contact_id=contact_id,
                error=str(e),
            )

        # 2. Fetch notes
        try:
            raw_notes = await ghl_service.get_contact_notes(contact_id)
            notes = raw_notes if isinstance(raw_notes, list) else []
        except Exception as e:
            logger.warning(
                "ghl_enrichment.notes_failed",
                contact_id=contact_id,
                error=str(e),
            )

        # 3. Fetch recent messages -- search for conversation if ID not provided
        resolved_conversation_id = conversation_id
        if not resolved_conversation_id and contact_id:
            try:
                conv = await ghl_service.search_conversations(contact_id)
                if conv:
                    resolved_conversation_id = conv.get("id", "")
            except Exception as e:
                logger.warning(
                    "ghl_enrichment.conversation_search_failed",
                    contact_id=contact_id,
                    error=str(e),
                )

        if resolved_conversation_id:
            try:
                messages = await ghl_service.get_conversation_messages(
                    resolved_conversation_id, limit=10
                )
            except Exception as e:
                logger.warning(
                    "ghl_enrichment.messages_failed",
                    conversation_id=resolved_conversation_id,
                    error=str(e),
                )

        # Assemble enrichedContext (same structure as n8n Stage 4)
        enriched = {
            "tags": contact.get("tags", []),
            "customFields": contact.get("customFields", {}),
            "source": contact.get("source", ""),
            "name": contact.get("name", ""),
            "email": contact.get("email", ""),
            "recentNotes": [
                n.get("body", "") for n in notes[:5] if isinstance(n, dict)
            ],
            "recentMessages": [
                {
                    "text": m.get("body", ""),
                    "direction": m.get("direction", ""),
                }
                for m in messages[:10]
                if isinstance(m, dict)
            ],
            "formName": GHLEnrichmentService._extract_custom_field(contact.get("customFields", []), "formName"),
        }

        logger.info(
            "ghl_enrichment.complete",
            contact_id=contact_id,
            tags_count=len(enriched["tags"]),
            notes_count=len(enriched["recentNotes"]),
            messages_count=len(enriched["recentMessages"]),
        )

        return enriched
