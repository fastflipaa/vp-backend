"""Alert manager for monitoring agent -- Slack webhooks + GHL notes with dedup.

Uses Redis SETNX with 30-minute TTL for alert deduplication.
Alert levels: INFO (Slack only), WARNING (Slack + GHL note), CRITICAL (Slack + GHL note + tag).
"""

from __future__ import annotations

import hashlib
import json
from enum import Enum

import httpx
import redis
import structlog

from app.config import settings

logger = structlog.get_logger()

DEDUP_TTL = 1800  # 30 minutes


class AlertLevel(str, Enum):
    INFO = "INFO"
    WARNING = "WARNING"
    CRITICAL = "CRITICAL"


# Slack emoji mapping
_LEVEL_EMOJI = {
    AlertLevel.INFO: ":information_source:",
    AlertLevel.WARNING: ":warning:",
    AlertLevel.CRITICAL: ":rotating_light:",
}


class AlertManager:
    """Sends alerts to Slack and GHL with Redis-based deduplication."""

    def __init__(self, redis_client: redis.Redis) -> None:
        self._redis = redis_client

    def _dedup_key(self, alert_type: str, entity_id: str) -> str:
        """Build dedup key: monitor:dedup:{hash(type+entity)}."""
        raw = f"{alert_type}:{entity_id}"
        h = hashlib.md5(raw.encode()).hexdigest()[:12]
        return f"monitor:dedup:{h}"

    def _is_duplicate(self, alert_type: str, entity_id: str) -> bool:
        """Check if this alert was already sent within DEDUP_TTL."""
        key = self._dedup_key(alert_type, entity_id)
        was_set = self._redis.set(key, "1", nx=True, ex=DEDUP_TTL)
        return not was_set  # If set() returned None, key already existed = duplicate

    async def send(
        self,
        alert_type: str,
        message: str,
        level: AlertLevel = AlertLevel.WARNING,
        contact_id: str | None = None,
        entity_id: str | None = None,
        extra: dict | None = None,
    ) -> bool:
        """Send an alert with deduplication.

        Args:
            alert_type: Category (e.g. "repetition", "hallucination", "error_rate").
            message: Human-readable alert message.
            level: INFO, WARNING, or CRITICAL.
            contact_id: GHL contact ID (for per-lead GHL actions).
            entity_id: Dedup entity (defaults to contact_id or "system").
            extra: Additional metadata for the Slack message.

        Returns:
            True if alert was sent, False if deduplicated.
        """
        dedup_id = entity_id or contact_id or "system"

        if self._is_duplicate(alert_type, dedup_id):
            logger.debug(
                "alert.deduplicated",
                alert_type=alert_type,
                entity_id=dedup_id,
            )
            return False

        # Slack alert (all levels)
        await self._send_slack(alert_type, message, level, extra)

        # GHL note (WARNING + CRITICAL only, requires contact_id)
        if level in (AlertLevel.WARNING, AlertLevel.CRITICAL) and contact_id:
            await self._add_ghl_note(contact_id, alert_type, message)

        # GHL tag (CRITICAL only, requires contact_id)
        if level == AlertLevel.CRITICAL and contact_id:
            await self._add_ghl_tag(contact_id, "needs-human-review")

        logger.info(
            "alert.sent",
            alert_type=alert_type,
            level=level.value,
            entity_id=dedup_id,
            contact_id=contact_id,
        )
        return True

    async def _send_slack(
        self,
        alert_type: str,
        message: str,
        level: AlertLevel,
        extra: dict | None = None,
    ) -> None:
        """Post to Slack webhook. Gracefully skip if not configured."""
        webhook_url = settings.SLACK_WEBHOOK_URL
        if not webhook_url:
            logger.warning("alert.slack_skipped", reason="no_webhook_url")
            return

        emoji = _LEVEL_EMOJI.get(level, ":bell:")
        blocks = [
            {
                "type": "header",
                "text": {"type": "plain_text", "text": f"{emoji} [{level.value}] {alert_type}"},
            },
            {
                "type": "section",
                "text": {"type": "mrkdwn", "text": message},
            },
        ]
        if extra:
            fields = [
                {"type": "mrkdwn", "text": f"*{k}:* {v}"}
                for k, v in extra.items()
            ]
            blocks.append({"type": "section", "fields": fields[:10]})  # Slack limit

        try:
            resp = httpx.post(
                webhook_url,
                json={"blocks": blocks, "text": f"[{level.value}] {alert_type}: {message}"},
                timeout=5.0,
            )
            resp.raise_for_status()
        except Exception as exc:
            logger.error("alert.slack_failed", error=str(exc), alert_type=alert_type)

    async def _add_ghl_note(self, contact_id: str, alert_type: str, message: str) -> None:
        """Add a note to the GHL contact."""
        from app.services.ghl_service import add_note
        try:
            body = f"[Monitor - {alert_type}] {message}"
            await add_note(contact_id, body[:500])  # GHL note limit safety
        except Exception as exc:
            logger.error("alert.ghl_note_failed", error=str(exc), contact_id=contact_id)

    async def _add_ghl_tag(self, contact_id: str, tag: str) -> None:
        """Add a tag to the GHL contact."""
        from app.services.ghl_service import add_tag
        try:
            await add_tag(contact_id, tag)
        except Exception as exc:
            logger.error("alert.ghl_tag_failed", error=str(exc), contact_id=contact_id)
