"""Slack alerting Celery task.

Sends alerts to a Slack webhook URL for system-down and circuit breaker
emergencies. Gracefully skips if SLACK_WEBHOOK_URL is not configured.
"""

from __future__ import annotations

import httpx
import structlog

from app.celery_app import celery_app
from app.config import settings

logger = structlog.get_logger()


@celery_app.task(
    name="alerting.slack",
    bind=True,
    max_retries=2,
    default_retry_delay=5,
)
def send_slack_alert(self, message: str) -> None:  # type: ignore[override]
    """Send an alert message to Slack via incoming webhook.

    Gracefully skips if SLACK_WEBHOOK_URL is empty or not configured.
    Retries up to 2 times on transient errors.
    """
    webhook_url = settings.SLACK_WEBHOOK_URL
    if not webhook_url:
        logger.warning("slack_alert_skipped", reason="no_webhook_url_configured")
        return

    try:
        response = httpx.post(
            webhook_url,
            json={"text": f":rotating_light: VP-Backend Alert: {message}"},
            timeout=5.0,
        )
        response.raise_for_status()
        logger.info("slack_alert_sent", message=message[:100])
    except Exception as exc:
        logger.error(
            "slack_alert_failed",
            error=str(exc),
            message=message[:100],
        )
        raise self.retry(exc=exc)
