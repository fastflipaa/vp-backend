"""GHL httpx client with tenacity retry and structured logging.

Module-level AsyncClient singleton targeting the GoHighLevel v2 API at
``services.leadconnectorhq.com``. All methods retry on 429 (rate limit)
and 5xx (server error) with exponential backoff + jitter via tenacity.
Connection and timeout errors are also retried.

Usage:
    from app.services.ghl_service import get_ghl_client, send_message
    client = get_ghl_client()
    result = await send_message("contact_abc", "Hola!", "SMS")
"""

from __future__ import annotations

import httpx
import structlog
from tenacity import (
    RetryError,
    retry,
    retry_if_exception,
    stop_after_attempt,
    wait_random_exponential,
)

from app.config import settings

logger = structlog.get_logger()

# --- Per-call client factory ---
# Each asyncio.run() creates a new event loop, so we cannot cache an
# AsyncClient across calls (it binds to the loop that created it).
# Instead, create a fresh client per call.  httpx.AsyncClient is cheap
# to instantiate and handles connection pooling internally.


def get_ghl_client() -> httpx.AsyncClient:
    """Return a fresh httpx AsyncClient for GHL API calls."""
    return httpx.AsyncClient(
        base_url="https://services.leadconnectorhq.com",
        headers={
            "Authorization": f"Bearer {settings.GHL_TOKEN}",
            "Version": "2021-07-28",
            "Content-Type": "application/json",
        },
        timeout=httpx.Timeout(30.0, connect=10.0),
    )


# --- Retry predicate ---


def is_retryable(exc: BaseException) -> bool:
    """Return True if the exception should trigger a tenacity retry.

    Retries on:
    - HTTP 429 (rate limit)
    - HTTP 5xx (server error)
    - Connection errors
    - Timeout errors
    """
    if isinstance(exc, httpx.HTTPStatusError):
        return exc.response.status_code == 429 or exc.response.status_code >= 500
    if isinstance(exc, (httpx.ConnectError, httpx.TimeoutException)):
        return True
    return False


_retry_decorator = retry(
    wait=wait_random_exponential(multiplier=1, max=30),
    stop=stop_after_attempt(3),
    retry=retry_if_exception(is_retryable),
    reraise=True,
)


# --- API methods ---


@_retry_decorator
async def send_message(
    contact_id: str, message: str, channel: str = "SMS"
) -> dict:
    """Send a message via GHL conversations API.

    Args:
        contact_id: GHL contact ID.
        message: Message text to send.
        channel: Delivery channel (SMS, WhatsApp, Email, etc.).

    Returns:
        Parsed JSON response from GHL.

    Raises:
        httpx.HTTPStatusError: On non-retryable HTTP errors (after 3 retries
            for retryable ones).
    """
    client = get_ghl_client()
    response = await client.post(
        "/conversations/messages",
        json={"type": channel, "contactId": contact_id, "message": message},
    )
    response.raise_for_status()
    logger.info(
        "ghl.send_message",
        contact_id=contact_id,
        channel=channel,
        status=response.status_code,
    )
    return response.json()


@_retry_decorator
async def get_contact(contact_id: str) -> dict:
    """Fetch a contact by ID from GHL.

    Returns:
        Parsed JSON response with contact data.
    """
    client = get_ghl_client()
    response = await client.get(f"/contacts/{contact_id}")
    response.raise_for_status()
    logger.debug("ghl.get_contact", contact_id=contact_id, status=response.status_code)
    return response.json()


@_retry_decorator
async def get_contact_notes(contact_id: str) -> list[dict]:
    """Fetch notes for a contact from GHL.

    Returns:
        List of note dicts.
    """
    client = get_ghl_client()
    response = await client.get(f"/contacts/{contact_id}/notes")
    response.raise_for_status()
    data = response.json()
    notes = data.get("notes", data) if isinstance(data, dict) else data
    logger.debug(
        "ghl.get_contact_notes",
        contact_id=contact_id,
        count=len(notes) if isinstance(notes, list) else 0,
    )
    return notes if isinstance(notes, list) else []


@_retry_decorator
async def get_conversation_messages(
    conversation_id: str, limit: int = 10
) -> list[dict]:
    """Fetch recent messages from a GHL conversation.

    Args:
        conversation_id: GHL conversation ID.
        limit: Max messages to return.

    Returns:
        List of message dicts.
    """
    client = get_ghl_client()
    response = await client.get(
        f"/conversations/{conversation_id}/messages",
        params={"limit": limit},
    )
    response.raise_for_status()
    data = response.json()
    messages = data.get("messages", data) if isinstance(data, dict) else data
    logger.debug(
        "ghl.get_conversation_messages",
        conversation_id=conversation_id,
        count=len(messages) if isinstance(messages, list) else 0,
    )
    return messages if isinstance(messages, list) else []


@_retry_decorator
async def search_conversations(contact_id: str) -> dict | None:
    """Search for a conversation by contact ID.

    Returns:
        First conversation dict, or None if no conversations found.
    """
    client = get_ghl_client()
    response = await client.get(
        "/conversations/search",
        params={"contactId": contact_id},
    )
    response.raise_for_status()
    data = response.json()
    conversations = data.get("conversations", [])
    if conversations:
        logger.debug(
            "ghl.search_conversations",
            contact_id=contact_id,
            found=True,
        )
        return conversations[0]
    logger.debug(
        "ghl.search_conversations",
        contact_id=contact_id,
        found=False,
    )
    return None


@_retry_decorator
async def add_tag(contact_id: str, tag: str) -> None:
    """Add a tag to a GHL contact."""
    client = get_ghl_client()
    response = await client.post(
        f"/contacts/{contact_id}/tags",
        json={"tags": [tag]},
    )
    response.raise_for_status()
    logger.info("ghl.add_tag", contact_id=contact_id, tag=tag)


@_retry_decorator
async def add_note(contact_id: str, body: str) -> None:
    """Add a note to a GHL contact."""
    client = get_ghl_client()
    response = await client.post(
        f"/contacts/{contact_id}/notes",
        json={"body": body},
    )
    response.raise_for_status()
    logger.info("ghl.add_note", contact_id=contact_id)


@_retry_decorator
async def update_contact(contact_id: str, data: dict) -> dict:
    """Update a GHL contact with arbitrary fields.

    Args:
        contact_id: GHL contact ID.
        data: Dict of fields to update.

    Returns:
        Parsed JSON response.
    """
    client = get_ghl_client()
    response = await client.patch(
        f"/contacts/{contact_id}",
        json=data,
    )
    response.raise_for_status()
    logger.info("ghl.update_contact", contact_id=contact_id)
    return response.json()
