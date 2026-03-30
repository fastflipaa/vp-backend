"""Human re-entry service -- builds context for AI resumption after human agent interaction.

When a human agent takes over a conversation (via human lock), the AI
pauses. After the lock expires or is released, the next AI message
should acknowledge the human interaction period.

This service:
1. Checks Redis for human_lock_history (persists 48h, outlives the 24h lock)
2. Fetches recent outbound messages from GHL conversation
3. Filters to human-sent messages (excludes bot/system/automation/workflow)
4. Optionally summarizes the human conversation via Claude Haiku
5. Returns a context dict that gets injected into the prompt pipeline

Fail-open: ANY error returns {"had_human_interaction": False} so the
pipeline is never blocked by a re-entry failure.
"""

from __future__ import annotations

import structlog

logger = structlog.get_logger()

# Guard TTL: prevent repeated re-entry acknowledgment (24h)
REENTRY_DONE_TTL = 86400


async def build_human_reentry_context(
    contact_id: str,
    conversation_id: str,
    redis_client,
    claude_service=None,
) -> dict:
    """Build context for AI re-entry after human agent interaction.

    Args:
        contact_id: GHL contact ID.
        conversation_id: GHL conversation ID.
        redis_client: Redis client (sync -- from Celery worker).
        claude_service: Optional ClaudeService for conversation summarization.

    Returns:
        Dict with keys:
        - had_human_interaction (bool): True if human agent was involved.
        - agent_name (str): Name of the human agent (if available).
        - human_summary (str): Summary of the human conversation.
        - messages_during_lock (int): Count of human messages found.
    """
    try:
        # Step 1: Check for human lock history (48h TTL key)
        history_key = f"human_lock_history:{contact_id}"
        lock_history = redis_client.get(history_key)

        if not lock_history:
            return {"had_human_interaction": False}

        # Step 2: Check reentry-already-handled guard
        done_key = f"human_reentry_done:{contact_id}"
        if redis_client.exists(done_key):
            return {"had_human_interaction": False}

        # Step 3: Fetch recent conversation messages from GHL
        from app.services import ghl_service

        messages = await ghl_service.get_conversation_messages(
            conversation_id, limit=20
        )

        # Step 4: Filter to human outbound messages (not bot/system/automation/workflow)
        excluded_sources = {"bot", "system", "automation", "workflow"}
        human_messages = [
            m
            for m in messages
            if m.get("source", "") not in excluded_sources
            and m.get("direction") == "outbound"
        ]

        if not human_messages:
            return {"had_human_interaction": False}

        # Step 5: Summarize via Claude (if available)
        summary = "un asesor atendio al lead previamente"
        if claude_service is not None:
            try:
                conversation_text = "\n".join(
                    f"[{m.get('source', 'agent')}] {m.get('body', m.get('message', ''))}"
                    for m in human_messages
                )
                summary = await claude_service.generate(
                    system_prompt=(
                        "Summarize this human agent conversation in 1-2 sentences. "
                        "Focus on what was discussed and any commitments made."
                    ),
                    user_message=conversation_text,
                    model="claude-haiku-4-20250514",
                    max_tokens=256,
                )
            except Exception:
                logger.exception(
                    "human_reentry_summarization_failed",
                    contact_id=contact_id,
                )
                summary = "un asesor atendio al lead previamente"

        # Step 6: Set reentry-done guard (prevent repeated acknowledgment)
        redis_client.set(done_key, "1", ex=REENTRY_DONE_TTL)

        # Step 7: Parse agent name from lock history string
        agent_name = "un asesor"
        if isinstance(lock_history, str) and lock_history.strip():
            # Lock history format is typically the agent info string
            agent_name = lock_history.strip() or "un asesor"

        logger.info(
            "human_reentry_context_built",
            contact_id=contact_id,
            agent_name=agent_name,
            human_messages_count=len(human_messages),
        )

        return {
            "had_human_interaction": True,
            "agent_name": agent_name,
            "human_summary": summary,
            "messages_during_lock": len(human_messages),
        }

    except Exception:
        logger.exception(
            "human_reentry_context_error",
            contact_id=contact_id,
        )
        # Fail-open: never block pipeline due to re-entry errors
        return {"had_human_interaction": False}
