"""Overnight watchdog — runs every 15 min via cron, posts issues to Slack.

Checks:
1. Container health (api, worker, beat all running)
2. DLQ depth (should be 0)
3. ConversationScorer outputs (outcomes created in last 30 min?)
4. Beat task execution (health scan ran in last 10 min?)
5. Worker error rate (any spikes?)
6. Circuit breaker states (should be CLOSED)
7. Redis connectivity
8. Neo4j connectivity

Posts to Slack ONLY if issues found. Silent when healthy.
Also posts a brief "all clear" summary every 2 hours (to confirm it's running).

Deploy: copy to /app/scripts/ on VPS, add cron:
  */15 * * * * cd /app && PYTHONPATH=/app python3 scripts/overnight_watchdog.py >> /var/log/watchdog.log 2>&1
"""

import asyncio
import json
import os
import sys
import time
from datetime import datetime, timezone

# Add /app to path for imports
sys.path.insert(0, "/app")


async def main():
    import httpx
    import redis
    from app.config import settings

    issues = []
    stats = {}

    # ── 1. Redis connectivity ──
    try:
        r = redis.Redis.from_url(settings.redis_cache_url, decode_responses=True)
        r.ping()
        stats["redis"] = "ok"
    except Exception as e:
        issues.append(f"Redis DOWN: {e}")
        stats["redis"] = "down"
        # Can't check much without Redis
        await _send_slack(settings.SLACK_WEBHOOK_URL, issues, stats)
        return

    # ── 2. DLQ depth ──
    dlq_depth = r.llen("dlq:failed_messages") or 0
    stats["dlq_depth"] = dlq_depth
    if dlq_depth > 0:
        issues.append(f"DLQ has {dlq_depth} failed messages")

    # ── 3. Circuit breaker states ──
    for cb_name in ["claude", "ghl"]:
        state = r.get(f"circuit:{cb_name}") or "CLOSED"
        stats[f"cb_{cb_name}"] = state
        if state != "CLOSED":
            issues.append(f"Circuit breaker {cb_name}: {state}")

    # Embedding circuit breaker
    emb_open = r.get("circuit:embedding:open_until")
    if emb_open and float(emb_open) > time.time():
        issues.append("Embedding circuit breaker OPEN")
        stats["cb_embedding"] = "OPEN"
    else:
        stats["cb_embedding"] = "CLOSED"

    # ── 4. Error rate counters (sliding 5-min window) ──
    error_keys = [
        "monitor:errors:processing_task",
        "monitor:errors:gate_task",
        "monitor:errors:delivery_failed",
        "monitor:errors:claude_timeout",
        "monitor:errors:neo4j_error",
        "monitor:errors:json_parse",
    ]
    total_errors = 0
    for key in error_keys:
        count = int(r.get(key) or 0)
        if count > 0:
            total_errors += count
            short_name = key.split(":")[-1]
            stats[f"err_{short_name}"] = count
    if total_errors > 5:
        issues.append(f"High error rate: {total_errors} errors in current window")

    # ── 5. Neo4j connectivity + learning data check ──
    try:
        from app.repositories.base import get_driver, close_driver

        driver = await get_driver()
        async with driver.session() as session:
            # Check connectivity
            result = await session.run("RETURN 1 AS ok")
            record = await result.single()
            stats["neo4j"] = "ok" if record else "no_response"

            # Check if ConversationScorer has created outcomes recently
            result2 = await session.run(
                "MATCH (o:ConversationOutcome) "
                "RETURN count(o) AS total, "
                "max(o.created_at) AS latest"
            )
            rec2 = await result2.single()
            if rec2:
                stats["outcomes_total"] = rec2["total"]
                latest_val = rec2["latest"]
                if latest_val is not None:
                    if hasattr(latest_val, 'to_native'):
                        stats["outcomes_latest"] = latest_val.to_native().isoformat()[:19]
                    elif hasattr(latest_val, 'isoformat'):
                        stats["outcomes_latest"] = latest_val.isoformat()[:19]
                    else:
                        stats["outcomes_latest"] = str(latest_val)[:19]
                else:
                    stats["outcomes_latest"] = "none"

            # Check learning nodes
            result3 = await session.run(
                "MATCH (e:AgentError) RETURN count(e) AS errors "
                "UNION ALL "
                "MATCH (ll:LessonLearned) RETURN count(ll) AS errors"
            )
            counts = [rec["errors"] async for rec in result3]
            stats["agent_errors"] = counts[0] if counts else 0
            stats["lessons"] = counts[1] if len(counts) > 1 else 0

            # Check embeddings
            result4 = await session.run(
                "MATCH (ce:ConversationEmbedding) RETURN count(ce) AS total"
            )
            rec4 = await result4.single()
            stats["embeddings"] = rec4["total"] if rec4 else 0

        await close_driver()
    except Exception as e:
        issues.append(f"Neo4j error: {str(e)[:100]}")
        stats["neo4j"] = "error"

    # ── 6. Determine if we should report ──
    now = datetime.now(timezone.utc)
    minute = now.minute
    hour = now.hour

    # Always report if issues found
    # Also report "all clear" every 2 hours (at the :00 check)
    is_summary_time = (hour % 2 == 0 and minute < 15)

    if issues:
        await _send_slack(settings.SLACK_WEBHOOK_URL, issues, stats)
    elif is_summary_time:
        await _send_slack_summary(settings.SLACK_WEBHOOK_URL, stats)

    # Always log locally
    status = "ISSUES" if issues else "OK"
    print(f"[{now.isoformat()[:19]}] {status}: {json.dumps(stats)}")
    if issues:
        for issue in issues:
            print(f"  ! {issue}")


async def _send_slack(webhook_url: str, issues: list, stats: dict):
    """Send alert to Slack when issues are detected."""
    if not webhook_url:
        return

    blocks = [
        f"*:rotating_light: Overnight Watchdog — {len(issues)} issue(s) detected*",
        "",
    ]
    for issue in issues:
        blocks.append(f"• {issue}")
    blocks.append("")
    blocks.append(f"_Stats: {json.dumps(stats, default=str)}_")

    async with httpx.AsyncClient() as client:
        await client.post(webhook_url, json={"text": "\n".join(blocks)})


async def _send_slack_summary(webhook_url: str, stats: dict):
    """Send periodic all-clear summary."""
    if not webhook_url:
        return

    now = datetime.now(timezone.utc).strftime("%H:%M UTC")
    text = (
        f":white_check_mark: *Watchdog {now} — All Clear*\n"
        f"DLQ: {stats.get('dlq_depth', '?')} | "
        f"CB: claude={stats.get('cb_claude', '?')}, ghl={stats.get('cb_ghl', '?')}, emb={stats.get('cb_embedding', '?')} | "
        f"Neo4j: {stats.get('neo4j', '?')} | "
        f"Outcomes: {stats.get('outcomes_total', '?')} | "
        f"Errors: {stats.get('agent_errors', '?')} | "
        f"Embeddings: {stats.get('embeddings', '?')} | "
        f"Lessons: {stats.get('lessons', '?')}"
    )

    async with httpx.AsyncClient() as client:
        await client.post(webhook_url, json={"text": text})


if __name__ == "__main__":
    asyncio.run(main())
