"""Worker startup smoke test + task failure observability.

Without these, broken components only surface on first task execution
(invisible to Coolify health checks) and uncaught task failures never
register in any ``monitor:errors:*`` counter (invisible to
``system_health_scan``).

REGRESSION CONTEXT: On Apr 5 2026 commit 5e64e63 added
``CLOSED.to(RE_ENGAGE)`` to ``ConversationSM``. CLOSED is ``final=True``
and python-statemachine forbids outgoing transitions from final states,
so ``ConversationSM.from_persisted_state(...)`` raised
``InvalidDefinition`` on every call. Worker boot succeeded (lazy import),
the in-task try/except in ``processing_task.process_message`` never
reached the structured error counter because the SM raised *before* the
try block, and ``system_health_scan`` reported ``processing_task=0``
errors and ``dlq_depth=0`` for 46 hours while 100% of customer messages
went unanswered. This module exists to make that failure mode impossible.

It does two things:

1. ``worker_process_init`` handler instantiates ``ConversationSM`` for
   every state at worker startup. If python-statemachine validation
   fails for any reason, the worker process exits non-zero, Coolify
   marks the container unhealthy, and the deploy fails. The bug is
   caught at deploy time, not in production.

2. ``task_failure`` handler catches *any* uncaught exception escaping
   *any* Celery task body and increments the ``monitor:errors:task_uncaught``
   counter. ``SystemHealthScanner`` reads this counter and alerts CRITICAL
   on any non-zero value, regardless of total error rate threshold.
"""

from __future__ import annotations

import redis
import structlog
from celery import signals

from app.config import settings

logger = structlog.get_logger()

# Per-worker Redis client used by the task_failure handler. Initialized in
# the worker_process_init handler below so each forked worker gets its own
# connection pool.
_failure_redis: redis.Redis | None = None

# Counter TTL matches SystemHealthScanner.WINDOW_SECONDS (5 minutes).
_FAILURE_KEY_TTL = 300


@signals.worker_process_init.connect
def _smoke_test_critical_components(**kwargs) -> None:
    """Pre-instantiate critical components at worker startup.

    Runs once per forked worker process. Any exception raised here
    causes the process to exit non-zero, which Coolify reports as
    unhealthy and the deploy is marked failed.

    Currently smoke-tests:
      * ``ConversationSM`` for every state value (catches
        python-statemachine graph validation errors)

    Add additional smoke tests here as new components become
    deploy-time critical.
    """
    global _failure_redis

    # 1. State machine -- python-statemachine validates on instantiation.
    # Test every state including final states (HANDOFF, CLOSED, BROKER)
    # and proactive states (FOLLOW_UP, RE_ENGAGE) to catch any future
    # graph validation regression.
    from app.state_machine.conversation_sm import ConversationSM

    smoke_states = (
        "GREETING",
        "QUALIFYING",
        "SCHEDULING",
        "QUALIFIED",
        "NON_RESPONSIVE",
        "RECOVERY",
        "FOLLOW_UP",
        "RE_ENGAGE",
        "HANDOFF",
        "CLOSED",
        "BROKER",
    )
    for state in smoke_states:
        ConversationSM.from_persisted_state(state, "startup_smoke_test")

    # 2. Initialize the Redis client used by the task_failure handler.
    pool = redis.ConnectionPool.from_url(
        settings.redis_cache_url,
        max_connections=5,
        decode_responses=True,
    )
    _failure_redis = redis.Redis(connection_pool=pool)

    logger.info("worker_smoke_test_passed", smoke_states_count=len(smoke_states))


@signals.task_failure.connect
def _on_task_failure(
    sender=None,
    task_id=None,
    exception=None,
    **kwargs,
) -> None:
    """Increment the global uncaught-failure counter on any task exception.

    Fires for any exception that escapes a Celery task body, including
    exceptions raised before any in-task try/except (e.g. SM
    instantiation, repository init, import-time validation). The
    ``monitor:errors:task_uncaught`` counter is read by
    ``SystemHealthScanner`` which treats any non-zero value as CRITICAL.
    """
    if _failure_redis is None:
        # Smoke test must have failed; the worker is shutting down anyway.
        return
    try:
        task_name = getattr(sender, "name", "unknown") if sender else "unknown"

        # Global counter -- system_health_scan reads this.
        _failure_redis.incr("monitor:errors:task_uncaught")
        _failure_redis.expire("monitor:errors:task_uncaught", _FAILURE_KEY_TTL)

        # Per-task-name counter for diagnostics.
        per_task_key = f"monitor:errors:task_uncaught:{task_name}"
        _failure_redis.incr(per_task_key)
        _failure_redis.expire(per_task_key, _FAILURE_KEY_TTL)

        logger.error(
            "celery_task_uncaught_failure",
            task_name=task_name,
            task_id=task_id,
            exception_type=type(exception).__name__ if exception else "unknown",
            exception=str(exception)[:500] if exception else "",
        )
    except Exception:
        # Observability code must NEVER break the worker.
        pass
