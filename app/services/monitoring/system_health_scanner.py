"""System health scanner -- error rates, circuit breakers, delivery failures.

Reads Redis sliding-window counters and circuit breaker states.
Designed for use inside a Celery Beat scheduled task (every 5 min).

Monitors:
1. Celery task failure rate (per 5-min window via Redis counters)
2. Circuit breaker states (reads existing circuit:{name} keys)
3. GHL delivery failure rate (counts DLQ entries in last window)
4. JSON parse failure tracking (when Claude returns text instead of JSON)
5. State machine transition error counting
"""

from __future__ import annotations

import time

import redis
import structlog

from app.services.monitoring.alert_manager import AlertLevel, AlertManager

logger = structlog.get_logger()

# Thresholds
ERROR_RATE_WARNING = 5       # 5 errors in 5 min window
ERROR_RATE_CRITICAL = 15     # 15 errors in 5 min window
DLQ_WARNING = 3              # 3 DLQ entries
DLQ_CRITICAL = 10            # 10 DLQ entries
WINDOW_SECONDS = 300         # 5-minute sliding window


class SystemHealthScanner:
    """Scans system health metrics from Redis."""

    def __init__(
        self,
        redis_client: redis.Redis,
        alert_manager: AlertManager,
    ) -> None:
        self._redis = redis_client
        self._alert = alert_manager

    async def scan(self) -> dict:
        """Run all system health checks.

        Returns dict with health metrics.
        """
        metrics = {}

        try:
            # 1. Circuit breaker states
            cb_states = self._check_circuit_breakers()
            metrics["circuit_breakers"] = cb_states

            # 2. Error rate counters
            error_counts = self._check_error_rates()
            metrics["error_counts"] = error_counts

            # 3. DLQ depth
            dlq_depth = self._check_dlq_depth()
            metrics["dlq_depth"] = dlq_depth

            # 4. JSON parse failures
            json_failures = self._check_json_failures()
            metrics["json_parse_failures"] = json_failures

            # 5. State machine errors
            sm_errors = self._check_sm_errors()
            metrics["state_machine_errors"] = sm_errors

        except Exception:
            logger.exception("system_health_scan.error")

        return metrics

    def _check_circuit_breakers(self) -> dict[str, str]:
        """Read circuit breaker states from Redis.

        Known circuit breakers: claude, ghl, neo4j (future).
        """
        states = {}
        for name in ("claude", "ghl"):
            state_key = f"circuit:{name}"
            state = self._redis.get(state_key)
            cb_state = state if state else "CLOSED"
            states[name] = cb_state

            if cb_state == "OPEN":
                # Fire-and-forget: we can't await in a sync method,
                # so we record it and the caller (scan) will alert
                self._redis.incr("monitor:cb_open_alerts")
                self._redis.expire("monitor:cb_open_alerts", WINDOW_SECONDS)

        return states

    def _check_error_rates(self) -> dict[str, int]:
        """Check sliding-window error counters.

        Error keys are incremented by the system (processing task failures,
        gate failures, etc.) using the pattern: monitor:errors:{category}
        """
        categories = [
            "processing_task",
            "gate_task",
            "delivery_failed",
            "claude_timeout",
            "neo4j_error",
        ]
        counts = {}
        for cat in categories:
            key = f"monitor:errors:{cat}"
            val = self._redis.get(key)
            count = int(val) if val else 0
            counts[cat] = count

        return counts

    def _check_dlq_depth(self) -> int:
        """Check the depth of the dead-letter queue."""
        try:
            depth = self._redis.llen("dlq:failed_messages") or 0
            return depth
        except Exception:
            logger.exception("dlq_depth_check_failed")
            return 0

    def _check_json_failures(self) -> int:
        """Check JSON parse failure counter in current window."""
        key = "monitor:errors:json_parse"
        val = self._redis.get(key)
        return int(val) if val else 0

    def _check_sm_errors(self) -> int:
        """Check state machine transition error counter."""
        key = "monitor:errors:sm_transition"
        val = self._redis.get(key)
        return int(val) if val else 0

    async def alert_on_issues(self, metrics: dict) -> int:
        """Evaluate metrics and send alerts for any issues found.

        Returns count of alerts sent.
        """
        alerts_sent = 0

        # Circuit breaker alerts
        for name, state in metrics.get("circuit_breakers", {}).items():
            if state == "OPEN":
                sent = await self._alert.send(
                    alert_type=f"circuit_breaker_{name}",
                    message=f"Circuit breaker `{name}` is OPEN. Requests to {name} are being blocked.",
                    level=AlertLevel.CRITICAL,
                    entity_id=f"circuit_{name}",
                    extra={"service": name, "state": state},
                )
                if sent:
                    alerts_sent += 1

        # Error rate alerts
        total_errors = sum(metrics.get("error_counts", {}).values())
        if total_errors >= ERROR_RATE_CRITICAL:
            sent = await self._alert.send(
                alert_type="high_error_rate",
                message=f"CRITICAL error rate: {total_errors} errors in last 5 minutes.",
                level=AlertLevel.CRITICAL,
                entity_id="error_rate",
                extra={k: str(v) for k, v in metrics.get("error_counts", {}).items() if v > 0},
            )
            if sent:
                alerts_sent += 1
        elif total_errors >= ERROR_RATE_WARNING:
            sent = await self._alert.send(
                alert_type="elevated_error_rate",
                message=f"Elevated error rate: {total_errors} errors in last 5 minutes.",
                level=AlertLevel.WARNING,
                entity_id="error_rate",
                extra={k: str(v) for k, v in metrics.get("error_counts", {}).items() if v > 0},
            )
            if sent:
                alerts_sent += 1

        # DLQ depth alerts
        dlq_depth = metrics.get("dlq_depth", 0)
        if dlq_depth >= DLQ_CRITICAL:
            sent = await self._alert.send(
                alert_type="dlq_critical",
                message=f"Dead-letter queue has {dlq_depth} failed messages. Delivery pipeline may be degraded.",
                level=AlertLevel.CRITICAL,
                entity_id="dlq",
                extra={"dlq_depth": str(dlq_depth)},
            )
            if sent:
                alerts_sent += 1
        elif dlq_depth >= DLQ_WARNING:
            sent = await self._alert.send(
                alert_type="dlq_warning",
                message=f"Dead-letter queue has {dlq_depth} failed messages.",
                level=AlertLevel.WARNING,
                entity_id="dlq",
                extra={"dlq_depth": str(dlq_depth)},
            )
            if sent:
                alerts_sent += 1

        # JSON parse failures
        json_failures = metrics.get("json_parse_failures", 0)
        if json_failures >= 3:
            sent = await self._alert.send(
                alert_type="json_parse_failures",
                message=f"Claude returned invalid JSON {json_failures} times in last window. Responses may be degraded.",
                level=AlertLevel.WARNING,
                entity_id="json_parse",
                extra={"count": str(json_failures)},
            )
            if sent:
                alerts_sent += 1

        # State machine errors
        sm_errors = metrics.get("state_machine_errors", 0)
        if sm_errors >= 3:
            sent = await self._alert.send(
                alert_type="state_machine_errors",
                message=f"State machine had {sm_errors} transition errors in last window.",
                level=AlertLevel.WARNING,
                entity_id="sm_errors",
                extra={"count": str(sm_errors)},
            )
            if sent:
                alerts_sent += 1

        logger.info(
            "system_health_scan.alerts_evaluated",
            total_errors=total_errors,
            dlq_depth=dlq_depth,
            alerts_sent=alerts_sent,
        )

        return alerts_sent

    # --- Public helpers for other tasks to increment error counters ---

    @staticmethod
    def increment_error(redis_client: redis.Redis, category: str) -> None:
        """Increment a sliding-window error counter.

        Call this from processing_task, gate_tasks, etc. when errors occur.
        Key auto-expires after WINDOW_SECONDS.

        Usage:
            SystemHealthScanner.increment_error(redis_client, "processing_task")
        """
        key = f"monitor:errors:{category}"
        redis_client.incr(key)
        redis_client.expire(key, WINDOW_SECONDS)
