"""SafetyPipeline - composable gate runner with fail-open semantics.

Executes a list of gate callables sequentially. Each gate returns a GateResult.
On unhandled exception, the gate fails OPEN (PASS with ERROR logged).
Circuit breaker trips after 3+ errors in a 5-minute window, bypassing all gates.
"""

from __future__ import annotations

import time
import uuid
from typing import Callable

import structlog

from app.gates.base import GateDecision, GateResult, PipelineResult

logger = structlog.get_logger()


class SafetyPipeline:
    """Runs an ordered list of safety gates against an inbound payload.

    Gates are (name, callable) tuples. Each callable signature:
        def gate(payload: dict, trace_id: str, **kwargs) -> GateResult

    Fail-open: exceptions produce GateDecision.ERROR (not BLOCK).
    Circuit breaker: 3+ errors in 5 minutes triggers emergency bypass.
    """

    def __init__(self, gates: list[tuple[str, Callable]]) -> None:
        self.gates = gates
        self._error_window: list[float] = []
        self._emergency_bypass: bool = False

    def run(self, payload: dict, trace_id: str | None = None) -> PipelineResult:
        """Execute all gates sequentially against the payload."""
        if trace_id is None:
            trace_id = str(uuid.uuid4())

        contact_id = payload.get("contactId", "unknown")
        message_id = payload.get("messageId", "unknown")

        # Emergency bypass check
        if self._emergency_bypass:
            logger.warning(
                "emergency_bypass_active",
                trace_id=trace_id,
                contact_id=contact_id,
                message_id=message_id,
            )
            return PipelineResult(
                trace_id=trace_id,
                contact_id=contact_id,
                message_id=message_id,
                overall_decision=GateDecision.PASS,
                gate_results=[],
                short_circuited_at="emergency_bypass",
                total_duration_ms=0.0,
            )

        pipeline_start = time.perf_counter()
        overall_decision = GateDecision.PASS
        gate_results: list[GateResult] = []
        short_circuited_at: str | None = None

        for gate_name, gate_callable in self.gates:
            gate_start = time.perf_counter()
            try:
                result = gate_callable(payload, trace_id)
            except Exception:
                duration_ms = (time.perf_counter() - gate_start) * 1000
                result = GateResult(
                    gate_name=gate_name,
                    decision=GateDecision.ERROR,
                    reason=f"unhandled_exception",
                    duration_ms=duration_ms,
                )
                logger.error(
                    "gate_exception",
                    gate_name=gate_name,
                    trace_id=trace_id,
                    contact_id=contact_id,
                    duration_ms=round(duration_ms, 2),
                    exc_info=True,
                )
                self._record_error()
            else:
                result.duration_ms = (time.perf_counter() - gate_start) * 1000

            gate_results.append(result)

            logger.info(
                "gate_result",
                gate_name=result.gate_name,
                decision=result.decision.value,
                reason=result.reason,
                duration_ms=round(result.duration_ms, 2),
                trace_id=trace_id,
                contact_id=contact_id,
            )

            if result.decision == GateDecision.BLOCK:
                overall_decision = GateDecision.BLOCK
                short_circuited_at = gate_name
                break
            # ERROR does NOT short-circuit -- fail-open semantics

        total_duration_ms = (time.perf_counter() - pipeline_start) * 1000

        return PipelineResult(
            trace_id=trace_id,
            contact_id=contact_id,
            message_id=message_id,
            overall_decision=overall_decision,
            gate_results=gate_results,
            short_circuited_at=short_circuited_at,
            total_duration_ms=total_duration_ms,
        )

    def _record_error(self) -> None:
        """Track gate errors for circuit breaker logic.

        If 3+ errors occur within a 5-minute window, activate emergency bypass
        and fire a Slack alert.
        """
        now = time.time()
        self._error_window.append(now)

        # Prune entries older than 5 minutes
        cutoff = now - 300
        self._error_window = [t for t in self._error_window if t >= cutoff]

        if len(self._error_window) >= 3:
            self._emergency_bypass = True
            logger.critical(
                "circuit_breaker_tripped",
                error_count=len(self._error_window),
                window_seconds=300,
            )
            # Fire Slack alert (lazy import to avoid circular dependency)
            try:
                from app.tasks.alerting_tasks import send_slack_alert

                send_slack_alert.delay(
                    "CIRCUIT BREAKER TRIPPED: 3+ gate errors in 5 minutes. "
                    "Emergency bypass activated -- all gates bypassed."
                )
            except ImportError:
                logger.warning("alerting_tasks_not_available")
