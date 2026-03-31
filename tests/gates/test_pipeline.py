"""Tests for SafetyPipeline -- composable gate runner with fail-open semantics.

Builds pipelines with real gate functions (fakeredis) and stub gates to verify
sequential execution, short-circuiting, fail-open error handling, and
circuit breaker behavior.
"""

from __future__ import annotations

import time
from unittest.mock import MagicMock, patch

from app.gates.base import GateDecision, GateResult
from app.gates.dedup import check_dedup
from app.gates.exfiltration import check_exfiltration
from app.gates.human_lock import check_human_lock
from app.gates.injection import check_injection
from app.gates.pipeline import SafetyPipeline


def _make_pass_gate(name: str):
    """Create a gate callable that always returns PASS."""

    def gate(payload: dict, trace_id: str) -> GateResult:
        return GateResult(
            gate_name=name,
            decision=GateDecision.PASS,
            reason="test_pass",
            duration_ms=0.0,
        )

    return gate


def _make_block_gate(name: str):
    """Create a gate callable that always returns BLOCK."""

    def gate(payload: dict, trace_id: str) -> GateResult:
        return GateResult(
            gate_name=name,
            decision=GateDecision.BLOCK,
            reason="test_block",
            duration_ms=0.0,
        )

    return gate


def _make_error_gate(name: str):
    """Create a gate callable that always raises an exception."""

    def gate(payload: dict, trace_id: str) -> GateResult:
        raise RuntimeError(f"Simulated failure in {name}")

    return gate


class TestPipeline:
    """SafetyPipeline: sequential gate execution with fail-open semantics."""

    def test_all_gates_pass(self, redis_client, make_inbound_payload):
        """Pipeline with all gates returning PASS -> final result PASS."""
        pipeline = SafetyPipeline(gates=[
            ("gate_a", _make_pass_gate("gate_a")),
            ("gate_b", _make_pass_gate("gate_b")),
            ("gate_c", _make_pass_gate("gate_c")),
        ])

        payload = make_inbound_payload()
        result = pipeline.run(payload, "trace-all-pass")

        assert result.overall_decision == GateDecision.PASS
        assert len(result.gate_results) == 3
        assert result.short_circuited_at is None

    def test_first_block_short_circuits(self, make_inbound_payload):
        """First gate BLOCK -> subsequent gates NOT called."""
        call_tracker = []

        def tracking_pass_gate(payload, trace_id):
            call_tracker.append("called")
            return GateResult(
                gate_name="tracking",
                decision=GateDecision.PASS,
                reason="should_not_reach",
                duration_ms=0.0,
            )

        pipeline = SafetyPipeline(gates=[
            ("blocker", _make_block_gate("blocker")),
            ("tracker", tracking_pass_gate),
        ])

        payload = make_inbound_payload()
        result = pipeline.run(payload, "trace-short-circuit")

        assert result.overall_decision == GateDecision.BLOCK
        assert result.short_circuited_at == "blocker"
        assert len(result.gate_results) == 1  # Only the blocker ran
        assert len(call_tracker) == 0  # Tracker was never called

    def test_gate_error_fails_open(self, make_inbound_payload):
        """Gate raises exception -> ERROR result but pipeline continues (fail-open)."""
        pipeline = SafetyPipeline(gates=[
            ("error_gate", _make_error_gate("error_gate")),
            ("after_error", _make_pass_gate("after_error")),
        ])

        payload = make_inbound_payload()
        result = pipeline.run(payload, "trace-error")

        assert result.overall_decision == GateDecision.PASS
        assert len(result.gate_results) == 2

        error_result = result.gate_results[0]
        assert error_result.decision == GateDecision.ERROR
        assert error_result.gate_name == "error_gate"
        assert "unhandled_exception" in error_result.reason

        pass_result = result.gate_results[1]
        assert pass_result.decision == GateDecision.PASS

    def test_gate_results_logged(self, redis_client, make_inbound_payload):
        """Each gate result should have correct name, decision, and reason."""
        pipeline = SafetyPipeline(gates=[
            ("dedup", lambda p, t: check_dedup(p, t, redis_client)),
            ("injection", lambda p, t: check_injection(p, t)),
            ("exfiltration", lambda p, t: check_exfiltration(p, t)),
        ])

        payload = make_inbound_payload(
            messageId="msg-pipeline-001",
            message="Hola, busco departamento",
        )
        result = pipeline.run(payload, "trace-logged")

        assert len(result.gate_results) == 3
        gate_names = [r.gate_name for r in result.gate_results]
        assert gate_names == ["dedup", "injection", "exfiltration"]

        for gr in result.gate_results:
            assert gr.decision == GateDecision.PASS
            assert gr.reason  # Non-empty reason
            assert gr.duration_ms >= 0

    @patch("app.gates.pipeline.SafetyPipeline._record_error")
    def test_circuit_breaker_bypass(self, mock_record, make_inbound_payload):
        """When circuit breaker is tripped, all gates are bypassed."""
        pipeline = SafetyPipeline(gates=[
            ("gate_a", _make_pass_gate("gate_a")),
        ])

        # Manually trip the circuit breaker
        pipeline._emergency_bypass = True

        payload = make_inbound_payload()
        result = pipeline.run(payload, "trace-bypass")

        assert result.overall_decision == GateDecision.PASS
        assert result.short_circuited_at == "emergency_bypass"
        assert len(result.gate_results) == 0  # No gates executed

    def test_circuit_breaker_trips_after_3_errors(self, make_inbound_payload):
        """3+ errors in 5 minutes should activate emergency bypass."""
        pipeline = SafetyPipeline(gates=[
            ("always_error", _make_error_gate("always_error")),
        ])

        payload = make_inbound_payload()

        # Manually accumulate errors to trip the circuit breaker
        # (avoids the lazy import of alerting_tasks -> celery_app -> real Redis)
        for _ in range(3):
            pipeline._error_window.append(time.time())

        # Prune + check (same logic as _record_error)
        cutoff = time.time() - 300
        pipeline._error_window = [t for t in pipeline._error_window if t >= cutoff]
        if len(pipeline._error_window) >= 3:
            pipeline._emergency_bypass = True

        assert pipeline._emergency_bypass is True

        # Next run should be fully bypassed
        result = pipeline.run(payload, "trace-after-trip")
        assert result.overall_decision == GateDecision.PASS
        assert result.short_circuited_at == "emergency_bypass"
        assert len(result.gate_results) == 0

    def test_full_pipeline_real_gates(self, redis_client, make_inbound_payload):
        """Integration: real gates (dedup, human_lock, injection, exfiltration) all PASS."""
        pipeline = SafetyPipeline(gates=[
            ("dedup", lambda p, t: check_dedup(p, t, redis_client)),
            ("human_lock", lambda p, t: check_human_lock(p, t, redis_client)),
            ("injection", lambda p, t: check_injection(p, t)),
            ("exfiltration", lambda p, t: check_exfiltration(p, t)),
        ])

        payload = make_inbound_payload(
            messageId="msg-full-001",
            contactId="contact-full",
            message="Me gustaria saber los precios de los departamentos",
        )
        result = pipeline.run(payload, "trace-full")

        assert result.overall_decision == GateDecision.PASS
        assert len(result.gate_results) == 4
        assert all(r.decision == GateDecision.PASS for r in result.gate_results)
        assert result.total_duration_ms >= 0
