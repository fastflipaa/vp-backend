"""Shadow comparison: Python gate decisions vs n8n gate decisions.

Compares each gate's Python decision against the corresponding n8n decision,
tracking agreements, disagreements, and known improvements (intentional
Python-over-n8n upgrades).
"""

from __future__ import annotations

import structlog

from app.gates.base import GateDecision, PipelineResult

logger = structlog.get_logger()


def compare_results(
    python_result: PipelineResult,
    n8n_decisions: dict[str, str],
) -> dict:
    """Compare Python pipeline results against n8n gate decisions.

    Args:
        python_result: The full PipelineResult from SafetyPipeline.run().
        n8n_decisions: Dict of gate_name -> decision string from n8n payload.

    Returns:
        Comparison dict with agreement_rate, disagreements, comparisons list.
    """
    comparisons: list[dict] = []
    agreements = 0
    disagreements = 0
    known_improvements = 0

    for gate_result in python_result.gate_results:
        gate_name = gate_result.gate_name
        python_decision = gate_result.decision.value
        n8n_decision = n8n_decisions.get(gate_name, "UNKNOWN")

        if python_decision == n8n_decision:
            match = True
            agreements += 1
            is_known_improvement = False
        else:
            match = False
            is_known_improvement = _is_known_improvement(
                gate_name, python_decision, n8n_decision
            )
            if is_known_improvement:
                known_improvements += 1
            else:
                disagreements += 1

        comparisons.append({
            "gate": gate_name,
            "python": python_decision,
            "n8n": n8n_decision,
            "match": match,
            "known_improvement": is_known_improvement,
        })

    total_gates = len(python_result.gate_results)
    agreement_rate = agreements / total_gates if total_gates > 0 else 0.0

    comparison = {
        "trace_id": python_result.trace_id,
        "contact_id": python_result.contact_id,
        "message_id": python_result.message_id,
        "total_gates": total_gates,
        "agreements": agreements,
        "disagreements": disagreements,
        "known_improvements": known_improvements,
        "agreement_rate": agreement_rate,
        "comparisons": comparisons,
    }

    logger.info(
        "shadow_comparison_complete",
        trace_id=python_result.trace_id,
        agreement_rate=round(agreement_rate, 4),
        agreements=agreements,
        disagreements=disagreements,
        known_improvements=known_improvements,
    )

    return comparison


def _is_known_improvement(
    gate_name: str,
    python_decision: str,
    n8n_decision: str,
) -> bool:
    """Determine if a disagreement is a known Python improvement over n8n.

    Known improvements:
    - rate_limit: Any disagreement -- Python uses sliding window vs n8n fixed window.
    - injection: Python BLOCK + n8n PASS -- Python has OWASP 2026 patterns n8n lacks.

    All other disagreements are genuine and need investigation.
    """
    # Rate limit: sliding window (Python) vs fixed window (n8n) -- any difference is expected
    if gate_name == "rate_limit":
        return True

    # Injection: Python blocks what n8n misses (OWASP 2026 additions)
    if gate_name == "injection" and python_decision == "BLOCK" and n8n_decision == "PASS":
        return True

    return False
