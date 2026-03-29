"""Safety gate base types for the composable pipeline architecture.

Defines the shared data types used across all gates and the pipeline runner.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any


class GateDecision(str, Enum):
    """Outcome of a single gate evaluation."""

    PASS = "PASS"
    BLOCK = "BLOCK"
    ERROR = "ERROR"


@dataclass
class GateResult:
    """Result from a single gate execution."""

    gate_name: str
    decision: GateDecision
    reason: str
    duration_ms: float
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class PipelineResult:
    """Aggregated result from running the full gate pipeline."""

    trace_id: str
    contact_id: str
    message_id: str
    overall_decision: GateDecision
    gate_results: list[GateResult] = field(default_factory=list)
    short_circuited_at: str | None = None
    total_duration_ms: float = 0.0
