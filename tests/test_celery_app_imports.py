"""Import-chain tests for celery_app and observability_signals.

These tests guarantee that the Celery worker entry-point modules
import cleanly. They are the *fastest* way to detect class-definition
failures (e.g. python-statemachine validation errors raised when
ConversationSM is referenced) and signal-handler registration bugs.

REGRESSION CONTEXT
------------------
On 2026-04-05 a state machine fix introduced ``CLOSED.to(RE_ENGAGE)``
which violated python-statemachine's ``final`` constraint. The error
was raised on every ``ConversationSM.from_persisted_state()`` call.
Existing parametrized tests in ``tests/state_machine/`` would have
caught it, but they were never run before the deploy. These import-
level tests are belt-and-suspenders: even a developer who only runs
``pytest -k import`` will see the failure.

These tests are fast (no external services), so they run in the
``test`` job of the CI deploy gate.
"""

from __future__ import annotations


def test_celery_app_imports_cleanly():
    """The celery_app module imports without raising.

    This is the entry point for Celery workers in production. If any
    task module, signal handler, or transitively-imported class
    definition raises during import, the worker process won't start.
    """
    from app import celery_app  # noqa: F401

    assert celery_app.celery_app is not None
    assert celery_app.celery_app.main == "vp_backend"


def test_observability_signals_imports_cleanly():
    """observability_signals registers worker_process_init smoke test.

    Importing this module attaches the smoke test handler that
    instantiates ConversationSM at worker boot. If the module itself
    fails to import, the smoke test never registers and broken SM
    code can ship.
    """
    from app import observability_signals  # noqa: F401

    # The signal handlers should be registered by import side effect.
    # We can't easily assert "registered" without poking at celery
    # internals, but a successful import is what we care about.


def test_ground_truth_check_task_imports():
    """scheduled.ground_truth_check imports and registers with Celery."""
    from app.tasks.scheduled import ground_truth_check

    assert hasattr(ground_truth_check, "ground_truth_check")
    # Celery task name should be set via @celery_app.task(name=...)
    assert ground_truth_check.ground_truth_check.name == "scheduled.ground_truth_check"


def test_ground_truth_tracker_imports():
    """ground_truth_tracker module exposes its public API."""
    from app.services.monitoring import ground_truth_tracker

    assert hasattr(ground_truth_tracker, "record_inbound")
    assert hasattr(ground_truth_tracker, "record_delivery")
    assert hasattr(ground_truth_tracker, "get_window_counts")
    assert hasattr(ground_truth_tracker, "cleanup_old_entries")


def test_conversation_sm_class_instantiates():
    """ConversationSM must instantiate without raising InvalidDefinition.

    This is the failure mode that took down production for 46 hours
    on Apr 5-7 2026. python-statemachine runs graph validation at
    instantiation time, so merely creating an instance catches any
    transition rule violation (final state with outgoing edge,
    unreachable state, etc.).

    Belt-and-suspenders for the worker_process_init smoke test in
    app/observability_signals.py.
    """
    from app.state_machine.conversation_sm import ConversationSM

    # Test every state value including final and externally-managed
    states = (
        "GREETING", "QUALIFYING", "SCHEDULING", "QUALIFIED",
        "NON_RESPONSIVE", "RECOVERY", "FOLLOW_UP", "RE_ENGAGE",
        "HANDOFF", "CLOSED", "BROKER",
    )
    for state in states:
        sm = ConversationSM.from_persisted_state(state, "test_smoke")
        assert sm.model.state == state
