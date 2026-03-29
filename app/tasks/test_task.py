"""Minimal echo task for Phase 14 verification."""

from app.celery_app import celery_app


@celery_app.task(name="test.echo")
def echo_task(message: str) -> dict:
    """Echo the input message back with a status field."""
    return {"echo": message, "status": "processed"}
