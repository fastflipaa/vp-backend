#!/usr/bin/env python
"""Celery worker liveness probe using active_queues check.

Detects the 'catatonic worker' bug where ping returns OK but the worker
has lost its queue consumer registration after a Redis restart.

Exit code 0 = healthy, 1 = unhealthy.
"""

import sys
from app.celery_app import celery_app


def check_worker_health() -> bool:
    """Check if worker is alive AND consuming queues."""
    try:
        inspector = celery_app.control.inspect(timeout=5.0)

        # Stage 1: Ping check
        ping_result = inspector.ping()
        if not ping_result:
            print("UNHEALTHY: Worker did not respond to ping")
            return False

        # Stage 2: Active queues check (detects catatonic state)
        queues = inspector.active_queues()
        if not queues:
            print("UNHEALTHY: No active queue consumers (catatonic state)")
            return False

        print(f"HEALTHY: {len(queues)} worker(s) with active queues")
        return True

    except Exception as e:
        print(f"UNHEALTHY: Health check error: {e}")
        return False


if __name__ == "__main__":
    sys.exit(0 if check_worker_health() else 1)
