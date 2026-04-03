"""Vive Polanco Backend - Celery application instance.

Configures Celery with Redis broker (db0) and result backend (db1).
Uses late ack, prefetch=1, and crash-recovery settings.
Includes Celery Beat schedule for daily shadow summary report.
"""

from celery import Celery
from celery.schedules import crontab

from app.config import settings

celery_app = Celery(
    "vp_backend",
    broker=settings.redis_broker_url,
    backend=settings.redis_result_url,
)

# Register tasks explicitly (autodiscover expects tasks.py, our files have custom names)
import app.tasks.test_task  # noqa: F401, E402
import app.tasks.gate_tasks  # noqa: F401, E402
import app.tasks.alerting_tasks  # noqa: F401, E402
import app.tasks.reporting_tasks  # noqa: F401, E402
import app.tasks.processing_task  # noqa: F401, E402
import app.tasks.scheduled.health_check  # noqa: F401, E402
import app.tasks.scheduled.morning_drain  # noqa: F401, E402
import app.tasks.scheduled.stale_reengagement  # noqa: F401, E402
import app.tasks.scheduled.followup_sequence  # noqa: F401, E402
import app.tasks.handoff_notification_task  # noqa: F401, E402
import app.tasks.pipeline_sync_task  # noqa: F401, E402
import app.tasks.doc_delivery_task  # noqa: F401, E402
import app.tasks.scheduled.conversation_quality_scan  # noqa: F401, E402
import app.tasks.scheduled.system_health_scan  # noqa: F401, E402
import app.tasks.scheduled.lead_operator  # noqa: F401, E402

celery_app.conf.update(
    # Serialization
    task_serializer="json",
    accept_content=["json"],
    result_serializer="json",
    # Timezone
    timezone="America/Mexico_City",
    enable_utc=True,
    # Results
    result_expires=3600,  # 1 hour
    # Worker settings
    worker_prefetch_multiplier=1,
    worker_max_tasks_per_child=100,
    worker_hijack_root_logger=False,
    # Broker settings
    broker_connection_retry_on_startup=True,
    broker_transport_options={"visibility_timeout": 3600},
    # Reliability: late ack + reject on worker lost
    task_acks_late=True,
    task_reject_on_worker_lost=True,
    # Queue routing: separate gate and processing workers for independent scaling
    task_routes={
        "gates.*": {"queue": "celery"},
        "processing.*": {"queue": "processing"},
        "reporting.*": {"queue": "celery"},
        "alerting.*": {"queue": "celery"},
        "scheduled.*": {"queue": "celery"},
        "notifications.*": {"queue": "celery"},
        "pipeline.*": {"queue": "celery"},
        "documents.*": {"queue": "processing"},
    },
)

# Celery Beat schedule -- runs on vp-beat container
celery_app.conf.beat_schedule = {
    "daily-shadow-summary": {
        "task": "reporting.daily_shadow_summary",
        "schedule": crontab(hour=8, minute=0),  # 8:00 AM CDMX (timezone already set to America/Mexico_City)
    },
    "health-check-5min": {
        "task": "scheduled.health_check",
        "schedule": crontab(minute="*/5"),
    },
    "morning-queue-drain": {
        "task": "scheduled.morning_drain",
        "schedule": crontab(hour=9, minute=1),
    },
    "stale-lead-reengagement": {
        "task": "scheduled.stale_reengagement",
        "schedule": crontab(hour=10, minute=0),
    },
    "followup-sequence-hourly": {
        "task": "scheduled.followup_check",
        "schedule": crontab(minute=0),
    },
    "conversation-quality-scan-15min": {
        "task": "scheduled.conversation_quality_scan",
        "schedule": crontab(minute="*/15"),
    },
    "system-health-scan-5min": {
        "task": "scheduled.system_health_scan",
        "schedule": crontab(minute="*/5"),
    },
    "lead-operator-sweep-3h": {
        "task": "scheduled.lead_operator_sweep",
        "schedule": crontab(minute=30, hour="*/3"),  # Every 3 hours at :30
    },
}
