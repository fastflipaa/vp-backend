"""Vive Polanco Backend - Celery application instance.

Configures Celery with Redis broker (db0) and result backend (db1).
Uses late ack, prefetch=1, and crash-recovery settings.
"""

from celery import Celery
from app.config import settings

celery_app = Celery(
    "vp_backend",
    broker=settings.redis_broker_url,
    backend=settings.redis_result_url,
)

# Autodiscover tasks in app.tasks package
celery_app.autodiscover_tasks(["app.tasks"])

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
)
