"""Canary mode monitoring endpoint.

Provides /canary/status for observing canary routing stats:
processed count, error rate, average latency, shadow count.
"""

from __future__ import annotations

import structlog
from fastapi import APIRouter
from fastapi.responses import JSONResponse

from app.config import settings
from app.dependencies import get_sync_redis
from app.services.canary_router import get_canary_stats

logger = structlog.get_logger()

router = APIRouter(prefix="/canary", tags=["canary"])


@router.get("/status")
async def canary_status() -> JSONResponse:
    """Return canary routing statistics.

    Reports:
    - canary_enabled: whether canary mode is active
    - canary_tag: the GHL tag used for canary routing
    - canary_processed_24h: messages processed in canary mode (24h)
    - canary_errors_24h: canary processing errors (24h)
    - canary_error_rate: errors / processed
    - canary_avg_latency_ms: average gate pipeline latency
    - shadow_count_24h: messages processed in shadow mode (24h)
    """
    try:
        redis_client = get_sync_redis()
        stats = get_canary_stats(redis_client)
    except Exception:
        logger.exception("canary_status_redis_error")
        stats = {
            "processed_24h": 0,
            "errors_24h": 0,
            "error_rate": 0.0,
            "avg_latency_ms": 0.0,
            "shadow_count_24h": 0,
        }

    return JSONResponse(
        status_code=200,
        content={
            "canary_enabled": settings.CANARY_ENABLED,
            "canary_tag": settings.CANARY_TAG,
            "canary_processed_24h": stats["processed_24h"],
            "canary_errors_24h": stats["errors_24h"],
            "canary_error_rate": stats["error_rate"],
            "canary_avg_latency_ms": stats["avg_latency_ms"],
            "shadow_count_24h": stats["shadow_count_24h"],
        },
    )
