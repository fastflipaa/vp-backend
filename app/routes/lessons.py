"""Lesson lifecycle admin endpoints.

Provides REST API for managing LessonLearned nodes:
- List candidates for review
- List active (approved + evergreen) lessons
- Approve, reject, and promote individual lessons

All endpoints are under /admin/lessons prefix.
No auth middleware yet (internal-only API behind VPN).
"""

from __future__ import annotations

import structlog
from fastapi import APIRouter, HTTPException

from app.repositories.base import get_driver
from app.repositories.learning_repository import LearningRepository

logger = structlog.get_logger()

router = APIRouter(prefix="/admin/lessons", tags=["lessons"])


@router.get("/candidates")
async def list_candidates():
    """List candidate lessons awaiting review."""
    try:
        driver = await get_driver()
        repo = LearningRepository(driver)
        lessons = await repo.get_lessons_by_status(["candidate"], limit=50)
        return {"lessons": lessons, "count": len(lessons)}
    except Exception as exc:
        logger.exception("lessons.list_candidates_failed")
        raise HTTPException(status_code=500, detail=str(exc))


@router.get("/active")
async def list_active():
    """List active lessons (approved + evergreen)."""
    try:
        driver = await get_driver()
        repo = LearningRepository(driver)
        lessons = await repo.get_lessons_by_status(
            ["approved", "evergreen"], limit=50
        )
        return {"lessons": lessons, "count": len(lessons)}
    except Exception as exc:
        logger.exception("lessons.list_active_failed")
        raise HTTPException(status_code=500, detail=str(exc))


@router.post("/{lesson_id}/approve")
async def approve_lesson(lesson_id: str):
    """Approve a candidate lesson (sets status=approved, confidence=0.8)."""
    try:
        driver = await get_driver()
        async with driver.session() as session:
            result = await session.execute_write(
                _approve_lesson_tx, lesson_id
            )
        if not result:
            raise HTTPException(status_code=404, detail="Lesson not found")
        logger.info("lessons.approved", lesson_id=lesson_id)
        return {"status": "approved", "lesson_id": lesson_id}
    except HTTPException:
        raise
    except Exception as exc:
        logger.exception("lessons.approve_failed", lesson_id=lesson_id)
        raise HTTPException(status_code=500, detail=str(exc))


@router.post("/{lesson_id}/reject")
async def reject_lesson(lesson_id: str):
    """Reject a candidate lesson (sets status=rejected, confidence=0.0)."""
    try:
        driver = await get_driver()
        async with driver.session() as session:
            result = await session.execute_write(
                _reject_lesson_tx, lesson_id
            )
        if not result:
            raise HTTPException(status_code=404, detail="Lesson not found")
        logger.info("lessons.rejected", lesson_id=lesson_id)
        return {"status": "rejected", "lesson_id": lesson_id}
    except HTTPException:
        raise
    except Exception as exc:
        logger.exception("lessons.reject_failed", lesson_id=lesson_id)
        raise HTTPException(status_code=500, detail=str(exc))


@router.post("/{lesson_id}/promote")
async def promote_lesson(lesson_id: str):
    """Promote a lesson to evergreen status."""
    try:
        driver = await get_driver()
        repo = LearningRepository(driver)
        promoted = await repo.promote_lesson_to_evergreen(lesson_id)
        if not promoted:
            raise HTTPException(status_code=404, detail="Lesson not found")
        logger.info("lessons.promoted", lesson_id=lesson_id)
        return {"status": "evergreen", "lesson_id": lesson_id}
    except HTTPException:
        raise
    except Exception as exc:
        logger.exception("lessons.promote_failed", lesson_id=lesson_id)
        raise HTTPException(status_code=500, detail=str(exc))


# -- Static transaction functions --


async def _approve_lesson_tx(tx, lesson_id: str) -> bool:
    result = await tx.run(
        """
        MATCH (ll:LessonLearned {id: $lid})
        SET ll.status = 'approved', ll.confidence = 0.8
        RETURN ll.id AS id
        """,
        lid=lesson_id,
    )
    record = await result.single()
    return record is not None


async def _reject_lesson_tx(tx, lesson_id: str) -> bool:
    result = await tx.run(
        """
        MATCH (ll:LessonLearned {id: $lid})
        SET ll.status = 'rejected', ll.confidence = 0.0
        RETURN ll.id AS id
        """,
        lid=lesson_id,
    )
    record = await result.single()
    return record is not None
