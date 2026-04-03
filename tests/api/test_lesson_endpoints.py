"""Tests for lesson admin endpoints -- GET/POST with mocked Neo4j driver.

Covers all 5 admin endpoints:
- GET /admin/lessons/candidates
- GET /admin/lessons/active
- POST /admin/lessons/{id}/approve
- POST /admin/lessons/{id}/reject
- POST /admin/lessons/{id}/promote
"""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from fastapi.testclient import TestClient

from app.main import app


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _mock_driver_with_session(execute_write_return=True):
    """Create a mock driver with properly chained async session."""
    mock_session = AsyncMock()
    mock_session.execute_write.return_value = execute_write_return

    mock_ctx = AsyncMock()
    mock_ctx.__aenter__ = AsyncMock(return_value=mock_session)
    mock_ctx.__aexit__ = AsyncMock(return_value=False)

    mock_driver = MagicMock()
    mock_driver.session.return_value = mock_ctx
    return mock_driver


# ---------------------------------------------------------------------------
# List candidates
# ---------------------------------------------------------------------------

class TestListCandidates:
    def test_get_candidates_returns_200(self):
        """GET /admin/lessons/candidates returns 200 with lessons."""
        mock_driver = AsyncMock()

        with patch("app.routes.lessons.get_driver", new_callable=AsyncMock, return_value=mock_driver), \
             patch("app.routes.lessons.LearningRepository") as MockRepo:
            mock_repo_instance = AsyncMock()
            mock_repo_instance.get_lessons_by_status.return_value = [
                {"id": "L1", "rule": "test rule", "status": "candidate"},
            ]
            MockRepo.return_value = mock_repo_instance

            client = TestClient(app)
            response = client.get("/admin/lessons/candidates")

        assert response.status_code == 200
        body = response.json()
        assert "lessons" in body
        assert body["count"] == 1

    def test_get_candidates_empty(self):
        """GET /admin/lessons/candidates returns 200 with count=0."""
        mock_driver = AsyncMock()

        with patch("app.routes.lessons.get_driver", new_callable=AsyncMock, return_value=mock_driver), \
             patch("app.routes.lessons.LearningRepository") as MockRepo:
            mock_repo_instance = AsyncMock()
            mock_repo_instance.get_lessons_by_status.return_value = []
            MockRepo.return_value = mock_repo_instance

            client = TestClient(app)
            response = client.get("/admin/lessons/candidates")

        assert response.status_code == 200
        assert response.json()["count"] == 0

    def test_get_candidates_error_returns_500(self):
        """GET /admin/lessons/candidates returns 500 on exception."""
        with patch("app.routes.lessons.get_driver", new_callable=AsyncMock, side_effect=Exception("DB down")):
            client = TestClient(app)
            response = client.get("/admin/lessons/candidates")

        assert response.status_code == 500


# ---------------------------------------------------------------------------
# List active
# ---------------------------------------------------------------------------

class TestListActive:
    def test_get_active_returns_approved_and_evergreen(self):
        """GET /admin/lessons/active returns 200 with active lessons."""
        mock_driver = AsyncMock()

        with patch("app.routes.lessons.get_driver", new_callable=AsyncMock, return_value=mock_driver), \
             patch("app.routes.lessons.LearningRepository") as MockRepo:
            mock_repo_instance = AsyncMock()
            mock_repo_instance.get_lessons_by_status.return_value = [
                {"id": "L1", "rule": "rule1", "status": "approved"},
                {"id": "L2", "rule": "rule2", "status": "evergreen"},
            ]
            MockRepo.return_value = mock_repo_instance

            client = TestClient(app)
            response = client.get("/admin/lessons/active")

        assert response.status_code == 200
        assert response.json()["count"] == 2

    def test_get_active_empty(self):
        """GET /admin/lessons/active returns 200 with count=0."""
        mock_driver = AsyncMock()

        with patch("app.routes.lessons.get_driver", new_callable=AsyncMock, return_value=mock_driver), \
             patch("app.routes.lessons.LearningRepository") as MockRepo:
            mock_repo_instance = AsyncMock()
            mock_repo_instance.get_lessons_by_status.return_value = []
            MockRepo.return_value = mock_repo_instance

            client = TestClient(app)
            response = client.get("/admin/lessons/active")

        assert response.status_code == 200
        assert response.json()["count"] == 0


# ---------------------------------------------------------------------------
# Approve lesson
# ---------------------------------------------------------------------------

class TestApproveLessonEndpoint:
    def test_approve_returns_200(self):
        """POST /admin/lessons/{id}/approve returns 200."""
        mock_driver = _mock_driver_with_session(execute_write_return=True)

        with patch("app.routes.lessons.get_driver", new_callable=AsyncMock, return_value=mock_driver):
            client = TestClient(app)
            response = client.post("/admin/lessons/test-id/approve")

        assert response.status_code == 200
        body = response.json()
        assert body["status"] == "approved"
        assert body["lesson_id"] == "test-id"

    def test_approve_nonexistent_returns_404(self):
        """POST /admin/lessons/{id}/approve returns 404 when not found."""
        mock_driver = _mock_driver_with_session(execute_write_return=False)

        with patch("app.routes.lessons.get_driver", new_callable=AsyncMock, return_value=mock_driver):
            client = TestClient(app)
            response = client.post("/admin/lessons/nonexistent/approve")

        assert response.status_code == 404


# ---------------------------------------------------------------------------
# Reject lesson
# ---------------------------------------------------------------------------

class TestRejectLessonEndpoint:
    def test_reject_returns_200(self):
        """POST /admin/lessons/{id}/reject returns 200."""
        mock_driver = _mock_driver_with_session(execute_write_return=True)

        with patch("app.routes.lessons.get_driver", new_callable=AsyncMock, return_value=mock_driver):
            client = TestClient(app)
            response = client.post("/admin/lessons/test-id/reject")

        assert response.status_code == 200
        body = response.json()
        assert body["status"] == "rejected"
        assert body["lesson_id"] == "test-id"

    def test_reject_nonexistent_returns_404(self):
        """POST /admin/lessons/{id}/reject returns 404 when not found."""
        mock_driver = _mock_driver_with_session(execute_write_return=False)

        with patch("app.routes.lessons.get_driver", new_callable=AsyncMock, return_value=mock_driver):
            client = TestClient(app)
            response = client.post("/admin/lessons/nonexistent/reject")

        assert response.status_code == 404


# ---------------------------------------------------------------------------
# Promote lesson
# ---------------------------------------------------------------------------

class TestPromoteLessonEndpoint:
    def test_promote_returns_200(self):
        """POST /admin/lessons/{id}/promote returns 200."""
        mock_driver = AsyncMock()

        with patch("app.routes.lessons.get_driver", new_callable=AsyncMock, return_value=mock_driver), \
             patch("app.routes.lessons.LearningRepository") as MockRepo:
            mock_repo_instance = AsyncMock()
            mock_repo_instance.promote_lesson_to_evergreen.return_value = True
            MockRepo.return_value = mock_repo_instance

            client = TestClient(app)
            response = client.post("/admin/lessons/test-id/promote")

        assert response.status_code == 200
        body = response.json()
        assert body["status"] == "evergreen"
        assert body["lesson_id"] == "test-id"

    def test_promote_nonexistent_returns_404(self):
        """POST /admin/lessons/{id}/promote returns 404 when not found."""
        mock_driver = AsyncMock()

        with patch("app.routes.lessons.get_driver", new_callable=AsyncMock, return_value=mock_driver), \
             patch("app.routes.lessons.LearningRepository") as MockRepo:
            mock_repo_instance = AsyncMock()
            mock_repo_instance.promote_lesson_to_evergreen.return_value = False
            MockRepo.return_value = mock_repo_instance

            client = TestClient(app)
            response = client.post("/admin/lessons/nonexistent/promote")

        assert response.status_code == 404
