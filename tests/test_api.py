"""Tests for the FastAPI REST API."""

import sys
import os
import json
import pytest
from pathlib import Path
from unittest.mock import patch, MagicMock

sys.path.insert(0, str(Path(__file__).parent.parent))

from fastapi.testclient import TestClient
from src.api.app import app


client = TestClient(app)


class TestAPI:
    def test_root(self):
        r = client.get("/")
        assert r.status_code == 200
        data = r.json()
        assert data["name"] == "AIGC Platform API"
        assert "endpoints" in data

    def test_list_templates(self):
        r = client.get("/api/templates")
        assert r.status_code == 200
        data = r.json()
        assert "templates" in data
        assert "nursing-ad" in data["templates"]

    @patch("src.api.app.get_agent")
    def test_create_project(self, mock_get_agent):
        """Verify project creation endpoint."""
        from src.pipeline.schema import VideoProject, Scene, SceneType
        mock_agent = MagicMock()
        mock_agent.from_template.return_value = VideoProject(
            title="Test",
            scenes=[
                Scene(id="s1", prompt="test", scene_type=SceneType.TITLE,
                      duration=3.0),
                Scene(id="s2", prompt="test2", duration=5.0),
            ],
        )
        mock_get_agent.return_value = mock_agent

        r = client.post("/api/project",
                        json={"source": "template", "value": "nursing-ad"})
        assert r.status_code == 200
        data = r.json()
        assert data["status"] == "created"
        assert data["scenes"] == 2

    def test_create_project_unknown_source(self):
        r = client.post("/api/project",
                        json={"source": "invalid", "value": "test"})
        assert r.status_code == 400

    def test_generate_no_project(self):
        """Generate without a project should 400."""
        with patch("src.api.app.get_agent") as mock_get_agent:
            mock_agent = MagicMock()
            mock_agent.generate.side_effect = ValueError("No project")
            mock_get_agent.return_value = mock_agent

            r = client.post("/api/generate")
            assert r.status_code == 400

    @patch("src.api.app.get_agent")
    def test_generate_success(self, mock_get_agent):
        mock_agent = MagicMock()
        mock_agent.generate.return_value = {
            "total": 6, "success": 6, "failed": 0,
            "elapsed_s": 120.0, "results": {},
        }
        mock_get_agent.return_value = mock_agent
        # Need a current_project for generate to not raise
        from src.pipeline.schema import VideoProject
        mock_agent.current_project = VideoProject(title="T", scenes=[])

        r = client.post("/api/generate")
        assert r.status_code == 200
        data = r.json()
        assert data["status"] == "completed"
        assert data["success"] == 6

    def test_evaluate_missing_video(self):
        r = client.post("/api/evaluate",
                        json={"video_path": "/nonexistent.mp4"})
        assert r.status_code == 404

    @patch("src.api.app.get_agent")
    def test_evaluate_success(self, mock_get_agent):
        mock_agent = MagicMock()
        mock_agent.evaluate.return_value = {
            "quality_score": 55.0,
            "duration_s": 26.0,
            "resolution": (1152, 768),
            "file_size_kb": 3200.0,
            "has_audio": True,
            "errors": [],
        }
        mock_get_agent.return_value = mock_agent

        with patch("os.path.exists", return_value=True):
            r = client.post("/api/evaluate",
                            json={"video_path": "/tmp/test.mp4"})
        assert r.status_code == 200
        data = r.json()
        assert data["quality_score"] == 55.0
        assert data["resolution"] == [1152, 768]

    def test_agent_log(self):
        r = client.get("/api/agent/log")
        assert r.status_code == 200
        assert "log" in r.json()

    def test_agent_summary(self):
        r = client.get("/api/agent/summary")
        assert r.status_code == 200
        assert "summary" in r.json()
