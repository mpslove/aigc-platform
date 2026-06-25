"""Tests for multi-modal API endpoints."""
import sys, os, json
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from fastapi.testclient import TestClient
from src.api.app import app

client = TestClient(app)


class TestUnderstandingAPI:
    def test_root_has_endpoints(self):
        resp = client.get("/")
        assert resp.status_code == 200
        data = resp.json()
        assert "endpoints" in data

    def test_templates(self):
        resp = client.get("/api/templates")
        assert resp.status_code == 200
        data = resp.json()
        assert "templates" in data

    def test_understand_image_not_found(self):
        resp = client.post("/api/understand", json={"image_path": "nonexistent.jpg"})
        assert resp.status_code == 404

    def test_visual_qa_image_not_found(self):
        resp = client.post("/api/visual-qa", json={
            "image_path": "nonexistent.jpg",
            "question": "What is this?",
        })
        assert resp.status_code == 404

    def test_rag_search_empty(self):
        resp = client.post("/api/rag/search", json={
            "query": "cat",
            "top_k": 5,
        })
        assert resp.status_code == 200
        data = resp.json()
        assert "results" in data

    def test_rag_stats(self):
        resp = client.get("/api/rag/stats")
        assert resp.status_code == 200
        data = resp.json()
        assert "total_items" in data
        assert "dimension" in data

    def test_rag_index(self):
        resp = client.post("/api/rag/index", json={"directory": "./assets"})
        assert resp.status_code == 200
        data = resp.json()
        assert data["status"] == "indexed"
