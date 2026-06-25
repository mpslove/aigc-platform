"""Tests for multi-modal understanding module."""
import sys, os, json
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import pytest
from src.understanding.qwen_engine import QwenVLEngine, _check_gpu


class TestQwenVLEngine:
    def test_default_init(self):
        engine = QwenVLEngine(use_api=True)
        assert engine is not None
        assert engine.available is True
        assert engine.model_size == "2B"

    def test_describe_image_mock(self):
        engine = QwenVLEngine(use_api=True)
        desc = engine.describe_image("nonexistent.jpg")
        assert isinstance(desc, str)
        assert len(desc) > 0

    def test_answer_question_mock(self):
        engine = QwenVLEngine(use_api=True)
        ans = engine.answer_question("test.jpg", "What is in this image?")
        assert isinstance(ans, str)

    def test_analyze_scene_mock(self):
        engine = QwenVLEngine(use_api=True)
        result = engine.analyze_scene("test.jpg")
        assert isinstance(result, dict)
        # Should have at least some keys
        assert len(result) > 0

    def test_extract_json(self):
        text = '{"key": "value"}'
        assert QwenVLEngine._extract_json(text) == text

    def test_extract_json_from_code_block(self):
        text = "```json\n{\"key\": \"value\"}\n```"
        assert QwenVLEngine._extract_json(text) == '{"key": "value"}'

    def test_gpu_check(self):
        info = _check_gpu()
        assert "cuda_available" in info
        assert "device" in info
        assert "dtype" in info
        assert "optimizations" in info

    def test_mock_response_has_description(self):
        engine = QwenVLEngine(use_api=True)
        messages = [
            {"role": "user", "content": [
                {"type": "image", "image": "test.jpg"},
                {"type": "text", "text": "Describe this image in detail..."},
            ]}
        ]
        resp = engine._mock_response(messages)
        assert "image" in resp.lower() or "scene" in resp.lower() or "subject" in resp.lower()

    def test_mock_response_has_json(self):
        engine = QwenVLEngine(use_api=True)
        messages = [
            {"role": "user", "content": [
                {"type": "image", "image": "test.jpg"},
                {"type": "text", "text": "Return JSON with keys: objects, scene_type"},
            ]}
        ]
        resp = engine._mock_response(messages)
        try:
            data = json.loads(resp)
            assert "objects" in data or "scene_type" in data
        except json.JSONDecodeError:
            pytest.fail(f"Expected JSON response, got: {resp}")
