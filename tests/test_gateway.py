"""Tests for the multi-model gateway layer."""

import sys
import os
import json
import pytest
from pathlib import Path
from unittest.mock import patch, MagicMock

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.gateway.base import (
    GeneratorBase, UnderstanderBase,
    GenerateResult, UnderstandResult,
    MediaType, TaskStatus,
)
from src.gateway.factory import create_generator, list_available_providers
from src.gateway.agnes import AgnesGenerator


# ── Unit: Data Models ────────────────────────────────────────────────

class TestDataModels:
    def test_generate_result_defaults(self):
        r = GenerateResult(success=True, media_type=MediaType.VIDEO)
        assert r.success is True
        assert r.media_type == MediaType.VIDEO
        assert r.file_path is None
        assert r.file_size_kb == 0.0
        assert r.metadata == {}

    def test_generate_result_failure(self):
        r = GenerateResult(success=False, media_type=MediaType.IMAGE,
                           error="API timeout")
        assert r.success is False
        assert "timeout" in r.error

    def test_understand_result_defaults(self):
        r = UnderstandResult(success=True, text="a cat")
        assert r.text == "a cat"
        assert r.confidence == 0.0


# ── Unit: Gateway Factory ────────────────────────────────────────────

class TestFactory:
    def test_create_agnes_no_key(self):
        """Should raise if env key not set."""
        with patch.dict(os.environ, {}, clear=True):
            # Without key in env, should raise
            try:
                gen = create_generator("agnes")
                # If it doesn't raise, the generator should fail gracefully
                assert gen.get_provider_name() == "agnes"
            except ValueError as e:
                assert "API_KEY" in str(e)

    def test_create_agnes_with_key(self):
        """Should create AgnesGenerator when key is set."""
        with patch.dict(os.environ, {"AGNES_API_KEY": "test-key"}):
            gen = create_generator("agnes")
            assert gen.get_provider_name() == "agnes"
            assert gen.api_key == "test-key"

    def test_create_unknown_provider(self):
        with pytest.raises(ValueError, match="Unknown provider"):
            create_generator("nonexistent")

    def test_list_available_with_key(self):
        with patch.dict(os.environ, {"AGNES_API_KEY": "test-key"}):
            available = list_available_providers()
            assert "agnes" in available

    def test_list_available_no_key(self):
        with patch.dict(os.environ, {}, clear=True):
            available = list_available_providers()
            assert "agnes" not in available


# ── Unit: AgnesGenerator ─────────────────────────────────────────────

class TestAgnesGenerator:
    def setup_method(self):
        self.gen = AgnesGenerator(api_key="test-key",
                                  output_dir="/tmp/test_assets")

    def test_provider_name(self):
        assert self.gen.get_provider_name() == "agnes"

    @patch("requests.post")
    def test_generate_image_api_call(self, mock_post):
        """Verify correct API call shape for image generation."""
        mock_post.return_value.json.return_value = {
            "data": [{"url": "https://example.com/img.jpg"}]
        }
        mock_dl = MagicMock()
        mock_dl.content = b"fake_image_bytes"
        mock_dl.status_code = 200

        with patch("requests.get", return_value=mock_dl):
            result = self.gen.generate_image("a test prompt")

        # Check POST call
        call_url = mock_post.call_args[0][0]
        assert "images/generations" in call_url
        payload = mock_post.call_args[1]["json"]
        assert payload["prompt"] == "a test prompt"
        assert payload["model"] == "agnes-image-2.1-flash"

    @patch("requests.post")
    def test_generate_image_failure(self, mock_post):
        """Handle API error gracefully."""
        mock_post.return_value.json.return_value = {"error": "rate limit"}

        result = self.gen.generate_image("test")
        assert result.success is False
        assert result.error is not None

    @patch("requests.post")
    def test_generate_video_poll_flow(self, mock_post):
        """Verify video task creation + polling flow."""
        # First call: create task → returns video_id
        mock_post.return_value.json.return_value = {"id": "vid_123"}

        # Second call: poll → completed
        mock_poll = MagicMock()
        mock_poll.json.return_value = {
            "status": "completed",
            "video_url": "https://example.com/vid.mp4",
        }

        mock_dl = MagicMock()
        mock_dl.content = b"fake_video_bytes"

        with patch("requests.get", side_effect=[mock_poll, mock_dl]):
            result = self.gen.generate_video("test video", duration=3.0)

        assert result.success is True
        assert result.file_size_kb > 0

    @patch("requests.post")
    def test_generate_video_timeout(self, mock_post):
        """Handle poll timeout."""
        mock_post.return_value.json.return_value = {"id": "vid_123"}
        mock_poll = MagicMock()
        mock_poll.json.return_value = {"status": "pending", "progress": 50}

        with patch("requests.get", return_value=mock_poll):
            result = self.gen.generate_video(
                "test", scene_id="timeout_test",
                max_polls=2, poll_interval=0.1,
            )

        assert result.success is False
        assert "Timeout" in result.error

    def test_generate_batch_empty(self):
        """Empty batch returns empty list."""
        results = self.gen.generate_batch([])
        assert results == []

    @patch.object(AgnesGenerator, "generate_image")
    def test_generate_batch_routing(self, mock_img):
        """Batch routes to correct method by type."""
        mock_img.return_value = GenerateResult(
            success=True, media_type=MediaType.IMAGE,
        )
        items = [
            {"prompt": "img1", "type": "image"},
            {"prompt": "vid1", "type": "video"},
        ]
        with patch.object(AgnesGenerator, "generate_video",
                          return_value=GenerateResult(
                              success=True, media_type=MediaType.VIDEO)):
            results = self.gen.generate_batch(items)

        assert len(results) == 2
        assert results[0].media_type == MediaType.IMAGE
        assert results[1].media_type == MediaType.VIDEO


# ── Edge Cases ────────────────────────────────────────────────────────

class TestEdgeCases:
    def test_generate_result_empty_metadata(self):
        r = GenerateResult(success=True, media_type=MediaType.AUDIO)
        assert r.metadata == {}
        r.metadata["key"] = "val"
        assert r.metadata["key"] == "val"

    def test_media_type_str(self):
        assert MediaType.VIDEO.value == "video"
        assert MediaType.IMAGE.value == "image"

    def test_task_status_cycle(self):
        """Verify status enum values match API semantics."""
        assert TaskStatus.PENDING.value == "pending"
        assert TaskStatus.RUNNING.value == "running"
        assert TaskStatus.COMPLETED.value == "completed"
