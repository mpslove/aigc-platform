"""Tests for the AIGC video pipeline module."""

import sys
import os
import json
import pytest
from pathlib import Path
from unittest.mock import patch, MagicMock, PropertyMock

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.pipeline.schema import (
    Scene, SceneType, GeneratorType, TextOverlay, VideoProject, Transition,
)
from src.pipeline.script_writer import ScriptWriter
from src.pipeline.generator import AssetProducer
from src.pipeline.composer import Composer
from src.gateway.base import GenerateResult, MediaType


# ── Data Model Tests ─────────────────────────────────────────────────

class TestSchema:
    def test_scene_defaults(self):
        s = Scene(id="s1", prompt="test")
        assert s.scene_type == SceneType.VIDEO
        assert s.duration == 4.0
        assert s.transition == Transition.CUT
        assert s.num_frames == 64  # min(64, 4*16=64) capped

    def test_scene_explicit_frames(self):
        s = Scene(id="s1", prompt="test", num_frames=96)
        assert s.num_frames == 96
        assert s.duration == 4.0

    def test_video_project_summary(self):
        scenes = [
            Scene(id="t", prompt="title", scene_type=SceneType.TITLE, duration=3.0),
            Scene(id="v", prompt="video", duration=5.0),
        ]
        p = VideoProject(title="Test", scenes=scenes)
        summary = p.summary()
        assert summary["scenes"] == 2
        assert summary["duration_s"] == 8.0
        assert summary["script_source"] == "manual"

    def test_scene_count_types(self):
        scenes = [
            Scene(id="t", prompt="t", scene_type=SceneType.TITLE),
            Scene(id="v1", prompt="v1"),
            Scene(id="v2", prompt="v2"),
            Scene(id="e", prompt="e", scene_type=SceneType.END),
        ]
        p = VideoProject(title="T", scenes=scenes)
        counts = p.scene_count()
        assert counts["title"] == 1
        assert counts["video"] == 2
        assert counts["end_card"] == 1

    def test_text_overlay_defaults(self):
        t = TextOverlay(text="hello")
        assert t.font_size == 36
        assert t.box is True
        assert t.x == "(w-text_w)/2"

    def test_transition_enum_values(self):
        assert Transition.CUT.value == "cut"
        assert Transition.FADE.value == "fade"
        assert Transition.CROSSFADE.value == "crossfade"


# ── ScriptWriter Tests ───────────────────────────────────────────────

class TestScriptWriter:
    def setup_method(self):
        self.writer = ScriptWriter()

    def test_list_templates(self):
        templates = self.writer.list_templates()
        assert "nursing-ad" in templates
        assert "product-ad" in templates
        assert "travel-ad" in templates

    def test_from_template_nursing(self):
        project = self.writer.from_template("nursing-ad")
        assert project.title == "Nursing Care Advertisement"
        assert len(project.scenes) == 6
        assert project.scenes[0].scene_type == SceneType.TITLE
        assert project.scenes[-1].scene_type == SceneType.END

    def test_from_template_unknown(self):
        with pytest.raises(ValueError, match="Unknown template"):
            self.writer.from_template("nonexistent")

    def test_from_prompt_nursing_keyword(self):
        p = self.writer.from_prompt("护理广告")
        assert "Nursing" in p.title

    def test_from_prompt_generic(self):
        p = self.writer.from_prompt("something random")
        # Falls back to nursing
        assert p.title is not None

    def test_to_dict_and_from_json(self, tmp_path):
        project = self.writer.from_template("travel-ad")
        json_path = tmp_path / "test_script.json"
        self.writer.save_script(project, str(json_path))
        assert json_path.exists()

        loaded = ScriptWriter.from_json(str(json_path))
        assert loaded.title == project.title
        assert len(loaded.scenes) == len(project.scenes)

    def test_project_scene_properties(self):
        project = self.writer.from_template("product-ad")
        for scene in project.scenes:
            assert scene.id
            assert scene.prompt
            assert scene.duration > 0
            assert scene.width > 0


# ── AssetProducer Tests ──────────────────────────────────────────────

class TestAssetProducer:
    def setup_method(self):
        self.mock_gen = MagicMock()
        self.mock_gen.get_provider_name.return_value = "agnes"
        self.producer = AssetProducer(self.mock_gen, "/tmp/test_assets")

    def test_produce_scene_video(self):
        """Verify video scene calls generate_video on generator."""
        self.mock_gen.generate_video.return_value = GenerateResult(
            success=True, media_type=MediaType.VIDEO,
            file_path="/tmp/test_assets/vid1.mp4",
        )
        scene = Scene(id="vid1", prompt="test video")
        result = self.producer.produce_scene(scene)
        assert result.success
        assert scene.asset_path == "/tmp/test_assets/vid1.mp4"
        self.mock_gen.generate_video.assert_called_once()

    def test_produce_scene_image(self):
        """Verify image scene calls generate_image."""
        self.mock_gen.generate_image.return_value = GenerateResult(
            success=True, media_type=MediaType.IMAGE,
            file_path="/tmp/test_assets/title_img.jpg",
        )
        scene = Scene(id="title", prompt="test title",
                      scene_type=SceneType.TITLE)
        result = self.producer.produce_scene(scene)
        assert result.success
        self.mock_gen.generate_image.assert_called_once()

    def test_produce_scene_failure(self):
        """Handle generation failure gracefully."""
        self.mock_gen.generate_video.return_value = GenerateResult(
            success=False, media_type=MediaType.VIDEO,
            error="API rate limit",
        )
        scene = Scene(id="fail1", prompt="fail")
        result = self.producer.produce_scene(scene)
        assert result.success is False
        assert "rate limit" in result.error

    def test_produce_all_tracks_results(self):
        """Verify produce_all returns results for all scenes."""
        self.mock_gen.generate_video.return_value = GenerateResult(
            success=True, media_type=MediaType.VIDEO,
            file_path="/tmp/test_assets/v.mp4",
        )
        self.mock_gen.generate_image.return_value = GenerateResult(
            success=True, media_type=MediaType.IMAGE,
            file_path="/tmp/test_assets/t.jpg",
        )

        project = VideoProject(
            title="Test",
            scenes=[
                Scene(id="t1", prompt="t1", scene_type=SceneType.TITLE),
                Scene(id="v1", prompt="v1"),
            ],
        )
        results = self.producer.produce_all(project)
        assert len(results) == 2
        assert all(r.success for r in results.values())

    def test_progress_save_and_load(self, tmp_path):
        """Verify progress file saves and restores state."""
        self.mock_gen.generate_video.return_value = GenerateResult(
            success=True, media_type=MediaType.VIDEO,
            file_path="/tmp/test_already_done.mp4",
        )

        progress = tmp_path / ".progress.json"
        producer = AssetProducer(self.mock_gen, str(tmp_path),
                                 progress_file=str(progress))

        scene = Scene(id="s_done", prompt="done")
        producer.produce_scene(scene)

        # New producer should detect existing asset
        with patch("os.path.exists", return_value=True):
            producer2 = AssetProducer(self.mock_gen, str(tmp_path),
                                      progress_file=str(progress))
            project = VideoProject(title="Resume", scenes=[scene])
            producer2._load_progress(project)
            assert project.scenes[0].asset_path is not None


# ── Composer Tests (unit, no FFmpeg) ─────────────────────────────────

class TestComposer:
    def setup_method(self):
        self.composer = Composer("/tmp/test_output")

    def test_build_drawtext_filter(self):
        """Verify drawtext filter string construction."""
        from src.pipeline.schema import TextOverlay
        overlays = [
            TextOverlay(text="Hello", font_size=36, y="100"),
        ]
        vf = self.composer.build_drawtext_filter(overlays, 10.0)
        assert vf is not None
        assert "drawtext=text='Hello'" in vf
        assert "fontsize=36" in vf
        assert "enable='between(t,0.0,10.0)" in vf  # float conversion

    def test_build_drawtext_no_box(self):
        from src.pipeline.schema import TextOverlay
        t = TextOverlay(text="no box", box=False, box_border_w=0)
        vf = self.composer.build_drawtext_filter([t], 5.0)
        assert ":box=1" not in vf

    def test_build_drawtext_empty(self):
        assert self.composer.build_drawtext_filter([], 10.0) is None

    def test_build_drawtext_multiple(self):
        from src.pipeline.schema import TextOverlay
        overlays = [
            TextOverlay(text="Line1", font_size=36, y="100"),
            TextOverlay(text="Line2", font_size=28, y="200",
                        start_time=2.0, end_time=8.0),
        ]
        vf = self.composer.build_drawtext_filter(overlays, 10.0)
        assert vf is not None
        assert "Line1" in vf
        assert "Line2" in vf
        # Comma-separated filters
        assert "," in vf or vf.count("drawtext") == 2

    def test_get_transition_filter(self):
        cut = self.composer._get_transition_filter(Transition.CUT, 0.5)
        assert cut == ""  # cut has no filter
        fade = self.composer._get_transition_filter(Transition.FADE, 1.0)
        assert "fade=t=out" in fade

    @patch("subprocess.run")
    def test_verify_compatibility(self, mock_run):
        """Verify compatibility check handles metadata."""
        mock_run.return_value = MagicMock(
            returncode=0,
            stderr="Stream #0:0: Video: h264, 1152x768, 24 fps",
        )
        with patch("os.path.getsize", return_value=102400):
            result = Composer.verify_compatibility(["a.mp4", "b.mp4"])
        assert result["segments"] == 2


# ── Edge Cases ───────────────────────────────────────────────────────

class TestPipelineEdgeCases:
    def test_empty_project(self):
        p = VideoProject(title="Empty")
        assert p.total_duration() == 0.0
        assert p.scene_count() == {}
        assert p.summary()["scenes"] == 0

    def test_single_scene_project(self):
        s = Scene(id="only", prompt="only one")
        p = VideoProject(title="Single", scenes=[s])
        assert p.total_duration() == 4.0
        assert p.scene_count()["video"] == 1

    def test_enum_coverage(self):
        """All scene types and transitions have expected values."""
        assert len(SceneType) == 4
        assert len(Transition) == 3
        assert len(GeneratorType) == 4

    def test_script_writer_to_dict_keys(self):
        writer = ScriptWriter()
        project = writer.from_template("nursing-ad")
        d = writer.to_dict(project)
        assert "title" in d
        assert "scenes" in d
        assert len(d["scenes"]) == 6
        for s in d["scenes"]:
            assert "id" in s
            assert "type" in s
            assert "prompt" in s
