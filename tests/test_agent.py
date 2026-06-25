"""Tests for the agent orchestrator module."""

import sys
import os
import json
import pytest
from pathlib import Path
from unittest.mock import patch, MagicMock, PropertyMock

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.agent.orchestrator import VideoAgent, ToolRegistry
from src.agent.tools import Tool, register_core_tools
from src.pipeline.schema import VideoProject, Scene, SceneType
from src.gateway.base import GenerateResult, MediaType


class TestToolRegistry:
    def test_register_and_get(self):
        registry = ToolRegistry()
        tool = Tool(name="test_tool", description="A test tool",
                    fn=lambda: "done")
        registry.register(tool)
        assert "test_tool" in registry
        assert registry["test_tool"]() == "done"

    def test_register_core_tools(self):
        registry = ToolRegistry()
        register_core_tools(registry)
        assert "list_templates" in registry
        assert "create_project" in registry
        assert "generate_assets" in registry
        assert "compose_video" in registry
        assert "evaluate_quality" in registry

    def test_tool_to_dict(self):
        tool = Tool(name="test", description="desc",
                    fn=lambda: None,
                    parameters={"type": "object"})
        d = tool.to_dict()
        assert d["name"] == "test"
        assert d["parameters"]["type"] == "object"


class TestVideoAgent:
    def setup_method(self):
        self.agent = VideoAgent(asset_dir="/tmp/test_assets",
                                output_dir="/tmp/test_output")

    def test_log(self):
        self.agent._log("init", "test detail")
        assert len(self.agent.log) == 1
        assert self.agent.log[0]["action"] == "init"

    def test_from_template(self):
        project = self.agent.from_template("nursing-ad")
        assert project is not None
        assert len(project.scenes) == 6
        assert self.agent.current_project is project
        assert len(self.agent.log) >= 1

    def test_from_topic(self):
        project = self.agent.from_topic("护理广告")
        assert project is not None
        assert len(project.scenes) >= 1

    def test_from_json(self, tmp_path):
        # First create a project and save it
        project = self.agent.from_template("travel-ad")
        json_path = tmp_path / "test.json"
        self.agent.script_writer.save_script(project, str(json_path))

        # Load it fresh
        agent2 = VideoAgent()
        loaded = agent2.from_json(str(json_path))
        assert loaded.title == project.title
        assert len(loaded.scenes) == len(project.scenes)

    def test_generate_no_project(self):
        with pytest.raises(ValueError, match="No project"):
            self.agent.generate()

    @patch.object(VideoAgent, "ensure_generator")
    def test_generate_calls_producer(self, mock_ensure):
        """Verify generate() produces assets for all scenes."""
        self.agent.from_template("nursing-ad")

        with patch("src.agent.orchestrator.AssetProducer") as mock_producer_cls:
            mock_producer = MagicMock()
            mock_producer_cls.return_value = mock_producer
            # Produce_all returns results
            mock_producer.produce_all.return_value = {
                s.id: GenerateResult(
                    success=True, media_type=MediaType.VIDEO,
                    file_path=f"/tmp/{s.id}.mp4",
                )
                for s in self.agent.current_project.scenes
            }
            result = self.agent.generate()

        assert result["total"] == 6
        assert result["success"] == 6
        assert result["failed"] == 0

    @patch.object(VideoAgent, "ensure_generator")
    def test_generate_partial_failure(self, mock_ensure):
        """Handle mixed success/failure in generation."""
        self.agent.from_template("nursing-ad")

        with patch("src.agent.orchestrator.AssetProducer") as mock_producer_cls:
            mock_producer = MagicMock()
            mock_producer_cls.return_value = mock_producer
            results = {}
            for i, s in enumerate(self.agent.current_project.scenes):
                results[s.id] = GenerateResult(
                    success=(i != 2),  # third scene fails
                    media_type=MediaType.VIDEO,
                    file_path=f"/tmp/{s.id}.mp4",
                    error="API error" if i == 2 else None,
                )
            mock_producer.produce_all.return_value = results
            result = self.agent.generate()

        assert result["success"] == 5
        assert result["failed"] == 1

    @patch("src.agent.orchestrator.Composer.get_video_info")
    @patch("src.agent.orchestrator.Composer.compose_project")
    def test_compose(self, mock_compose, mock_info):
        """Verify compose produces output path."""
        self.agent.from_template("nursing-ad")
        # Set asset paths so compose doesn't fail
        for s in self.agent.current_project.scenes:
            s.asset_path = f"/tmp/{s.id}.mp4"

        mock_compose.return_value = "/tmp/test_output/nursing_ad_final.mp4"
        mock_info.return_value = {
            "duration_str": "00:00:26.20",
            "size_kb": 3200,
        }

        result = self.agent.compose()
        assert result == "/tmp/test_output/nursing_ad_final.mp4"
        assert len(self.agent.log) >= 1

    def test_compose_no_project(self):
        with pytest.raises(ValueError, match="No project"):
            self.agent.compose()

    @patch("src.agent.orchestrator.VideoQualityAnalyzer.analyze")
    def test_evaluate(self, mock_analyze):
        from src.eval.metrics import VideoQualityScore
        mock_analyze.return_value = VideoQualityScore(
            path="/tmp/video.mp4",
            duration_s=26.0,
            resolution=(1152, 768),
            file_size_kb=3200,
            fps=24,
            has_audio=True,
        )

        result = self.agent.evaluate("/tmp/video.mp4")
        assert result["quality_score"] > 0
        assert result["duration_s"] == 26.0
        assert result["resolution"] == (1152, 768)

    @patch.object(VideoAgent, "ensure_generator")
    @patch("src.agent.orchestrator.AssetProducer")
    @patch("src.agent.orchestrator.Composer.compose_project")
    @patch("src.agent.orchestrator.Composer.get_video_info")
    @patch("src.agent.orchestrator.VideoQualityAnalyzer.analyze")
    def test_end_to_end(self, mock_analyze, mock_info,
                        mock_compose, mock_producer_cls, mock_ensure):
        """Full workflow smoke test."""
        from src.eval.metrics import VideoQualityScore
        mock_producer = MagicMock()
        mock_producer_cls.return_value = mock_producer
        mock_producer.produce_all.return_value = {
            f"s{i}": GenerateResult(success=True, media_type=MediaType.VIDEO,
                                     file_path=f"/tmp/s{i}.mp4")
            for i in range(6)
        }
        mock_compose.return_value = "/tmp/output.mp4"
        mock_info.return_value = {
            "duration_str": "00:00:26",
            "size_kb": 3000,
        }
        mock_analyze.return_value = VideoQualityScore(
            path="/tmp/output.mp4", duration_s=26.0,
            resolution=(1152, 768), file_size_kb=3000,
        )

        result = self.agent.run_end_to_end("template", "nursing-ad")
        assert "video_path" in result
        assert "generation" in result
        assert "evaluation" in result

    def test_summary_format(self):
        self.agent._log("init", "started")
        self.agent._log("done", "finished", "warn")
        summary = self.agent.summary()
        assert "AIGC Video Agent Summary" in summary
        assert "started" in summary
        assert "finished" in summary

    def test_summary_with_project(self):
        self.agent.from_template("travel-ad")
        summary = self.agent.summary()
        assert "Travel" in summary
        assert summary.count("Scenes:") >= 1


class TestToolFunctionality:
    def test_list_templates(self):
        registry = ToolRegistry()
        register_core_tools(registry)
        result = registry["list_templates"]()
        assert "templates" in result
        assert "nursing-ad" in result["templates"]

    def test_create_project_tool(self):
        registry = ToolRegistry()
        register_core_tools(registry)
        result = registry["create_project"](source="template",
                                            value="nursing-ad")
        assert result["status"] == "created"
