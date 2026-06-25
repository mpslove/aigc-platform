"""Tests for the eval/quality module."""

import sys
import os
import json
import pytest
from pathlib import Path
from unittest.mock import patch, MagicMock

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.eval.metrics import (
    VideoQualityAnalyzer, VideoQualityScore,
    BenchmarkRunner, BenchmarkResult,
)
from src.gateway.base import GenerateResult, MediaType


class TestVideoQualityScore:
    def test_default_score(self):
        s = VideoQualityScore(path="test.mp4")
        assert s.quality_score == 45  # 50 baseline - 5 no-audio

    def test_high_resolution_bonus(self):
        s = VideoQualityScore(path="test.mp4", resolution=(1920, 1080),
                              has_audio=True)
        assert s.quality_score == 60  # 50 + 10(hires) no-audio already True

    def test_low_resolution_penalty(self):
        s = VideoQualityScore(path="test.mp4", resolution=(320, 240),
                              has_audio=True)
        assert s.quality_score == 30  # 50 - 20(lowres)

    def test_multiple_factors(self):
        s = VideoQualityScore(path="test.mp4", resolution=(1920, 1080),
                              has_audio=True, errors=["e1"])
        assert s.quality_score == 55  # 50 + 10 - 5

    def test_no_audio_penalty(self):
        s = VideoQualityScore(path="test.mp4", has_audio=False)
        assert s.quality_score == 45  # 50 - 5

    def test_score_clamped_low(self):
        s = VideoQualityScore(path="test.mp4", errors=["a"] * 20)
        assert s.quality_score == 0

    def test_score_clamped_high(self):
        s = VideoQualityScore(path="test.mp4", resolution=(3840, 2160),
                              has_audio=True)
        assert s.quality_score == 60  # 50 + 10, capped at 100


class TestVideoQualityAnalyzer:
    def setup_method(self):
        self.analyzer = VideoQualityAnalyzer()

    def _make_fake_video(self, path: str, size: int = 512000):
        """Create a fake file for os.path.getsize to find."""
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "wb") as f:
            f.write(b"x" * size)

    def test_analyze_basic(self, tmp_path):
        """Verify successful analysis parses FFmpeg output."""
        video = tmp_path / "test.mp4"
        self._make_fake_video(str(video))

        with patch.object(self.analyzer, "analyze", wraps=self.analyzer.analyze):
            # Mock subprocess.run for FFmpeg parsing
            with patch("src.eval.metrics.subprocess.run") as mock_run:
                mock_run.return_value = MagicMock(
                    returncode=0,
                    stderr=(
                        "Duration: 00:00:05.20, start: 0.0, bitrate: 1024 kb/s\n"
                        "Stream #0:0: Video: h264 (libx264), yuv420p, 1152x768, 24 fps\n"
                        "Stream #0:1: Audio: aac, 44100 Hz"
                    ),
                )
                score = self.analyzer.analyze(str(video))

        assert score.duration_s == pytest.approx(5.2, rel=0.01)
        assert score.resolution == (1152, 768)
        assert score.fps == 24.0
        assert score.bitrate_kbps == 1024
        assert score.has_audio is True
        assert score.errors == []

    def test_analyze_no_audio(self, tmp_path):
        """Detect missing audio."""
        video = tmp_path / "test.mp4"
        self._make_fake_video(str(video))

        with patch("src.eval.metrics.subprocess.run") as mock_run:
            mock_run.return_value = MagicMock(
                returncode=0,
                stderr=("Stream #0:0: Video: h264, 640x480\n"
                        "Duration: 00:00:03.00"),
            )
            score = self.analyzer.analyze(str(video))

        assert score.has_audio is False
        assert score.resolution == (640, 480)

    def test_analyze_missing_metadata(self, tmp_path):
        """Handle video with minimal metadata."""
        video = tmp_path / "test.mp4"
        self._make_fake_video(str(video), 256000)

        with patch("src.eval.metrics.subprocess.run") as mock_run:
            mock_run.return_value = MagicMock(
                returncode=0,
                stderr="Output #0, to 'test.mp4':",
            )
            score = self.analyzer.analyze(str(video))

        assert score.duration_s == 0.0
        assert score.resolution == (0, 0)

    def test_analyze_missing_file(self):
        score = self.analyzer.analyze("/tmp/nonexistent_test_video.mp4")
        assert "File not found" in score.errors

    def test_analyze_ffmpeg_not_found(self, tmp_path):
        """Handle missing ffmpeg binary."""
        video = tmp_path / "test.mp4"
        self._make_fake_video(str(video))

        with patch("src.eval.metrics.subprocess.run",
                   side_effect=FileNotFoundError("ffmpeg")):
            score = self.analyzer.analyze(str(video))

        assert "FFmpeg not found" in score.errors


class TestBenchmarkRunner:
    def setup_method(self):
        self.mock_gen = MagicMock()
        self.mock_gen.get_provider_name.return_value = "agnes"
        self.runner = BenchmarkRunner(self.mock_gen)

    def test_run_generates_scores(self):
        """Verify benchmark run produces scores for each attempt."""
        self.mock_gen.generate_video.return_value = GenerateResult(
            success=True, media_type=MediaType.VIDEO,
            file_path="/tmp/bench_test.mp4",
        )

        with patch.object(self.runner.analyzer, "analyze",
                          return_value=VideoQualityScore(
                              path="/tmp/bench_test.mp4",
                              duration_s=5.0, resolution=(1152, 768),
                              file_size_kb=500, fps=24, has_audio=True,
                          )):
            result = self.runner.run("test prompt", num_runs=3)

        assert len(result.scores) == 3
        assert result.success_rate == 1.0
        assert result.average_quality > 0

    def test_run_with_failures(self):
        """Benchmark handles generation failures."""
        returns = [
            GenerateResult(success=True, media_type=MediaType.VIDEO,
                           file_path="/tmp/ok.mp4"),
            GenerateResult(success=False, media_type=MediaType.VIDEO,
                           error="API error"),
            GenerateResult(success=True, media_type=MediaType.VIDEO,
                           file_path="/tmp/ok2.mp4"),
        ]
        self.mock_gen.generate_video.side_effect = returns

        with patch.object(self.runner.analyzer, "analyze",
                          return_value=VideoQualityScore(
                              path="/tmp/ok.mp4", duration_s=5.0,
                          )):
            result = self.runner.run("test", num_runs=3)

        assert len(result.scores) == 3
        assert result.success_rate < 1.0

    def test_report_format(self):
        """Verify report generates readable text."""
        self.mock_gen.generate_video.return_value = GenerateResult(
            success=True, media_type=MediaType.VIDEO,
            file_path="/tmp/r.mp4",
        )
        with patch.object(self.runner.analyzer, "analyze",
                          return_value=VideoQualityScore(
                              path="/tmp/r.mp4", duration_s=5.0,
                              resolution=(1152, 768), file_size_kb=500,
                          )):
            self.runner.run("test prompt", num_runs=2)

        report = self.runner.report()
        assert "AIGC Benchmark Report" in report
        assert "test prompt" in report
        assert "Avg Quality" in report

    def test_export_json(self, tmp_path):
        """Verify JSON export works."""
        self.mock_gen.generate_video.return_value = GenerateResult(
            success=True, media_type=MediaType.VIDEO,
            file_path="/tmp/r.mp4",
        )
        with patch.object(self.runner.analyzer, "analyze",
                          return_value=VideoQualityScore(
                              path="/tmp/r.mp4", duration_s=5.0,
                              resolution=(1152, 768),
                          )):
            self.runner.run("test", num_runs=1)

        json_path = tmp_path / "bench.json"
        self.runner.export_json(str(json_path))
        data = json.loads(json_path.read_text())
        assert len(data) == 1
        assert data[0]["prompt"] == "test"


class TestBenchmarkResult:
    def test_empty_scores(self):
        br = BenchmarkResult(prompt="p", provider="a", scores=[])
        assert br.average_quality == 0.0
        assert br.success_rate == 0.0

    def test_mixed_scores(self):
        scores = [
            VideoQualityScore(path="a.mp4", errors=[]),
            VideoQualityScore(path="b.mp4", errors=["err"]),
        ]
        br = BenchmarkResult(prompt="p", provider="a", scores=scores)
        assert br.success_rate == 0.5
        assert br.average_quality > 0
