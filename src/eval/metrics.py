"""Eval: quality evaluation for AIGC-generated videos and images.

Provides:
  - VideoQualityMetrics — technical quality scoring
  - PromptAlignmentScorer — how well output matches prompt
  - BenchmarkRunner — compare multiple generations on same prompt
"""

import os
import json
import subprocess
import re
from pathlib import Path
from typing import Optional
from dataclasses import dataclass, field


@dataclass
class VideoQualityScore:
    """Quality score for a single video."""
    path: str
    duration_s: float = 0.0
    resolution: tuple = (0, 0)
    file_size_kb: float = 0.0
    bitrate_kbps: float = 0.0
    fps: float = 0.0
    has_audio: bool = False
    errors: list[str] = field(default_factory=list)

    @property
    def quality_score(self) -> float:
        """Composite quality score 0-100."""
        score = 50.0  # baseline

        # Penalty: low resolution
        if self.resolution[0] > 0:
            pixels = self.resolution[0] * self.resolution[1]
            if pixels < 480 * 360:
                score -= 20
            elif pixels < 854 * 480:
                score -= 10
            elif pixels >= 1920 * 1080:
                score += 10

        # Penalty: missing audio when expected
        if not self.has_audio:
            score -= 5

        # Penalty: errors
        score -= len(self.errors) * 5

        return max(0, min(100, score))


@dataclass
class BenchmarkResult:
    """Result of a benchmark run."""
    prompt: str
    provider: str
    scores: list[VideoQualityScore]
    average_quality: float = 0.0
    success_rate: float = 0.0

    def __post_init__(self):
        if self.scores:
            self.average_quality = (
                sum(s.quality_score for s in self.scores) / len(self.scores)
            )
            self.success_rate = (
                sum(1 for s in self.scores if s.errors == []) / len(self.scores)
            )


class VideoQualityAnalyzer:
    """Analyze technical quality of generated videos."""

    def analyze(self, video_path: str) -> VideoQualityScore:
        """Extract quality metrics from a video file."""
        score = VideoQualityScore(path=video_path)

        if not os.path.exists(video_path):
            score.errors.append("File not found")
            return score

        score.file_size_kb = os.path.getsize(video_path) / 1024

        try:
            r = subprocess.run(
                ["ffmpeg", "-i", video_path],
                capture_output=True, text=True, timeout=15,
            )
            stderr = r.stderr
        except FileNotFoundError:
            score.errors.append("FFmpeg not found")
            return score
        except subprocess.TimeoutExpired:
            score.errors.append("FFmpeg timed out")
            return score

        # Duration
        m = re.search(r"Duration: (\d+):(\d+):(\d+\.\d+)", stderr)
        if m:
            h, m_, s = m.groups()
            score.duration_s = int(h) * 3600 + int(m_) * 60 + float(s)
        else:
            score.errors.append("Could not parse duration")

        # Resolution
        m = re.search(r"Stream .* Video:.* (\d+)x(\d+)", stderr)
        if m:
            score.resolution = (int(m.group(1)), int(m.group(2)))

        # FPS
        m = re.search(r"(\d+(?:\.\d+)?) fps", stderr)
        if m:
            score.fps = float(m.group(1))

        # Bitrate
        m = re.search(r"(\d+) kb/s", stderr)
        if m:
            score.bitrate_kbps = float(m.group(1))
        elif score.duration_s > 0 and score.file_size_kb > 0:
            score.bitrate_kbps = (score.file_size_kb * 8) / score.duration_s

        # Audio
        score.has_audio = bool(re.search(r"Audio:", stderr))

        return score


class BenchmarkRunner:
    """Run benchmarks: generate same prompt multiple times and compare."""

    def __init__(self, generator, analyzer: Optional[VideoQualityAnalyzer] = None):
        self.generator = generator
        self.analyzer = analyzer or VideoQualityAnalyzer()
        self.history: list[BenchmarkResult] = []

    def run(self, prompt: str, provider: str = "agnes",
            num_runs: int = 3, **gen_kwargs) -> BenchmarkResult:
        """Generate same prompt N times and score quality."""
        scores = []
        for i in range(num_runs):
            scene_id = f"bench_{prompt[:8]}_{i}"
            result = self.generator.generate_video(
                prompt, scene_id=scene_id, **gen_kwargs,
            )
            if result.success and result.file_path:
                s = self.analyzer.analyze(result.file_path)
                scores.append(s)
            else:
                s = VideoQualityScore(
                    path="",
                    errors=[result.error or "Generation failed"],
                )
                scores.append(s)

        br = BenchmarkResult(prompt=prompt, provider=provider, scores=scores)
        self.history.append(br)
        return br

    def compare_providers(self, prompt: str,
                          generators: dict[str, object],
                          **gen_kwargs) -> dict[str, BenchmarkResult]:
        """Compare same prompt across multiple generator backends."""
        results = {}
        for name, gen in generators.items():
            original = self.generator
            self.generator = gen
            results[name] = self.run(prompt, provider=name, **gen_kwargs)
            self.generator = original
        return results

    def report(self) -> str:
        """Generate a text report of all benchmark runs."""
        lines = ["=== AIGC Benchmark Report ===\n"]
        for br in self.history:
            lines.append(f"\nPrompt: {br.prompt[:60]}")
            lines.append(f"Provider: {br.provider}")
            lines.append(f"Runs: {len(br.scores)}")
            lines.append(f"Avg Quality: {br.average_quality:.1f}/100")
            lines.append(f"Success Rate: {br.success_rate * 100:.0f}%")
            for i, s in enumerate(br.scores):
                lines.append(
                    f"  Run {i}: {s.resolution[0]}x{s.resolution[1]}, "
                    f"{s.duration_s:.1f}s, {s.file_size_kb:.0f}KB, "
                    f"score={s.quality_score:.0f}"
                    + (" ERR" if s.errors else "")
                )
        return "\n".join(lines)

    def save_report(self, path: str):
        """Save benchmark report to file."""
        with open(path, "w") as f:
            f.write(self.report())

    def export_json(self, path: str):
        """Export benchmark data as JSON."""
        data = []
        for br in self.history:
            data.append({
                "prompt": br.prompt,
                "provider": br.provider,
                "average_quality": br.average_quality,
                "success_rate": br.success_rate,
                "runs": [
                    {"duration_s": s.duration_s,
                     "resolution": list(s.resolution),
                     "file_size_kb": s.file_size_kb,
                     "quality_score": s.quality_score,
                     "errors": s.errors}
                    for s in br.scores
                ],
            })
        with open(path, "w") as f:
            json.dump(data, f, indent=2)
