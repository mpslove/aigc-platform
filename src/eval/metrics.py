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


# ── VQA: Visual Question Answering quality evaluation ──────────────

@dataclass
class VQAResult:
    """Visual quality assessment result via multimodal VLM."""
    path: str
    prompt: str = ""
    theme_consistency: float = 0.0      # 0-1: how well content matches expected theme
    visual_quality: float = 0.0         # 0-1: clarity, composition, aesthetics
    text_readability: float = 0.0       # 0-1: are on-screen texts readable
    scene_coherence: float = 0.0        # 0-1: do scenes flow logically
    issues: list[str] = field(default_factory=list)
    raw_judgments: list[dict] = field(default_factory=list)

    @property
    def composite_score(self) -> float:
        """Weighted composite 0-100."""
        return round(max(0, min(100, (
            self.theme_consistency * 35
            + self.visual_quality * 30
            + self.text_readability * 15
            + self.scene_coherence * 20
        ) * 100)), 1)


class VQAEvaluator:
    """Use Qwen2-VL to assess visual content quality.

    Extracts key frames from a video, then asks structured questions
    about theme alignment, visual quality, and text readability.
    Falls back to heuristic scoring when VLM is unavailable.
    """

    # Questions asked per frame for each dimension
    _QUESTIONS = {
        "theme_consistency": "Does this image match the theme '{prompt}'? Rate 0-10.",
        "visual_quality": "Rate the visual quality of this image 0-10 (clarity, composition).",
        "text_readability": "Is any text in this image readable and well-placed? Rate 0-10, or 5 if no text.",
        "scene_coherence": "Does this image fit naturally into a professional video? Rate 0-10.",
    }

    def __init__(self, engine=None):
        """Args:
            engine: QwenVLEngine instance. If None, lazy-loads via get_qwen_engine().
        """
        self._engine = engine

    @property
    def engine(self):
        if self._engine is None:
            # Import here to avoid circular import
            from src.api.app import get_qwen_engine
            self._engine = get_qwen_engine()
        return self._engine

    def _extract_key_frames(self, video_path: str, n_frames: int = 3) -> list[str]:
        """Extract N evenly-spaced key frames from video via ffmpeg."""
        import tempfile
        tmpdir = tempfile.mkdtemp(prefix="vqa_frames_")
        frame_paths = []
        try:
            # Get duration
            r = subprocess.run(
                ["ffmpeg", "-i", video_path],
                capture_output=True, text=True, timeout=15,
            )
            m = re.search(r"Duration: (\d+):(\d+):(\d+\.\d+)", r.stderr)
            if not m:
                return []
            h, mn, s = m.groups()
            duration = int(h) * 3600 + int(mn) * 60 + float(s)

            for i in range(n_frames):
                ts = duration * (i + 1) / (n_frames + 1)
                out_path = os.path.join(tmpdir, f"frame_{i}.jpg")
                subprocess.run(
                    ["ffmpeg", "-ss", str(ts), "-i", video_path,
                     "-frames:v", "1", "-q:v", "2", out_path],
                    capture_output=True, timeout=15,
                )
                if os.path.exists(out_path):
                    frame_paths.append(out_path)
        except (FileNotFoundError, subprocess.TimeoutExpired):
            pass
        return frame_paths

    def _parse_score(self, answer: str) -> float:
        """Extract 0-10 numeric score from VLM answer text."""
        # Look for explicit ratings like "7/10", "Rating: 8", "I'd give it a 6"
        m = re.search(r'(\d+(?:\.\d+)?)(?:\s*/\s*10)?', answer)
        if m:
            val = float(m.group(1))
            if val > 10:
                val = val / 10  # normalize e.g. "75/100" → 7.5
            return min(val / 10.0, 1.0)
        return 0.5  # default middle when no score found

    def evaluate_video(self, video_path: str, prompt: str = "",
                       n_frames: int = 3) -> VQAResult:
        """Evaluate visual content quality of a video using VLM.

        Args:
            video_path: Path to the video file.
            prompt: Original generation prompt (for theme consistency check).
            n_frames: Number of key frames to extract and evaluate.

        Returns:
            VQAResult with dimension scores and composite.
        """
        result = VQAResult(path=video_path, prompt=prompt)

        if not os.path.exists(video_path):
            result.issues.append("Video file not found")
            return result

        # Extract frames
        frames = self._extract_key_frames(video_path, n_frames)
        if not frames:
            # Fallback: use file-level metrics only
            result.issues.append("Could not extract frames for VQA")
            result.theme_consistency = 0.5
            result.visual_quality = 0.5
            result.text_readability = 0.5
            result.scene_coherence = 0.5
            return result

        # Run VQA on each frame for each dimension
        dimension_scores = {k: [] for k in self._QUESTIONS}

        for frame_path in frames:
            for dim, question_template in self._QUESTIONS.items():
                question = question_template.format(prompt=prompt or "general")
                try:
                    answer = self.engine.answer_question(frame_path, question)
                    score = self._parse_score(answer)
                    dimension_scores[dim].append(score)
                    result.raw_judgments.append({
                        "frame": os.path.basename(frame_path),
                        "dimension": dim,
                        "question": question,
                        "answer": answer[:200],
                        "score": round(score, 3),
                    })
                except Exception as e:
                    result.issues.append(f"VQA failed for {dim}: {e}")
                    dimension_scores[dim].append(0.5)

        # Average across frames
        for dim, scores in dimension_scores.items():
            if scores:
                avg = sum(scores) / len(scores)
                setattr(result, dim, round(avg, 3))

        # Identify issues from low scores
        thresholds = {
            "theme_consistency": 0.5,
            "visual_quality": 0.4,
            "text_readability": 0.3,
            "scene_coherence": 0.5,
        }
        for dim, threshold in thresholds.items():
            val = getattr(result, dim)
            if val < threshold:
                result.issues.append(
                    f"Low {dim}: {val:.2f} < {threshold}"
                )

        return result

    def evaluate_image(self, image_path: str, prompt: str = "") -> VQAResult:
        """Evaluate visual quality of a single image using VLM."""
        result = VQAResult(path=image_path, prompt=prompt)

        if not os.path.exists(image_path):
            result.issues.append("Image file not found")
            return result

        for dim, question_template in self._QUESTIONS.items():
            question = question_template.format(prompt=prompt or "general")
            try:
                answer = self.engine.answer_question(image_path, question)
                score = self._parse_score(answer)
                setattr(result, dim, round(score, 3))
                result.raw_judgments.append({
                    "dimension": dim,
                    "question": question,
                    "answer": answer[:200],
                    "score": round(score, 3),
                })
            except Exception as e:
                result.issues.append(f"VQA failed for {dim}: {e}")
                setattr(result, dim, 0.5)

        return result
