"""Orchestrator — the main agent loop for automated video production.

The VideoAgent coordinates the full workflow:
  1. Accept a user request (topic, template, or custom script)
  2. (Optional) Enhance prompts via LLM
  3. Generate all assets via the gateway
  4. Compose the final video
  5. Evaluate quality
  6. Report results
"""

import os
import json
import time
import threading
from pathlib import Path
from typing import Optional, Callable

from src.pipeline.schema import VideoProject
from src.pipeline.script_writer import ScriptWriter
from src.pipeline.generator import AssetProducer
from src.pipeline.composer import Composer
from src.eval.metrics import VideoQualityAnalyzer
from src.gateway.base import GeneratorBase
from src.gateway.factory import create_generator


class ToolRegistry(dict):
    """Dictionary-based tool registry with helpful errors."""

    def register(self, tool):
        self[tool.name] = tool


class VideoAgent:
    """Orchestrator for automated AIGC video production."""

    def __init__(self, generator: Optional[GeneratorBase] = None,
                 asset_dir: str = "./assets",
                 output_dir: str = "./output"):
        self.generator = generator
        self.asset_dir = asset_dir
        self.output_dir = output_dir
        self.script_writer = ScriptWriter()
        self.composer = Composer(work_dir=output_dir)
        self.quality_analyzer = VideoQualityAnalyzer()
        self.tools = ToolRegistry()

        # Execution state
        self.current_project: Optional[VideoProject] = None
        self.log: list[dict] = []
        self._lock = threading.Lock()

    def _log(self, action: str, detail: str, status: str = "ok"):
        self.log.append({
            "time": time.strftime("%H:%M:%S"),
            "action": action,
            "detail": detail,
            "status": status,
        })
        # Cap log at 100 entries to prevent memory leak
        if len(self.log) > 100:
            self.log = self.log[-50:]

    def ensure_generator(self):
        """Thread-safe lazy-init generator."""
        if self.generator is not None:
            return
        with self._lock:
            if self.generator is not None:
                return
            self.generator = create_generator(
                "agnes", output_dir=self.asset_dir,
            )
            self._log("init", f"Generator: {self.generator.get_provider_name()}")

    # ── Phase 1: Create Project ───────────────────────────────────────

    def from_template(self, template_name: str) -> VideoProject:
        """Create project from named template."""
        project = self.script_writer.from_template(template_name)
        self.current_project = project
        self._log("project", f"From template '{template_name}': "
                             f"{len(project.scenes)} scenes, "
                             f"{project.total_duration():.0f}s")
        return project

    def from_topic(self, topic: str) -> VideoProject:
        """Create project from a topic description."""
        project = self.script_writer.from_prompt(topic)
        self.current_project = project
        self._log("project", f"From topic '{topic}': "
                             f"{len(project.scenes)} scenes, "
                             f"{project.total_duration():.0f}s")
        return project

    def from_json(self, json_path: str) -> VideoProject:
        """Load project from JSON file."""
        project = ScriptWriter.from_json(json_path)
        self.current_project = project
        self._log("project", f"Loaded from JSON: {len(project.scenes)} scenes")
        return project

    # ── Phase 2: Generate Assets ──────────────────────────────────────

    def generate(self, project: Optional[VideoProject] = None,
                 skip_existing: bool = True) -> dict:
        """Generate all assets for the project."""
        proj = project or self.current_project
        if proj is None:
            raise ValueError("No project. Call from_template/topic/json first.")

        self.ensure_generator()
        producer = AssetProducer(
            self.generator,
            asset_dir=self.asset_dir,
            progress_file=os.path.join(self.asset_dir, ".progress.json")
            if skip_existing else None,
        )

        start = time.time()
        results = producer.produce_all(proj)
        elapsed = time.time() - start

        success = sum(1 for r in results.values() if r.success)
        failed = sum(1 for r in results.values() if not r.success)

        self._log("generate",
                  f"{success}/{len(results)} assets ({elapsed:.0f}s)")
        if failed:
            self._log("generate", f"{failed} failures", "warn")

        return {
            "total": len(results),
            "success": success,
            "failed": failed,
            "elapsed_s": elapsed,
            "results": {
                sid: {"success": r.success, "path": r.file_path,
                      "error": r.error}
                for sid, r in results.items()
            },
        }

    # ── Phase 3: Compose ──────────────────────────────────────────────

    def compose(self, project: Optional[VideoProject] = None,
                output_filename: Optional[str] = None) -> str:
        """Compose final video from generated assets."""
        proj = project or self.current_project
        if proj is None:
            raise ValueError("No project set.")

        if output_filename:
            proj.output_filename = output_filename

        start = time.time()
        output_path = self.composer.compose_project(proj, self.asset_dir)
        elapsed = time.time() - start

        info = Composer.get_video_info(output_path)
        self._log("compose",
                  f"{output_path} ({info.get('duration_str', '?')}, "
                  f"{info.get('size_kb', 0):.0f}KB, {elapsed:.0f}s)")

        return output_path

    # ── Phase 4: Evaluate ─────────────────────────────────────────────

    def evaluate(self, video_path: str) -> dict:
        """Evaluate quality of a generated video."""
        score = self.quality_analyzer.analyze(video_path)
        self._log("evaluate",
                  f"Quality: {score.quality_score:.0f}/100 "
                  f"({score.resolution[0]}x{score.resolution[1]}, "
                  f"{score.duration_s:.1f}s)")
        return {
            "path": score.path,
            "quality_score": score.quality_score,
            "duration_s": score.duration_s,
            "resolution": score.resolution,
            "file_size_kb": score.file_size_kb,
            "has_audio": score.has_audio,
            "errors": score.errors,
        }

    # ── End-to-End ────────────────────────────────────────────────────

    def run_end_to_end(self, source: str = "template",
                       value: str = "nursing-ad",
                       output_filename: Optional[str] = None) -> dict:
        """Run the full workflow: create → generate → compose → evaluate."""
        self.log = []
        start = time.time()

        # Phase 1: Project
        if source == "template":
            project = self.from_template(value)
        elif source == "topic":
            project = self.from_topic(value)
        elif source == "json":
            project = self.from_json(value)
        else:
            raise ValueError(f"Unknown source: {source}")

        # Phase 2: Generate
        gen_result = self.generate(project)
        if gen_result["failed"] > 0:
            self._log("end_to_end", f"{gen_result['failed']} failed assets",
                      "warn")

        # Phase 3: Compose
        video_path = self.compose(project, output_filename)

        # Phase 4: Evaluate
        eval_result = self.evaluate(video_path)

        total_elapsed = time.time() - start
        self._log("end_to_end",
                  f"Done in {total_elapsed:.0f}s → {video_path}")

        return {
            "video_path": video_path,
            "elapsed_s": total_elapsed,
            "generation": gen_result,
            "evaluation": eval_result,
            "project_summary": self.current_project.summary(),
            "log": self.log,
        }

    def summary(self) -> str:
        """Generate a human-readable summary of the last run."""
        lines = ["=== AIGC Video Agent Summary ===\n"]
        if self.current_project:
            s = self.current_project.summary()
            lines.append(f"Project: {s['title']}")
            lines.append(f"Scenes: {s['scenes']} ({s['duration_s']:.0f}s)")
        lines.append("")
        for entry in self.log:
            icon = {"ok": "✓", "warn": "⚠"}.get(entry["status"], "?")
            lines.append(f"  {icon} [{entry['time']}] {entry['action']}: "
                         f"{entry['detail']}")
        return "\n".join(lines)
