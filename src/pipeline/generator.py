"""AssetProducer — coordinates AI asset generation for a VideoProject.

Orchestrates the gateway to produce all assets for a project:
  - Routes each scene to the right generator
  - Tracks progress and handles failures
  - Supports partial recovery (skip already-generated assets)
"""

import os
import json
import time
from pathlib import Path
from typing import Optional

from .schema import Scene, SceneType, GeneratorType, VideoProject
from src.gateway.base import GeneratorBase, GenerateResult, MediaType


class AssetProducer:
    """Produce all assets for a project via the generator backend."""

    def __init__(self, generator: GeneratorBase, asset_dir: str = "./assets",
                 progress_file: Optional[str] = None):
        self.generator = generator
        self.asset_dir = asset_dir
        self.progress_file = progress_file
        os.makedirs(asset_dir, exist_ok=True)

        # Track results
        self.results: dict[str, GenerateResult] = {}
        self._start_time: float = 0.0

    def produce_scene(self, scene: Scene) -> GenerateResult:
        """Generate asset for a single scene."""
        if scene.scene_type in (SceneType.TITLE, SceneType.END, SceneType.IMAGE):
            scene_id = f"{scene.id}_{scene.scene_type.value}"
            result = self.generator.generate_image(
                scene.prompt,
                scene_id=scene_id,
                size=f"{scene.width}x{scene.height}",
            )
            # Map file path back to scene
            if result.success and result.file_path:
                scene.asset_path = result.file_path
        else:
            result = self.generator.generate_video(
                scene.prompt,
                scene_id=scene.id,
                width=scene.width,
                height=scene.height,
                duration=scene.duration,
                frame_rate=scene.frame_rate,
                num_frames=scene.num_frames,
            )
            if result.success and result.file_path:
                scene.asset_path = result.file_path

        self.results[scene.id] = result
        self._save_progress()
        return result

    def produce_all(self, project: VideoProject, max_workers: int = 3) -> dict[str, GenerateResult]:
        """Generate all assets for a project (concurrent by default).

        Args:
            max_workers: Thread pool size for concurrent generation.
                         1 = serial (backward compatible), 3 = default concurrent.
        """
        from concurrent.futures import ThreadPoolExecutor, as_completed

        self._start_time = time.time()
        self.results = {}
        self._load_progress(project)

        total = len(project.scenes)
        # Collect scenes that need generation
        pending = []
        for i, scene in enumerate(project.scenes):
            if scene.asset_path and os.path.exists(scene.asset_path):
                print(f"  [{i+1}/{total}] {scene.id} — already exists, skip")
                continue
            pending.append((i, scene))

        if not pending:
            elapsed = time.time() - self._start_time
            print(f"  All {total} assets already generated. ({elapsed:.0f}s)")
            return self.results

        print(f"  Generating {len(pending)}/{total} assets "
              f"(workers={max_workers})...")

        # Thread-safe result collection
        import threading
        lock = threading.Lock()

        def _gen(scene: Scene, idx: int) -> tuple[str, GenerateResult]:
            result = self.produce_scene(scene)
            return scene.id, result

        # Dispatch
        with ThreadPoolExecutor(max_workers=max_workers) as pool:
            futures = {
                pool.submit(_gen, scene, idx): scene.id
                for idx, scene in pending
            }
            for future in as_completed(futures):
                sid = futures[future]
                try:
                    scene_id, result = future.result()
                    with lock:
                        self.results[scene_id] = result
                except Exception as exc:
                    with lock:
                        self.results[sid] = GenerateResult(
                            success=False, error=str(exc)
                        )
                    print(f"    FAIL {sid}: {exc}")

        elapsed = time.time() - self._start_time
        success = sum(1 for r in self.results.values() if r.success)
        failed = sum(1 for r in self.results.values() if not r.success)
        print(f"  Produced {success}/{total} assets in {elapsed:.0f}s")
        if failed:
            for sid, r in self.results.items():
                if not r.success:
                    print(f"    FAIL {sid}: {r.error[:100]}")
        return self.results

    def _progress_path(self) -> str:
        if self.progress_file:
            return self.progress_file
        return os.path.join(self.asset_dir, ".progress.json")

    def _save_progress(self):
        """Save progress so generation can be resumed."""
        data = {
            "timestamp": time.time(),
            "results": {
                sid: {
                    "success": r.success,
                    "file_path": r.file_path,
                    "error": r.error,
                }
                for sid, r in self.results.items()
            },
        }
        try:
            with open(self._progress_path(), "w") as f:
                json.dump(data, f)
        except Exception:
            import logging
            logging.getLogger(__name__).warning(
                f"Failed to save progress to {self._progress_path()}"
            )

    def _load_progress(self, project: VideoProject):
        """Load previous progress and mark existing assets."""
        path = self._progress_path()
        if not os.path.exists(path):
            return
        try:
            with open(path) as f:
                data = json.load(f)
        except Exception:
            return

        # Mark scenes with existing files as complete
        for scene in project.scenes:
            entry = data.get("results", {}).get(scene.id, {})
            if entry.get("success") and entry.get("file_path"):
                fp = entry["file_path"]
                if os.path.exists(fp):
                    scene.asset_path = fp
                    self.results[scene.id] = GenerateResult(
                        success=True,
                        media_type=(MediaType.IMAGE
                                    if scene.scene_type in (SceneType.TITLE,
                                                            SceneType.END,
                                                            SceneType.IMAGE)
                                    else MediaType.VIDEO),
                        file_path=fp,
                    )
