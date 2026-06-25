"""Agnes AI backend — implements GeneratorBase for Agnes API.

Supports:
  - Text-to-video (agnes-video-v2.0)
  - Text-to-image (agnes-image-2.1-flash)

Architecture:
  generate_video() — async POST → poll until complete → download
  generate_image() — synchronous POST → download URL
"""

import os
import time
import json
import requests
from pathlib import Path
from typing import Optional

from .base import GeneratorBase, GenerateResult, MediaType, TaskStatus


class AgnesGenerator(GeneratorBase):
    """Generator backend for Agnes AI API."""

    BASE = "https://apihub.agnes-ai.com"

    def __init__(self, api_key: Optional[str] = None,
                 output_dir: str = "./assets"):
        self.api_key = api_key or os.environ.get("AGNES_API_KEY", "")
        if not self.api_key:
            raise ValueError("AGNES_API_KEY not set")
        self.output_dir = output_dir
        self.headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }

    def get_provider_name(self) -> str:
        return "agnes"

    # ── Image ──────────────────────────────────────────────────────────

    def generate_image(self, prompt: str, **kwargs) -> GenerateResult:
        """Generate image from prompt. Synchronous — single POST."""
        model = kwargs.get("model", "agnes-image-2.1-flash")
        size = kwargs.get("size", "1024x768")
        scene_id = kwargs.get("scene_id", f"img_{int(time.time())}")
        os.makedirs(self.output_dir, exist_ok=True)

        payload = {
            "model": model,
            "prompt": prompt,
            "size": size,
            "extra_body": {"response_format": "url"},
        }
        try:
            r = requests.post(
                f"{self.BASE}/v1/images/generations",
                headers=self.headers, json=payload, timeout=60,
            )
            d = r.json()
            url = d.get("data", [{}])[0].get("url")
            if not url:
                return GenerateResult(
                    success=False, media_type=MediaType.IMAGE,
                    error=f"No URL in response: {json.dumps(d)[:200]}",
                )
            out_path = os.path.join(self.output_dir, f"{scene_id}.jpg")
            dl = requests.get(url, timeout=30)
            with open(out_path, "wb") as f:
                f.write(dl.content)
            sz = len(dl.content) / 1024
            return GenerateResult(
                success=True, media_type=MediaType.IMAGE,
                file_path=out_path, file_size_kb=sz,
            )
        except Exception as e:
            return GenerateResult(
                success=False, media_type=MediaType.IMAGE, error=str(e),
            )

    # ── Video ──────────────────────────────────────────────────────────

    def generate_video(self, prompt: str, **kwargs) -> GenerateResult:
        """Generate video from prompt. Async POST + poll."""
        model = kwargs.get("model", "agnes-video-v2.0")
        width = kwargs.get("width", 1152)
        height = kwargs.get("height", 768)
        duration = kwargs.get("duration", 5.0)
        frame_rate = kwargs.get("frame_rate", 24)
        scene_id = kwargs.get("scene_id", f"vid_{int(time.time())}")
        max_polls = kwargs.get("max_polls", 40)   # reduced for faster runs
        poll_interval = kwargs.get("poll_interval", 5)
        os.makedirs(self.output_dir, exist_ok=True)

        num_frames = kwargs.get("num_frames", int(duration * frame_rate))
        # API limit: max 32 frames for this model
        num_frames = min(32, num_frames)
        # Must be multiple of 8
        num_frames = max(8, (num_frames // 8) * 8)
        payload = {
            "model": model,
            "prompt": prompt,
            "width": width,
            "height": height,
            "num_frames": num_frames,
            "frame_rate": frame_rate,
        }

        try:
            # POST — create task
            r = requests.post(
                f"{self.BASE}/v1/videos",
                headers=self.headers, json=payload, timeout=300,
            )
            d = r.json()
            video_id = d.get("video_id") or d.get("id")
            if not video_id:
                return GenerateResult(
                    success=False, media_type=MediaType.VIDEO,
                    error=f"Create failed: {json.dumps(d)[:200]}",
                )

            # Poll until done
            for _ in range(max_polls):
                time.sleep(poll_interval)
                r = requests.get(
                    f"{self.BASE}/agnesapi",
                    headers=self.headers,
                    params={"video_id": video_id, "model_name": model},
                    timeout=30,
                )
                sd = r.json()
                status = sd.get("status", "?")
                if status == "completed":
                    url = (sd.get("video_url") or sd.get("url")
                           or sd.get("remixed_from_video_id"))
                    if not url:
                        return GenerateResult(
                            success=False, media_type=MediaType.VIDEO,
                            error=f"No URL in completed response: {sd}",
                        )
                    out_path = os.path.join(self.output_dir, f"{scene_id}.mp4")
                    dl = requests.get(url, timeout=120)
                    with open(out_path, "wb") as f:
                        f.write(dl.content)
                    sz = len(dl.content) / 1024
                    return GenerateResult(
                        success=True, media_type=MediaType.VIDEO,
                        file_path=out_path, file_size_kb=sz,
                        width=width, height=height, duration_s=duration,
                    )
                elif status in ("failed", "cancelled"):
                    return GenerateResult(
                        success=False, media_type=MediaType.VIDEO,
                        error=json.dumps(sd.get("error", "")),
                    )

            return GenerateResult(
                success=False, media_type=MediaType.VIDEO,
                error=f"Timeout after {max_polls * poll_interval}s",
            )
        except Exception as e:
            return GenerateResult(
                success=False, media_type=MediaType.VIDEO, error=str(e),
            )

    # ── Batch ─────────────────────────────────────────────────────────

    def generate_batch(self, prompts: list[dict]) -> list[GenerateResult]:
        """Generate multiple assets. Each dict: {prompt, type, **kwargs}."""
        results = []
        for item in prompts:
            media_type = item.get("type", "video")
            if media_type == "image":
                results.append(self.generate_image(
                    item["prompt"], **{k: v for k, v in item.items()
                                       if k not in ("prompt", "type")},
                ))
            else:
                results.append(self.generate_video(
                    item["prompt"], **{k: v for k, v in item.items()
                                       if k not in ("prompt", "type")},
                ))
        return results
