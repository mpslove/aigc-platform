"""Composer — advanced video composition engine (FFmpeg).

Upgraded from the original aigc-video-studio composer with:
  - Transition effects (fade, crossfade between clips)
  - Background music overlay with volume control
  - Timed subtitle/text overlays
  - Video metadata extraction
  - Support for title/end card creation from static images
"""

import os
import re
import subprocess
import json
from pathlib import Path
from typing import Optional

from .schema import Scene, SceneType, VideoProject, Transition


class Composer:
    """FFmpeg-based video composition with transitions and music."""

    def __init__(self, work_dir: str = "./output"):
        self.work_dir = work_dir
        os.makedirs(work_dir, exist_ok=True)

    # ── Text overlay ──────────────────────────────────────────────────

    def build_drawtext_filter(self, overlays: list,
                              video_duration: float) -> Optional[str]:
        """Build FFmpeg drawtext filter string from text overlays."""
        if not overlays:
            return None

        filters = []
        for ot in overlays:
            enable = f"between(t,{ot.start_time},{ot.end_time or video_duration})"
            # FFmpeg drawtext requires escaping: \ : ' % [ ]
            esc_text = (ot.text
                        .replace("\\", r"\\\\")
                        .replace(":", r"\\:")
                        .replace("'", r"\'")
                        .replace("%", r"\\%")
                        .replace("[", r"\\[")
                        .replace("]", r"\\]"))
            box = (f":box=1:boxcolor={ot.box_color}"
                   f":boxborderw={ot.box_border_w}") if ot.box else ""
            filters.append(
                f"drawtext=text='{esc_text}'"
                f":fontcolor={ot.font_color}"
                f":fontsize={ot.font_size}"
                f":x={ot.x}:y={ot.y}"
                f"{box}"
                f":enable='{enable}'"
            )
        return ",".join(filters)

    # ── Title / end card ──────────────────────────────────────────────

    def make_static_segment(self, image_path: str, scene: Scene,
                            output_path: str) -> str:
        """Create video segment from a static image + overlays."""
        vf = self.build_drawtext_filter(scene.text_overlays, scene.duration)
        cmd = [
            "ffmpeg", "-y",
            "-loop", "1", "-i", image_path,
            "-c:v", "libx264",
            "-t", str(scene.duration),
            "-pix_fmt", "yuv420p",
            "-r", str(scene.frame_rate),
        ]
        if vf:
            cmd.extend(["-vf", vf])
        cmd.append(output_path)

        r = subprocess.run(cmd, capture_output=True, text=True)
        if r.returncode != 0:
            raise RuntimeError(f"FFmpeg static segment failed: "
                               f"{r.stderr[:200]}")
        return output_path

    # ── Transition between segments ───────────────────────────────────

    def _get_transition_filter(self, t: Transition, duration: float) -> str:
        """Get FFmpeg filter for transition effect."""
        if t == Transition.FADE:
            return f"fade=t=out:st=0:d={duration}"
        elif t == Transition.CROSSFADE:
            return f"fade=t=in:st=0:d={duration}"
        return ""

    # ── Concat with transitions ───────────────────────────────────────

    def concat_segments(self, segment_paths: list[str],
                        transitions: list[Transition],
                        output_path: str,
                        xfade_duration: float = 0.5) -> str:
        """Concatenate video segments with optional transitions.

        Uses concat demuxer for cut-only segments (most common).
        For transitions, uses xfade filter graph for real crossfades.
        """
        has_transitions = transitions and not all(
            t == Transition.CUT for t in transitions
        )
        if not has_transitions:
            return self._concat_cut(segment_paths, output_path)
        return self._concat_xfade(segment_paths, transitions,
                                  output_path, xfade_duration)

    def _concat_xfade(self, segment_paths: list[str],
                       transitions: list[Transition],
                       output_path: str,
                       xfade_duration: float = 0.5) -> str:
        """Concatenate video segments using FFmpeg xfade filter (real crossfade).

        Builds a complex filter graph that chains xfade filters between
        adjacent segments, with audio acrossfade for smooth audio blending.

        Requires FFmpeg 4.4+ for the xfade filter.
        """
        n = len(segment_paths)
        if n < 2:
            # Single segment — just copy
            import shutil
            shutil.copy2(segment_paths[0], output_path)
            return output_path

        # Build input arguments
        inputs = []
        for p in segment_paths:
            inputs.extend(["-i", p])

        # Build xfade filter chain
        # xfade needs: offset (when transition starts in first clip)
        # = clip_duration - xfade_duration for each pair
        filter_parts = []
        audio_parts = []

        # Get durations for offset calculation
        durations = []
        for p in segment_paths:
            info = self.get_video_info(p)
            durations.append(info.get("duration_s", 2.0))

        # Video xfade chain
        current_label = "[0:v]"
        offset = durations[0] - xfade_duration

        for i in range(1, n):
            transition = transitions[i - 1] if i - 1 < len(transitions) else Transition.CUT
            out_label = f"[v{i}]" if i < n - 1 else "[vout]"

            if transition == Transition.CROSSFADE:
                xfade_effect = "crossfade"
            elif transition == Transition.FADE:
                xfade_effect = "fadeblack"  # fade to black then in
            else:
                # CUT — use crossfade with duration 0 (instant cut)
                # Can't duraion=0 in xfade, so use very short
                xfade_effect = "smoothleft"

            offset = max(offset, 0.1)  # safety: offset must be > 0
            filter_parts.append(
                f"{current_label}[{i}:v]xfade=transition={xfade_effect}"
                f":duration={xfade_duration}:offset={offset:.3f}{out_label}"
            )
            current_label = out_label

            # Accumulate offset for next pair
            if i < n - 1:
                offset += durations[i] - xfade_duration * 2

        # Audio acrossfade chain
        current_audio = "[0:a]"
        for i in range(1, n):
            out_label = f"[a{i}]" if i < n - 1 else "[aout]"
            audio_parts.append(
                f"{current_audio}[{i}:a]acrossfade=d={xfade_duration}"
                f":c1=tri:c2=tri{out_label}"
            )
            current_audio = out_label

        # Combine video + audio filters
        video_filter = ";".join(filter_parts)
        audio_filter = ";".join(audio_parts)
        filter_complex = f"{video_filter};{audio_filter}"

        cmd = [
            "ffmpeg", "-y",
            *inputs,
            "-filter_complex", filter_complex,
            "-map", "[vout]",
            "-map", "[aout]",
            "-c:v", "libx264", "-preset", "medium",
            "-pix_fmt", "yuv420p",
            "-c:a", "aac", "-b:a", "128k",
            output_path,
        ]

        r = subprocess.run(cmd, capture_output=True, text=True, timeout=300)
        if r.returncode != 0:
            # Fallback to simple concat on xfade failure
            import logging
            logging.getLogger(__name__).warning(
                f"xfade failed ({r.stderr[:150]}), falling back to concat"
            )
            return self._concat_cut(segment_paths, output_path)

        return output_path

    def _concat_cut(self, segment_paths: list[str],
                    output_path: str) -> str:
        """Simple concat — no transitions."""
        import tempfile
        fd, concat_file = tempfile.mkstemp(suffix=".txt", dir=self.work_dir,
                                           prefix="concat_")
        os.close(fd)
        try:
            with open(concat_file, "w") as f:
                for p in segment_paths:
                    abs_p = os.path.abspath(p).replace("\\", "/")
                    f.write(f"file '{abs_p}'\n")

            cmd = [
                "ffmpeg", "-y",
                "-f", "concat", "-safe", "0",
                "-i", concat_file,
                "-c:v", "libx264", "-preset", "medium",
                "-pix_fmt", "yuv420p",
                output_path,
            ]
            r = subprocess.run(cmd, capture_output=True, text=True)
            if r.returncode != 0:
                raise RuntimeError(f"FFmpeg concat failed: {r.stderr[:200]}")
            return output_path
        finally:
            try:
                os.unlink(concat_file)
            except OSError:
                pass
    # ── Background music ──────────────────────────────────────────────

    def add_background_music(self, video_path: str,
                             music_path: str,
                             volume: float = 0.3,
                             output_path: Optional[str] = None) -> str:
        """Add background music to a video."""
        if output_path is None:
            base, ext = os.path.splitext(video_path)
            output_path = f"{base}_with_music{ext}"

        cmd = [
            "ffmpeg", "-y",
            "-i", video_path,
            "-i", music_path,
            "-filter_complex",
            f"[1:a]volume={volume}[music];"
            f"[0:a][music]amix=inputs=2:duration=first",
            "-c:v", "copy",
            "-c:a", "aac", "-b:a", "128k",
            output_path,
        ]
        r = subprocess.run(cmd, capture_output=True, text=True)
        if r.returncode != 0:
            raise RuntimeError(f"FFmpeg add music failed: {r.stderr[:200]}")
        return output_path

    # ── Full project composition ──────────────────────────────────────

    def compose_project(self, project: VideoProject,
                        asset_dir: str = "./assets") -> str:
        """Full composition: build segments from assets, concat, add music."""
        os.makedirs(project.output_dir, exist_ok=True)
        segments = []
        transitions = []

        for i, scene in enumerate(project.scenes):
            asset_path = scene.asset_path
            if not asset_path or not os.path.exists(asset_path):
                raise FileNotFoundError(
                    f"Asset for scene '{scene.id}' not found at "
                    f"{asset_path}. Run AssetProducer first."
                )

            seg_path = os.path.join(self.work_dir, f"seg_{scene.id}.mp4")

            if scene.scene_type in (SceneType.TITLE, SceneType.END, SceneType.IMAGE):
                self.make_static_segment(asset_path, scene, seg_path)
            else:
                # Video clips used directly
                seg_path = asset_path

            segments.append(seg_path)
            transitions.append(scene.transition)

        # Concat
        output_path = os.path.join(project.output_dir, project.output_filename)
        self.concat_segments(segments, transitions, output_path)

        # Add background music if specified
        if project.bg_music and os.path.exists(project.bg_music):
            output_path = self.add_background_music(
                output_path, project.bg_music, project.music_volume,
            )

        return output_path

    # ── Metadata ──────────────────────────────────────────────────────

    @staticmethod
    def get_video_info(video_path: str) -> dict:
        """Extract duration, size, resolution from a video file."""
        info = {"path": video_path}
        info["size_kb"] = os.path.getsize(video_path) / 1024

        r = subprocess.run(
            ["ffmpeg", "-i", video_path],
            capture_output=True, text=True,
        )
        stderr = r.stderr

        m = re.search(r"Duration: (\d+):(\d+):(\d+\.\d+)", stderr)
        if m:
            h, m_, s = m.groups()
            info["duration_s"] = int(h) * 3600 + int(m_) * 60 + float(s)
            info["duration_str"] = f"{h}:{m_}:{s}"
        m = re.search(r"Stream .* Video:.* (\d+)x(\d+)", stderr)
        if m:
            info["width"] = int(m.group(1))
            info["height"] = int(m.group(2))

        m = re.search(r"(\d+) fps", stderr)
        if m:
            info["fps"] = float(m.group(1))

        return info

    @staticmethod
    def verify_compatibility(segment_paths: list[str]) -> dict:
        """Check if all segments have compatible codecs/resolutions."""
        infos = []
        for p in segment_paths:
            infos.append(Composer.get_video_info(p))

        if not infos:
            return {"compatible": True, "message": "no segments"}

        ref = infos[0]
        issues = []
        for i, info in enumerate(infos[1:], 1):
            if info.get("width") != ref.get("width"):
                issues.append(f"seg {i}: {info.get('width')}x{info.get('height')} "
                              f"!= ref {ref.get('width')}x{ref.get('height')}")

        return {
            "compatible": len(issues) == 0,
            "segments": len(infos),
            "issues": issues,
        }
