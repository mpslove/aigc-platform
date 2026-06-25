"""Data schema for AIGC video production pipeline.

Extends the original aigc-video-studio model with:
  - Transition types (dissolve, fade, crossfade)
  - Background music support
  - Asset tracking (produced vs pending)
  - Script auto-generation metadata
"""

from dataclasses import dataclass, field
from typing import Optional
from enum import Enum


class SceneType(Enum):
    VIDEO = "video"
    TITLE = "title"
    END = "end_card"
    IMAGE = "image"


class GeneratorType(Enum):
    AGNES_VIDEO = "agnes-video-v2.0"
    AGNES_IMAGE = "agnes-image-2.1-flash"
    COMFYUI = "comfyui"
    LOCAL = "local"          # Pre-existing file


class Transition(Enum):
    CUT = "cut"
    FADE = "fade"            # fade to black
    CROSSFADE = "crossfade"  # dissolve between clips


@dataclass
class TextOverlay:
    """Text overlay on a scene."""
    text: str
    font_size: int = 36
    font_color: str = "white"
    x: str = "(w-text_w)/2"
    y: str = "(h-text_h)/2"
    box: bool = True
    box_color: str = "black@0.5"
    box_border_w: int = 8
    start_time: float = 0.0
    end_time: Optional[float] = None


@dataclass
class Scene:
    """A single scene in the video."""
    id: str
    prompt: str
    scene_type: SceneType = SceneType.VIDEO
    generator: GeneratorType = GeneratorType.AGNES_VIDEO
    duration: float = 4.0  # shorter default for API limits
    width: int = 1152
    height: int = 768
    frame_rate: int = 16   # lower FPS for faster gen
    num_frames: int = 0
    text_overlays: list[TextOverlay] = field(default_factory=list)
    transition: Transition = Transition.CUT
    transition_duration: float = 0.5
    asset_path: Optional[str] = None    # set after generation

    def __post_init__(self):
        if self.num_frames == 0:
            self.num_frames = max(16, min(64, int(self.duration * self.frame_rate)))


@dataclass
class VideoProject:
    """Complete video project definition."""
    title: str
    topic: str = ""                     # original topic for script gen
    scenes: list[Scene] = field(default_factory=list)
    output_filename: str = "output.mp4"
    output_dir: str = "./output"
    bg_music: Optional[str] = None
    music_volume: float = 0.3
    script_source: str = "manual"       # 'manual' or 'auto'

    def total_duration(self) -> float:
        return sum(s.duration for s in self.scenes)

    def scene_count(self) -> dict:
        """Count scenes by type."""
        counts = {}
        for s in self.scenes:
            counts[s.scene_type.value] = counts.get(s.scene_type.value, 0) + 1
        return counts

    def summary(self) -> dict:
        return {
            "title": self.title,
            "topic": self.topic,
            "scenes": len(self.scenes),
            "duration_s": self.total_duration(),
            "output": f"{self.output_dir}/{self.output_filename}",
            "script_source": self.script_source,
        }
