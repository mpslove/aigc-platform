"""Abstract base classes for multi-model inference gateway.

Design:
  GeneratorBase — for generative models (text-to-image, text-to-video, etc.)
  UnderstanderBase — for understanding models (image caption, VQA, OCR, etc.)

Each backend (Agnes, ComfyUI, OpenAI) implements these interfaces.
This lets the rest of the system swap backends without changing code.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum
from typing import Optional


class MediaType(Enum):
    IMAGE = "image"
    VIDEO = "video"
    AUDIO = "audio"


class TaskStatus(Enum):
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


@dataclass
class GenerateResult:
    """Result from a generation task."""
    success: bool
    media_type: MediaType
    file_path: Optional[str] = None
    url: Optional[str] = None
    duration_s: Optional[float] = None      # video duration
    width: int = 0
    height: int = 0
    file_size_kb: float = 0.0
    error: Optional[str] = None
    metadata: dict = field(default_factory=dict)


@dataclass
class UnderstandResult:
    """Result from an understanding/analysis task."""
    success: bool
    text: Optional[str] = None               # caption, OCR text, etc.
    confidence: float = 0.0
    error: Optional[str] = None
    metadata: dict = field(default_factory=dict)


class GeneratorBase(ABC):
    """Abstract generator — produce images, videos, audio from prompts."""

    @abstractmethod
    def generate_image(self, prompt: str, **kwargs) -> GenerateResult:
        ...

    @abstractmethod
    def generate_video(self, prompt: str, **kwargs) -> GenerateResult:
        ...

    @abstractmethod
    def get_provider_name(self) -> str:
        """Return backend identifier, e.g. 'agnes', 'comfyui'."""
        ...


class UnderstanderBase(ABC):
    """Abstract understander — caption, VQA, OCR, etc."""

    @abstractmethod
    def caption(self, media_path: str) -> UnderstandResult:
        """Generate caption for an image or video."""
        ...

    @abstractmethod
    def vqa(self, media_path: str, question: str) -> UnderstandResult:
        """Visual question answering."""
        ...

    @abstractmethod
    def ocr(self, image_path: str) -> UnderstandResult:
        """Extract text from image."""
        ...
