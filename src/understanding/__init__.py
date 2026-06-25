"""Multi-modal understanding module — image/video analysis and visual QA.

Provides high-level analysis APIs built on top of Qwen2-VL engine:
  - Image captioning and detailed description
  - Visual question answering
  - Scene analysis (objects, actions, quality)
  - Video keyframe extraction and analysis
"""

from .qwen_engine import QwenVLEngine

__all__ = ["QwenVLEngine", "analyze_image", "answer_question"]


def analyze_image(image_path: str, engine: QwenVLEngine = None) -> dict:
    """Convenience: analyze an image and return structured result.

    Args:
        image_path: Path to image file.
        engine: QwenVLEngine instance. Creates default if None.

    Returns:
        Dict with keys: description, analysis (structured), path.
    """
    eng = engine or QwenVLEngine()
    return {
        "description": eng.describe_image(image_path),
        "analysis": eng.analyze_scene(image_path),
        "path": image_path,
    }


def answer_question(image_path: str, question: str, engine: QwenVLEngine = None) -> str:
    """Convenience: answer a question about an image."""
    eng = engine or QwenVLEngine()
    return eng.answer_question(image_path, question)
