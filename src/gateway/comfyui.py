"""ComfyUI backend — stub for local GPU-based generation.

This backend connects to a running ComfyUI instance via its WebSocket API.
Requires ComfyUI installed at COMFYUI_PATH (env or config).
Not available in the current environment (no GPU).
"""

from .base import GeneratorBase, GenerateResult, MediaType


class ComfyUIGenerator(GeneratorBase):
    """Generator backend for local ComfyUI (stub — GPU required)."""

    def __init__(self, server_url: str = "http://127.0.0.1:8188",
                 output_dir: str = "./assets"):
        self.server_url = server_url
        self.output_dir = output_dir

    def get_provider_name(self) -> str:
        return "comfyui"

    def generate_image(self, prompt: str, **kwargs) -> GenerateResult:
        return GenerateResult(
            success=False, media_type=MediaType.IMAGE,
            error="ComfyUI requires GPU — not available in current env. "
                  "Set up ComfyUI at the configured path and restart.",
        )

    def generate_video(self, prompt: str, **kwargs) -> GenerateResult:
        return GenerateResult(
            success=False, media_type=MediaType.VIDEO,
            error="ComfyUI requires GPU — not available in current env.",
        )
