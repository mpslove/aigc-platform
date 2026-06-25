"""Gateway factory — create backends from config."""

import os
from typing import Optional

from .base import GeneratorBase
from .agnes import AgnesGenerator


def create_generator(provider: str = "agnes",
                     api_key: Optional[str] = None,
                     output_dir: str = "./assets") -> GeneratorBase:
    """Factory: create a generator by provider name.

    Args:
        provider: 'agnes', 'comfyui'
        api_key: API key (falls back to env / config)
        output_dir: where to save generated assets

    Returns:
        GeneratorBase instance

    Raises:
        ValueError on unknown provider
    """
    providers = {
        "agnes": AgnesGenerator,
    }
    cls = providers.get(provider)
    if cls is None:
        raise ValueError(f"Unknown provider: {provider}. "
                         f"Available: {list(providers.keys())}")

    if api_key is None:
        api_key = os.environ.get(f"{provider.upper()}_API_KEY", "")

    return cls(api_key=api_key, output_dir=output_dir)


def list_available_providers() -> list[str]:
    """Return list of provider names that are configured."""
    available = []
    for name, default_var in [("agnes", "AGNES_API_KEY")]:
        if os.environ.get(default_var):
            available.append(name)
    return available
