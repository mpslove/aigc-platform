"""Configuration and environment management."""
import os
import json
import yaml
from pathlib import Path
from typing import Optional


def find_project_root() -> Path:
    """Find project root by looking for config/providers.yaml."""
    cwd = Path.cwd()
    for parent in [cwd] + list(cwd.parents):
        if (parent / "config" / "providers.yaml").exists():
            return parent
    # Fallback: check env
    env_root = os.environ.get("AIGC_PLATFORM_ROOT")
    if env_root:
        return Path(env_root)
    # Last resort: D:\aigc-platform
    return Path("D:/aigc-platform")


def load_config() -> dict:
    """Load providers.yaml and merge with environment variables."""
    root = find_project_root()
    config_path = root / "config" / "providers.yaml"

    config = {}
    if config_path.exists():
        with open(config_path) as f:
            config = yaml.safe_load(f) or {}

    providers = config.get("providers", {})

    # Environment overrides — env vars take precedence
    key_map = {
        "AGNES_API_KEY": ("agnes", "api_key"),
        "AIGC_ASSETS_DIR": ("global", "assets_dir"),
    }
    for env_var, (section, key) in key_map.items():
        if env_var in os.environ:
            if section not in providers:
                providers[section] = {}
            providers[section][key] = os.environ[env_var]

    config["providers"] = providers
    return config


def get_api_key(provider: str) -> str:
    """Get API key for a provider from config or env."""
    config = load_config()
    key = config.get("providers", {}).get(provider, {}).get("api_key", "")
    env_var = f"{provider.upper()}_API_KEY"
    return os.environ.get(env_var, key)
