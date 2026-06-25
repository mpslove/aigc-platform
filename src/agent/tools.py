"""Tool definitions for the AIGC agent.

Each tool is a callable with a name and description.
Tools are registered with the orchestrator and can be called
by the agent loop.
"""

import os
import json
from typing import Callable, Optional


class Tool:
    """A registered tool the agent can call."""

    def __init__(self, name: str, description: str,
                 fn: Callable, parameters: Optional[dict] = None):
        self.name = name
        self.description = description
        self.fn = fn
        self.parameters = parameters or {}

    def __call__(self, **kwargs):
        return self.fn(**kwargs)

    def to_dict(self) -> dict:
        return {
            "name": self.name,
            "description": self.description,
            "parameters": self.parameters,
        }


def register_core_tools(registry: dict):
    """Register all core tools into the given registry dict."""
    tools = [
        Tool(
            name="list_templates",
            description="List available video project templates",
            fn=lambda **kw: {"templates": [
                "nursing-ad", "product-ad", "travel-ad"]},
        ),
        Tool(
            name="create_project",
            description="Create a VideoProject from a template or topic",
            fn=lambda **kw: _create_project(**kw),
            parameters={
                "type": "object",
                "properties": {
                    "source": {
                        "type": "string",
                        "enum": ["template", "topic", "json"],
                    },
                    "value": {"type": "string"},
                },
            },
        ),
        Tool(
            name="generate_assets",
            description="Generate all assets for a project",
            fn=lambda **kw: _generate_assets(**kw),
        ),
        Tool(
            name="compose_video",
            description="Compose final video from generated assets",
            fn=lambda **kw: _compose_video(**kw),
        ),
        Tool(
            name="evaluate_quality",
            description="Evaluate quality of a generated video",
            fn=lambda **kw: _evaluate(**kw),
        ),
    ]
    for tool in tools:
        registry[tool.name] = tool


def _create_project(source: str = "template",
                    value: str = "nursing-ad",
                    **kwargs):
    """Create a project — stub, actual creation in orchestrator."""
    return {"status": "created", "source": source, "value": value}


def _generate_assets(project_json: str = None, **kwargs):
    """Stub — actual generation in orchestrator."""
    return {"status": "pending", "message": "Call orchestrator.generate()"}


def _compose_video(project_json: str = None, **kwargs):
    """Stub — actual composition in orchestrator."""
    return {"status": "pending", "message": "Call orchestrator.compose()"}


def _evaluate(video_path: str = None, **kwargs):
    """Stub — actual evaluation in orchestrator."""
    return {"status": "pending", "message": "Call orchestrator.evaluate()"}
