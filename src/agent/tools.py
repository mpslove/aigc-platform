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
    """Register all core tools into the given registry dict.
    
    Includes:
      - Video production: templates, generate, compose, evaluate
      - Multi-modal understanding: analyze_image, search_images, visual_qa
      - Retrieval: rag_search, index_assets
    """
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
        # ── Multi-modal understanding tools ──
        Tool(
            name="analyze_image",
            description="Describe and analyze an image in detail (objects, scene, quality)",
            fn=lambda **kw: _analyze_image(**kw),
            parameters={
                "type": "object",
                "properties": {
                    "image_path": {"type": "string"},
                },
                "required": ["image_path"],
            },
        ),
        Tool(
            name="visual_qa",
            description="Answer a question about an image",
            fn=lambda **kw: _visual_qa(**kw),
            parameters={
                "type": "object",
                "properties": {
                    "image_path": {"type": "string"},
                    "question": {"type": "string"},
                },
                "required": ["image_path", "question"],
            },
        ),
        Tool(
            name="search_images",
            description="Search indexed images by text query (cross-modal retrieval)",
            fn=lambda **kw: _search_images(**kw),
            parameters={
                "type": "object",
                "properties": {
                    "query": {"type": "string"},
                    "top_k": {"type": "integer", "default": 5},
                },
                "required": ["query"],
            },
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


# ── Multi-modal understanding tool implementations ─────────────────


def _analyze_image(image_path: str = None, **kwargs):
    """Analyze an image: description + structured scene analysis."""
    if not image_path or not os.path.exists(image_path):
        return {"error": f"Image not found: {image_path}"}

    from src.understanding import QwenVLEngine
    engine = QwenVLEngine()
    result = engine.analyze_scene(image_path)
    return {
        "image": image_path,
        "analysis": result,
    }


def _visual_qa(image_path: str = None, question: str = None, **kwargs):
    """Answer a question about an image."""
    if not image_path or not os.path.exists(image_path):
        return {"error": f"Image not found: {image_path}"}
    if not question:
        return {"error": "No question provided"}

    from src.understanding import QwenVLEngine
    engine = QwenVLEngine()
    answer = engine.answer_question(image_path, question)
    return {
        "image": image_path,
        "question": question,
        "answer": answer,
    }


def _search_images(query: str = None, top_k: int = 5, **kwargs):
    """Search indexed images by text query using Visual RAG."""
    if not query:
        return {"error": "No query provided"}

    from src.rag.visual_rag import VisualRAG
    rag = VisualRAG()

    # Index assets directory if not already loaded
    import os as _os
    asset_dir = _os.environ.get("AIGC_ASSETS_DIR", "./assets")
    if _os.path.exists(asset_dir):
        rag.index_directory(asset_dir)

    results = rag.search(query, top_k=top_k, rerank=True)
    return {
        "query": query,
        "results": [
            {
                "id": r["id"],
                "path": r["path"],
                "caption": r["caption"],
                "score": round(r["score"], 4),
                "rerank_score": round(r.get("rerank_score", 0), 4),
            }
            for r in results
        ],
        "total": len(results),
    }
