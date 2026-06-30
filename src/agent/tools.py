"""Tool definitions for the AIGC agent.

Each tool is a callable with a name, description, and JSON schema.
Tools are registered with the orchestrator and called by the ReAct loop
based on LLM tool-selection decisions.

Key design:
  - Every tool returns a dict (never raises — errors captured in dict)
  - Tools are stateless; state lives in VideoAgent
  - JSON schema for each tool enables LLM function calling
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
        try:
            return self.fn(**kwargs)
        except Exception as e:
            return {"error": str(e), "tool": self.name}

    def to_dict(self) -> dict:
        return {
            "name": self.name,
            "description": self.description,
            "parameters": self.parameters,
        }

    def to_openai_function(self) -> dict:
        """Convert to OpenAI function-calling format."""
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": self.parameters,
            }
        }


def register_core_tools(registry: dict, agent=None):
    """Register all core tools into the given registry dict.

    All production tools now delegate to the *agent* (VideoAgent) instead
    of being stubs. Multi-modal tools keep their own impl.

    Args:
        registry: ToolRegistry dict to insert tools into
        agent: VideoAgent instance — required for production tools
    """
    tools = [
        # ── Production tools (delegate to agent) ──
        Tool(
            name="list_templates",
            description="List available video project templates (nursing-ad, product-ad, travel-ad)",
            fn=lambda **kw: _list_templates(agent),
        ),
        Tool(
            name="create_project",
            description="Create a video project from a template or topic description",
            fn=lambda **kw: _create_project(agent, **kw),
            parameters={
                "type": "object",
                "properties": {
                    "source": {
                        "type": "string",
                        "enum": ["template", "topic"],
                        "description": "Create from template name or free-text topic",
                    },
                    "value": {
                        "type": "string",
                        "description": "Template name (e.g. nursing-ad) or topic description",
                    },
                },
                "required": ["source", "value"],
            },
        ),
        Tool(
            name="generate_assets",
            description="Generate all image/video assets for the current project. Call after create_project.",
            fn=lambda **kw: _generate_assets(agent),
        ),
        Tool(
            name="compose_video",
            description="Compose generated assets into a final video with transitions and music",
            fn=lambda **kw: _compose_video(agent, **kw),
            parameters={
                "type": "object",
                "properties": {
                    "output_filename": {
                        "type": "string",
                        "description": "Optional output filename (default: output.mp4)",
                    },
                },
            },
        ),
        Tool(
            name="evaluate_quality",
            description="Evaluate technical quality of a generated video (resolution, duration, audio)",
            fn=lambda **kw: _evaluate(agent, **kw),
            parameters={
                "type": "object",
                "properties": {
                    "video_path": {
                        "type": "string",
                        "description": "Path to the video file to evaluate",
                    },
                },
                "required": ["video_path"],
            },
        ),
        # ── Multi-modal understanding tools ──
        Tool(
            name="analyze_image",
            description="Describe and analyze an image in detail (objects, scene, quality). Uses Qwen2-VL.",
            fn=lambda **kw: _analyze_image(**kw),
            parameters={
                "type": "object",
                "properties": {
                    "image_path": {"type": "string", "description": "Path to image file"},
                },
                "required": ["image_path"],
            },
        ),
        Tool(
            name="visual_qa",
            description="Answer a question about an image using Qwen2-VL",
            fn=lambda **kw: _visual_qa(**kw),
            parameters={
                "type": "object",
                "properties": {
                    "image_path": {"type": "string", "description": "Path to image file"},
                    "question": {"type": "string", "description": "Question about the image"},
                },
                "required": ["image_path", "question"],
            },
        ),
        Tool(
            name="search_images",
            description="Search indexed images by text query (cross-modal retrieval via CLIP+FAISS+Rerank)",
            fn=lambda **kw: _search_images(**kw),
            parameters={
                "type": "object",
                "properties": {
                    "query": {"type": "string", "description": "Text search query"},
                    "top_k": {"type": "integer", "default": 5, "description": "Max results"},
                },
                "required": ["query"],
            },
        ),
    ]
    for tool in tools:
        registry[tool.name] = tool


# ── Production tool implementations (delegate to agent) ──────────


def _list_templates(agent):
    """List available templates."""
    if agent is None:
        return {"templates": ["nursing-ad", "product-ad", "travel-ad"]}
    return {"templates": agent.script_writer.list_templates()}


def _create_project(agent, source: str = "template",
                    value: str = "nursing-ad", **kwargs):
    """Create project — delegates to VideoAgent."""
    if agent is None:
        return {"error": "No agent instance", "status": "failed"}
    try:
        if source == "template":
            project = agent.from_template(value)
        elif source == "topic":
            project = agent.from_topic(value)
        else:
            return {"error": f"Unknown source: {source}", "status": "failed"}
        return {
            "status": "created",
            "title": project.title,
            "scenes": len(project.scenes),
            "duration_s": project.total_duration(),
            "scenes_detail": [
                {"id": s.id, "type": s.scene_type.value, "duration": s.duration}
                for s in project.scenes
            ],
        }
    except Exception as e:
        return {"error": str(e), "status": "failed"}


def _generate_assets(agent, **kwargs):
    """Generate assets — delegates to VideoAgent."""
    if agent is None:
        return {"error": "No agent instance", "status": "failed"}
    try:
        return agent.generate()
    except Exception as e:
        return {"error": str(e), "status": "failed"}


def _compose_video(agent, output_filename: str = None, **kwargs):
    """Compose video — delegates to VideoAgent."""
    if agent is None:
        return {"error": "No agent instance", "status": "failed"}
    try:
        path = agent.compose(output_filename=output_filename)
        return {"status": "completed", "video_path": path}
    except Exception as e:
        return {"error": str(e), "status": "failed"}


def _evaluate(agent, video_path: str = None, **kwargs):
    """Evaluate quality — delegates to VideoAgent."""
    if agent is None:
        return {"error": "No agent instance", "status": "failed"}
    if not video_path:
        return {"error": "video_path required", "status": "failed"}
    try:
        return agent.evaluate(video_path)
    except Exception as e:
        return {"error": str(e), "status": "failed"}


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
    asset_dir = os.environ.get("AIGC_ASSETS_DIR", "./assets")
    if os.path.exists(asset_dir):
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
