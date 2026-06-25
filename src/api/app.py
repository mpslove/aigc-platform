"""FastAPI REST API for the AIGC platform.

Provides:
  - GET  /api/templates      — list available templates
  - POST /api/project        — create a project from template/topic
  - POST /api/generate       — generate assets
  - POST /api/compose        — compose final video
  - GET  /api/status/{job}   — check generation status
  - POST /api/evaluate       -- evaluate video quality
  - POST /api/e2e            -- full end-to-end pipeline
  - POST /api/understand     -- multi-modal image understanding (Qwen2-VL)
  - POST /api/visual-qa      -- visual question answering
  - POST /api/rag/search     -- cross-modal image retrieval (text to image)
  - POST /api/rag/index      -- index image directory for search
  - GET  /api/rag/stats      -- RAG index statistics
"""

import os
import sys
import json
import time
from pathlib import Path
from typing import Optional

# Allow running directly
sys.path.insert(0, str(Path(__file__).parent.parent))

from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from pydantic import BaseModel

from src.agent.orchestrator import VideoAgent
from src.pipeline.script_writer import ScriptWriter

app = FastAPI(title="AIGC Platform API", version="1.0.0")

# Agent instance (lazy init)
_agent: Optional[VideoAgent] = None


def get_agent() -> VideoAgent:
    global _agent
    if _agent is None:
        _agent = VideoAgent(
            asset_dir=os.environ.get("AIGC_ASSETS_DIR", "./assets"),
            output_dir=os.environ.get("AIGC_OUTPUT_DIR", "./output"),
        )
    return _agent


# ── Request/Response Models ─────────────────────────────────────────

class ProjectRequest(BaseModel):
    source: str = "template"  # template, topic, json
    value: str = "nursing-ad"


class GenerateResponse(BaseModel):
    status: str
    total: int = 0
    success: int = 0
    failed: int = 0
    elapsed_s: float = 0.0
    results: dict = {}


class EvalRequest(BaseModel):
    video_path: str


class EvalResponse(BaseModel):
    quality_score: float
    duration_s: float
    resolution: list = [0, 0]
    file_size_kb: float
    has_audio: bool
    errors: list = []


# ── Routes ───────────────────────────────────────────────────────────

@app.get("/")
def root():
    return {
        "name": "AIGC Platform API",
        "version": "1.0.0",
        "endpoints": [
            "GET  /api/templates",
            "POST /api/project",
            "POST /api/generate",
            "POST /api/compose",
            "POST /api/evaluate",
            "POST /api/e2e",
        ],
    }


@app.get("/api/templates")
def list_templates():
    writer = ScriptWriter()
    return {"templates": writer.list_templates()}


@app.post("/api/project")
def create_project(req: ProjectRequest):
    agent = get_agent()
    try:
        if req.source == "template":
            project = agent.from_template(req.value)
        elif req.source == "topic":
            project = agent.from_topic(req.value)
        else:
            raise HTTPException(400, f"Unknown source: {req.source}")
    except ValueError as e:
        raise HTTPException(404, str(e))

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


@app.post("/api/generate")
def generate_assets(project_json: Optional[str] = None):
    agent = get_agent()
    if project_json:
        try:
            project = ScriptWriter.from_json(project_json)
            agent.current_project = project
        except Exception as e:
            raise HTTPException(400, f"Invalid project JSON: {e}")

    try:
        result = agent.generate()
    except ValueError as e:
        raise HTTPException(400, str(e))

    return GenerateResponse(
        status="completed" if result["failed"] == 0 else "partial",
        **result,
    )


@app.post("/api/compose")
def compose_video(output_filename: Optional[str] = None):
    agent = get_agent()
    try:
        path = agent.compose(output_filename=output_filename)
    except (ValueError, FileNotFoundError, RuntimeError) as e:
        raise HTTPException(400, str(e))

    return {"status": "completed", "video_path": path}


@app.post("/api/evaluate")
def evaluate_video(req: EvalRequest):
    if not os.path.exists(req.video_path):
        raise HTTPException(404, f"Video not found: {req.video_path}")

    agent = get_agent()
    result = agent.evaluate(req.video_path)
    return EvalResponse(
        quality_score=result["quality_score"],
        duration_s=result["duration_s"],
        resolution=list(result["resolution"]),
        file_size_kb=result["file_size_kb"],
        has_audio=result["has_audio"],
        errors=result["errors"],
    )


@app.post("/api/e2e")
def end_to_end(req: ProjectRequest,
               output_filename: Optional[str] = None):
    """Full pipeline: create -> generate -> compose -> evaluate."""
    agent = get_agent()
    try:
        result = agent.run_end_to_end(
            source=req.source, value=req.value,
            output_filename=output_filename,
        )
    except Exception as e:
        raise HTTPException(500, f"Pipeline failed: {e}")

    return result


@app.get("/api/download/{video_name:path}")
def download_video(video_name: str):
    """Download a generated video."""
    agent = get_agent()
    # Path traversal protection
    safe_name = os.path.basename(video_name)
    safe_path = os.path.normpath(os.path.join(agent.output_dir, safe_name))
    if not safe_path.startswith(os.path.normpath(agent.output_dir)):
        raise HTTPException(403, "Access denied")
    video_path = safe_path
    if not os.path.exists(video_path):
        raise HTTPException(404, f"Video not found: {safe_name}")
    return FileResponse(video_path, media_type="video/mp4")


@app.get("/api/agent/log")
def get_agent_log():
    agent = get_agent()
    return {"log": agent.log}


@app.get("/api/agent/summary")
def get_agent_summary():
    agent = get_agent()
    return {"summary": agent.summary()}


# ── Multi-modal understanding endpoints ────────────────────────────


class ImageRequest(BaseModel):
    image_path: str


class VisualQARequest(BaseModel):
    image_path: str
    question: str


class RAGSearchRequest(BaseModel):
    query: str
    top_k: int = 10
    rerank: bool = True


class RAGIndexRequest(BaseModel):
    directory: str = "./assets"


@app.post("/api/understand", tags=["Multi-modal Understanding"])
def understand_image(req: ImageRequest):
    """Analyze an image using Qwen2-VL: description + scene analysis."""
    if not os.path.exists(req.image_path):
        raise HTTPException(404, f"Image not found: {req.image_path}")

    from src.understanding import QwenVLEngine
    engine = QwenVLEngine()
    return {
        "description": engine.describe_image(req.image_path),
        "analysis": engine.analyze_scene(req.image_path),
    }


@app.post("/api/visual-qa", tags=["Multi-modal Understanding"])
def visual_qa(req: VisualQARequest):
    """Answer a question about an image using Qwen2-VL."""
    if not os.path.exists(req.image_path):
        raise HTTPException(404, f"Image not found: {req.image_path}")

    from src.understanding import QwenVLEngine
    engine = QwenVLEngine()
    return {
        "question": req.question,
        "answer": engine.answer_question(req.image_path, req.question),
    }


@app.post("/api/rag/search", tags=["RAG"])
def rag_search(req: RAGSearchRequest):
    """Cross-modal search: find images by text query using Visual RAG.

    Uses CLIP embeddings + FAISS index + cross-encoder reranking.
    """
    from src.rag.visual_rag import VisualRAG
    rag = VisualRAG()
    if os.path.exists("./assets"):
        rag.index_directory("./assets")

    results = rag.search(req.query, top_k=req.top_k, rerank=req.rerank)
    return {
        "query": req.query,
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


@app.post("/api/rag/index", tags=["RAG"])
def rag_index(req: RAGIndexRequest):
    """Index a directory of images for cross-modal search."""
    from src.rag.visual_rag import VisualRAG
    rag = VisualRAG()
    rag.index_directory(req.directory)
    return {"status": "indexed", "items": len(rag.items), "directory": req.directory}


@app.get("/api/rag/stats", tags=["RAG"])
def rag_stats():
    """Get RAG index statistics."""
    from src.rag.visual_rag import VisualRAG
    rag = VisualRAG()
    return rag.stats
