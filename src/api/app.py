"""FastAPI REST API for the AIGC platform.

Provides:
  - GET  /api/templates      — list available templates
  - POST /api/project        — create a project from template/topic
  - POST /api/generate       — generate assets
  - POST /api/compose        — compose final video
  - GET  /api/status/{job}   — check generation status
  - POST /api/evaluate       -- evaluate video quality
  - POST /api/e2e            -- full end-to-end pipeline
  - POST /api/e2e-react      -- ReAct loop (LLM-driven decisions)
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
import uuid
import threading
from pathlib import Path
from typing import Optional
from enum import Enum

# Allow running directly
sys.path.insert(0, str(Path(__file__).parent.parent))

from fastapi import FastAPI, HTTPException, BackgroundTasks
from fastapi.responses import FileResponse
from pydantic import BaseModel

from src.agent.orchestrator import VideoAgent
from src.pipeline.script_writer import ScriptWriter

app = FastAPI(title="AIGC Platform API", version="1.0.0")

# ── Singleton Managers ──────────────────────────────────────────────

_agent: Optional[VideoAgent] = None
_agent_lock = threading.Lock()

# RAG singleton (with disk persistence)
_rag = None
_rag_lock = threading.Lock()
_rag_index_path = os.environ.get("AIGC_RAG_INDEX_PATH", "./.cache/rag_index")

# QwenVL singleton
_qwen_engine = None
_qwen_lock = threading.Lock()


def get_agent() -> VideoAgent:
    """Thread-safe agent singleton."""
    global _agent
    if _agent is not None:
        return _agent
    with _agent_lock:
        if _agent is not None:
            return _agent
        _agent = VideoAgent(
            asset_dir=os.environ.get("AIGC_ASSETS_DIR", "./assets"),
            output_dir=os.environ.get("AIGC_OUTPUT_DIR", "./output"),
        )
        return _agent


def get_rag():
    """Thread-safe RAG singleton with disk persistence.

    On first call: load from disk cache if available, else create fresh.
    On subsequent calls: return the same instance with incremental index.
    """
    global _rag
    if _rag is not None:
        return _rag
    with _rag_lock:
        if _rag is not None:
            return _rag
        from src.rag.visual_rag import VisualRAG
        # Try loading from disk cache
        if os.path.exists(_rag_index_path + ".json"):
            _rag = VisualRAG(index_path=_rag_index_path)
            return _rag
        _rag = VisualRAG()
        return _rag


def get_qwen_engine():
    """Thread-safe QwenVLEngine singleton.

    Avoids re-creating QwenVLEngine on every request (which causes
    cold GPU/CPU model loading each time).
    """
    global _qwen_engine
    if _qwen_engine is not None:
        return _qwen_engine
    with _qwen_lock:
        if _qwen_engine is not None:
            return _qwen_engine
        from src.understanding import QwenVLEngine
        _qwen_engine = QwenVLEngine()
        return _qwen_engine


def _ensure_rag_indexed():
    """Ensure RAG has the assets directory indexed (incremental)."""
    rag = get_rag()
    asset_dir = os.environ.get("AIGC_ASSETS_DIR", "./assets")
    if os.path.exists(asset_dir):
        new = rag.index_directory(asset_dir, incremental=True)
        if new > 0:
            # Persist updated index to disk
            try:
                os.makedirs(os.path.dirname(_rag_index_path) or ".", exist_ok=True)
                rag.save(_rag_index_path)
            except Exception:
                pass  # Non-critical: save failure shouldn't block search


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


class ReActRequest(BaseModel):
    request: str
    max_steps: int = 12


# ── Async Job Tracker ───────────────────────────────────────────────

class JobStatus(str, Enum):
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"


class JobInfo(BaseModel):
    job_id: str
    status: JobStatus = JobStatus.PENDING
    created_at: float = 0.0
    finished_at: Optional[float] = None
    result: Optional[dict] = None
    error: Optional[str] = None


# In-memory job store (thread-safe dict)
_jobs: dict[str, JobInfo] = {}
_jobs_lock = threading.Lock()


def _create_job() -> str:
    """Create a new job entry, return job_id."""
    job_id = uuid.uuid4().hex[:12]
    job = JobInfo(job_id=job_id, created_at=time.time())
    with _jobs_lock:
        _jobs[job_id] = job
    return job_id


def _update_job(job_id: str, status: JobStatus,
                result: Optional[dict] = None, error: Optional[str] = None):
    with _jobs_lock:
        if job_id in _jobs:
            _jobs[job_id].status = status
            _jobs[job_id].finished_at = time.time()
            _jobs[job_id].result = result
            _jobs[job_id].error = error


def _run_e2e_job(job_id: str, source: str, value: str,
                output_filename: Optional[str] = None):
    """Background worker for e2e pipeline."""
    try:
        _update_job(job_id, JobStatus.RUNNING)
        agent = get_agent()
        result = agent.run_end_to_end(
            source=source, value=value,
            output_filename=output_filename,
        )
        _update_job(job_id, JobStatus.COMPLETED, result=result)
    except Exception as e:
        _update_job(job_id, JobStatus.FAILED, error=str(e))


def _run_react_job(job_id: str, request: str, max_steps: int):
    """Background worker for ReAct loop."""
    try:
        _update_job(job_id, JobStatus.RUNNING)
        agent = get_agent()
        result = agent.run(user_request=request, max_steps=max_steps)
        _update_job(job_id, JobStatus.COMPLETED, result=result)
    except Exception as e:
        _update_job(job_id, JobStatus.FAILED, error=str(e))


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
            "POST /api/e2e-react",
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


@app.post("/api/e2e-react")
def end_to_end_react(req: ReActRequest):
    """ReAct loop: LLM decides each step with tool selection + error recovery."""
    agent = get_agent()
    try:
        result = agent.run(
            user_request=req.request,
            max_steps=req.max_steps,
        )
    except Exception as e:
        raise HTTPException(500, f"ReAct loop failed: {e}")

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

    engine = get_qwen_engine()
    return {
        "description": engine.describe_image(req.image_path),
        "analysis": engine.analyze_scene(req.image_path),
    }


@app.post("/api/visual-qa", tags=["Multi-modal Understanding"])
def visual_qa(req: VisualQARequest):
    """Answer a question about an image using Qwen2-VL."""
    if not os.path.exists(req.image_path):
        raise HTTPException(404, f"Image not found: {req.image_path}")

    engine = get_qwen_engine()
    return {
        "question": req.question,
        "answer": engine.answer_question(req.image_path, req.question),
    }


@app.post("/api/rag/search", tags=["RAG"])
def rag_search(req: RAGSearchRequest):
    """Cross-modal search: find images by text query using Visual RAG.

    Uses CLIP embeddings + FAISS index + cross-encoder reranking.
    RAG index is cached at app level with incremental updates.
    """
    rag = get_rag()
    _ensure_rag_indexed()

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
    """Index a directory of images for cross-modal search.

    Incremental: only adds new files not already indexed.
    Persists index to disk after indexing.
    """
    rag = get_rag()
    new_count = rag.index_directory(req.directory, incremental=True)
    # Persist to disk
    try:
        os.makedirs(os.path.dirname(_rag_index_path) or ".", exist_ok=True)
        rag.save(_rag_index_path)
    except Exception:
        pass
    return {
        "status": "indexed",
        "total_items": len(rag.items),
        "new_items": new_count,
        "directory": req.directory,
    }


@app.get("/api/rag/stats", tags=["RAG"])
def rag_stats():
    """Get RAG index statistics."""
    rag = get_rag()
    return rag.stats


# ── Async Job Endpoints ─────────────────────────────────────────────

@app.post("/api/e2e/async", tags=["Async Jobs"])
def e2e_async(req: ProjectRequest, bg: BackgroundTasks,
              output_filename: Optional[str] = None):
    """Submit end-to-end pipeline as background job. Returns job_id immediately."""
    job_id = _create_job()
    bg.add_task(_run_e2e_job, job_id, req.source, req.value, output_filename)
    return {"job_id": job_id, "status": "pending"}


@app.post("/api/e2e-react/async", tags=["Async Jobs"])
def e2e_react_async(req: ReActRequest, bg: BackgroundTasks):
    """Submit ReAct loop as background job. Returns job_id immediately."""
    job_id = _create_job()
    bg.add_task(_run_react_job, job_id, req.request, req.max_steps)
    return {"job_id": job_id, "status": "pending"}


@app.get("/api/jobs/{job_id}", tags=["Async Jobs"])
def get_job_status(job_id: str):
    """Check status of a background job."""
    with _jobs_lock:
        job = _jobs.get(job_id)
    if not job:
        raise HTTPException(404, f"Job not found: {job_id}")
    resp = {"job_id": job.job_id, "status": job.status.value,
            "created_at": job.created_at}
    if job.finished_at:
        resp["finished_at"] = job.finished_at
        resp["elapsed_s"] = round(job.finished_at - job.created_at, 2)
    if job.result:
        resp["result"] = job.result
    if job.error:
        resp["error"] = job.error
    return resp


@app.get("/api/jobs", tags=["Async Jobs"])
def list_jobs():
    """List all background jobs."""
    with _jobs_lock:
        return [
            {
                "job_id": j.job_id,
                "status": j.status.value,
                "created_at": j.created_at,
                "finished_at": j.finished_at,
            }
            for j in _jobs.values()
        ]
