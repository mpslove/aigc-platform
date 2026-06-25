"""RAG: multi-modal retrieval augmented generation.

Provides:
  - VectorStore — FAISS-based embedding storage and retrieval
  - VisualRAG — image/video frame indexing + retrieval
  - Reranker — re-rank results for quality

Note: this module requires optional dependencies (faiss-cpu, sentence-transformers).
It degrades gracefully with a fallback dict-based store when not available.
"""

import json
import os
import re
from pathlib import Path
from typing import Optional
from dataclasses import dataclass, field

from src.utils.config import find_project_root


@dataclass
class RetrievalResult:
    """Single retrieval result."""
    content: str
    score: float
    source: str              # file path or URL
    modality: str = "text"   # text, image, video


@dataclass
class DocumentChunk:
    """A chunk of indexed content."""
    id: str
    text: str
    source: str
    modality: str = "text"
    metadata: dict = field(default_factory=dict)


class VectorStore:
    """Vector store for embedding retrieval.

    Uses an in-memory dict-based store as a universal fallback.
    When faiss-cpu + sentence-transformers are available, uses
    real vector similarity search.

    Design rationale:
      - Dict fallback works on any system (including no-GPU CI)
      - FAISS backend activated via env flag USE_FAISS=1
      - No hard dependency on heavy ML packages
    """

    def __init__(self, index_path: Optional[str] = None):
        self.index_path = index_path
        self.documents: list[DocumentChunk] = []
        self._faiss_available = False
        self._model = None
        self._index = None
        self._try_init_faiss()

    def _try_init_faiss(self):
        """Try to initialize FAISS + sentence-transformers."""
        if not os.environ.get("USE_FAISS"):
            return
        try:
            import faiss
            from sentence_transformers import SentenceTransformer
            self._model = SentenceTransformer(
                "all-MiniLM-L6-v2", device="cpu",
            )
            self._faiss_available = True
        except ImportError:
            pass

    @property
    def is_faiss_available(self) -> bool:
        return self._faiss_available

    def add_document(self, chunk: DocumentChunk):
        """Add a single document chunk to the store."""
        self.documents.append(chunk)

    def add_documents(self, chunks: list[DocumentChunk]):
        """Add multiple document chunks."""
        self.documents.extend(chunks)

    def search(self, query: str, top_k: int = 5) -> list[RetrievalResult]:
        """Search for documents matching the query."""
        if not self.documents:
            return []

        if self._faiss_available:
            return self._faiss_search(query, top_k)
        return self._text_search(query, top_k)

    def _text_search(self, query: str, top_k: int = 5) -> list[RetrievalResult]:
        """Simple keyword-based fallback search."""
        query_lower = query.lower()
        query_terms = set(query_lower.split())

        scored = []
        for doc in self.documents:
            text_lower = doc.text.lower()
            # Simple TF overlap score
            term_matches = sum(
                1 for t in query_terms if t in text_lower
            )
            if term_matches > 0:
                score = term_matches / max(len(query_terms), 1)
                scored.append(RetrievalResult(
                    content=doc.text,
                    score=score,
                    source=doc.source,
                    modality=doc.modality,
                ))

        # Sort by score descending
        scored.sort(key=lambda x: x.score, reverse=True)
        return scored[:top_k]

    def _faiss_search(self, query: str,
                      top_k: int = 5) -> list[RetrievalResult]:
        """VAISS vector similarity search."""
        import numpy as np
        query_emb = self._model.encode([query])[0]

        if self._index is None:
            # Build index from documents
            if not self.documents:
                return []
            texts = [d.text for d in self.documents]
            embeddings = self._model.encode(texts)
            import faiss
            dim = embeddings.shape[1]
            self._index = faiss.IndexFlatL2(dim)
            self._index.add(np.array(embeddings).astype("float32"))

        distances, indices = self._index.search(
            np.array([query_emb]).astype("float32"), min(top_k, len(self.documents)),
        )

        results = []
        for i, idx in enumerate(indices[0]):
            if idx < 0 or idx >= len(self.documents):
                continue
            doc = self.documents[idx]
            score = 1.0 / (1.0 + distances[0][i])  # convert L2 to similarity
            results.append(RetrievalResult(
                content=doc.text,
                score=float(score),
                source=doc.source,
                modality=doc.modality,
            ))
        return results

    def save_index(self, path: Optional[str] = None):
        """Save document store to JSON."""
        save_path = path or self.index_path
        if not save_path:
            return
        data = [
            {"id": d.id, "text": d.text, "source": d.source,
             "modality": d.modality, "metadata": d.metadata}
            for d in self.documents
        ]
        with open(save_path, "w") as f:
            json.dump(data, f, indent=2, ensure_ascii=False)

    def load_index(self, path: Optional[str] = None):
        """Load document store from JSON."""
        load_path = path or self.index_path
        if not load_path or not os.path.exists(load_path):
            return
        with open(load_path) as f:
            data = json.load(f)
        self.documents = [
            DocumentChunk(**d) for d in data
        ]

    def clear(self):
        """Clear all documents and index."""
        self.documents.clear()
        self._index = None


class VisualRAG:
    """RAG for image/video content using captions as text proxies.

    Since we can't run vision models locally, this stores image captions
    (from APIs or manual annotations) and searches them via VectorStore.
    """

    def __init__(self, store: Optional[VectorStore] = None):
        self.store = store or VectorStore()

    def index_image(self, image_path: str, caption: str,
                    metadata: Optional[dict] = None):
        """Index an image by its caption text."""
        chunk = DocumentChunk(
            id=os.path.basename(image_path),
            text=caption,
            source=image_path,
            modality="image",
            metadata=metadata or {},
        )
        self.store.add_document(chunk)

    def index_video_segment(self, video_path: str, scene_id: str,
                            prompt: str, **metadata):
        """Index a video segment by its generation prompt."""
        chunk = DocumentChunk(
            id=scene_id,
            text=prompt,
            source=video_path,
            modality="video",
            metadata=metadata,
        )
        self.store.add_document(chunk)

    def index_project_scenes(self, scenes: list,
                             asset_dir: str = "./assets"):
        """Index all scenes from a VideoProject."""
        for scene in scenes:
            asset_path = scene.asset_path or f"{asset_dir}/{scene.id}"
            caption = scene.prompt
            self.index_video_segment(
                asset_path, scene.id, caption,
                scene_type=scene.scene_type.value,
                duration=scene.duration,
            )

    def search(self, query: str, top_k: int = 5) -> list[RetrievalResult]:
        """Search indexed visual content."""
        return self.store.search(query, top_k)


class Reranker:
    """Simple re-ranker for retrieval results.

    Re-scores based on:
      - Query term overlap (boost)
      - Source diversity (penalize same source repeatedly)
      - Modality diversity (prefer mix of modalities)
    """

    def rerank(self, query: str,
               results: list[RetrievalResult]) -> list[RetrievalResult]:
        if not results:
            return results

        query_terms = set(query.lower().split())
        seen_sources = set()
        seen_modalities = set()

        for r in results:
            # Boost: term overlap in content
            content_terms = set(r.content.lower().split())
            overlap = len(query_terms & content_terms)
            r.score += overlap * 0.01

            # Penalty: duplicate source
            if r.source in seen_sources:
                r.score *= 0.8
            seen_sources.add(r.source)

            # Bonus: modality diversity (prefer)
            if r.modality not in seen_modalities:
                r.score *= 1.05
            seen_modalities.add(r.modality)

        results.sort(key=lambda x: x.score, reverse=True)
        return results
