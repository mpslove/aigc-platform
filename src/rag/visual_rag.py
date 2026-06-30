"""Visual RAG — End-to-end multi-modal retrieval pipeline.

Orchestrates embedding extraction → FAISS indexing → retrieval → reranking
to enable cross-modal search: text→image, image→image, image→text.

Pipeline:
  Image → CLIP Embedder → FAISS Index → Top-K Search → Reranker → Results

Resume alignment: 基于FAISS构建多模态向量库，搭配Rerank重排序搭建Visual RAG闭环。
"""

import json
import logging
import os
from dataclasses import dataclass
import numpy as np
from pathlib import Path
from typing import Optional

from src.rag.embedder import CLIPEmbedder, EMBED_DIM
from src.rag.reranker import Reranker

logger = logging.getLogger(__name__)

try:
    import faiss
    _FAISS_AVAILABLE = True
except ImportError:
    _FAISS_AVAILABLE = False
    logger.warning("FAISS not installed. Using brute-force numpy search (slow).")


@dataclass
class IndexedItem:
    """A single item in the vector index."""
    id: str
    path: str
    caption: str
    modality: str  # image, video_frame, text
    metadata: dict  # extra info (size, tags, source_url, etc.)


class VisualRAG:
    """Multi-modal RAG pipeline: index images/videos → search by text or image.

    Usage:
        rag = VisualRAG()
        rag.index_directory("./assets")
        results = rag.search("a cat on the beach", top_k=5)
    """

    def __init__(
        self,
        embedder: Optional[CLIPEmbedder] = None,
        reranker: Optional[Reranker] = None,
        index_path: Optional[str] = None,
    ):
        self.embedder = embedder or CLIPEmbedder()
        self.reranker = reranker or Reranker()
        self.items: list[IndexedItem] = []
        self.index: Optional[faiss.Index] = None
        self._dim = EMBED_DIM

        if index_path and os.path.exists(index_path):
            self.load(index_path)

    def add_item(
        self,
        id: str,
        path: str,
        caption: str = "",
        modality: str = "image",
        metadata: Optional[dict] = None,
        embedding: Optional[np.ndarray] = None,
    ):
        """Add a single item to the index."""
        if embedding is None and os.path.exists(path):
            embedding = self.embedder.embed_image(path)
        elif embedding is None:
            embedding = self.embedder._mock_embedding()[0]

        item = IndexedItem(
            id=id,
            path=path,
            caption=caption,
            modality=modality,
            metadata=metadata or {},
        )
        self.items.append(item)

        # Build or extend FAISS index
        emb = embedding.reshape(1, -1).astype(np.float32)
        if self.index is None:
            if _FAISS_AVAILABLE:
                self.index = faiss.IndexFlatIP(self._dim)
            else:
                self.index = _NumpyIndex(self._dim)
            self.index.add(emb)
        else:
            self.index.add(emb)

    def index_directory(
        self,
        dir_path: str,
        recursive: bool = True,
        image_extensions: tuple = (".jpg", ".jpeg", ".png", ".webp", ".bmp"),
        incremental: bool = True,
    ) -> int:
        """Index all images in a directory.

        Args:
            incremental: If True, only index files not already in self.items.
                         If False, re-index everything (slower, full rebuild).

        Returns:
            Number of newly indexed items.
        """
        path = Path(dir_path)
        if not path.exists():
            logger.warning(f"Directory not found: {dir_path}")
            return 0

        pattern = "**/*" if recursive else "*"
        files = [f for f in path.glob(pattern)
                 if f.suffix.lower() in image_extensions and f.is_file()]

        # Incremental: skip already-indexed paths
        if incremental and self.items:
            existing_paths = {it.path for it in self.items}
            files = [f for f in files if str(f) not in existing_paths]

        if not files:
            logger.info(f"No new images to index in {dir_path}.")
            return 0

        logger.info(f"Indexing {len(files)} new images from {dir_path}...")
        for f in files:
            self.add_item(
                id=f.name,
                path=str(f),
                caption=f.stem.replace("_", " ").replace("-", " "),
                modality="image",
                metadata={"size_kb": f.stat().st_size / 1024},
            )

        logger.info(f"Index: {len(self.items)} items ({len(files)} new).")
        return len(files)

    def search(
        self,
        query: str,
        top_k: int = 10,
        rerank: bool = True,
    ) -> list[dict]:
        """Search index by text query.

        Steps:
          1. Embed query text → FAISS search → top-K candidates
          2. (Optional) Rerank candidates with cross-encoder
          3. Return sorted results with scores

        Args:
            query: Text query (natural language).
            top_k: Number of results to return.
            rerank: Whether to apply cross-encoder reranking.

        Returns:
            List of dicts with keys: id, path, caption, score, rerank_score, ...
        """
        if not self.items or self.index is None:
            return []

        # Step 1: Embed query
        query_emb = self.embedder.embed_text(query).reshape(1, -1).astype(np.float32)
        k = min(top_k * 3, len(self.items))  # Retrieve more for reranking
        scores, indices = self.index.search(query_emb, k)

        # Step 2: Build candidate list
        candidates = []
        for score, idx in zip(scores[0], indices[0]):
            if idx < 0 or idx >= len(self.items):
                continue
            item = self.items[idx]
            candidates.append({
                "id": item.id,
                "path": item.path,
                "caption": item.caption,
                "modality": item.modality,
                "metadata": item.metadata,
                "score": float(score),
            })

        # Step 3: Rerank
        if rerank and candidates:
            candidates = self.reranker.rerank(query, candidates, text_key="caption")

        return candidates[:top_k]

    def search_by_image(
        self,
        image_path: str,
        top_k: int = 10,
    ) -> list[dict]:
        """Search index by image (image→image retrieval)."""
        if not self.items or self.index is None:
            return []
        query_emb = self.embedder.embed_image(image_path).reshape(1, -1).astype(np.float32)
        k = min(top_k, len(self.items))
        scores, indices = self.index.search(query_emb, k)

        results = []
        for score, idx in zip(scores[0], indices[0]):
            if idx < 0 or idx >= len(self.items):
                continue
            item = self.items[idx]
            results.append({
                "id": item.id,
                "path": item.path,
                "caption": item.caption,
                "modality": item.modality,
                "score": float(score),
            })
        return results

    def save(self, path: str):
        """Save index and metadata to disk."""
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        data = {
            "items": [
                {
                    "id": it.id,
                    "path": it.path,
                    "caption": it.caption,
                    "modality": it.modality,
                    "metadata": it.metadata,
                }
                for it in self.items
            ],
        }
        with open(path + ".json", "w") as f:
            json.dump(data, f, indent=2)
        if _FAISS_AVAILABLE and self.index is not None:
            faiss.write_index(self.index, path + ".faiss")
        logger.info(f"Index saved to {path}")

    def load(self, path: str):
        """Load index and metadata from disk."""
        json_path = path + ".json"
        faiss_path = path + ".faiss"
        if os.path.exists(json_path):
            with open(json_path) as f:
                data = json.load(f)
            self.items = [IndexedItem(**it) for it in data["items"]]
        if _FAISS_AVAILABLE and os.path.exists(faiss_path):
            self.index = faiss.read_index(faiss_path)
        else:
            # Rebuild from items
            embeddings = np.array([
                self.embedder.embed_image(it.path) for it in self.items
            ], dtype=np.float32)
            if _FAISS_AVAILABLE:
                self.index = faiss.IndexFlatIP(self._dim)
            else:
                self.index = _NumpyIndex(self._dim)
            self.index.add(embeddings)
        logger.info(f"Index loaded: {len(self.items)} items.")

    @property
    def stats(self) -> dict:
        """Index statistics."""
        return {
            "total_items": len(self.items),
            "embeddings_available": self.embedder.available,
            "reranker_available": self.reranker.available,
            "faiss_available": _FAISS_AVAILABLE,
            "device": self.embedder.device,
            "dimension": self._dim,
        }


class _NumpyIndex:
    """Brute-force FAISS fallback using numpy dot product.

    Used when FAISS is not installed. Slower but functionally equivalent.
    """
    def __init__(self, dim: int):
        self.dim = dim
        self.vectors: list[np.ndarray] = []
        self.ntotal = 0

    def add(self, embeddings: np.ndarray):
        for i in range(embeddings.shape[0]):
            self.vectors.append(embeddings[i])
        self.ntotal = len(self.vectors)

    def search(self, query: np.ndarray, k: int) -> tuple:
        if not self.vectors:
            return np.array([[]]), np.array([[]])
        stack = np.stack(self.vectors, axis=0)
        scores = query @ stack.T  # dot product
        top_k = min(k, len(self.vectors))
        indices = np.argsort(-scores[0])[:top_k]
        return scores[:, indices], indices.reshape(1, -1)
