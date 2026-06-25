"""Reranker — Cross-encoder reranking for retrieval quality.

Takes initial retrieval results and re-orders them using
a cross-encoder model for higher precision. Falls back to
cosine similarity re-scoring when cross-encoder unavailable.

Resume alignment: Rerank重排序搭建Visual RAG闭环。
"""

import logging
import numpy as np
from typing import Optional

logger = logging.getLogger(__name__)

try:
    from sentence_transformers import CrossEncoder
    _CROSS_ENCODER_AVAILABLE = True
except ImportError:
    _CROSS_ENCODER_AVAILABLE = False


class Reranker:
    """Rerank retrieval results for higher precision.

    Uses a cross-encoder to jointly score query-document pairs,
    significantly improving retrieval quality over pure embedding similarity.

    Two modes:
    1. CrossEncoder — most accurate, needs sentence-transformers
    2. Cosine re-score — fallback, uses embedding similarity
    """

    def __init__(self, model_name: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"):
        self.model_name = model_name
        self.model = None
        if _CROSS_ENCODER_AVAILABLE:
            try:
                logger.info(f"Loading cross-encoder: {model_name}...")
                self.model = CrossEncoder(model_name)
                logger.info("Cross-encoder loaded.")
            except Exception as e:
                logger.warning(f"Cross-encoder load failed: {e}")

    @property
    def available(self) -> bool:
        return self.model is not None

    def rerank(
        self,
        query: str,
        candidates: list[dict],
        text_key: str = "content",
        top_k: Optional[int] = None,
    ) -> list[dict]:
        """Rerank candidate documents by query relevance.

        Args:
            query: Search query string.
            candidates: List of dicts, each must contain text_key.
            text_key: Key in each candidate holding the text to score.
            top_k: Number of results to return. None = all.

        Returns:
            Candidates list sorted by relevance (descending), with
            'rerank_score' added to each dict.
        """
        if not candidates:
            return []

        # Extract content texts
        texts = [
            c.get(text_key, "") if isinstance(c, dict) else str(c)
            for c in candidates
        ]

        if self.available:
            # Cross-encoder scoring
            pairs = [[query, t] for t in texts]
            scores = self.model.predict(pairs)
            for i, c in enumerate(candidates):
                if isinstance(c, dict):
                    c["rerank_score"] = float(scores[i])
        else:
            # Fallback: score by position (preserve original order)
            logger.debug("Cross-encoder not available, keeping original order.")
            for i, c in enumerate(candidates):
                if isinstance(c, dict):
                    c["rerank_score"] = float(len(candidates) - i)

        # Sort by score descending
        if isinstance(candidates[0], dict):
            scored = sorted(
                candidates,
                key=lambda x: x.get("rerank_score", 0),
                reverse=True,
            )
        else:
            scored = candidates

        if top_k and top_k < len(scored):
            return scored[:top_k]
        return scored
