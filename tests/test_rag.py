"""Tests for Visual RAG and embedder modules."""
import sys, os, json
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import pytest
import numpy as np
from src.rag.embedder import CLIPEmbedder, EMBED_DIM
from src.rag.visual_rag import VisualRAG, _NumpyIndex
from src.rag.reranker import Reranker


class TestCLIPEmbedder:
    def test_default_init(self):
        embedder = CLIPEmbedder()
        assert embedder is not None
        assert embedder.device in ("cpu", "cuda", "mps")
        assert embedder.available is False  # No CLIP installed in CI

    def test_mock_embedding_shape(self):
        embedder = CLIPEmbedder()
        emb = embedder._mock_embedding()
        assert emb.shape == (1, EMBED_DIM)
        assert emb.dtype == np.float32
        # Check normalization
        norm = np.linalg.norm(emb[0])
        assert abs(norm - 1.0) < 0.01

    def test_mock_embedding_batch(self):
        embedder = CLIPEmbedder()
        embs = embedder._mock_embedding(3)
        assert embs.shape == (3, EMBED_DIM)

    def test_embed_image_mock(self):
        embedder = CLIPEmbedder()
        emb = embedder.embed_image("nonexistent.jpg")
        assert emb.shape == (EMBED_DIM,)

    def test_embed_text_mock(self):
        embedder = CLIPEmbedder()
        emb = embedder.embed_text("a cat on the beach")
        assert emb.shape == (EMBED_DIM,)

    def test_device_detection(self):
        from src.rag.embedder import _get_device
        device = _get_device()
        assert device in ("cpu", "cuda", "mps")


class TestNumpyIndex:
    def test_add_and_search(self):
        idx = _NumpyIndex(EMBED_DIM)
        vec = np.random.randn(1, EMBED_DIM).astype(np.float32)
        vec = vec / np.linalg.norm(vec)
        idx.add(vec)
        assert idx.ntotal == 1

        query = np.random.randn(1, EMBED_DIM).astype(np.float32)
        scores, indices = idx.search(query, 1)
        assert indices.shape[1] == 1

    def test_empty_search(self):
        idx = _NumpyIndex(EMBED_DIM)
        scores, indices = idx.search(np.random.randn(1, EMBED_DIM), 1)
        assert scores.shape[1] == 0


class TestVisualRAG:
    def test_default_init(self):
        rag = VisualRAG()
        assert rag is not None
        assert rag._dim == EMBED_DIM
        assert rag.items == []

    def test_add_item_and_search(self):
        rag = VisualRAG()
        rag.add_item(
            id="test1",
            path="test.jpg",
            caption="test image",
            modality="image",
        )
        assert len(rag.items) == 1
        assert rag.index is not None

    def test_search_with_results(self):
        rag = VisualRAG()
        rag.add_item(id="cat", path="cat.jpg", caption="a cat on the beach")
        rag.add_item(id="dog", path="dog.jpg", caption="a dog in the park")
        results = rag.search("cat", top_k=5)
        assert len(results) > 0
        assert results[0]["caption"] == "a cat on the beach"

    def test_search_image_by_text(self):
        rag = VisualRAG()
        rag.add_item(id="img1", path="test1.jpg", caption="sunset over mountains")
        rag.add_item(id="img2", path="test2.jpg", caption="ocean waves at sunrise")
        results = rag.search("sunrise", top_k=2)
        assert len(results) >= 1

    def test_search_empty_index(self):
        rag = VisualRAG()
        assert rag.search("anything") == []

    def test_stats(self):
        rag = VisualRAG()
        stats = rag.stats
        assert stats["total_items"] == 0
        assert stats["dimension"] == EMBED_DIM


class TestReranker:
    def test_default_init(self):
        reranker = Reranker()
        assert reranker is not None

    def test_rerank_empty(self):
        reranker = Reranker()
        assert reranker.rerank("test", []) == []

    def test_rerank_preserves_order(self):
        reranker = Reranker()
        candidates = [
            {"content": "a cat on the beach", "score": 0.9},
            {"content": "a dog in the park", "score": 0.5},
        ]
        results = reranker.rerank("cat", candidates)
        assert len(results) == 2
        assert "rerank_score" in results[0]

    def test_rerank_with_dict_items(self):
        reranker = Reranker()
        items = [
            {"id": "1", "content": "document about AI"},
            {"id": "2", "content": "document about cooking"},
        ]
        results = reranker.rerank("AI", items)
        assert len(results) == 2
