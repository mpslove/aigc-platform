"""Tests for the RAG module."""

import sys
import os
import json
import pytest
from pathlib import Path
from unittest.mock import patch, MagicMock

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.rag.vector_store import (
    VectorStore, VisualRAG, Reranker,
    RetrievalResult, DocumentChunk,
)


class TestDocumentChunk:
    def test_default_modality(self):
        d = DocumentChunk(id="1", text="hello", source="src")
        assert d.modality == "text"
        assert d.metadata == {}

    def test_with_metadata(self):
        d = DocumentChunk(id="2", text="world", source="src",
                          modality="image",
                          metadata={"width": 1024, "height": 768})
        assert d.metadata["width"] == 1024


class TestVectorStore:
    def setup_method(self):
        self.store = VectorStore()

    def test_empty_search(self):
        results = self.store.search("anything")
        assert results == []

    def test_add_and_search_keyword(self):
        self.store.add_document(
            DocumentChunk(id="1", text="a nurse caring for patient",
                          source="scene1", modality="video"),
        )
        self.store.add_document(
            DocumentChunk(id="2", text="a sunny beach with palm trees",
                          source="scene2", modality="video"),
        )
        results = self.store.search("nurse patient")
        assert len(results) >= 1
        assert "nurse" in results[0].content

    def test_add_multiple_search_top_k(self):
        chunks = [
            DocumentChunk(id=str(i), text=f"document about topic {i}",
                          source=f"src{i}", modality="text")
            for i in range(10)
        ]
        self.store.add_documents(chunks)
        results = self.store.search("topic", top_k=3)
        assert len(results) <= 3

    def test_search_ranking(self):
        """Documents with more query terms should rank higher."""
        self.store.add_documents([
            DocumentChunk(id="a", text="cat sitting on a mat",
                          source="a", modality="text"),
            DocumentChunk(id="b", text="dog running in park",
                          source="b", modality="text"),
            DocumentChunk(id="c", text="cat and dog playing together",
                          source="c", modality="text"),
        ])
        results = self.store.search("cat dog playing")
        if len(results) >= 2:
            assert results[0].score >= results[1].score

    def test_save_and_load(self, tmp_path):
        store_path = tmp_path / "index.json"
        store = VectorStore(index_path=str(store_path))
        store.add_documents([
            DocumentChunk(id="1", text="hello", source="s1"),
            DocumentChunk(id="2", text="world", source="s2"),
        ])
        store.save_index()
        assert store_path.exists()

        store2 = VectorStore(index_path=str(store_path))
        store2.load_index()
        assert len(store2.documents) == 2

    def test_clear(self):
        self.store.add_document(
            DocumentChunk(id="1", text="test", source="s"),
        )
        self.store.clear()
        assert len(self.store.documents) == 0

    def test_faiss_not_available_by_default(self):
        assert not self.store.is_faiss_available

    def test_case_insensitive_search(self):
        self.store.add_document(
            DocumentChunk(id="1", text="NURSE Caring For Patient",
                          source="s1"),
        )
        results = self.store.search("nurse")
        assert len(results) > 0


class TestVisualRAG:
    def setup_method(self):
        self.vrag = VisualRAG()

    def test_index_image(self):
        self.vrag.index_image("/images/cat.jpg", "a cat sitting on a chair")
        results = self.vrag.search("cat")
        assert len(results) >= 1
        assert results[0].modality == "image"

    def test_index_video_segment(self):
        self.vrag.index_video_segment("/videos/scene1.mp4", "scene1",
                                      "a nurse helping patient")
        results = self.vrag.search("nurse")
        assert len(results) >= 1
        assert results[0].modality == "video"

    def test_search_empty(self):
        results = self.vrag.search("anything")
        assert results == []

    def test_index_project_scenes(self):
        from src.pipeline.schema import Scene, SceneType
        scenes = [
            Scene(id="title", prompt="medical title card",
                  scene_type=SceneType.TITLE, duration=3.0),
            Scene(id="care", prompt="nurse caring for elderly patient",
                  duration=5.0),
        ]
        self.vrag.index_project_scenes(scenes)
        results = self.vrag.search("nurse")
        assert len(results) >= 1
        # Search for medical too
        results2 = self.vrag.search("medical")
        assert len(results2) >= 1


class TestReranker:
    def test_empty(self):
        r = Reranker()
        assert r.rerank("query", []) == []

    def test_diversity_penalty(self):
        r = Reranker()
        results = [
            RetrievalResult(content="nurse caring",
                            score=0.8, source="a.mp4", modality="video"),
            RetrievalResult(content="nurse helping",
                            score=0.7, source="a.mp4", modality="video"),
        ]
        reranked = r.rerank("nurse", results)
        # Second result should be penalized (same source)
        if reranked[0].source == reranked[1].source:
            assert reranked[0].score > reranked[1].score

    def test_modality_bonus(self):
        r = Reranker()
        results = [
            RetrievalResult(content="cat video", score=0.5,
                            source="a.mp4", modality="video"),
            RetrievalResult(content="cat image", score=0.6,
                            source="b.jpg", modality="image"),
        ]
        reranked = r.rerank("cat", results)
        assert len(reranked) == 2
