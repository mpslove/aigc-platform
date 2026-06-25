"""RAG: Multi-modal retrieval augmented generation.

Provides:
  - CLIPEmbedder — text/image embedding extraction (CPU/GPU)
  - VectorStore — FAISS-based embedding storage and retrieval
  - VisualRAG — end-to-end multi-modal retrieval pipeline
  - Reranker — cross-encoder reranking for precision
"""
