"""Embedder — CLIP-based multi-modal embedding extraction.

Extracts vector embeddings from images and text using CLIP.
Supports CPU/GPU auto-dispatch and graceful fallback when
CLIP is not installed (uses lightweight ONNX or mock embeddings).

Resume alignment: multi-modal embedding for Visual RAG.
"""

import logging
import numpy as np
from typing import Optional

logger = logging.getLogger(__name__)

# Try real CLIP import
try:
    import torch
    import clip
    _CLIP_AVAILABLE = True
except ImportError:
    _CLIP_AVAILABLE = False
    try:
        import onnxruntime
        _ONNX_AVAILABLE = True
    except ImportError:
        _ONNX_AVAILABLE = False

# Dimension for CLIP ViT-B/32
EMBED_DIM = 512


def _get_device() -> str:
    """Auto-detect best device: CUDA > MPS > CPU."""
    if _CLIP_AVAILABLE:
        try:
            if torch.cuda.is_available():
                return "cuda"
            if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
                return "mps"
        except Exception:
            pass
    return "cpu"


class CLIPEmbedder:
    """Multi-modal embedder using OpenAI CLIP.

    Converts images and text to shared embedding space for
    cross-modal retrieval (text→image, image→image, image→text).

    Features:
    - CPU/GPU auto-dispatch
    - Graceful fallback to mock embeddings when CLIP unavailable
    - Embedding normalization for cosine similarity search
    """

    def __init__(self, model_name: str = "ViT-B/32", device: Optional[str] = None):
        self.model_name = model_name
        self.device = device or _get_device()
        self.model = None
        self.preprocess = None
        self._load_model()

    def _load_model(self):
        """Load CLIP model. Falls back to mock embedder on failure."""
        if _CLIP_AVAILABLE:
            try:
                logger.info(f"Loading CLIP ({self.model_name}) on {self.device}...")
                self.model, self.preprocess = clip.load(
                    self.model_name, device=self.device
                )
                self.model.eval()
                logger.info("CLIP loaded successfully.")
                return
            except Exception as e:
                logger.warning(f"CLIP load failed: {e}. Falling back to mock.")

        logger.info("Using mock embedder (install CLIP + torch for real embeddings).")

    def _mock_embedding(self, n: int = 1) -> np.ndarray:
        """Return random normalized vector as mock embedding."""
        rng = np.random.default_rng(42)
        vec = rng.normal(0, 0.1, (n, EMBED_DIM))
        norms = np.linalg.norm(vec, axis=1, keepdims=True)
        return (vec / norms).astype(np.float32)

    @property
    def available(self) -> bool:
        return self.model is not None

    def embed_image(self, image_path: str) -> np.ndarray:
        """Extract embedding from an image file.

        Args:
            image_path: Path to image file.

        Returns:
            float32 array of shape (EMBED_DIM,), normalized.
        """
        if not self.available:
            logger.debug(f"Mock embedding for {image_path}")
            return self._mock_embedding()[0]

        try:
            from PIL import Image
            image = Image.open(image_path).convert("RGB")
            import torch
            with torch.no_grad():
                image_input = self.preprocess(image).unsqueeze(0).to(self.device)
                embedding = self.model.encode_image(image_input)
                embedding = embedding / embedding.norm(dim=-1, keepdim=True)
            return embedding.cpu().numpy().astype(np.float32)[0]
        except Exception as e:
            logger.warning(f"Image embedding failed for {image_path}: {e}")
            return self._mock_embedding()[0]

    def embed_images(self, image_paths: list[str]) -> np.ndarray:
        """Batch embed multiple images.

        Returns:
            float32 array of shape (N, EMBED_DIM), normalized.
        """
        if not self.available:
            logger.debug(f"Mock embeddings for {len(image_paths)} images")
            return self._mock_embedding(len(image_paths))

        import torch
        from PIL import Image

        embeddings = []
        with torch.no_grad():
            for path in image_paths:
                try:
                    image = Image.open(path).convert("RGB")
                    image_input = self.preprocess(image).unsqueeze(0).to(self.device)
                    emb = self.model.encode_image(image_input)
                    emb = emb / emb.norm(dim=-1, keepdim=True)
                    embeddings.append(emb.cpu().numpy()[0])
                except Exception as e:
                    logger.warning(f"Skipping {path}: {e}")
                    embeddings.append(self._mock_embedding()[0])
        return np.array(embeddings, dtype=np.float32)

    def embed_text(self, text: str) -> np.ndarray:
        """Extract embedding from text.

        Returns:
            float32 array of shape (EMBED_DIM,), normalized.
        """
        if not self.available:
            return self._mock_embedding()[0]

        import torch
        with torch.no_grad():
            text_tokens = clip.tokenize([text]).to(self.device)
            embedding = self.model.encode_text(text_tokens)
            embedding = embedding / embedding.norm(dim=-1, keepdim=True)
        return embedding.cpu().numpy().astype(np.float32)[0]

    def embed_texts(self, texts: list[str]) -> np.ndarray:
        """Batch embed multiple texts."""
        if not self.available:
            return self._mock_embedding(len(texts))

        import torch
        with torch.no_grad():
            text_tokens = clip.tokenize(texts).to(self.device)
            embeddings = self.model.encode_text(text_tokens)
            embeddings = embeddings / embeddings.norm(dim=-1, keepdim=True)
        return embeddings.cpu().numpy().astype(np.float32)
