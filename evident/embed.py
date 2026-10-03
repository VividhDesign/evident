"""Bi-encoder embeddings with an on-disk cache (embedding a corpus is the slowest step)."""

from __future__ import annotations

import hashlib
import re

import numpy as np

from . import config


def best_device() -> str:
    import torch

    if torch.backends.mps.is_available():
        return "mps"
    return "cuda" if torch.cuda.is_available() else "cpu"


class Embedder:
    def __init__(self, model_name: str = config.EMBED_MODEL, device: str | None = None, batch_size: int = 64):
        from sentence_transformers import SentenceTransformer

        self.model_name = model_name
        self.device = device or best_device()
        self.model = SentenceTransformer(model_name, device=self.device)
        self.batch_size = batch_size
        # BGE v1.5 models expect this instruction on queries (not on passages).
        self.query_prefix = "Represent this sentence for searching relevant passages: " if "bge" in model_name.lower() else ""
        get_dim = getattr(self.model, "get_embedding_dimension", None) or self.model.get_sentence_embedding_dimension
        self.dim = get_dim()

    def _encode(self, texts: list[str], progress: bool = False) -> np.ndarray:
        return self.model.encode(texts, batch_size=self.batch_size, normalize_embeddings=True,
                                 convert_to_numpy=True, show_progress_bar=progress).astype(np.float32)

    def embed_documents(self, texts: list[str], cache: bool = True) -> np.ndarray:
        if not cache:
            return self._encode(texts, progress=len(texts) > 1000)
        h = hashlib.sha1(self.model_name.encode())
        for t in texts:
            h.update(t.encode())
            h.update(b"\0")
        slug = re.sub(r"[^A-Za-z0-9]+", "_", self.model_name)
        path = config.CACHE_DIR / "embeddings" / f"{slug}-{len(texts)}-{h.hexdigest()[:16]}.npy"
        if path.exists():
            return np.load(path)
        vectors = self._encode(texts, progress=len(texts) > 1000)
        path.parent.mkdir(parents=True, exist_ok=True)
        np.save(path, vectors)
        return vectors

    def embed_queries(self, texts: list[str]) -> np.ndarray:
        return self._encode([self.query_prefix + t for t in texts])
