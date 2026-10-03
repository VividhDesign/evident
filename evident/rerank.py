"""Cross-encoder reranking: reads (query, passage) together, so it is far more accurate than
the bi-encoder but too slow to run over the whole corpus. We apply it to the top candidates."""

from __future__ import annotations

from typing import Sequence

import numpy as np

from . import config
from .embed import best_device


class Reranker:
    def __init__(self, model: str = config.DEFAULT_RERANKER, device: str | None = None, max_length: int = 512,
                 batch_size: int = 64):
        from sentence_transformers import CrossEncoder

        self.model_name = config.RERANK_MODELS.get(model, model)
        self.model = CrossEncoder(self.model_name, device=device or best_device(), max_length=max_length)
        self.batch_size = batch_size

    def score(self, pairs: Sequence[tuple[str, str]], progress: bool = False) -> np.ndarray:
        if not pairs:
            return np.zeros(0, dtype=np.float32)
        return np.asarray(self.model.predict(list(pairs), batch_size=self.batch_size, show_progress_bar=progress),
                          dtype=np.float32)
