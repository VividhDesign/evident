"""Dense (vector) search backends.

StrataDense serves queries from the Strata HNSW index (approximate, sub-linear). ExactDense
is brute-force matrix multiplication and acts as the ground truth for how much retrieval
quality the approximate index costs.
"""

from __future__ import annotations

import os
from pathlib import Path

import numpy as np


class StrataDense:
    name = "strata"

    def __init__(self, vectors: np.ndarray, M: int = 16, ef_construction: int = 200, ef: int = 128,
                 index_path: str | Path | None = None):
        """Builds the HNSW index, or loads it from `index_path` if a snapshot exists there (the file is
        written after a build). EVIDENT_THREADS caps build threads on small, shared CPUs."""
        import strata

        self.ef = ef
        if index_path is not None and Path(index_path).exists():
            self.index = strata.Index.load(str(index_path))
            if len(self.index) == len(vectors) and self.index.dim == vectors.shape[1]:
                return
        self.index = strata.Index(vectors.shape[1], "cosine", M=M, ef_construction=ef_construction,
                                  capacity=len(vectors))
        self.index.add(vectors, num_threads=int(os.environ.get("EVIDENT_THREADS", "0")))  # ids = corpus positions
        if index_path is not None:
            Path(index_path).parent.mkdir(parents=True, exist_ok=True)
            self.index.save(str(index_path))

    def search(self, queries: np.ndarray, k: int, exclude: list[set[int]] | None = None):
        """Returns (ids, similarities), each (n_queries, k); -1 pads missing results."""
        if exclude is None or not any(exclude):
            ids, dist = self.index.search(queries, k=k, ef=max(self.ef, k))
            return ids, 1.0 - dist
        ids = np.full((len(queries), k), -1, dtype=np.int64)
        sims = np.full((len(queries), k), -np.inf, dtype=np.float32)
        for i, q in enumerate(queries):
            ex = np.fromiter(exclude[i], dtype=np.int64) if exclude[i] else None
            r_ids, r_dist = self.index.search(q[None], k=k, ef=max(self.ef, k), exclude_ids=ex)
            ids[i], sims[i] = r_ids[0], 1.0 - r_dist[0]
        return ids, sims


class ExactDense:
    name = "exact"

    def __init__(self, vectors: np.ndarray):
        self.vectors = vectors

    def search(self, queries: np.ndarray, k: int, exclude: list[set[int]] | None = None):
        sims = queries @ self.vectors.T
        if exclude:
            for i, ex in enumerate(exclude):
                if ex:
                    sims[i, list(ex)] = -np.inf
        k = min(k, sims.shape[1])
        top = np.argpartition(-sims, k - 1, axis=1)[:, :k]
        top_sims = np.take_along_axis(sims, top, axis=1)
        order = np.argsort(-top_sims, axis=1, kind="stable")
        return np.take_along_axis(top, order, axis=1), np.take_along_axis(top_sims, order, axis=1)


def make_dense(backend: str, vectors: np.ndarray, **kwargs):
    if backend == "strata":
        return StrataDense(vectors, **kwargs)
    if backend == "exact":
        return ExactDense(vectors)  # nothing to build or cache
    raise ValueError(f"unknown dense backend {backend!r}")
