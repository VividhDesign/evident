"""Dense (vector) search backends.

StrataDense serves queries from the Strata HNSW index (approximate, sub-linear). ExactDense
is brute-force matrix multiplication and acts as the ground truth for how much retrieval
quality the approximate index costs.
"""

from __future__ import annotations

import numpy as np


class StrataDense:
    name = "strata"

    def __init__(self, vectors: np.ndarray, M: int = 16, ef_construction: int = 200, ef: int = 128):
        import strata

        self.index = strata.Index(vectors.shape[1], "cosine", M=M, ef_construction=ef_construction,
                                  capacity=len(vectors))
        self.index.add(vectors)  # ids 0..n-1 are positions in the corpus
        self.ef = ef

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
        return ExactDense(vectors)
    raise ValueError(f"unknown dense backend {backend!r}")
