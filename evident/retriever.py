"""The retrieval pipeline: BM25 and/or dense candidates -> fusion -> optional cross-encoder rerank.

Modes
  bm25            lexical only
  dense           bi-encoder + vector index only
  hybrid          RRF of BM25 and dense
  hybrid_convex   min-max-normalised weighted sum of BM25 and dense (alpha tuned on held-out data)
  bm25_rerank     cross-encoder over BM25 top-N (the classic BEIR "BM25+CE" baseline)
  hybrid_rerank   cross-encoder over hybrid top-N
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Sequence

import numpy as np

from .bm25 import BM25
from .data import Doc
from .dense import make_dense
from .embed import Embedder
from .fusion import convex, rrf

MODES = ("bm25", "dense", "hybrid", "hybrid_convex", "bm25_rerank", "hybrid_rerank")
_USES_DENSE = {"dense", "hybrid", "hybrid_convex", "hybrid_rerank"}
_USES_SPARSE = {"bm25", "hybrid", "hybrid_convex", "bm25_rerank", "hybrid_rerank"}
_RERANKS = {"bm25_rerank", "hybrid_rerank"}


@dataclass
class Hit:
    idx: int  # position of the document in the corpus
    score: float


class Retriever:
    def __init__(self, docs: Sequence[Doc], embedder: Embedder | None = None, dense_backend: str = "strata",
                 reranker=None, candidates: int = 100, rerank_depth: int = 50, alpha: float = 0.5,
                 rrf_k: int = 60, cache_embeddings: bool = True):
        self.docs = list(docs)
        self.texts = [d.full_text for d in self.docs]
        self.embedder = embedder or Embedder()
        self.candidates, self.rerank_depth, self.alpha, self.rrf_k = candidates, rerank_depth, alpha, rrf_k
        self.build_seconds: dict[str, float] = {}

        t0 = time.perf_counter()
        self.bm25 = BM25().fit(self.texts)
        self.build_seconds["bm25_index"] = time.perf_counter() - t0
        t0 = time.perf_counter()
        self.doc_vectors = self.embedder.embed_documents(self.texts, cache=cache_embeddings)
        self.build_seconds["embed_corpus"] = time.perf_counter() - t0
        t0 = time.perf_counter()
        self.dense = make_dense(dense_backend, self.doc_vectors)
        self.build_seconds[f"{dense_backend}_index"] = time.perf_counter() - t0
        self._reranker = reranker

    @property
    def reranker(self):
        if self._reranker is None or isinstance(self._reranker, str):
            from .rerank import Reranker

            self._reranker = Reranker(self._reranker) if isinstance(self._reranker, str) else Reranker()
        return self._reranker

    # --- fusion -------------------------------------------------------------------------------

    def _fuse(self, mode: str, sparse: tuple[np.ndarray, np.ndarray] | None,
              dense: tuple[np.ndarray, np.ndarray] | None) -> list[tuple[int, float]]:
        if mode in ("bm25", "bm25_rerank"):
            return list(zip(sparse[0].tolist(), sparse[1].tolist()))
        d_ids, d_sims = dense
        dense_pairs = [(int(i), float(s)) for i, s in zip(d_ids, d_sims) if i >= 0]
        if mode == "dense":
            return dense_pairs
        if mode == "hybrid_convex":
            return convex(dict(dense_pairs), dict(zip(sparse[0].tolist(), sparse[1].tolist())), self.alpha)
        return rrf([sparse[0].tolist(), [i for i, _ in dense_pairs]], k=self.rrf_k)

    # --- batch path (evaluation) ----------------------------------------------------------------

    def run(self, queries: Sequence[str], mode: str, k: int = 100,
            exclude: Sequence[set[int]] | None = None) -> list[list[Hit]]:
        """Retrieves for many queries at once (batched embedding and reranking)."""
        if mode not in MODES:
            raise ValueError(f"mode must be one of {MODES}")
        exclude = list(exclude) if exclude is not None else [set() for _ in queries]
        depth = max(k, self.candidates)
        dense_ids = dense_sims = None
        if mode in _USES_DENSE:
            qv = self.embedder.embed_queries(list(queries))
            dense_ids, dense_sims = self.dense.search(qv, depth, exclude)
        fused = []
        for i, q in enumerate(queries):
            sparse = self.bm25.search(q, depth, exclude[i]) if mode in _USES_SPARSE else None
            dense = (dense_ids[i], dense_sims[i]) if dense_ids is not None else None
            fused.append(self._fuse(mode, sparse, dense))

        if mode in _RERANKS:
            pairs, owners = [], []
            for i, cands in enumerate(fused):
                for doc, _ in cands[: self.rerank_depth]:
                    pairs.append((queries[i], self.texts[doc]))
                    owners.append((i, doc))
            scores = self.reranker.score(pairs, progress=len(pairs) > 5000)
            reranked: list[list[tuple[int, float]]] = [[] for _ in queries]
            for (i, doc), s in zip(owners, scores):
                reranked[i].append((doc, float(s)))
            for i in range(len(fused)):
                reranked[i].sort(key=lambda kv: -kv[1])
                # keep un-reranked tail after the reranked head so recall@100 is still defined
                head = {d for d, _ in reranked[i]}
                fused[i] = reranked[i] + [(d, s) for d, s in fused[i] if d not in head]
        return [[Hit(d, s) for d, s in cands[:k]] for cands in fused]

    # --- single-query path (serving, latency measurement) ---------------------------------------

    def retrieve(self, query: str, mode: str = "hybrid_rerank", k: int = 10,
                 exclude: set[int] | None = None) -> tuple[list[Hit], dict[str, float]]:
        """Returns hits and per-stage wall-clock seconds."""
        if mode not in MODES:
            raise ValueError(f"mode must be one of {MODES}")
        exclude = exclude or set()
        timings: dict[str, float] = {}
        depth = max(k, self.candidates)
        dense = sparse = None
        if mode in _USES_DENSE:
            t0 = time.perf_counter()
            qv = self.embedder.embed_queries([query])
            timings["embed_query"] = time.perf_counter() - t0
            t0 = time.perf_counter()
            ids, sims = self.dense.search(qv, depth, [exclude])
            dense = (ids[0], sims[0])
            timings["vector_search"] = time.perf_counter() - t0
        if mode in _USES_SPARSE:
            t0 = time.perf_counter()
            sparse = self.bm25.search(query, depth, exclude)
            timings["bm25"] = time.perf_counter() - t0
        t0 = time.perf_counter()
        cands = self._fuse(mode, sparse, dense)
        timings["fusion"] = time.perf_counter() - t0
        if mode in _RERANKS:
            t0 = time.perf_counter()
            head = cands[: self.rerank_depth]
            scores = self.reranker.score([(query, self.texts[d]) for d, _ in head])
            cands = sorted(((d, float(s)) for (d, _), s in zip(head, scores)), key=lambda kv: -kv[1])
            timings["rerank"] = time.perf_counter() - t0
        timings["total"] = sum(timings.values())
        return [Hit(d, s) for d, s in cands[:k]], timings
