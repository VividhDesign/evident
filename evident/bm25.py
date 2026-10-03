"""Okapi BM25, implemented with a sparse document-term matrix.

score(q, d) = sum over query terms t of
    idf(t) * tf(t, d) * (k1 + 1) / (tf(t, d) + k1 * (1 - b + b * |d| / avgdl))
idf(t) = ln(1 + (N - df(t) + 0.5) / (df(t) + 0.5))         (Lucene's non-negative variant)

All the per-(term, doc) weights are precomputed at index time into a CSC matrix, so a
query is just a sum of a few sparse columns.
"""

from __future__ import annotations

import os
from collections import Counter
from pathlib import Path

import numpy as np
from scipy import sparse

from .text import tokenize


class BM25:
    def __init__(self, k1: float = 0.9, b: float = 0.4):
        # k1 = 0.9, b = 0.4 are Pyserini's defaults for BEIR.
        self.k1, self.b = k1, b
        self.vocab: dict[str, int] = {}
        self.weights: sparse.csc_matrix | None = None
        self.n_docs = 0

    def fit(self, texts: list[str]) -> "BM25":
        rows, cols, tfs = [], [], []
        doc_len = np.zeros(len(texts), dtype=np.float32)
        for i, text in enumerate(texts):
            tokens = tokenize(text)
            doc_len[i] = len(tokens)
            for term, tf in Counter(tokens).items():
                j = self.vocab.setdefault(term, len(self.vocab))
                rows.append(i)
                cols.append(j)
                tfs.append(tf)
        self.n_docs = len(texts)
        rows, cols = np.asarray(rows), np.asarray(cols)
        tf = np.asarray(tfs, dtype=np.float32)

        df = np.bincount(cols, minlength=len(self.vocab)).astype(np.float32)
        self.idf = np.log1p((self.n_docs - df + 0.5) / (df + 0.5)).astype(np.float32)
        avgdl = float(doc_len.mean()) if self.n_docs else 1.0
        norm = self.k1 * (1 - self.b + self.b * doc_len[rows] / avgdl)
        w = self.idf[cols] * tf * (self.k1 + 1) / (tf + norm)
        self.weights = sparse.csc_matrix((w, (rows, cols)), shape=(self.n_docs, len(self.vocab)), dtype=np.float32)
        return self

    def save(self, path: str | Path) -> None:
        """Stores the fitted index (weights, vocabulary, idf) so it can be loaded without re-tokenising."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        w = self.weights
        terms = np.array(sorted(self.vocab, key=self.vocab.get), dtype=str)
        tmp = path.with_name(path.name + ".tmp.npz")
        np.savez_compressed(tmp, data=w.data, indices=w.indices, indptr=w.indptr, shape=np.array(w.shape),
                            idf=self.idf, params=np.array([self.k1, self.b]), terms=terms)
        os.replace(tmp, path)

    @classmethod
    def load(cls, path: str | Path) -> "BM25":
        z = np.load(path)
        bm = cls(k1=float(z["params"][0]), b=float(z["params"][1]))
        bm.weights = sparse.csc_matrix((z["data"], z["indices"], z["indptr"]), shape=tuple(z["shape"]))
        bm.idf = z["idf"]
        bm.vocab = {str(t): i for i, t in enumerate(z["terms"])}
        bm.n_docs = int(z["shape"][0])
        return bm

    def scores(self, query: str) -> np.ndarray:
        """BM25 score of every document for `query` (repeated query terms count repeatedly)."""
        cols = [self.vocab[t] for t in tokenize(query) if t in self.vocab]
        if not cols:
            return np.zeros(self.n_docs, dtype=np.float32)
        return np.asarray(self.weights[:, cols].sum(axis=1)).ravel()

    def search(self, query: str, k: int = 100, exclude: set[int] | None = None) -> tuple[np.ndarray, np.ndarray]:
        s = self.scores(query)
        if exclude:
            s[list(exclude)] = -np.inf
        k = min(k, self.n_docs)
        top = np.argpartition(-s, k - 1)[:k]
        top = top[np.argsort(-s[top], kind="stable")]
        keep = s[top] > 0  # documents sharing no term with the query are not matches
        return top[keep], s[top][keep]
