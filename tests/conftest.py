import hashlib

import numpy as np
import pytest

from evident.data import Doc


class HashEmbedder:
    """Bag-of-words hashing embedder: deterministic, no model download, good enough to make
    documents sharing words land near each other."""

    model_name = "hash"
    device = "cpu"
    dim = 64

    def _vec(self, text):
        v = np.zeros(self.dim, dtype=np.float32)
        for w in text.lower().split():
            v[int(hashlib.md5(w.strip(".,?!").encode()).hexdigest(), 16) % self.dim] += 1
        n = np.linalg.norm(v)
        return v / n if n else v

    def embed_documents(self, texts, cache=True):
        return np.stack([self._vec(t) for t in texts])

    def embed_queries(self, texts):
        return np.stack([self._vec(t) for t in texts])


class FakeReranker:
    """Scores by word overlap with the query."""

    def score(self, pairs, progress=False):
        return np.array([len(set(q.lower().split()) & set(d.lower().split())) for q, d in pairs], dtype=np.float32)


@pytest.fixture
def docs():
    texts = [
        "The Roth IRA is funded with after-tax dollars and withdrawals in retirement are tax free.",
        "A traditional IRA gives a tax deduction now and withdrawals are taxed as income.",
        "Index funds track a market index and usually have low expense ratios.",
        "Compound interest means earning interest on previously earned interest.",
        "Bonds pay fixed coupons and their prices fall when interest rates rise.",
        "Dollar cost averaging invests a fixed amount at regular intervals.",
    ]
    return [Doc(id=f"d{i}", text=t) for i, t in enumerate(texts)]


@pytest.fixture
def retriever(docs):
    from evident.retriever import Retriever

    return Retriever(docs, embedder=HashEmbedder(), dense_backend="strata", reranker=FakeReranker(), candidates=6,
                     rerank_depth=6, cache_embeddings=False)
