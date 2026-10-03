import math

import numpy as np
import pytest

from evident.bm25 import BM25
from evident.data import chunk_text
from evident.dense import ExactDense, StrataDense
from evident.fusion import convex, rrf
from evident.metrics import evaluate_run, mrr_at_k, ndcg_at_k, recall_at_k
from evident.retriever import MODES
from evident.text import tokenize


def test_tokenize_drops_stopwords_and_stems():
    assert tokenize("The runners were running to the markets!") == ["runner", "were", "run", "market"]


def test_bm25_matches_hand_computation():
    texts = ["apple banana apple", "banana cherry", "cherry cherry cherry date"]
    bm = BM25(k1=1.2, b=0.75).fit(texts)
    n, avgdl = 3, (3 + 2 + 4) / 3
    idf = lambda df: math.log(1 + (n - df + 0.5) / (df + 0.5))  # noqa: E731
    tf_part = lambda tf, dl: tf * 2.2 / (tf + 1.2 * (1 - 0.75 + 0.75 * dl / avgdl))  # noqa: E731
    expected_doc0 = idf(1) * tf_part(2, 3) + idf(2) * tf_part(1, 3)  # "apple banana"
    scores = bm.scores("apple banana")
    assert scores[0] == pytest.approx(expected_doc0, rel=1e-5)
    assert scores[1] == pytest.approx(idf(2) * tf_part(1, 2), rel=1e-5)
    assert scores[2] == 0.0
    idx, s = bm.search("cherry", k=3)
    assert list(idx) == [2, 1] and s[0] > s[1]
    idx, _ = bm.search("cherry", k=3, exclude={2})
    assert list(idx) == [1]
    assert len(bm.search("zebra", k=3)[0]) == 0


def test_metrics_on_known_rankings():
    qrel = {"a": 1, "b": 2}
    assert ndcg_at_k(["b", "a", "x"], qrel) == pytest.approx(1.0)
    expected = (1 / math.log2(2) + 2 / math.log2(3)) / (2 / math.log2(2) + 1 / math.log2(3))
    assert ndcg_at_k(["a", "b"], qrel) == pytest.approx(expected)
    assert recall_at_k(["x", "a"], qrel, 2) == 0.5
    assert mrr_at_k(["x", "y", "b"], qrel) == pytest.approx(1 / 3)
    assert mrr_at_k(["x"], qrel) == 0.0
    run = {"q1": ["a", "b"], "q2": ["z"]}
    res = evaluate_run(run, {"q1": qrel, "q2": {"y": 1}, "q3": {"n": 0}})
    assert res["recall@10"] == pytest.approx(0.5)  # q3 has no relevant docs and is skipped


def test_rrf_and_convex_fusion():
    fused = rrf([[1, 2, 3], [3, 1, 4]], k=60)
    order = [d for d, _ in fused]
    assert order[0] == 1 and set(order) == {1, 2, 3, 4}
    assert dict(fused)[1] == pytest.approx(1 / 61 + 1 / 62)
    assert [d for d, _ in convex({1: 0.9, 2: 0.1}, {2: 10.0, 3: 5.0}, alpha=1.0)][0] == 1
    assert [d for d, _ in convex({1: 0.9, 2: 0.1}, {2: 10.0, 3: 5.0}, alpha=0.0)][0] == 2


def test_strata_dense_agrees_with_exact():
    rng = np.random.default_rng(0)
    v = rng.standard_normal((2000, 32)).astype(np.float32)
    v /= np.linalg.norm(v, axis=1, keepdims=True)
    q = v[:20] + 0.01
    a_ids, a_s = StrataDense(v).search(q, 10)
    b_ids, b_s = ExactDense(v).search(q, 10)
    assert np.mean([len(set(x) & set(y)) / 10 for x, y in zip(a_ids, b_ids)]) > 0.95
    ex_ids, _ = StrataDense(v).search(q[:2], 5, exclude=[{0}, set()])
    assert 0 not in ex_ids[0]


def test_every_mode_returns_ranked_hits(retriever):
    for mode in MODES:
        hits = retriever.run(["tax free retirement withdrawals"], mode, k=3)[0]
        assert 1 <= len(hits) <= 3
        hits1, timings = retriever.retrieve("tax free retirement withdrawals", mode, k=3)
        assert timings["total"] >= 0
    top = retriever.run(["roth ira after-tax"], "hybrid_rerank", k=1)[0][0]
    assert retriever.docs[top.idx].id == "d0"
    hits = retriever.run(["roth ira after-tax"], "hybrid", k=6, exclude=[{0}])[0]
    assert all(h.idx != 0 for h in hits)


def test_chunking_overlap_and_limits():
    text = " ".join(f"Sentence number {i} is here." for i in range(100))
    chunks = chunk_text(text, max_words=50, overlap=10)
    assert all(len(c.split()) <= 60 for c in chunks)
    assert len(chunks) > 5
    first_tail = chunks[0].split()[-10:]
    assert chunks[1].split()[:10] == first_tail
    long_sentence = "word " * 500
    assert all(len(c.split()) <= 120 for c in chunk_text(long_sentence, max_words=100, overlap=20))
    with pytest.raises(ValueError):
        chunk_text("x", max_words=10, overlap=10)
