"""Combining rankings from lexical (BM25) and semantic (dense) retrieval."""

from __future__ import annotations

from typing import Sequence


def rrf(rankings: Sequence[Sequence[int]], k: int = 60) -> list[tuple[int, float]]:
    """Reciprocal Rank Fusion (Cormack et al., SIGIR 2009): score(d) = sum_r 1 / (k + rank_r(d)).

    Uses only ranks, so BM25 scores (unbounded) and cosine similarities (in [-1, 1]) never
    need to be put on a common scale.
    """
    scores: dict[int, float] = {}
    for ranking in rankings:
        for rank, doc in enumerate(ranking, start=1):
            if doc < 0:
                continue
            scores[doc] = scores.get(doc, 0.0) + 1.0 / (k + rank)
    return sorted(scores.items(), key=lambda kv: (-kv[1], kv[0]))


def _minmax(scores: dict[int, float]) -> dict[int, float]:
    if not scores:
        return {}
    lo, hi = min(scores.values()), max(scores.values())
    if hi == lo:
        return {d: 1.0 for d in scores}
    return {d: (s - lo) / (hi - lo) for d, s in scores.items()}


def convex(dense: dict[int, float], sparse: dict[int, float], alpha: float) -> list[tuple[int, float]]:
    """alpha * minmax(dense) + (1 - alpha) * minmax(sparse); a missing score counts as 0.

    Needs alpha tuned on held-out queries (see eval: tuned on dev/train, reported on test).
    """
    d, s = _minmax(dense), _minmax(sparse)
    fused = {doc: alpha * d.get(doc, 0.0) + (1 - alpha) * s.get(doc, 0.0) for doc in set(d) | set(s)}
    return sorted(fused.items(), key=lambda kv: (-kv[1], kv[0]))
