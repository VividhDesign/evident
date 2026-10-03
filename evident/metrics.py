"""IR metrics with trec_eval conventions (what BEIR / pytrec_eval report).

nDCG@k uses linear gain (gain = relevance grade) and log2(rank + 1) discounting; the ideal
DCG is computed from all judged relevant documents of the query.
"""

from __future__ import annotations

import math
from typing import Mapping, Sequence

import numpy as np


def ndcg_at_k(ranked: Sequence[str], qrel: Mapping[str, int], k: int = 10) -> float:
    dcg = sum(qrel.get(d, 0) / math.log2(i + 2) for i, d in enumerate(ranked[:k]))
    ideal = sorted((r for r in qrel.values() if r > 0), reverse=True)[:k]
    idcg = sum(r / math.log2(i + 2) for i, r in enumerate(ideal))
    return dcg / idcg if idcg > 0 else 0.0


def recall_at_k(ranked: Sequence[str], qrel: Mapping[str, int], k: int) -> float:
    relevant = {d for d, r in qrel.items() if r > 0}
    if not relevant:
        return 0.0
    return len(relevant.intersection(ranked[:k])) / len(relevant)


def mrr_at_k(ranked: Sequence[str], qrel: Mapping[str, int], k: int = 10) -> float:
    for i, d in enumerate(ranked[:k]):
        if qrel.get(d, 0) > 0:
            return 1.0 / (i + 1)
    return 0.0


def evaluate_run(run: Mapping[str, Sequence[str]], qrels: Mapping[str, Mapping[str, int]]) -> dict[str, float]:
    """Mean metrics over queries that have at least one relevant document."""
    per_query = {"ndcg@10": [], "recall@10": [], "recall@100": [], "mrr@10": []}
    for qid, qrel in qrels.items():
        if not any(r > 0 for r in qrel.values()):
            continue
        ranked = list(run.get(qid, []))
        per_query["ndcg@10"].append(ndcg_at_k(ranked, qrel, 10))
        per_query["recall@10"].append(recall_at_k(ranked, qrel, 10))
        per_query["recall@100"].append(recall_at_k(ranked, qrel, 100))
        per_query["mrr@10"].append(mrr_at_k(ranked, qrel, 10))
    return {m: float(np.mean(v)) if v else 0.0 for m, v in per_query.items()}


def percentile_ms(seconds: Sequence[float], p: float) -> float:
    return float(np.percentile(np.asarray(seconds) * 1000.0, p)) if len(seconds) else 0.0
