"""Retrieval evaluation on BEIR: quality (nDCG/recall/MRR), latency per stage, and how much
quality the approximate vector index (Strata HNSW) costs versus exact search."""

from __future__ import annotations

import json
import platform
import random
import time
from pathlib import Path

import numpy as np

from . import config
from .data import load_beir
from .dense import ExactDense
from .embed import Embedder
from .metrics import evaluate_run, percentile_ms
from .retriever import Retriever

DEFAULT_MODES = ("bm25", "dense", "hybrid", "hybrid_convex", "bm25_rerank", "hybrid_rerank")


def _run_to_ids(retriever: Retriever, qids, hits) -> dict[str, list[str]]:
    return {qid: [retriever.docs[h.idx].id for h in hs] for qid, hs in zip(qids, hits)}


def tune_alpha(retriever: Retriever, dataset: str, max_queries: int = 500, seed: int = 0) -> tuple[float, dict]:
    """Grid-searches the convex-fusion weight on a held-out split (dev, else train), never on test."""
    split = "dev" if (config.BEIR_DIR / dataset / "qrels" / "dev.tsv").exists() else "train"
    held_out = load_beir(dataset, split)
    qids = sorted(held_out.queries)
    random.Random(seed).shuffle(qids)
    qids = qids[:max_queries]
    texts = [held_out.queries[q] for q in qids]
    qrels = {q: held_out.qrels[q] for q in qids}
    scores = {}
    for alpha in np.round(np.linspace(0, 1, 11), 2):
        retriever.alpha = float(alpha)
        scores[float(alpha)] = evaluate_run(_run_to_ids(retriever, qids, retriever.run(texts, "hybrid_convex", 100)), qrels)["ndcg@10"]
    best = max(scores, key=scores.get)
    retriever.alpha = best
    return best, {"split": split, "queries": len(qids), "ndcg@10_by_alpha": scores}


def measure_latency(retriever: Retriever, queries: list[str], mode: str) -> dict:
    retriever.retrieve(queries[0], mode)  # warm-up (model kernels, caches)
    totals, stages = [], {}
    for q in queries:
        _, t = retriever.retrieve(q, mode, k=10)
        totals.append(t["total"])
        for name, secs in t.items():
            if name != "total":
                stages.setdefault(name, []).append(secs)
    return {"p50_ms": percentile_ms(totals, 50), "p95_ms": percentile_ms(totals, 95),
            "stage_mean_ms": {k: float(np.mean(v) * 1000) for k, v in stages.items()}}


def evaluate(dataset: str, modes=DEFAULT_MODES, rerankers=("minilm", "bge"),
             latency_queries: int = 100, out_dir: Path = config.RESULTS_DIR) -> dict:
    """Rerank modes are evaluated once per reranker and reported as e.g. "hybrid_rerank:bge"."""
    from .rerank import Reranker

    ds = load_beir(dataset, "test")
    print(f"[{dataset}] {len(ds.docs):,} docs, {len(ds.queries)} test queries", flush=True)
    embedder = Embedder()
    retriever = Retriever(ds.docs, embedder, dense_backend="strata")
    print(f"[{dataset}] built indexes: " + ", ".join(f"{k} {v:.1f}s" for k, v in retriever.build_seconds.items()), flush=True)

    qids = sorted(ds.queries)
    texts = [ds.queries[q] for q in qids]
    sample = random.Random(1).sample(texts, min(latency_queries, len(texts)))
    report = {"dataset": dataset, "docs": len(ds.docs), "queries": len(qids), "embed_model": embedder.model_name,
              "rerankers": {r: config.RERANK_MODELS.get(r, r) for r in rerankers},
              "rerank_depth": retriever.rerank_depth, "device": embedder.device, "machine": platform.platform(),
              "build_seconds": retriever.build_seconds, "systems": {}}

    if "hybrid_convex" in modes:
        alpha, info = tune_alpha(retriever, dataset)
        report["alpha_tuning"] = {"best_alpha": alpha, **info}
        print(f"[{dataset}] convex alpha tuned on {info['split']}: {alpha}", flush=True)

    runs = [(m, None) for m in modes if not m.endswith("_rerank")]
    runs += [(m, r) for r in rerankers for m in modes if m.endswith("_rerank")]
    loaded: dict[str, Reranker] = {}
    for mode, rr in runs:
        if rr is not None:
            retriever._reranker = loaded.setdefault(rr, Reranker(rr))
        name = mode if rr is None else f"{mode}:{rr}"
        t0 = time.perf_counter()
        hits = retriever.run(texts, mode, k=100)
        batch_s = time.perf_counter() - t0
        metrics = evaluate_run(_run_to_ids(retriever, qids, hits), ds.qrels)
        latency = measure_latency(retriever, sample, mode)
        report["systems"][name] = {**metrics, "latency": latency, "batch_seconds": batch_s}
        print(f"[{dataset}] {name:22s} nDCG@10={metrics['ndcg@10']:.4f} R@100={metrics['recall@100']:.4f} "
              f"p50={latency['p50_ms']:.1f}ms", flush=True)

    # How much does approximate search cost? Same embeddings, exact brute-force search.
    strata_dense = retriever.dense
    strata_hits = retriever.run(texts, "dense", k=10)
    retriever.dense = ExactDense(retriever.doc_vectors)
    exact_hits = retriever.run(texts, "dense", k=100)
    exact_metrics = evaluate_run(_run_to_ids(retriever, qids, exact_hits), ds.qrels)
    overlap = np.mean([len({h.idx for h in a} & {h.idx for h in b[:10]}) / 10 for a, b in zip(strata_hits, exact_hits)])
    retriever.dense = strata_dense
    report["ann_fidelity"] = {"exact_ndcg@10": exact_metrics["ndcg@10"],
                              "strata_ndcg@10": report["systems"].get("dense", {}).get("ndcg@10"),
                              "top10_overlap_with_exact": float(overlap), "strata_ef": strata_dense.ef}
    print(f"[{dataset}] ANN fidelity: top-10 overlap {overlap:.4f}, exact nDCG@10 {exact_metrics['ndcg@10']:.4f}", flush=True)

    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / f"retrieval_{dataset}.json").write_text(json.dumps(report, indent=2))
    return report


SYSTEM_NAMES = {
    "bm25": "BM25 (from scratch)", "dense": "Dense: bge-small + Strata HNSW", "hybrid": "Hybrid (RRF)",
    "hybrid_convex": "Hybrid (convex, α tuned on dev)",
    "bm25_rerank:minilm": "BM25 → MiniLM cross-encoder", "hybrid_rerank:minilm": "Hybrid → MiniLM cross-encoder",
    "bm25_rerank:bge": "BM25 → bge-reranker-base", "hybrid_rerank:bge": "Hybrid → bge-reranker-base",
}


def markdown_table(reports: list[dict]) -> str:
    datasets = [r["dataset"] for r in reports]
    lines = ["| System | " + " | ".join(f"{d} nDCG@10" for d in datasets) + " | " +
             " | ".join(f"{d} R@100" for d in datasets) + f" | p50 latency ({datasets[-1]}) |",
             "|---|" + "---:|" * (2 * len(datasets) + 1)]
    for key, label in SYSTEM_NAMES.items():
        if not all(key in r["systems"] for r in reports):
            continue
        best = [max(s["ndcg@10"] for s in r["systems"].values()) for r in reports]
        cells = [(f"**{r['systems'][key]['ndcg@10']:.3f}**" if r["systems"][key]["ndcg@10"] == b
                  else f"{r['systems'][key]['ndcg@10']:.3f}") for r, b in zip(reports, best)]
        cells += [f"{r['systems'][key]['recall@100']:.3f}" for r in reports]
        lat = reports[-1]["systems"][key]["latency"]["p50_ms"]
        lines.append(f"| {label} | " + " | ".join(cells) + f" | {lat:.1f} ms |")
    return "\n".join(lines)
