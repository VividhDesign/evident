"""End-to-end answer evaluation: faithfulness, relevance, citations, abstention, latency, cost.

Conditions (same questions, same generator, same judge):
  bm25                      answers grounded in BM25 top-k
  hybrid_convex             answers grounded in the best retrieval system from eval-retrieval
                            (BM25 + dense, weight tuned on the dev split)
  evidence_removed          hybrid_convex, but every document judged relevant for the question is
                            filtered out of retrieval (Strata deny-filter + BM25 mask). A grounded
                            system should abstain more often here instead of making things up.
"""

from __future__ import annotations

import json
import random
from pathlib import Path

import numpy as np

from . import config
from .data import load_beir
from .judge import Judge
from .llm import LLM, load_prices
from .metrics import percentile_ms
from .rag import RAG, RAGAnswer
from .retriever import Retriever


def _mean(xs):
    xs = [x for x in xs if x is not None]
    return float(np.mean(xs)) if xs else None


def summarize(rows: list[dict]) -> dict:
    answered = [r for r in rows if not r["abstained"]]
    e2e = [r["timings"].get("retrieve.total", 0) + r["timings"].get("generate", 0) for r in rows]
    gen = [r["timings"].get("generate", 0) for r in rows]
    prices = load_prices()
    tok_in = float(np.mean([r["input_tokens"] for r in rows]))
    tok_out = float(np.mean([r["output_tokens"] for r in rows]))
    return {
        "n": len(rows),
        "abstention_rate": sum(r["abstained"] for r in rows) / len(rows),
        "context_hit_rate": float(np.mean([r["context_has_relevant"] for r in rows])),
        "faithfulness": _mean([r["faithfulness"] for r in answered]),
        "fully_faithful_rate": _mean([None if r["faithfulness"] is None else float(r["faithfulness"] == 1.0) for r in answered]),
        "answer_relevance": _mean([r["relevance"] for r in answered]),
        "citation_precision": _mean([r["citation_precision"] for r in answered]),
        "citation_coverage": _mean([r["citation_coverage"] for r in answered]),
        "invalid_citations": int(sum(r["invalid_citations"] for r in rows)),
        "latency_ms": {"retrieval_p50": percentile_ms([r["timings"].get("retrieve.total", 0) for r in rows], 50),
                       "generation_p50": percentile_ms(gen, 50), "generation_p95": percentile_ms(gen, 95),
                       "end_to_end_p50": percentile_ms(e2e, 50), "end_to_end_p95": percentile_ms(e2e, 95)},
        "tokens_per_query": {"input": tok_in, "output": tok_out},
        "estimated_usd_per_1k_queries": {name: (tok_in * p[0] + tok_out * p[1]) / 1e6 * 1000 for name, p in prices.items()},
    }


def _record(condition: str, qid: str, ans: RAGAnswer, judgement, relevant: set[str]) -> dict:
    return {
        "condition": condition, "qid": qid, "question": ans.question, "answer": ans.answer,
        "abstained": ans.abstained, "sources": [s.doc_id for s in ans.sources],
        "context_has_relevant": any(s.doc_id in relevant for s in ans.sources),
        "faithfulness": judgement.faithfulness, "claims": judgement.claims, "relevance": judgement.relevance,
        "relevance_reason": judgement.relevance_reason, "citation_precision": judgement.citation_precision,
        "citation_pairs": judgement.citation_pairs, "citation_coverage": ans.citation_coverage,
        "invalid_citations": ans.invalid_citations, "timings": ans.timings,
        "input_tokens": ans.input_tokens, "output_tokens": ans.output_tokens, "model": ans.model,
    }


def evaluate(dataset: str = "fiqa", n: int = 100, abstention_n: int = 50, k: int = 5, seed: int = 0,
             modes=("bm25", "hybrid_convex"), generator: LLM | None = None, judge_llm: LLM | None = None,
             reranker: str = config.DEFAULT_RERANKER, out_dir: Path = config.RESULTS_DIR) -> dict:
    ds = load_beir(dataset, "test")
    generator = generator or LLM(config.LLM_PROVIDER, config.LLM_MODEL)
    judge_llm = judge_llm or LLM(config.JUDGE_PROVIDER, config.JUDGE_MODEL)
    retriever = Retriever(ds.docs, reranker=reranker, alpha=config.tuned_alpha(dataset))
    rag = RAG(retriever, generator, mode="hybrid_convex", k=k)
    judge = Judge(judge_llm)

    qids = sorted(ds.queries)
    random.Random(seed).shuffle(qids)
    sample = qids[:n]
    out_dir.mkdir(parents=True, exist_ok=True)
    records_path = out_dir / f"generation_{dataset}_records.jsonl"
    rows: list[dict] = []

    # Two phases: generate every answer, then judge every answer. On a single machine the
    # generator and judge models don't both fit in GPU memory; interleaving them would reload
    # a model on every call and pollute the generation latency with model-load time.
    jobs = [(mode, qid, mode, None) for mode in modes for qid in sample]
    jobs += [("evidence_removed", qid, "hybrid_convex", {ds.doc_index[d] for d in ds.qrels[qid] if d in ds.doc_index})
             for qid in sample[:abstention_n]]
    answers: list[tuple[str, str, RAGAnswer]] = []
    print(f"[{dataset}] generating {len(jobs)} answers with {generator.name}", flush=True)
    for i, (condition, qid, mode, exclude) in enumerate(jobs, 1):
        answers.append((condition, qid, rag.answer(ds.queries[qid], exclude=exclude, mode=mode)))
        if i % 25 == 0:
            print(f"  generated {i}/{len(jobs)}", flush=True)

    print(f"[{dataset}] judging with {judge_llm.name}", flush=True)
    records_path.write_text("")
    for i, (condition, qid, ans) in enumerate(answers, 1):
        relevant = {d for d, r in ds.qrels[qid].items() if r > 0}
        row = _record(condition, qid, ans, judge.judge(ans), relevant)
        rows.append(row)
        with open(records_path, "a") as f:
            f.write(json.dumps(row) + "\n")
        if i % 25 == 0:
            print(f"  judged {i}/{len(answers)}", flush=True)

    report = {
        "dataset": dataset, "questions": n, "k": k, "seed": seed, "generator": generator.name,
        "judge": judge_llm.name, "retrieval_alpha": retriever.alpha,
        "conditions": {c: summarize([r for r in rows if r["condition"] == c])
                       for c in dict.fromkeys(r["condition"] for r in rows)},
    }
    (out_dir / f"generation_{dataset}.json").write_text(json.dumps(report, indent=2))
    return report


def paired_bootstrap(records_path: Path, n_boot: int = 10_000, seed: int = 0) -> dict:
    """95% bootstrap CIs for paired differences between conditions on the same questions."""
    rows = [json.loads(line) for line in open(records_path) if line.strip()]
    by: dict[str, dict[str, dict]] = {}
    for r in rows:
        by.setdefault(r["condition"], {})[r["qid"]] = r
    rng = np.random.default_rng(seed)

    def ci(diffs):
        d = np.asarray(diffs, dtype=float)
        means = d[rng.integers(0, len(d), (n_boot, len(d)))].mean(axis=1)
        return {"mean": float(d.mean()), "ci95": [float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))],
                "n": int(len(d))}

    def compare(a: str, b: str) -> dict:
        qs = sorted(set(by.get(a, {})) & set(by.get(b, {})))
        A, B = by[a], by[b]
        out = {"context_hit_rate": ci([float(B[q]["context_has_relevant"]) - float(A[q]["context_has_relevant"]) for q in qs]),
               "abstention_rate": ci([float(B[q]["abstained"]) - float(A[q]["abstained"]) for q in qs])}
        for metric in ("relevance", "faithfulness"):
            both = [q for q in qs if A[q][metric] is not None and B[q][metric] is not None]
            if both:
                out[metric] = ci([B[q][metric] - A[q][metric] for q in both])
        both = [q for q in qs if A[q]["faithfulness"] is not None and B[q]["faithfulness"] is not None]
        if both:
            out["fully_faithful_rate"] = ci([float(B[q]["faithfulness"] == 1) - float(A[q]["faithfulness"] == 1) for q in both])
        return out

    result = {}
    if "bm25" in by and "hybrid_convex" in by:
        result["hybrid_convex_minus_bm25"] = compare("bm25", "hybrid_convex")
    if "evidence_removed" in by and "hybrid_convex" in by:
        result["evidence_removed_minus_hybrid_convex"] = compare("hybrid_convex", "evidence_removed")
    result["claims_judged"] = sum(len(r["claims"]) for r in rows)
    result["claims_unsupported"] = sum(1 for r in rows for c in r["claims"] if not c["supported"])
    return result


def markdown_table(report: dict) -> str:
    names = {"bm25": "BM25 context", "hybrid_convex": "Hybrid context (best retriever)",
             "hybrid_rerank": "Hybrid + rerank context", "evidence_removed": "Hybrid, relevant docs hidden"}
    fmt = lambda v, pct=False: "—" if v is None else (f"{v:.0%}" if pct else f"{v:.2f}")  # noqa: E731
    lines = ["| Condition | n | Context has relevant doc | Abstained | Faithfulness | Fully faithful | "
             "Relevance (1-5) | Citation precision | p50 latency |",
             "|---|---:|---:|---:|---:|---:|---:|---:|---:|"]
    for cond, s in report["conditions"].items():
        lines.append(f"| {names.get(cond, cond)} | {s['n']} | {fmt(s['context_hit_rate'], True)} | "
                     f"{fmt(s['abstention_rate'], True)} | {fmt(s['faithfulness'])} | {fmt(s['fully_faithful_rate'], True)} | "
                     f"{fmt(s['answer_relevance'])} | {fmt(s['citation_precision'])} | "
                     f"{s['latency_ms']['end_to_end_p50'] / 1000:.1f}s |")
    return "\n".join(lines)
