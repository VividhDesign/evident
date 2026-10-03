"""Command line interface.

  evident eval-retrieval --datasets scifact,nfcorpus,fiqa
  evident eval-generation --dataset fiqa --n 100
  evident ask "What is a Roth IRA?" --corpus fiqa
  evident ingest ./my_pdfs --name notes && evident ask "..." --corpus notes
  evident calibrate-judge --n 30
"""

from __future__ import annotations

import argparse
import json
import random
import sys
import textwrap

from . import config


def _corpus_docs(name: str):
    from .data import load_beir
    from .ingest import load_corpus

    if name in config.BEIR_DATASETS:
        return load_beir(name, "test").docs
    return load_corpus(name)


def cmd_eval_retrieval(args):
    from .eval_retrieval import evaluate, markdown_table

    reports = [evaluate(d.strip(), rerankers=tuple(args.rerankers.split(","))) for d in args.datasets.split(",")]
    table = markdown_table(reports)
    print("\n" + table)
    (config.RESULTS_DIR / "retrieval_table.md").write_text(table + "\n")


def cmd_eval_generation(args):
    from .eval_generation import evaluate, markdown_table, paired_bootstrap
    from .llm import LLM

    report = evaluate(args.dataset, n=args.n, abstention_n=args.abstention_n, k=args.k,
                      generator=LLM(args.provider, args.model), judge_llm=LLM(args.judge_provider, args.judge_model),
                      reranker=args.reranker)
    table = markdown_table(report)
    print("\n" + table)
    (config.RESULTS_DIR / f"generation_{args.dataset}_table.md").write_text(table + "\n")
    stats = paired_bootstrap(config.RESULTS_DIR / f"generation_{args.dataset}_records.jsonl")
    (config.RESULTS_DIR / f"generation_{args.dataset}_stats.json").write_text(json.dumps(stats, indent=2))


def cmd_ask(args):
    from .llm import LLM
    from .rag import RAG
    from .retriever import Retriever

    retriever = Retriever(_corpus_docs(args.corpus), reranker=args.reranker, alpha=config.tuned_alpha(args.corpus))
    rag = RAG(retriever, LLM(args.provider, args.model), mode=args.mode, k=args.k)
    ans = rag.answer(args.question)
    print("\n" + textwrap.fill(ans.answer, 100) + "\n")
    for s in ans.sources:
        print(f"[{s.number}] {s.doc_id} (score {s.score:.3f}): {textwrap.shorten(s.text, 160)}")
    t = ans.timings
    print(f"\nretrieval {t.get('retrieve.total', 0) * 1000:.0f} ms · generation {t.get('generate', 0):.1f} s · "
          f"{ans.input_tokens} in / {ans.output_tokens} out tokens · cost "
          f"{'n/a' if ans.cost_usd is None else f'${ans.cost_usd:.5f}'}")


def cmd_ingest(args):
    from .ingest import ingest

    n = ingest(args.folder, args.name, max_words=args.chunk_words, overlap=args.overlap)
    print(f"indexed {n} chunks as corpus '{args.name}'")


def cmd_calibrate_judge(args):
    """Hand-label a sample of judged claims to measure how often the judge agrees with you."""
    path = config.RESULTS_DIR / f"generation_{args.dataset}_records.jsonl"
    rows = [json.loads(line) for line in open(path) if line.strip()]
    items = [(r, c) for r in rows for c in r.get("claims", [])]
    random.Random(0).shuffle(items)
    agree = total = 0
    labels = []
    for row, claim in items[: args.n]:
        print("\n" + "=" * 100)
        print("QUESTION:", row["question"])
        print("ANSWER:", textwrap.fill(row["answer"], 100))
        print("\nCLAIM:", claim["claim"])
        ans = input("Is this claim supported by the cited sources? Inspect sources in the app if needed. [y/n/s=skip] ")
        if ans.lower().startswith("s"):
            continue
        human = ans.lower().startswith("y")
        agree += human == bool(claim["supported"])
        total += 1
        labels.append({"qid": row["qid"], "claim": claim["claim"], "human": human, "judge": bool(claim["supported"])})
    if total:
        print(f"\njudge/human agreement: {agree}/{total} = {agree / total:.0%}")
        (config.RESULTS_DIR / "judge_calibration.json").write_text(json.dumps(
            {"agreement": agree / total, "n": total, "labels": labels}, indent=2))


def main(argv=None):
    ap = argparse.ArgumentParser(prog="evident", description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)

    p = sub.add_parser("eval-retrieval", help="BEIR retrieval benchmark")
    p.add_argument("--datasets", default="scifact,nfcorpus,fiqa")
    p.add_argument("--rerankers", default="minilm,bge")
    p.set_defaults(fn=cmd_eval_retrieval)

    p = sub.add_parser("eval-generation", help="answer quality benchmark with an LLM judge")
    p.add_argument("--dataset", default="fiqa")
    p.add_argument("--n", type=int, default=100)
    p.add_argument("--abstention-n", type=int, default=50)
    p.add_argument("--k", type=int, default=5)
    p.add_argument("--provider", default=config.LLM_PROVIDER)
    p.add_argument("--model", default=config.LLM_MODEL)
    p.add_argument("--judge-provider", default=config.JUDGE_PROVIDER)
    p.add_argument("--judge-model", default=config.JUDGE_MODEL)
    p.add_argument("--reranker", default=config.DEFAULT_RERANKER)
    p.set_defaults(fn=cmd_eval_generation)

    p = sub.add_parser("ask", help="answer a question with citations")
    p.add_argument("question")
    p.add_argument("--corpus", default="fiqa")
    p.add_argument("--mode", default="hybrid_convex")
    p.add_argument("--k", type=int, default=5)
    p.add_argument("--provider", default=config.LLM_PROVIDER)
    p.add_argument("--model", default=config.LLM_MODEL)
    p.add_argument("--reranker", default=config.DEFAULT_RERANKER)
    p.set_defaults(fn=cmd_ask)

    p = sub.add_parser("ingest", help="index a folder of PDF/Markdown/text files")
    p.add_argument("folder")
    p.add_argument("--name", required=True)
    p.add_argument("--chunk-words", type=int, default=180)
    p.add_argument("--overlap", type=int, default=40)
    p.set_defaults(fn=cmd_ingest)

    p = sub.add_parser("calibrate-judge", help="measure LLM-judge agreement with your own labels")
    p.add_argument("--dataset", default="fiqa")
    p.add_argument("--n", type=int, default=30)
    p.set_defaults(fn=cmd_calibrate_judge)

    args = ap.parse_args(argv)
    args.fn(args)


if __name__ == "__main__":
    sys.exit(main())
