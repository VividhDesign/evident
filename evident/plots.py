"""Charts for the README (light + dark variants):  python -m evident.plots"""

from __future__ import annotations

import json

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from . import config  # noqa: E402

THEMES = {
    "light": {"surface": "#fcfcfb", "text": "#0b0b0b", "muted": "#52514e", "grid": "#e6e5e1",
              "bar": "#2a78d6", "bar_best": "#104281"},
    "dark": {"surface": "#1a1a19", "text": "#ffffff", "muted": "#c3c2b7", "grid": "#33332f",
             "bar": "#3987e5", "bar_best": "#9ec5f4"},
}
SYSTEMS = [
    ("bm25", "BM25 (from scratch)"),
    ("dense", "Dense (Strata HNSW)"),
    ("hybrid", "Hybrid, RRF"),
    ("hybrid_convex", "Hybrid, tuned α"),
    ("hybrid_rerank:minilm", "Hybrid → MiniLM CE"),
    ("hybrid_rerank:bge", "Hybrid → bge reranker"),
]
TITLES = {"scifact": "SciFact (science)", "nfcorpus": "NFCorpus (medical)", "fiqa": "FiQA (finance)"}


def retrieval_chart(mode: str) -> None:
    t = THEMES[mode]
    reports = [json.loads((config.RESULTS_DIR / f"retrieval_{d}.json").read_text()) for d in TITLES]
    fig, axes = plt.subplots(1, 3, figsize=(10.5, 3.6), dpi=150, sharey=True)
    fig.patch.set_facecolor(t["surface"])
    labels = [label for _, label in SYSTEMS][::-1]
    for ax, rep in zip(axes, reports):
        ax.set_facecolor(t["surface"])
        vals = [rep["systems"][key]["ndcg@10"] for key, _ in SYSTEMS][::-1]
        best = max(vals)
        colors = [t["bar_best"] if v == best else t["bar"] for v in vals]
        ax.barh(labels, vals, color=colors, height=0.62, edgecolor=t["surface"], linewidth=2)
        for y, v in enumerate(vals):
            ax.text(v + best * 0.02, y, f"{v:.3f}", va="center", fontsize=8.5, color=t["text"],
                    fontweight="bold" if v == best else "normal")
        ax.set_xlim(0, best * 1.25)
        ax.set_title(TITLES[rep["dataset"]], loc="left", fontsize=10, color=t["text"])
        ax.tick_params(which="both", colors=t["muted"], labelsize=8.5, length=0)
        ax.xaxis.set_visible(False)
        for side in ("top", "right", "bottom"):
            ax.spines[side].set_visible(False)
        ax.spines["left"].set_color(t["grid"])
    fig.suptitle("Retrieval quality, nDCG@10 on BEIR test sets (higher is better)", x=0.01, ha="left",
                 fontsize=11, color=t["text"])
    fig.tight_layout()
    out = config.ROOT / "docs" / "img"
    out.mkdir(parents=True, exist_ok=True)
    fig.savefig(out / f"retrieval-ndcg-{mode}.png", facecolor=t["surface"])
    plt.close(fig)


def main():
    for mode in THEMES:
        retrieval_chart(mode)
    print("charts written to", config.ROOT / "docs" / "img")


if __name__ == "__main__":
    main()
