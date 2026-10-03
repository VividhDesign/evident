"""Paths and defaults. Everything can be overridden with EVIDENT_* environment variables."""

from __future__ import annotations

import os
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = Path(os.environ.get("EVIDENT_DATA", ROOT / "data"))
BEIR_DIR = DATA_DIR / "beir"
CACHE_DIR = DATA_DIR / "cache"
INDEX_DIR = DATA_DIR / "indexes"
RESULTS_DIR = Path(os.environ.get("EVIDENT_RESULTS", ROOT / "results"))

EMBED_MODEL = os.environ.get("EVIDENT_EMBED_MODEL", "BAAI/bge-small-en-v1.5")
RERANK_MODELS = {
    "minilm": "cross-encoder/ms-marco-MiniLM-L-6-v2",
    "bge": "BAAI/bge-reranker-base",
}
DEFAULT_RERANKER = os.environ.get("EVIDENT_RERANKER", "minilm")

# Generation (local by default; see evident/llm.py for hosted providers)
LLM_PROVIDER = os.environ.get("EVIDENT_LLM_PROVIDER", "ollama")
LLM_MODEL = os.environ.get("EVIDENT_LLM_MODEL", "qwen3.5")
JUDGE_PROVIDER = os.environ.get("EVIDENT_JUDGE_PROVIDER", "ollama")
JUDGE_MODEL = os.environ.get("EVIDENT_JUDGE_MODEL", "gemma4:12b")

BEIR_DATASETS = ("scifact", "nfcorpus", "fiqa")
BEIR_URL = "https://public.ukp.informatik.tu-darmstadt.de/thakur/BEIR/datasets/{name}.zip"


def tuned_alpha(dataset: str, default: float = 0.5) -> float:
    """Convex-fusion weight tuned on the dataset's dev/train split by `evident eval-retrieval`."""
    import json

    path = RESULTS_DIR / f"retrieval_{dataset}.json"
    if path.exists():
        return float(json.loads(path.read_text()).get("alpha_tuning", {}).get("best_alpha", default))
    return default
