"""Index your own documents: chunk -> store chunks -> embed (cached) on first use."""

from __future__ import annotations

import json
import re

from . import config
from .data import Doc, load_documents
from .embed import Embedder


def _corpus_path(name: str):
    if not re.fullmatch(r"[A-Za-z0-9_-]{1,64}", name):
        raise ValueError("corpus name must match [A-Za-z0-9_-]{1,64}")
    return config.INDEX_DIR / name / "docs.jsonl"


def ingest(folder: str, name: str, max_words: int = 180, overlap: int = 40) -> int:
    docs = load_documents(folder, max_words=max_words, overlap=overlap)
    if not docs:
        raise ValueError(f"no .pdf/.md/.txt content found in {folder}")
    path = _corpus_path(name)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        for d in docs:
            f.write(json.dumps({"id": d.id, "title": d.title, "text": d.text, "metadata": d.metadata}) + "\n")
    Embedder().embed_documents([d.full_text for d in docs])  # warm the embedding cache
    return len(docs)


def load_corpus(name: str) -> list[Doc]:
    path = _corpus_path(name)
    if not path.exists():
        raise FileNotFoundError(f"unknown corpus '{name}' (BEIR: {', '.join(config.BEIR_DATASETS)}; or run `evident ingest`)")
    return [Doc(**json.loads(line)) for line in open(path) if line.strip()]


def list_corpora() -> list[str]:
    if not config.INDEX_DIR.exists():
        return []
    return sorted(p.parent.name for p in config.INDEX_DIR.glob("*/docs.jsonl"))
