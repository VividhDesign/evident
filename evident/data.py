"""Corpora: BEIR benchmark datasets and user documents (PDF / Markdown / text)."""

from __future__ import annotations

import csv
import io
import json
import re
import urllib.request
import zipfile
from dataclasses import dataclass, field
from pathlib import Path

from . import config


@dataclass
class Doc:
    id: str
    text: str
    title: str = ""
    metadata: dict = field(default_factory=dict)

    @property
    def full_text(self) -> str:
        return f"{self.title}\n{self.text}".strip() if self.title else self.text


@dataclass
class Dataset:
    name: str
    docs: list[Doc]
    queries: dict[str, str]  # query id -> text (only queries that have qrels in `split`)
    qrels: dict[str, dict[str, int]]  # query id -> {doc id: relevance}
    split: str

    def __post_init__(self):
        self.doc_index = {d.id: i for i, d in enumerate(self.docs)}


def download_beir(name: str) -> Path:
    target = config.BEIR_DIR / name
    if (target / "corpus.jsonl").exists():
        return target
    config.BEIR_DIR.mkdir(parents=True, exist_ok=True)
    url = config.BEIR_URL.format(name=name)
    print(f"downloading {url} ...")
    with urllib.request.urlopen(url) as resp:
        zipfile.ZipFile(io.BytesIO(resp.read())).extractall(config.BEIR_DIR)
    return target


def _read_qrels(path: Path) -> dict[str, dict[str, int]]:
    qrels: dict[str, dict[str, int]] = {}
    with open(path, newline="") as f:
        reader = csv.reader(f, delimiter="\t")
        next(reader)  # header
        for qid, did, score in reader:
            qrels.setdefault(qid, {})[did] = int(score)
    return qrels


def load_beir(name: str, split: str = "test") -> Dataset:
    root = download_beir(name)
    docs = []
    with open(root / "corpus.jsonl") as f:
        for line in f:
            d = json.loads(line)
            docs.append(Doc(id=str(d["_id"]), text=d.get("text", ""), title=d.get("title", ""),
                            metadata={"source": name}))
    qrels_path = root / "qrels" / f"{split}.tsv"
    if not qrels_path.exists():
        raise FileNotFoundError(f"{name} has no '{split}' split")
    qrels = _read_qrels(qrels_path)
    queries = {}
    with open(root / "queries.jsonl") as f:
        for line in f:
            q = json.loads(line)
            if str(q["_id"]) in qrels:
                queries[str(q["_id"])] = q["text"]
    qrels = {q: r for q, r in qrels.items() if q in queries}
    return Dataset(name=name, docs=docs, queries=queries, qrels=qrels, split=split)


# --- user documents -------------------------------------------------------------------------

_SENTENCE_END = re.compile(r"(?<=[.!?])\s+")


def chunk_text(text: str, max_words: int = 180, overlap: int = 40) -> list[str]:
    """Splits text into ~max_words chunks on paragraph, then sentence boundaries.

    Consecutive chunks share `overlap` words so a fact straddling a boundary is still
    retrievable from at least one chunk. A single sentence longer than max_words is split
    on word boundaries.
    """
    if overlap >= max_words:
        raise ValueError("overlap must be smaller than max_words")
    units: list[str] = []
    for para in re.split(r"\n\s*\n", text):
        para = " ".join(para.split())
        if not para:
            continue
        for sent in _SENTENCE_END.split(para):
            words = sent.split()
            for i in range(0, len(words), max_words):
                units.append(" ".join(words[i:i + max_words]))

    chunks: list[str] = []
    current: list[str] = []
    for unit in units:
        words = unit.split()
        if current and len(current) + len(words) > max_words:
            chunks.append(" ".join(current))
            current = current[-overlap:] if overlap else []
        current.extend(words)
    if current and (not chunks or " ".join(current) != chunks[-1]):
        chunks.append(" ".join(current))
    return chunks


def load_documents(folder: str | Path, max_words: int = 180, overlap: int = 40) -> list[Doc]:
    """Reads .pdf, .md and .txt files under `folder` into chunked Docs."""
    folder = Path(folder)
    docs: list[Doc] = []
    for path in sorted(folder.rglob("*")):
        suffix = path.suffix.lower()
        if suffix == ".pdf":
            from pypdf import PdfReader

            pages = [(i + 1, p.extract_text() or "") for i, p in enumerate(PdfReader(path).pages)]
        elif suffix in (".md", ".txt", ".markdown"):
            pages = [(None, path.read_text(errors="ignore"))]
        else:
            continue
        for page, text in pages:
            for j, chunk in enumerate(chunk_text(text, max_words, overlap)):
                doc_id = f"{path.relative_to(folder)}{f'#p{page}' if page else ''}#{j}"
                docs.append(Doc(id=doc_id, text=chunk, title=path.stem,
                                metadata={"source": str(path.relative_to(folder)), "page": page}))
    return docs
