"""Answer generation with numbered citations and explicit abstention."""

from __future__ import annotations

import re
import time
from dataclasses import dataclass, field

from .retriever import Retriever

ABSTAIN_TEXT = "I don't have enough information in the provided sources to answer that."

SYSTEM_PROMPT = f"""You answer questions using ONLY the numbered sources provided by the user.
Rules:
1. Every sentence of your answer must end with the number(s) of the source(s) that support it, like [1] or [2][3].
2. Only state what the sources say. Do not add outside knowledge, even if you are sure it is true.
3. If the sources do not contain the information needed to answer, reply with exactly: "{ABSTAIN_TEXT}"
4. Be concise: at most 4 sentences."""

_CITATION = re.compile(r"\[(\d+)\]")
_LEADING_CITATIONS = re.compile(r"^((?:\[\d+\]\s*)+)")
_SENTENCE = re.compile(r"(?<=[.!?])\s+(?=[A-Z0-9\"'(\[])")
_ABSTAIN_PATTERNS = re.compile(
    r"(don.t|do not) have enough information|not enough information|sources do not (contain|provide|mention)|"
    r"cannot answer|can.t answer|unable to answer", re.IGNORECASE)


def truncate_words(text: str, max_words: int) -> str:
    words = text.split()
    return text if len(words) <= max_words else " ".join(words[:max_words]) + " ..."


def build_messages(question: str, passages: list[str]) -> list[dict]:
    sources = "\n\n".join(f"[{i + 1}] {p}" for i, p in enumerate(passages))
    return [{"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": f"Sources:\n{sources}\n\nQuestion: {question}"}]


def is_abstention(answer: str) -> bool:
    return bool(_ABSTAIN_PATTERNS.search(answer)) and len(answer.split()) < 60


def split_sentences(text: str) -> list[str]:
    return [s.strip() for s in _SENTENCE.split(text.strip()) if s.strip()]


def sentence_citations(answer: str) -> list[tuple[str, list[int]]]:
    """[(sentence text without markers, [cited source numbers])].

    Markers placed after the full stop ("... tax free. [1] Next ...") belong to the
    previous sentence, so leading markers are moved back to it.
    """
    out: list[tuple[str, list[int]]] = []
    for s in split_sentences(answer):
        lead = _LEADING_CITATIONS.match(s)
        if lead and out:
            prev_text, prev_cites = out[-1]
            out[-1] = (prev_text, prev_cites + [int(c) for c in _CITATION.findall(lead.group(1))])
            s = s[lead.end():].strip()
            if not s:
                continue
        out.append((_CITATION.sub("", s).strip(), [int(c) for c in _CITATION.findall(s)]))
    return out


@dataclass
class Source:
    number: int
    doc_index: int
    doc_id: str
    text: str
    score: float


@dataclass
class RAGAnswer:
    question: str
    answer: str
    sources: list[Source]
    abstained: bool
    citations: list[tuple[str, list[int]]]
    timings: dict[str, float] = field(default_factory=dict)
    input_tokens: int = 0
    output_tokens: int = 0
    cost_usd: float | None = 0.0
    model: str = ""
    cached: bool = False  # answer served from the LLM response cache

    @property
    def invalid_citations(self) -> int:
        return sum(1 for _, cites in self.citations for c in cites if not 1 <= c <= len(self.sources))

    @property
    def citation_coverage(self) -> float:
        """Fraction of answer sentences carrying at least one valid citation."""
        if not self.citations:
            return 0.0
        ok = sum(1 for _, cites in self.citations if any(1 <= c <= len(self.sources) for c in cites))
        return ok / len(self.citations)


class RAG:
    def __init__(self, retriever: Retriever, llm, mode: str = "hybrid_convex", k: int = 5,
                 max_passage_words: int = 220, max_tokens: int = 400):
        self.retriever, self.llm, self.mode, self.k = retriever, llm, mode, k
        self.max_passage_words, self.max_tokens = max_passage_words, max_tokens

    def answer(self, question: str, exclude: set[int] | None = None, mode: str | None = None) -> RAGAnswer:
        hits, timings = self.retriever.retrieve(question, mode or self.mode, k=self.k, exclude=exclude)
        timings = {f"retrieve.{k}": v for k, v in timings.items()}
        sources = [Source(i + 1, h.idx, self.retriever.docs[h.idx].id,
                          truncate_words(self.retriever.texts[h.idx], self.max_passage_words), h.score)
                   for i, h in enumerate(hits)]
        t0 = time.perf_counter()
        resp = self.llm.chat(build_messages(question, [s.text for s in sources]), max_tokens=self.max_tokens)
        timings["generate"] = resp.latency_s if not getattr(resp, "cached", False) else time.perf_counter() - t0
        text = resp.text.strip()
        return RAGAnswer(question=question, answer=text, sources=sources, abstained=is_abstention(text),
                         citations=sentence_citations(text), timings=timings, input_tokens=resp.input_tokens,
                         output_tokens=resp.output_tokens, cost_usd=self.llm.cost_usd(resp.input_tokens, resp.output_tokens),
                         model=resp.model, cached=bool(getattr(resp, "cached", False)))
