"""LLM-as-judge metrics for generated answers.

  faithfulness        share of the answer's atomic claims that the sources support
                      (claim decomposition + verification, as in RAGAS)
  answer relevance    1-5 rubric: does the answer address the question (correctness aside)
  citation precision  share of (sentence, cited source) pairs where that source supports
                      that sentence

The judge should be a different model family from the generator to limit self-preference
bias (default: generator qwen3.5, judge gemma4:12b). Use `evident calibrate-judge` to hand-label
a sample and measure judge/human agreement before trusting the numbers.
"""

from __future__ import annotations

from dataclasses import dataclass

from .llm import extract_json
from .rag import RAGAnswer

FAITHFULNESS = """You check whether an answer is supported by source passages.

Sources:
{sources}

Answer:
{answer}

Split the answer into its individual factual claims (ignore citation markers such as [1]).
For each claim decide whether the sources explicitly state or directly imply it. A claim that
adds anything the sources do not say is NOT supported, even if it is true in general.
Respond with JSON only, in this form:
{{"claims": [{{"claim": "<claim>", "supported": true}}]}}"""

RELEVANCE = """Question: {question}

Answer: {answer}

How directly and completely does the answer address the question? Ignore whether it is true.
5 = fully answers it; 4 = mostly; 3 = partially; 2 = barely; 1 = does not answer it.
Respond with JSON only: {{"score": <1-5>, "reason": "<one short sentence>"}}"""

CITATIONS = """For each numbered pair below, decide whether the SOURCE supports the STATEMENT
(states it or directly implies it).

{pairs}

Respond with JSON only, one verdict per pair, in this form:
{{"verdicts": [{{"pair": 1, "supported": true}}]}}"""


@dataclass
class Judgement:
    faithfulness: float | None
    claims: list[dict]
    relevance: float | None
    relevance_reason: str
    citation_precision: float | None
    citation_pairs: int


class Judge:
    def __init__(self, llm):
        self.llm = llm

    def _json(self, prompt: str, max_tokens: int = 700):
        resp = self.llm.chat([{"role": "user", "content": prompt}], max_tokens=max_tokens, json_mode=True)
        return extract_json(resp.text) or {}

    def faithfulness(self, ans: RAGAnswer) -> tuple[float | None, list[dict]]:
        sources = "\n\n".join(f"[{s.number}] {s.text}" for s in ans.sources)
        data = self._json(FAITHFULNESS.format(sources=sources, answer=ans.answer), max_tokens=900)
        claims = [c for c in data.get("claims", []) if isinstance(c, dict) and "supported" in c]
        if not claims:
            return None, []
        return sum(bool(c["supported"]) for c in claims) / len(claims), claims

    def relevance(self, ans: RAGAnswer) -> tuple[float | None, str]:
        data = self._json(RELEVANCE.format(question=ans.question, answer=ans.answer), max_tokens=150)
        try:
            score = float(data["score"])
        except (KeyError, TypeError, ValueError):
            return None, ""
        return min(max(score, 1.0), 5.0), str(data.get("reason", ""))

    def citation_precision(self, ans: RAGAnswer) -> tuple[float | None, int]:
        pairs = [(sentence, ans.sources[c - 1]) for sentence, cites in ans.citations
                 for c in dict.fromkeys(cites) if 1 <= c <= len(ans.sources) and sentence]
        if not pairs:
            return None, 0
        listing = "\n\n".join(f"PAIR {i + 1}\nSTATEMENT: {s}\nSOURCE: {src.text}" for i, (s, src) in enumerate(pairs))
        data = self._json(CITATIONS.format(pairs=listing), max_tokens=60 + 25 * len(pairs))
        verdicts = {}
        for v in data.get("verdicts", []):
            if isinstance(v, dict) and "pair" in v and "supported" in v:
                try:
                    verdicts[int(v["pair"])] = bool(v["supported"])
                except (TypeError, ValueError):
                    continue
        judged = [verdicts[i + 1] for i in range(len(pairs)) if i + 1 in verdicts]
        if not judged:
            return None, len(pairs)
        return sum(judged) / len(judged), len(pairs)

    def judge(self, ans: RAGAnswer) -> Judgement:
        if ans.abstained:
            return Judgement(None, [], None, "abstained", None, 0)
        faith, claims = self.faithfulness(ans)
        rel, reason = self.relevance(ans)
        cp, n_pairs = self.citation_precision(ans)
        return Judgement(faith, claims, rel, reason, cp, n_pairs)
