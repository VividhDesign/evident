# Evaluation methodology

Every number in the README comes from `evident eval-retrieval` or `evident eval-generation`. This page explains
what each number measures, how it is computed, and where it can mislead.

## Retrieval

**Data.** Three BEIR test sets from different domains, each with human relevance judgements (qrels):

| Dataset | Domain | Docs | Test queries | Query style |
|---|---|---:|---:|---|
| SciFact | biomedical claims | 5,183 | 300 | a scientific claim; find abstracts that support or refute it |
| NFCorpus | nutrition / medicine | 3,633 | 323 | lay health questions; graded relevance (1-2) |
| FiQA-2018 | personal finance | 57,638 | 648 | forum questions; answers are other users' posts |

**Metrics.**
- nDCG@10 is the primary BEIR metric. It uses trec_eval conventions: linear gain, log2 discount, and the ideal
  ranking built from all judged-relevant documents.
- Recall@100 measures how much a reranker *could* recover from the candidate set.
- MRR@10 is also computed.

All metrics are implemented in `evident/metrics.py` and unit-tested against hand-computed values.

**Systems.**
- BM25 is written from scratch: Lucene-style IDF, k1=0.9, b=0.4, Lucene's English stopwords, Snowball stemming.
- Dense retrieval uses `BAAI/bge-small-en-v1.5` (33M parameters, 384-d) with the query instruction prefix, served
  from the Strata HNSW index (M=16, efC=200, ef=128).
- Two hybrids:
  - **RRF** (k=60), which needs no tuning;
  - a **convex combination** of min-max-normalised scores. Its weight α is grid-searched on the **dev split**
    (the train split for SciFact, which has no dev), then frozen for the test run. Tuning on test would inflate the
    score.
- Two cross-encoder rerankers, applied to the top 50 candidates:
  - `ms-marco-MiniLM-L-6-v2` (22M parameters);
  - `bge-reranker-base` (278M parameters).

**Sanity checks against published numbers.**

| | SciFact | NFCorpus | FiQA |
|---|---:|---:|---:|
| Our BM25 (from scratch) | 0.680 | 0.321 | 0.238 |
| Pyserini BM25 "flat" ([Kamalloo et al., 2023](https://arxiv.org/abs/2306.07471)) | 0.679 | 0.322 | 0.236 |
| Our BM25 → MiniLM cross-encoder (top-50) | 0.687 | 0.350 | 0.329 |
| BEIR paper BM25+CE (top-100) ([Thakur et al., 2021](https://arxiv.org/abs/2104.08663)) | 0.688 | 0.350 | 0.347 |

Our BM25 is within ±0.002 of Pyserini on all three datasets. The reranked numbers match on two datasets; on FiQA
we rerank 50 candidates instead of 100, so fewer relevant documents reach the reranker. This agreement is strong
evidence that the tokenizer, the BM25 math and the metric code are right.

**Approximate-search cost.** The same embeddings are searched exactly (brute-force matrix product) and with
Strata. We report the top-10 overlap and both nDCG values, so any quality lost to the approximate index is
measured rather than assumed.

**Latency.**
- Per-stage wall-clock times come from 100 single-query requests after a warm-up, on an Apple M5 Pro.
- Embedding and cross-encoder inference run on the Apple GPU (PyTorch MPS).
- We report p50/p95 totals and the mean per stage. Batch evaluation runs are timed separately and are *not*
  latency numbers.

## Generation

**Setup.**
- 100 FiQA test questions, sampled with a fixed seed.
- Top-5 sources, each truncated to 220 words.
- Generator: `qwen3.5` (9.7B) via Ollama, temperature 0, thinking disabled.
- Judge: `gemma4:12b`. It is a **different model family** from the generator, to reduce the known bias of judges
  toward their own outputs.

**Prompt contract.** Every sentence must cite sources like `[2]`. Only facts in the sources may be used. If the
sources don't contain the answer, the model must reply with a fixed abstention sentence.

**Conditions.**
1. `bm25`: the context comes from BM25.
2. `hybrid_convex`: the context comes from the best retriever in the retrieval benchmark (BM25 + dense, α tuned
   on FiQA dev).
3. `evidence_removed`: same as 2, but every document judged relevant for the question is hidden from retrieval.
   That uses Strata's deny-filter for dense search and a mask for BM25. A well-grounded system should abstain
   more often here instead of inventing an answer.

All answers are generated first, then all are judged. The two models don't fit in GPU memory together, and
alternating between them would reload a model on every call and inflate the measured generation latency.

**Metrics.**

| Metric | How it's computed | Judge calls |
|---|---|---|
| Context hit rate | share of questions where at least one judged-relevant doc is among the 5 sources | none (uses qrels) |
| Abstention rate | share of answers that are the abstention sentence (regex) | none |
| Faithfulness | the judge splits the answer into atomic claims and marks each as supported or not by the sources; score = supported / total. Averaged over non-abstaining answers | 1 per answer |
| Fully faithful rate | share of answers with no unsupported claim | — |
| Answer relevance | 1-5 rubric: does the answer address the question (correctness aside) | 1 per answer |
| Citation precision | for every (sentence, cited source) pair, does that source support that sentence? | 1 per answer (batched pairs) |
| Citation coverage / invalid citations | share of sentences with a valid citation; citations to sources that don't exist | none |
| Latency | retrieval and generation wall-clock, p50/p95 | — |
| Cost | mean tokens per query × published API prices (`evident/prices.json`) | — |

## Known limitations (read before quoting numbers)

- **Judge reliability.** An LLM judge is a model, not ground truth. `evident calibrate-judge` lets you hand-label
  a sample of claims and measure judge/human agreement. Do that before relying on small differences.
- **"Evidence removed" is approximate.** FiQA has many near-duplicate posts. Hiding the *judged* relevant ones
  does not guarantee the remaining corpus lacks the answer, so abstention here is expected to rise, not to reach
  100%.
- **Sample size.** With 100 questions, a difference of a few points is within noise. The per-question records
  (`results/generation_fiqa_records.jsonl`) allow paired comparisons and bootstrap confidence intervals.
- **FiQA questions are often opinion-seeking** ("is it smart to..."), so a perfect answer is a balanced summary of
  the sources, not a single fact.
