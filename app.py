"""Evident + Strata demo UI:  streamlit run app.py

Locally the default LLM is Ollama. The hosted demo sets EVIDENT_HOSTED=1: it uses Groq (key from the
GROQ_API_KEY secret), limits questions per visitor, and falls back to retrieval-only mode when no key
is configured, so the search and the vector-database playground always work.
"""

from __future__ import annotations

import json
import os
import re
import time
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st

# Streamlit Community Cloud checks the repo out under /mount/src. The hosted settings must be in
# the environment before evident.config is imported.
ON_STREAMLIT_CLOUD = Path(__file__).resolve().as_posix().startswith("/mount/src/")
HOSTED = os.environ.get("EVIDENT_HOSTED") == "1" or ON_STREAMLIT_CLOUD
DEMO_DATA_REPO = "vividh111/evident-demo-data"  # precomputed embeddings (Hugging Face dataset)
if HOSTED:
    os.environ.setdefault("EVIDENT_DATA", str(Path.home() / ".cache" / "evident-demo"))
    os.environ.setdefault("EVIDENT_LLM_PROVIDER", "groq")
    os.environ.setdefault("EVIDENT_LLM_MODEL", "llama-3.1-8b-instant")
    os.environ.setdefault("EVIDENT_THREADS", "2")  # if an index must be built, don't oversubscribe a shared CPU
    try:  # Streamlit secrets -> environment, for the LLM client
        for key in ("GROQ_API_KEY", "GEMINI_API_KEY", "OPENAI_API_KEY"):
            if key in st.secrets and not os.environ.get(key):
                os.environ[key] = str(st.secrets[key])
    except Exception:  # no secrets configured
        pass

from evident import config  # noqa: E402
from evident.data import load_beir  # noqa: E402
from evident.ingest import list_corpora, load_corpus  # noqa: E402
from evident.llm import LLM, PROVIDERS  # noqa: E402
from evident.rag import RAG  # noqa: E402
from evident.retriever import MODES, Retriever  # noqa: E402

HOSTED = os.environ.get("EVIDENT_HOSTED") == "1"
QUESTION_LIMIT = 20  # per browser session on the hosted demo (protects the free API quota)
STRATA_REPO = "https://github.com/VividhDesign/strata"
EVIDENT_REPO = "https://github.com/VividhDesign/evident"
STRATA_IMG = "https://raw.githubusercontent.com/VividhDesign/strata/main/docs/img/{}-light.png"

st.set_page_config(page_title="Evident + Strata demo", layout="wide")


@st.cache_resource(show_spinner="Syncing precomputed embeddings and indexes (first start only)...")
def ensure_demo_data() -> None:
    """On the hosted demo, fetch precomputed embeddings and prebuilt BM25 / Strata indexes instead of
    rebuilding them on a small shared CPU. The BEIR corpora come from their original source on first use."""
    if not HOSTED:
        return
    from huggingface_hub import snapshot_download  # incremental: only fetches files that changed

    snapshot_download(DEMO_DATA_REPO, repo_type="dataset", local_dir=str(config.DATA_DIR),
                      allow_patterns=["cache/*", "indexes/*"])


ensure_demo_data()


# ---------------------------------------------------------------------------------------------
# Cached resources
# ---------------------------------------------------------------------------------------------

@st.cache_resource(show_spinner="Loading corpus and building BM25 + Strata indexes (first visit only)...")
def get_retriever(corpus: str, reranker: str) -> Retriever:
    docs = load_beir(corpus, "test").docs if corpus in config.BEIR_DATASETS else load_corpus(corpus)
    return Retriever(docs, reranker=reranker, alpha=config.tuned_alpha(corpus))


@st.cache_resource
def get_llm(provider: str, model: str):
    """(LLM, None) or (None, reason it is unavailable)."""
    try:
        return LLM(provider, model), None
    except Exception as e:  # missing API key, unknown provider, ...
        return None, str(e)


@st.cache_data
def evaluated_answers() -> list[dict]:
    path = config.RESULTS_DIR / "generation_fiqa_records.jsonl"
    if not path.exists():
        return []
    rows = [json.loads(line) for line in open(path) if line.strip()]
    rows = [r for r in rows if r["condition"] == "hybrid_convex" and not r["abstained"]]
    return sorted(rows, key=lambda r: (-(r["relevance"] or 0), r["question"]))


@st.cache_data(show_spinner="Embedding 200 test queries and timing every ef setting...")
def ef_sweep(corpus: str, reranker: str) -> pd.DataFrame:
    retriever = get_retriever(corpus, reranker)
    queries = sorted(load_beir(corpus, "test").queries.values())[:200]
    qv = retriever.embedder.embed_queries(queries)
    sims = qv @ retriever.doc_vectors.T
    truth = np.argpartition(-sims, 9, axis=1)[:, :10]
    t0 = time.perf_counter()
    for q in qv:  # (results assigned so Streamlit "magic" doesn't render them)
        _ = np.argpartition(-(retriever.doc_vectors @ q), 9)[:10]
    exact_us = (time.perf_counter() - t0) / len(qv) * 1e6
    index = retriever.dense.index
    rows = []
    for ef in (10, 16, 32, 64, 128, 256, 512):
        t0 = time.perf_counter()
        ids, _ = index.search(qv, k=10, ef=ef, num_threads=1)
        us = (time.perf_counter() - t0) / len(qv) * 1e6
        recall = float(np.mean([len(set(a) & set(b)) / 10 for a, b in zip(ids, truth)]))
        rows.append({"ef": ef, "recall@10": round(recall, 4), "µs per query": round(us, 1),
                     "speedup vs exact": round(exact_us / us, 1)})
    return pd.DataFrame(rows)


def render_citations(text: str) -> str:
    return re.sub(r"\[(\d+)\]", r"<sup><b>[\1]</b></sup>", text)


# ---------------------------------------------------------------------------------------------
# Sidebar
# ---------------------------------------------------------------------------------------------

with st.sidebar:
    st.header("Settings")
    corpora = list(config.BEIR_DATASETS) + list_corpora()
    corpus = st.selectbox("Corpus", corpora, index=corpora.index("fiqa"),
                          help="fiqa: 57,638 finance forum answers · scifact: 5,183 science abstracts · "
                               "nfcorpus: 3,633 medical articles")
    mode = st.selectbox("Retrieval", MODES, index=MODES.index("hybrid_convex"),
                        help="hybrid_convex was the best system on all three benchmark datasets")
    k = st.slider("Sources given to the LLM", 1, 10, 5)
    rerankers = ["minilm"] if HOSTED else list(config.RERANK_MODELS)
    reranker = st.selectbox("Reranker (for *_rerank modes)", rerankers, index=0)
    providers = [p for p in PROVIDERS if p != "ollama"] if HOSTED else list(PROVIDERS)
    provider = st.selectbox("LLM provider", providers, index=0)
    default_model = {"ollama": config.LLM_MODEL, "groq": "llama-3.1-8b-instant",
                     "gemini": "gemini-2.5-flash", "openai": "gpt-4o-mini"}[provider]
    model = st.text_input("Model", default_model)
    st.divider()
    st.markdown(f"**Code:** [Evident (RAG)]({EVIDENT_REPO}) · [Strata (vector DB)]({STRATA_REPO})")

ask_tab, strata_tab, bench_tab, how_tab = st.tabs(
    ["Ask", "Vector DB playground (Strata)", "Benchmarks", "How it works"])

# ---------------------------------------------------------------------------------------------
# Ask
# ---------------------------------------------------------------------------------------------

with ask_tab:
    st.title("Evident")
    st.caption("Retrieval-augmented generation where every component is measured · retrieval runs on "
               "Strata, a vector database I wrote from scratch in C++")
    llm, llm_error = get_llm(provider, model)
    if llm is None:
        st.info("No LLM is configured on this deployment, so **retrieval-only mode** is on: you'll see the "
                "sources the system would cite. Fully generated answers from the evaluation are browsable below.")
    question = st.text_input("Question", "Is it better to pay off my student loans early or invest?")
    asked = st.session_state.setdefault("asked", 0)
    if st.button("Ask", type="primary") and question.strip():
        if HOSTED and asked >= QUESTION_LIMIT:
            st.warning(f"This public demo allows {QUESTION_LIMIT} questions per visit. Reload the page to continue.")
        else:
            st.session_state["asked"] = asked + 1
            retriever = get_retriever(corpus, reranker)
            if llm is not None:
                with st.spinner("Retrieving and generating..."):
                    ans = RAG(retriever, llm, mode=mode, k=k).answer(question)
                if ans.abstained:
                    st.warning(ans.answer)
                else:
                    st.markdown(render_citations(ans.answer), unsafe_allow_html=True)
                t = ans.timings
                stages = [("Retrieval", f"{t.get('retrieve.total', 0) * 1000:.0f} ms")]
                if "retrieve.rerank" in t:
                    stages.append(("of which rerank", f"{t['retrieve.rerank'] * 1000:.0f} ms"))
                stages.append(("Generation", "cached" if ans.cached else f"{t.get('generate', 0):.1f} s"))
                for col, (label, value) in zip(st.columns(len(stages)), stages):
                    col.metric(label, value)
                cost = "cost n/a" if ans.cost_usd is None else (
                    "$0 (local model)" if ans.cost_usd == 0 else f"${ans.cost_usd:.6f}")
                st.caption(f"{ans.input_tokens:,} input + {ans.output_tokens:,} output tokens · {cost} · {ans.model}")
                if ans.invalid_citations:
                    st.error(f"{ans.invalid_citations} citation(s) point to sources that do not exist.")
                sources = [(s.number, s.doc_id, s.score, s.text) for s in ans.sources]
            else:
                hits, t = retriever.retrieve(question, mode, k=k)
                st.metric("Retrieval", f"{t['total'] * 1000:.0f} ms")
                sources = [(i + 1, retriever.docs[h.idx].id, h.score, retriever.texts[h.idx]) for i, h in enumerate(hits)]
            st.subheader("Sources")
            for number, doc_id, score, text in sources:
                with st.expander(f"[{number}] {doc_id} · score {score:.3f}", expanded=number <= 2):
                    st.write(text)

    examples = evaluated_answers()
    if examples:
        st.divider()
        st.subheader("Browse evaluated answers")
        st.caption("100 FiQA questions were answered by qwen3.5 and graded claim by claim by a different model "
                   "(gemma4:12b). These are the hybrid-retrieval answers, with the judge's verdicts.")
        choice = st.selectbox("Question", [r["question"] for r in examples], key="example")
        row = next(r for r in examples if r["question"] == choice)
        st.markdown(render_citations(row["answer"]), unsafe_allow_html=True)
        c = st.columns(3)
        c[0].metric("Faithfulness", "—" if row["faithfulness"] is None else f"{row['faithfulness']:.0%}")
        c[1].metric("Relevance", "—" if row["relevance"] is None else f"{row['relevance']:.0f} / 5")
        c[2].metric("Citation precision", "—" if row["citation_precision"] is None else f"{row['citation_precision']:.0%}")
        if row["claims"]:
            st.dataframe(pd.DataFrame([{"claim": cl["claim"], "supported by sources": "yes" if cl["supported"] else "NO"}
                                       for cl in row["claims"]]), hide_index=True, width="stretch")

# ---------------------------------------------------------------------------------------------
# Strata playground
# ---------------------------------------------------------------------------------------------

with strata_tab:
    st.title("Strata vector search, live")
    st.caption("Approximate nearest-neighbour search with an HNSW graph index, written from scratch in C++ "
               "with SIMD kernels. Compare it against exact brute-force search over the same embeddings.")
    retriever = get_retriever(corpus, reranker)
    index = retriever.dense.index
    s = index.stats()
    c = st.columns(5)
    c[0].metric("Vectors", f"{s['size']:,}")
    c[1].metric("Dimensions", index.dim)
    c[2].metric("Graph levels", s["max_level"] + 1)
    c[3].metric("Avg links / node", f"{s['mean_degree_level0']:.1f}")
    c[4].metric("Index memory", f"{s['memory_bytes'] / 2**20:.0f} MiB")

    query = st.text_input("Search query", "how are dividends taxed", key="pg_query")
    ef = st.select_slider("ef: how many candidates the graph search keeps (higher = more accurate, slower)",
                          options=[10, 16, 32, 64, 128, 256, 512], value=64)
    if query.strip():
        qv = retriever.embedder.embed_queries([query])
        t0 = time.perf_counter()
        ids, dist = index.search(qv, k=10, ef=ef, num_threads=1)
        strata_us = (time.perf_counter() - t0) * 1e6
        t0 = time.perf_counter()
        sims = retriever.doc_vectors @ qv[0]
        exact = np.argpartition(-sims, 9)[:10]
        exact_us = (time.perf_counter() - t0) * 1e6
        exact_set = set(exact.tolist())
        found = [int(i) for i in ids[0] if i >= 0]
        c = st.columns(4)
        c[0].metric("Strata search", f"{strata_us:,.0f} µs")
        c[1].metric("Exact search", f"{exact_us:,.0f} µs")
        c[2].metric("Speedup", f"{exact_us / strata_us:.1f}×")
        c[3].metric("Same top-10 as exact", f"{len(exact_set & set(found))} / 10")
        st.dataframe(pd.DataFrame([{
            "rank": r + 1, "doc": retriever.docs[i].id, "similarity": round(1 - float(d), 4),
            "also in exact top-10": "yes" if i in exact_set else "no",
            "text": retriever.texts[i][:180] + ("..." if len(retriever.texts[i]) > 180 else "")}
            for r, (i, d) in enumerate(zip(found, dist[0]))]), hide_index=True, width="stretch")
        st.caption("Timings are single queries on this server, so they jitter; the sweep below averages 200 queries.")

    st.subheader("The accuracy / speed trade-off")
    st.write("Each row runs 200 benchmark queries through Strata at one `ef` setting and checks how many of the "
             "true 10 nearest neighbours (from exact search) it found.")
    if st.button("Run the sweep"):
        import altair as alt

        df = ef_sweep(corpus, reranker)
        st.dataframe(df, hide_index=True, width="stretch")
        plot = df.rename(columns={"µs per query": "us", "recall@10": "recall"})
        base = alt.Chart(plot).encode(
            x=alt.X("us:Q", title="Microseconds per query (lower is faster)", scale=alt.Scale(domain=[0, float(plot.us.max()) * 1.1])),
            y=alt.Y("recall:Q", title="Recall@10 vs exact search", scale=alt.Scale(domain=[float(plot.recall.min()) - 0.03, 1.0])),
            tooltip=["ef", "recall", "us"])
        labels = base.mark_text(dy=-12, fontSize=11).encode(text=alt.Text("ef:Q", format="d"))
        st.altair_chart((base.mark_line(point=True) + labels).properties(height=320,
                        title="Each point is one ef setting (labelled)"), width="stretch")

# ---------------------------------------------------------------------------------------------
# Benchmarks
# ---------------------------------------------------------------------------------------------

with bench_tab:
    st.title("Measured results")
    st.caption("Measured on an Apple M5 Pro. Raw JSON for every number is in the two GitHub repos.")
    st.subheader("Strata vs FAISS vs hnswlib: 1M SIFT vectors")
    st.image(STRATA_IMG.format("sift-recall-qps"))
    st.markdown("At equal recall, Strata matches FAISS and is ~1.9× faster than hnswlib; it builds the 1M-vector "
                "index in 20.5 s on 15 threads. Its SIMD kernels are 11–15× faster than scalar code, and the REST "
                f"server sustains 76K requests/s at 2 ms p99. [Full benchmark write-up]({STRATA_REPO}#benchmarks)")
    c = st.columns(2)
    c[0].image(STRATA_IMG.format("ablation-heuristic"), caption="Neighbour-selection heuristic: +10 points recall")
    c[1].image(STRATA_IMG.format("build-scaling"), caption="Parallel build: 9.6× on 15 threads")
    st.subheader("Evident retrieval quality (BEIR)")
    chart = config.ROOT / "docs" / "img" / "retrieval-ndcg-light.png"
    if chart.exists():
        st.image(str(chart))
    table = config.RESULTS_DIR / "retrieval_table.md"
    if table.exists():
        st.markdown(table.read_text())
    for path in sorted(config.RESULTS_DIR.glob("generation_*_table.md")):
        st.subheader("Answer quality (FiQA, 100 questions)")
        st.markdown(path.read_text())

# ---------------------------------------------------------------------------------------------
# How it works
# ---------------------------------------------------------------------------------------------

with how_tab:
    st.title("How it works")
    st.markdown(f"""
**1. Strata ([code]({STRATA_REPO})).** A vector database written from scratch in C++17:
- an HNSW graph index with NEON/AVX2 SIMD distance kernels;
- a parallel build with 1-byte per-node spinlocks;
- a write-ahead log with crash recovery, metadata filters, a REST server and Python bindings.

**2. Evident ([code]({EVIDENT_REPO})).** For each question, it runs these steps:
1. Embed the question with `bge-small-en-v1.5`.
2. Search Strata for semantically similar passages, and score every passage with a from-scratch BM25.
3. Fuse both rankings with a weight tuned on held-out queries.
4. Pass the top passages to the LLM as numbered sources. The answer must cite them, or say it doesn't know.

**3. Measured, not assumed.**
- Retrieval is benchmarked on 3 BEIR datasets. The BM25 matches Pyserini's published scores within ±0.002.
- Answers are graded claim by claim by a second LLM.
- Neither cross-encoder reranker beat the tuned hybrid, so it isn't the default.

Built by **Vividh Yadav** · B.Tech AI & ML, BIT Mesra.
""")
