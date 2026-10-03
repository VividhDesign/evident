"""Evident demo UI:  streamlit run app.py"""

from __future__ import annotations

import json
import re

import streamlit as st

from evident import config
from evident.data import load_beir
from evident.ingest import list_corpora, load_corpus
from evident.llm import LLM, PROVIDERS
from evident.rag import RAG
from evident.retriever import MODES, Retriever

st.set_page_config(page_title="Evident", layout="wide")


@st.cache_resource(show_spinner="Building BM25 + Strata indexes (embeddings are cached after the first run)...")
def get_retriever(corpus: str, reranker: str) -> Retriever:
    docs = load_beir(corpus, "test").docs if corpus in config.BEIR_DATASETS else load_corpus(corpus)
    return Retriever(docs, reranker=reranker, alpha=config.tuned_alpha(corpus))


@st.cache_resource
def get_llm(provider: str, model: str) -> LLM:
    return LLM(provider, model)


def render_answer(text: str) -> str:
    return re.sub(r"\[(\d+)\]", r"<sup><b>[\1]</b></sup>", text)


ask_tab, eval_tab = st.tabs(["Ask", "Evaluation results"])

with st.sidebar:
    st.header("Settings")
    corpora = list(config.BEIR_DATASETS) + list_corpora()
    corpus = st.selectbox("Corpus", corpora, index=corpora.index("fiqa"))
    mode = st.selectbox("Retrieval", MODES, index=MODES.index("hybrid_convex"))
    k = st.slider("Sources given to the LLM", 1, 10, 5)
    reranker = st.selectbox("Reranker", list(config.RERANK_MODELS), index=0)
    provider = st.selectbox("LLM provider", list(PROVIDERS), index=0)
    model = st.text_input("Model", config.LLM_MODEL if provider == "ollama" else "llama-3.1-8b-instant")
    st.caption("Hosted providers read their API key from the environment (GROQ_API_KEY, GEMINI_API_KEY, OPENAI_API_KEY).")

with ask_tab:
    st.title("Evident")
    st.caption("Hybrid retrieval (BM25 + Strata vector index, optional cross-encoder) → cited answer")
    question = st.text_input("Question", "Is it better to pay off my student loans early or invest?")
    if st.button("Ask", type="primary") and question.strip():
        retriever = get_retriever(corpus, reranker)
        rag = RAG(retriever, get_llm(provider, model), mode=mode, k=k)
        with st.spinner("Retrieving and generating..."):
            ans = rag.answer(question)
        if ans.abstained:
            st.warning(ans.answer)
        else:
            st.markdown(render_answer(ans.answer), unsafe_allow_html=True)
        t = ans.timings
        stages = [("Retrieval", f"{t.get('retrieve.total', 0) * 1000:.0f} ms")]
        if "retrieve.rerank" in t:
            stages.append(("of which rerank", f"{t['retrieve.rerank'] * 1000:.0f} ms"))
        stages.append(("Generation", "cached" if ans.cached else f"{t.get('generate', 0):.1f} s"))
        for col, (label, value) in zip(st.columns(len(stages)), stages):
            col.metric(label, value)
        cost = "cost n/a (no price configured)" if ans.cost_usd is None else (
            "$0 (local model)" if ans.cost_usd == 0 else f"${ans.cost_usd:.5f}")
        st.caption(f"{ans.input_tokens:,} input + {ans.output_tokens:,} output tokens · {cost} · {ans.model}")
        if ans.invalid_citations:
            st.error(f"{ans.invalid_citations} citation(s) point to sources that do not exist.")
        st.subheader("Sources")
        for s in ans.sources:
            with st.expander(f"[{s.number}] {s.doc_id} · score {s.score:.3f}", expanded=s.number <= 2):
                st.write(s.text)

with eval_tab:
    st.title("Measured results")
    table = config.RESULTS_DIR / "retrieval_table.md"
    if table.exists():
        st.subheader("Retrieval quality (BEIR test sets)")
        chart = config.ROOT / "docs" / "img" / "retrieval-ndcg-light.png"
        if chart.exists():
            st.image(str(chart))
        st.markdown(table.read_text())
    for path in sorted(config.RESULTS_DIR.glob("generation_*_table.md")):
        st.subheader(f"Answer quality ({path.stem.split('_')[1]})")
        st.markdown(path.read_text())
    for path in sorted(config.RESULTS_DIR.glob("retrieval_*.json")):
        with st.expander(f"Raw: {path.name}"):
            st.json(json.loads(path.read_text()))
    if not table.exists():
        st.info("Run `evident eval-retrieval` and `evident eval-generation` to populate this tab.")
