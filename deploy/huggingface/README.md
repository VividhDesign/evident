---
title: Evident + Strata
emoji: 🔎
colorFrom: blue
colorTo: indigo
sdk: docker
app_port: 7860
pinned: true
license: mit
short_description: Measured RAG on a from-scratch C++ vector database
---

# Evident + Strata: live demo

- **Ask:** cited answers from hybrid retrieval (BM25 + dense) over 57,638 finance documents.
- **Vector DB playground:** live HNSW search in Strata vs exact search, with an accuracy/speed sweep.
- **Benchmarks:** measured results for both projects.

Code: [Evident](https://github.com/VividhDesign/evident) · [Strata](https://github.com/VividhDesign/strata)

To enable generated answers, add a `GROQ_API_KEY` secret in this Space's settings. Without it, the demo runs in
retrieval-only mode.
