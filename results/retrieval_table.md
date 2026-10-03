| System | scifact nDCG@10 | nfcorpus nDCG@10 | fiqa nDCG@10 | scifact R@100 | nfcorpus R@100 | fiqa R@100 | p50 latency (fiqa) |
|---|---:|---:|---:|---:|---:|---:|---:|
| BM25 (from scratch) | 0.680 | 0.321 | 0.238 | 0.922 | 0.246 | 0.537 | 0.4 ms |
| Dense: bge-small + Strata HNSW | 0.713 | 0.345 | 0.400 | 0.945 | 0.309 | 0.686 | 5.9 ms |
| Hybrid (RRF) | 0.727 | 0.361 | 0.360 | 0.965 | 0.313 | 0.691 | 6.6 ms |
| Hybrid (convex, α tuned on dev) | **0.733** | **0.365** | **0.415** | 0.963 | 0.316 | 0.689 | 6.9 ms |
| BM25 → MiniLM cross-encoder | 0.687 | 0.350 | 0.329 | 0.922 | 0.246 | 0.537 | 192.1 ms |
| Hybrid → MiniLM cross-encoder | 0.694 | 0.357 | 0.372 | 0.965 | 0.313 | 0.691 | 203.5 ms |
| BM25 → bge-reranker-base | 0.710 | 0.321 | 0.315 | 0.922 | 0.246 | 0.537 | 1179.6 ms |
| Hybrid → bge-reranker-base | 0.718 | 0.324 | 0.347 | 0.965 | 0.313 | 0.691 | 1160.0 ms |
