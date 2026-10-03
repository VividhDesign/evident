| Condition | n | Context has relevant doc | Abstained | Faithfulness | Fully faithful | Relevance (1-5) | Citation precision | p50 latency |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| BM25 context | 100 | 43% | 16% | 1.00 | 99% | 4.02 | 0.91 | 4.7s |
| Hybrid context (best retriever) | 100 | 57% | 8% | 0.99 | 96% | 4.34 | 0.88 | 5.2s |
| Hybrid, relevant docs hidden | 50 | 0% | 8% | 0.97 | 87% | 4.13 | 0.84 | 3.5s |
