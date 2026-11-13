## Benchmark: git-engram vs. standard RAG

Fixture: `shopfront` synthetic monorepo -- 37 commits, 31 files, 153 indexed documents, 7 graveyard failures, 32 ground-truth queries (8 causal, 8 regression, 8 coupled-file, 8 exact-identifier). Hardware: Darwin arm64, Python 3.14.10. Latency = warm, single-thread wall time per query over 20 repeats.

| Configuration | Recall@3 | Recall@5 | MRR | nDCG@5 | Neg. Warning Hit | False Alarms | Co-Change Disc. | Index RAM (KB) | P50 (ms) | P95 (ms) |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Dense RAG (float32 cosine) | 0.583 | 0.703 | 0.764 | 0.629 | 87% | 6% | 0% | 229.5 | 0.23 | 0.37 |
| BM25 only (SQLite FTS5) | 0.651 | 0.734 | 0.885 | 0.739 | 67% | 0% | 0% | 47.0 | 0.31 | 0.49 |
| Hybrid (BM25 + float32, RRF) | 0.630 | 0.719 | 0.829 | 0.704 | 73% | 6% | 0% | 276.5 | 0.40 | 0.63 |
| **git-engram** (full) | 0.740 | 0.781 | 0.878 | 0.793 | 93% | 6% | 62% | 95.5 | 1.76 | 2.65 |

### nDCG@5 by query category

| Configuration | causal | regression | coupled | error-code |
|---|---:|---:|---:|---:|
| Dense RAG (float32 cosine) | 0.628 | 0.788 | 0.370 | 0.730 |
| BM25 only (SQLite FTS5) | 0.812 | 0.907 | 0.435 | 0.803 |
| Hybrid (BM25 + float32, RRF) | 0.754 | 0.909 | 0.412 | 0.741 |
| **git-engram** (full) | 0.948 | 0.860 | 0.579 | 0.786 |

### Ablation study (git-engram minus one component)

| Variant | Recall@5 | MRR | nDCG@5 | Co-Change Disc. | Neg. Warning Hit |
|---|---:|---:|---:|---:|---:|
| git-engram (full) | 0.781 | 0.878 | 0.793 | 62% | 93% |
| - co-change graph path | 0.740 | 0.896 | 0.777 | 12% | 93% |
| - interaction reranker | 0.734 | 0.707 | 0.621 | 62% | 73% |
| - MMR file diversity | 0.781 | 0.885 | 0.797 | 38% | 93% |
| - time decay (plain RRF) | 0.724 | 0.855 | 0.760 | 62% | 93% |
| float32 exact instead of 1-bit+int8 | 0.781 | 0.878 | 0.793 | 62% | 93% |

### `engram_preflight_check` latency: P50 **1.40 ms**, P95 **1.92 ms** (max 3.18 ms) over 7 targets

### Two-stage quantized search at scale (synthetic Dirichlet topic mixtures of the corpus embeddings)

| Vectors | Recall@10 vs float32-384d | Recall@10 vs float32-256d | Two-stage P50 (ms) | Float32 P50 (ms) | Two-stage RAM (MB) | Float32 RAM (MB) |
|---:|---:|---:|---:|---:|---:|---:|
| 10,000 | 0.831 | 0.982 | 0.20 | 0.27 | 2.8 | 14.6 |
| 100,000 | 0.738 | 0.968 | 0.66 | 2.53 | 27.8 | 146.5 |

### Context firewall

12/12 injection payloads detected, 12/12 neutralised (stripped or redacted); 0/12 benign snippets modified, 1/12 flagged.
