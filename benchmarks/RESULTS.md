## Benchmark: git-engram vs. standard RAG

Fixture: `shopfront` synthetic monorepo -- 37 commits, 31 files, 153 indexed documents, 7 graveyard failures. Hardware: Darwin arm64, Python 3.14.10. Latency = warm, single-thread wall time per query over 20 repeats.

### Held-out test split (24 unseen queries: 6 causal, 6 regression, 6 coupled-file, 6 exact-identifier)

Written and committed before the query router existed; never used to choose any weight or threshold.

| Configuration | Recall@3 | Recall@5 | MRR | nDCG@5 | Neg. Warning Hit | False Alarms | Co-Change Disc. | Index RAM (KB) | P50 (ms) | P95 (ms) |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Dense RAG (float32 cosine) | 0.632 | 0.771 | 0.751 | 0.682 | 100% | 5% | 33% | 229.5 | 0.19 | 0.27 |
| BM25 only (SQLite FTS5) | 0.625 | 0.785 | 0.830 | 0.743 | 80% | 0% | 33% | 47.0 | 0.47 | 0.65 |
| Hybrid (BM25 + float32, RRF) | 0.632 | 0.764 | 0.801 | 0.733 | 80% | 5% | 33% | 276.5 | 0.65 | 0.89 |
| **git-engram** (full) | 0.708 | 0.722 | 0.931 | 0.792 | 100% | 0% | 67% | 95.5 | 2.68 | 3.93 |

| nDCG@5 | causal | regression | coupled | error-code |
|---|---:|---:|---:|---:|
| Dense RAG (float32 cosine) | 0.661 | 0.685 | 0.526 | 0.857 |
| BM25 only (SQLite FTS5) | 0.781 | 0.827 | 0.565 | 0.800 |
| Hybrid (BM25 + float32, RRF) | 0.757 | 0.767 | 0.546 | 0.863 |
| **git-engram** (full) | 0.902 | 0.786 | 0.667 | 0.812 |

Router classes on the test split: causal 6, change 6, exact 7, general 1, symptom 4

### Dev / calibration split (32 queries, 8 per category)

Every weight and threshold was chosen while looking at these queries: scores are optimistic.

| Configuration | Recall@3 | Recall@5 | MRR | nDCG@5 | Neg. Warning Hit | False Alarms | Co-Change Disc. | Index RAM (KB) | P50 (ms) | P95 (ms) |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Dense RAG (float32 cosine) | 0.583 | 0.703 | 0.764 | 0.629 | 87% | 6% | 0% | 229.5 | 0.23 | 0.30 |
| BM25 only (SQLite FTS5) | 0.651 | 0.734 | 0.885 | 0.739 | 67% | 0% | 0% | 47.0 | 0.43 | 0.59 |
| Hybrid (BM25 + float32, RRF) | 0.630 | 0.719 | 0.829 | 0.704 | 73% | 6% | 0% | 276.5 | 0.50 | 0.67 |
| **git-engram** (full) | 0.755 | 0.807 | 0.924 | 0.829 | 93% | 6% | 75% | 95.5 | 1.85 | 2.39 |

| nDCG@5 | causal | regression | coupled | error-code |
|---|---:|---:|---:|---:|
| Dense RAG (float32 cosine) | 0.628 | 0.788 | 0.370 | 0.730 |
| BM25 only (SQLite FTS5) | 0.812 | 0.907 | 0.435 | 0.803 |
| Hybrid (BM25 + float32, RRF) | 0.754 | 0.909 | 0.412 | 0.741 |
| **git-engram** (full) | 0.948 | 0.913 | 0.671 | 0.786 |

### Ablation study (git-engram minus one component)

| Variant | nDCG@5 dev | nDCG@5 test | Recall@5 test | MRR test | Co-Change Disc. test | Neg. Warning Hit test |
|---|---:|---:|---:|---:|---:|---:|
| git-engram (full) | 0.829 | 0.792 | 0.722 | 0.931 | 67% | 100% |
| - co-change graph path | 0.785 | 0.790 | 0.729 | 0.918 | 50% | 100% |
| - interaction reranker | 0.647 | 0.464 | 0.597 | 0.521 | 67% | 80% |
| - MMR file diversity | 0.831 | 0.825 | 0.778 | 0.931 | 67% | 100% |
| - time decay (plain RRF) | 0.802 | 0.806 | 0.743 | 0.944 | 83% | 100% |
| float32 exact instead of 1-bit+int8 | 0.829 | 0.792 | 0.722 | 0.931 | 67% | 100% |
| - query-adaptive router (fixed weights) | 0.793 | 0.759 | 0.729 | 0.882 | 67% | 100% |

### `engram_preflight_check` latency: P50 **2.29 ms**, P95 **4.15 ms** (max 4.44 ms) over 7 targets

### Two-stage quantized search at scale (synthetic Dirichlet topic mixtures of the corpus embeddings)

| Vectors | Recall@10 vs float32-384d | Recall@10 vs float32-256d | Two-stage P50 (ms) | Float32 P50 (ms) | Two-stage RAM (MB) | Float32 RAM (MB) |
|---:|---:|---:|---:|---:|---:|---:|
| 10,000 | 0.831 | 0.982 | 0.26 | 0.35 | 2.8 | 14.6 |
| 100,000 | 0.738 | 0.968 | 0.98 | 3.79 | 27.8 | 146.5 |

### Context firewall

12/12 injection payloads detected, 12/12 neutralised (stripped or redacted); 0/12 benign snippets modified, 1/12 flagged.
