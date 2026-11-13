## Benchmark: git-engram vs. standard RAG

Fixture: `shopfront` synthetic monorepo -- 47 commits, 35 files, 179 indexed documents, 9 graveyard failures. Hardware: Darwin arm64, Python 3.14.10. Latency = warm, single-thread wall time per query over 20 repeats.

### Held-out test split v2 (24 unseen queries: 6 causal, 6 regression, 6 coupled-file, 6 exact-identifier)

Committed (with a fixture extension) before the commit slot-budget fix existed; never used to choose any weight or threshold. Half of it targets facts no other split mentions.

| Configuration | Recall@3 | Recall@5 | MRR | nDCG@5 | Neg. Warning Hit | False Alarms | Co-Change Disc. | Index RAM (KB) | P50 (ms) | P95 (ms) |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Dense RAG (float32 cosine) | 0.708 | 0.812 | 0.894 | 0.748 | 100% | 11% | 33% | 268.5 | 0.16 | 0.19 |
| BM25 only (SQLite FTS5) | 0.771 | 0.833 | 0.874 | 0.784 | 83% | 0% | 33% | 55.5 | 0.25 | 0.34 |
| Hybrid (BM25 + float32, RRF) | 0.771 | 0.833 | 0.887 | 0.781 | 100% | 6% | 50% | 324.0 | 0.31 | 0.41 |
| **git-engram** (full) | 0.771 | 0.854 | 0.903 | 0.854 | 100% | 17% | 83% | 112.0 | 1.44 | 1.91 |

| nDCG@5 | causal | regression | coupled | error-code |
|---|---:|---:|---:|---:|
| Dense RAG (float32 cosine) | 0.800 | 0.708 | 0.599 | 0.885 |
| BM25 only (SQLite FTS5) | 0.857 | 0.804 | 0.573 | 0.903 |
| Hybrid (BM25 + float32, RRF) | 0.811 | 0.792 | 0.637 | 0.885 |
| **git-engram** (full) | 0.896 | 0.806 | 0.845 | 0.868 |

Router classes on the v2 split: causal 5, change 6, exact 5, general 7, symptom 1

### Retired held-out split v1 (24 queries)

Unseen when first reported; its per-query results were then used to diagnose commit-capsule crowding, so it is no longer held-out. Shown for continuity only.

| Configuration | Recall@3 | Recall@5 | MRR | nDCG@5 | Neg. Warning Hit | False Alarms | Co-Change Disc. | Index RAM (KB) | P50 (ms) | P95 (ms) |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Dense RAG (float32 cosine) | 0.611 | 0.750 | 0.751 | 0.675 | 100% | 0% | 33% | 268.5 | 0.17 | 0.22 |
| BM25 only (SQLite FTS5) | 0.625 | 0.785 | 0.830 | 0.745 | 80% | 0% | 33% | 55.5 | 0.31 | 0.36 |
| Hybrid (BM25 + float32, RRF) | 0.611 | 0.764 | 0.829 | 0.743 | 80% | 0% | 33% | 324.0 | 0.37 | 0.45 |
| **git-engram** (full) | 0.708 | 0.806 | 0.924 | 0.826 | 100% | 32% | 83% | 112.0 | 1.74 | 2.09 |

| nDCG@5 | causal | regression | coupled | error-code |
|---|---:|---:|---:|---:|
| Dense RAG (float32 cosine) | 0.661 | 0.656 | 0.526 | 0.855 |
| BM25 only (SQLite FTS5) | 0.781 | 0.827 | 0.565 | 0.804 |
| Hybrid (BM25 + float32, RRF) | 0.757 | 0.761 | 0.597 | 0.857 |
| **git-engram** (full) | 0.881 | 0.824 | 0.777 | 0.824 |

### Dev / calibration split (32 queries, 8 per category)

Every weight and threshold was chosen while looking at these queries: scores are optimistic.

| Configuration | Recall@3 | Recall@5 | MRR | nDCG@5 | Neg. Warning Hit | False Alarms | Co-Change Disc. | Index RAM (KB) | P50 (ms) | P95 (ms) |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Dense RAG (float32 cosine) | 0.583 | 0.703 | 0.764 | 0.625 | 87% | 0% | 0% | 268.5 | 0.19 | 0.28 |
| BM25 only (SQLite FTS5) | 0.661 | 0.734 | 0.888 | 0.740 | 67% | 0% | 0% | 55.5 | 0.26 | 0.41 |
| Hybrid (BM25 + float32, RRF) | 0.630 | 0.719 | 0.829 | 0.704 | 73% | 0% | 0% | 324.0 | 0.36 | 0.54 |
| **git-engram** (full) | 0.760 | 0.833 | 0.911 | 0.837 | 87% | 0% | 75% | 112.0 | 1.51 | 2.10 |

| nDCG@5 | causal | regression | coupled | error-code |
|---|---:|---:|---:|---:|
| Dense RAG (float32 cosine) | 0.614 | 0.788 | 0.369 | 0.730 |
| BM25 only (SQLite FTS5) | 0.812 | 0.907 | 0.440 | 0.803 |
| Hybrid (BM25 + float32, RRF) | 0.754 | 0.909 | 0.412 | 0.741 |
| **git-engram** (full) | 0.931 | 0.865 | 0.769 | 0.786 |

### Ablation study (git-engram minus one component)

| Variant | nDCG@5 dev | nDCG@5 v2 | Recall@5 v2 | MRR v2 | Co-Change Disc. v2 | Neg. Warning Hit v2 |
|---|---:|---:|---:|---:|---:|---:|
| git-engram (full) | 0.837 | 0.854 | 0.854 | 0.903 | 83% | 100% |
| - co-change graph path | 0.829 | 0.863 | 0.854 | 0.924 | 83% | 100% |
| - interaction reranker | 0.572 | 0.712 | 0.938 | 0.734 | 83% | 100% |
| - MMR file diversity | 0.836 | 0.866 | 0.938 | 0.899 | 83% | 100% |
| - time decay (plain RRF) | 0.839 | 0.854 | 0.854 | 0.903 | 83% | 100% |
| float32 exact instead of 1-bit+int8 | 0.837 | 0.854 | 0.854 | 0.903 | 83% | 100% |
| - query-adaptive router (fixed weights) | 0.773 | 0.831 | 0.833 | 0.897 | 67% | 100% |
| - commit slot budget (unconstrained top-k) | 0.800 | 0.816 | 0.792 | 0.896 | 50% | 100% |

### `engram_preflight_check` latency: P50 **1.16 ms**, P95 **1.59 ms** (max 1.94 ms) over 7 targets

### Two-stage quantized search at scale (synthetic Dirichlet topic mixtures of the corpus embeddings)

| Vectors | Recall@10 vs float32-384d | Recall@10 vs float32-256d | Two-stage P50 (ms) | Float32 P50 (ms) | Two-stage RAM (MB) | Float32 RAM (MB) |
|---:|---:|---:|---:|---:|---:|---:|
| 10,000 | 0.844 | 0.985 | 0.18 | 0.24 | 2.8 | 14.6 |
| 100,000 | 0.746 | 0.976 | 0.60 | 2.35 | 27.8 | 146.5 |

### Context firewall

12/12 injection payloads detected, 12/12 neutralised (stripped or redacted); 0/12 benign snippets modified, 1/12 flagged.
