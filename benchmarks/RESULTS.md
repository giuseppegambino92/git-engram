## Benchmark: git-engram vs. standard RAG

Fixture: `shopfront` synthetic monorepo -- 58 commits, 42 files, 217 indexed documents, 13 graveyard failures. Hardware: Darwin arm64, Python 3.14.10. Latency = warm, single-thread wall time per query over 20 repeats.

### Held-out test split v4 (30 unseen queries: 6 causal, 6 behavioral, 6 regression, 6 coupled-file, 6 exact-identifier)

Committed (with a rate-limit / signed-webhook fixture extension) before code-first routing and commit-relevance gating existed; never used to choose any weight or threshold. Includes sibling-symbol queries that measure graveyard false alarms.

| Configuration | Recall@3 | Recall@5 | MRR | nDCG@5 | Neg. Warning Hit | False Alarms | Co-Change Disc. | Index RAM (KB) | P50 (ms) | P95 (ms) |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Dense RAG (float32 cosine) | 0.661 | 0.744 | 0.848 | 0.722 | 100% | 18% | 33% | 325.5 | 0.18 | 0.29 |
| BM25 only (SQLite FTS5) | 0.661 | 0.789 | 0.844 | 0.757 | 50% | 14% | 17% | 65.0 | 0.25 | 0.39 |
| Hybrid (BM25 + float32, RRF) | 0.678 | 0.744 | 0.892 | 0.747 | 88% | 14% | 17% | 390.5 | 0.31 | 0.45 |
| **git-engram** (full) | 0.694 | 0.856 | 0.864 | 0.817 | 62% | 14% | 67% | 133.3 | 1.47 | 1.99 |

| nDCG@5 | causal | behavioral | regression | coupled | error-code |
|---|---:|---:|---:|---:|---:|
| Dense RAG (float32 cosine) | 0.824 | 0.991 | 0.610 | 0.481 | 0.703 |
| BM25 only (SQLite FTS5) | 0.888 | 0.889 | 0.764 | 0.440 | 0.806 |
| Hybrid (BM25 + float32, RRF) | 0.870 | 0.907 | 0.713 | 0.442 | 0.803 |
| **git-engram** (full) | 0.888 | 0.989 | 0.696 | 0.652 | 0.859 |

Router classes on the v4 split: behavioral 6, causal 5, change 6, exact 6, general 5, symptom 2

### Retired held-out split v3 (24 queries)

Unseen when first reported; its causal regression and false alarms were then diagnosed.

| Configuration | Recall@3 | Recall@5 | MRR | nDCG@5 | Neg. Warning Hit | False Alarms | Co-Change Disc. | Index RAM (KB) | P50 (ms) | P95 (ms) |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Dense RAG (float32 cosine) | 0.688 | 0.771 | 0.722 | 0.691 | 100% | 24% | 17% | 325.5 | 0.18 | 0.25 |
| BM25 only (SQLite FTS5) | 0.750 | 0.812 | 0.823 | 0.763 | 57% | 12% | 17% | 65.0 | 0.26 | 0.35 |
| Hybrid (BM25 + float32, RRF) | 0.688 | 0.792 | 0.785 | 0.738 | 86% | 12% | 17% | 390.5 | 0.37 | 0.49 |
| **git-engram** (full) | 0.694 | 0.896 | 0.896 | 0.840 | 71% | 18% | 83% | 133.3 | 1.49 | 1.86 |

| nDCG@5 | causal | regression | coupled | error-code |
|---|---:|---:|---:|---:|
| Dense RAG (float32 cosine) | 0.674 | 0.769 | 0.371 | 0.951 |
| BM25 only (SQLite FTS5) | 0.784 | 0.846 | 0.487 | 0.934 |
| Hybrid (BM25 + float32, RRF) | 0.785 | 0.791 | 0.442 | 0.934 |
| **git-engram** (full) | 0.855 | 0.861 | 0.716 | 0.928 |

### Retired held-out split v2 (24 queries)

Unseen when first reported; its false alarms were then diagnosed, so it is no longer held-out.

| Configuration | Recall@3 | Recall@5 | MRR | nDCG@5 | Neg. Warning Hit | False Alarms | Co-Change Disc. | Index RAM (KB) | P50 (ms) | P95 (ms) |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Dense RAG (float32 cosine) | 0.708 | 0.833 | 0.888 | 0.758 | 100% | 6% | 50% | 325.5 | 0.19 | 0.25 |
| BM25 only (SQLite FTS5) | 0.771 | 0.833 | 0.874 | 0.784 | 67% | 6% | 33% | 65.0 | 0.27 | 0.42 |
| Hybrid (BM25 + float32, RRF) | 0.750 | 0.833 | 0.888 | 0.784 | 100% | 6% | 50% | 390.5 | 0.35 | 0.46 |
| **git-engram** (full) | 0.750 | 0.938 | 0.897 | 0.857 | 83% | 6% | 83% | 133.3 | 1.51 | 2.07 |

| nDCG@5 | causal | regression | coupled | error-code |
|---|---:|---:|---:|---:|
| Dense RAG (float32 cosine) | 0.819 | 0.690 | 0.639 | 0.885 |
| BM25 only (SQLite FTS5) | 0.857 | 0.804 | 0.573 | 0.903 |
| Hybrid (BM25 + float32, RRF) | 0.811 | 0.783 | 0.637 | 0.903 |
| **git-engram** (full) | 0.846 | 0.867 | 0.813 | 0.902 |

### Retired held-out split v1 (24 queries)

Unseen when first reported; then used to diagnose commit-capsule crowding. Continuity only.

| Configuration | Recall@3 | Recall@5 | MRR | nDCG@5 | Neg. Warning Hit | False Alarms | Co-Change Disc. | Index RAM (KB) | P50 (ms) | P95 (ms) |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Dense RAG (float32 cosine) | 0.590 | 0.750 | 0.696 | 0.637 | 80% | 5% | 33% | 325.5 | 0.19 | 0.32 |
| BM25 only (SQLite FTS5) | 0.625 | 0.764 | 0.830 | 0.739 | 80% | 0% | 33% | 65.0 | 0.32 | 0.40 |
| Hybrid (BM25 + float32, RRF) | 0.632 | 0.785 | 0.817 | 0.747 | 80% | 0% | 33% | 390.5 | 0.40 | 0.46 |
| **git-engram** (full) | 0.701 | 0.819 | 0.918 | 0.823 | 80% | 5% | 83% | 133.3 | 1.63 | 2.40 |

| nDCG@5 | causal | regression | coupled | error-code |
|---|---:|---:|---:|---:|
| Dense RAG (float32 cosine) | 0.590 | 0.667 | 0.488 | 0.805 |
| BM25 only (SQLite FTS5) | 0.763 | 0.818 | 0.561 | 0.816 |
| Hybrid (BM25 + float32, RRF) | 0.739 | 0.790 | 0.546 | 0.913 |
| **git-engram** (full) | 0.819 | 0.789 | 0.849 | 0.836 |

### Dev / calibration split (40 queries, 8 per category)

Every weight and threshold was chosen while looking at these queries: scores are optimistic.

| Configuration | Recall@3 | Recall@5 | MRR | nDCG@5 | Neg. Warning Hit | False Alarms | Co-Change Disc. | Index RAM (KB) | P50 (ms) | P95 (ms) |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Dense RAG (float32 cosine) | 0.667 | 0.771 | 0.799 | 0.696 | 87% | 4% | 0% | 325.5 | 0.18 | 0.27 |
| BM25 only (SQLite FTS5) | 0.729 | 0.787 | 0.910 | 0.792 | 60% | 0% | 0% | 65.0 | 0.28 | 0.41 |
| Hybrid (BM25 + float32, RRF) | 0.725 | 0.775 | 0.864 | 0.761 | 73% | 4% | 0% | 390.5 | 0.38 | 0.49 |
| **git-engram** (full) | 0.833 | 0.867 | 0.975 | 0.890 | 73% | 0% | 62% | 133.3 | 1.63 | 2.00 |

| nDCG@5 | causal | regression | coupled | error-code | behavioral |
|---|---:|---:|---:|---:|---:|
| Dense RAG (float32 cosine) | 0.663 | 0.788 | 0.397 | 0.770 | 0.862 |
| BM25 only (SQLite FTS5) | 0.812 | 0.907 | 0.440 | 0.803 | 1.000 |
| Hybrid (BM25 + float32, RRF) | 0.764 | 0.871 | 0.437 | 0.779 | 0.954 |
| **git-engram** (full) | 0.912 | 0.913 | 0.783 | 0.844 | 1.000 |

### Ablation study (git-engram minus one component)

| Variant | nDCG@5 dev | nDCG@5 v4 | Recall@5 v4 | MRR v4 | Co-Change v4 | Neg. Hit v4 | False Alarms v4 |
|---|---:|---:|---:|---:|---:|---:|---:|
| git-engram (full) | 0.890 | 0.817 | 0.856 | 0.864 | 67% | 62% | 14% |
| - code-first behavioral route | 0.893 | 0.756 | 0.839 | 0.787 | 67% | 62% | 14% |
| - named-symbol pinning | 0.882 | 0.798 | 0.856 | 0.839 | 67% | 62% | 14% |
| - commit-relevance gate (any top-3 commit) | 0.890 | 0.817 | 0.856 | 0.864 | 67% | 100% | 27% |
| file-level graveyard anchoring (legacy) | 0.890 | 0.817 | 0.856 | 0.864 | 67% | 100% | 36% |
| - commit slot budget (unconstrained top-k) | 0.867 | 0.772 | 0.822 | 0.822 | 50% | 62% | 14% |
| - query-adaptive router (fixed weights) | 0.838 | 0.697 | 0.806 | 0.732 | 50% | 50% | 14% |
| - exact-identifier rerank profile | 0.890 | 0.805 | 0.839 | 0.864 | 67% | 62% | 14% |
| + MMR file diversity (opt-in) | 0.897 | 0.781 | 0.778 | 0.864 | 67% | 62% | 14% |
| - co-change graph path | 0.865 | 0.854 | 0.856 | 0.944 | 50% | 75% | 14% |
| - interaction reranker | 0.654 | 0.716 | 0.828 | 0.744 | 50% | 62% | 9% |
| - time decay (plain RRF) | 0.891 | 0.800 | 0.822 | 0.864 | 50% | 62% | 14% |
| float32 exact instead of 1-bit+int8 | 0.890 | 0.817 | 0.856 | 0.864 | 67% | 62% | 14% |

### `engram_preflight_check` latency: P50 **1.34 ms**, P95 **1.84 ms** (max 2.34 ms) over 7 targets

### Two-stage quantized search at scale (synthetic Dirichlet topic mixtures of the corpus embeddings)

| Vectors | Recall@10 vs float32-384d | Recall@10 vs float32-256d | Two-stage P50 (ms) | Float32 P50 (ms) | Two-stage RAM (MB) | Float32 RAM (MB) |
|---:|---:|---:|---:|---:|---:|---:|
| 10,000 | 0.870 | 0.985 | 0.20 | 0.25 | 2.8 | 14.6 |
| 100,000 | 0.726 | 0.976 | 0.65 | 2.52 | 27.8 | 146.5 |

### Context firewall

12/12 injection payloads detected, 12/12 neutralised (stripped or redacted); 0/12 benign snippets modified, 1/12 flagged.
