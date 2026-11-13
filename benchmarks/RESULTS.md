## Benchmark: git-engram vs. standard RAG

Fixture: `shopfront` synthetic monorepo -- 65 commits, 45 files, 240 indexed documents, 15 graveyard failures. Hardware: Darwin arm64, Python 3.14.10. Latency = warm, single-thread wall time per query over 20 repeats.

### Held-out test split v5 (30 unseen queries: 6 causal, 6 behavioral, 6 natural-language bug reports, 6 coupled-file, 6 exact-identifier)

Committed (with a promotions / CSV-export fixture extension) before the direct top-commit warning trigger and broadened symptom routing existed; never used to choose any weight or threshold. Includes sibling-symbol queries that measure graveyard false alarms.

| Configuration | Recall@3 | Recall@5 | MRR | nDCG@5 | Neg. Warning Hit | False Alarms | Co-Change Disc. | Index RAM (KB) | P50 (ms) | P95 (ms) |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Dense RAG (float32 cosine) | 0.778 | 0.817 | 0.850 | 0.771 | 100% | 12% | 67% | 360.0 | 0.16 | 0.23 |
| BM25 only (SQLite FTS5) | 0.756 | 0.817 | 0.874 | 0.803 | 83% | 0% | 67% | 74.0 | 0.30 | 0.40 |
| Hybrid (BM25 + float32, RRF) | 0.761 | 0.833 | 0.899 | 0.826 | 83% | 4% | 67% | 434.0 | 0.38 | 0.56 |
| **git-engram** (full) | 0.800 | 0.872 | 0.967 | 0.902 | 100% | 8% | 83% | 149.4 | 1.60 | 2.18 |

| nDCG@5 | causal | behavioral | regression | coupled | error-code |
|---|---:|---:|---:|---:|---:|
| Dense RAG (float32 cosine) | 0.754 | 1.000 | 0.764 | 0.564 | 0.774 |
| BM25 only (SQLite FTS5) | 0.902 | 0.929 | 0.828 | 0.468 | 0.889 |
| Hybrid (BM25 + float32, RRF) | 0.922 | 0.994 | 0.802 | 0.526 | 0.886 |
| **git-engram** (full) | 0.942 | 0.991 | 0.878 | 0.824 | 0.874 |

Router classes on the v5 split: behavioral 6, causal 5, change 5, exact 5, general 3, symptom 6

### Retired held-out split v4 (30 queries)

Unseen when first reported; its warning-recall drop and regression gap were then diagnosed.

| Configuration | Recall@3 | Recall@5 | MRR | nDCG@5 | Neg. Warning Hit | False Alarms | Co-Change Disc. | Index RAM (KB) | P50 (ms) | P95 (ms) |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Dense RAG (float32 cosine) | 0.661 | 0.744 | 0.826 | 0.708 | 100% | 23% | 33% | 360.0 | 0.18 | 0.27 |
| BM25 only (SQLite FTS5) | 0.644 | 0.789 | 0.835 | 0.753 | 50% | 18% | 17% | 74.0 | 0.33 | 0.49 |
| Hybrid (BM25 + float32, RRF) | 0.678 | 0.744 | 0.892 | 0.734 | 88% | 14% | 17% | 434.0 | 0.38 | 0.56 |
| **git-engram** (full) | 0.683 | 0.844 | 0.900 | 0.832 | 100% | 14% | 83% | 149.4 | 1.65 | 2.11 |

| nDCG@5 | causal | behavioral | regression | coupled | error-code |
|---|---:|---:|---:|---:|---:|
| Dense RAG (float32 cosine) | 0.824 | 0.991 | 0.592 | 0.430 | 0.703 |
| BM25 only (SQLite FTS5) | 0.888 | 0.889 | 0.764 | 0.419 | 0.806 |
| Hybrid (BM25 + float32, RRF) | 0.868 | 0.907 | 0.679 | 0.415 | 0.803 |
| **git-engram** (full) | 0.888 | 0.989 | 0.824 | 0.654 | 0.806 |

### Retired held-out split v3 (24 queries)

Unseen when first reported; its causal regression and false alarms were then diagnosed.

| Configuration | Recall@3 | Recall@5 | MRR | nDCG@5 | Neg. Warning Hit | False Alarms | Co-Change Disc. | Index RAM (KB) | P50 (ms) | P95 (ms) |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Dense RAG (float32 cosine) | 0.667 | 0.771 | 0.720 | 0.691 | 100% | 24% | 17% | 360.0 | 0.18 | 0.27 |
| BM25 only (SQLite FTS5) | 0.750 | 0.812 | 0.795 | 0.751 | 57% | 12% | 17% | 74.0 | 0.30 | 0.43 |
| Hybrid (BM25 + float32, RRF) | 0.708 | 0.812 | 0.755 | 0.730 | 86% | 12% | 17% | 434.0 | 0.37 | 0.50 |
| **git-engram** (full) | 0.715 | 0.896 | 0.899 | 0.842 | 86% | 24% | 83% | 149.4 | 1.64 | 2.27 |

| nDCG@5 | causal | regression | coupled | error-code |
|---|---:|---:|---:|---:|
| Dense RAG (float32 cosine) | 0.674 | 0.775 | 0.364 | 0.951 |
| BM25 only (SQLite FTS5) | 0.784 | 0.848 | 0.436 | 0.934 |
| Hybrid (BM25 + float32, RRF) | 0.785 | 0.814 | 0.386 | 0.934 |
| **git-engram** (full) | 0.855 | 0.869 | 0.716 | 0.928 |

### Retired held-out split v2 (24 queries)

Unseen when first reported; its false alarms were then diagnosed, so it is no longer held-out.

| Configuration | Recall@3 | Recall@5 | MRR | nDCG@5 | Neg. Warning Hit | False Alarms | Co-Change Disc. | Index RAM (KB) | P50 (ms) | P95 (ms) |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Dense RAG (float32 cosine) | 0.688 | 0.833 | 0.914 | 0.764 | 100% | 6% | 50% | 360.0 | 0.18 | 0.20 |
| BM25 only (SQLite FTS5) | 0.771 | 0.833 | 0.873 | 0.784 | 67% | 6% | 33% | 74.0 | 0.33 | 0.39 |
| Hybrid (BM25 + float32, RRF) | 0.750 | 0.833 | 0.894 | 0.788 | 100% | 6% | 50% | 434.0 | 0.39 | 0.60 |
| **git-engram** (full) | 0.750 | 0.938 | 0.925 | 0.872 | 100% | 6% | 83% | 149.4 | 1.52 | 1.94 |

| nDCG@5 | causal | regression | coupled | error-code |
|---|---:|---:|---:|---:|
| Dense RAG (float32 cosine) | 0.817 | 0.725 | 0.639 | 0.876 |
| BM25 only (SQLite FTS5) | 0.857 | 0.804 | 0.573 | 0.903 |
| Hybrid (BM25 + float32, RRF) | 0.809 | 0.801 | 0.637 | 0.903 |
| **git-engram** (full) | 0.846 | 0.927 | 0.813 | 0.902 |

### Retired held-out split v1 (24 queries)

Unseen when first reported; then used to diagnose commit-capsule crowding. Continuity only.

| Configuration | Recall@3 | Recall@5 | MRR | nDCG@5 | Neg. Warning Hit | False Alarms | Co-Change Disc. | Index RAM (KB) | P50 (ms) | P95 (ms) |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Dense RAG (float32 cosine) | 0.590 | 0.750 | 0.744 | 0.658 | 80% | 5% | 33% | 360.0 | 0.18 | 0.24 |
| BM25 only (SQLite FTS5) | 0.646 | 0.764 | 0.856 | 0.757 | 80% | 0% | 33% | 74.0 | 0.36 | 0.47 |
| Hybrid (BM25 + float32, RRF) | 0.632 | 0.771 | 0.842 | 0.751 | 80% | 0% | 33% | 434.0 | 0.44 | 0.63 |
| **git-engram** (full) | 0.743 | 0.799 | 0.931 | 0.830 | 80% | 5% | 83% | 149.4 | 1.75 | 2.34 |

| nDCG@5 | causal | regression | coupled | error-code |
|---|---:|---:|---:|---:|
| Dense RAG (float32 cosine) | 0.608 | 0.664 | 0.526 | 0.833 |
| BM25 only (SQLite FTS5) | 0.763 | 0.811 | 0.561 | 0.894 |
| Hybrid (BM25 + float32, RRF) | 0.739 | 0.784 | 0.561 | 0.917 |
| **git-engram** (full) | 0.819 | 0.804 | 0.861 | 0.836 |

### Dev / calibration split (48 queries, 8 per category)

Every weight and threshold was chosen while looking at these queries: scores are optimistic.

| Configuration | Recall@3 | Recall@5 | MRR | nDCG@5 | Neg. Warning Hit | False Alarms | Co-Change Disc. | Index RAM (KB) | P50 (ms) | P95 (ms) |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Dense RAG (float32 cosine) | 0.618 | 0.708 | 0.736 | 0.634 | 70% | 4% | 0% | 360.0 | 0.20 | 0.37 |
| BM25 only (SQLite FTS5) | 0.681 | 0.760 | 0.848 | 0.735 | 60% | 4% | 0% | 74.0 | 0.34 | 0.48 |
| Hybrid (BM25 + float32, RRF) | 0.670 | 0.729 | 0.787 | 0.687 | 75% | 4% | 0% | 434.0 | 0.41 | 0.57 |
| **git-engram** (full) | 0.764 | 0.826 | 0.882 | 0.818 | 90% | 7% | 62% | 149.4 | 1.62 | 2.21 |

| nDCG@5 | causal | regression | coupled | error-code | behavioral |
|---|---:|---:|---:|---:|---:|
| Dense RAG (float32 cosine) | 0.663 | 0.561 | 0.388 | 0.770 | 0.862 |
| BM25 only (SQLite FTS5) | 0.812 | 0.678 | 0.440 | 0.803 | 1.000 |
| Hybrid (BM25 + float32, RRF) | 0.762 | 0.608 | 0.408 | 0.779 | 0.954 |
| **git-engram** (full) | 0.912 | 0.694 | 0.766 | 0.844 | 1.000 |

### Ablation study (git-engram minus one component)

| Variant | nDCG@5 dev | nDCG@5 v5 | Recall@5 v5 | MRR v5 | Co-Change v5 | Neg. Hit v5 | False Alarms v5 |
|---|---:|---:|---:|---:|---:|---:|---:|
| git-engram (full) | 0.818 | 0.902 | 0.872 | 0.967 | 83% | 100% | 8% |
| - code-first behavioral route | 0.820 | 0.855 | 0.872 | 0.900 | 83% | 100% | 4% |
| - named-symbol pinning | 0.812 | 0.902 | 0.872 | 0.967 | 83% | 100% | 8% |
| - direct top-commit warning trigger | 0.818 | 0.902 | 0.872 | 0.967 | 83% | 67% | 8% |
| - broad symptom routing (old rules) | 0.812 | 0.893 | 0.872 | 0.950 | 83% | 83% | 8% |
| symptom co-change expansion on (graph 0.3) | 0.816 | 0.902 | 0.872 | 0.967 | 83% | 100% | 8% |
| - commit-relevance gate (any top-3 commit) | 0.818 | 0.902 | 0.872 | 0.967 | 83% | 100% | 42% |
| file-level graveyard anchoring (legacy) | 0.818 | 0.902 | 0.872 | 0.967 | 83% | 100% | 50% |
| - commit slot budget (unconstrained top-k) | 0.802 | 0.900 | 0.872 | 0.967 | 83% | 100% | 8% |
| - query-adaptive router (fixed weights) | 0.765 | 0.826 | 0.889 | 0.867 | 83% | 83% | 8% |
| - exact-identifier rerank profile | 0.818 | 0.896 | 0.872 | 0.967 | 83% | 100% | 12% |
| + MMR file diversity (opt-in) | 0.818 | 0.886 | 0.844 | 0.967 | 83% | 100% | 8% |
| - co-change graph path | 0.794 | 0.894 | 0.861 | 0.961 | 67% | 100% | 8% |
| - interaction reranker | 0.606 | 0.855 | 0.894 | 0.906 | 83% | 83% | 4% |
| - time decay (plain RRF) | 0.821 | 0.889 | 0.861 | 0.967 | 67% | 100% | 4% |
| float32 exact instead of 1-bit+int8 | 0.818 | 0.902 | 0.872 | 0.967 | 83% | 100% | 8% |

### `engram_preflight_check` latency: P50 **1.11 ms**, P95 **1.83 ms** (max 2.27 ms) over 7 targets

### Two-stage quantized search at scale (synthetic Dirichlet topic mixtures of the corpus embeddings)

| Vectors | Recall@10 vs float32-384d | Recall@10 vs float32-256d | Two-stage P50 (ms) | Float32 P50 (ms) | Two-stage RAM (MB) | Float32 RAM (MB) |
|---:|---:|---:|---:|---:|---:|---:|
| 10,000 | 0.874 | 0.990 | 0.18 | 0.23 | 2.8 | 14.6 |
| 100,000 | 0.733 | 0.970 | 0.62 | 2.44 | 27.8 | 146.5 |

### Context firewall

12/12 injection payloads detected, 12/12 neutralised (stripped or redacted); 0/12 benign snippets modified, 1/12 flagged.
