## Benchmark: git-engram vs. standard RAG

Fixture: `shopfront` synthetic monorepo -- 52 commits, 39 files, 198 indexed documents, 11 graveyard failures. Hardware: Darwin arm64, Python 3.14.10. Latency = warm, single-thread wall time per query over 20 repeats.

### Held-out test split v3 (24 unseen queries: 6 causal, 6 regression, 6 coupled-file, 6 exact-identifier)

Committed (with an inventory fixture extension) before symbol-level graveyard anchoring, the MMR removal and exact-route tuning existed; never used to choose any weight or threshold. Includes sibling-symbol queries that measure graveyard false alarms.

| Configuration | Recall@3 | Recall@5 | MRR | nDCG@5 | Neg. Warning Hit | False Alarms | Co-Change Disc. | Index RAM (KB) | P50 (ms) | P95 (ms) |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Dense RAG (float32 cosine) | 0.688 | 0.771 | 0.708 | 0.682 | 100% | 24% | 17% | 297.0 | 0.21 | 0.33 |
| BM25 only (SQLite FTS5) | 0.771 | 0.812 | 0.826 | 0.766 | 57% | 12% | 17% | 64.2 | 0.37 | 0.54 |
| Hybrid (BM25 + float32, RRF) | 0.688 | 0.812 | 0.771 | 0.734 | 100% | 12% | 17% | 361.2 | 0.43 | 0.59 |
| **git-engram** (full) | 0.653 | 0.875 | 0.835 | 0.789 | 86% | 29% | 83% | 126.6 | 1.79 | 2.52 |

| nDCG@5 | causal | regression | coupled | error-code |
|---|---:|---:|---:|---:|
| Dense RAG (float32 cosine) | 0.674 | 0.718 | 0.384 | 0.951 |
| BM25 only (SQLite FTS5) | 0.784 | 0.858 | 0.487 | 0.934 |
| Hybrid (BM25 + float32, RRF) | 0.785 | 0.775 | 0.442 | 0.934 |
| **git-engram** (full) | 0.650 | 0.861 | 0.716 | 0.928 |

Router classes on the v3 split: causal 6, change 6, exact 6, general 3, symptom 3

### Retired held-out split v2 (24 queries)

Unseen when first reported; its false alarms were then diagnosed, so it is no longer held-out.

| Configuration | Recall@3 | Recall@5 | MRR | nDCG@5 | Neg. Warning Hit | False Alarms | Co-Change Disc. | Index RAM (KB) | P50 (ms) | P95 (ms) |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Dense RAG (float32 cosine) | 0.708 | 0.833 | 0.887 | 0.758 | 100% | 6% | 50% | 297.0 | 0.23 | 0.28 |
| BM25 only (SQLite FTS5) | 0.771 | 0.833 | 0.874 | 0.784 | 83% | 6% | 33% | 64.2 | 0.39 | 0.54 |
| Hybrid (BM25 + float32, RRF) | 0.771 | 0.812 | 0.888 | 0.772 | 100% | 11% | 33% | 361.2 | 0.48 | 0.65 |
| **git-engram** (full) | 0.750 | 0.917 | 0.894 | 0.855 | 100% | 17% | 83% | 126.6 | 1.95 | 2.40 |

| nDCG@5 | causal | regression | coupled | error-code |
|---|---:|---:|---:|---:|
| Dense RAG (float32 cosine) | 0.819 | 0.690 | 0.639 | 0.885 |
| BM25 only (SQLite FTS5) | 0.857 | 0.804 | 0.573 | 0.903 |
| Hybrid (BM25 + float32, RRF) | 0.811 | 0.792 | 0.598 | 0.885 |
| **git-engram** (full) | 0.914 | 0.867 | 0.773 | 0.868 |

### Retired held-out split v1 (24 queries)

Unseen when first reported; then used to diagnose commit-capsule crowding. Continuity only.

| Configuration | Recall@3 | Recall@5 | MRR | nDCG@5 | Neg. Warning Hit | False Alarms | Co-Change Disc. | Index RAM (KB) | P50 (ms) | P95 (ms) |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Dense RAG (float32 cosine) | 0.590 | 0.771 | 0.724 | 0.667 | 100% | 5% | 33% | 297.0 | 0.22 | 0.31 |
| BM25 only (SQLite FTS5) | 0.625 | 0.764 | 0.830 | 0.739 | 80% | 0% | 33% | 64.2 | 0.45 | 0.58 |
| Hybrid (BM25 + float32, RRF) | 0.611 | 0.750 | 0.823 | 0.738 | 80% | 0% | 33% | 361.2 | 0.55 | 0.68 |
| **git-engram** (full) | 0.708 | 0.819 | 0.903 | 0.829 | 100% | 21% | 83% | 126.6 | 2.16 | 3.16 |

| nDCG@5 | causal | regression | coupled | error-code |
|---|---:|---:|---:|---:|
| Dense RAG (float32 cosine) | 0.661 | 0.664 | 0.488 | 0.855 |
| BM25 only (SQLite FTS5) | 0.763 | 0.827 | 0.561 | 0.804 |
| Hybrid (BM25 + float32, RRF) | 0.757 | 0.773 | 0.510 | 0.910 |
| **git-engram** (full) | 0.881 | 0.804 | 0.804 | 0.825 |

### Dev / calibration split (32 queries, 8 per category)

Every weight and threshold was chosen while looking at these queries: scores are optimistic.

| Configuration | Recall@3 | Recall@5 | MRR | nDCG@5 | Neg. Warning Hit | False Alarms | Co-Change Disc. | Index RAM (KB) | P50 (ms) | P95 (ms) |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Dense RAG (float32 cosine) | 0.583 | 0.714 | 0.780 | 0.644 | 87% | 0% | 0% | 297.0 | 0.19 | 0.29 |
| BM25 only (SQLite FTS5) | 0.661 | 0.734 | 0.888 | 0.740 | 67% | 0% | 0% | 64.2 | 0.37 | 0.51 |
| Hybrid (BM25 + float32, RRF) | 0.641 | 0.719 | 0.843 | 0.713 | 73% | 0% | 0% | 361.2 | 0.43 | 0.60 |
| **git-engram** (full) | 0.792 | 0.833 | 0.932 | 0.849 | 87% | 6% | 62% | 126.6 | 1.76 | 2.23 |

| nDCG@5 | causal | regression | coupled | error-code |
|---|---:|---:|---:|---:|
| Dense RAG (float32 cosine) | 0.649 | 0.788 | 0.370 | 0.770 |
| BM25 only (SQLite FTS5) | 0.812 | 0.907 | 0.440 | 0.803 |
| Hybrid (BM25 + float32, RRF) | 0.757 | 0.909 | 0.408 | 0.779 |
| **git-engram** (full) | 0.945 | 0.873 | 0.734 | 0.844 |

### Ablation study (git-engram minus one component)

| Variant | nDCG@5 dev | nDCG@5 v3 | Recall@5 v3 | MRR v3 | Co-Change v3 | Neg. Hit v3 | False Alarms v3 |
|---|---:|---:|---:|---:|---:|---:|---:|
| git-engram (full) | 0.849 | 0.789 | 0.875 | 0.835 | 83% | 86% | 29% |
| - commit slot budget (unconstrained top-k) | 0.830 | 0.755 | 0.792 | 0.833 | 50% | 86% | 35% |
| - query-adaptive router (fixed weights) | 0.795 | 0.772 | 0.854 | 0.835 | 50% | 86% | 41% |
| - exact-identifier rerank profile | 0.838 | 0.789 | 0.875 | 0.835 | 83% | 86% | 29% |
| file-level graveyard anchoring (legacy) | 0.849 | 0.789 | 0.875 | 0.835 | 83% | 86% | 41% |
| + MMR file diversity (opt-in) | 0.856 | 0.752 | 0.771 | 0.832 | 83% | 86% | 29% |
| - co-change graph path | 0.823 | 0.767 | 0.833 | 0.826 | 50% | 86% | 35% |
| - interaction reranker | 0.533 | 0.731 | 0.875 | 0.753 | 83% | 86% | 24% |
| - time decay (plain RRF) | 0.859 | 0.776 | 0.833 | 0.835 | 67% | 86% | 35% |
| float32 exact instead of 1-bit+int8 | 0.849 | 0.789 | 0.875 | 0.835 | 83% | 86% | 29% |

### `engram_preflight_check` latency: P50 **1.48 ms**, P95 **2.65 ms** (max 3.16 ms) over 7 targets

### Two-stage quantized search at scale (synthetic Dirichlet topic mixtures of the corpus embeddings)

| Vectors | Recall@10 vs float32-384d | Recall@10 vs float32-256d | Two-stage P50 (ms) | Float32 P50 (ms) | Two-stage RAM (MB) | Float32 RAM (MB) |
|---:|---:|---:|---:|---:|---:|---:|
| 10,000 | 0.856 | 0.984 | 0.21 | 0.27 | 2.8 | 14.6 |
| 100,000 | 0.753 | 0.964 | 0.85 | 3.18 | 27.8 | 146.5 |

### Context firewall

12/12 injection payloads detected, 12/12 neutralised (stripped or redacted); 0/12 benign snippets modified, 1/12 flagged.
