"""Engine configuration (all knobs of the retrieval math live here, nothing is hidden)."""

from __future__ import annotations

import math
from pathlib import Path

from pydantic import BaseModel, Field


class EmbeddingConfig(BaseModel):
    backend: str = Field(default="hashing-lsa", description="'hashing-lsa' (offline, deterministic) or 'onnx'")
    onnx_model_dir: str | None = Field(default=None, description="Directory holding model.onnx + tokenizer.json")
    full_dim: int = 384
    mrl_dim: int = 256
    hash_buckets: int = 4096
    char_ngram: int = 4
    char_ngram_weight: float = 0.35
    svd_sample: int = 6000
    svd_oversample: int = 16
    svd_power_iters: int = 3
    seed: int = 1729


class RetrievalConfig(BaseModel):
    rrf_k: float = 60.0
    w_bm25: float = 1.0
    w_vec: float = 1.0
    w_graph: float = 1.0
    adaptive_routing: bool = True
    static_link_discount: float = 0.6
    recency_alpha: float = 0.25
    recency_half_life_days: float = 120.0
    stage1_candidates: int = 400
    path_depth: int = 50
    graph_seeds: int = 6
    graph_exclude_top: int = 6
    graph_decay: float = 0.85
    graph_min_weight: float = 0.15
    rerank_top_n: int = 30
    mmr: bool = False  # opt-in: MMR file diversity lowered held-out nDCG@5 on two consecutive unseen splits
    diversity_penalty: float = 0.10
    # Slot budget: at most this many commit capsules inside the first ``slot_window`` results, per router class.
    slot_window: int = 5
    commit_slots: dict[str, int] = Field(
        default_factory=lambda: {"causal": 3, "behavioral": 2, "symptom": 1, "change": 0, "exact": 2, "general": 2}
    )
    # Named-symbol pinning: a symbol the query names exactly is guaranteed a slot at or above ``pin_position``.
    pin_position: int = 1
    pin_classes: list[str] = Field(default_factory=lambda: ["behavioral", "exact", "change"])
    attach_commits: int = 2
    rerank: bool = True
    cross_encoder_dir: str | None = None
    parallel: bool = True

    @property
    def recency_lambda(self) -> float:
        return math.log(2.0) / (self.recency_half_life_days * 86400.0)


class GraphConfig(BaseModel):
    half_life_days: float = 365.0
    max_files_per_commit: int = 40
    min_support: int = 2
    max_edges_per_node: int = 24

    @property
    def decay_lambda(self) -> float:
        return math.log(2.0) / (self.half_life_days * 86400.0)


class IndexConfig(BaseModel):
    max_commits: int = 5000
    max_file_bytes: int = 512_000
    blame: bool = True
    allow_grammar_download: bool = False
    tree_sitter: bool = True
    skip_globs: list[str] = Field(
        default_factory=lambda: [
            "*.lock",
            "package-lock.json",
            "pnpm-lock.yaml",
            "yarn.lock",
            "Cargo.lock",
            "poetry.lock",
            "uv.lock",
            "*.min.js",
            "*.map",
            "*.png",
            "*.jpg",
            "*.jpeg",
            "*.gif",
            "*.ico",
            "*.pdf",
            "*.zip",
            "*.gz",
            "*.woff",
            "*.woff2",
            "*.ttf",
            "*.so",
            "*.dylib",
            "*.dll",
            "*.exe",
            "*.bin",
            "*.onnx",
            "*.pyc",
            ".engram/*",
        ]
    )


class GraveyardConfig(BaseModel):
    max_stderr_bytes: int = 65_536
    max_diff_bytes: int = 131_072
    semantic_threshold: float = 0.40
    context_weight: float = 0.30
    context_hits: int = 3
    sibling_weight: float = 0.25
    commit_context: bool = True
    # Commit-relevance gate: a retrieved commit may credit a failure only if its rerank score is at least this
    # fraction of the best one AND the failure's function is also among the top ``gate_code_hits`` code hits.
    commit_gating: bool = True
    commit_gate: float = 0.9
    gate_code_hits: int = 5
    anchor: str = "symbol"  # "symbol" (default) or "file" (legacy: any failure in a top-3 file gets credit)
    max_commit_context_symbols: int = 8
    textual_weight: float = 0.60
    max_warnings: int = 3


class EngramConfig(BaseModel):
    embedding: EmbeddingConfig = Field(default_factory=EmbeddingConfig)
    retrieval: RetrievalConfig = Field(default_factory=RetrievalConfig)
    graph: GraphConfig = Field(default_factory=GraphConfig)
    index: IndexConfig = Field(default_factory=IndexConfig)
    graveyard: GraveyardConfig = Field(default_factory=GraveyardConfig)

    @classmethod
    def load(cls, repo_root: Path) -> EngramConfig:
        path = repo_root / ".engram" / "config.json"
        if path.exists():
            return cls.model_validate_json(path.read_text(encoding="utf-8"))
        return cls()
