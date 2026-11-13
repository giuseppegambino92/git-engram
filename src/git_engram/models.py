"""Typed records exchanged between the indexer, retrieval engine, MCP server and CLI."""

from __future__ import annotations

from pydantic import BaseModel, Field


class Symbol(BaseModel):
    path: str
    qualname: str
    name: str
    kind: str
    language: str
    start_line: int
    end_line: int
    signature: str = ""
    docstring: str = ""
    content: str = ""
    parent: str | None = None

    @property
    def key(self) -> str:
        return f"{self.path}::{self.qualname}"


class FileChange(BaseModel):
    path: str
    old_path: str | None = None
    change: str = "M"
    added: int = 0
    removed: int = 0
    new_ranges: list[tuple[int, int]] = Field(default_factory=list)
    old_ranges: list[tuple[int, int]] = Field(default_factory=list)
    added_lines: list[tuple[int, str]] = Field(default_factory=list)
    removed_lines: list[tuple[int, str]] = Field(default_factory=list)
    old_touch: list[int] = Field(default_factory=list)


class Commit(BaseModel):
    hash: str
    parents: list[str] = Field(default_factory=list)
    author: str = ""
    ts: float = 0.0
    subject: str = ""
    body: str = ""
    files: list[FileChange] = Field(default_factory=list)


class Hit(BaseModel):
    doc_id: int
    doc_key: str
    kind: str
    path: str | None
    symbol: str | None
    title: str
    snippet: str
    ts: float | None
    score: float
    fused: float = 0.0
    recency: float = 1.0
    bm25_rank: int | None = None
    bm25_score: float | None = None
    vec_rank: int | None = None
    hamming: int | None = None
    cosine: float | None = None
    graph_rank: int | None = None
    graph_activation: float | None = None
    graph_via: str | None = None
    rerank_score: float | None = None
    attached_commits: list[dict] = Field(default_factory=list)


class CouplingAlert(BaseModel):
    source: str
    target: str
    weight: float
    confidence: float
    support: int
    hidden: bool
    last_ts: float


class FailureWarning(BaseModel):
    failure_id: int
    ts: float
    files: list[str]
    symbols: list[str]
    command: str
    exit_code: int
    signature: str
    notes: str
    diff_excerpt: str
    occurrences: int
    match_reason: str
    score: float


class InvariantHit(BaseModel):
    rule_id: str
    scope: str
    description: str
    severity: str
    matched: str


class SecurityFinding(BaseModel):
    rule: str
    detail: str
    severity: str


class SearchResponse(BaseModel):
    query: str
    hits: list[Hit]
    coupling: list[CouplingAlert] = Field(default_factory=list)
    warnings: list[FailureWarning] = Field(default_factory=list)
    invariants: list[InvariantHit] = Field(default_factory=list)
    security: list[SecurityFinding] = Field(default_factory=list)
    timings_ms: dict[str, float] = Field(default_factory=dict)
    query_class: str | None = None
    fusion_weights: dict[str, float] = Field(default_factory=dict)
