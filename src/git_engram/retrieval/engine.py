"""Tri-path hybrid retrieval: FTS5 BM25 || two-stage quantized vectors -> co-change spreading activation
-> Time-Decayed RRF -> interaction / cross-encoder reranking."""

from __future__ import annotations

import time
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field

import numpy as np

from ..config import RetrievalConfig
from ..embeddings import Embedder, QuantizedVectorIndex
from ..models import Hit
from ..storage.db import EngramDB
from .fusion import td_rrf
from .graph import CoChangeGraph
from .lexical import bm25_search
from .reranker import Candidate, Reranker
from .router import QueryProfile, classify


@dataclass(frozen=True)
class RetrievalMode:
    name: str
    bm25: bool = True
    vector: str | None = "two_stage"  # "two_stage" | "float32" | None
    graph: bool = True
    time_decay: bool = True
    rerank: bool = True
    negative: bool = True
    diversify: bool = False
    route: bool = True
    slot_budget: bool = True
    exclude_kinds: frozenset[str] = frozenset({"failure"})


FULL = RetrievalMode("git-engram")
DENSE_F32 = RetrievalMode(
    "dense-float32",
    bm25=False,
    vector="float32",
    graph=False,
    time_decay=False,
    rerank=False,
    negative=False,
    route=False,
    slot_budget=False,
    exclude_kinds=frozenset(),
)
BM25_ONLY = RetrievalMode(
    "bm25",
    bm25=True,
    vector=None,
    graph=False,
    time_decay=False,
    rerank=False,
    negative=False,
    route=False,
    slot_budget=False,
    exclude_kinds=frozenset(),
)
HYBRID = RetrievalMode(
    "hybrid",
    bm25=True,
    vector="float32",
    graph=False,
    time_decay=False,
    rerank=False,
    negative=False,
    route=False,
    slot_budget=False,
    exclude_kinds=frozenset(),
)
MODES = {m.name: m for m in (FULL, DENSE_F32, BM25_ONLY, HYBRID)}


@dataclass(slots=True)
class DocMeta:
    kind: str
    path: str | None
    symbol: str | None
    title: str
    ts: float | None
    doc_key: str


@dataclass
class RetrievalResult:
    hits: list[Hit]
    query_vec: np.ndarray | None
    timings_ms: dict[str, float] = field(default_factory=dict)
    seed_files: list[str] = field(default_factory=list)
    profile: QueryProfile | None = None


class HybridRetriever:
    def __init__(
        self,
        db: EngramDB,
        embedder: Embedder,
        vindex: QuantizedVectorIndex,
        graph: CoChangeGraph,
        cfg: RetrievalConfig,
        ref_ts: float,
        reranker: Reranker | None = None,
    ):
        self.db = db
        self.embedder = embedder
        self.vindex = vindex
        self.graph = graph
        self.cfg = cfg
        self.ref_ts = ref_ts
        self.reranker = reranker
        self.pool = ThreadPoolExecutor(max_workers=1, thread_name_prefix="engram-vec") if cfg.parallel else None
        self.meta: dict[int, DocMeta] = {}
        self.by_key: dict[str, int] = {}
        self.by_path: dict[str, list[int]] = defaultdict(list)
        for r in db.query("SELECT id, doc_key, kind, path, symbol, title, ts FROM documents"):
            m = DocMeta(r["kind"], r["path"], r["symbol"], r["title"], r["ts"], r["doc_key"])
            self.meta[int(r["id"])] = m
            self.by_key[m.doc_key] = int(r["id"])
            if m.path and m.kind in {"symbol", "file"}:
                self.by_path[m.path].append(int(r["id"]))

    def register(
        self, doc_id: int, doc_key: str, kind: str, path: str | None, symbol: str | None, title: str, ts: float | None
    ) -> None:
        self.meta[doc_id] = DocMeta(kind, path, symbol, title, ts, doc_key)
        self.by_key[doc_key] = doc_id

    def close(self) -> None:
        if self.pool is not None:
            self.pool.shutdown(wait=False)

    # --------------------------------------------------------------- paths
    def _vector(self, q: np.ndarray, mode: RetrievalMode, depth: int):
        exclude = set(mode.exclude_kinds) or None
        if mode.vector == "float32":
            return self.vindex.exact_float32_search(q, depth, exclude=exclude)
        return self.vindex.search(q, depth, self.cfg.stage1_candidates, exclude=exclude)

    def _graph_list(
        self, seeds_ranked: list[int], q: np.ndarray | None, file_hint: str | None, limit: int, exclude_paths: set[str]
    ) -> tuple[list[int], dict[int, tuple[float, str]]]:
        file_seeds: dict[str, float] = {}
        sym_seeds: dict[str, float] = {}
        n = len(seeds_ranked)
        for rank, doc in enumerate(seeds_ranked):
            m = self.meta.get(doc)
            if m is None or not m.path:
                continue
            a = 1.0 - rank / max(n, 1) * 0.5
            file_seeds[m.path] = max(file_seeds.get(m.path, 0.0), a)
            if m.kind == "symbol" and m.symbol:
                sym_seeds[f"{m.path}::{m.symbol}"] = a
        if file_hint:
            file_seeds[file_hint] = 1.0
        cfg = self.cfg
        file_act = self.graph.spread(
            file_seeds, "file", cfg.graph_decay, cfg.graph_min_weight, cfg.static_link_discount
        )
        sym_act = self.graph.spread(
            sym_seeds, "symbol", cfg.graph_decay, cfg.graph_min_weight, cfg.static_link_discount
        )
        seed_paths = exclude_paths | ({file_hint} if file_hint else set())
        scored: dict[int, tuple[float, str]] = {}
        for node, (act, src, _) in sym_act.items():
            if node.split("::", 1)[0] in seed_paths:
                continue
            doc = self.by_key.get(f"sym:{node}")
            if doc is not None:
                scored[doc] = (act, f"symbol co-change with {src}")
        top_files = sorted(((p, v) for p, v in file_act.items() if p not in seed_paths), key=lambda kv: -kv[1][0])[
            :limit
        ]
        for path, (act, src, _) in top_files:
            docs = self.by_path.get(path, [])
            if not docs:
                continue
            if q is not None and len(docs) > 1:
                cos = self.vindex.cosine_for(q, docs)
                rep = max(docs, key=lambda d: cos.get(d, -1.0))
            else:
                rep = docs[0]
            if act > scored.get(rep, (0.0, ""))[0]:
                scored[rep] = (act, f"file co-change with {src}")
        ranked = sorted(scored, key=lambda d: -scored[d][0])[:limit]
        return ranked, scored

    # ------------------------------------------------------------- retrieve
    def retrieve(
        self, query: str, top_k: int = 10, mode: RetrievalMode = FULL, file_hint: str | None = None
    ) -> RetrievalResult:
        cfg = self.cfg
        timings: dict[str, float] = {}
        t0 = time.perf_counter()
        depth = cfg.path_depth
        q = self.embedder.embed_query(query) if (mode.vector or mode.graph or mode.rerank) else None
        timings["embed"] = (time.perf_counter() - t0) * 1000

        vec_hits = []
        future = None
        t1 = time.perf_counter()
        if mode.vector and q is not None:
            if self.pool is not None:
                future = self.pool.submit(self._vector, q, mode, depth)
            else:
                vec_hits = self._vector(q, mode, depth)
        bm25_hits = bm25_search(self.db, query, depth, exclude=set(mode.exclude_kinds) or None) if mode.bm25 else []
        timings["bm25"] = (time.perf_counter() - t1) * 1000
        if future is not None:
            vec_hits = future.result()
        timings["vector"] = (time.perf_counter() - t1) * 1000

        rankings: dict[str, list[int]] = {}
        if mode.bm25:
            rankings["bm25"] = [d for d, _ in bm25_hits]
        if mode.vector:
            rankings["vec"] = [h.doc_id for h in vec_hits]
        weights = {"bm25": cfg.w_bm25, "vec": cfg.w_vec, "graph": cfg.w_graph}
        profile = classify(query) if (mode.route and cfg.adaptive_routing) else None
        if profile is not None:
            weights = {m: weights[m] * profile.weights[m] for m in weights}

        graph_info: dict[int, tuple[float, str]] = {}
        seed_files: list[str] = []
        t2 = time.perf_counter()
        if mode.graph:
            prelim = td_rrf(rankings, weights, cfg.rrf_k)
            seeds = [d for d, _, _ in prelim[: cfg.graph_seeds]]
            seed_files = [self.meta[d].path for d in seeds if d in self.meta and self.meta[d].path]
            exclude_paths = {
                self.meta[d].path for d, _, _ in prelim[: cfg.graph_exclude_top] if d in self.meta and self.meta[d].path
            }
            ranked, graph_info = self._graph_list(seeds, q, file_hint, depth, exclude_paths)
            excluded = set(mode.exclude_kinds)
            rankings["graph"] = [d for d in ranked if d in self.meta and self.meta[d].kind not in excluded]
        timings["graph"] = (time.perf_counter() - t2) * 1000

        t3 = time.perf_counter()
        ts = {d: self.meta[d].ts for ids in rankings.values() for d in ids if d in self.meta}
        fused = td_rrf(
            rankings,
            weights,
            cfg.rrf_k,
            ts,
            self.ref_ts,
            cfg.recency_alpha if mode.time_decay else 0.0,
            cfg.recency_lambda,
        )
        fused = [f for f in fused if f[0] in self.meta]
        timings["fusion"] = (time.perf_counter() - t3) * 1000

        bm25_rank = {d: (i + 1, s) for i, (d, s) in enumerate(bm25_hits)}
        vec_rank = {h.doc_id: (i + 1, h) for i, h in enumerate(vec_hits)}
        graph_rank = {d: i + 1 for i, d in enumerate(rankings.get("graph", []))}

        t4 = time.perf_counter()
        rerank_scores: dict[int, float] = {}
        order = [d for d, _, _ in fused]
        if mode.rerank and self.reranker is not None and fused:
            top = fused[: cfg.rerank_top_n]
            ids = [d for d, _, _ in top]
            bodies = self._bodies(ids)
            dense = self.vindex.cosine_for(q, ids) if q is not None else {}
            cands = [
                Candidate(
                    d,
                    self.meta[d].kind,
                    self.meta[d].title,
                    self.meta[d].symbol or "",
                    self.meta[d].path or "",
                    bodies.get(d, ""),
                    score,
                    dense.get(d, 0.0),
                    min(1.0, graph_info.get(d, (0.0, ""))[0]),
                )
                for d, score, _ in top
            ]
            scores = self.reranker.rerank(query, cands, profile.intent if profile else "auto")
            rerank_scores = {c.doc_id: s for c, s in zip(cands, scores, strict=True)}
            head = sorted(ids, key=lambda d: (-rerank_scores[d], d))
            if mode.diversify or cfg.mmr:
                head = self._diversify(head, rerank_scores)
            order = head + [d for d, _, _ in fused[cfg.rerank_top_n :]]
        if mode.slot_budget:
            order = self._slot_budget(order, profile.klass if profile else "general")
        timings["rerank"] = (time.perf_counter() - t4) * 1000

        fused_map = {d: (s, rf) for d, s, rf in fused}
        selected = order[:top_k]
        bodies = self._bodies(selected)
        hits: list[Hit] = []
        for d in selected:
            m = self.meta[d]
            fs, rf = fused_map[d]
            br = bm25_rank.get(d)
            vr = vec_rank.get(d)
            gi = graph_info.get(d)
            hits.append(
                Hit(
                    doc_id=d,
                    doc_key=m.doc_key,
                    kind=m.kind,
                    path=m.path,
                    symbol=m.symbol,
                    title=m.title,
                    snippet=make_snippet(m.kind, bodies.get(d, "")),
                    ts=m.ts,
                    score=rerank_scores.get(d, fs),
                    fused=fs,
                    recency=rf,
                    bm25_rank=br[0] if br else None,
                    bm25_score=br[1] if br else None,
                    vec_rank=vr[0] if vr else None,
                    hamming=(vr[1].hamming if vr and vr[1].hamming >= 0 else None),
                    cosine=vr[1].cosine if vr else None,
                    graph_rank=graph_rank.get(d),
                    graph_activation=gi[0] if gi else None,
                    graph_via=gi[1] if gi else None,
                    rerank_score=rerank_scores.get(d),
                )
            )
        if mode.slot_budget and cfg.attach_commits > 0:
            self._attach_commits(hits, order[: cfg.rerank_top_n])
        timings["total"] = (time.perf_counter() - t0) * 1000
        return RetrievalResult(hits, q, timings, seed_files, profile)

    def _slot_budget(self, order: list[int], klass: str) -> list[int]:
        """Stratified top-K: commit capsules may hold at most ``commit_slots[klass]`` of the first ``slot_window``
        positions. Excess commits are deferred (not dropped) to just after the window, keeping their order."""
        window = self.cfg.slot_window
        cap = self.cfg.commit_slots.get(klass, window)
        head: list[int] = []
        deferred: list[int] = []
        used = 0
        i = 0
        while i < len(order) and len(head) < window:
            d = order[i]
            i += 1
            if self.meta[d].kind == "commit":
                if used >= cap:
                    deferred.append(d)
                    continue
                used += 1
            head.append(d)
        return head + deferred + order[i:]

    def _attach_commits(self, hits: list[Hit], candidates: list[int]) -> None:
        """Parent-symbol attachment: each code hit carries the commits that modified it, preferring commits that were
        themselves retrieval candidates, so causal context survives even when a capsule gave up its slot."""
        targets = [(h.path, h.symbol) for h in hits if h.kind == "symbol" and h.path and h.symbol]
        if not targets:
            return
        cand_rank = {self.meta[d].doc_key[7:]: r for r, d in enumerate(candidates) if self.meta[d].kind == "commit"}
        placeholders = ",".join("(?, ?)" for _ in targets)
        rows = self.db.query(
            "SELECT sc.path, sc.qualname, c.hash, c.subject, c.intent, c.ts FROM symbol_commits sc "
            f"JOIN commits c ON c.hash = sc.hash WHERE (sc.path, sc.qualname) IN (VALUES {placeholders})",
            [x for pair in targets for x in pair],
        )
        by_symbol: dict[tuple[str, str], list] = defaultdict(list)
        for r in rows:
            by_symbol[(r["path"], r["qualname"])].append(r)
        shown = {h.doc_key for h in hits}
        load_bearing = {"bugfix", "security", "revert"}
        for h in hits:
            commits = by_symbol.get((h.path, h.symbol), [])
            ranked = sorted(
                commits,
                key=lambda r: (
                    r["hash"] not in cand_rank,
                    cand_rank.get(r["hash"], 0),
                    r["intent"] not in load_bearing,
                    -r["ts"],
                ),
            )
            picked = [r for r in ranked if r["hash"] in cand_rank or r["intent"] in load_bearing]
            h.attached_commits = [
                {
                    "hash": r["hash"],
                    "subject": r["subject"],
                    "intent": r["intent"],
                    "ts": r["ts"],
                    "also_listed": f"commit:{r['hash']}" in shown,
                }
                for r in picked[: self.cfg.attach_commits]
            ]

    def _diversify(self, ranked: list[int], scores: dict[int, float]) -> list[int]:
        """Greedy MMR over file identity: each already-selected code doc from the same file costs
        ``diversity_penalty``. Commits/failures/invariants are distinct evidence types and are never penalised."""
        penalty = self.cfg.diversity_penalty
        if penalty <= 0:
            return ranked
        remaining = list(ranked)
        per_path: dict[str, int] = defaultdict(int)
        out: list[int] = []
        while remaining:

            def adjusted(d: int) -> float:
                m = self.meta[d]
                same = per_path[m.path] if m.path and m.kind in {"symbol", "file"} else 0
                return scores[d] - penalty * same

            best = max(remaining, key=lambda d: (adjusted(d), -d))
            remaining.remove(best)
            m = self.meta[best]
            if m.path and m.kind in {"symbol", "file"}:
                per_path[m.path] += 1
            out.append(best)
        return out

    def _bodies(self, ids: list[int]) -> dict[int, str]:
        if not ids:
            return {}
        marks = ",".join("?" * len(ids))
        return {
            int(r["id"]): r["body"] for r in self.db.query(f"SELECT id, body FROM documents WHERE id IN ({marks})", ids)
        }


def make_snippet(kind: str, body: str, max_lines: int = 14, max_chars: int = 1200) -> str:
    lines = [ln.rstrip() for ln in body.splitlines() if ln.strip()]
    text = "\n".join(lines[:max_lines])
    return text[:max_chars]
