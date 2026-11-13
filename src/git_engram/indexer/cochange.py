r"""Temporal Co-Change Graph.

For nodes A, B (files or symbols) with commit sets C_A, C_B:

    W(A, B) = sum_{c in C_A ∩ C_B} exp(-lambda * (t_now - t_c)) / sqrt(|C_A| * |C_B|)

i.e. an exponentially time-decayed Ochiai (cosine) coupling coefficient: 1.0 means "always changed
together, recently"; old co-changes fade with half-life ``ln 2 / lambda``. We also store the directed
confidence P(B | A) = |C_A ∩ C_B| / |C_A| ("B changes in 85% of the commits that touch A").

Bulk commits touching more than ``max_files_per_commit`` nodes (formatting sweeps, renames, vendoring)
are excluded because they create O(n^2) spurious edges.
"""

from __future__ import annotations

import math
import re
from collections import defaultdict
from dataclasses import dataclass
from itertools import combinations
from pathlib import PurePosixPath

from ..config import GraphConfig
from ..storage.db import EngramDB


@dataclass(slots=True)
class Edge:
    src: str
    dst: str
    weight: float
    support: int
    confidence: float
    last_ts: float
    static_link: bool


def compute_edges(transactions: list[tuple[float, list[str]]], ref_ts: float, cfg: GraphConfig) -> list[Edge]:
    """Pure function: (timestamp, nodes) transactions -> symmetric weighted edges (both directions)."""
    lam = cfg.decay_lambda
    counts: dict[str, int] = defaultdict(int)
    num: dict[tuple[str, str], float] = defaultdict(float)
    support: dict[tuple[str, str], int] = defaultdict(int)
    last: dict[tuple[str, str], float] = {}
    for ts, nodes in transactions:
        uniq = sorted(set(nodes))
        if not uniq or len(uniq) > cfg.max_files_per_commit:
            continue
        for n in uniq:
            counts[n] += 1
        if len(uniq) < 2:
            continue
        decay = math.exp(-lam * max(0.0, ref_ts - ts))
        for a, b in combinations(uniq, 2):
            num[(a, b)] += decay
            support[(a, b)] += 1
            last[(a, b)] = max(last.get((a, b), 0.0), ts)
    edges: list[Edge] = []
    for (a, b), s in support.items():
        if s < cfg.min_support:
            continue
        w = num[(a, b)] / math.sqrt(counts[a] * counts[b])
        edges.append(Edge(a, b, w, s, s / counts[a], last[(a, b)], False))
        edges.append(Edge(b, a, w, s, s / counts[b], last[(a, b)], False))
    by_src: dict[str, list[Edge]] = defaultdict(list)
    for e in edges:
        by_src[e.src].append(e)
    pruned: list[Edge] = []
    for src_edges in by_src.values():
        src_edges.sort(key=lambda e: (-e.weight, e.dst))
        pruned.extend(src_edges[: cfg.max_edges_per_node])
    return pruned


def _module_tokens(path: str) -> set[str]:
    p = PurePosixPath(path)
    stem = p.stem.lower()
    return {stem, re.sub(r"[_\-.]", "", stem)} - {"index", "__init__", "mod", "main", "lib", ""}


def statically_linked(a: str, b: str, imports: dict[str, set[str]]) -> bool:
    """True when either file's import list references the other module (direct, non-hidden coupling)."""
    for src, dst in ((a, b), (b, a)):
        targets = _module_tokens(dst)
        for imp in imports.get(src, ()):
            parts = {x.lower() for x in re.split(r"[/.:\\]+", imp) if x}
            if targets & parts:
                return True
    return False


def rebuild_cochange(db: EngramDB, cfg: GraphConfig, ref_ts: float, current_files: set[str]) -> dict[str, int]:
    file_tx: dict[str, tuple[float, list[str]]] = {}
    for row in db.query("SELECT c.hash, c.ts, f.path FROM commits c JOIN commit_files f ON f.hash = c.hash"):
        if row["path"] not in current_files:
            continue
        file_tx.setdefault(row["hash"], (row["ts"], []))[1].append(row["path"])
    sym_tx: dict[str, tuple[float, list[str]]] = {}
    for row in db.query("SELECT hash, ts, path, qualname FROM symbol_commits"):
        if row["path"] not in current_files or row["qualname"] == "<module>":
            continue
        sym_tx.setdefault(row["hash"], (row["ts"], []))[1].append(f"{row['path']}::{row['qualname']}")
    imports = {
        r["path"]: set(filter(None, r["imports"].split("\n"))) for r in db.query("SELECT path, imports FROM files")
    }

    file_edges = compute_edges(list(file_tx.values()), ref_ts, cfg)
    sym_edges = compute_edges(list(sym_tx.values()), ref_ts, cfg)
    for e in file_edges:
        e.static_link = statically_linked(e.src, e.dst, imports)
    for e in sym_edges:
        e.static_link = statically_linked(e.src.split("::")[0], e.dst.split("::")[0], imports) or (
            e.src.split("::")[0] == e.dst.split("::")[0]
        )
    with db.transaction() as conn:
        conn.execute("DELETE FROM cochange_edges")
        conn.executemany(
            "INSERT INTO cochange_edges(level, src, dst, weight, support, confidence, last_ts, static_link) "
            "VALUES(?,?,?,?,?,?,?,?)",
            [
                ("file", e.src, e.dst, e.weight, e.support, e.confidence, e.last_ts, int(e.static_link))
                for e in file_edges
            ]
            + [
                ("symbol", e.src, e.dst, e.weight, e.support, e.confidence, e.last_ts, int(e.static_link))
                for e in sym_edges
            ],
        )
    return {"file_edges": len(file_edges), "symbol_edges": len(sym_edges)}
