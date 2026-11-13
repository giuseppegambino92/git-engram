"""Standard IR metrics over ranked documents with graded, de-duplicated relevance targets."""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass


@dataclass(frozen=True)
class RankedDoc:
    doc_key: str
    kind: str
    path: str | None


def target_matches(target: str, doc: RankedDoc, commit_keys: dict[str, str]) -> bool:
    if target.startswith("commit:"):
        return doc.doc_key == commit_keys.get(target)
    if target.startswith("sym:"):
        return doc.doc_key == target
    if target.startswith("file:"):
        return doc.kind in {"symbol", "file"} and doc.path == target[5:]
    raise ValueError(target)


def first_ranks(targets: Sequence[str], ranked: Sequence[RankedDoc], commit_keys: dict[str, str]) -> dict[str, int]:
    """1-based rank at which each target is first satisfied (absent if never)."""
    found: dict[str, int] = {}
    for i, doc in enumerate(ranked, start=1):
        for t in targets:
            if t not in found and target_matches(t, doc, commit_keys):
                found[t] = i
    return found


def recall_at(found: dict[str, int], n_targets: int, k: int) -> float:
    return sum(1 for r in found.values() if r <= k) / n_targets if n_targets else 0.0


def reciprocal_rank(found: dict[str, int]) -> float:
    return 1.0 / min(found.values()) if found else 0.0


def ndcg_at(relevant: dict[str, int], ranked: Sequence[RankedDoc], commit_keys: dict[str, str], k: int = 5) -> float:
    """nDCG@k with gain (2^grade - 1); each target is credited once, at the first document satisfying it."""
    credited: set[str] = set()
    dcg = 0.0
    for i, doc in enumerate(ranked[:k], start=1):
        best = 0
        best_t = None
        for t, grade in relevant.items():
            if t not in credited and target_matches(t, doc, commit_keys) and grade > best:
                best, best_t = grade, t
        if best_t is not None:
            credited.add(best_t)
            dcg += (2**best - 1) / math.log2(i + 1)
    ideal = sorted(relevant.values(), reverse=True)[:k]
    idcg = sum((2**g - 1) / math.log2(i + 1) for i, g in enumerate(ideal, start=1))
    return dcg / idcg if idcg else 0.0


def percentile(values: Sequence[float], p: float) -> float:
    if not values:
        return 0.0
    xs = sorted(values)
    pos = (len(xs) - 1) * p / 100.0
    lo, hi = math.floor(pos), math.ceil(pos)
    return xs[lo] + (xs[hi] - xs[lo]) * (pos - lo)
