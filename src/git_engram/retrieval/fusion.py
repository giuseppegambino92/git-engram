r"""Time-Decayed Reciprocal Rank Fusion.

    Score(d) = sum_m  w_m / (k + rank_m(d))  *  (1 + alpha * exp(-lambda * dt_d))

``dt_d`` is measured against the repository's own clock (timestamp of the newest indexed commit), not
wall-clock time, so results are reproducible and an old-but-quiet repository is not uniformly penalised.
"""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence


def recency_factor(ts: float | None, ref_ts: float, alpha: float, lam: float) -> float:
    if ts is None or alpha == 0.0:
        return 1.0
    return 1.0 + alpha * math.exp(-lam * max(0.0, ref_ts - ts))


def td_rrf(
    rankings: Mapping[str, Sequence[int]],
    weights: Mapping[str, float],
    k: float,
    timestamps: Mapping[int, float | None] | None = None,
    ref_ts: float = 0.0,
    alpha: float = 0.0,
    lam: float = 0.0,
) -> list[tuple[int, float, float]]:
    """Fuse ranked lists of doc ids. Returns ``(doc_id, score, recency_factor)`` sorted best-first."""
    base: dict[int, float] = {}
    for name, ranked in rankings.items():
        w = weights.get(name, 1.0)
        for rank, doc in enumerate(ranked, start=1):
            base[doc] = base.get(doc, 0.0) + w / (k + rank)
    out = []
    for doc, score in base.items():
        rf = recency_factor((timestamps or {}).get(doc), ref_ts, alpha, lam)
        out.append((doc, score * rf, rf))
    out.sort(key=lambda x: (-x[1], x[0]))
    return out
