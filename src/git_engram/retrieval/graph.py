"""Path 3 -- spreading activation over the Temporal Co-Change Graph (held in memory as adjacency lists)."""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass

from ..storage.db import EngramDB


@dataclass(slots=True)
class Neighbor:
    node: str
    weight: float
    confidence: float
    support: int
    last_ts: float
    static_link: bool


class CoChangeGraph:
    def __init__(self, adjacency: dict[str, dict[str, list[Neighbor]]]):
        self.adj = adjacency

    @classmethod
    def from_db(cls, db: EngramDB) -> CoChangeGraph:
        adj: dict[str, dict[str, list[Neighbor]]] = {"file": defaultdict(list), "symbol": defaultdict(list)}
        for r in db.query("SELECT * FROM cochange_edges ORDER BY level, src, weight DESC"):
            adj[r["level"]][r["src"]].append(
                Neighbor(r["dst"], r["weight"], r["confidence"], r["support"], r["last_ts"], bool(r["static_link"]))
            )
        return cls({k: dict(v) for k, v in adj.items()})

    def neighbors(self, node: str, level: str = "file", min_weight: float = 0.0) -> list[Neighbor]:
        return [n for n in self.adj.get(level, {}).get(node, []) if n.weight >= min_weight]

    def edge_count(self) -> int:
        return sum(len(v) for level in self.adj.values() for v in level.values())

    def memory_bytes(self) -> int:
        # adjacency entry ~ (pointer + 5 numeric fields + string ref); estimate with 64 bytes per edge.
        return self.edge_count() * 64

    def spread(
        self,
        seeds: dict[str, float],
        level: str = "file",
        decay: float = 0.85,
        min_weight: float = 0.1,
        static_discount: float = 1.0,
    ) -> dict[str, tuple[float, str, Neighbor]]:
        """One-hop spreading activation: a(v) = decay * sum_s a(s) * W(s, v) * (static_discount if import-linked).

        Import-linked neighbours are discounted because they are already reachable through static analysis;
        the graph path exists to surface *hidden* (purely temporal) couplings.

        Returns ``node -> (activation, strongest_source, strongest_edge)``.
        """
        act: dict[str, float] = defaultdict(float)
        best: dict[str, tuple[float, str, Neighbor]] = {}
        for src, a in seeds.items():
            for nb in self.neighbors(src, level, min_weight):
                contribution = decay * a * nb.weight * (static_discount if nb.static_link else 1.0)
                act[nb.node] += contribution
                if nb.node not in best or contribution > best[nb.node][0]:
                    best[nb.node] = (contribution, src, nb)
        return {node: (value, best[node][1], best[node][2]) for node, value in act.items()}
