"""Path 1 -- lexical exact search: SQLite FTS5 BM25 over the code-aware term stream."""

from __future__ import annotations

from ..storage.db import EngramDB
from ..text.analyzer import analyze_query

# Column weights for bm25(): title, symbol, path, body.
BM25_WEIGHTS = (3.0, 4.0, 2.0, 1.0)


def fts_match_expression(terms: list[str]) -> str:
    return " OR ".join('"' + t.replace('"', '""') + '"' for t in terms)


def bm25_search(
    db: EngramDB, query: str, limit: int = 50, kinds: set[str] | None = None, exclude: set[str] | None = None
) -> list[tuple[int, float]]:
    """Return ``(doc_id, bm25)`` best-first; bm25 is negated so that larger is better."""
    terms = analyze_query(query)
    if not terms:
        return []
    clauses, params = ["documents_fts MATCH ?"], [fts_match_expression(terms)]
    if kinds:
        clauses.append(f"d.kind IN ({','.join('?' * len(kinds))})")
        params.extend(sorted(kinds))
    if exclude:
        clauses.append(f"d.kind NOT IN ({','.join('?' * len(exclude))})")
        params.extend(sorted(exclude))
    w = ", ".join(str(x) for x in BM25_WEIGHTS)
    sql = (
        f"SELECT d.id AS id, bm25(documents_fts, {w}) AS s FROM documents_fts "
        f"JOIN documents d ON d.id = documents_fts.rowid WHERE {' AND '.join(clauses)} ORDER BY s LIMIT ?"
    )
    params.append(limit)
    return [(int(r["id"]), -float(r["s"])) for r in db.query(sql, params)]
