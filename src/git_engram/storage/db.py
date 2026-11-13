"""SQLite storage: FTS5 lexical index, quantized vector BLOBs, causal history, co-change graph,
OKF invariants and the Negative-RAG graveyard, all in one WAL-mode file under ``.engram/``."""

from __future__ import annotations

import json
import sqlite3
import threading
from collections.abc import Iterable, Iterator, Sequence
from contextlib import contextmanager
from pathlib import Path
from typing import Any

from ..text.analyzer import analyze

SCHEMA_VERSION = 1

SCHEMA = """
CREATE TABLE IF NOT EXISTS meta (key TEXT PRIMARY KEY, value BLOB);

CREATE TABLE IF NOT EXISTS files (
    path TEXT PRIMARY KEY,
    language TEXT NOT NULL,
    content_hash TEXT NOT NULL,
    size INTEGER NOT NULL,
    n_lines INTEGER NOT NULL,
    imports TEXT NOT NULL DEFAULT ''
);

CREATE TABLE IF NOT EXISTS symbols (
    id INTEGER PRIMARY KEY,
    path TEXT NOT NULL,
    qualname TEXT NOT NULL,
    name TEXT NOT NULL,
    kind TEXT NOT NULL,
    language TEXT NOT NULL,
    signature TEXT NOT NULL,
    docstring TEXT NOT NULL,
    start_line INTEGER NOT NULL,
    end_line INTEGER NOT NULL,
    content TEXT NOT NULL,
    parent TEXT,
    UNIQUE (path, qualname)
);
CREATE INDEX IF NOT EXISTS symbols_name ON symbols(name);

CREATE TABLE IF NOT EXISTS commits (
    hash TEXT PRIMARY KEY,
    parents TEXT NOT NULL,
    author TEXT NOT NULL,
    ts REAL NOT NULL,
    subject TEXT NOT NULL,
    body TEXT NOT NULL,
    intent TEXT NOT NULL,
    n_files INTEGER NOT NULL,
    capsule TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS commits_ts ON commits(ts);

CREATE TABLE IF NOT EXISTS commit_files (
    hash TEXT NOT NULL,
    path TEXT NOT NULL,
    change TEXT NOT NULL,
    added INTEGER NOT NULL,
    removed INTEGER NOT NULL,
    PRIMARY KEY (hash, path)
);
CREATE INDEX IF NOT EXISTS commit_files_path ON commit_files(path);

CREATE TABLE IF NOT EXISTS symbol_commits (
    path TEXT NOT NULL,
    qualname TEXT NOT NULL,
    hash TEXT NOT NULL,
    ts REAL NOT NULL,
    added INTEGER NOT NULL,
    removed INTEGER NOT NULL,
    excerpt TEXT NOT NULL,
    PRIMARY KEY (path, qualname, hash)
);
CREATE INDEX IF NOT EXISTS symbol_commits_hash ON symbol_commits(hash);

CREATE TABLE IF NOT EXISTS symbol_blame (
    path TEXT NOT NULL,
    qualname TEXT NOT NULL,
    hash TEXT NOT NULL,
    lines INTEGER NOT NULL,
    PRIMARY KEY (path, qualname, hash)
);

CREATE TABLE IF NOT EXISTS cochange_edges (
    level TEXT NOT NULL,
    src TEXT NOT NULL,
    dst TEXT NOT NULL,
    weight REAL NOT NULL,
    support INTEGER NOT NULL,
    confidence REAL NOT NULL,
    last_ts REAL NOT NULL,
    static_link INTEGER NOT NULL,
    PRIMARY KEY (level, src, dst)
);

CREATE TABLE IF NOT EXISTS documents (
    id INTEGER PRIMARY KEY,
    doc_key TEXT NOT NULL UNIQUE,
    kind TEXT NOT NULL,
    path TEXT,
    symbol TEXT,
    title TEXT NOT NULL,
    body TEXT NOT NULL,
    ts REAL
);
CREATE INDEX IF NOT EXISTS documents_path ON documents(path);
CREATE INDEX IF NOT EXISTS documents_kind ON documents(kind);

CREATE VIRTUAL TABLE IF NOT EXISTS documents_fts USING fts5(
    title, symbol, path, body,
    tokenize = "porter unicode61 tokenchars '_'"
);

CREATE TABLE IF NOT EXISTS vectors (
    doc_id INTEGER PRIMARY KEY,
    bin BLOB NOT NULL,
    i8 BLOB NOT NULL,
    i8_scale REAL NOT NULL,
    f32 BLOB NOT NULL
);

CREATE TABLE IF NOT EXISTS failures (
    id INTEGER PRIMARY KEY,
    ts REAL NOT NULL,
    fingerprint TEXT NOT NULL UNIQUE,
    command TEXT NOT NULL,
    exit_code INTEGER NOT NULL,
    attempted_diff TEXT NOT NULL,
    stderr TEXT NOT NULL,
    signature TEXT NOT NULL,
    failing_tests TEXT NOT NULL,
    notes TEXT NOT NULL,
    source TEXT NOT NULL,
    head TEXT NOT NULL,
    occurrences INTEGER NOT NULL DEFAULT 1,
    last_ts REAL NOT NULL
);

CREATE TABLE IF NOT EXISTS failure_targets (
    failure_id INTEGER NOT NULL,
    path TEXT NOT NULL,
    symbol TEXT NOT NULL DEFAULT '',
    PRIMARY KEY (failure_id, path, symbol)
);
CREATE INDEX IF NOT EXISTS failure_targets_path ON failure_targets(path);

CREATE TABLE IF NOT EXISTS invariants (
    rule_id TEXT PRIMARY KEY,
    scope TEXT NOT NULL,
    description TEXT NOT NULL,
    severity TEXT NOT NULL,
    created_ts REAL NOT NULL,
    source TEXT NOT NULL
);
"""

DERIVED_TABLES = (
    "files",
    "symbols",
    "commits",
    "commit_files",
    "symbol_commits",
    "symbol_blame",
    "cochange_edges",
    "documents",
    "documents_fts",
    "vectors",
)


class EngramDB:
    """Thin, explicit wrapper over a single sqlite3 connection (thread-safe via an RLock)."""

    def __init__(self, path: Path | str):
        self.path = Path(path)
        if str(path) != ":memory:":
            self.path.parent.mkdir(parents=True, exist_ok=True)
        self.conn = sqlite3.connect(str(path), check_same_thread=False, isolation_level=None)
        self.conn.row_factory = sqlite3.Row
        self.lock = threading.RLock()
        self.conn.execute("PRAGMA journal_mode=WAL")
        self.conn.execute("PRAGMA synchronous=NORMAL")
        self.conn.execute("PRAGMA temp_store=MEMORY")
        self.conn.execute("PRAGMA mmap_size=268435456")
        self.conn.executescript(SCHEMA)
        if self.get_meta("schema_version") is None:
            self.set_meta("schema_version", str(SCHEMA_VERSION))

    def close(self) -> None:
        with self.lock:
            self.conn.close()

    @contextmanager
    def transaction(self) -> Iterator[sqlite3.Connection]:
        with self.lock:
            self.conn.execute("BEGIN")
            try:
                yield self.conn
            except BaseException:
                self.conn.execute("ROLLBACK")
                raise
            else:
                self.conn.execute("COMMIT")

    def execute(self, sql: str, params: Sequence[Any] | dict[str, Any] = ()) -> sqlite3.Cursor:
        with self.lock:
            return self.conn.execute(sql, params)

    def query(self, sql: str, params: Sequence[Any] | dict[str, Any] = ()) -> list[sqlite3.Row]:
        with self.lock:
            return self.conn.execute(sql, params).fetchall()

    def query_one(self, sql: str, params: Sequence[Any] | dict[str, Any] = ()) -> sqlite3.Row | None:
        with self.lock:
            return self.conn.execute(sql, params).fetchone()

    # ------------------------------------------------------------------ meta
    def get_meta(self, key: str) -> Any:
        row = self.query_one("SELECT value FROM meta WHERE key = ?", (key,))
        return None if row is None else row["value"]

    def set_meta(self, key: str, value: Any) -> None:
        self.execute(
            "INSERT INTO meta(key, value) VALUES(?, ?) ON CONFLICT(key) DO UPDATE SET value = excluded.value",
            (key, value),
        )

    def get_meta_json(self, key: str, default: Any = None) -> Any:
        raw = self.get_meta(key)
        return default if raw is None else json.loads(raw)

    def set_meta_json(self, key: str, value: Any) -> None:
        self.set_meta(key, json.dumps(value))

    # ------------------------------------------------------------- documents
    def upsert_document(
        self,
        conn: sqlite3.Connection,
        doc_key: str,
        kind: str,
        path: str | None,
        symbol: str | None,
        title: str,
        body: str,
        ts: float | None,
    ) -> int:
        """Insert or replace a retrievable document and its FTS5 row. Returns the document id."""
        row = conn.execute("SELECT id FROM documents WHERE doc_key = ?", (doc_key,)).fetchone()
        if row is not None:
            doc_id = int(row[0])
            conn.execute(
                "UPDATE documents SET kind=?, path=?, symbol=?, title=?, body=?, ts=? WHERE id=?",
                (kind, path, symbol, title, body, ts, doc_id),
            )
            conn.execute("DELETE FROM documents_fts WHERE rowid = ?", (doc_id,))
            conn.execute("DELETE FROM vectors WHERE doc_id = ?", (doc_id,))
        else:
            cur = conn.execute(
                "INSERT INTO documents(doc_key, kind, path, symbol, title, body, ts) VALUES(?,?,?,?,?,?,?)",
                (doc_key, kind, path, symbol, title, body, ts),
            )
            doc_id = int(cur.lastrowid)
        conn.execute(
            "INSERT INTO documents_fts(rowid, title, symbol, path, body) VALUES(?,?,?,?,?)",
            (
                doc_id,
                " ".join(analyze(title)),
                " ".join(analyze(symbol or "")),
                " ".join(analyze(path or "")),
                " ".join(analyze(body)),
            ),
        )
        return doc_id

    def delete_documents(self, conn: sqlite3.Connection, doc_ids: Iterable[int]) -> None:
        ids = list(doc_ids)
        for start in range(0, len(ids), 500):
            chunk = ids[start : start + 500]
            marks = ",".join("?" * len(chunk))
            conn.execute(f"DELETE FROM documents_fts WHERE rowid IN ({marks})", chunk)
            conn.execute(f"DELETE FROM vectors WHERE doc_id IN ({marks})", chunk)
            conn.execute(f"DELETE FROM documents WHERE id IN ({marks})", chunk)

    def reset_derived(self) -> None:
        with self.transaction() as conn:
            for table in DERIVED_TABLES:
                conn.execute(f"DELETE FROM {table}")
            for key in ("last_head", "embedder_state", "embedder_kind", "ref_ts", "lm_state"):
                conn.execute("DELETE FROM meta WHERE key = ?", (key,))

    def fts_bytes(self) -> int:
        row = self.query_one("SELECT COALESCE(SUM(LENGTH(block)), 0) AS n FROM documents_fts_data")
        return int(row["n"]) if row else 0

    def file_bytes(self) -> int:
        if str(self.path) == ":memory:":
            return 0
        total = 0
        for suffix in ("", "-wal"):
            p = Path(str(self.path) + suffix)
            if p.exists():
                total += p.stat().st_size
        return total
