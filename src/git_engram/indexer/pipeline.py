"""Indexing pipeline: git history -> AST symbols -> blame -> co-change graph -> embeddings -> firewall LM.

Incremental by default: only commits in ``last_head..HEAD`` are parsed, only files whose content hash
changed are re-chunked, and only documents without vectors are embedded. A full rebuild happens on
first run, on ``--rebuild``, after history rewrites (last indexed HEAD is no longer an ancestor), or
when the hashing embedder's vocabulary has drifted by more than 30% since it was fitted.
"""

from __future__ import annotations

import fnmatch
import hashlib
import time
from collections import defaultdict
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from ..config import EngramConfig
from ..documents import doc_text, embed_documents, file_document, symbol_document
from ..embeddings import Embedder, encode_vectors, load_rotation, make_rotation
from ..graveyard.store import Graveyard
from ..invariants import InvariantStore
from ..security.sanitizer import CharTrigramLM
from ..storage.db import EngramDB
from .ast_chunker import CODE_LANGUAGES, chunk_source, detect_language
from .cochange import rebuild_cochange
from .gitio import Git
from .history import HistoryExtractor

ProgressFn = Callable[[str, int, int], None]


@dataclass
class IndexStats:
    mode: str = "full"
    files: int = 0
    changed_files: int = 0
    symbols: int = 0
    commits: int = 0
    new_commits: int = 0
    documents: int = 0
    embedded: int = 0
    file_edges: int = 0
    symbol_edges: int = 0
    failures: int = 0
    invariants: int = 0
    parsers: dict[str, int] = field(default_factory=dict)
    timings_ms: dict[str, float] = field(default_factory=dict)


class Indexer:
    def __init__(
        self,
        root: Path,
        db: EngramDB,
        git: Git,
        cfg: EngramConfig,
        embedder: Embedder,
        graveyard: Graveyard,
        invariants: InvariantStore,
        progress: ProgressFn | None = None,
    ):
        self.root = root
        self.db = db
        self.git = git
        self.cfg = cfg
        self.embedder = embedder
        self.graveyard = graveyard
        self.invariants = invariants
        self.progress = progress or (lambda stage, done, total: None)
        allow = cfg.index.allow_grammar_download
        use_ts = cfg.index.tree_sitter
        self.chunker = lambda path, src: chunk_source(path, src, allow_download=allow, prefer_tree_sitter=use_ts)
        self.mrl_dim = min(cfg.embedding.mrl_dim, getattr(embedder, "dim", cfg.embedding.full_dim))

    def skip(self, path: str) -> bool:
        name = path.rsplit("/", 1)[-1]
        return any(fnmatch.fnmatch(path, g) or fnmatch.fnmatch(name, g) for g in self.cfg.index.skip_globs)

    # ------------------------------------------------------------------ run
    def run(self, rebuild: bool = False) -> IndexStats:
        stats = IndexStats()
        t_all = time.perf_counter()
        head = self.git.head()
        last = self.db.get_meta("last_head")
        has_embedder = (
            self.db.get_meta("embedder_state") is not None and self.db.get_meta("embedder_kind") == self.embedder.kind
        )
        full = rebuild or last is None or not has_embedder
        if not full and head and last and last != head and not self.git.is_ancestor(last, head):
            full = True
        stats.mode = "full" if full else "incremental"
        if not full:
            # A fresh process holds an unfitted embedder: restore the persisted projection before embedding new docs.
            self.embedder.load_state(self.db.get_meta("embedder_state"))
        if full:
            self.db.reset_derived()

        def timed(name: str, fn: Callable[[], None]) -> None:
            t0 = time.perf_counter()
            fn()
            stats.timings_ms[name] = (time.perf_counter() - t0) * 1000

        if head and (full or last != head):
            timed("history", lambda: self._index_history(None if full else f"{last}..{head}", stats))
        changed: list[str] = []
        timed("ast", lambda: changed.extend(self._index_files(stats)))
        timed("timestamps", self._update_timestamps)
        if self.cfg.index.blame and head:
            timed("blame", lambda: self._index_blame(changed if not full else None))
        ref_ts = self._ref_ts()
        current = {r["path"] for r in self.db.query("SELECT path FROM files")}
        graph_stats: dict[str, int] = {}
        timed("cochange", lambda: graph_stats.update(rebuild_cochange(self.db, self.cfg.graph, ref_ts, current)))
        stats.file_edges = graph_stats.get("file_edges", 0)
        stats.symbol_edges = graph_stats.get("symbol_edges", 0)
        timed("memory_docs", lambda: self._index_memory_documents(stats, full))
        timed("embeddings", lambda: self._index_embeddings(stats, full))
        if full or self.db.get_meta("lm_state") is None:
            timed("firewall_lm", self._train_lm)

        self.db.set_meta("last_head", head)
        self.db.set_meta("ref_ts", ref_ts)
        self.db.set_meta("indexed_at", time.time())
        stats.files = self._count("files")
        stats.symbols = self._count("symbols")
        stats.commits = self._count("commits")
        stats.documents = self._count("documents")
        stats.timings_ms["total"] = (time.perf_counter() - t_all) * 1000
        return stats

    def _count(self, table: str) -> int:
        return int(self.db.query_one(f"SELECT COUNT(*) AS n FROM {table}")["n"])

    def _ref_ts(self) -> float:
        row = self.db.query_one("SELECT MAX(ts) AS t FROM commits")
        return float(row["t"]) if row and row["t"] is not None else time.time()

    # -------------------------------------------------------------- history
    def _index_history(self, revision_range: str | None, stats: IndexStats) -> None:
        alias: dict[str, str] = {} if revision_range is None else dict(self.db.get_meta_json("path_alias", {}))
        known_alias = dict(alias)
        extractor = HistoryExtractor(self.git, self.chunker, self.skip)
        total = self.cfg.index.max_commits
        batch: list = []

        def flush() -> None:
            with self.db.transaction() as conn:
                for hc in batch:
                    c = hc.commit
                    conn.execute(
                        "INSERT OR REPLACE INTO commits(hash, parents, author, ts, subject, body, intent, n_files, "
                        "capsule)"
                        " VALUES(?,?,?,?,?,?,?,?,?)",
                        (
                            c.hash,
                            " ".join(c.parents),
                            c.author,
                            c.ts,
                            c.subject,
                            c.body,
                            hc.intent,
                            len(hc.file_paths),
                            hc.capsule,
                        ),
                    )
                    conn.executemany(
                        "INSERT OR REPLACE INTO commit_files(hash, path, change, added, removed) VALUES(?,?,?,?,?)",
                        [(c.hash, p, ch, a, r) for p, ch, a, r in hc.file_paths],
                    )
                    conn.executemany(
                        "INSERT OR REPLACE INTO symbol_commits(path, qualname, hash, ts, added, removed, excerpt) "
                        "VALUES(?,?,?,?,?,?,?)",
                        [(t.path, t.qualname, t.hash, t.ts, t.added, t.removed, t.excerpt) for t in hc.touches],
                    )
                    primary = _primary_file(hc.file_paths)
                    self.db.upsert_document(
                        conn, f"commit:{c.hash}", "commit", primary, None, c.subject, hc.capsule, c.ts
                    )
            batch.clear()

        for i, hc in enumerate(extractor.iter_history(self.cfg.index.max_commits, revision_range, alias)):
            batch.append(hc)
            stats.new_commits += 1
            if len(batch) >= 200:
                flush()
            if i % 25 == 0:
                self.progress("history", i, total)
        flush()
        self.progress("history", stats.new_commits, stats.new_commits)
        new_alias = {k: v for k, v in alias.items() if known_alias.get(k) != v}
        if revision_range is not None and new_alias:
            with self.db.transaction() as conn:
                for old, new in new_alias.items():
                    for table in ("commit_files", "symbol_commits", "symbol_blame"):
                        conn.execute(f"UPDATE OR REPLACE {table} SET path = ? WHERE path = ?", (new, old))
                    conn.execute("UPDATE documents SET path = ? WHERE path = ? AND kind = 'commit'", (new, old))
        self.db.set_meta_json("path_alias", alias)

    # ---------------------------------------------------------------- files
    def _index_files(self, stats: IndexStats) -> list[str]:
        paths = [p for p in self.git.ls_files() if not self.skip(p)]
        existing = {r["path"]: r["content_hash"] for r in self.db.query("SELECT path, content_hash FROM files")}
        present: set[str] = set()
        changed: list[str] = []
        for i, path in enumerate(paths):
            if i % 50 == 0:
                self.progress("ast", i, len(paths))
            full = self.root / path
            try:
                if not full.is_file() or full.stat().st_size > self.cfg.index.max_file_bytes:
                    continue
                raw = full.read_bytes()
            except OSError:
                continue
            if b"\x00" in raw[:8192]:
                continue
            present.add(path)
            digest = hashlib.sha1(raw).hexdigest()
            if existing.get(path) == digest:
                continue
            changed.append(path)
            self._replace_file(path, raw.decode("utf-8", "replace"), digest, stats)
        self.progress("ast", len(paths), len(paths))
        removed = set(existing) - present
        if removed:
            with self.db.transaction() as conn:
                for path in removed:
                    self._delete_file(conn, path)
        stats.changed_files = len(changed)
        return changed

    def _delete_file(self, conn, path: str) -> None:
        ids = [
            r[0]
            for r in conn.execute(
                "SELECT id FROM documents WHERE path = ? AND kind IN ('symbol', 'file')", (path,)
            ).fetchall()
        ]
        self.db.delete_documents(conn, ids)
        conn.execute("DELETE FROM symbols WHERE path = ?", (path,))
        conn.execute("DELETE FROM files WHERE path = ?", (path,))
        conn.execute("DELETE FROM symbol_blame WHERE path = ?", (path,))

    def _replace_file(self, path: str, source: str, digest: str, stats: IndexStats) -> None:
        language = detect_language(path)
        try:
            result = self.chunker(path, source)
        except Exception:
            from .ast_chunker import chunk_text

            result = chunk_text(path, source, language)
        stats.parsers[result.parser.split(":")[0]] = stats.parsers.get(result.parser.split(":")[0], 0) + 1
        with self.db.transaction() as conn:
            self._delete_file(conn, path)
            conn.execute(
                "INSERT INTO files(path, language, content_hash, size, n_lines, imports) VALUES(?,?,?,?,?,?)",
                (path, language, digest, len(source), source.count("\n") + 1, "\n".join(result.imports)),
            )
            for s in result.symbols:
                conn.execute(
                    "INSERT OR REPLACE INTO symbols(path, qualname, name, kind, language, signature, docstring, "
                    "start_line, end_line, content, parent) VALUES(?,?,?,?,?,?,?,?,?,?,?)",
                    (
                        s.path,
                        s.qualname,
                        s.name,
                        s.kind,
                        s.language,
                        s.signature,
                        s.docstring,
                        s.start_line,
                        s.end_line,
                        s.content,
                        s.parent,
                    ),
                )
                if language in CODE_LANGUAGES:
                    key, kind, p, sym, title, body = symbol_document(s)
                    self.db.upsert_document(conn, key, kind, p, sym, title, body, None)
            key, kind, p, sym, title, body = file_document(path, language, result.symbols, source, result.imports)
            self.db.upsert_document(conn, key, kind, p, sym or None, title, body, None)

    def _update_timestamps(self) -> None:
        with self.db.transaction() as conn:
            conn.execute(
                "UPDATE documents SET ts = (SELECT MAX(sc.ts) FROM symbol_commits sc WHERE sc.path = documents.path "
                "AND sc.qualname = documents.symbol) WHERE kind = 'symbol'"
            )
            conn.execute(
                "UPDATE documents SET ts = (SELECT MAX(c.ts) FROM commit_files f JOIN commits c ON c.hash = f.hash "
                "WHERE f.path = documents.path) WHERE kind IN ('symbol', 'file') AND ts IS NULL"
            )

    # ---------------------------------------------------------------- blame
    def _index_blame(self, only: list[str] | None) -> None:
        rows = self.db.query("SELECT path, language FROM files")
        targets = [r["path"] for r in rows if r["language"] in CODE_LANGUAGES and (only is None or r["path"] in only)]
        if not targets:
            return
        symbols: dict[str, list[tuple[str, int, int]]] = defaultdict(list)
        for r in self.db.query("SELECT path, qualname, start_line, end_line FROM symbols WHERE qualname != '<module>'"):
            symbols[r["path"]].append((r["qualname"], r["start_line"], r["end_line"]))
        with ThreadPoolExecutor(max_workers=8) as pool:
            results = list(pool.map(self.git.blame_lines, targets))
        records: list[tuple[str, str, str, int]] = []
        for path, owners in zip(targets, results, strict=True):
            for qual, start, end in symbols.get(path, []):
                counts: dict[str, int] = defaultdict(int)
                for ln in range(start, min(end, len(owners)) + 1):
                    h = owners[ln - 1]
                    if h and h != "0" * 40:
                        counts[h] += 1
                records.extend((path, qual, h, n) for h, n in counts.items())
        with self.db.transaction() as conn:
            for path in targets:
                conn.execute("DELETE FROM symbol_blame WHERE path = ?", (path,))
            conn.executemany(
                "INSERT OR REPLACE INTO symbol_blame(path, qualname, hash, lines) VALUES(?,?,?,?)", records
            )

    # ------------------------------------------------------- memory documents
    def _index_memory_documents(self, stats: IndexStats, full: bool) -> None:
        """Graveyard failures and OKF invariants are not derived from git: (re)materialise their documents."""
        self.invariants.sync_from_file()
        with self.db.transaction() as conn:
            for r in self.db.query("SELECT id FROM failures"):
                doc = self.graveyard.document(int(r["id"]))
                if doc and (
                    full or not conn.execute("SELECT 1 FROM documents WHERE doc_key = ?", (doc[0],)).fetchone()
                ):
                    self.db.upsert_document(conn, *doc)
            live = set()
            for inv in self.invariants.all():
                doc = self.invariants.document(inv)
                live.add(doc[0])
                self.db.upsert_document(conn, *doc)
            stale = [
                r[0]
                for r in conn.execute("SELECT id, doc_key FROM documents WHERE kind = 'invariant'").fetchall()
                if r[1] not in live
            ]
            self.db.delete_documents(conn, stale)
        stats.failures = self._count("failures")
        stats.invariants = self._count("invariants")

    # ------------------------------------------------------------ embeddings
    def _index_embeddings(self, stats: IndexStats, full: bool) -> None:
        n_docs = self._count("documents")
        fit_docs = int(self.db.get_meta("fit_docs") or 0)
        missing = [
            r[0]
            for r in self.db.query(
                "SELECT d.id FROM documents d LEFT JOIN vectors v ON v.doc_id = d.id WHERE v.doc_id IS NULL"
            )
        ]
        drift = fit_docs > 0 and len(missing) > 0.3 * fit_docs and self.embedder.kind == "hashing-lsa"
        if full or drift or self.db.get_meta("embedder_state") is None:
            rows = self.db.query("SELECT id, title, symbol, path, body FROM documents ORDER BY id")
            texts = [doc_text(r["title"], r["symbol"], r["path"], r["body"]) for r in rows]
            self.progress("embeddings", 0, len(texts))
            self.embedder.fit(texts)
            vecs = self.embedder.embed(texts) if texts else np.zeros((0, self.embedder.dim), np.float32)
            rotation = make_rotation(self.mrl_dim, self.cfg.embedding.seed)
            self.db.set_meta("bin_rotation", rotation.astype(np.float32).tobytes())
            self.db.set_meta("embedder_state", self.embedder.state_bytes())
            self.db.set_meta("embedder_kind", self.embedder.kind)
            self.db.set_meta("fit_docs", len(texts))
            encoded = encode_vectors(vecs, self.mrl_dim, rotation) if len(texts) else []
            with self.db.transaction() as conn:
                conn.execute("DELETE FROM vectors")
                conn.executemany(
                    "INSERT INTO vectors(doc_id, bin, i8, i8_scale, f32) VALUES(?,?,?,?,?)",
                    [(r["id"], e.bin, e.i8, e.scale, e.f32) for r, e in zip(rows, encoded, strict=True)],
                )
            stats.embedded = len(texts)
        elif missing:
            rotation = load_rotation(self.db, self.mrl_dim)
            with self.db.transaction() as conn:
                stats.embedded = embed_documents(self.db, conn, self.embedder, self.mrl_dim, missing, rotation)
        self.progress("embeddings", n_docs, n_docs)

    def _train_lm(self) -> None:
        lm = CharTrigramLM()
        rows = self.db.query("SELECT body FROM documents WHERE kind IN ('symbol', 'file', 'commit') ORDER BY id")
        lm.fit([r["body"] for r in rows])
        self.db.set_meta("lm_state", lm.state_bytes())


def _primary_file(files: list[tuple[str, str, int, int]]) -> str | None:
    if not files:
        return None

    def rank(f: tuple[str, str, int, int]) -> tuple[int, int]:
        is_test = int("test" in f[0].lower() or f[0].endswith((".md", ".lock")))
        return (is_test, -(f[2] + f[3]))

    return sorted(files, key=rank)[0][0]
