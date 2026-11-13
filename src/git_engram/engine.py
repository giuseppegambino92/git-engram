"""`Engram`: the facade tying indexer, retrieval, graveyard, invariants and the context firewall together."""

from __future__ import annotations

import threading
import time
from pathlib import Path

from pydantic import BaseModel, Field

from .config import EngramConfig
from .documents import embed_documents, load_rotation
from .embeddings import QuantizedVectorIndex, create_embedder
from .graveyard.exec_watcher import CapturedRun, run_watched
from .graveyard.store import Graveyard, RecordResult
from .indexer.ast_chunker import chunk_source
from .indexer.gitio import Git
from .indexer.history import LOAD_BEARING_INTENTS
from .indexer.pipeline import Indexer, IndexStats, ProgressFn
from .invariants import InvariantStore
from .models import CouplingAlert, FailureWarning, Hit, InvariantHit, SearchResponse, SecurityFinding
from .retrieval.engine import FULL, HybridRetriever, RetrievalMode
from .retrieval.graph import CoChangeGraph
from .retrieval.lexical import bm25_search
from .retrieval.reranker import InteractionReranker, OnnxCrossEncoderReranker, idf_coverage
from .security.sanitizer import CharTrigramLM, ContextFirewall
from .storage.db import EngramDB

ENGRAM_DIR = ".engram"


class HistoryEntry(BaseModel):
    hash: str
    ts: float
    subject: str
    intent: str
    added: int = 0
    removed: int = 0
    excerpt: str = ""


class OwnershipEntry(BaseModel):
    hash: str
    subject: str
    intent: str
    ts: float
    lines: int
    share: float


class SymbolContext(BaseModel):
    qualname: str
    kind: str
    path: str
    start_line: int
    end_line: int
    signature: str
    docstring: str
    content: str
    history: list[HistoryEntry] = Field(default_factory=list)
    ownership: list[OwnershipEntry] = Field(default_factory=list)


class PreflightReport(BaseModel):
    file_path: str
    symbol_name: str | None
    intended_change: str | None
    indexed: bool
    language: str | None = None
    symbols: list[SymbolContext] = Field(default_factory=list)
    file_symbols: list[str] = Field(default_factory=list)
    file_history: list[HistoryEntry] = Field(default_factory=list)
    load_bearing: list[str] = Field(default_factory=list)
    coupling: list[CouplingAlert] = Field(default_factory=list)
    warnings: list[FailureWarning] = Field(default_factory=list)
    invariants: list[InvariantHit] = Field(default_factory=list)
    related: list[Hit] = Field(default_factory=list)
    security: list[SecurityFinding] = Field(default_factory=list)
    latency_ms: float = 0.0


class Engram:
    def __init__(self, path: Path | str = ".", config: EngramConfig | None = None):
        self.git = Git.discover(path)
        self.root = self.git.root
        self.cfg = config or EngramConfig.load(self.root)
        (self.root / ENGRAM_DIR).mkdir(exist_ok=True)
        ignore = self.root / ENGRAM_DIR / ".gitignore"
        if not ignore.exists():
            ignore.write_text(
                "# git-engram local index (invariants.okf.json is meant to be committed)\nengram.db*\n*.tmp\n"
            )
        self.db = EngramDB(self.root / ENGRAM_DIR / "engram.db")
        self.embedder = create_embedder(self.cfg.embedding)
        idx = self.cfg.index
        self.graveyard = Graveyard(
            self.db,
            self.git,
            lambda p, s: chunk_source(
                p, s, allow_download=idx.allow_grammar_download, prefer_tree_sitter=idx.tree_sitter
            ),
        )
        self.invariants = InvariantStore(self.db, self.root)
        self.firewall = ContextFirewall()
        self._retriever: HybridRetriever | None = None
        self._lock = threading.RLock()
        self._refreshing = False
        self.mrl_dim = min(self.cfg.embedding.mrl_dim, getattr(self.embedder, "dim", self.cfg.embedding.full_dim))

    # ------------------------------------------------------------ lifecycle
    def close(self) -> None:
        if self._retriever is not None:
            self._retriever.close()
        self.db.close()

    def __enter__(self) -> Engram:
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()

    @property
    def is_indexed(self) -> bool:
        return self.db.get_meta("last_head") is not None or self.db.get_meta("embedder_state") is not None

    def index(self, rebuild: bool = False, progress: ProgressFn | None = None) -> IndexStats:
        with self._lock:
            indexer = Indexer(
                self.root, self.db, self.git, self.cfg, self.embedder, self.graveyard, self.invariants, progress
            )
            stats = indexer.run(rebuild=rebuild)
            self.reload()
            return stats

    def reload(self) -> None:
        with self._lock:
            state = self.db.get_meta("embedder_state")
            if state is not None:
                self.embedder.load_state(state)
            lm_state = self.db.get_meta("lm_state")
            self.firewall = ContextFirewall(CharTrigramLM.from_state(lm_state) if lm_state else None)
            reranker = None
            if self.cfg.retrieval.rerank:
                ce = getattr(self.cfg.retrieval, "cross_encoder_dir", None)
                reranker = OnnxCrossEncoderReranker(ce) if ce else InteractionReranker(self.embedder.term_idf)
            if self._retriever is not None:
                self._retriever.close()
            self._retriever = HybridRetriever(
                self.db,
                self.embedder,
                QuantizedVectorIndex.from_db(self.db, self.mrl_dim),
                CoChangeGraph.from_db(self.db),
                self.cfg.retrieval,
                float(self.db.get_meta("ref_ts") or time.time()),
                reranker,
            )

    @property
    def retriever(self) -> HybridRetriever:
        if self._retriever is None:
            if not self.is_indexed:
                self.index()
            else:
                self.reload()
        assert self._retriever is not None
        return self._retriever

    def head_signature(self) -> str:
        """Cheap (no subprocess) HEAD fingerprint: contents of .git/HEAD plus the ref it points to."""
        git_dir = self.root / ".git"
        if git_dir.is_file():  # worktree / submodule: "gitdir: <path>"
            git_dir = (self.root / git_dir.read_text().split(":", 1)[1].strip()).resolve()
        try:
            head = (git_dir / "HEAD").read_text().strip()
        except OSError:
            return ""
        if head.startswith("ref: "):
            ref = git_dir / head[5:]
            try:
                return ref.read_text().strip()
            except OSError:
                packed = git_dir / "packed-refs"
                if packed.exists():
                    for line in packed.read_text().splitlines():
                        if line.endswith(" " + head[5:]):
                            return line.split(" ", 1)[0]
                return head
        return head

    def ensure_fresh(self, background: bool = True) -> bool:
        """Re-index incrementally if HEAD moved since the last index. Returns True if a refresh started."""
        if self._refreshing:
            return False
        if self.head_signature() == (self.db.get_meta("last_head") or ""):
            return False
        self._refreshing = True

        def work() -> None:
            try:
                self.index()
            finally:
                self._refreshing = False

        if background:
            threading.Thread(target=work, name="engram-refresh", daemon=True).start()
        else:
            work()
        return True

    # ------------------------------------------------------------- helpers
    def rel(self, file_path: str) -> str:
        p = Path(file_path)
        if p.is_absolute():
            try:
                return p.resolve().relative_to(self.root.resolve()).as_posix()
            except ValueError:
                return p.as_posix()
        return p.as_posix().lstrip("./") if not file_path.startswith("../") else file_path

    def _clean(self, text: str, findings: list[SecurityFinding]) -> str:
        result = self.firewall.scan(text)
        findings.extend(result.findings)
        return result.text

    def _store_doc(self, doc: tuple) -> int:
        if self.db.get_meta("embedder_state") is not None:
            _ = self.retriever  # loads embedder state before embedding the new document
        rotation = load_rotation(self.db, self.mrl_dim)
        with self.db.transaction() as conn:
            doc_id = self.db.upsert_document(conn, *doc)
            if self.db.get_meta("embedder_state") is not None:
                embed_documents(self.db, conn, self.embedder, self.mrl_dim, [doc_id], rotation)
        if self._retriever is not None:
            self._retriever.vindex = QuantizedVectorIndex.from_db(self.db, self.mrl_dim)
            self._retriever.register(doc_id, doc[0], doc[1], doc[2], doc[3], doc[4], doc[6])
        return doc_id

    # --------------------------------------------------------- negative RAG
    def graveyard_matches(
        self,
        query: str | None,
        path: str | None = None,
        symbol: str | None = None,
        context_files: list[str] | None = None,
        query_vec=None,
    ) -> list[FailureWarning]:
        cfg = self.cfg.graveyard
        scores: dict[int, float] = {}
        reasons: dict[int, list[str]] = {}

        structural: dict[int, tuple[float, str]] = {}

        def note(fid: int, reason: str) -> None:
            reasons.setdefault(fid, [])
            if reason not in reasons[fid]:
                reasons[fid].append(reason)

        def add_structural(fid: int, value: float, reason: str) -> None:
            # Structural evidence types are alternatives: keep the strongest, never sum them.
            if value > structural.get(fid, (0.0, ""))[0]:
                structural[fid] = (value, reason)

        def add(fid: int, value: float, reason: str) -> None:
            scores[fid] = scores.get(fid, 0.0) + value
            note(fid, reason)

        if path:
            for r in self.db.query("SELECT failure_id, symbol FROM failure_targets WHERE path = ?", (path,)):
                fid = int(r["failure_id"])
                sym = r["symbol"] or ""
                if symbol and sym and (sym == symbol or sym.endswith("." + symbol) or symbol.endswith("." + sym)):
                    add_structural(fid, 0.75, f"same symbol {path}::{sym}")
                else:
                    add_structural(fid, 0.5, f"same file {path}")
        for cf in (context_files or [])[:3]:
            if cf == path:
                continue
            for r in self.db.query("SELECT DISTINCT failure_id FROM failure_targets WHERE path = ?", (cf,)):
                add_structural(int(r["failure_id"]), cfg.context_weight, f"targets retrieved file {cf}")
        for fid, (value, reason) in structural.items():
            add(fid, value, reason)
        if query:
            # Textual evidence is absolute (not rank-normalised) so that "no related failure" stays silent:
            # max(dense cosine, IDF-weighted share of query terms present in the failure record).
            retriever = self.retriever
            q = query_vec if query_vec is not None else self.embedder.embed_query(query)
            textual: dict[int, tuple[float, str]] = {}
            for h in retriever.vindex.search(q, 8, retriever.cfg.stage1_candidates, kinds={"failure"}):
                if h.doc_id in retriever.meta and h.cosine > 0.0:
                    fid = int(retriever.meta[h.doc_id].doc_key.split(":", 1)[1])
                    textual[fid] = (h.cosine, f"semantic match cos={h.cosine:.2f}")
            lex = bm25_search(self.db, query, 8, kinds={"failure"})
            bodies = retriever._bodies([d for d, _ in lex])
            for doc_id, _ in lex:
                m = retriever.meta.get(doc_id)
                if m is None:
                    continue
                fid = int(m.doc_key.split(":", 1)[1])
                cov = idf_coverage(query, f"{m.title}\n{bodies.get(doc_id, '')}", self.embedder.term_idf)
                if cov > textual.get(fid, (0.0, ""))[0]:
                    textual[fid] = (cov, f"error-signature term overlap {cov:.0%}")
            for fid, (value, reason) in textual.items():
                if value > 0.15:
                    add(fid, cfg.textual_weight * value, reason)
        selected = sorted((fid for fid, s in scores.items() if s >= cfg.semantic_threshold), key=lambda f: -scores[f])[
            : cfg.max_warnings
        ]
        out: list[FailureWarning] = []
        for fid in selected:
            row = self.db.query_one("SELECT * FROM failures WHERE id = ?", (fid,))
            if row is None:
                continue
            targets = self.graveyard.targets_of(fid)
            diff_lines = [
                ln
                for ln in row["attempted_diff"].splitlines()
                if ln.startswith(("+", "-")) and not ln.startswith(("+++", "---"))
            ]
            out.append(
                FailureWarning(
                    failure_id=fid,
                    ts=row["ts"],
                    files=sorted({p for p, _ in targets}),
                    symbols=sorted({f"{p}::{s}" for p, s in targets if s}),
                    command=row["command"],
                    exit_code=row["exit_code"],
                    signature=row["signature"],
                    notes=row["notes"],
                    diff_excerpt="\n".join(diff_lines[:16]),
                    occurrences=row["occurrences"],
                    match_reason="; ".join(reasons[fid]),
                    score=round(scores[fid], 4),
                )
            )
        return out

    def coupling_for(self, path: str, limit: int = 6) -> list[CouplingAlert]:
        graph = self.retriever.graph
        return [
            CouplingAlert(
                source=path,
                target=n.node,
                weight=round(n.weight, 4),
                confidence=round(n.confidence, 4),
                support=n.support,
                hidden=not n.static_link,
                last_ts=n.last_ts,
            )
            for n in graph.neighbors(path, "file", self.cfg.retrieval.graph_min_weight)[:limit]
        ]

    # --------------------------------------------------------------- search
    def search(self, query: str, top_k: int = 8, file: str | None = None, mode: RetrievalMode = FULL) -> SearchResponse:
        t0 = time.perf_counter()
        file = self.rel(file) if file else None
        result = self.retriever.retrieve(query, top_k, mode, file_hint=file)
        findings: list[SecurityFinding] = []
        for h in result.hits:
            h.title = self._clean(h.title, findings)
            h.snippet = self._clean(h.snippet, findings)
        coupling: list[CouplingAlert] = []
        warnings: list[FailureWarning] = []
        invariants: list[InvariantHit] = []
        if mode.negative:
            t1 = time.perf_counter()
            hit_files = [h.path for h in result.hits if h.path]
            anchor_files = ([file] if file else []) + [p for p in dict.fromkeys(hit_files) if p != file][:2]
            seen: set[str] = set()
            for p in anchor_files:
                for alert in self.coupling_for(p, 4):
                    if alert.target not in seen and alert.target not in anchor_files:
                        seen.add(alert.target)
                        coupling.append(alert)
            warnings = self.graveyard_matches(query, file, None, hit_files, result.query_vec)
            for w in warnings:
                w.signature = self._clean(w.signature, findings)
                w.diff_excerpt = self._clean(w.diff_excerpt, findings)
                w.notes = self._clean(w.notes, findings)
            seen_rules: set[str] = set()
            for p in dict.fromkeys(([file] if file else []) + hit_files[:5]):
                for inv in self.invariants.match(p):
                    if inv.rule_id not in seen_rules:
                        seen_rules.add(inv.rule_id)
                        inv.description = self._clean(inv.description, findings)
                        invariants.append(inv)
            result.timings_ms["negative_rag"] = (time.perf_counter() - t1) * 1000
        result.timings_ms["end_to_end"] = (time.perf_counter() - t0) * 1000
        profile = result.profile
        return SearchResponse(
            query_class=profile.klass if profile else None,
            fusion_weights=profile.weights if profile else {},
            query=query,
            hits=result.hits,
            coupling=coupling[:6],
            warnings=warnings,
            invariants=invariants,
            security=findings,
            timings_ms=result.timings_ms,
        )

    # ------------------------------------------------------------ preflight
    def _history(self, path: str, qualname: str, limit: int = 6) -> list[HistoryEntry]:
        rows = self.db.query(
            "SELECT c.hash, c.ts, c.subject, c.intent, sc.added, sc.removed, sc.excerpt FROM symbol_commits sc "
            "JOIN commits c ON c.hash = sc.hash WHERE sc.path = ? AND sc.qualname = ? ORDER BY c.ts DESC LIMIT ?",
            (path, qualname, limit),
        )
        return [HistoryEntry(**dict(r)) for r in rows]

    def _ownership(self, path: str, qualname: str, limit: int = 5) -> list[OwnershipEntry]:
        rows = self.db.query(
            "SELECT b.hash, b.lines, c.subject, c.intent, c.ts FROM symbol_blame b JOIN commits c ON c.hash = b.hash "
            "WHERE b.path = ? AND b.qualname = ? ORDER BY b.lines DESC",
            (path, qualname),
        )
        total = sum(r["lines"] for r in rows) or 1
        return [
            OwnershipEntry(
                hash=r["hash"],
                subject=r["subject"],
                intent=r["intent"],
                ts=r["ts"],
                lines=r["lines"],
                share=r["lines"] / total,
            )
            for r in rows[:limit]
        ]

    def preflight(
        self, file_path: str, symbol_name: str | None = None, intended_change_query: str | None = None
    ) -> PreflightReport:
        t0 = time.perf_counter()
        retriever = self.retriever
        path = self.rel(file_path)
        findings: list[SecurityFinding] = []
        file_row = self.db.query_one("SELECT language FROM files WHERE path = ?", (path,))
        report = PreflightReport(
            file_path=path,
            symbol_name=symbol_name,
            intended_change=intended_change_query,
            indexed=file_row is not None,
            language=file_row["language"] if file_row else None,
        )
        rows = []
        if symbol_name:
            rows = self.db.query(
                "SELECT * FROM symbols WHERE path = ? AND (qualname = ? OR name = ? OR qualname LIKE ?) "
                "ORDER BY (qualname = ?) DESC, start_line LIMIT 3",
                (path, symbol_name, symbol_name, f"%.{symbol_name}", symbol_name),
            )
        report.file_symbols = [
            r["qualname"]
            for r in self.db.query(
                "SELECT qualname FROM symbols WHERE path = ? AND qualname != '<module>' ORDER BY start_line LIMIT 60",
                (path,),
            )
        ]
        for r in rows:
            ctx = SymbolContext(
                qualname=r["qualname"],
                kind=r["kind"],
                path=path,
                start_line=r["start_line"],
                end_line=r["end_line"],
                signature=self._clean(r["signature"], findings),
                docstring=self._clean(r["docstring"], findings),
                content=self._clean(r["content"][:2500], findings),
                history=self._history(path, r["qualname"]),
                ownership=self._ownership(path, r["qualname"]),
            )
            for h in ctx.history:
                h.subject = self._clean(h.subject, findings)
                h.excerpt = self._clean(h.excerpt, findings)
            for o in ctx.ownership:
                o.subject = self._clean(o.subject, findings)
                if o.intent in LOAD_BEARING_INTENTS:
                    report.load_bearing.append(
                        f"{o.lines} line(s) of {ctx.qualname} ({o.share:.0%}) were written by {o.hash[:10]} "
                        f'[{o.intent}] "{o.subject}" -- removing them may re-open that {o.intent}.'
                    )
            for h in ctx.history:
                if h.intent in LOAD_BEARING_INTENTS and not any(h.hash[:10] in lb for lb in report.load_bearing):
                    report.load_bearing.append(
                        f'{ctx.qualname} was changed by {h.hash[:10]} [{h.intent}] "{h.subject}".'
                    )
            report.symbols.append(ctx)
        if not rows:
            report.file_history = [
                HistoryEntry(**dict(r))
                for r in self.db.query(
                    "SELECT c.hash, c.ts, c.subject, c.intent, f.added, f.removed FROM commit_files f "
                    "JOIN commits c ON c.hash = f.hash WHERE f.path = ? ORDER BY c.ts DESC LIMIT 8",
                    (path,),
                )
            ]
            for h in report.file_history:
                h.subject = self._clean(h.subject, findings)
                if h.intent in LOAD_BEARING_INTENTS:
                    report.load_bearing.append(f'{path} was changed by {h.hash[:10]} [{h.intent}] "{h.subject}".')
        report.coupling = self.coupling_for(path)
        if symbol_name and rows:
            for r in rows[:1]:
                for nb in retriever.graph.neighbors(
                    f"{path}::{r['qualname']}", "symbol", self.cfg.retrieval.graph_min_weight
                )[:4]:
                    report.coupling.append(
                        CouplingAlert(
                            source=f"{path}::{r['qualname']}",
                            target=nb.node,
                            weight=round(nb.weight, 4),
                            confidence=round(nb.confidence, 4),
                            support=nb.support,
                            hidden=not nb.static_link,
                            last_ts=nb.last_ts,
                        )
                    )
        qualname = rows[0]["qualname"] if rows else symbol_name
        q_vec = self.embedder.embed_query(intended_change_query) if intended_change_query else None
        report.warnings = self.graveyard_matches(intended_change_query, path, qualname, None, q_vec)
        for w in report.warnings:
            w.signature = self._clean(w.signature, findings)
            w.diff_excerpt = self._clean(w.diff_excerpt, findings)
            w.notes = self._clean(w.notes, findings)
        report.invariants = self.invariants.match(path, qualname)
        for inv in report.invariants:
            inv.description = self._clean(inv.description, findings)
        if intended_change_query:
            res = retriever.retrieve(intended_change_query, 4, FULL, file_hint=path)
            for h in res.hits:
                h.title = self._clean(h.title, findings)
                h.snippet = self._clean(h.snippet, findings)
            report.related = [h for h in res.hits if not (h.path == path and h.symbol == qualname)][:3]
        report.security = findings
        report.latency_ms = round((time.perf_counter() - t0) * 1000, 3)
        return report

    # ---------------------------------------------------------- write paths
    def record_failure(
        self,
        *,
        file_path: str | None,
        attempted_diff: str,
        error_trace: str,
        notes: str = "",
        command: str = "agent-reported",
        exit_code: int = 1,
        symbol: str | None = None,
        source: str = "manual",
        ts: float | None = None,
    ) -> RecordResult:
        cfg = self.cfg.graveyard
        result = self.graveyard.record(
            command=command,
            exit_code=exit_code,
            attempted_diff=attempted_diff,
            error_output=error_trace,
            notes=notes,
            source=source,
            file_path=self.rel(file_path) if file_path else None,
            symbol=symbol,
            ts=ts,
            max_diff_bytes=cfg.max_diff_bytes,
            max_stderr_bytes=cfg.max_stderr_bytes,
        )
        doc = self.graveyard.document(result.failure_id)
        if doc is not None:
            self._store_doc(doc)
        return result

    def record_invariant(
        self, rule_id: str, scope: str | list[str], description: str, severity: str = "warning", source: str = "agent"
    ) -> dict:
        inv = self.invariants.record(rule_id, scope, description, severity, source)
        self._store_doc(self.invariants.document(inv))
        return inv

    def exec(
        self, argv: list[str], echo: bool = True, record_clean: bool = False, notes: str = ""
    ) -> tuple[CapturedRun, RecordResult | None]:
        run = run_watched(argv, cwd=self.root, echo=echo, max_bytes=self.cfg.graveyard.max_stderr_bytes)
        if run.exit_code == 0:
            return run, None
        diff = self.git.diff_head()
        for path in [p for p in self.git.untracked() if not p.startswith(ENGRAM_DIR + "/")][:20]:
            full = self.root / path
            try:
                if full.is_file() and full.stat().st_size < 65_536:
                    body = full.read_text("utf-8")
                    lines = body.splitlines()
                    diff += (
                        f"\ndiff --git a/{path} b/{path}\nnew file mode 100644\n--- /dev/null\n+++ b/{path}\n"
                        f"@@ -0,0 +1,{len(lines)} @@\n" + "\n".join("+" + ln for ln in lines) + "\n"
                    )
            except (OSError, UnicodeDecodeError):
                continue
        if not diff.strip() and not record_clean:
            return run, None
        result = self.record_failure(
            file_path=None,
            attempted_diff=diff,
            error_trace=run.combined,
            notes=notes,
            command=run.command,
            exit_code=run.exit_code,
            source="exec",
        )
        return run, result

    # ---------------------------------------------------------------- stats
    def stats(self) -> dict:
        retriever = self.retriever
        counts = {
            t: int(self.db.query_one(f"SELECT COUNT(*) AS n FROM {t}")["n"])
            for t in ("files", "symbols", "commits", "documents", "vectors", "cochange_edges", "failures", "invariants")
        }
        kinds = {r["kind"]: r["n"] for r in self.db.query("SELECT kind, COUNT(*) AS n FROM documents GROUP BY kind")}
        return {
            "root": str(self.root),
            "head": self.db.get_meta("last_head"),
            "counts": counts,
            "documents_by_kind": kinds,
            "embedder": self.embedder.kind,
            "mrl_dim": self.mrl_dim,
            "full_dim": getattr(self.embedder, "dim", None),
            "memory_bytes": {
                "vectors_quantized": retriever.vindex.memory_bytes(),
                "vectors_float32_equivalent": len(retriever.vindex) * (getattr(self.embedder, "dim", 0) * 4),
                "fts5_index": self.db.fts_bytes(),
                "cochange_graph": retriever.graph.memory_bytes(),
                "db_file": self.db.file_bytes(),
            },
            "graveyard": self.graveyard.stats(),
        }
