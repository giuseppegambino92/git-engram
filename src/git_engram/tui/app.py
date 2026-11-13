"""`git-engram tui`: interactive Textual dashboard over a live engine.

Tabs: live hybrid search with per-path score breakdown, the temporal co-change graph, the Negative-RAG graveyard
with syntax-highlighted failed diffs, and an index overview. Install with ``pip install 'git-engram[tui]'``.
"""

from __future__ import annotations

import time

from rich.console import Group
from rich.panel import Panel
from rich.syntax import Syntax
from rich.table import Table
from rich.text import Text
from textual.app import App, ComposeResult
from textual.binding import Binding
from textual.containers import Horizontal, Vertical, VerticalScroll
from textual.widgets import DataTable, Footer, Header, Input, Static, TabbedContent, TabPane, Tree

from ..engine import Engram
from ..models import SearchResponse


def _date(ts: float | None) -> str:
    return time.strftime("%Y-%m-%d", time.gmtime(ts)) if ts else "-"


class EngramApp(App):
    """Interactive explorer for a git-engram index."""

    TITLE = "git-engram"
    CSS = """
    #search-input { margin: 0 1; }
    #search-status { height: 1; margin: 0 1; color: $text-muted; }
    #results { height: 1fr; }
    #hit-detail, #failure-detail, #overview-body { padding: 0 1; }
    .left { width: 45%; }
    .right { width: 55%; border-left: solid $primary-darken-2; }
    #coupling-tree { height: 1fr; }
    """
    BINDINGS = [
        # Function keys: printable keys would be swallowed by the search box while it has focus.
        Binding("f1", "show('search')", "Search"),
        Binding("f2", "show('graph')", "Co-change"),
        Binding("f3", "show('graveyard')", "Graveyard"),
        Binding("f4", "show('overview')", "Overview"),
        Binding("escape", "blur_search", "Leave search box", show=False),
        Binding("ctrl+q", "quit", "Quit"),
        Binding("q", "quit", "Quit", show=False),
        Binding("/", "focus_search", "Search", show=False),
    ]

    def __init__(self, engine: Engram, initial_query: str = ""):
        super().__init__()
        self.engine = engine
        self.initial_query = initial_query
        self.last_response: SearchResponse | None = None

    # ----------------------------------------------------------------- layout
    def compose(self) -> ComposeResult:
        yield Header(show_clock=False)
        with TabbedContent(initial="search"):
            with TabPane("Search [F1]", id="search"):
                yield Input(placeholder="why does ... / ERR_CODE / add a field to ...", id="search-input")
                yield Static("", id="search-status")
                with Horizontal():
                    with Vertical(classes="left"):
                        yield DataTable(id="results", cursor_type="row", zebra_stripes=True)
                    with VerticalScroll(classes="right"):
                        yield Static("Type a query. Results update as you type.", id="hit-detail")
            with TabPane("Co-change graph [F2]", id="graph"):
                with Horizontal():
                    with Vertical(classes="left"):
                        yield DataTable(id="hotspots", cursor_type="row", zebra_stripes=True)
                    with Vertical(classes="right"):
                        yield Tree("select a file", id="coupling-tree")
            with TabPane("Graveyard [F3]", id="graveyard"):
                with Horizontal():
                    with Vertical(classes="left"):
                        yield DataTable(id="failures", cursor_type="row", zebra_stripes=True)
                    with VerticalScroll(classes="right"):
                        yield Static("No failure selected.", id="failure-detail")
            with TabPane("Overview [F4]", id="overview"):
                with VerticalScroll():
                    yield Static("", id="overview-body")
        yield Footer()

    def on_mount(self) -> None:
        self.sub_title = str(self.engine.root)
        results = self.query_one("#results", DataTable)
        results.add_columns("#", "kind", "result", "BM25", "Ham", "cos", "graph", "TD-RRF", "rerank")
        hot = self.query_one("#hotspots", DataTable)
        hot.add_columns("file", "edges", "max W", "hidden")
        fails = self.query_one("#failures", DataTable)
        fails.add_columns("id", "date", "seen", "command", "targets")
        self._load_hotspots()
        self._load_failures()
        self._load_overview()
        if self.initial_query:
            self.query_one("#search-input", Input).value = self.initial_query
            self.run_search(self.initial_query)
        self.query_one("#search-input", Input).focus()

    # ------------------------------------------------------------------ actions
    def action_focus_search(self) -> None:
        self.query_one(TabbedContent).active = "search"
        self.query_one("#search-input", Input).focus()

    def action_show(self, tab: str) -> None:
        self.query_one(TabbedContent).active = tab
        target = {"search": "#search-input", "graph": "#hotspots", "graveyard": "#failures"}.get(tab)
        if target:
            self.query_one(target).focus()

    def action_blur_search(self) -> None:
        self.query_one("#results", DataTable).focus()

    # ------------------------------------------------------------------- search
    def on_input_changed(self, event: Input.Changed) -> None:
        if event.input.id == "search-input" and len(event.value.strip()) >= 3:
            self.run_search(event.value.strip())

    def on_input_submitted(self, event: Input.Submitted) -> None:
        if event.input.id == "search-input" and event.value.strip():
            self.run_search(event.value.strip())

    def run_search(self, query: str) -> SearchResponse:
        resp = self.engine.search(query, top_k=12)
        self.last_response = resp
        table = self.query_one("#results", DataTable)
        table.clear()
        for i, h in enumerate(resp.hits, start=1):
            label = h.title if h.kind == "commit" else f"{h.title}  [{h.path}]"
            table.add_row(
                str(i),
                h.kind,
                label[:70],
                f"#{h.bm25_rank}" if h.bm25_rank else "-",
                str(h.hamming) if h.hamming is not None else "-",
                f"{h.cosine:.2f}" if h.cosine is not None else "-",
                f"{h.graph_activation:.2f}" if h.graph_activation else "-",
                f"{h.fused:.4f}",
                f"{h.rerank_score:.3f}" if h.rerank_score is not None else "-",
                key=str(i - 1),
            )
        weights = " ".join(f"{k}={v:.1f}" for k, v in resp.fusion_weights.items())
        extras = []
        if resp.warnings:
            extras.append(f"[red]{len(resp.warnings)} failed attempt(s)[/red]")
        if resp.invariants:
            extras.append(f"[yellow]{len(resp.invariants)} invariant(s)[/yellow]")
        if resp.coupling:
            extras.append(f"{len(resp.coupling)} coupling alert(s)")
        status = self.query_one("#search-status", Static)
        status.update(
            f"class [b]{resp.query_class or '-'}[/b]  weights {weights}  "
            f"[b]{resp.timings_ms.get('end_to_end', 0):.2f} ms[/b]  " + "  ".join(extras)
        )
        self._show_hit(0)
        return resp

    def on_data_table_row_highlighted(self, event: DataTable.RowHighlighted) -> None:
        table_id = event.data_table.id
        if event.row_key is None or event.row_key.value is None:
            return
        if table_id == "results":
            self._show_hit(int(event.row_key.value))
        elif table_id == "hotspots":
            self._show_coupling(event.row_key.value)
        elif table_id == "failures":
            self._show_failure(int(event.row_key.value))

    def _show_hit(self, index: int) -> None:
        resp = self.last_response
        detail = self.query_one("#hit-detail", Static)
        if resp is None or not resp.hits:
            detail.update("No results.")
            return
        h = resp.hits[min(index, len(resp.hits) - 1)]
        breakdown = Table.grid(padding=(0, 2))
        breakdown.add_column(style="bold")
        breakdown.add_column()
        rows = [
            ("BM25", f"rank {h.bm25_rank}  score {h.bm25_score:.2f}" if h.bm25_rank else "-"),
            ("vector", f"rank {h.vec_rank}  hamming {h.hamming}  int8 cos {h.cosine:.3f}" if h.vec_rank else "-"),
            (
                "co-change",
                f"rank {h.graph_rank}  activation {h.graph_activation:.2f}  ({h.graph_via})" if h.graph_rank else "-",
            ),
            ("TD-RRF", f"{h.fused:.4f}  (recency x{h.recency:.2f})"),
            ("rerank", f"{h.rerank_score:.3f}" if h.rerank_score is not None else "-"),
            ("date", _date(h.ts)),
        ]
        for k, v in rows:
            breakdown.add_row(k, v)
        parts: list = [
            Text(h.title, style="bold"),
            Text(f"{h.path or ''} {h.symbol or ''}", style="dim"),
            breakdown,
            Panel(
                Syntax(
                    h.snippet, "python" if (h.path or "").endswith(".py") else "text", word_wrap=True, theme="ansi_dark"
                ),
                title=h.kind,
            ),
        ]
        if h.attached_commits:
            why = Table("commit", "intent", "why this code exists", title="attached history", expand=True)
            for c in h.attached_commits:
                why.add_row(c["hash"][:10], c["intent"], c["subject"])
            parts.append(why)
        for w in resp.warnings:
            parts.append(
                Panel(
                    Text(f"{w.notes}\n{w.signature}\n\n{w.diff_excerpt}"),
                    border_style="red",
                    title=f"do not repeat: failure #{w.failure_id} ({w.occurrences}x)",
                )
            )
        for inv in resp.invariants:
            parts.append(Panel(inv.description, border_style="yellow", title=f"{inv.rule_id} [{inv.severity}]"))
        if resp.coupling:
            ct = Table("coupled file", "W", "P", "link", title="co-change alerts", expand=True)
            for c in resp.coupling:
                ct.add_row(c.target, f"{c.weight:.2f}", f"{c.confidence:.0%}", "hidden" if c.hidden else "import")
            parts.append(ct)
        detail.update(Group(*parts))

    # -------------------------------------------------------------------- graph
    def _load_hotspots(self) -> None:
        table = self.query_one("#hotspots", DataTable)
        rows = self.engine.db.query(
            "SELECT src, COUNT(*) AS n, MAX(weight) AS w, SUM(static_link = 0) AS hidden FROM cochange_edges "
            "WHERE level = 'file' GROUP BY src ORDER BY w DESC, n DESC"
        )
        for r in rows:
            table.add_row(r["src"], str(r["n"]), f"{r['w']:.2f}", str(r["hidden"]), key=r["src"])
        if rows:
            self._show_coupling(rows[0]["src"])

    def _show_coupling(self, path: str) -> None:
        tree = self.query_one("#coupling-tree", Tree)
        tree.clear()
        tree.root.set_label(Text(path, style="bold"))
        graph = self.engine.retriever.graph
        for nb in graph.neighbors(path, "file"):
            style = "red" if not nb.static_link else "green"
            node = tree.root.add(
                Text.assemble(
                    (nb.node, style),
                    f"  W={nb.weight:.2f}  P={nb.confidence:.0%}  n={nb.support}  ",
                    ("hidden" if not nb.static_link else "import-linked", "dim"),
                ),
                expand=True,
            )
            for sym_edge in self.engine.db.query(
                "SELECT src, dst, weight FROM cochange_edges WHERE level='symbol' AND src LIKE ? AND dst LIKE ? "
                "ORDER BY weight DESC LIMIT 6",
                (f"{path}::%", f"{nb.node}::%"),
            ):
                node.add_leaf(
                    f"{sym_edge['src'].split('::', 1)[1]} <-> {sym_edge['dst'].split('::', 1)[1]}"
                    f"  W={sym_edge['weight']:.2f}"
                )
        tree.root.expand()

    # ---------------------------------------------------------------- graveyard
    def _load_failures(self) -> None:
        table = self.query_one("#failures", DataTable)
        for r in self.engine.db.query("SELECT * FROM failures ORDER BY last_ts DESC"):
            targets = ", ".join(f"{p}::{s}" if s else p for p, s in self.engine.graveyard.targets_of(r["id"])[:2])
            table.add_row(
                str(r["id"]), _date(r["ts"]), f"{r['occurrences']}x", r["command"][:40], targets[:60], key=str(r["id"])
            )
        first = self.engine.db.query_one("SELECT id FROM failures ORDER BY last_ts DESC LIMIT 1")
        if first:
            self._show_failure(int(first["id"]))

    def _show_failure(self, failure_id: int) -> None:
        row = self.engine.db.query_one("SELECT * FROM failures WHERE id = ?", (failure_id,))
        detail = self.query_one("#failure-detail", Static)
        if row is None:
            detail.update("No failure selected.")
            return
        targets = ", ".join(f"{p}::{s}" if s else p for p, s in self.engine.graveyard.targets_of(failure_id))
        header = Text.assemble(
            (f"failure #{failure_id}", "bold red"),
            f"  {_date(row['ts'])}  seen {row['occurrences']}x  exit {row['exit_code']}\n",
            ("$ " + row["command"], "bold"),
            f"\ntargets: {targets}",
        )
        parts: list = [header]
        if row["notes"]:
            parts.append(Panel(row["notes"], title="what was tried"))
        parts.append(Panel(Text(row["signature"], style="red"), title="failure signature"))
        diff = row["attempted_diff"] or "(no diff captured)"
        parts.append(Panel(Syntax(diff[:20000], "diff", theme="ansi_dark", word_wrap=True), title="attempted diff"))
        detail.update(Group(*parts))

    # ----------------------------------------------------------------- overview
    def _load_overview(self) -> None:
        s = self.engine.stats()
        mem, c = s["memory_bytes"], s["counts"]
        grid = Table.grid(padding=(0, 2))
        grid.add_column(style="bold")
        grid.add_column()
        for k, v in [
            ("repository", s["root"]),
            ("indexed HEAD", (s["head"] or "-")[:12]),
            ("files / symbols / commits", f"{c['files']} / {c['symbols']} / {c['commits']}"),
            ("documents", ", ".join(f"{k}:{v}" for k, v in sorted(s["documents_by_kind"].items()))),
            ("embedder", f"{s['embedder']} {s['full_dim']}d -> MRL {s['mrl_dim']}d (1-bit + int8)"),
            (
                "quantized vectors",
                f"{mem['vectors_quantized'] / 1024:.1f} KB (float32 {mem['vectors_float32_equivalent'] / 1024:.1f} KB)",
            ),
            ("FTS5 index", f"{mem['fts5_index'] / 1024:.1f} KB"),
            ("co-change edges", str(c["cochange_edges"])),
            ("graveyard", f"{s['graveyard']['failures']} failures / {s['graveyard']['occurrences']} occurrences"),
        ]:
            grid.add_row(k, str(v))
        fragile = Table("file", "fixes", "security", "reverts", title="load-bearing hotspots", expand=True)
        for r in self.engine.db.query(
            "SELECT f.path, SUM(c.intent='bugfix') AS fx, SUM(c.intent='security') AS sec, "
            "SUM(c.intent='revert') AS rv "
            "FROM commit_files f JOIN commits c ON c.hash = f.hash WHERE c.intent IN ('bugfix','security','revert') "
            "AND f.path IN (SELECT path FROM files) GROUP BY f.path ORDER BY (fx + 2*sec + 2*rv) DESC LIMIT 10"
        ):
            fragile.add_row(r["path"], str(r["fx"]), str(r["sec"]), str(r["rv"]))
        invs = Table("rule", "severity", "scope", "description", title="OKF invariants", expand=True)
        for inv in self.engine.invariants.all():
            invs.add_row(inv["rule_id"], inv["severity"], inv["scope"], inv["description"])
        self.query_one("#overview-body", Static).update(Group(Panel(grid, title="index"), fragile, invs))


def run_tui(engine: Engram, initial_query: str = "") -> None:
    EngramApp(engine, initial_query).run()
