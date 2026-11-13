"""`git-engram` command line interface (also usable as `git engram ...`)."""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

from rich import box
from rich.console import Console, Group
from rich.panel import Panel
from rich.progress import BarColumn, Progress, SpinnerColumn, TextColumn, TimeElapsedColumn
from rich.table import Table
from rich.text import Text

from .. import __version__
from ..indexer.gitio import GitError

console = Console(stderr=False)
err = Console(stderr=True)

STAGES = {
    "history": "Git history -> causal capsules",
    "ast": "AST symbol chunking",
    "embeddings": "Embeddings + quantization",
}


def _engine(args: argparse.Namespace):
    from ..engine import Engram

    try:
        return Engram(args.repo)
    except GitError as exc:
        err.print(f"[red]error:[/red] {exc}")
        raise SystemExit(2) from exc


def _ensure_index(engine) -> None:
    if not engine.is_indexed:
        err.print("[yellow]No index yet - indexing first...[/yellow]")
        _run_index(engine, rebuild=False, quiet=True)


def _date(ts: float | None) -> str:
    return time.strftime("%Y-%m-%d", time.gmtime(ts)) if ts else "-"


# ------------------------------------------------------------------------------------- index
def _run_index(engine, rebuild: bool, quiet: bool = False):
    progress = Progress(
        SpinnerColumn(),
        TextColumn("[bold]{task.description}"),
        BarColumn(),
        TextColumn("{task.completed}/{task.total}"),
        TimeElapsedColumn(),
        console=err,
        transient=quiet,
    )
    tasks: dict[str, int] = {}

    def on_progress(stage: str, done: int, total: int) -> None:
        if stage not in tasks:
            tasks[stage] = progress.add_task(STAGES.get(stage, stage), total=max(total, 1))
        progress.update(tasks[stage], completed=min(done, total), total=max(total, 1))

    with progress:
        stats = engine.index(rebuild=rebuild, progress=on_progress)
    return stats


def cmd_index(args: argparse.Namespace) -> int:
    engine = _engine(args)
    stats = _run_index(engine, args.rebuild)
    s = engine.stats()
    table = Table(title=f"git-engram index ({stats.mode})", box=box.SIMPLE_HEAVY)
    table.add_column("metric")
    table.add_column("value", justify="right")
    for label, value in [
        ("files", stats.files),
        ("files re-parsed", stats.changed_files),
        ("AST symbols", stats.symbols),
        ("commits (new)", f"{stats.commits} ({stats.new_commits})"),
        ("documents", stats.documents),
        ("vectors embedded", stats.embedded),
        ("co-change edges file/symbol", f"{stats.file_edges}/{stats.symbol_edges}"),
        ("graveyard failures", stats.failures),
        ("invariants", stats.invariants),
        ("parsers", ", ".join(f"{k}:{v}" for k, v in sorted(stats.parsers.items())) or "-"),
        ("quantized vectors RAM", f"{s['memory_bytes']['vectors_quantized'] / 1024:.1f} KB"),
        ("float32 equivalent", f"{s['memory_bytes']['vectors_float32_equivalent'] / 1024:.1f} KB"),
    ]:
        table.add_row(label, str(value))
    console.print(table)
    timing = "  ".join(f"{k}={v:.0f}ms" for k, v in stats.timings_ms.items())
    console.print(f"[dim]{timing}[/dim]")
    return 0


# ------------------------------------------------------------------------------------- query
def cmd_query(args: argparse.Namespace) -> int:
    from ..render import render_search
    from ..retrieval.engine import MODES

    engine = _engine(args)
    _ensure_index(engine)
    mode = MODES[{"full": "git-engram", "bm25": "bm25", "dense": "dense-float32", "hybrid": "hybrid"}[args.mode]]
    resp = engine.search(args.question, top_k=args.top_k, file=args.file, mode=mode)
    if args.json:
        console.print_json(resp.model_dump_json())
        return 0
    if args.raw:
        print(render_search(resp))
        return 0
    table = Table(title=f"Hybrid retrieval: {args.question!r}", box=box.SIMPLE_HEAVY, show_lines=False)
    for col, justify in [
        ("#", "right"),
        ("kind", "left"),
        ("result", "left"),
        ("BM25", "right"),
        ("Hamming", "right"),
        ("cosine", "right"),
        ("graph", "right"),
        ("TD-RRF", "right"),
        ("recency", "right"),
        ("rerank", "right"),
        ("date", "left"),
    ]:
        table.add_column(col, justify=justify, overflow="fold")
    for i, h in enumerate(resp.hits, start=1):
        where = f"{h.path}::{h.symbol}" if h.symbol else (h.path or "")
        label = Text(h.title, style="bold")
        if where and h.kind != "commit":
            label.append(f"\n{where}", style="dim")
        bm25 = f"#{h.bm25_rank} {h.bm25_score:.1f}" if h.bm25_rank else "-"
        ham = str(h.hamming) if h.hamming is not None else "-"
        cos = f"#{h.vec_rank} {h.cosine:.3f}" if h.vec_rank else "-"
        graph = f"#{h.graph_rank} {h.graph_activation:.2f}" if h.graph_rank else "-"
        table.add_row(
            str(i),
            h.kind,
            label,
            bm25,
            ham,
            cos,
            graph,
            f"{h.fused:.4f}",
            f"x{h.recency:.2f}",
            f"{h.rerank_score:.3f}" if h.rerank_score is not None else "-",
            _date(h.ts),
        )
    console.print(table)
    if resp.coupling:
        ct = Table(title="Co-change coupling alerts", box=box.MINIMAL)
        for col in ("from", "coupled file", "W", "P(co-change)", "commits", "link"):
            ct.add_column(col)
        for c in resp.coupling:
            ct.add_row(
                c.source,
                c.target,
                f"{c.weight:.2f}",
                f"{c.confidence:.0%}",
                str(c.support),
                "[red]hidden[/red]" if c.hidden else "import",
            )
        console.print(ct)
    for w in resp.warnings:
        body = Text()
        body.append(f"{w.command} (exit {w.exit_code}, {w.occurrences}x, {_date(w.ts)})\n", style="bold")
        if w.notes:
            body.append(f"{w.notes}\n")
        body.append(w.signature[:600] + "\n", style="red")
        body.append(w.diff_excerpt[:600], style="dim")
        body.append(f"\nmatched: {w.match_reason}", style="italic")
        console.print(
            Panel(body, title=f"[red]NEGATIVE RAG - failed attempt #{w.failure_id}: do not repeat", border_style="red")
        )
    for inv in resp.invariants:
        console.print(
            Panel(
                inv.description,
                title=f"invariant {inv.rule_id} [{inv.severity}] scope {inv.matched}",
                border_style="yellow",
            )
        )
    for f in resp.security:
        console.print(f"[magenta]firewall[/magenta] [{f.severity}] {f.rule}: {f.detail[:120]}")
    t = resp.timings_ms
    console.print("[dim]latency: " + "  ".join(f"{k}={v:.2f}ms" for k, v in t.items()) + "[/dim]")
    return 0


# --------------------------------------------------------------------------------- preflight
def cmd_preflight(args: argparse.Namespace) -> int:
    from ..render import render_preflight

    engine = _engine(args)
    _ensure_index(engine)
    report = engine.preflight(args.file, args.symbol, args.intent)
    if args.json:
        console.print_json(report.model_dump_json())
    else:
        print(render_preflight(report))
    return 0


# -------------------------------------------------------------------------------------- exec
def cmd_exec(args: argparse.Namespace) -> int:
    argv = list(args.command)
    if argv and argv[0] == "--":
        argv = argv[1:]
    if not argv:
        err.print("usage: git-engram exec -- <command> [args...]")
        return 2
    engine = _engine(args)
    _ensure_index(engine)
    run, result = engine.exec(argv, echo=True, record_clean=args.record_clean, notes=args.notes or "")
    if result is not None:
        state = "already known" if result.duplicate else "recorded"
        targets = ", ".join(f"{p}::{s}" if s else p for p, s in result.targets[:6])
        err.print(
            f"[red]git-engram:[/red] exit {run.exit_code} -> failure #{result.failure_id} {state} in the "
            f"Graveyard ({result.occurrences}x). targets: {targets or '-'}"
        )
    elif run.exit_code != 0:
        err.print(
            f"[yellow]git-engram:[/yellow] exit {run.exit_code} on a clean working tree - nothing recorded "
            "(use --record-clean to record anyway)"
        )
    return run.exit_code


# ------------------------------------------------------------------------------- invariants
def cmd_invariant(args: argparse.Namespace) -> int:
    engine = _engine(args)
    if args.action == "add":
        if not (args.rule_id and args.scope and args.description):
            err.print("usage: git-engram invariant add RULE_ID --scope GLOBS --description TEXT [--severity S]")
            return 2
        _ensure_index(engine)
        inv = engine.record_invariant(args.rule_id, args.scope, args.description, args.severity, source="cli")
        console.print(f"[green]stored[/green] {inv['rule_id']} [{inv['severity']}] scope={inv['scope']}")
    elif args.action == "remove":
        ok = engine.invariants.remove(args.rule_id or "")
        console.print("[green]removed[/green]" if ok else "[yellow]not found[/yellow]")
        return 0 if ok else 1
    else:
        table = Table(title="OKF invariants (.engram/invariants.okf.json)", box=box.SIMPLE)
        for col in ("rule", "severity", "scope", "description"):
            table.add_column(col, overflow="fold")
        for inv in engine.invariants.all():
            table.add_row(inv["rule_id"], inv["severity"], inv["scope"], inv["description"])
        console.print(table)
    return 0


def cmd_graveyard(args: argparse.Namespace) -> int:
    engine = _engine(args)
    rows = engine.db.query("SELECT * FROM failures ORDER BY last_ts DESC LIMIT ?", (args.limit,))
    table = Table(title="Negative-RAG Graveyard", box=box.SIMPLE)
    for col in ("id", "date", "seen", "command", "targets", "signature"):
        table.add_column(col, overflow="fold")
    for r in rows:
        targets = ", ".join(f"{p}::{s}" if s else p for p, s in engine.graveyard.targets_of(r["id"])[:4])
        table.add_row(
            str(r["id"]),
            _date(r["ts"]),
            f"{r['occurrences']}x",
            r["command"],
            targets,
            (r["signature"].splitlines() or [""])[0][:100],
        )
    console.print(table)
    return 0


# ------------------------------------------------------------------------------- benchmark
def cmd_benchmark(args: argparse.Namespace) -> int:
    from ..evals.benchmark_runner import run_benchmark

    with err.status("Building the shopfront fixture, indexing and running the evaluation suite..."):
        payload = run_benchmark(
            repeats=args.repeats, scale=not args.no_scale, sizes=tuple(int(x) for x in args.sizes.split(","))
        )
    print(payload["markdown"])
    if args.out:
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        Path(args.out).write_text(payload["markdown"], encoding="utf-8")
        err.print(f"[green]wrote[/green] {args.out}")
    if args.json:
        Path(args.json).parent.mkdir(parents=True, exist_ok=True)
        Path(args.json).write_text(
            json.dumps({k: v for k, v in payload.items() if k != "markdown"}, indent=2, default=str), encoding="utf-8"
        )
        err.print(f"[green]wrote[/green] {args.json}")
    return 0


# --------------------------------------------------------------------------------- inspect
def cmd_inspect(args: argparse.Namespace) -> int:
    engine = _engine(args)
    _ensure_index(engine)
    s = engine.stats()
    mem = s["memory_bytes"]
    c = s["counts"]
    summary = Table.grid(padding=(0, 2))
    summary.add_column(style="bold")
    summary.add_column()
    for k, v in [
        ("repository", s["root"]),
        ("indexed HEAD", (s["head"] or "-")[:12]),
        ("files / symbols", f"{c['files']} / {c['symbols']}"),
        ("commits", c["commits"]),
        ("documents", ", ".join(f"{k}:{v}" for k, v in sorted(s["documents_by_kind"].items()))),
        ("embedder", f"{s['embedder']} {s['full_dim']}d -> MRL {s['mrl_dim']}d (1-bit + int8)"),
        (
            "quantized vectors",
            f"{mem['vectors_quantized'] / 1024:.1f} KB "
            f"(float32 would be {mem['vectors_float32_equivalent'] / 1024:.1f} KB)",
        ),
        ("FTS5 index", f"{mem['fts5_index'] / 1024:.1f} KB"),
        ("co-change graph", f"{c['cochange_edges']} edges"),
        ("database file", f"{mem['db_file'] / 1024:.1f} KB"),
    ]:
        summary.add_row(k, str(v))
    console.print(Panel(summary, title=f"git-engram {__version__}", border_style="cyan"))

    hot = Table(title="Hidden coupling hotspots (temporal co-change, no import link)", box=box.SIMPLE)
    for col in ("file A", "file B", "W", "P(B|A)", "co-commits", "last"):
        hot.add_column(col, overflow="fold")
    for r in engine.db.query(
        "SELECT * FROM cochange_edges WHERE level='file' AND static_link=0 AND src < dst ORDER BY weight DESC LIMIT ?",
        (args.limit,),
    ):
        hot.add_row(
            r["src"], r["dst"], f"{r['weight']:.2f}", f"{r['confidence']:.0%}", str(r["support"]), _date(r["last_ts"])
        )
    fragile = Table(title="Load-bearing hotspots (bugfix / security / revert commits per file)", box=box.SIMPLE)
    for col in ("file", "fixes", "security", "reverts", "last fix"):
        fragile.add_column(col)
    for r in engine.db.query(
        "SELECT f.path, SUM(c.intent='bugfix') AS fixes, SUM(c.intent='security') AS sec, "
        "SUM(c.intent='revert') AS rev, MAX(c.ts) AS last FROM commit_files f JOIN commits c ON c.hash=f.hash "
        "WHERE c.intent IN ('bugfix','security','revert') AND f.path IN (SELECT path FROM files) "
        "GROUP BY f.path ORDER BY (fixes + 2*sec + 2*rev) DESC, last DESC LIMIT ?",
        (args.limit,),
    ):
        fragile.add_row(r["path"], str(r["fixes"]), str(r["sec"]), str(r["rev"]), _date(r["last"]))
    g = s["graveyard"]
    grave = Table(
        title=f"Graveyard: {g['failures']} failures, {g['occurrences']} occurrences, {g['files']} files", box=box.SIMPLE
    )
    for col in ("id", "seen", "command", "signature"):
        grave.add_column(col, overflow="fold")
    for r in engine.db.query("SELECT * FROM failures ORDER BY occurrences DESC, last_ts DESC LIMIT ?", (args.limit,)):
        grave.add_row(str(r["id"]), f"{r['occurrences']}x", r["command"], (r["signature"].splitlines() or [""])[0][:90])
    console.print(Group(hot, fragile, grave))
    invs = engine.invariants.all()
    if invs:
        console.print(
            f"[yellow]{len(invs)} invariant(s):[/yellow] " + ", ".join(f"{i['rule_id']}[{i['severity']}]" for i in invs)
        )
    return 0


def cmd_mcp(args: argparse.Namespace) -> int:
    from ..mcp.server import run_stdio

    run_stdio(args.repo)
    return 0


# ----------------------------------------------------------------------------------- parser
def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="git-engram", description="Spatio-temporal, causal & negative RAG memory for AI coding agents."
    )
    parser.add_argument("--version", action="version", version=f"git-engram {__version__}")
    parser.add_argument("-C", "--repo", default=".", help="repository path (default: current directory)")
    sub = parser.add_subparsers(dest="cmd", required=True)

    p = sub.add_parser("index", help="index AST symbols, git history and the co-change graph")
    p.add_argument("path", nargs="?", help="repository path (overrides -C)")
    p.add_argument("--rebuild", action="store_true", help="drop derived data and rebuild from scratch")
    p.set_defaults(func=cmd_index)

    p = sub.add_parser("query", help="hybrid search with a full scoring breakdown")
    p.add_argument("question")
    p.add_argument("--file", help="file you are working on (seeds the co-change graph)")
    p.add_argument("-k", "--top-k", type=int, default=8)
    p.add_argument("--mode", choices=["full", "bm25", "dense", "hybrid"], default="full")
    p.add_argument("--json", action="store_true", help="machine-readable output")
    p.add_argument("--raw", action="store_true", help="print the exact context block an MCP client receives")
    p.set_defaults(func=cmd_query)

    p = sub.add_parser("preflight", help="what an agent must know before editing a file/symbol")
    p.add_argument("file")
    p.add_argument("-s", "--symbol")
    p.add_argument("-i", "--intent", help="plain-language description of the intended change")
    p.add_argument("--json", action="store_true")
    p.set_defaults(func=cmd_preflight)

    p = sub.add_parser("exec", help="run a test/build command; on failure record diff + error in the Graveyard")
    p.add_argument("--notes", help="what this attempt was trying to do")
    p.add_argument("--record-clean", action="store_true", help="record failures even with a clean working tree")
    p.add_argument("command", nargs=argparse.REMAINDER)
    p.set_defaults(func=cmd_exec)

    p = sub.add_parser("invariant", help="manage OKF invariants")
    p.add_argument("action", choices=["add", "list", "remove"])
    p.add_argument("rule_id", nargs="?")
    p.add_argument("--scope", help="comma-separated globs, optional '::symbol'")
    p.add_argument("--description")
    p.add_argument("--severity", choices=["info", "warning", "critical"], default="warning")
    p.set_defaults(func=cmd_invariant)

    p = sub.add_parser("graveyard", help="list recorded failed attempts")
    p.add_argument("--limit", type=int, default=20)
    p.set_defaults(func=cmd_graveyard)

    p = sub.add_parser("benchmark", help="run the built-in evaluation suite and print a Markdown report")
    p.add_argument("--repeats", type=int, default=20, help="latency repeats per query")
    p.add_argument("--no-scale", action="store_true", help="skip the 10k/100k vector scale test")
    p.add_argument("--sizes", default="10000,100000", help="vector counts for the scale test")
    p.add_argument("--out", help="write the Markdown report to this path")
    p.add_argument("--json", help="write raw results as JSON to this path")
    p.set_defaults(func=cmd_benchmark)

    p = sub.add_parser("inspect", help="coupling hotspots, load-bearing files, graveyard and memory footprint")
    p.add_argument("--limit", type=int, default=10)
    p.set_defaults(func=cmd_inspect)

    p = sub.add_parser("mcp", help="run the MCP server over stdio")
    p.set_defaults(func=cmd_mcp)
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    if getattr(args, "path", None):
        args.repo = args.path
    try:
        return int(args.func(args) or 0)
    except KeyboardInterrupt:
        return 130


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
