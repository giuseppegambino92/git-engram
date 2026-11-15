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
from ..i18n import SUPPORTED_LANGS, get_lang, resolve_lang, set_lang, t
from ..indexer.gitio import GitError

console = Console(stderr=False)
err = Console(stderr=True)

STAGES = ("history", "ast", "embeddings")


def _engine(args: argparse.Namespace):
    from ..engine import Engram

    try:
        return Engram(args.repo)
    except GitError as exc:
        err.print(f"[red]{t('cli.error')}[/red] {exc}")
        raise SystemExit(2) from exc


def _ensure_index(engine) -> None:
    if not engine.is_indexed:
        err.print(f"[yellow]{t('cli.no_index')}[/yellow]")
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
            tasks[stage] = progress.add_task(t(f"cli.stage.{stage}") if stage in STAGES else stage, total=max(total, 1))
        progress.update(tasks[stage], completed=min(done, total), total=max(total, 1))

    with progress:
        stats = engine.index(rebuild=rebuild, progress=on_progress)
    return stats


def cmd_index(args: argparse.Namespace) -> int:
    engine = _engine(args)
    stats = _run_index(engine, args.rebuild)
    s = engine.stats()
    table = Table(title=t("cli.index.title", mode=stats.mode), box=box.SIMPLE_HEAVY)
    table.add_column(t("cli.index.metric"))
    table.add_column(t("cli.index.value"), justify="right")
    for label, value in [
        (t("cli.index.files"), stats.files),
        (t("cli.index.files_reparsed"), stats.changed_files),
        (t("cli.index.symbols"), stats.symbols),
        (t("cli.index.commits"), f"{stats.commits} ({stats.new_commits})"),
        (t("cli.index.documents"), stats.documents),
        (t("cli.index.embedded"), stats.embedded),
        (t("cli.index.edges"), f"{stats.file_edges}/{stats.symbol_edges}"),
        (t("cli.index.failures"), stats.failures),
        (t("cli.index.invariants"), stats.invariants),
        (t("cli.index.parsers"), ", ".join(f"{k}:{v}" for k, v in sorted(stats.parsers.items())) or "-"),
        (t("cli.index.ram_quantized"), f"{s['memory_bytes']['vectors_quantized'] / 1024:.1f} KB"),
        (t("cli.index.ram_float32"), f"{s['memory_bytes']['vectors_float32_equivalent'] / 1024:.1f} KB"),
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
    table = Table(title=t("cli.query.title", question=repr(args.question)), box=box.SIMPLE_HEAVY, show_lines=False)
    for col, justify in [
        ("#", "right"),
        (t("col.kind"), "left"),
        (t("col.result"), "left"),
        ("BM25", "right"),
        ("Hamming", "right"),
        (t("col.cosine"), "right"),
        (t("col.graph"), "right"),
        ("TD-RRF", "right"),
        (t("col.recency"), "right"),
        ("rerank", "right"),
        (t("col.date"), "left"),
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
        ct = Table(title=t("cli.query.coupling_title"), box=box.MINIMAL)
        for col in (
            t("cli.query.from"),
            t("col.coupled_file"),
            "W",
            "P(co-change)",
            t("cli.query.commits"),
            t("col.link"),
        ):
            ct.add_column(col)
        for c in resp.coupling:
            ct.add_row(
                c.source,
                c.target,
                f"{c.weight:.2f}",
                f"{c.confidence:.0%}",
                str(c.support),
                f"[red]{t('link.hidden')}[/red]" if c.hidden else t("link.import"),
            )
        console.print(ct)
    for w in resp.warnings:
        body = Text()
        body.append(f"{w.command} (exit {w.exit_code}, {w.occurrences}x, {_date(w.ts)})\n", style="bold")
        if w.notes:
            body.append(f"{w.notes}\n")
        body.append(w.signature[:600] + "\n", style="red")
        body.append(w.diff_excerpt[:600], style="dim")
        body.append("\n" + t("cli.query.matched", reason=w.match_reason), style="italic")
        title = t("cli.query.warning_title", failure_id=w.failure_id)
        console.print(Panel(body, title=f"[red]{title}", border_style="red"))
    for inv in resp.invariants:
        console.print(
            Panel(
                inv.description,
                title=t("cli.query.invariant_title", rule_id=inv.rule_id, severity=inv.severity, scope=inv.matched),
                border_style="yellow",
            )
        )
    for f in resp.security:
        console.print(f"[magenta]firewall[/magenta] [{f.severity}] {f.rule}: {f.detail[:120]}")
    if resp.query_class:
        w = "  ".join(f"{k}={v:.1f}" for k, v in resp.fusion_weights.items())
        console.print(
            f"[cyan]{t('cli.query.class')}[/cyan] {resp.query_class}   [cyan]{t('cli.query.weights')}[/cyan] {w}"
        )
    latency = "  ".join(f"{k}={v:.2f}ms" for k, v in resp.timings_ms.items())
    console.print(f"[dim]{t('cli.query.latency')} {latency}[/dim]")
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
        err.print(t("cli.exec.usage"))
        return 2
    engine = _engine(args)
    _ensure_index(engine)
    run, result = engine.exec(argv, echo=True, record_clean=args.record_clean, notes=args.notes or "")
    if result is not None:
        state = t("cli.exec.known") if result.duplicate else t("cli.exec.recorded")
        targets = ", ".join(f"{p}::{s}" if s else p for p, s in result.targets[:6])
        message = t(
            "cli.exec.failure",
            exit_code=run.exit_code,
            failure_id=result.failure_id,
            state=state,
            occurrences=result.occurrences,
            targets=targets or "-",
        )
        err.print(f"[red]git-engram:[/red] {message}")
    elif run.exit_code != 0:
        err.print(f"[yellow]git-engram:[/yellow] {t('cli.exec.clean', exit_code=run.exit_code)}")
    return run.exit_code


# ------------------------------------------------------------------------------- invariants
def cmd_invariant(args: argparse.Namespace) -> int:
    engine = _engine(args)
    if args.action == "add":
        if not (args.rule_id and args.scope and args.description):
            err.print(t("cli.invariant.usage"))
            return 2
        _ensure_index(engine)
        inv = engine.record_invariant(args.rule_id, args.scope, args.description, args.severity, source="cli")
        console.print(
            f"[green]{t('cli.invariant.stored')}[/green] {inv['rule_id']} [{inv['severity']}] scope={inv['scope']}"
        )
    elif args.action == "remove":
        ok = engine.invariants.remove(args.rule_id or "")
        console.print(
            f"[green]{t('cli.invariant.removed')}[/green]" if ok else f"[yellow]{t('cli.invariant.not_found')}[/yellow]"
        )
        return 0 if ok else 1
    else:
        table = Table(title=t("cli.invariant.title"), box=box.SIMPLE)
        for col in (t("col.rule"), t("col.severity"), t("col.scope"), t("col.description")):
            table.add_column(col, overflow="fold")
        for inv in engine.invariants.all():
            table.add_row(inv["rule_id"], inv["severity"], inv["scope"], inv["description"])
        console.print(table)
    return 0


def cmd_graveyard(args: argparse.Namespace) -> int:
    engine = _engine(args)
    rows = engine.db.query("SELECT * FROM failures ORDER BY last_ts DESC LIMIT ?", (args.limit,))
    table = Table(title=t("cli.graveyard.title"), box=box.SIMPLE)
    for col in ("id", t("col.date"), t("col.seen"), t("col.command"), t("col.targets"), t("col.signature")):
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

    with err.status(t("cli.benchmark.status")):
        payload = run_benchmark(
            repeats=args.repeats, scale=not args.no_scale, sizes=tuple(int(x) for x in args.sizes.split(","))
        )
    print(payload["markdown"])
    if args.out:
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        Path(args.out).write_text(payload["markdown"], encoding="utf-8")
        err.print(f"[green]{t('cli.wrote')}[/green] {args.out}")
    if args.json:
        Path(args.json).parent.mkdir(parents=True, exist_ok=True)
        Path(args.json).write_text(
            json.dumps({k: v for k, v in payload.items() if k != "markdown"}, indent=2, default=str), encoding="utf-8"
        )
        err.print(f"[green]{t('cli.wrote')}[/green] {args.json}")
    return 0


# --------------------------------------------------------------------------------- inspect
def cmd_inspect(args: argparse.Namespace) -> int:
    if getattr(args, "tui", False):
        args.query = None
        return cmd_tui(args)
    engine = _engine(args)
    _ensure_index(engine)
    s = engine.stats()
    mem = s["memory_bytes"]
    c = s["counts"]
    summary = Table.grid(padding=(0, 2))
    summary.add_column(style="bold")
    summary.add_column()
    for k, v in [
        (t("label.repository"), s["root"]),
        (t("label.indexed_head"), (s["head"] or "-")[:12]),
        (t("cli.inspect.files_symbols"), f"{c['files']} / {c['symbols']}"),
        (t("label.commits"), c["commits"]),
        (t("label.documents"), ", ".join(f"{k}:{v}" for k, v in sorted(s["documents_by_kind"].items()))),
        (t("label.embedder"), f"{s['embedder']} {s['full_dim']}d -> MRL {s['mrl_dim']}d (1-bit + int8)"),
        (
            t("label.quantized_vectors"),
            t(
                "cli.inspect.quantized",
                quantized=f"{mem['vectors_quantized'] / 1024:.1f}",
                float32=f"{mem['vectors_float32_equivalent'] / 1024:.1f}",
            ),
        ),
        (t("label.fts5"), f"{mem['fts5_index'] / 1024:.1f} KB"),
        (t("cli.inspect.graph"), t("cli.inspect.graph_edges", edges=c["cochange_edges"])),
        (t("cli.inspect.db_file"), f"{mem['db_file'] / 1024:.1f} KB"),
    ]:
        summary.add_row(k, str(v))
    console.print(Panel(summary, title=f"git-engram {__version__}", border_style="cyan"))

    hot = Table(title=t("cli.inspect.hidden_title"), box=box.SIMPLE)
    for col in ("file A", "file B", "W", "P(B|A)", t("cli.inspect.co_commits"), t("cli.inspect.last")):
        hot.add_column(col, overflow="fold")
    for r in engine.db.query(
        "SELECT * FROM cochange_edges WHERE level='file' AND static_link=0 AND src < dst ORDER BY weight DESC LIMIT ?",
        (args.limit,),
    ):
        hot.add_row(
            r["src"], r["dst"], f"{r['weight']:.2f}", f"{r['confidence']:.0%}", str(r["support"]), _date(r["last_ts"])
        )
    fragile = Table(title=t("cli.inspect.fragile_title"), box=box.SIMPLE)
    for col in (t("col.file"), t("col.fixes"), t("col.security"), t("col.reverts"), t("cli.inspect.last_fix")):
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
        title=t("cli.inspect.graveyard_title", failures=g["failures"], occurrences=g["occurrences"], files=g["files"]),
        box=box.SIMPLE,
    )
    for col in ("id", t("col.seen"), t("col.command"), t("col.signature")):
        grave.add_column(col, overflow="fold")
    for r in engine.db.query("SELECT * FROM failures ORDER BY occurrences DESC, last_ts DESC LIMIT ?", (args.limit,)):
        grave.add_row(str(r["id"]), f"{r['occurrences']}x", r["command"], (r["signature"].splitlines() or [""])[0][:90])
    console.print(Group(hot, fragile, grave))
    invs = engine.invariants.all()
    if invs:
        console.print(
            f"[yellow]{t('cli.inspect.invariants', count=len(invs))}[/yellow] "
            + ", ".join(f"{i['rule_id']}[{i['severity']}]" for i in invs)
        )
    return 0


def cmd_tui(args: argparse.Namespace) -> int:
    try:
        from ..tui.app import run_tui
    except ImportError:
        err.print(f"[red]{t('cli.tui_missing')}[/red] pip install 'git-engram[tui]'")
        return 2
    engine = _engine(args)
    _ensure_index(engine)
    run_tui(engine, args.query or "", lang=get_lang())
    return 0


def cmd_mcp(args: argparse.Namespace) -> int:
    from ..mcp.server import run_stdio

    run_stdio(args.repo)
    return 0


# ----------------------------------------------------------------------------------- parser
class _HelpFormatter(argparse.HelpFormatter):
    """Localised ``usage:`` prefix; argparse's own error messages stay in English."""

    def add_usage(self, usage, actions, groups, prefix=None):
        super().add_usage(usage, actions, groups, t("cli.usage") if prefix is None else prefix)


def _localize(parser: argparse.ArgumentParser) -> argparse.ArgumentParser:
    """Translated section titles and ``-h`` help (the parser is built after the language is known)."""
    parser._positionals.title = t("cli.positionals")
    parser._optionals.title = t("cli.options")
    parser.add_argument("-h", "--help", action="help", default=argparse.SUPPRESS, help=t("cli.help.help"))
    return parser


def _lang_option(parser: argparse.ArgumentParser, default: str | None) -> None:
    parser.add_argument("--lang", choices=SUPPORTED_LANGS, default=default, help=t("cli.help.lang"))


def build_parser() -> argparse.ArgumentParser:
    parser = _localize(
        argparse.ArgumentParser(
            prog="git-engram", description=t("cli.description"), formatter_class=_HelpFormatter, add_help=False
        )
    )
    parser.add_argument("--version", action="version", version=f"git-engram {__version__}", help=t("cli.help.version"))
    parser.add_argument("-C", "--repo", default=".", help=t("cli.help.repo"))
    _lang_option(parser, default=None)
    sub = parser.add_subparsers(dest="cmd", required=True)

    def command(name: str, help_key: str) -> argparse.ArgumentParser:
        p = _localize(sub.add_parser(name, help=t(help_key), formatter_class=_HelpFormatter, add_help=False))
        _lang_option(p, default=argparse.SUPPRESS)  # also accepted after the subcommand; never overrides the global
        return p

    p = command("index", "cli.help.index")
    p.add_argument("path", nargs="?", help=t("cli.help.index.path"))
    p.add_argument("--rebuild", action="store_true", help=t("cli.help.index.rebuild"))
    p.set_defaults(func=cmd_index)

    p = command("query", "cli.help.query")
    p.add_argument("question")
    p.add_argument("--file", help=t("cli.help.query.file"))
    p.add_argument("-k", "--top-k", type=int, default=8)
    p.add_argument("--mode", choices=["full", "bm25", "dense", "hybrid"], default="full")
    p.add_argument("--json", action="store_true", help=t("cli.help.query.json"))
    p.add_argument("--raw", action="store_true", help=t("cli.help.query.raw"))
    p.set_defaults(func=cmd_query)

    p = command("preflight", "cli.help.preflight")
    p.add_argument("file")
    p.add_argument("-s", "--symbol")
    p.add_argument("-i", "--intent", help=t("cli.help.preflight.intent"))
    p.add_argument("--json", action="store_true")
    p.set_defaults(func=cmd_preflight)

    p = command("exec", "cli.help.exec")
    p.add_argument("--notes", help=t("cli.help.exec.notes"))
    p.add_argument("--record-clean", action="store_true", help=t("cli.help.exec.record_clean"))
    p.add_argument("command", nargs=argparse.REMAINDER)
    p.set_defaults(func=cmd_exec)

    p = command("invariant", "cli.help.invariant")
    p.add_argument("action", choices=["add", "list", "remove"])
    p.add_argument("rule_id", nargs="?")
    p.add_argument("--scope", help=t("cli.help.invariant.scope"))
    p.add_argument("--description")
    p.add_argument("--severity", choices=["info", "warning", "critical"], default="warning")
    p.set_defaults(func=cmd_invariant)

    p = command("graveyard", "cli.help.graveyard")
    p.add_argument("--limit", type=int, default=20)
    p.set_defaults(func=cmd_graveyard)

    p = command("benchmark", "cli.help.benchmark")
    p.add_argument("--repeats", type=int, default=20, help=t("cli.help.benchmark.repeats"))
    p.add_argument("--no-scale", action="store_true", help=t("cli.help.benchmark.no_scale"))
    p.add_argument("--sizes", default="10000,100000", help=t("cli.help.benchmark.sizes"))
    p.add_argument("--out", help=t("cli.help.benchmark.out"))
    p.add_argument("--json", help=t("cli.help.benchmark.json"))
    p.set_defaults(func=cmd_benchmark)

    p = command("inspect", "cli.help.inspect")
    p.add_argument("--limit", type=int, default=10)
    p.add_argument("--tui", action="store_true", help=t("cli.help.inspect.tui"))
    p.set_defaults(func=cmd_inspect)

    p = command("tui", "cli.help.tui")
    p.add_argument("query", nargs="?", help=t("cli.help.tui.query"))
    p.set_defaults(func=cmd_tui)

    p = command("mcp", "cli.help.mcp")
    p.set_defaults(func=cmd_mcp)
    return parser


def _requested_lang(argv: list[str]) -> str | None:
    """``--lang`` value from argv, read before the (localised) parser exists. Validation is left to the parser."""
    pre = argparse.ArgumentParser(add_help=False)
    pre.add_argument("--lang")
    known, _ = pre.parse_known_args(argv)
    return known.lang


def main(argv: list[str] | None = None) -> int:
    argv = sys.argv[1:] if argv is None else list(argv)
    set_lang(resolve_lang(_requested_lang(argv)))  # help texts are translated too
    parser = build_parser()
    args = parser.parse_args(argv)
    set_lang(resolve_lang(getattr(args, "lang", None)))
    if getattr(args, "path", None):
        args.repo = args.path
    try:
        return int(args.func(args) or 0)
    except KeyboardInterrupt:
        return 130


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
