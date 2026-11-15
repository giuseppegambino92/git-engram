"""Regenerate the README screenshots from real runs, in every supported language.

    python scripts/make_screenshots.py                 # writes docs/images/<lang>/*.svg
    python scripts/make_screenshots.py --langs it      # one language only
    python scripts/make_screenshots.py --workdir /tmp/engram-shots   # keep (and reuse) the indexed repositories

Nothing is hand-written: the dashboard screenshots drive the real Textual app headlessly (``App.run_test`` +
``App.save_screenshot``) over the benchmark repository (``git_engram.evals.fixture``: built, indexed and populated with
graveyard failures and invariants exactly as the test suite does), and the CLI screenshot records a real
``git-engram query`` over a clone of this repository's own git history (dogfooding) with Rich's ``export_svg``.
Needs the dev extras (``pip install -e ".[dev]"``) and git; it never touches the network.
"""

from __future__ import annotations

import argparse
import asyncio
import io
import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

from rich.console import Console

from git_engram.cli import main as cli
from git_engram.engine import Engram
from git_engram.evals.fixture import build_shopfront, populate_memory
from git_engram.i18n import DEFAULT_LANG, ENV_VAR, SUPPORTED_LANGS
from git_engram.tui.app import EngramApp

REPO_ROOT = Path(__file__).resolve().parents[1]
TUI_SIZE = (150, 42)  # same geometry as the original screenshots
SEARCH_QUERY = "use a fresh idempotency key for each payment retry"
COCHANGE_FILE = "backend/api/schemas.py"
CLI_QUERY = "why does the retriever defer commits instead of dropping them"
CLI_FILE = "src/git_engram/retrieval/engine.py"
CLI_WIDTH = 150


def shopfront_engine(workdir: Path) -> Engram:
    """The benchmark repository, indexed, with its graveyard and invariants (reused if already in ``workdir``)."""
    root = workdir / "shopfront"  # the directory name is what the dashboard header shows
    if not (root / ".engram").exists():
        shutil.rmtree(root, ignore_errors=True)
        info = build_shopfront(root)
        engine = Engram(root)
        engine.index()
        populate_memory(engine, info)
        return engine
    return Engram(root)


def self_engine(workdir: Path) -> Engram:
    """A clone of this repository's committed history (working-tree changes are not part of it), indexed."""
    root = workdir / "git-engram"
    if not (root / ".git").exists():
        subprocess.run(["git", "clone", "--quiet", str(REPO_ROOT), str(root)], check=True)
    engine = Engram(root)
    engine.index()  # incremental when reused
    return engine


async def tui_screenshots(engine: Engram, lang: str, out: Path) -> list[Path]:
    app = EngramApp(engine, lang=lang)
    written: list[Path] = []
    async with app.run_test(size=TUI_SIZE) as pilot:
        # Search tab: type the query like a user would (search-as-you-type), then submit.
        await pilot.click("#search-input")
        await pilot.press(*("space" if ch == " " else ch for ch in SEARCH_QUERY))
        await pilot.press("enter")
        await pilot.pause()
        written.append(Path(app.save_screenshot("tui-search.svg", str(out))))

        # Co-change tab: select the file whose hidden couplings the README discusses.
        await pilot.press("f2")
        hotspots = app.query_one("#hotspots")
        hotspots.move_cursor(row=hotspots.get_row_index(COCHANGE_FILE))
        await pilot.pause()
        written.append(Path(app.save_screenshot("tui-cochange.svg", str(out))))

        # Graveyard tab: select the failed idempotency-key attempt.
        await pilot.press("f3")
        row = engine.db.query_one("SELECT id FROM failures WHERE notes LIKE '%idempotency key%' ORDER BY id LIMIT 1")
        failures = app.query_one("#failures")
        failures.move_cursor(row=failures.get_row_index(str(row["id"])))
        await pilot.pause()
        written.append(Path(app.save_screenshot("tui-graveyard.svg", str(out))))
        await pilot.press("ctrl+q")
    return written


def cli_screenshot(engine: Engram, lang: str, out: Path) -> Path:
    """Run the real CLI entry point with its consoles swapped for recording ones, then export the recording."""
    lang_flag = [] if lang == DEFAULT_LANG else ["--lang", lang]  # English is what users get without the option
    argv = [*lang_flag, "query", CLI_QUERY, "--file", CLI_FILE, "-k", "6"]
    recorder = Console(file=io.StringIO(), record=True, width=CLI_WIDTH, force_terminal=True, color_system="truecolor")
    saved = cli.console, cli.err
    cli.console = cli.err = recorder
    try:
        code = cli.main(["-C", str(engine.root), *argv])
    finally:
        cli.console, cli.err = saved
    if code != 0:
        raise SystemExit(f"git-engram {' '.join(argv)} exited with {code}")
    path = out / "cli-query.svg"
    title = " ".join(["git-engram", *lang_flag, "query", "..."])
    path.write_text(recorder.export_svg(title=title), encoding="utf-8")
    return path


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--out", type=Path, default=REPO_ROOT / "docs" / "images", help="output root")
    parser.add_argument("--langs", default=",".join(SUPPORTED_LANGS), help="comma-separated languages")
    parser.add_argument("--workdir", type=Path, help="where to build the repositories (default: a temp dir)")
    args = parser.parse_args(argv)

    langs = [lang.strip() for lang in args.langs.split(",") if lang.strip()]
    unknown = sorted(set(langs) - set(SUPPORTED_LANGS))
    if unknown:
        parser.error(f"unsupported language(s): {', '.join(unknown)}")
    os.environ.pop(ENV_VAR, None)  # each screenshot's language comes from its own run, never from the caller's shell
    workdir = args.workdir or Path(tempfile.mkdtemp(prefix="engram-shots-"))
    workdir.mkdir(parents=True, exist_ok=True)

    shop = shopfront_engine(workdir)
    own = self_engine(workdir)
    try:
        for lang in langs:
            out = args.out / lang
            out.mkdir(parents=True, exist_ok=True)
            for path in [*asyncio.run(tui_screenshots(shop, lang, out)), cli_screenshot(own, lang, out)]:
                print(path)
    finally:
        shop.close()
        own.close()
        if args.workdir is None:
            shutil.rmtree(workdir, ignore_errors=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
