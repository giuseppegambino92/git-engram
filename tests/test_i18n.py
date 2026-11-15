"""Localisation: catalog parity, language selection, and Italian vs. English output of the real CLI and TUI."""

from __future__ import annotations

import asyncio
import html
import json
import re

import pytest

from git_engram import i18n
from git_engram.cli import main as cli
from git_engram.i18n import MESSAGES, SUPPORTED_LANGS, placeholders, resolve_lang, set_lang, t, translate


def screen_text(app) -> str:
    """Visible text of the current screen (parsed from Textual's SVG export)."""
    svg = app.export_screenshot()
    return html.unescape(" ".join(re.findall(r">([^<>]+)</text>", svg))).replace("\xa0", " ")


@pytest.fixture(autouse=True)
def _restore_language(monkeypatch):
    monkeypatch.delenv(i18n.ENV_VAR, raising=False)
    yield
    set_lang(i18n.DEFAULT_LANG)


def test_every_key_exists_in_every_language_with_the_same_placeholders():
    english = MESSAGES["en"]
    assert set(MESSAGES) == set(SUPPORTED_LANGS)
    for lang in SUPPORTED_LANGS:
        catalog = MESSAGES[lang]
        assert set(catalog) == set(english), (
            f"{lang}: missing {set(english) - set(catalog)}, extra {set(catalog) - set(english)}"
        )
        for key, template in catalog.items():
            assert template.strip(), f"{lang}:{key} is empty"
            assert placeholders(template) == placeholders(english[key]), f"{lang}:{key} placeholders differ"


def test_italian_catalog_is_actually_translated():
    same = [k for k in MESSAGES["en"] if MESSAGES["en"][k] == MESSAGES["it"][k]]
    # Technical terms (Graveyard, embedder, commit, ...) legitimately stay identical; most entries must not.
    assert len(same) < len(MESSAGES["en"]) * 0.2, same


def test_language_resolution(monkeypatch):
    assert resolve_lang() == "en"
    monkeypatch.setenv(i18n.ENV_VAR, "it_IT.UTF-8")
    assert resolve_lang() == "it"
    assert resolve_lang("en") == "en"  # the CLI option wins over the environment
    monkeypatch.setenv(i18n.ENV_VAR, "fr")
    assert resolve_lang() == "en"  # unsupported values fall back to English
    assert set_lang("IT") == "it" and t("cli.error") == "errore:"
    assert translate("en", "tui.status.failed", count=2) == "2 failed attempt(s)"
    assert translate("xx", "cli.error") == "error:"
    with pytest.raises(KeyError):
        t("no.such.key")


def _graveyard_output(capsys, shopfront, *flags: str) -> str:
    assert cli.main(["-C", str(shopfront.info.root), *flags, "graveyard"]) == 0
    return capsys.readouterr().out


def test_cli_graveyard_english_by_default_and_italian_on_request(capsys, monkeypatch, shopfront):
    english = _graveyard_output(capsys, shopfront)
    assert "Negative-RAG Graveyard" in english and "command" in english and "comando" not in english

    italian = _graveyard_output(capsys, shopfront, "--lang", "it")
    assert "Graveyard del Negative RAG" in italian and "comando" in italian and "firma" in italian

    monkeypatch.setenv(i18n.ENV_VAR, "it")
    assert "Graveyard del Negative RAG" in _graveyard_output(capsys, shopfront)
    assert "Negative-RAG Graveyard" in _graveyard_output(capsys, shopfront, "--lang", "en")  # option beats env
    # Repository data is never translated: the same failures (here: their dates) appear in both languages.
    assert "2025-12-09" in english and "2025-12-09" in italian


def test_cli_help_is_translated(capsys):
    with pytest.raises(SystemExit):
        cli.main(["--lang", "it", "--help"])
    out = capsys.readouterr().out
    assert "uso: git-engram" in out and "opzioni" in out and "dashboard interattiva" in out
    with pytest.raises(SystemExit):
        cli.main(["--help"])
    out = capsys.readouterr().out
    assert "usage: git-engram" in out and "interactive dashboard" in out


def test_machine_readable_output_is_identical_in_every_language(capsys, shopfront):
    def run(output_flag: str, *lang_flags: str) -> str:
        argv = ["-C", str(shopfront.info.root), *lang_flags, "query", "ERR_CART_STALE", "-k", "3", output_flag]
        assert cli.main(argv) == 0
        return capsys.readouterr().out

    def stable(payload: str) -> dict:
        data = json.loads(payload)
        data.pop("timings_ms")  # wall-clock latency differs between any two runs
        return data

    assert stable(run("--json")) == stable(run("--json", "--lang", "it"))

    def strip_latency(block: str) -> str:
        return re.sub(r'latency_ms="[^"]*"', "", block)

    raw_en, raw_it = run("--raw"), run("--raw", "--lang", "it")
    assert strip_latency(raw_en) == strip_latency(raw_it) and "<engram_search" in raw_it


def test_tui_renders_italian_and_defaults_to_english(shopfront):
    pytest.importorskip("textual")
    from git_engram.tui.app import EngramApp

    async def render(lang: str | None) -> tuple[str, str]:
        app = EngramApp(shopfront.engine, initial_query="ERR_CART_STALE", lang=lang)
        async with app.run_test(size=(160, 48)) as pilot:
            await pilot.pause()
            search = screen_text(app)
            await pilot.press("f4")
            await pilot.pause()
            overview = screen_text(app)
            await pilot.press("ctrl+q")
        return search, overview

    search, overview = asyncio.run(render(None))
    assert "Search" in search and "Overview" in search and "weights" in search
    assert "Quit" in search  # footer
    assert "load-bearing hotspots" in overview and "OKF invariants" in overview

    search, overview = asyncio.run(render("it"))
    assert "Ricerca" in search and "Panoramica" in search and "pesi" in search
    assert "Esci" in search and "Co-change graph" not in search and "Quit" not in search
    assert "hotspot portanti" in overview and "invarianti OKF" in overview and "HEAD indicizzato" in overview

    set_lang("it")  # the process-wide language (set by --lang / GIT_ENGRAM_LANG) is the default for the app
    search, _ = asyncio.run(render(None))
    assert "Ricerca" in search
