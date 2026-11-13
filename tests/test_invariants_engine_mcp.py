import asyncio
import json
import sys
from pathlib import Path

import pytest

from git_engram.engine import Engram
from git_engram.invariants import compile_scope, glob_to_regex
from git_engram.render import render_preflight


@pytest.mark.parametrize(
    "pattern,path,ok",
    [
        ("backend/payments/**", "backend/payments/retry.py", True),
        ("backend/payments/**", "backend/payments/sub/x.py", True),
        ("backend/payments/**", "backend/paymentsx/x.py", False),
        ("*.sql", "backend/db/schema.sql", True),
        ("backend/*.py", "backend/a/b.py", False),
        ("src/**/test_*.py", "src/test_a.py", True),
    ],
)
def test_glob_semantics(pattern, path, ok):
    assert bool(glob_to_regex(pattern).match(path)) is ok


def test_symbol_scopes():
    (sc,) = compile_scope("backend/api/schemas.py::Order*")
    assert sc.symbol_re.match("OrderSchema") and not sc.symbol_re.match("UserSchema")


def test_invariant_roundtrip_through_okf_file(mini_repo):
    mini_repo.write("a/x.py", "def f():\n    return 1\n")
    mini_repo.commit("feat: x")
    with Engram(mini_repo.root) as e:
        e.index()
        e.record_invariant("NO-NET", "a/**", "Module a must not do network I/O", "critical")
        data = json.loads((mini_repo.root / ".engram/invariants.okf.json").read_text())
        assert data["format"] == "okf/1" and data["invariants"][0]["scope"] == ["a/**"]
        assert [h.rule_id for h in e.invariants.match("a/x.py")] == ["NO-NET"]
        assert e.invariants.match("b/y.py") == []
        with pytest.raises(ValueError):
            e.record_invariant("bad id!", "a/**", "x")
        with pytest.raises(ValueError):
            e.record_invariant("OK", "a/**", "x", severity="fatal")
    with Engram(mini_repo.root) as e2:  # reloaded from the committed OKF file
        assert [i["rule_id"] for i in e2.invariants.all()] == ["NO-NET"]
        assert any(h.doc_key == "inv:NO-NET" for h in e2.search("network I/O module").hits)


def test_preflight_report_contents(shopfront):
    report = shopfront.engine.preflight("backend/api/schemas.py", "OrderSchema", "add a discount field")
    hidden = {c.target for c in report.coupling if c.hidden}
    assert {"mobile/src/parsers/orderParser.ts", "web/src/api/orderTypes.ts"} <= hidden
    assert [i.rule_id for i in report.invariants] == ["INV-API-002"]
    assert report.symbols[0].qualname == "OrderSchema" and report.symbols[0].history
    text = render_preflight(report)
    assert "<co_change_coupling>" in text and "[hidden: no import link]" in text and "<invariants" in text

    report = shopfront.engine.preflight(
        str(shopfront.info.root / "backend/auth/token.py"), "is_token_expired", "remove the clock skew grace period"
    )
    assert report.file_path == "backend/auth/token.py"
    assert any("bugfix" in lb and "clock skew" in lb for lb in report.load_bearing)
    assert report.warnings and report.warnings[0].failure_id == shopfront.failure_ids["F1-clock-skew"]
    assert "<anti_context_do_not_repeat>" in render_preflight(report)


def test_preflight_latency(shopfront):
    e = shopfront.engine
    e.preflight("backend/payments/retry.py", "charge_with_retry", "change retries")
    report = e.preflight("backend/payments/retry.py", "charge_with_retry", "change retries")
    assert report.latency_ms < 50.0  # generous CI bound; typical warm P50 is ~1.5 ms


def test_preflight_unknown_file(shopfront):
    report = shopfront.engine.preflight("does/not/exist.py")
    assert not report.indexed and "not in the index" in render_preflight(report)


def test_incremental_index_add_modify_delete_rename(mini_repo):
    mini_repo.write("svc/a.py", "def alpha():\n    return 1\n")
    mini_repo.write("svc/b.py", "def beta():\n    return 2\n")
    mini_repo.commit("feat: a and b")
    with Engram(mini_repo.root) as e:
        assert e.index().mode == "full"
        mini_repo.write("svc/a.py", "def alpha():\n    return 10\n\n\ndef gamma():\n    return 3\n")
        (mini_repo.root / "svc/b.py").unlink()
        mini_repo.commit("fix: alpha off by nine; drop b")
        stats = e.index()
        assert stats.mode == "incremental" and stats.new_commits == 1 and stats.changed_files == 1
        assert e.db.query_one("SELECT COUNT(*) AS n FROM symbols WHERE path='svc/b.py'")["n"] == 0
        assert e.retriever.by_key.get("sym:svc/a.py::gamma") is not None
        mini_repo.git("mv", "svc/a.py", "svc/core.py")
        mini_repo.commit("refactor: rename a -> core")
        e.index()
        rows = e.db.query("SELECT DISTINCT path FROM symbol_commits WHERE qualname='alpha'")
        assert [r["path"] for r in rows] == ["svc/core.py"]  # history rewritten onto the new path
        hist = e.preflight("svc/core.py", "alpha").symbols[0].history
        assert {h.subject for h in hist} >= {"feat: a and b", "fix: alpha off by nine; drop b"}
        assert e.index().new_commits == 0


def test_record_failure_is_searchable_immediately(mini_repo):
    mini_repo.write("m.py", "def pay():\n    return charge()\n")
    mini_repo.commit("feat: pay")
    with Engram(mini_repo.root) as e:
        e.index()
        diff = (
            "diff --git a/m.py b/m.py\n--- a/m.py\n+++ b/m.py\n@@ -2 +2 @@\n"
            "-    return charge()\n+    return charge_async()\n"
        )
        res = e.record_failure(
            file_path="m.py",
            attempted_diff=diff,
            error_trace="RuntimeError: no running loop",
            notes="switched pay() to the async gateway client",
        )
        assert res.targets == [("m.py", "pay")]
        resp = e.search("async gateway client no running loop")
        assert res.failure_id in {w.failure_id for w in resp.warnings}


def _call(server, name, args):
    from mcp import Client

    async def go():
        async with Client(server) as client:
            tools = await client.list_tools()
            result = await client.call_tool(name, args)
            return [t.name for t in tools.tools], result

    return asyncio.run(go())


def test_mcp_tools_in_process(shopfront):
    from git_engram.mcp.server import build_server

    server, holder = build_server(shopfront.info.root, eager=False)
    holder._engine, holder._started = shopfront.engine, True
    holder._ready.set()
    names, result = _call(
        server,
        "engram_preflight_check",
        {
            "file_path": "backend/utils/money.py",
            "symbol_name": "tax_cents",
            "intended_change_query": "use ROUND_HALF_UP",
        },
    )
    assert names == [
        "engram_preflight_check",
        "engram_search_causal_history",
        "engram_record_failure",
        "engram_record_invariant",
    ]
    text = result.content[0].text
    assert "<engram_preflight" in text and "INV-MONEY-003" in text and "<anti_context_do_not_repeat>" in text
    _, result = _call(server, "engram_search_causal_history", {"query": "INC-2291", "top_k": 3})
    assert "clock skew" in result.content[0].text


def test_mcp_stdio_subprocess(mini_repo):
    from mcp import Client
    from mcp.client.stdio import StdioServerParameters

    mini_repo.write("svc/pay.py", "def pay(order):\n    return order.total\n")
    mini_repo.commit("feat: pay")
    exe = str(Path(sys.executable).with_name("git-engram"))

    async def go():
        params = StdioServerParameters(command=exe, args=["-C", str(mini_repo.root), "mcp"])
        async with Client(params) as client:
            r1 = await client.call_tool(
                "engram_record_invariant",
                {
                    "rule_id": "PAY-1",
                    "scope": "svc/**",
                    "invariant_description": "amounts in cents",
                    "severity": "critical",
                },
            )
            r2 = await client.call_tool(
                "engram_record_failure",
                {
                    "file_path": "svc/pay.py",
                    "symbol_name": "pay",
                    "attempted_diff": (
                        "--- a/svc/pay.py\n+++ b/svc/pay.py\n@@ -2 +2 @@\n"
                        "-    return order.total\n+    return order.total / 100\n"
                    ),
                    "error_trace": "AssertionError: expected cents",
                    "notes": "converted to dollars",
                },
            )
            r3 = await client.call_tool("engram_preflight_check", {"file_path": "svc/pay.py", "symbol_name": "pay"})
            return r1.content[0].text, r2.content[0].text, r3.content[0].text

    inv, fail, pre = asyncio.run(go())
    assert "PAY-1" in inv and "recorded as failure #1" in fail
    assert "PAY-1" in pre and "converted to dollars" in pre


def test_incremental_index_in_a_fresh_process(mini_repo):
    """Regression: a new Engram instance (unfitted embedder) must be able to index new commits incrementally."""
    import subprocess
    import sys
    from pathlib import Path

    mini_repo.write("svc/a.py", "def alpha():\n    return 1\n")
    mini_repo.commit("feat: alpha")
    with Engram(mini_repo.root) as e:
        assert e.index().mode == "full"
    mini_repo.write("svc/b.py", "def beta_refund_window():\n    return 30\n")
    mini_repo.commit("feat: refund window for beta customers")
    with Engram(mini_repo.root) as fresh:
        stats = fresh.index()
        assert stats.mode == "incremental" and stats.new_commits == 1 and stats.embedded > 0
        assert fresh.db.get_meta("last_head") == fresh.git.head()
        assert any(h.symbol == "beta_refund_window" for h in fresh.search("refund window").hits)
    mini_repo.write("svc/c.py", "def gamma():\n    return 3\n")
    mini_repo.commit("feat: gamma")
    exe = str(Path(sys.executable).with_name("git-engram"))
    proc = subprocess.run([exe, "-C", str(mini_repo.root), "index"], capture_output=True, text=True)
    assert proc.returncode == 0, proc.stderr
    assert "incremental" in proc.stdout
