import subprocess
import sys
from pathlib import Path

from git_engram.engine import Engram
from git_engram.graveyard.exec_watcher import run_watched
from git_engram.graveyard.signature import diff_core, extract_signature, fingerprint

PYTEST_OUT = """tests/test_x.py::test_a FAILED
E       AssertionError: assert 83 == 82
tests/test_x.py:12: AssertionError
FAILED tests/test_x.py::test_a - AssertionError: assert 83 == 82
"""
PY_TRACE = """Traceback (most recent call last):
  File "/srv/app/cart.py", line 41, in checkout
    run()
RuntimeError: Event loop is closed
"""
GO_OUT = """WARNING: DATA RACE
--- FAIL: TestConsumer (0.01s)
panic: runtime error: index out of range [3] with length 3
"""
JEST_OUT = """ FAIL  src/order.test.ts
  ● parseOrder › keeps cents
    TypeError: Cannot read properties of undefined (reading 'total')
"""
RUST_OUT = """error[E0382]: borrow of moved value: `v`
thread 'main' panicked at 'index out of bounds', src/main.rs:4:5
"""


def test_signature_extraction_across_toolchains():
    s = extract_signature(PYTEST_OUT)
    assert "tests/test_x.py::test_a" in s.failing_tests and any("AssertionError" in e for e in s.exceptions)
    s = extract_signature(PY_TRACE)
    assert s.exceptions == ["RuntimeError: Event loop is closed"] and "cart.py:checkout" in s.frames
    s = extract_signature(GO_OUT)
    assert "WARNING: DATA RACE" in s.exceptions and "TestConsumer" in s.failing_tests
    assert any(e.startswith("panic:") for e in s.exceptions)
    s = extract_signature(JEST_OUT)
    assert "parseOrder › keeps cents" in s.failing_tests and any(e.startswith("TypeError") for e in s.exceptions)
    s = extract_signature("  ● parseOrder › cents\n\n    Expected: 1999\n    Received: 19.99\n\nTests: 1 failed")
    assert s.exceptions == ["expectation failed: expected 1999, received 19.99"]
    s = extract_signature(RUST_OUT)
    assert any("E0382" in e for e in s.exceptions) and any("panic in main" in e for e in s.exceptions)
    s = extract_signature("\x1b[31mERROR\x1b[0m: build failed: missing dependency foo")
    assert "\x1b" not in s.summary and "build failed" in s.summary


def test_fingerprint_ignores_volatile_details():
    a = extract_signature("Error at 0x7ffee1 after 1234 retries in 52ms\nFAILED t.py::x - KeyError")
    b = extract_signature("Error at 0x1a2b3c after 9876 retries in 17ms\nFAILED t.py::x - KeyError")
    d1 = "@@ -1 +1 @@\n-  a = 1\n+  a = 2\n"
    d2 = "@@ -5 +5 @@\n-a=1\n+a=2\n"
    assert diff_core(d1) == diff_core(d2)
    assert fingerprint(a, ["x.py"], d1) == fingerprint(b, ["x.py"], d2)
    assert fingerprint(a, ["y.py"], d1) != fingerprint(a, ["x.py"], d1)


def test_run_watched_tees_and_preserves_exit_code(capfd):
    run = run_watched([sys.executable, "-c", "import sys; print('out'); print('boom', file=sys.stderr); sys.exit(3)"])
    assert run.exit_code == 3 and "out" in run.stdout_tail and "boom" in run.stderr_tail
    captured = capfd.readouterr()
    assert "out" in captured.out and "boom" in captured.err


def _make_repo(mini_repo):
    mini_repo.write("app/rate.py", 'def rate(x):\n    """Apply the 10% fee."""\n    return x * 1.10\n')
    mini_repo.write(
        "tests/test_rate.py", "from app.rate import rate\n\n\ndef test_rate():\n    assert rate(10) == 11.0\n"
    )
    mini_repo.commit("feat: rate")
    return Engram(mini_repo.root)


def test_engine_exec_records_failed_attempt_with_symbol_targets(mini_repo):
    engine = _make_repo(mini_repo)
    engine.index()
    (mini_repo.root / "app/rate.py").write_text('def rate(x):\n    """Apply the 10% fee."""\n    return x * 1.2\n')
    cmd = [
        sys.executable,
        "-c",
        "import sys; print('FAILED tests/test_rate.py::test_rate - assert 12.0 == 11.0');"
        "print('AssertionError: assert 12.0 == 11.0', file=sys.stderr); sys.exit(1)",
    ]
    run, result = engine.exec(cmd, echo=False, notes="tried a 20% fee")
    assert run.exit_code == 1 and result is not None and not result.duplicate
    assert ("app/rate.py", "rate") in result.targets
    run2, again = engine.exec(cmd, echo=False)
    assert again is not None and again.duplicate and again.occurrences == 2
    warnings = engine.preflight("app/rate.py", "rate", "change the fee").warnings
    assert warnings and warnings[0].failure_id == result.failure_id and "same symbol" in warnings[0].match_reason
    assert "x * 1.2" in warnings[0].diff_excerpt
    engine.close()


def test_exec_success_and_clean_tree_record_nothing(mini_repo):
    engine = _make_repo(mini_repo)
    engine.index()
    ok, rec = engine.exec([sys.executable, "-c", "pass"], echo=False)
    assert ok.exit_code == 0 and rec is None
    bad, rec = engine.exec([sys.executable, "-c", "raise SystemExit(2)"], echo=False)
    assert bad.exit_code == 2 and rec is None  # clean tree: not an attempted change
    bad, rec = engine.exec([sys.executable, "-c", "raise SystemExit(2)"], echo=False, record_clean=True)
    assert rec is not None
    engine.close()


def test_cli_exec_propagates_exit_code_and_records(mini_repo):
    engine = _make_repo(mini_repo)
    engine.index()
    engine.close()
    (mini_repo.root / "app/rate.py").write_text("def rate(x):\n    return x\n")
    exe = Path(sys.executable).with_name("git-engram")
    proc = subprocess.run(
        [
            str(exe),
            "-C",
            str(mini_repo.root),
            "exec",
            "--",
            sys.executable,
            "-c",
            "import sys; sys.stderr.write('ValueError: fee missing\\n'); sys.exit(7)",
        ],
        capture_output=True,
        text=True,
    )
    assert proc.returncode == 7
    assert "ValueError: fee missing" in proc.stderr and "recorded in the Graveyard" in proc.stderr
    with Engram(mini_repo.root) as e:
        assert e.graveyard.stats()["failures"] == 1


def test_untracked_files_are_part_of_the_snapshot(mini_repo):
    engine = _make_repo(mini_repo)
    engine.index()
    (mini_repo.root / "app/new_helper.py").write_text("def helper():\n    return 42\n")
    _, rec = engine.exec([sys.executable, "-c", "raise SystemExit(1)"], echo=False)
    assert rec is not None and any(p == "app/new_helper.py" for p, _ in rec.targets)
    engine.close()
