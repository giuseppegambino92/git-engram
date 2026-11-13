"""Failure-signature extraction: turn noisy test/build output into a stable, embeddable error signature."""

from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass, field

_ANSI = re.compile(r"\x1b\[[0-9;?]*[ -/]*[@-~]")
_PY_EXC = re.compile(
    r"^(?:E\s+)?([A-Za-z_][\w.]*(?:Error|Exception|Exit|Interrupt|Warning|Failure|Timeout))\b(?::\s*(.*))?$"
)
_PY_FRAME = re.compile(r'^\s*File "([^"]+)", line (\d+), in (\S+)')
_PYTEST_FAILED = re.compile(r"^(?:FAILED|ERROR)\s+(\S+?)(?:\s+-\s+(.*))?$")
_PYTEST_LOC = re.compile(r"^([\w/.\-]+\.py):(\d+):\s+(\w+)")
_GO_FAIL = re.compile(r"^\s*--- FAIL: (\S+)")
_GO_PANIC = re.compile(r"^panic: (.*)")
_GO_RACE = re.compile(r"WARNING: DATA RACE")
_RUST_PANIC = re.compile(r"thread '([^']+)' panicked at (?:'([^']*)'|(.*))")
_RUST_ERR = re.compile(r"^error(\[E\d{4}\])?: (.*)")
_TS_ERR = re.compile(r"error (TS\d{4}): (.*)")
_JEST_FAIL = re.compile(r"^\s*●\s+(.*)")
_JEST_EXPECT = re.compile(r"^\s*(Expected|Received)(?: [\w ]+)?:\s+(.*)$")
_JS_ERR = re.compile(r"^\s*(TypeError|ReferenceError|SyntaxError|RangeError|AssertionError|Error): (.*)")
_GENERIC = re.compile(r"\b(error|failed|failure|fatal|exception|panic|segmentation fault|assert)\b", re.I)

_NORMALISERS = [
    (re.compile(r"0x[0-9a-fA-F]+"), "<HEX>"),
    (re.compile(r"\b[0-9a-f]{12,64}\b"), "<HASH>"),
    (re.compile(r"(/[\w.\-]+)+/([\w.\-]+)"), r"\2"),
    (re.compile(r"\b\d+(\.\d+)?(ms|s|µs|ns)\b"), "<DUR>"),
    (re.compile(r"\b\d{3,}\b"), "<N>"),
    (re.compile(r"\s+"), " "),
]


@dataclass(slots=True)
class FailureSignature:
    summary: str
    exceptions: list[str] = field(default_factory=list)
    failing_tests: list[str] = field(default_factory=list)
    frames: list[str] = field(default_factory=list)

    @property
    def normalized(self) -> str:
        text = " | ".join(self.exceptions + self.failing_tests)
        for pattern, repl in _NORMALISERS:
            text = pattern.sub(repl, text)
        return text.strip()


def strip_ansi(text: str) -> str:
    return _ANSI.sub("", text)


def extract_signature(output: str, max_lines: int = 40) -> FailureSignature:
    text = strip_ansi(output)
    exceptions: list[str] = []
    tests: list[str] = []
    frames: list[str] = []
    generic: list[str] = []
    expect: dict[str, str] = {}

    def add(bucket: list[str], value: str) -> None:
        value = value.strip()[:240]
        if value and value not in bucket:
            bucket.append(value)

    for line in text.splitlines():
        stripped = line.strip()
        if not stripped:
            continue
        if m := _PY_FRAME.match(line):
            add(frames, f"{m.group(1).rsplit('/', 1)[-1]}:{m.group(3)}")
        elif m := _PYTEST_FAILED.match(stripped):
            add(tests, m.group(1))
            if m.group(2):
                add(exceptions, m.group(2))
        elif m := _PY_EXC.match(stripped):
            add(exceptions, f"{m.group(1)}: {m.group(2)}" if m.group(2) else m.group(1))
        elif m := _PYTEST_LOC.match(stripped):
            add(frames, f"{m.group(1).rsplit('/', 1)[-1]}:{m.group(2)}:{m.group(3)}")
        elif m := _GO_FAIL.match(line):
            add(tests, m.group(1))
        elif m := _GO_PANIC.match(stripped):
            add(exceptions, f"panic: {m.group(1)}")
        elif _GO_RACE.search(stripped):
            add(exceptions, "WARNING: DATA RACE")
        elif m := _RUST_PANIC.search(stripped):
            add(exceptions, f"panic in {m.group(1)}: {m.group(2) or m.group(3)}")
        elif m := _TS_ERR.search(stripped):
            add(exceptions, f"{m.group(1)}: {m.group(2)}")
        elif m := _RUST_ERR.match(stripped):
            add(exceptions, f"error{m.group(1) or ''}: {m.group(2)}")
        elif m := _JS_ERR.match(stripped):
            add(exceptions, f"{m.group(1)}: {m.group(2)}")
        elif m := _JEST_FAIL.match(line):
            add(tests, m.group(1))
        elif m := _JEST_EXPECT.match(line):
            expect[m.group(1)] = m.group(2).strip()
            if "Expected" in expect and "Received" in expect:
                add(
                    exceptions,
                    f"expectation failed: expected {expect.pop('Expected')}, received {expect.pop('Received')}",
                )
        elif _GENERIC.search(stripped) and len(generic) < 8:
            add(generic, stripped)
    if not exceptions and generic:
        exceptions = generic[-4:]
    if not exceptions and not tests:
        tail = [ln for ln in text.splitlines() if ln.strip()][-6:]
        exceptions = [t.strip()[:240] for t in tail]
    summary_lines = [
        *(f"error: {e}" for e in exceptions[-6:]),
        *(f"failing: {t}" for t in tests[:10]),
        *(f"at {f}" for f in frames[-6:]),
    ]
    return FailureSignature("\n".join(summary_lines[:max_lines]), exceptions[-6:], tests[:20], frames[-10:])


def diff_core(diff: str) -> str:
    """Whitespace-insensitive changed lines only (ignores hunk offsets, context and index headers)."""
    core = []
    for line in diff.splitlines():
        if line.startswith(("+++", "---")):
            continue
        if line.startswith(("+", "-")):
            core.append(line[0] + re.sub(r"\s+", "", line[1:]))
    return "\n".join(core)


def fingerprint(signature: FailureSignature, paths: list[str], diff: str) -> str:
    h = hashlib.sha1()
    h.update(signature.normalized.encode())
    h.update("\x00".join(sorted(set(paths))).encode())
    h.update(diff_core(diff).encode())
    return h.hexdigest()
