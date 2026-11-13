"""`git-engram exec -- <cmd>`: transparent test/build wrapper that tees output and captures failures."""

from __future__ import annotations

import os
import shlex
import signal
import subprocess
import sys
import threading
import time
from collections import deque
from dataclasses import dataclass
from typing import BinaryIO


@dataclass(slots=True)
class CapturedRun:
    command: str
    exit_code: int
    stdout_tail: str
    stderr_tail: str
    duration_s: float

    @property
    def combined(self) -> str:
        """stderr first (compiler/runtime errors), then stdout tail (pytest/jest/go test reports)."""
        parts = [p for p in (self.stderr_tail.strip(), self.stdout_tail.strip()) if p]
        return "\n".join(parts)


def _pump(src: BinaryIO, sink: BinaryIO | None, buffer: deque[bytes], limit: int) -> None:
    size = 0
    for chunk in iter(lambda: src.readline(), b""):
        if sink is not None:
            try:
                sink.write(chunk)
                sink.flush()
            except (BrokenPipeError, ValueError):
                sink = None
        buffer.append(chunk)
        size += len(chunk)
        while size > limit and len(buffer) > 1:
            size -= len(buffer.popleft())
    src.close()


def run_watched(
    argv: list[str],
    cwd: str | os.PathLike[str] | None = None,
    echo: bool = True,
    max_bytes: int = 65_536,
    env: dict[str, str] | None = None,
) -> CapturedRun:
    if not argv:
        raise ValueError("no command given; usage: git-engram exec -- <command> [args...]")
    start = time.perf_counter()
    proc = subprocess.Popen(argv, cwd=cwd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, env=env)
    out_buf: deque[bytes] = deque()
    err_buf: deque[bytes] = deque()
    out_sink = sys.stdout.buffer if echo and hasattr(sys.stdout, "buffer") else None
    err_sink = sys.stderr.buffer if echo and hasattr(sys.stderr, "buffer") else None
    threads = [
        threading.Thread(target=_pump, args=(proc.stdout, out_sink, out_buf, max_bytes), daemon=True),
        threading.Thread(target=_pump, args=(proc.stderr, err_sink, err_buf, max_bytes), daemon=True),
    ]
    for t in threads:
        t.start()
    try:
        code = proc.wait()
    except KeyboardInterrupt:
        proc.send_signal(signal.SIGINT)
        code = proc.wait()
    for t in threads:
        t.join(timeout=5)
    return CapturedRun(
        command=shlex.join(argv),
        exit_code=code,
        stdout_tail=b"".join(out_buf).decode("utf-8", "replace"),
        stderr_tail=b"".join(err_buf).decode("utf-8", "replace"),
        duration_s=time.perf_counter() - start,
    )
