"""Raw git plumbing (subprocess only, no GitPython): streaming log/patch parser, cat-file batch, blame."""

from __future__ import annotations

import os
import re
import subprocess
from collections.abc import Iterator
from pathlib import Path

from ..models import Commit, FileChange

_GIT_ENV = {**os.environ, "GIT_PAGER": "cat", "LC_ALL": "C", "GIT_CONFIG_NOSYSTEM": "1"}
_REC = b"\x00\x01"
_HUNK = re.compile(r"^@@ -(\d+)(?:,(\d+))? \+(\d+)(?:,(\d+))? @@")
_ZERO = "0" * 40
EXCERPT_LINES = 400


class GitError(RuntimeError):
    pass


def _clean_path(raw: str) -> str:
    raw = raw.strip()
    if raw.startswith('"') and raw.endswith('"'):
        raw = (
            raw[1:-1]
            .encode("latin-1", "backslashreplace")
            .decode("unicode_escape")
            .encode("latin-1")
            .decode("utf-8", "replace")
        )
    return raw


class Git:
    def __init__(self, root: Path | str):
        self.root = Path(root)

    @classmethod
    def discover(cls, path: Path | str) -> Git:
        out = subprocess.run(
            ["git", "-C", str(path), "rev-parse", "--show-toplevel"], capture_output=True, text=True, env=_GIT_ENV
        )
        if out.returncode != 0:
            raise GitError(f"not a git repository: {path}")
        return cls(Path(out.stdout.strip()))

    def run(self, *args: str, check: bool = True, input_text: str | None = None) -> str:
        proc = subprocess.run(
            ["git", "-C", str(self.root), "-c", "core.quotepath=false", *args],
            capture_output=True,
            env=_GIT_ENV,
            input=input_text.encode() if input_text is not None else None,
        )
        if check and proc.returncode != 0:
            raise GitError(f"git {' '.join(args)} failed: {proc.stderr.decode('utf-8', 'replace').strip()}")
        return proc.stdout.decode("utf-8", "replace")

    def head(self) -> str | None:
        out = subprocess.run(
            ["git", "-C", str(self.root), "rev-parse", "--verify", "-q", "HEAD"],
            capture_output=True,
            text=True,
            env=_GIT_ENV,
        )
        return out.stdout.strip() or None

    def is_ancestor(self, older: str, newer: str) -> bool:
        proc = subprocess.run(
            ["git", "-C", str(self.root), "merge-base", "--is-ancestor", older, newer],
            capture_output=True,
            env=_GIT_ENV,
        )
        return proc.returncode == 0

    def ls_files(self) -> list[str]:
        out = self.run("ls-files", "-z", "--cached", "--others", "--exclude-standard")
        return sorted({p for p in out.split("\0") if p})

    def diff_head(self) -> str:
        if self.head() is None:
            return ""
        return self.run("diff", "HEAD", "--no-color", "--no-ext-diff", check=False)

    def untracked(self) -> list[str]:
        out = self.run("ls-files", "-z", "--others", "--exclude-standard", check=False)
        return [p for p in out.split("\0") if p]

    def show_file(self, rev: str, path: str) -> str | None:
        proc = subprocess.run(["git", "-C", str(self.root), "show", f"{rev}:{path}"], capture_output=True, env=_GIT_ENV)
        return proc.stdout.decode("utf-8", "replace") if proc.returncode == 0 else None

    # ------------------------------------------------------------------ log
    def iter_commits(self, max_commits: int, revision_range: str | None = None) -> Iterator[Commit]:
        """Stream commits newest-first with ``-U0`` patches; never materialises the whole log in memory."""
        if self.head() is None:
            return
        fmt = "%x00%x01%H%x1f%P%x1f%at%x1f%an%x1f%s%x1f%b%x1e"
        cmd = [
            "git",
            "-C",
            str(self.root),
            "-c",
            "core.quotepath=false",
            "log",
            "--no-merges",
            "--no-color",
            "-M",
            "-U0",
            "--no-ext-diff",
            "--no-textconv",
            f"--format={fmt}",
            "-p",
            f"--max-count={max_commits}",
        ]
        if revision_range:
            cmd.append(revision_range)
        proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL, env=_GIT_ENV)
        assert proc.stdout is not None
        buf = b""
        try:
            while True:
                chunk = proc.stdout.read(1 << 16)
                if not chunk:
                    break
                buf += chunk
                while True:
                    start = buf.find(_REC)
                    nxt = buf.find(_REC, start + 2) if start >= 0 else -1
                    if nxt < 0:
                        break
                    yield parse_commit_record(buf[start + 2 : nxt].decode("utf-8", "replace"))
                    buf = buf[nxt:]
            start = buf.find(_REC)
            if start >= 0 and len(buf) > start + 2:
                yield parse_commit_record(buf[start + 2 :].decode("utf-8", "replace"))
        finally:
            proc.stdout.close()
            proc.wait()

    # ---------------------------------------------------------------- blame
    def blame_lines(self, path: str) -> list[str]:
        """Commit hash owning each line of the *working tree* version of ``path`` (index = line - 1)."""
        proc = subprocess.run(
            ["git", "-C", str(self.root), "blame", "--porcelain", "-w", "--", path], capture_output=True, env=_GIT_ENV
        )
        if proc.returncode != 0:
            return []
        owners: dict[int, str] = {}
        current = _ZERO
        for raw in proc.stdout.decode("utf-8", "replace").split("\n"):
            if not raw or raw.startswith("\t"):
                continue
            parts = raw.split(" ")
            if len(parts) >= 3 and len(parts[0]) == 40 and all(c in "0123456789abcdef" for c in parts[0]):
                current = parts[0]
                final_line = int(parts[2])
                count = int(parts[3]) if len(parts) >= 4 else 1
                for k in range(count):
                    owners[final_line + k] = current
        if not owners:
            return []
        return [owners.get(i, _ZERO) for i in range(1, max(owners) + 1)]


def parse_commit_record(record: str) -> Commit:
    header, _, patch = record.partition("\x1e")
    fields = header.split("\x1f")
    while len(fields) < 6:
        fields.append("")
    commit = Commit(
        hash=fields[0].strip(),
        parents=fields[1].split(),
        ts=float(fields[2] or 0),
        author=fields[3],
        subject=fields[4].strip(),
        body=fields[5].strip(),
    )
    commit.files = parse_patch(patch)
    return commit


def parse_patch(patch: str) -> list[FileChange]:
    files: list[FileChange] = []
    cur: FileChange | None = None
    old_ln = new_ln = 0
    for line in patch.split("\n"):
        if line.startswith("diff --git "):
            rest = line[len("diff --git ") :]
            a, _, b = rest.partition(" b/")
            cur = FileChange(path=_clean_path(b), old_path=_clean_path(a[2:] if a.startswith("a/") else a))
            files.append(cur)
            continue
        if cur is None:
            continue
        if line.startswith("new file mode"):
            cur.change = "A"
        elif line.startswith("deleted file mode"):
            cur.change = "D"
        elif line.startswith("rename from "):
            cur.old_path = _clean_path(line[len("rename from ") :])
            cur.change = "R"
        elif line.startswith("rename to "):
            cur.path = _clean_path(line[len("rename to ") :])
            cur.change = "R"
        elif line.startswith("--- "):
            target = line[4:]
            if target != "/dev/null":
                cur.old_path = _clean_path(target[2:] if target.startswith("a/") else target)
        elif line.startswith("+++ "):
            target = line[4:]
            if target != "/dev/null":
                cur.path = _clean_path(target[2:] if target.startswith("b/") else target)
        elif line.startswith("@@"):
            m = _HUNK.match(line)
            if m:
                a, b = int(m.group(1)), int(m.group(2) if m.group(2) is not None else 1)
                c, d = int(m.group(3)), int(m.group(4) if m.group(4) is not None else 1)
                cur.old_ranges.append((max(a, 1), max(a, 1) + max(b, 1) - 1))
                cur.new_ranges.append((max(c, 1), max(c, 1) + max(d, 1) - 1))
                # For a zero-length side, git reports the line *after which* the change happens.
                old_ln = a if b > 0 else a + 1
                new_ln = c if d > 0 else c + 1
        elif line.startswith("+"):
            cur.old_touch.append(max(old_ln - 1, 1))
            cur.added += 1
            if len(cur.added_lines) < EXCERPT_LINES:
                cur.added_lines.append((new_ln, line[1:]))
            new_ln += 1
        elif line.startswith(" "):
            old_ln += 1
            new_ln += 1
        elif line.startswith("-"):
            cur.old_touch.append(old_ln)
            cur.removed += 1
            if len(cur.removed_lines) < EXCERPT_LINES:
                cur.removed_lines.append((old_ln, line[1:]))
            old_ln += 1
    for f in files:
        if f.change == "M" and f.old_path and f.old_path != f.path:
            f.change = "R"
        if f.change == "A":
            f.old_ranges = []
        if f.change == "D":
            f.new_ranges = []
    return files


class CatFile:
    """Persistent ``git cat-file --batch`` reader: one process for thousands of historical blobs."""

    def __init__(self, root: Path | str):
        self.proc = subprocess.Popen(
            ["git", "-C", str(root), "cat-file", "--batch"],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
            env=_GIT_ENV,
        )

    def read(self, rev: str, path: str, max_bytes: int = 1_000_000) -> tuple[str, str] | None:
        """Return ``(blob_oid, text)`` for ``rev:path`` or ``None`` when missing/binary/too large."""
        assert self.proc.stdin is not None and self.proc.stdout is not None
        self.proc.stdin.write(f"{rev}:{path}\n".encode())
        self.proc.stdin.flush()
        header = self.proc.stdout.readline().decode("utf-8", "replace").strip()
        parts = header.split()
        if len(parts) != 3 or parts[1] == "missing":
            return None
        size = int(parts[2])
        data = self.proc.stdout.read(size)
        self.proc.stdout.read(1)
        if parts[1] != "blob" or size > max_bytes or b"\x00" in data[:8192]:
            return None
        return parts[0], data.decode("utf-8", "replace")

    def close(self) -> None:
        if self.proc.stdin:
            self.proc.stdin.close()
        self.proc.wait(timeout=10)

    def __enter__(self) -> CatFile:
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()
