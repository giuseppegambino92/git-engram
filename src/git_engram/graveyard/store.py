"""The Graveyard Index: persistent memory of failed coding attempts (Negative RAG)."""

from __future__ import annotations

import time
from dataclasses import dataclass

from ..indexer.ast_chunker import chunk_source, symbols_overlapping
from ..indexer.gitio import Git, parse_patch
from ..storage.db import EngramDB
from .signature import extract_signature, fingerprint


@dataclass(slots=True)
class RecordResult:
    failure_id: int
    fingerprint: str
    duplicate: bool
    occurrences: int
    targets: list[tuple[str, str]]
    signature: str


class Graveyard:
    def __init__(self, db: EngramDB, git: Git | None, chunker=None):
        self.db = db
        self.git = git
        self.chunker = chunker or chunk_source

    def _targets(self, diff: str, file_path: str | None, symbol: str | None) -> list[tuple[str, str]]:
        targets: dict[tuple[str, str], None] = {}
        head_cache: dict[str, list] = {}
        for fc in parse_patch(diff):
            path = fc.path if fc.change != "D" else (fc.old_path or fc.path)
            touched_lines = sorted(set(fc.old_touch))
            ranges = [(ln, ln) for ln in touched_lines] or fc.old_ranges or fc.new_ranges
            symbols = []
            if self.git is not None and fc.change != "A":
                if path not in head_cache:
                    src = self.git.show_file("HEAD", fc.old_path or path)
                    head_cache[path] = self.chunker(path, src).symbols if src else []
                symbols = head_cache[path]
            elif self.git is not None:
                full = self.git.root / path
                if full.exists():
                    symbols = self.chunker(path, full.read_text("utf-8", "replace")).symbols
                    ranges = [(ln, ln) for ln, _ in fc.added_lines] or fc.new_ranges
            touched = symbols_overlapping(symbols, ranges)
            if touched:
                for s in touched:
                    targets[(path, s.qualname)] = None
            else:
                targets[(path, "")] = None
        if file_path:
            if symbol:
                targets.pop((file_path, ""), None)
                targets[(file_path, symbol)] = None
            elif not any(p == file_path for p, _ in targets):
                targets[(file_path, "")] = None
        return list(targets)

    def record(
        self,
        *,
        command: str,
        exit_code: int,
        attempted_diff: str,
        error_output: str,
        notes: str = "",
        source: str = "manual",
        file_path: str | None = None,
        symbol: str | None = None,
        ts: float | None = None,
        max_diff_bytes: int = 131_072,
        max_stderr_bytes: int = 65_536,
    ) -> RecordResult:
        now = ts if ts is not None else time.time()
        diff = attempted_diff[:max_diff_bytes]
        stderr = error_output[-max_stderr_bytes:]
        sig = extract_signature(stderr)
        targets = self._targets(diff, file_path, symbol)
        fp = fingerprint(sig, [p for p, _ in targets], diff)
        head = (self.git.head() if self.git else None) or ""
        with self.db.transaction() as conn:
            row = conn.execute("SELECT id, occurrences FROM failures WHERE fingerprint = ?", (fp,)).fetchone()
            if row is not None:
                occ = int(row["occurrences"]) + 1
                conn.execute(
                    "UPDATE failures SET occurrences = ?, last_ts = ?, notes = CASE WHEN ? != '' THEN ? "
                    "ELSE notes END WHERE id = ?",
                    (occ, now, notes, notes, row["id"]),
                )
                return RecordResult(int(row["id"]), fp, True, occ, targets, sig.summary)
            cur = conn.execute(
                "INSERT INTO failures(ts, fingerprint, command, exit_code, attempted_diff, stderr, signature, "
                "failing_tests, notes, source, head, occurrences, last_ts) VALUES(?,?,?,?,?,?,?,?,?,?,?,1,?)",
                (
                    now,
                    fp,
                    command,
                    exit_code,
                    diff,
                    stderr,
                    sig.summary,
                    "\n".join(sig.failing_tests),
                    notes,
                    source,
                    head,
                    now,
                ),
            )
            fid = int(cur.lastrowid)
            conn.executemany(
                "INSERT OR IGNORE INTO failure_targets(failure_id, path, symbol) VALUES(?,?,?)",
                [(fid, p, s) for p, s in targets],
            )
        return RecordResult(fid, fp, False, 1, targets, sig.summary)

    def targets_of(self, failure_id: int) -> list[tuple[str, str]]:
        return [
            (r["path"], r["symbol"])
            for r in self.db.query("SELECT path, symbol FROM failure_targets WHERE failure_id = ?", (failure_id,))
        ]

    def document(self, failure_id: int) -> tuple[str, str, str | None, str | None, str, str, float] | None:
        row = self.db.query_one("SELECT * FROM failures WHERE id = ?", (failure_id,))
        if row is None:
            return None
        targets = self.targets_of(failure_id)
        first_path = targets[0][0] if targets else None
        first_sym = next((s for _, s in targets if s), None)
        title = f"FAILED ATTEMPT: {row['command']} (exit {row['exit_code']})"
        removed, added = [], []
        for line in row["attempted_diff"].splitlines():
            if line.startswith("-") and not line.startswith("---"):
                removed.append(line)
            elif line.startswith("+") and not line.startswith("+++"):
                added.append(line)
        body = "\n".join(
            filter(
                None,
                [
                    row["notes"],
                    row["signature"],
                    "Failing tests: " + row["failing_tests"].replace("\n", ", ") if row["failing_tests"] else "",
                    "Targets: " + ", ".join(f"{p}::{s}" if s else p for p, s in targets),
                    "Removed lines:\n" + "\n".join(removed[:30]) if removed else "",
                    "Added lines:\n" + "\n".join(added[:30]) if added else "",
                ],
            )
        )
        return f"fail:{failure_id}", "failure", first_path, first_sym, title, body, float(row["ts"])

    def stats(self) -> dict[str, int]:
        row = self.db.query_one("SELECT COUNT(*) AS n, COALESCE(SUM(occurrences), 0) AS occ FROM failures")
        files = self.db.query_one("SELECT COUNT(DISTINCT path) AS n FROM failure_targets")
        return {"failures": int(row["n"]), "occurrences": int(row["occ"]), "files": int(files["n"])}
