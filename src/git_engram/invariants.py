"""Deterministic architectural invariants in Open Knowledge Format (OKF).

Invariants live in ``.engram/invariants.okf.json`` (commit it: it is team knowledge) and are mirrored into
SQLite. Scopes are glob patterns over repo-relative paths, optionally narrowed to symbols with ``::``:

* ``backend/payments/**``                      every file below the directory
* ``*.sql``                                    any SQL file at any depth
* ``backend/auth/token.py::is_token_expired``  a single symbol
* ``backend/api/schemas.py::Order*``           symbol glob

Matching uses pre-compiled regular expressions; no retrieval or scoring is involved, so a matching
invariant is *always* returned for a touched file (deterministic, ~microseconds).
"""

from __future__ import annotations

import json
import re
import time
from dataclasses import dataclass
from pathlib import Path

from .models import InvariantHit
from .storage.db import EngramDB

OKF_FORMAT = "okf/1"
SEVERITIES = ("info", "warning", "critical")


def glob_to_regex(pattern: str) -> re.Pattern[str]:
    pattern = pattern.strip().lstrip("./")
    out = []
    i = 0
    while i < len(pattern):
        c = pattern[i]
        if pattern.startswith("**/", i):
            out.append("(?:.*/)?")
            i += 3
            continue
        if pattern.startswith("**", i):
            out.append(".*")
            i += 2
            continue
        if c == "*":
            out.append("[^/]*")
        elif c == "?":
            out.append("[^/]")
        else:
            out.append(re.escape(c))
        i += 1
    body = "".join(out)
    if "/" not in pattern:
        body = "(?:.*/)?" + body
    return re.compile(f"^{body}$")


@dataclass(slots=True)
class CompiledScope:
    raw: str
    path_re: re.Pattern[str]
    symbol_re: re.Pattern[str] | None


def compile_scope(scope: str) -> list[CompiledScope]:
    compiled = []
    for part in [p.strip() for p in re.split(r"[,\n]", scope) if p.strip()]:
        path_part, _, sym_part = part.partition("::")
        sym_re = (
            re.compile("^" + re.escape(sym_part).replace(r"\*", ".*").replace(r"\?", ".") + "$") if sym_part else None
        )
        compiled.append(CompiledScope(part, glob_to_regex(path_part or "**"), sym_re))
    return compiled


class InvariantStore:
    def __init__(self, db: EngramDB, repo_root: Path):
        self.db = db
        self.path = repo_root / ".engram" / "invariants.okf.json"
        self._compiled: list[tuple[dict, list[CompiledScope]]] = []
        self._mtime: float | None = None
        self.sync_from_file()

    def sync_from_file(self) -> None:
        if self.path.exists():
            mtime = self.path.stat().st_mtime
            if mtime != self._mtime:
                data = json.loads(self.path.read_text(encoding="utf-8"))
                with self.db.transaction() as conn:
                    conn.execute("DELETE FROM invariants")
                    for inv in data.get("invariants", []):
                        scope = inv["scope"] if isinstance(inv["scope"], str) else ",".join(inv["scope"])
                        conn.execute(
                            "INSERT INTO invariants(rule_id, scope, description, severity, created_ts, source) "
                            "VALUES(?,?,?,?,?,?)",
                            (
                                inv["rule_id"],
                                scope,
                                inv["description"],
                                inv.get("severity", "warning"),
                                float(inv.get("created_ts", time.time())),
                                inv.get("source", "okf"),
                            ),
                        )
                self._mtime = mtime
        self._compile()

    def _compile(self) -> None:
        self._compiled = [
            (dict(r), compile_scope(r["scope"])) for r in self.db.query("SELECT * FROM invariants ORDER BY rule_id")
        ]

    def _write_file(self) -> None:
        rows = self.db.query("SELECT * FROM invariants ORDER BY rule_id")
        payload = {
            "format": OKF_FORMAT,
            "invariants": [
                {
                    "rule_id": r["rule_id"],
                    "scope": [s.strip() for s in r["scope"].split(",") if s.strip()],
                    "description": r["description"],
                    "severity": r["severity"],
                    "created_ts": r["created_ts"],
                    "source": r["source"],
                }
                for r in rows
            ],
        }
        self.path.parent.mkdir(parents=True, exist_ok=True)
        tmp = self.path.with_suffix(".tmp")
        tmp.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
        tmp.replace(self.path)
        self._mtime = self.path.stat().st_mtime

    def record(
        self, rule_id: str, scope: str | list[str], description: str, severity: str = "warning", source: str = "agent"
    ) -> dict:
        if not re.fullmatch(r"[A-Za-z0-9][\w.\-]{0,63}", rule_id):
            raise ValueError("rule_id must be 1-64 chars of [A-Za-z0-9_.-] starting with an alphanumeric")
        if severity not in SEVERITIES:
            raise ValueError(f"severity must be one of {SEVERITIES}")
        scope_str = scope if isinstance(scope, str) else ",".join(scope)
        if not compile_scope(scope_str):
            raise ValueError("scope must contain at least one glob pattern")
        self.db.execute(
            "INSERT INTO invariants(rule_id, scope, description, severity, created_ts, source) VALUES(?,?,?,?,?,?) "
            "ON CONFLICT(rule_id) DO UPDATE SET scope=excluded.scope, description=excluded.description, "
            "severity=excluded.severity, source=excluded.source",
            (rule_id, scope_str, description.strip(), severity, time.time(), source),
        )
        self._write_file()
        self._compile()
        return dict(self.db.query_one("SELECT * FROM invariants WHERE rule_id = ?", (rule_id,)))

    def remove(self, rule_id: str) -> bool:
        cur = self.db.execute("DELETE FROM invariants WHERE rule_id = ?", (rule_id,))
        self._write_file()
        self._compile()
        return cur.rowcount > 0

    def all(self) -> list[dict]:
        return [inv for inv, _ in self._compiled]

    def match(self, path: str, symbol: str | None = None) -> list[InvariantHit]:
        hits: list[InvariantHit] = []
        path = path.lstrip("./")
        for inv, scopes in self._compiled:
            for sc in scopes:
                if not sc.path_re.match(path):
                    continue
                if sc.symbol_re is not None and not (symbol and sc.symbol_re.match(symbol)):
                    continue
                hits.append(
                    InvariantHit(
                        rule_id=inv["rule_id"],
                        scope=inv["scope"],
                        description=inv["description"],
                        severity=inv["severity"],
                        matched=sc.raw,
                    )
                )
                break
        order = {s: i for i, s in enumerate(reversed(SEVERITIES))}
        return sorted(hits, key=lambda h: order.get(h.severity, 9))

    def document(self, inv: dict) -> tuple[str, str, None, None, str, str, float]:
        body = f"{inv['description']}\nScope: {inv['scope']}\nSeverity: {inv['severity']}"
        return f"inv:{inv['rule_id']}", "invariant", None, None, f"invariant {inv['rule_id']}", body, inv["created_ts"]
