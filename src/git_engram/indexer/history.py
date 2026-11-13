"""Causal commit extraction: per-symbol semantic diffs, intent classification and Causal Capsules.

For every commit, each touched file is parsed *at that commit* (new side) and *at its parent* (old side)
with the same AST chunker used for the working tree. Hunk line ranges are then resolved to the most
specific enclosing symbol, giving a ``git log -L``-style evolution trail for every function, class and
method without running one ``git log -L`` per symbol.
"""

from __future__ import annotations

import re
import time
from collections import OrderedDict
from collections.abc import Callable, Iterator
from dataclasses import dataclass, field

from ..models import Commit, FileChange, Symbol
from .ast_chunker import ChunkResult, chunk_source, symbols_overlapping
from .gitio import CatFile, Git

_CONVENTIONAL = re.compile(r"^(\w+)(?:\(([^)]*)\))?!?:\s*", re.IGNORECASE)
_TYPE_INTENT = {
    "fix": "bugfix",
    "bugfix": "bugfix",
    "hotfix": "bugfix",
    "feat": "feature",
    "feature": "feature",
    "perf": "perf",
    "refactor": "refactor",
    "revert": "revert",
    "docs": "docs",
    "doc": "docs",
    "test": "test",
    "tests": "test",
    "chore": "chore",
    "build": "chore",
    "ci": "chore",
    "deps": "chore",
    "style": "chore",
    "security": "security",
    "sec": "security",
}
_KEYWORD_INTENTS: list[tuple[str, re.Pattern[str]]] = [
    ("revert", re.compile(r"^revert\b|this reverts commit", re.IGNORECASE)),
    (
        "security",
        re.compile(
            r"\b(security|cve-\d{4}-\d+|xss|csrf|ssrf|injection|vulnerab\w*|sec-\d+|exploit|"
            r"replay attack|privilege escalation)\b",
            re.IGNORECASE,
        ),
    ),
    (
        "bugfix",
        re.compile(
            r"\b(fix(e[sd])?|bug|crash\w*|regression|incident|inc-\d+|hotfix|broken|race condition|"
            r"leak|panic|off-by-one|deadlock|corrupt\w*)\b",
            re.IGNORECASE,
        ),
    ),
    ("perf", re.compile(r"\b(perf|performance|speed ?up|faster|optimi[sz]\w*|latency)\b", re.IGNORECASE)),
    ("refactor", re.compile(r"\b(refactor\w*|cleanup|clean up|rename\w*|extract\w*)\b", re.IGNORECASE)),
    ("feature", re.compile(r"\b(add\w*|implement\w*|introduc\w*|support\w*|new)\b", re.IGNORECASE)),
]
LOAD_BEARING_INTENTS = {"bugfix", "security", "revert"}


def classify_intent(subject: str, body: str = "") -> str:
    m = _CONVENTIONAL.match(subject)
    text = f"{subject}\n{body}"
    if m:
        intent = _TYPE_INTENT.get(m.group(1).lower())
        scope = (m.group(2) or "").lower()
        if intent in {"bugfix", "chore", None} and (
            scope in {"security", "sec"} or _KEYWORD_INTENTS[1][1].search(text)
        ):
            return "security"
        if intent:
            return intent
    for intent, pattern in _KEYWORD_INTENTS:
        if pattern.search(subject) or (intent in {"revert", "security"} and pattern.search(body)):
            return intent
    return "change"


@dataclass(slots=True)
class SymbolTouch:
    path: str
    qualname: str
    hash: str
    ts: float
    added: int
    removed: int
    excerpt: str


@dataclass(slots=True)
class HistoryCommit:
    commit: Commit
    intent: str
    capsule: str
    file_paths: list[tuple[str, str, int, int]]  # (current_path, change, added, removed)
    touches: list[SymbolTouch] = field(default_factory=list)


class _ChunkCache:
    def __init__(self, capacity: int = 2048):
        self.capacity = capacity
        self.data: OrderedDict[str, list[Symbol]] = OrderedDict()

    def get(self, oid: str, path: str, text: str, chunker: Callable[[str, str], ChunkResult]) -> list[Symbol]:
        key = f"{oid}:{path}"
        hit = self.data.get(key)
        if hit is not None:
            self.data.move_to_end(key)
            return hit
        try:
            symbols = chunker(path, text).symbols
        except Exception:
            symbols = []
        self.data[key] = symbols
        if len(self.data) > self.capacity:
            self.data.popitem(last=False)
        return symbols


def _excerpt_for(symbol: Symbol | None, lines: list[tuple[int, str]], prefix: str, limit: int) -> list[str]:
    out = []
    for ln, text in lines:
        if symbol is None or symbol.start_line <= ln <= symbol.end_line:
            if text.strip():
                out.append(f"{prefix} {text.rstrip()[:160]}")
        if len(out) >= limit:
            break
    return out


def build_capsule(
    commit: Commit,
    intent: str,
    files: list[tuple[str, str, int, int]],
    touches: list[SymbolTouch],
    file_changes: list[FileChange],
) -> str:
    """Deterministic, human- and LLM-readable explanation of *why* code changed in this commit."""
    date = time.strftime("%Y-%m-%d", time.gmtime(commit.ts))
    lines = [f"[{intent}] {commit.subject}", f"commit {commit.hash[:10]} | {date} | {commit.author}"]
    if commit.body:
        lines.append(commit.body.strip()[:1500])
    if touches:
        lines.append("Touched symbols:")
        for t in touches[:20]:
            lines.append(f"  {t.path}::{t.qualname} (+{t.added} -{t.removed})")
    if files:
        lines.append("Files: " + ", ".join(f"{p} (+{a} -{r})" for p, _, a, r in files[:25]))
    excerpt: list[str] = []
    for fc in file_changes[:6]:
        body = _excerpt_for(None, fc.removed_lines, "-", 6) + _excerpt_for(None, fc.added_lines, "+", 10)
        if body:
            excerpt.append(f"  {fc.path}")
            excerpt.extend(f"    {b}" for b in body)
        if len(excerpt) > 60:
            break
    if excerpt:
        lines.append("Diff excerpt:")
        lines.extend(excerpt)
    return "\n".join(lines)


class HistoryExtractor:
    def __init__(
        self,
        git: Git,
        chunker: Callable[[str, str], ChunkResult] | None = None,
        skip: Callable[[str], bool] | None = None,
    ):
        self.git = git
        self.chunker = chunker or (lambda p, s: chunk_source(p, s))
        self.skip = skip or (lambda p: False)
        self.cache = _ChunkCache()

    def iter_history(
        self, max_commits: int, revision_range: str | None = None, alias: dict[str, str] | None = None
    ) -> Iterator[HistoryCommit]:
        """Newest-first stream of attributed commits. ``alias`` maps historical paths to current paths."""
        alias = alias if alias is not None else {}

        def resolve(path: str) -> str:
            seen = 0
            while path in alias and seen < 64:
                path = alias[path]
                seen += 1
            return path

        with CatFile(self.git.root) as cat:
            for commit in self.git.iter_commits(max_commits, revision_range):
                parent = commit.parents[0] if commit.parents else None
                files: list[tuple[str, str, int, int]] = []
                touches: list[SymbolTouch] = []
                kept: list[FileChange] = []
                for fc in commit.files:
                    current = resolve(fc.path)
                    if fc.change == "R" and fc.old_path and fc.old_path != fc.path:
                        alias[fc.old_path] = current
                    if self.skip(current):
                        continue
                    files.append((current, fc.change, fc.added, fc.removed))
                    kept.append(fc)
                    touches.extend(self._attribute(cat, commit, parent, fc, current))
                intent = classify_intent(commit.subject, commit.body)
                capsule = build_capsule(commit, intent, files, touches, kept)
                yield HistoryCommit(commit=commit, intent=intent, capsule=capsule, file_paths=files, touches=touches)

    def _attribute(
        self, cat: CatFile, commit: Commit, parent: str | None, fc: FileChange, current: str
    ) -> list[SymbolTouch]:
        merged: dict[str, SymbolTouch] = {}
        new_syms: list[Symbol] = []
        old_syms: list[Symbol] = []
        if fc.change != "D" and fc.new_ranges:
            blob = cat.read(commit.hash, fc.path)
            if blob:
                new_syms = self.cache.get(blob[0], fc.path, blob[1], self.chunker)
        if fc.change != "A" and parent and fc.old_ranges and fc.old_path:
            blob = cat.read(parent, fc.old_path)
            if blob:
                old_syms = self.cache.get(blob[0], fc.old_path, blob[1], self.chunker)
        for side, syms, ranges, lines, prefix in (
            ("new", new_syms, fc.new_ranges, fc.added_lines, "+"),
            ("old", old_syms, fc.old_ranges, fc.removed_lines, "-"),
        ):
            for sym in symbols_overlapping(syms, ranges):
                n = sum(1 for ln, _ in lines if sym.start_line <= ln <= sym.end_line)
                ex = _excerpt_for(sym, lines, prefix, 8)
                t = merged.get(sym.qualname)
                if t is None:
                    t = SymbolTouch(current, sym.qualname, commit.hash, commit.ts, 0, 0, "")
                    merged[sym.qualname] = t
                if side == "new":
                    t.added += n
                else:
                    t.removed += n
                if ex:
                    t.excerpt = "\n".join(filter(None, [t.excerpt, *ex]))
        return list(merged.values())
