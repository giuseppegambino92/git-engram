"""Retrievable document builders + embedding/quantization write path shared by all subsystems."""

from __future__ import annotations

import sqlite3
from collections.abc import Sequence

import numpy as np

from .embeddings import Embedder, encode_vectors, load_rotation
from .models import Symbol
from .storage.db import EngramDB


def symbol_document(sym: Symbol) -> tuple[str, str, str, str, str, str]:
    """(doc_key, kind, path, symbol, title, body)."""
    title = f"{sym.kind} {sym.qualname}" if sym.qualname != "<module>" else f"module {sym.path}"
    parts = [sym.signature]
    if sym.docstring:
        parts.append(sym.docstring)
    parts.append(sym.content)
    return f"sym:{sym.path}::{sym.qualname}", "symbol", sym.path, sym.qualname, title, "\n".join(parts)


def file_document(
    path: str, language: str, symbols: Sequence[Symbol], content: str, imports: Sequence[str]
) -> tuple[str, str, str, str, str, str]:
    lines = [f"{language} file {path}"]
    module = next((s for s in symbols if s.qualname == "<module>"), None)
    if module is not None and module.docstring:
        lines.append(module.docstring)
    named = [s for s in symbols if s.qualname != "<module>" and not s.qualname.startswith("<lines")]
    if named:
        lines.append("Symbols:")
        lines.extend(f"  {s.kind} {s.qualname}: {s.signature[:160]}" for s in named[:80])
        if imports:
            lines.append("Imports: " + ", ".join(imports[:40]))
    else:
        lines.append(content[:6000])
    return f"file:{path}", "file", path, "", path, "\n".join(lines)


def doc_text(title: str, symbol: str | None, path: str | None, body: str) -> str:
    return "\n".join(x for x in (title, symbol or "", path or "", body[:12000]) if x)


def embed_documents(
    db: EngramDB,
    conn: sqlite3.Connection,
    embedder: Embedder,
    mrl_dim: int,
    doc_ids: Sequence[int],
    rotation: np.ndarray,
    batch: int = 512,
) -> int:
    """Embed, Matryoshka-truncate, quantize (int8 + 1-bit) and persist vectors for the given documents."""
    done = 0
    ids = list(doc_ids)
    for start in range(0, len(ids), batch):
        chunk = ids[start : start + batch]
        marks = ",".join("?" * len(chunk))
        rows = conn.execute(
            f"SELECT id, title, symbol, path, body FROM documents WHERE id IN ({marks})", chunk
        ).fetchall()
        if not rows:
            continue
        texts = [doc_text(r["title"], r["symbol"], r["path"], r["body"]) for r in rows]
        vecs = embedder.embed(texts)
        encoded = encode_vectors(vecs, mrl_dim, rotation)
        conn.executemany(
            "INSERT OR REPLACE INTO vectors(doc_id, bin, i8, i8_scale, f32) VALUES(?,?,?,?,?)",
            [(r["id"], e.bin, e.i8, e.scale, e.f32) for r, e in zip(rows, encoded, strict=True)],
        )
        done += len(rows)
    return done


__all__ = ["doc_text", "embed_documents", "file_document", "load_rotation", "symbol_document"]
