"""Optional neural backends on raw ONNX Runtime + HuggingFace ``tokenizers`` (no torch, no frameworks).

A model directory must contain ``model.onnx`` and ``tokenizer.json`` (any exported BERT-style encoder,
e.g. nomic-embed-text-v1.5, mxbai-embed-large-v1 or bge-small for embeddings; ms-marco-MiniLM-L-6-v2 for
the cross-encoder). An optional ``engram_model.json`` may declare ``query_prefix``/``document_prefix``
(e.g. ``"search_query: "`` for Nomic) and ``max_length``.

Install with ``pip install 'git-engram[onnx]'``.
"""

from __future__ import annotations

import json
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import numpy as np

from .quantization import l2_normalize


class _OnnxEncoder:
    def __init__(self, model_dir: str | Path, max_length: int | None = None):
        try:
            import onnxruntime as ort
            from tokenizers import Tokenizer
        except ImportError as exc:  # pragma: no cover - exercised only without the extra installed
            raise ImportError("ONNX backend requires `pip install 'git-engram[onnx]'`") from exc
        self.model_dir = Path(model_dir)
        meta_path = self.model_dir / "engram_model.json"
        self.meta: dict[str, Any] = json.loads(meta_path.read_text()) if meta_path.exists() else {}
        self.max_length = int(max_length or self.meta.get("max_length", 512))
        options = ort.SessionOptions()
        options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        self.session = ort.InferenceSession(
            str(self.model_dir / "model.onnx"), options, providers=["CPUExecutionProvider"]
        )
        self.input_names = {i.name for i in self.session.get_inputs()}
        self.tokenizer = Tokenizer.from_file(str(self.model_dir / "tokenizer.json"))
        self.tokenizer.enable_truncation(max_length=self.max_length)
        self.tokenizer.enable_padding()

    def _feeds(self, encodings: list[Any]) -> tuple[dict[str, np.ndarray], np.ndarray]:
        ids = np.array([e.ids for e in encodings], dtype=np.int64)
        mask = np.array([e.attention_mask for e in encodings], dtype=np.int64)
        feeds = {"input_ids": ids, "attention_mask": mask}
        if "token_type_ids" in self.input_names:
            feeds["token_type_ids"] = np.array([e.type_ids for e in encodings], dtype=np.int64)
        return {k: v for k, v in feeds.items() if k in self.input_names}, mask


class OnnxEmbedder(_OnnxEncoder):
    """Mean-pooled sentence embeddings. MRL truncation is applied later by the vector index."""

    kind = "onnx"

    def __init__(self, model_dir: str | Path, max_length: int | None = None):
        super().__init__(model_dir, max_length)
        self.query_prefix = str(self.meta.get("query_prefix", ""))
        self.document_prefix = str(self.meta.get("document_prefix", ""))
        probe = self.embed(["dimension probe"])
        self.dim = int(probe.shape[1])

    def fit(self, texts: Sequence[str]) -> None:
        """Neural encoders are pre-trained; nothing to fit on the corpus."""

    def embed(self, texts: Sequence[str], is_query: bool = False, batch_size: int = 32) -> np.ndarray:
        prefix = self.query_prefix if is_query else self.document_prefix
        chunks: list[np.ndarray] = []
        for start in range(0, len(texts), batch_size):
            batch = [prefix + t for t in texts[start : start + batch_size]]
            feeds, mask = self._feeds(self.tokenizer.encode_batch(batch))
            out = self.session.run(None, feeds)[0]
            if out.ndim == 3:
                m = mask[..., None].astype(np.float32)
                out = (out * m).sum(axis=1) / np.maximum(m.sum(axis=1), 1e-9)
            chunks.append(np.asarray(out, dtype=np.float32))
        if not chunks:
            return np.zeros((0, getattr(self, "dim", 0)), dtype=np.float32)
        return l2_normalize(np.vstack(chunks))

    def embed_query(self, text: str) -> np.ndarray:
        return self.embed([text], is_query=True)[0]

    def term_idf(self, term: str) -> float:
        return 1.0

    def state_bytes(self) -> bytes:
        return json.dumps({"model_dir": str(self.model_dir.resolve())}).encode()

    def load_state(self, blob: bytes) -> None:
        """Stateless beyond the model directory, which is passed at construction."""


class OnnxCrossEncoder(_OnnxEncoder):
    """Pair classifier producing one relevance logit per (query, passage)."""

    name = "onnx-cross-encoder"

    def score(self, query: str, passages: Sequence[str], batch_size: int = 16) -> np.ndarray:
        scores: list[np.ndarray] = []
        for start in range(0, len(passages), batch_size):
            pairs = [(query, p) for p in passages[start : start + batch_size]]
            feeds, _ = self._feeds(self.tokenizer.encode_batch(pairs))
            logits = np.asarray(self.session.run(None, feeds)[0], dtype=np.float32)
            scores.append(logits[:, 0] if logits.ndim == 2 else logits.reshape(-1))
        return np.concatenate(scores) if scores else np.zeros(0, dtype=np.float32)
