"""Embedding backends, Matryoshka/int8/1-bit quantization and the two-stage vector index."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Protocol

import numpy as np

from ..config import EmbeddingConfig
from .hashing import HashingLSAEmbedder
from .vector_index import QuantizedVectorIndex, VectorHit, encode_vectors, load_rotation, make_rotation


class Embedder(Protocol):
    kind: str
    dim: int

    def fit(self, texts: Sequence[str]) -> None: ...
    def embed(self, texts: Sequence[str], is_query: bool = False) -> np.ndarray: ...
    def embed_query(self, text: str) -> np.ndarray: ...
    def term_idf(self, term: str) -> float: ...
    def state_bytes(self) -> bytes: ...
    def load_state(self, blob: bytes) -> None: ...


def create_embedder(cfg: EmbeddingConfig) -> Embedder:
    if cfg.backend == "hashing-lsa":
        return HashingLSAEmbedder(cfg)
    if cfg.backend == "onnx":
        if not cfg.onnx_model_dir:
            raise ValueError("embedding.backend='onnx' requires embedding.onnx_model_dir")
        from .onnx_backend import OnnxEmbedder

        return OnnxEmbedder(cfg.onnx_model_dir)
    raise ValueError(f"unknown embedding backend {cfg.backend!r}")


__all__ = [
    "Embedder",
    "HashingLSAEmbedder",
    "QuantizedVectorIndex",
    "VectorHit",
    "create_embedder",
    "encode_vectors",
    "load_rotation",
    "make_rotation",
]
