"""Deterministic offline embedder: signed feature hashing + TF-IDF + randomized truncated SVD (LSA).

Why this works as a Matryoshka model: the SVD components are sorted by singular value, i.e. by the
share of corpus variance they explain, so any prefix of the embedding is the best rank-``d``
approximation of the full one (Eckart-Young). Truncating 384d -> 256d -> 128d therefore degrades
gracefully, which is exactly the MRL property neural models are trained to have.

Features per text: stemmed sub-word terms and intact compound identifiers (weight 1) plus boundary-
marked character n-grams (weight ``char_ngram_weight``) which give robustness to morphology
(``expiry`` / ``expired`` / ``expiration``). Features are hashed with CRC32 into ``hash_buckets``
signed buckets, weighted by sublinear TF x smoothed IDF and L2-normalised before projection.
"""

from __future__ import annotations

import io
import math
import zlib
from collections import Counter
from collections.abc import Sequence

import numpy as np

from ..config import EmbeddingConfig
from ..text.analyzer import dense_terms
from .quantization import l2_normalize


class HashingLSAEmbedder:
    kind = "hashing-lsa"

    def __init__(self, cfg: EmbeddingConfig):
        self.cfg = cfg
        self.buckets = cfg.hash_buckets
        self.dim = cfg.full_dim
        self.idf: np.ndarray | None = None
        self.proj: np.ndarray | None = None
        self.rank = 0
        self._hash_cache: dict[str, tuple[int, float]] = {}

    # ------------------------------------------------------------ features
    def _bucket(self, feature: str) -> tuple[int, float]:
        cached = self._hash_cache.get(feature)
        if cached is None:
            h = zlib.crc32(feature.encode("utf-8"))
            sign = 1.0 if (zlib.crc32(feature.encode("utf-8"), 0x9E3779B9) & 1) else -1.0
            cached = (h % self.buckets, sign)
            if len(self._hash_cache) < 500_000:
                self._hash_cache[feature] = cached
        return cached

    def features(self, text: str) -> Counter[str]:
        counts: Counter[str] = Counter()
        n = self.cfg.char_ngram
        for term in dense_terms(text):
            counts["w:" + term] += 1
            if term.isalpha() and len(term) >= n + 1:
                marked = f"<{term}>"
                for i in range(len(marked) - n + 1):
                    counts["c:" + marked[i : i + n]] += 1
        return counts

    def _sparse_tf(self, text: str) -> tuple[np.ndarray, np.ndarray]:
        counts = self.features(text)
        if not counts:
            return np.zeros(0, dtype=np.int64), np.zeros(0, dtype=np.float32)
        acc: dict[int, float] = {}
        cw = self.cfg.char_ngram_weight
        for feature, tf in counts.items():
            idx, sign = self._bucket(feature)
            weight = (1.0 + math.log(tf)) * (cw if feature[0] == "c" else 1.0)
            acc[idx] = acc.get(idx, 0.0) + sign * weight
        idx = np.fromiter(acc.keys(), dtype=np.int64, count=len(acc))
        val = np.fromiter(acc.values(), dtype=np.float32, count=len(acc))
        return idx, val

    def _tfidf_matrix(self, texts: Sequence[str]) -> np.ndarray:
        assert self.idf is not None
        x = np.zeros((len(texts), self.buckets), dtype=np.float32)
        for row, text in enumerate(texts):
            idx, val = self._sparse_tf(text)
            if idx.size:
                x[row, idx] = val * self.idf[idx]
        return l2_normalize(x)

    # ----------------------------------------------------------------- fit
    def fit(self, texts: Sequence[str]) -> None:
        cfg = self.cfg
        rng = np.random.default_rng(cfg.seed)
        n_docs = len(texts)
        df = np.zeros(self.buckets, dtype=np.float64)
        for text in texts:
            idx, _ = self._sparse_tf(text)
            df[idx] += 1.0
        self.idf = (np.log((1.0 + n_docs) / (1.0 + df)) + 1.0).astype(np.float32)

        if n_docs > cfg.svd_sample:
            sample_idx = np.sort(rng.choice(n_docs, size=cfg.svd_sample, replace=False))
            sample = [texts[i] for i in sample_idx]
        else:
            sample = list(texts)
        x = self._tfidf_matrix(sample) if sample else np.zeros((1, self.buckets), dtype=np.float32)

        k = max(1, min(self.dim, x.shape[0]))
        width = min(k + cfg.svd_oversample, x.shape[0], self.buckets)
        omega = rng.standard_normal((self.buckets, width)).astype(np.float32)
        y = x @ omega
        q, _ = np.linalg.qr(y)
        for _ in range(cfg.svd_power_iters):
            z, _ = np.linalg.qr(x.T @ q)
            q, _ = np.linalg.qr(x @ z)
        b = q.T @ x
        _, s, vt = np.linalg.svd(b, full_matrices=False)
        keep = int(np.sum(s > max(float(s[0]) * 1e-5, 1e-9))) if s.size else 0
        k_eff = min(k, keep)
        vt = vt[:k_eff]
        # Deterministic sign convention (svd_flip): largest |loading| of every component is positive.
        signs = np.sign(vt[np.arange(k_eff), np.argmax(np.abs(vt), axis=1)])
        signs[signs == 0] = 1.0
        vt = vt * signs[:, None]
        proj = np.zeros((self.buckets, self.dim), dtype=np.float32)
        proj[:, :k_eff] = vt.T
        self.proj = proj
        self.rank = k_eff

    # --------------------------------------------------------------- embed
    def embed(self, texts: Sequence[str], is_query: bool = False, batch_size: int = 512) -> np.ndarray:
        if self.proj is None or self.idf is None:
            raise RuntimeError("HashingLSAEmbedder.embed called before fit()/load()")
        out = np.zeros((len(texts), self.dim), dtype=np.float32)
        for start in range(0, len(texts), batch_size):
            chunk = texts[start : start + batch_size]
            out[start : start + len(chunk)] = self._tfidf_matrix(chunk) @ self.proj
        return l2_normalize(out)

    def embed_query(self, text: str) -> np.ndarray:
        """Single-query fast path: gather only the projection rows of the non-zero buckets."""
        if self.proj is None or self.idf is None:
            raise RuntimeError("HashingLSAEmbedder.embed_query called before fit()/load()")
        idx, val = self._sparse_tf(text)
        if idx.size == 0:
            return np.zeros(self.dim, dtype=np.float32)
        w = val * self.idf[idx]
        w /= max(float(np.linalg.norm(w)), 1e-12)
        v = w @ self.proj[idx]
        return l2_normalize(v)

    def term_idf(self, term: str) -> float:
        if self.idf is None:
            return 1.0
        idx, _ = self._bucket("w:" + term)
        return float(self.idf[idx])

    # --------------------------------------------------------------- state
    def state_bytes(self) -> bytes:
        if self.proj is None or self.idf is None:
            raise RuntimeError("cannot serialise an unfitted embedder")
        buf = io.BytesIO()
        np.savez_compressed(buf, idf=self.idf, proj=self.proj.astype(np.float16), rank=np.array([self.rank]))
        return buf.getvalue()

    def load_state(self, blob: bytes) -> None:
        data = np.load(io.BytesIO(blob))
        self.idf = data["idf"].astype(np.float32)
        self.proj = data["proj"].astype(np.float32)
        self.rank = int(data["rank"][0])
        self.buckets, self.dim = self.proj.shape
