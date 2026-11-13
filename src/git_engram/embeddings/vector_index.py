"""Two-stage quantized vector search over SQLite BLOBs held in contiguous NumPy arrays.

Stage 1: Hamming distance on 1-bit packed MRL vectors (XOR + popcount) -> top-C candidates.
         Bits are signs of the vector after a fixed random orthogonal rotation (SimHash / RaBitQ-style):
         rotation spreads energy evenly over all dimensions, so P(bit differs) = angle / pi and the Hamming
         distance is an unbiased estimator of the angle even for spectrally concentrated (e.g. LSA) embeddings.
Stage 2: asymmetric cosine of the float32 query against the int8 MRL vectors of those candidates.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from ..storage.db import EngramDB
from .quantization import (
    binarize,
    hamming_bit_sliced,
    int8_cosine,
    l2_normalize,
    mrl_truncate,
    packed_width,
    quantize_int8,
    smallest_k_small_ints,
    to_bit_slices,
    top_k_indices,
)

KIND_CODES = {"symbol": 0, "file": 1, "commit": 2, "failure": 3, "invariant": 4}


@dataclass(slots=True)
class VectorHit:
    doc_id: int
    hamming: int
    cosine: float


@dataclass(slots=True)
class EncodedVector:
    bin: bytes
    i8: bytes
    scale: float
    f32: bytes


def make_rotation(dim: int, seed: int = 0xE16A) -> np.ndarray:
    """Random orthogonal matrix (QR of a Gaussian matrix with sign-corrected diagonal -> Haar distributed)."""
    rng = np.random.default_rng(seed)
    q, r = np.linalg.qr(rng.standard_normal((dim, dim)))
    return (q * np.sign(np.diag(r))).astype(np.float32)


def encode_vectors(full: np.ndarray, mrl_dim: int, rotation: np.ndarray) -> list[EncodedVector]:
    full = l2_normalize(np.atleast_2d(full))
    trunc = mrl_truncate(full, mrl_dim)
    bits = binarize(trunc @ rotation)
    q, scale = quantize_int8(trunc)
    return [
        EncodedVector(bits[i].tobytes(), q[i].tobytes(), float(scale[i]), full[i].astype(np.float32).tobytes())
        for i in range(full.shape[0])
    ]


def load_rotation(db: EngramDB, mrl_dim: int) -> np.ndarray:
    raw = db.get_meta("bin_rotation")
    if raw is None:
        return make_rotation(mrl_dim)
    rot = np.frombuffer(raw, dtype=np.float32)
    return rot.reshape(mrl_dim, mrl_dim).copy() if rot.size == mrl_dim * mrl_dim else make_rotation(mrl_dim)


class QuantizedVectorIndex:
    def __init__(self, mrl_dim: int, rotation: np.ndarray):
        self.mrl_dim = mrl_dim
        self.rotation = np.asarray(rotation, dtype=np.float32)
        self.doc_ids = np.zeros(0, dtype=np.int64)
        self.kinds = np.zeros(0, dtype=np.int8)
        self.bit_slices = np.zeros((packed_width(mrl_dim) // 8, 0), dtype=np.uint64)
        self.i8 = np.zeros((0, mrl_dim), dtype=np.int8)
        self.i8_norms = np.zeros(0, dtype=np.float32)
        self._f32: np.ndarray | None = None
        self._row_of: dict[int, int] = {}
        self._db: EngramDB | None = None

    @classmethod
    def from_db(cls, db: EngramDB, mrl_dim: int) -> QuantizedVectorIndex:
        index = cls(mrl_dim, load_rotation(db, mrl_dim))
        index._db = db
        rows = db.query(
            "SELECT v.doc_id, d.kind, v.bin, v.i8 FROM vectors v JOIN documents d ON d.id = v.doc_id ORDER BY v.doc_id"
        )
        n = len(rows)
        width = packed_width(mrl_dim)
        index.doc_ids = np.fromiter((r["doc_id"] for r in rows), dtype=np.int64, count=n)
        index.kinds = np.fromiter((KIND_CODES.get(r["kind"], 9) for r in rows), dtype=np.int8, count=n)
        packed = (
            np.frombuffer(b"".join(r["bin"] for r in rows), dtype=np.uint8).reshape(n, width)
            if n
            else np.zeros((0, width), dtype=np.uint8)
        )
        index.bit_slices = to_bit_slices(packed)
        index.i8 = (
            np.frombuffer(b"".join(r["i8"] for r in rows), dtype=np.int8).reshape(n, mrl_dim).copy()
            if n
            else np.zeros((0, mrl_dim), dtype=np.int8)
        )
        index.i8_norms = np.linalg.norm(index.i8.astype(np.float32), axis=1) if n else np.zeros(0, np.float32)
        index._row_of = {int(d): i for i, d in enumerate(index.doc_ids)}
        return index

    def __len__(self) -> int:
        return int(self.doc_ids.shape[0])

    def memory_bytes(self, include_f32: bool = False) -> int:
        total = self.bit_slices.nbytes + self.i8.nbytes + self.i8_norms.nbytes + self.doc_ids.nbytes + self.kinds.nbytes
        if include_f32 and self._f32 is not None:
            total += self._f32.nbytes
        return int(total)

    def _mask(self, kinds: set[str] | None, exclude: set[str] | None) -> np.ndarray | None:
        if not kinds and not exclude:
            return None
        mask = np.ones(len(self), dtype=bool)
        if kinds:
            mask &= np.isin(self.kinds, [KIND_CODES[k] for k in kinds])
        if exclude:
            mask &= ~np.isin(self.kinds, [KIND_CODES[k] for k in exclude])
        return mask

    def search(
        self,
        query_full: np.ndarray,
        top_k: int = 50,
        candidates: int = 100,
        kinds: set[str] | None = None,
        exclude: set[str] | None = None,
    ) -> list[VectorHit]:
        if len(self) == 0:
            return []
        q = mrl_truncate(query_full, self.mrl_dim)
        if not np.any(q):
            return []
        q_bits = binarize(q @ self.rotation)[0]
        dist = hamming_bit_sliced(self.bit_slices, q_bits)
        max_d = self.bit_slices.shape[0] * 64
        mask = self._mask(kinds, exclude)
        if mask is not None:
            dist = np.where(mask, dist, max_d + 1).astype(np.uint16)
            available = int(mask.sum())
        else:
            available = len(self)
        if available == 0:
            return []
        cand = smallest_k_small_ints(dist, min(max(candidates, top_k), available), max_d + 1)
        cos = int8_cosine(q, self.i8[cand], self.i8_norms[cand])
        order = top_k_indices(cos, top_k)
        return [VectorHit(int(self.doc_ids[cand[i]]), int(dist[cand[i]]), float(cos[i])) for i in order]

    def cosine_for(self, query_full: np.ndarray, doc_ids: list[int]) -> dict[int, float]:
        rows = [self._row_of[d] for d in doc_ids if d in self._row_of]
        if not rows:
            return {}
        q = mrl_truncate(query_full, self.mrl_dim)
        cos = int8_cosine(q, self.i8[rows], self.i8_norms[rows])
        return {int(self.doc_ids[r]): float(c) for r, c in zip(rows, cos, strict=True)}

    # -------------------------------------------------------- baselines
    def load_float32(self) -> np.ndarray:
        if self._f32 is None:
            if self._db is None:
                raise RuntimeError("float32 vectors require a DB-backed index")
            rows = self._db.query("SELECT doc_id, f32 FROM vectors ORDER BY doc_id")
            self._f32 = (
                np.vstack([np.frombuffer(r["f32"], dtype=np.float32) for r in rows])
                if rows
                else np.zeros((0, 1), dtype=np.float32)
            )
        return self._f32

    def exact_float32_search(
        self,
        query_full: np.ndarray,
        top_k: int = 50,
        exclude: set[str] | None = None,
    ) -> list[VectorHit]:
        """Naive dense RAG baseline: exhaustive full-dimension float32 cosine."""
        f32 = self.load_float32()
        if f32.shape[0] == 0:
            return []
        q = l2_normalize(np.asarray(query_full, dtype=np.float32))
        sims = f32 @ q
        mask = self._mask(None, exclude)
        if mask is not None:
            sims = np.where(mask, sims, -np.inf)
        order = top_k_indices(sims, top_k)
        return [VectorHit(int(self.doc_ids[i]), -1, float(sims[i])) for i in order if np.isfinite(sims[i])]
