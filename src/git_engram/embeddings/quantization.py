"""Matryoshka truncation, int8 scalar quantization and 1-bit binary quantization in pure NumPy.

Storage layout per vector (defaults: full_dim=384, mrl_dim=256):

* ``f32``  -- full-dimension float32 embedding (1536 B). Kept on disk only, used by baselines/re-quantization.
* ``i8``   -- MRL-truncated, re-normalised vector, symmetric per-vector int8 (256 B).
* ``bin``  -- MRL-truncated vector, randomly rotated, 1 bit/dim (sign), packed (32 B).

Hamming distance is computed as ``popcount(xor(a, b))`` over packed uint64 words, which NumPy >= 2.0
exposes as the vectorised ``np.bitwise_count`` ufunc (a byte look-up table is used on older NumPy).
"""

from __future__ import annotations

import numpy as np

_POPCOUNT_LUT = np.array([bin(i).count("1") for i in range(256)], dtype=np.uint8)
HAS_BITWISE_COUNT = hasattr(np, "bitwise_count")


def l2_normalize(x: np.ndarray, axis: int = -1, eps: float = 1e-12) -> np.ndarray:
    x = np.asarray(x, dtype=np.float32)
    norms = np.linalg.norm(x, axis=axis, keepdims=True)
    return x / np.maximum(norms, eps)


def mrl_truncate(x: np.ndarray, dim: int) -> np.ndarray:
    """Matryoshka truncation: keep the leading ``dim`` coordinates and re-normalise to the unit sphere."""
    x = np.asarray(x, dtype=np.float32)
    if dim > x.shape[-1]:
        raise ValueError(f"mrl dim {dim} exceeds embedding dim {x.shape[-1]}")
    return l2_normalize(x[..., :dim])


def quantize_int8(x: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Symmetric per-vector scalar quantization: x ~= q * scale with q in [-127, 127]."""
    x = np.atleast_2d(np.asarray(x, dtype=np.float32))
    scale = np.max(np.abs(x), axis=1) / 127.0
    scale = np.where(scale == 0, 1.0, scale).astype(np.float32)
    q = np.clip(np.rint(x / scale[:, None]), -127, 127).astype(np.int8)
    return q, scale


def dequantize_int8(q: np.ndarray, scale: np.ndarray) -> np.ndarray:
    return np.atleast_2d(q).astype(np.float32) * np.asarray(scale, dtype=np.float32)[:, None]


def packed_width(dim: int) -> int:
    """Bytes per packed binary vector, padded to a multiple of 8 so rows view cleanly as uint64."""
    return ((dim + 63) // 64) * 8


def binarize(x: np.ndarray, threshold: np.ndarray | None = None) -> np.ndarray:
    """1-bit quantization: bit_j = x_j > threshold_j, packed big-endian into uint8 (zero padded to 64 bits)."""
    x = np.atleast_2d(np.asarray(x, dtype=np.float32))
    thr = np.zeros(x.shape[1], dtype=np.float32) if threshold is None else np.asarray(threshold, dtype=np.float32)
    bits = np.packbits(x > thr, axis=1)
    width = packed_width(x.shape[1])
    if bits.shape[1] < width:
        bits = np.pad(bits, ((0, 0), (0, width - bits.shape[1])))
    return np.ascontiguousarray(bits, dtype=np.uint8)


def hamming_distances(db_bits: np.ndarray, q_bits: np.ndarray) -> np.ndarray:
    """Hamming distance between one packed query and every packed row: popcount(db XOR q)."""
    db_bits = np.ascontiguousarray(db_bits, dtype=np.uint8)
    q_bits = np.ascontiguousarray(np.asarray(q_bits, dtype=np.uint8).reshape(-1))
    if db_bits.shape[0] == 0:
        return np.zeros(0, dtype=np.int32)
    if db_bits.shape[1] % 8 == 0:
        words = db_bits.view(np.uint64)
        qwords = q_bits.view(np.uint64)
        xored = np.bitwise_xor(words, qwords)
    else:
        xored = np.bitwise_xor(db_bits, q_bits)
    if HAS_BITWISE_COUNT:
        return np.bitwise_count(xored).sum(axis=1, dtype=np.int32)
    as_bytes = xored.view(np.uint8).reshape(db_bits.shape[0], -1)
    return _POPCOUNT_LUT[as_bytes].sum(axis=1, dtype=np.int32)


def to_bit_slices(packed: np.ndarray) -> np.ndarray:
    """Row-major packed bits (n, bytes) -> bit-sliced column-major uint64 words (n_words, n).

    Each row of the result is one contiguous 64-bit lane across *all* vectors, so XOR/popcount run as long,
    SIMD-friendly 1-D loops instead of iterating a tiny inner dimension (~13x faster at 100k vectors).
    """
    packed = np.ascontiguousarray(packed, dtype=np.uint8)
    if packed.shape[1] % 8:
        packed = np.pad(packed, ((0, 0), (0, 8 - packed.shape[1] % 8)))
    return np.ascontiguousarray(packed.view(np.uint64).T)


def hamming_bit_sliced(slices: np.ndarray, q_bits: np.ndarray) -> np.ndarray:
    """Hamming distances (uint16) of a packed query against bit-sliced database lanes."""
    n_words, n = slices.shape
    qwords = np.ascontiguousarray(np.asarray(q_bits, dtype=np.uint8).reshape(-1)).view(np.uint64)
    acc = np.zeros(n, dtype=np.uint16)
    if n == 0:
        return acc
    buf = np.empty(n, dtype=np.uint64)
    cnt = np.empty(n, dtype=np.uint8)
    for j in range(n_words):
        np.bitwise_xor(slices[j], qwords[j], out=buf)
        if HAS_BITWISE_COUNT:
            np.bitwise_count(buf, out=cnt)
        else:
            cnt[:] = _POPCOUNT_LUT[buf.view(np.uint8).reshape(n, 8)].sum(axis=1, dtype=np.uint8)
        np.add(acc, cnt, out=acc, casting="unsafe")
    return acc


def smallest_k_small_ints(values: np.ndarray, k: int, max_value: int) -> np.ndarray:
    """Indices of the k smallest small-integer values via counting select (O(n), no comparison sort).

    Builds the histogram of distances, finds the cut-off distance whose cumulative count reaches k, and
    takes every index strictly below it plus enough ties at the cut-off. Result is sorted by value.
    """
    n = values.shape[0]
    if n == 0 or k <= 0:
        return np.zeros(0, dtype=np.int64)
    if k >= n:
        return np.argsort(values, kind="stable")
    hist = np.bincount(values, minlength=max_value + 1)
    cut = int(np.searchsorted(np.cumsum(hist), k))
    below = np.flatnonzero(values < cut)
    ties = np.flatnonzero(values == cut)[: k - below.shape[0]]
    picked = np.concatenate([below, ties])
    return picked[np.argsort(values[picked], kind="stable")]


def hamming_distances_reference(db_bits: np.ndarray, q_bits: np.ndarray) -> np.ndarray:
    """Slow but obviously-correct reference implementation (unpackbits + sum), used in tests."""
    a = np.unpackbits(np.atleast_2d(db_bits), axis=1)
    b = np.unpackbits(np.asarray(q_bits, dtype=np.uint8).reshape(1, -1), axis=1)
    return (a != b).sum(axis=1).astype(np.int32)


def int8_cosine(query: np.ndarray, db_i8: np.ndarray, db_norms: np.ndarray) -> np.ndarray:
    """Asymmetric cosine: float32 query against int8 documents. The per-vector scale cancels out."""
    q = np.asarray(query, dtype=np.float32).reshape(-1)
    qn = float(np.linalg.norm(q))
    if qn == 0.0 or db_i8.shape[0] == 0:
        return np.zeros(db_i8.shape[0], dtype=np.float32)
    dots = db_i8.astype(np.float32) @ q
    return dots / (np.maximum(db_norms, 1e-12) * qn)


def top_k_indices(scores: np.ndarray, k: int, largest: bool = True) -> np.ndarray:
    """Indices of the k best scores, sorted best-first (argpartition then a small argsort)."""
    n = scores.shape[0]
    if n == 0 or k <= 0:
        return np.zeros(0, dtype=np.int64)
    k = min(k, n)
    keyed = -scores if largest else scores
    if k < n:
        part = np.argpartition(keyed, k - 1)[:k]
    else:
        part = np.arange(n)
    return part[np.argsort(keyed[part], kind="stable")]
