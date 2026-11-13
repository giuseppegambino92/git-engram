import numpy as np
import pytest

from git_engram.embeddings.quantization import (
    binarize,
    dequantize_int8,
    hamming_bit_sliced,
    hamming_distances,
    hamming_distances_reference,
    int8_cosine,
    l2_normalize,
    mrl_truncate,
    packed_width,
    quantize_int8,
    smallest_k_small_ints,
    to_bit_slices,
    top_k_indices,
)
from git_engram.embeddings.vector_index import make_rotation


@pytest.fixture
def rng():
    return np.random.default_rng(42)


def test_mrl_truncation_renormalises(rng):
    x = l2_normalize(rng.standard_normal((50, 384)))
    t = mrl_truncate(x, 128)
    assert t.shape == (50, 128)
    np.testing.assert_allclose(np.linalg.norm(t, axis=1), 1.0, atol=1e-5)
    np.testing.assert_allclose(t, l2_normalize(x[:, :128]), atol=1e-6)
    with pytest.raises(ValueError):
        mrl_truncate(x, 500)


def test_int8_roundtrip_error_is_bounded_by_half_step(rng):
    x = l2_normalize(rng.standard_normal((200, 256)))
    q, scale = quantize_int8(x)
    assert q.dtype == np.int8 and q.min() >= -127 and q.max() <= 127
    err = np.abs(dequantize_int8(q, scale) - x)
    assert np.all(err <= scale[:, None] / 2 + 1e-7)


def test_int8_asymmetric_cosine_matches_float(rng):
    x = l2_normalize(rng.standard_normal((300, 256)))
    q = l2_normalize(rng.standard_normal(256))
    qi, _ = quantize_int8(x)
    approx = int8_cosine(q, qi, np.linalg.norm(qi.astype(np.float32), axis=1))
    np.testing.assert_allclose(approx, x @ q, atol=0.01)


def test_zero_vector_quantizes_safely():
    q, scale = quantize_int8(np.zeros((1, 16)))
    assert np.all(q == 0) and scale[0] == 1.0
    assert int8_cosine(np.zeros(16), q, np.zeros(1))[0] == 0.0


@pytest.mark.parametrize("dim", [8, 64, 100, 256, 384])
def test_binarize_packs_to_64bit_multiple(rng, dim):
    bits = binarize(rng.standard_normal((5, dim)))
    assert bits.shape == (5, packed_width(dim)) and bits.shape[1] % 8 == 0
    unpacked = np.unpackbits(bits, axis=1)[:, :dim]
    assert unpacked.sum() > 0


@pytest.mark.parametrize("n", [1, 3, 64, 1000])
def test_hamming_kernels_match_reference(rng, n):
    db = binarize(rng.standard_normal((n, 256)))
    q = binarize(rng.standard_normal(256))[0]
    ref = hamming_distances_reference(db, q)
    np.testing.assert_array_equal(hamming_distances(db, q), ref)
    np.testing.assert_array_equal(hamming_bit_sliced(to_bit_slices(db), q), ref)


def test_hamming_extremes():
    a = binarize(np.ones((1, 256)))
    b = binarize(-np.ones((1, 256)))
    assert hamming_bit_sliced(to_bit_slices(a), a[0])[0] == 0
    assert hamming_bit_sliced(to_bit_slices(a), b[0])[0] == 256  # needs >uint8 accumulation


def test_counting_select_equals_sort(rng):
    values = rng.integers(0, 257, size=5000).astype(np.uint16)
    for k in (1, 10, 100, 4999, 5000, 6000):
        picked = smallest_k_small_ints(values, k, 257)
        assert len(picked) == min(k, 5000)
        assert sorted(values[picked].tolist()) == sorted(values.tolist())[: min(k, 5000)]
        assert np.all(np.diff(values[picked].astype(int)) >= 0)


def test_top_k_indices_sorted(rng):
    s = rng.standard_normal(1000)
    idx = top_k_indices(s, 10)
    np.testing.assert_array_equal(idx, np.argsort(-s)[:10])
    np.testing.assert_array_equal(top_k_indices(s, 5, largest=False), np.argsort(s)[:5])


def test_rotation_is_orthogonal_and_deterministic():
    r1, r2 = make_rotation(64, seed=3), make_rotation(64, seed=3)
    np.testing.assert_array_equal(r1, r2)
    np.testing.assert_allclose(r1 @ r1.T, np.eye(64), atol=1e-5)


def test_simhash_hamming_estimates_angle(rng):
    """With a random rotation, E[hamming / d] = angle / pi (Charikar 2002)."""
    d = 256
    rot = make_rotation(d, seed=11)
    base = l2_normalize(rng.standard_normal(d))
    for target_cos in (0.9, 0.5, 0.0):
        noise = l2_normalize(rng.standard_normal(d) - base * (rng.standard_normal(d) @ base))
        other = l2_normalize(target_cos * base + np.sqrt(1 - target_cos**2) * noise)
        estimates = []
        for seed in range(30):
            r = make_rotation(d, seed=seed) if seed else rot
            h = hamming_distances_reference(binarize(other[None] @ r), binarize(base[None] @ r)[0])[0]
            estimates.append(h / d)
        expected = np.arccos(target_cos) / np.pi
        assert abs(np.mean(estimates) - expected) < 0.03
