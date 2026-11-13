import numpy as np

from git_engram.config import EmbeddingConfig
from git_engram.documents import embed_documents
from git_engram.embeddings import HashingLSAEmbedder, QuantizedVectorIndex, make_rotation
from git_engram.storage.db import EngramDB

CORPUS = [
    "def is_token_expired(token, now): return now - CLOCK_SKEW_SECONDS >= token['exp']",
    "fix(auth): tolerate clock skew when validating token expiry; mobile clients were logged out",
    "def charge_with_retry(client, order): reuse the idempotency key across retries",
    "fix(payments): customers were charged twice when the gateway timed out",
    "class TTLCache: def get(self, key): with self._lock: return self._data.get(key)",
    "Revert perf(cache): skip the lock -- race condition corrupted cache entries",
    "export interface Order { totalCents: number; currency: string }",
    "func (c *Consumer) HandleOne(ctx context.Context) error { ack after handler }",
] * 3


def fitted() -> HashingLSAEmbedder:
    emb = HashingLSAEmbedder(EmbeddingConfig(full_dim=64, mrl_dim=32, hash_buckets=1024))
    emb.fit(CORPUS)
    return emb


def test_embedder_is_deterministic_and_normalised():
    a, b = fitted(), fitted()
    va, vb = a.embed(CORPUS), b.embed(CORPUS)
    np.testing.assert_allclose(va, vb, atol=1e-6)
    np.testing.assert_allclose(np.linalg.norm(va, axis=1), 1.0, atol=1e-5)


def test_query_fast_path_matches_batch_path():
    emb = fitted()
    for q in ("clock skew token", "double charge payment retry", "unknownzzz"):
        np.testing.assert_allclose(emb.embed_query(q), emb.embed([q], is_query=True)[0], atol=1e-5)


def test_semantic_neighbourhood():
    emb = fitted()
    docs = emb.embed(CORPUS[:8])
    sims = docs @ emb.embed_query("token expired because of clock skew")
    assert int(np.argmax(sims)) in (0, 1)
    sims = docs @ emb.embed_query("payment charged twice on retry")
    assert int(np.argmax(sims)) in (2, 3)


def test_matryoshka_prefixes_carry_most_energy():
    """SVD components are ordered by singular value, so leading dims dominate (the MRL property)."""
    import random

    rng = random.Random(5)
    vocab = " ".join(CORPUS).replace("(", " ").replace(")", " ").split()
    corpus = [" ".join(rng.choice(vocab) for _ in range(30)) for _ in range(120)]
    emb = HashingLSAEmbedder(EmbeddingConfig(full_dim=64, mrl_dim=32, hash_buckets=1024))
    emb.fit(corpus)
    v = emb.embed(corpus)
    energy = (v**2).sum(axis=0)
    assert energy[:8].sum() > energy[8:16].sum() > energy[32:40].sum()


def test_state_roundtrip():
    emb = fitted()
    clone = HashingLSAEmbedder(EmbeddingConfig(full_dim=64, mrl_dim=32, hash_buckets=1024))
    clone.load_state(emb.state_bytes())
    np.testing.assert_allclose(clone.embed(CORPUS[:4]), emb.embed(CORPUS[:4]), atol=2e-3)  # float16 storage


def test_two_stage_index_agrees_with_exact_search(tmp_path):
    emb = HashingLSAEmbedder(EmbeddingConfig(full_dim=64, mrl_dim=32, hash_buckets=1024))
    emb.fit(CORPUS)
    db = EngramDB(tmp_path / "e.db")
    rot = make_rotation(32)
    db.set_meta("bin_rotation", rot.tobytes())
    with db.transaction() as conn:
        ids = [
            db.upsert_document(conn, f"sym:d{i}", "symbol", f"f{i}.py", f"s{i}", f"t{i}", text, 0.0)
            for i, text in enumerate(CORPUS[:8])
        ]
        embed_documents(db, conn, emb, 32, ids, rot)
    index = QuantizedVectorIndex.from_db(db, 32)
    assert len(index) == 8 and index.memory_bytes() < index.load_float32().nbytes
    q = emb.embed_query("idempotency key retry charge")
    two = index.search(q, top_k=3, candidates=8)
    exact = index.exact_float32_search(q, top_k=3)
    assert two[0].doc_id == exact[0].doc_id
    assert {h.doc_id for h in two} == {h.doc_id for h in exact}
    assert index.search(q, 3, kinds={"commit"}) == []
    db.close()
