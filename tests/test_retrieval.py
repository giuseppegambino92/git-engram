import math

import pytest

from git_engram.retrieval.engine import BM25_ONLY, DENSE_F32, FULL, HYBRID
from git_engram.retrieval.fusion import recency_factor, td_rrf
from git_engram.retrieval.graph import CoChangeGraph, Neighbor
from git_engram.retrieval.lexical import fts_match_expression


def test_td_rrf_math():
    fused = td_rrf({"bm25": [1, 2], "vec": [2, 3]}, {"bm25": 1.0, "vec": 2.0}, k=60)
    scores = {d: s for d, s, _ in fused}
    assert scores[1] == pytest.approx(1 / 61)
    assert scores[2] == pytest.approx(1 / 62 + 2 / 61)
    assert scores[3] == pytest.approx(2 / 62)
    assert [d for d, _, _ in fused] == [2, 3, 1]


def test_time_decay_boost():
    lam = math.log(2) / 100.0
    assert recency_factor(100.0, 100.0, 0.5, lam) == pytest.approx(1.5)
    assert recency_factor(0.0, 100.0, 0.5, lam) == pytest.approx(1.25)
    assert recency_factor(None, 100.0, 0.5, lam) == 1.0
    fused = td_rrf({"a": [1, 2]}, {"a": 1.0}, 60, {1: 0.0, 2: 1000.0}, 1000.0, 0.5, lam)
    assert [d for d, _, _ in fused] == [2, 1]  # recent doc overtakes despite worse rank


def test_spreading_activation_with_static_discount():
    g = CoChangeGraph(
        {
            "file": {
                "a": [Neighbor("b", 0.8, 1.0, 4, 0.0, False), Neighbor("c", 0.8, 1.0, 4, 0.0, True)],
                "x": [Neighbor("b", 0.5, 1.0, 2, 0.0, False)],
            },
            "symbol": {},
        }
    )
    act = g.spread({"a": 1.0, "x": 0.5}, "file", decay=0.5, min_weight=0.1, static_discount=0.5)
    assert act["b"][0] == pytest.approx(0.5 * 1.0 * 0.8 + 0.5 * 0.5 * 0.5)
    assert act["b"][1] == "a"
    assert act["c"][0] == pytest.approx(0.5 * 0.8 * 0.5)


def test_fts_expression_quotes_terms():
    assert fts_match_expression(['a"b', "c"]) == '"a""b" OR "c"'


@pytest.mark.parametrize(
    "query,expected_key_part",
    [
        ("ERR_CART_STALE", "cart/service.py::CartStaleError"),
        ("E4012", "gateway.py::user_message_for"),
        ("why does TTLCache.get take the lock", "Revert"),
    ],
)
def test_full_mode_finds_expected(shopfront, query, expected_key_part):
    resp = shopfront.engine.search(query, top_k=5)
    keys = [h.doc_key for h in resp.hits] + [h.title for h in resp.hits]
    assert any(expected_key_part in k for k in keys), keys


def test_failures_are_routed_to_negative_channel(shopfront):
    # Query with unambiguous evidence; paraphrased symptoms can sit near the 0.40 threshold (see README limitations).
    resp = shopfront.engine.search("generate a fresh idempotency key for every payment retry attempt", top_k=8)
    assert all(h.kind != "failure" for h in resp.hits)
    assert shopfront.failure_ids["F2-idempotency"] in {w.failure_id for w in resp.warnings}
    assert any(i.rule_id == "INV-PAY-001" for i in resp.invariants)


def test_score_breakdown_and_latency_budget(shopfront):
    e = shopfront.engine
    e.search("clock skew token expiry")  # warm caches
    resp = e.search("clock skew token expiry", top_k=5)
    top = resp.hits[0]
    assert top.bm25_rank is not None and top.vec_rank is not None and top.rerank_score is not None
    assert top.hamming is not None and 0 <= top.hamming <= 256
    assert resp.timings_ms["end_to_end"] < 50.0  # generous CI bound; typical warm latency is 1-5 ms


@pytest.mark.parametrize("mode", [BM25_ONLY, DENSE_F32, HYBRID, FULL])
def test_all_modes_return_results(shopfront, mode):
    res = shopfront.engine.retriever.retrieve("idempotency key retry", 5, mode)
    assert 0 < len(res.hits) <= 5


def test_graph_path_surfaces_hidden_coupled_file(shopfront):
    res = shopfront.engine.retriever.retrieve("tune the exponential Backoff delays of the queue worker", 10, FULL)
    graph_hits = {h.path for h in res.hits if h.graph_rank}
    assert "worker/config/worker.yaml" in graph_hits


def test_file_hint_seeds_coupling(shopfront):
    resp = shopfront.engine.search("add a field", top_k=5, file="backend/api/schemas.py")
    surfaced = {c.target for c in resp.coupling} | {h.path for h in resp.hits}
    assert {"mobile/src/parsers/orderParser.ts", "web/src/api/orderTypes.ts"} <= surfaced


def test_slot_budget_caps_commits_and_attaches_them_to_symbols(shopfront):
    from dataclasses import replace

    from git_engram.retrieval.engine import FULL

    e = shopfront.engine
    q = "add a discount_cents field to the order schema returned by the API"  # router class: change -> 0 commits
    capped = e.retriever.retrieve(q, 8, FULL)
    assert capped.profile.klass == "change"
    assert all(h.kind != "commit" for h in capped.hits[:5])
    uncapped = e.retriever.retrieve(q, 8, replace(FULL, slot_budget=False))
    assert {h.doc_key for h in uncapped.hits} != {h.doc_key for h in capped.hits} or uncapped.hits[0].kind != "commit"
    schema = next(h for h in capped.hits if h.symbol == "OrderSchema")
    assert schema.attached_commits and all("hash" in c and "subject" in c for c in schema.attached_commits)

    why = e.retriever.retrieve("why does TTLCache.get take the lock even though it only reads", 8, FULL)
    assert why.profile.klass == "behavioral"  # names a function, no history cue -> code-first
    assert why.hits[0].doc_key == "sym:backend/cache/lru.py::TTLCache.get"
    assert sum(h.kind == "commit" for h in why.hits[:5]) <= e.cfg.retrieval.commit_slots["behavioral"]
    assert any(h.kind == "commit" for h in why.hits[:5])  # the load-bearing fix commit stays visible
    hist = e.retriever.retrieve("why did the TTLCache.get lock get restored in the end", 8, FULL)
    assert hist.profile.klass == "causal"
    assert sum(h.kind == "commit" for h in hist.hits[:5]) <= e.cfg.retrieval.commit_slots["causal"]
    assert any(h.kind == "commit" for h in hist.hits[:5])


def test_slot_budget_defers_instead_of_dropping(shopfront):
    r = shopfront.engine.retriever
    commits = [d for d, m in r.meta.items() if m.kind == "commit"][:4]
    code = [d for d, m in r.meta.items() if m.kind == "symbol"][:4]
    order = commits + code
    out = r._slot_budget(order, "symptom")  # cap 1 commit in the first 5
    assert sorted(out) == sorted(order)
    assert sum(r.meta[d].kind == "commit" for d in out[:5]) == 1
    assert out[:5] == [commits[0], *code] and out[5:] == commits[1:]


def test_graveyard_anchors_on_symbols_not_files(shopfront):
    e = shopfront.engine
    f1 = shopfront.failure_ids["F1-clock-skew"]  # targets token.py::is_token_expired
    resp = e.search("how does issue_token build and sign the session payload", top_k=8)
    assert f1 not in {w.failure_id for w in resp.warnings}  # sibling function in the same file
    assert all("targets retrieved file" not in w.match_reason for w in resp.warnings)
    # The failing function itself still triggers the warning through the retrieved symbol.
    resp = e.search("simplify is_token_expired by removing CLOCK_SKEW_SECONDS", top_k=8)
    assert f1 in {w.failure_id for w in resp.warnings}


def test_module_level_failures_fall_back_to_file_anchoring(shopfront):
    e = shopfront.engine
    f11 = shopfront.failure_ids["F11-ttl-mismatch"]  # diff sits on a module-level constant
    assert ("backend/inventory/reservations.py", "<module>") in e.graveyard.targets_of(f11)
    resp = e.search("change RESERVATION_TTL_SECONDS in the inventory reservations module", top_k=8)
    reasons = {w.failure_id: w.match_reason for w in resp.warnings}
    assert f11 in reasons


def test_bulk_commits_contribute_no_context(shopfront):
    from git_engram.models import Hit

    e = shopfront.engine

    def hit(key, kind, path=None, symbol=None, score=1.0):
        return Hit(
            doc_id=0,
            doc_key=key,
            kind=kind,
            path=path,
            symbol=symbol,
            title="",
            snippet="",
            ts=None,
            score=score,
            rerank_score=score,
        )

    bootstrap = shopfront.info.commit_by_prefix("feat: bootstrap shopfront")
    assert e._retrieved_context([hit(f"commit:{bootstrap}", "commit")], 3) == (set(), set())
    focused = shopfront.info.commit_by_prefix("fix(auth): tolerate 30s clock skew")
    code = hit(
        "sym:backend/auth/token.py::is_token_expired", "symbol", "backend/auth/token.py", "is_token_expired", 0.5
    )
    symbols, _ = e._retrieved_context([hit(f"commit:{focused}", "commit"), code], 3)
    assert ("backend/auth/token.py", "is_token_expired") in symbols


def test_commit_relevance_gate_blocks_sibling_commits(shopfront):
    """A sibling function's fix commit must not credit that sibling's failure (TTLCache.set vs the get() revert)."""
    from git_engram.models import Hit

    e = shopfront.engine
    revert = shopfront.info.commit_by_prefix('Revert "perf(cache)')

    def hit(key, kind, path=None, symbol=None, score=1.0):
        return Hit(
            doc_id=0,
            doc_key=key,
            kind=kind,
            path=path,
            symbol=symbol,
            title="",
            snippet="",
            ts=None,
            score=score,
            rerank_score=score,
        )

    asked = hit("sym:backend/cache/lru.py::TTLCache.set", "symbol", "backend/cache/lru.py", "TTLCache.set", 1.0)
    hits = [asked, hit(f"commit:{revert}", "commit", score=0.95)]
    symbols, _ = e._retrieved_context(hits, 3)
    assert ("backend/cache/lru.py", "TTLCache.get") not in symbols  # get() is not among the top code hits
    e.cfg.graveyard.commit_gating = False
    try:
        ungated, _ = e._retrieved_context(hits, 3)
    finally:
        e.cfg.graveyard.commit_gating = True
    assert ("backend/cache/lru.py", "TTLCache.get") in ungated
    weak = [
        asked,
        hit(f"commit:{revert}", "commit", score=0.2),
        hit("sym:backend/cache/lru.py::TTLCache.get", "symbol", "backend/cache/lru.py", "TTLCache.get", 0.9),
    ]
    symbols, _ = e._retrieved_context(weak, 3)
    assert ("backend/cache/lru.py", "TTLCache.get") in symbols  # still credited through the direct symbol hit


def test_named_symbol_is_pinned_for_code_first_queries(shopfront):
    from git_engram.retrieval.engine import FULL

    r = shopfront.engine.retriever
    res = r.retrieve("why does TTLCache.set evict entries when it is full", 8, FULL)
    assert res.profile.klass == "behavioral"
    assert res.hits[0].doc_key == "sym:backend/cache/lru.py::TTLCache.set"
    assert r.named_symbols("why does the cache get slow") == []  # generic words never pin
    assert r.named_symbols("look at sign_payload") == [r.by_key["sym:backend/auth/token.py::sign_payload"]]


def test_preflight_sibling_vs_file_level(shopfront):
    e = shopfront.engine
    f1 = shopfront.failure_ids["F1-clock-skew"]
    sibling = e.preflight("backend/auth/token.py", "issue_token")
    assert f1 not in {w.failure_id for w in sibling.warnings}
    whole_file = e.preflight("backend/auth/token.py")
    assert f1 in {w.failure_id for w in whole_file.warnings}


def test_mmr_is_opt_in():
    from git_engram.config import RetrievalConfig
    from git_engram.retrieval.engine import FULL

    assert FULL.diversify is False and RetrievalConfig().mmr is False
