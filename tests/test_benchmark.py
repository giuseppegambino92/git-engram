from git_engram.evals.dataset import DEV_QUERIES as QUERIES
from git_engram.evals.metrics import RankedDoc, first_ranks, ndcg_at, percentile, recall_at, reciprocal_rank


def test_metrics_math():
    ranked = [
        RankedDoc("sym:a::f", "symbol", "a"),
        RankedDoc("commit:x", "commit", "a"),
        RankedDoc("file:b", "file", "b"),
    ]
    keys = {"commit:fix": "commit:x"}
    rel = {"commit:fix": 2, "file:b": 1, "sym:zzz::q": 1}
    found = first_ranks(list(rel), ranked, keys)
    assert found == {"commit:fix": 2, "file:b": 3}
    assert recall_at(found, 3, 2) == 1 / 3 and recall_at(found, 3, 5) == 2 / 3
    assert reciprocal_rank(found) == 0.5
    import math

    dcg = 3 / math.log2(3) + 1 / math.log2(4)
    idcg = 3 / math.log2(2) + 1 / math.log2(3) + 1 / math.log2(4)
    assert abs(ndcg_at(rel, ranked, keys, 5) - dcg / idcg) < 1e-9
    assert percentile([1, 2, 3, 4], 50) == 2.5


def test_dataset_is_balanced():
    cats = {}
    for q in QUERIES:
        cats[q.category] = cats.get(q.category, 0) + 1
        assert q.relevant
    assert len(QUERIES) >= 25 and set(cats.values()) == {8}


def test_benchmark_end_to_end(tmp_path):
    from git_engram.evals.benchmark_runner import run_benchmark

    payload = run_benchmark(workdir=tmp_path, repeats=1, scale=True, sizes=(2000,))
    # Calibration contract: on the dev split the full system must not lose to any baseline.
    dev = {r["name"]: r for r in payload["results_dev"]}
    ours = dev["**git-engram** (full)"]
    for name, r in dev.items():
        assert ours["ndcg5"] >= r["ndcg5"], name
        assert ours["cochange"] >= r["cochange"], name
    assert ours["neg_hit"] >= 0.8
    # Held-out split: only structural checks here (its numbers are reported, never asserted/tuned against).
    test = {r["name"]: r for r in payload["results"]}
    assert set(test) == set(dev) and all(len(r["per_query"]) == 24 for r in test.values())
    assert test["**git-engram** (full)"]["p95_ms"] < 50
    assert payload["preflight"]["p50_ms"] < 20
    assert payload["scale"][0]["recall@10_same_dim"] > 0.8
    md = payload["markdown"]
    assert "Held-out test split" in md and "Dev / calibration split" in md and "query-adaptive router" in md


def test_splits_are_disjoint_and_ground_truth_exists(shopfront):
    """Validates held-out targets against the index WITHOUT running retrieval on them."""
    from git_engram.evals.dataset import DEV_QUERIES, TEST_QUERIES

    assert not {q.qid for q in DEV_QUERIES} & {q.qid for q in TEST_QUERIES}
    assert not {q.text for q in DEV_QUERIES} & {q.text for q in TEST_QUERIES}
    cats = {}
    for q in TEST_QUERIES:
        cats[q.category] = cats.get(q.category, 0) + 1
    assert set(cats.values()) == {6}
    e = shopfront.engine
    keys = {r["doc_key"] for r in e.db.query("SELECT doc_key FROM documents")}
    paths = {r["path"] for r in e.db.query("SELECT path FROM files")}
    for q in DEV_QUERIES + TEST_QUERIES:
        for target in q.relevant:
            if target.startswith("commit:"):
                assert shopfront.info.commit_by_prefix(target[7:]), target
            elif target.startswith("sym:"):
                assert target in keys, target
            else:
                assert target[5:] in paths, target
        if q.coupled:
            assert q.coupled in paths
        for tag in q.failures + q.acceptable:
            assert tag in shopfront.failure_ids
