from git_engram.evals.dataset import QUERIES
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
    results = {r["name"]: r for r in payload["results"]}
    ours = results["**git-engram** (full)"]
    for name, r in results.items():
        if r is not ours:
            assert ours["ndcg5"] >= r["ndcg5"], name
            assert ours["cochange"] >= r["cochange"], name
    assert ours["neg_hit"] >= 0.8 and ours["p95_ms"] < 50
    assert payload["preflight"]["p50_ms"] < 20
    assert payload["scale"][0]["recall@10_same_dim"] > 0.8
    assert "| **git-engram** (full) |" in payload["markdown"] and "Ablation study" in payload["markdown"]
