import pytest

from git_engram.retrieval.router import WEIGHTS, classify


@pytest.mark.parametrize(
    "query,klass",
    [
        ("why does the importer skip empty rows", "causal"),
        ("why does parse_invoice strip trailing zeros", "behavioral"),
        ("why is Invoice.total computed lazily", "behavioral"),
        ("why was parse_invoice changed to strip zeros", "causal"),  # history cue wins
        ("why did the InvoiceCache lock get reverted", "causal"),
        ("what is the reason we pin numpy below 2", "causal"),
        ("explain the history of the retry policy", "causal"),
        ("add a timeout parameter to fetch_user", "change"),
        ("rename the Invoice.total field", "change"),
        ("update config/app.yaml with the new region", "change"),  # imperative wins over the path
        ("ERR_QUOTA_EXCEEDED", "exact"),
        ("INC-4410", "exact"),
        ("9f3c2a1b7d0e", "exact"),
        ("ValueError: invalid literal for int()", "exact"),
        ("src/billing/invoice.py", "exact"),
        ("'connection reset by peer'", "exact"),
        ("parseInvoiceLines", "exact"),
        ("users are charged twice when the network drops", "symptom"),
        ("dashboard is slow after the upgrade", "symptom"),
        ("pagination of the admin table", "general"),
    ],
)
def test_classification(query, klass):
    assert classify(query).klass == klass


def test_weights_and_intent():
    exact, symptom, change = classify("E1234"), classify("emails go missing"), classify("add a column")
    assert exact.weights["bm25"] > exact.weights["vec"]
    assert symptom.weights["graph"] < WEIGHTS["general"]["graph"] < change.weights["graph"]
    assert symptom.intent == "causal" and change.intent == "change" and exact.intent == "exact"


def test_router_is_reported_and_can_be_disabled(shopfront):
    from dataclasses import replace

    from git_engram.retrieval.engine import FULL

    e = shopfront.engine
    resp = e.search("ERR_CART_STALE")
    assert resp.query_class == "exact" and resp.fusion_weights["bm25"] > resp.fusion_weights["vec"]
    res = e.retriever.retrieve("ERR_CART_STALE", 5, replace(FULL, route=False))
    assert res.profile is None
