import pytest

from git_engram.indexer.ast_chunker import (
    chunk_fallback,
    chunk_python,
    chunk_source,
    chunk_tree_sitter,
    detect_language,
    find_block_end,
    symbols_overlapping,
    tree_sitter_available,
)
from git_engram.text.analyzer import analyze, analyze_query, dense_terms, is_rare_token, split_identifier


def test_identifier_expansion():
    assert analyze("is_token_expired") == ["is_token_expired", "is", "token", "expired"]
    assert analyze("parseHTTPResponse") == ["parsehttpresponse", "parse", "http", "response"]
    assert analyze("INC-2291") == ["inc_2291", "inc", "2291"]
    assert analyze("ERR_CART_STALE") == ["err_cart_stale", "err", "cart", "stale"]
    assert split_identifier("XMLHttpRequest") == ("xml", "http", "request")


def test_commit_hash_keeps_short_form():
    h = "9f3c2a1b7d0e4f5a6b7c8d9e0f1a2b3c4d5e6f7a"
    assert analyze(h) == [h, "9f3c2a1"]
    assert analyze("9f3c2a1") == ["9f3c2a1"]


def test_query_analysis_drops_stopwords_and_dedupes():
    assert analyze_query("Why does the cache take the lock?") == ["cache", "take", "lock"]
    assert analyze_query("token token TOKEN") == ["token"]


def test_rare_tokens():
    for tok in ("ERR_CART_STALE", "E4012", "INC-2291", "camelCase", "CartService", "RuntimeError"):
        assert is_rare_token(tok), tok
    for tok in ("the", "cache", "Order", "fix"):
        assert not is_rare_token(tok), tok


def test_dense_terms_stem_and_filter_noise():
    terms = dense_terms("def expired(self): return self.expiring")
    assert "expir" in terms and "self" not in terms and "def" not in terms


PY = '''"""Module doc."""
import os
from a.b import c

LIMIT = 3


@decorator
def top(x: int) -> int:
    """Top doc."""
    return x


class Box:
    """Box doc."""
    size = 1

    def get(self, key):
        return key

    async def put(self, key,
                  value):
        return value
'''


def test_python_chunker_symbols_and_module_chunk():
    res = chunk_python("pkg/mod.py", PY)
    by = {s.qualname: s for s in res.symbols}
    assert set(by) == {"<module>", "top", "Box", "Box.get", "Box.put"}
    assert by["top"].start_line == 8 and by["top"].docstring == "Top doc." and by["top"].kind == "function"
    assert by["Box.get"].kind == "method" and by["Box.get"].parent == "Box"
    assert by["Box.put"].signature.startswith("async def put(self, key,") and by["Box.put"].signature.endswith(":")
    assert "LIMIT = 3" in by["<module>"].content and "def top" not in by["<module>"].content
    assert "size = 1" in by["Box"].content and "return key" not in by["Box"].content
    assert {"os", "a.b", "c"} <= set(res.imports)


def test_symbols_overlapping_prefers_most_specific():
    syms = chunk_python("m.py", PY).symbols
    names = lambda rs: sorted(s.qualname for s in symbols_overlapping(syms, rs))  # noqa: E731
    assert names([(18, 18)]) == ["Box.get"]
    assert names([(16, 16)]) == ["Box"]
    assert names([(5, 5)]) == ["<module>"]
    assert names([(10, 11), (18, 19)]) == ["Box.get", "top"]
    assert names([(10, 19)]) == ["<module>", "Box", "Box.get", "top"]  # blank lines 12-13 are module-level


TS = """import { http } from "./http";

// Parses an order.
export function parseOrder(json: any): Order {
  const s = "}{ not a brace";
  return { total: json.total }; /* } */
}

export const toCents = (x: number): number => Math.round(x * 100);

export class OrderStore {
  private items: Order[] = [];
  add(order: Order): void {
    this.items.push(order);
  }
}

export interface Order { total: number }
"""


def test_typescript_fallback_parser():
    res = chunk_fallback("web/order.ts", TS, "typescript")
    by = {s.qualname: s for s in res.symbols}
    assert {"parseOrder", "toCents", "OrderStore", "OrderStore.add", "Order"} <= set(by)
    assert by["parseOrder"].start_line == 4 and by["parseOrder"].end_line == 7
    assert by["parseOrder"].docstring == "Parses an order."
    assert by["OrderStore.add"].kind == "method"
    assert "./http" in res.imports


@pytest.mark.skipif(not tree_sitter_available("typescript"), reason="tree-sitter typescript grammar not cached")
def test_tree_sitter_matches_fallback_qualnames():
    ts = chunk_tree_sitter("web/order.ts", TS, "typescript")
    fb = chunk_fallback("web/order.ts", TS, "typescript")
    assert ts is not None
    assert {s.qualname for s in ts.symbols} >= {s.qualname for s in fb.symbols} - {"<module>"}


GO = """package queue

// Consumer pulls messages.
type Consumer struct {
	broker Broker
}

// HandleOne processes one message.
func (c *Consumer) HandleOne(ctx context.Context) error {
	if x := "}"; x != "" {
		return nil
	}
	return nil
}

func NewConsumer() *Consumer { return &Consumer{} }
"""


def test_go_receiver_methods_are_qualified():
    fb = {s.qualname for s in chunk_fallback("q.go", GO, "go").symbols}
    assert {"Consumer", "Consumer.HandleOne", "NewConsumer"} <= fb
    if tree_sitter_available("go"):
        ts = {s.qualname for s in chunk_tree_sitter("q.go", GO, "go").symbols}
        assert {"Consumer", "Consumer.HandleOne", "NewConsumer"} <= ts


RUST = """pub struct Ledger { total: i64 }

impl Ledger {
    pub fn add(&mut self, cents: i64) {
        self.total += cents;
    }
}

pub async fn sync_ledger() -> Result<(), String> { Ok(()) }
"""


def test_rust_fallback():
    names = {s.qualname for s in chunk_fallback("l.rs", RUST, "rust").symbols}
    assert {"Ledger", "sync_ledger"} <= names
    assert any(n.endswith("add") for n in names)


def test_brace_matcher_ignores_strings_and_comments():
    lines = ["function f() {", '  const a = "{{{";', "  // }", "  /* } */", "  return `}`;", "}", "after"]
    assert find_block_end(lines, 0) == 5


def test_text_files_and_language_detection():
    assert detect_language("deploy/values.yaml") == "yaml"
    assert detect_language("Dockerfile") == "dockerfile"
    res = chunk_source("deploy/values.yaml", "a: 1\nb: 2\n")
    assert res.parser == "text:yaml" and res.symbols[0].qualname == "<module>"
    broken = chunk_source("bad.py", "def f(:\n  pass\n")
    assert broken.parser == "text:python"


def test_tree_sitter_and_structural_fallback_agree_on_fixture(shopfront):
    """CI runners have no cached grammars: qualnames must not depend on which parser is available."""
    from git_engram.indexer.ast_chunker import CODE_LANGUAGES

    root = shopfront.info.root
    for path in shopfront.engine.git.ls_files():
        language = detect_language(path)
        if language not in CODE_LANGUAGES or language == "python":
            continue
        src = (root / path).read_text(encoding="utf-8")
        fallback = {s.qualname for s in chunk_fallback(path, src, language).symbols}
        if tree_sitter_available(language):
            ts = {s.qualname for s in chunk_tree_sitter(path, src, language).symbols}
            assert ts == fallback, (path, ts ^ fallback)


def test_brace_matcher_skips_destructured_parameters():
    lines = [
        "export function Button({ id, n }: { id: string; n: number }) {",
        "  const onClick = async () => {",
        "    await go(id);",
        "  };",
        "  return <b onClick={onClick} />;",
        "}",
    ]
    assert find_block_end(lines, 0) == 5
    names = {s.qualname for s in chunk_fallback("b.tsx", "\n".join(lines), "tsx").symbols}
    assert names == {"Button"}  # whole file is one function: no module-level remainder
