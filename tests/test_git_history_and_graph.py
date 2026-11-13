import math

import pytest

from git_engram.config import GraphConfig
from git_engram.engine import Engram
from git_engram.indexer.cochange import compute_edges, statically_linked
from git_engram.indexer.gitio import Git, parse_patch
from git_engram.indexer.history import HistoryExtractor, classify_intent

PATCH = """diff --git a/src/a.py b/src/a.py
index 111..222 100644
--- a/src/a.py
+++ b/src/a.py
@@ -3,2 +3,3 @@ def f():
-    x = 1
-    y = 2
+    x = 10
+    y = 20
+    z = 30
@@ -20,0 +22,1 @@ def g():
+    return 1
diff --git a/old/name.ts b/new/name.ts
similarity index 90%
rename from old/name.ts
rename to new/name.ts
--- a/old/name.ts
+++ b/new/name.ts
@@ -1 +1 @@
-export const a = 1;
+export const a = 2;
diff --git a/gone.go b/gone.go
deleted file mode 100644
--- a/gone.go
+++ /dev/null
@@ -1,2 +0,0 @@
-package x
-func A() {}
diff --git a/new.md b/new.md
new file mode 100644
--- /dev/null
+++ b/new.md
@@ -0,0 +1,2 @@
+# hi
+there
"""


def test_parse_patch_ranges_renames_adds_deletes():
    files = {f.path: f for f in parse_patch(PATCH)}
    a = files["src/a.py"]
    assert a.change == "M" and a.added == 4 and a.removed == 2
    assert a.old_ranges == [(3, 4), (20, 20)] and a.new_ranges == [(3, 5), (22, 22)]
    assert a.added_lines[0] == (3, "    x = 10") and a.removed_lines[1] == (4, "    y = 2")
    assert sorted(set(a.old_touch)) == [3, 4, 20]
    r = files["new/name.ts"]
    assert r.change == "R" and r.old_path == "old/name.ts"
    assert files["gone.go"].change == "D" and files["gone.go"].new_ranges == []
    assert files["new.md"].change == "A" and files["new.md"].old_ranges == []


@pytest.mark.parametrize(
    "subject,body,intent",
    [
        ("fix(auth): tolerate clock skew", "", "bugfix"),
        ("fix(auth): block replay attacks (SEC-118)", "", "security"),
        ('Revert "perf: x"', "This reverts commit abc.", "revert"),
        ("perf(cache): skip lock", "", "perf"),
        ("feat(api): add currency", "", "feature"),
        ("Handle crash when cart is empty", "", "bugfix"),
        ("docs: readme", "", "docs"),
        ("update stuff", "", "change"),
    ],
)
def test_intent_classification(subject, body, intent):
    assert classify_intent(subject, body) == intent


def test_history_attributes_hunks_to_symbols_and_follows_renames(mini_repo):
    mini_repo.write("lib/calc.py", "def add(a, b):\n    return a + b\n\n\ndef sub(a, b):\n    return a - b\n")
    mini_repo.commit("feat: calc")
    mini_repo.write("lib/calc.py", "def add(a, b):\n    return a + b\n\n\ndef sub(a, b):\n    return b - a\n")
    mini_repo.commit("fix: sub had swapped operands")
    mini_repo.git("mv", "lib/calc.py", "lib/math_ops.py")
    mini_repo.commit("refactor: rename calc module")
    mini_repo.write("lib/math_ops.py", "def add(a, b):\n    return a + b + 0\n\n\ndef sub(a, b):\n    return b - a\n")
    mini_repo.commit("chore: touch add")
    commits = list(HistoryExtractor(Git(mini_repo.root)).iter_history(100))
    assert [c.commit.subject for c in commits][0] == "chore: touch add"
    by_subject = {c.commit.subject: c for c in commits}
    fix = by_subject["fix: sub had swapped operands"]
    assert fix.intent == "bugfix"
    assert [(t.path, t.qualname, t.added, t.removed) for t in fix.touches] == [("lib/math_ops.py", "sub", 1, 1)]
    assert "+     return b - a" in fix.touches[0].excerpt.replace("+    return", "+     return")
    assert "lib/math_ops.py::sub" in fix.capsule and "[bugfix]" in fix.capsule
    initial = by_subject["feat: calc"]
    assert {t.qualname for t in initial.touches} == {"add", "sub"}
    assert {p for p, *_ in initial.file_paths} == {"lib/math_ops.py"}  # historical path resolved to current name


def test_cochange_weight_formula():
    cfg = GraphConfig(half_life_days=10.0, min_support=2, max_files_per_commit=3)
    day = 86400.0
    ref = 100 * day
    tx = [(ref, ["a", "b"]), (ref - 10 * day, ["a", "b"]), (ref, ["a"]), (ref, ["a", "b", "c", "d"])]
    edges = {(e.src, e.dst): e for e in compute_edges(tx, ref, cfg)}
    lam = math.log(2) / (10 * day)
    expected = (1.0 + math.exp(-lam * 10 * day)) / math.sqrt(3 * 2)  # |C_a|=3, |C_b|=2 (bulk commit dropped)
    assert edges[("a", "b")].weight == pytest.approx(expected) == pytest.approx(1.5 / math.sqrt(6))
    assert edges[("a", "b")].support == 2
    assert edges[("a", "b")].confidence == pytest.approx(2 / 3) and edges[("b", "a")].confidence == 1.0
    assert ("a", "c") not in edges


def test_static_link_detection():
    imports = {"backend/auth/refresh.py": {"backend.auth.token", "issue_token"}}
    assert statically_linked("backend/auth/refresh.py", "backend/auth/token.py", imports)
    assert not statically_linked("backend/api/schemas.py", "mobile/src/parsers/orderParser.ts", imports)


def test_fixture_history_and_hidden_coupling(shopfront):
    e: Engram = shopfront.engine
    rows = e.db.query("SELECT * FROM cochange_edges WHERE level='file' AND src='backend/api/schemas.py'")
    targets = {r["dst"]: r for r in rows}
    assert "mobile/src/parsers/orderParser.ts" in targets
    edge = targets["mobile/src/parsers/orderParser.ts"]
    assert edge["static_link"] == 0 and edge["support"] >= 5 and edge["weight"] > 0.5
    blame = e.db.query(
        "SELECT b.lines, c.intent FROM symbol_blame b JOIN commits c ON c.hash=b.hash "
        "WHERE b.path='backend/auth/token.py' AND b.qualname='is_token_expired'"
    )
    assert any(r["intent"] == "bugfix" and r["lines"] >= 1 for r in blame)
    revert = e.db.query_one("SELECT intent FROM commits WHERE subject LIKE 'Revert%'")
    assert revert["intent"] == "revert"
