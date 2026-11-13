"""Query-adaptive fusion router: a deterministic classifier that sets TD-RRF path weights per query.

Different questions need different evidence:

==========  ==========================================================  ==============================
class       trigger (first match wins)                                    what it changes
==========  ==========================================================  ==============================
behavioral  "why does/is <named function> ..." with no history cue        code-first: the named symbol is
                                                                          pinned to rank 1, at most 2
                                                                          commits in the top 5
causal      "why / reason / purpose / motivation / what changed ..."      graph down-weighted; reranker
                                                                          prefers commits and failures
change      imperative "add / change / rename / tune / raise ..."         graph up: hidden co-changes
                                                                          are the point of the query
exact       commit hash, error code (ERR_X, E4012, INC-123), exception    BM25 up, dense/graph down; reranker
            name, quoted string, file path                                weights verbatim-token hits
symptom     natural-language bug report ("billed twice", "logged out",    graph down; reranker prefers
            "crashes", "KeyError under load", ...) without identifiers    the commits that fixed it
general     anything else                                                 neutral weights
==========  ==========================================================  ==============================

The rules use only surface features of the query, so routing costs microseconds and never depends on the
corpus. Weights were set on the *dev* split only; the held-out split measures whether they generalise.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

from ..text.analyzer import STOPWORDS, is_rare_token, raw_tokens

_CAUSAL = re.compile(
    r"\b(why|reason|reasons|purpose|motivation|rationale|what\s+(?:did|does)\s+\S+\s+(?:change|do|fix)|"
    r"how\s+come|history\s+of|introduced|explain)\b",
    re.I,
)
_BEHAVIOR = re.compile(r"\bwhy\s+(?:does|do|is|are|must|would|should|can)\b", re.I)
_HISTORY = re.compile(
    r"\b(was|were|did|has|have|had|got|change[ds]?|changing|revert\w*|broke|broken|break|introduced|added|removed|"
    r"switched|replaced|put\s+back|doubled|increased|raised|lowered|shortened|led\s+to|history|regress\w*|"
    r"fix(?:ed|es)?)\b",
    re.I,
)
_DOTTED = re.compile(r"\b[A-Za-z_]\w*\.[A-Za-z_]\w*\b")
_CHANGE = re.compile(
    r"^\s*(?:please\s+)?(add|change|modify|update|rename|remove|delete|refactor|tune|raise|lower|increase|"
    r"decrease|extend|implement|move|replace|edit|make|redesign|show|support|migrate|bump|split)\b",
    re.I,
)
_SYMPTOM = re.compile(
    r"\b(twice|double|duplicate[ds]?|lost|losing|disappear\w*|corrupt\w*|crash\w*|broke|broken|breaks|fail\w*|"
    r"wrong|stale|out[- ]of[- ]date|logged\s+out|hang\w*|freez\w*|leak\w*|slow|timeout|timed\s+out|race|"
    r"off\s+by|cent\s+off|incorrect|missing|not\s+(?:found|work\w*|match\w*)|does\s?n[o']t|can\s?n[o']t|"
    r"unable|error|exception|bug|regression|after\s+(?:the|an?|we)|under\s+(?:load|traffic)|users?|customers?|"
    r"people)\b",
    re.I,
)
# Natural-language bug reports: silent drops, stale state, boundary cases, "kept / stopped / ended up" narratives.
_SYMPTOM_NL = re.compile(
    r"\b(silently|dropped|drops?|stale|stuck|forever|for\s+ages|kept|keeps|stopped|stops|never|again|second\s+time|"
    r"in\s+a\s+(?:tight\s+)?loop|hammer\w*|ran\s+out|out\s+of\s+memory|throttled|kicked\s+out|ended\s+(?:up|right)|"
    r"negative|shifted|vague|generic\s+error|measurably|accepted|rejected|charged|billed|priced|undercharged|"
    r"overcharged|blocked|locked|redelivered|replayed|expired|instead\s+of|edge\s+case|boundary|unexpected\w*|"
    r"whenever|only\s+(?:produced|showed)|went\s+(?:missing|wrong))\b",
    re.I,
)
_PAST_EVENT = re.compile(r"\b(?:was|were|got|had|became|turned)\b|\b[a-z]{3,}ed\b", re.I)
_HASH = re.compile(r"\b[0-9a-f]{7,40}\b")
_ERROR_CODE = re.compile(r"\b(?:[A-Z][A-Z0-9]*_[A-Z0-9_]+|E\d{3,5}|[A-Z]{2,6}-\d{2,6}|TS\d{4}|E0\d{3})\b")
_EXCEPTION = re.compile(r"\b(?:[A-Z]\w*(?:Error|Exception|Exit|Warning)|panic|WARNING|FAIL(?:ED)?|Traceback)\b")
_QUOTED = re.compile(r"[\"'`][^\"'`]{3,}[\"'`]")
_PATH = re.compile(r"\b[\w.-]+/[\w./-]+|\b\w+\.(?:py|ts|tsx|js|go|rs|java|sql|yaml|yml|json|toml|html|md)\b")

WEIGHTS: dict[str, dict[str, float]] = {
    "causal": {"bm25": 1.0, "vec": 1.0, "graph": 0.5},
    "behavioral": {"bm25": 1.2, "vec": 1.0, "graph": 0.3},
    "exact": {"bm25": 2.0, "vec": 0.4, "graph": 0.2},
    "change": {"bm25": 1.0, "vec": 1.0, "graph": 1.3},
    "symptom": {"bm25": 1.2, "vec": 1.0, "graph": 0.0},  # co-change expansion off: coupled files crowd out fixes
    "general": {"bm25": 1.0, "vec": 1.0, "graph": 1.0},
}
RERANK_INTENT = {
    "causal": "causal",
    "behavioral": "neutral",  # code-first comes from named-symbol pinning; a relevant fix commit may still rank
    "symptom": "causal",
    "change": "change",
    "exact": "exact",
    "general": "auto",
}


ENABLE_BEHAVIORAL = True  # switches used by the ablation study
ENABLE_BROAD_SYMPTOM = True


@dataclass(frozen=True)
class QueryProfile:
    klass: str
    weights: dict[str, float]
    intent: str
    features: dict[str, bool] = field(default_factory=dict)


def query_features(query: str) -> dict[str, bool]:
    tokens = [t for t in raw_tokens(query) if t.lower() not in STOPWORDS]
    words = raw_tokens(query)
    named = (
        any(is_rare_token(t) for t in words)
        or bool(_DOTTED.search(query))
        or any(t[0].isupper() and t.lower() not in STOPWORDS for t in words[1:])
    )
    return {
        "behavior": bool(_BEHAVIOR.search(query)),
        "history": bool(_HISTORY.search(query)),
        "named_symbol": named,
        "causal": bool(_CAUSAL.search(query)),
        "change": bool(_CHANGE.search(query)),
        "symptom": bool(_SYMPTOM.search(query)),
        "symptom_nl": bool(_SYMPTOM_NL.search(query)),
        "past_event": bool(_PAST_EVENT.search(query)),
        "hash": bool(_HASH.search(query)) and any(c.isdigit() for c in _HASH.search(query).group(0)),
        "error_code": bool(_ERROR_CODE.search(query)),
        "exception": bool(_EXCEPTION.search(query)),
        "quoted": bool(_QUOTED.search(query)),
        "path": bool(_PATH.search(query)),
        "identifier": any(is_rare_token(t) for t in tokens),
        "short": len(tokens) <= 3,
    }


def _is_bug_report(f: dict[str, bool], query: str) -> bool:
    """A natural-language bug report: symptom vocabulary, or a past-tense narrative of at least five words that names
    no code identifier (identifier-bearing queries are handled by the exact/behavioral routes)."""
    if f["symptom_nl"]:
        return True
    return f["past_event"] and not f["identifier"] and len(raw_tokens(query)) >= 5


def classify(query: str) -> QueryProfile:
    f = query_features(query)
    strong_exact = f["hash"] or f["error_code"] or f["exception"] or f["quoted"] or f["path"]
    if ENABLE_BEHAVIORAL and f["causal"] and f["behavior"] and f["named_symbol"] and not f["history"]:
        klass = "behavioral"  # "why does <function> do X": answered by the code, possibly plus its fix commit
    elif f["causal"]:
        klass = "causal"
    elif f["change"]:  # imperative change request ("add X to schema.sql") even if it names a path/identifier
        klass = "change"
    elif strong_exact or (f["identifier"] and f["short"]):
        klass = "exact"
    elif f["symptom"] or (ENABLE_BROAD_SYMPTOM and _is_bug_report(f, query)):
        klass = "symptom"
    else:
        klass = "general"
    return QueryProfile(klass, dict(WEIGHTS[klass]), RERANK_INTENT[klass], f)
