"""Pluggable second-stage rerankers over the fused top-N candidates.

``InteractionReranker`` (default, offline, deterministic) scores each (query, document) *pair* jointly,
the way a cross-encoder does, from interpretable interaction features instead of attention:

==============  =================================================================================
feature         definition
==============  =================================================================================
prior           fused TD-RRF score / max fused score in the candidate set
coverage        IDF-weighted fraction of query terms present anywhere in the document
field           IDF-weighted fraction of query terms present in title / symbol / path
proximity       fraction of adjacent query-term pairs that are also adjacent in the document
exact           fraction of "rare" query tokens (identifiers, error codes, hashes) found verbatim
dense           asymmetric int8 cosine between query and document embeddings
intent          agreement between query intent (causal "why/bug" vs. "change/where") and doc kind
graph           spreading-activation strength (capped at 1) if the candidate was reached through the graph
==============  =================================================================================

``OnnxCrossEncoderReranker`` wraps a real cross-encoder (e.g. ms-marco-MiniLM-L-6-v2 exported to ONNX)
and blends its sigmoid score with the fused prior.
"""

from __future__ import annotations

import math
import re
from collections import OrderedDict
from itertools import pairwise
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import Protocol

from ..text.analyzer import STOPWORDS, analyze, is_rare_token, raw_tokens, stem

_CAUSAL = re.compile(
    r"\b(why|reason|purpose|history|introduced|regress\w*|bug\w*|broke\w*|incident|crash\w*|"
    r"fail\w*|error\w*|wrong|instead|motivation|caused?|outage|exception|panic\w*)\b",
    re.I,
)
_CHANGE = re.compile(
    r"\b(add|change|modify|update|rename|remove|delete|refactor|where|implement|extend|"
    r"move|replace|edit)\b",
    re.I,
)


@dataclass(slots=True)
class Candidate:
    doc_id: int
    kind: str
    title: str
    symbol: str
    path: str
    body: str
    prior: float
    dense: float
    graph: float


def _norm_term(term: str) -> str:
    return stem(term) if term.isalpha() else term


def idf_coverage(query: str, text: str, idf: Callable[[str], float]) -> float:
    """IDF-weighted fraction of (stemmed, non-stopword) query terms that occur in ``text``."""
    q_terms = list(dict.fromkeys(_norm_term(t) for t in analyze(query) if t not in STOPWORDS and len(t) >= 2))
    if not q_terms:
        return 0.0
    doc_terms = {_norm_term(t) for t in analyze(text[:6000])}
    weights = [idf(t) for t in q_terms]
    return sum(w for t, w in zip(q_terms, weights, strict=True) if t in doc_terms) / (sum(weights) or 1.0)


class Reranker(Protocol):
    name: str

    def rerank(self, query: str, candidates: Sequence[Candidate], intent: str = "auto") -> list[float]: ...


class InteractionReranker:
    name = "interaction"
    WEIGHTS = {
        "prior": 0.28,
        "coverage": 0.20,
        "field": 0.08,
        "proximity": 0.06,
        "exact": 0.10,
        "dense": 0.08,
        "intent": 0.05,
        "graph": 0.15,
    }

    def __init__(self, idf: Callable[[str], float] | None = None, cache_size: int = 8192):
        self.idf = idf or (lambda t: 1.0)
        self._cache: OrderedDict[int, tuple[frozenset[str], frozenset[str], frozenset[tuple[str, str]], str]] = (
            OrderedDict()
        )
        self.cache_size = cache_size

    def _doc_features(self, c: Candidate):
        hit = self._cache.get(c.doc_id)
        if hit is not None:
            self._cache.move_to_end(c.doc_id)
            return hit
        body = c.body[:4000]
        seq = [_norm_term(t) for t in analyze(body)]
        field_terms = frozenset(_norm_term(t) for t in analyze(f"{c.title} {c.symbol} {c.path}"))
        terms = frozenset(seq) | field_terms
        bigrams = frozenset(pairwise(seq))
        lowered = f"{c.title}\n{c.symbol}\n{c.path}\n{body}".lower()
        value = (terms, field_terms, bigrams, lowered)
        self._cache[c.doc_id] = value
        if len(self._cache) > self.cache_size:
            self._cache.popitem(last=False)
        return value

    def features(self, query: str, c: Candidate, max_prior: float, intent_hint: str = "auto") -> dict[str, float]:
        q_terms: list[str] = []
        for t in analyze(query):
            if t in STOPWORDS or len(t) < 2:
                continue
            n = _norm_term(t)
            if n not in q_terms:
                q_terms.append(n)
        terms, field_terms, bigrams, lowered = self._doc_features(c)
        weights = [self.idf(t) for t in q_terms]
        total = sum(weights) or 1.0
        coverage = sum(w for t, w in zip(q_terms, weights, strict=True) if t in terms) / total
        field = sum(w for t, w in zip(q_terms, weights, strict=True) if t in field_terms) / total
        q_seq = [_norm_term(t) for t in analyze(query) if t not in STOPWORDS]
        pairs = list(pairwise(q_seq))
        proximity = sum(1 for p in pairs if p in bigrams) / len(pairs) if pairs else 0.0
        rare = [r.lower() for r in raw_tokens(query) if is_rare_token(r)]
        exact = sum(1 for r in rare if r in lowered) / len(rare) if rare else 0.0
        if intent_hint == "auto":
            causal = bool(_CAUSAL.search(query))
            change = bool(_CHANGE.search(query))
        else:  # set by the query router
            causal, change = intent_hint == "causal", intent_hint == "change"
        if causal and not change:
            intent = 1.0 if c.kind in {"commit", "failure"} else 0.0
        elif change and not causal:
            intent = 1.0 if c.kind in {"symbol", "file"} else 0.0
        else:
            intent = 0.5
        return {
            "prior": c.prior / max_prior if max_prior > 0 else 0.0,
            "coverage": coverage,
            "field": field,
            "proximity": proximity,
            "exact": exact,
            "dense": max(0.0, c.dense),
            "intent": intent,
            "graph": c.graph,
        }

    def rerank(self, query: str, candidates: Sequence[Candidate], intent: str = "auto") -> list[float]:
        max_prior = max((c.prior for c in candidates), default=1.0)
        out = []
        for c in candidates:
            f = self.features(query, c, max_prior, intent)
            out.append(sum(self.WEIGHTS[k] * v for k, v in f.items()))
        return out


class OnnxCrossEncoderReranker:
    name = "onnx-cross-encoder"

    def __init__(self, model_dir: str, prior_weight: float = 0.4):
        from ..embeddings.onnx_backend import OnnxCrossEncoder

        self.model = OnnxCrossEncoder(model_dir)
        self.prior_weight = prior_weight

    def rerank(self, query: str, candidates: Sequence[Candidate], intent: str = "auto") -> list[float]:
        passages = [f"{c.title}\n{c.path}\n{c.body[:1500]}" for c in candidates]
        logits = self.model.score(query, passages)
        max_prior = max((c.prior for c in candidates), default=1.0) or 1.0
        return [
            self.prior_weight * (c.prior / max_prior) + (1 - self.prior_weight) / (1 + math.exp(-float(z)))
            for c, z in zip(candidates, logits, strict=True)
        ]
