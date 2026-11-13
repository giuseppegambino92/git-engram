"""Code-aware text analysis shared by FTS5 indexing, query parsing and the hashing embedder.

SQLite's Python binding cannot register native FTS5 tokenizers, so the "custom tokenizer" is applied
on the Python side: every document and query is expanded into a canonical term stream that FTS5 then
indexes with ``unicode61 tokenchars '_'``. The expansion keeps compound identifiers intact *and* emits
their parts, so all of these match the way a developer expects:

* ``is_token_expired``  -> ``is_token_expired is token expired``
* ``parseHTTPResponse`` -> ``parsehttpresponse parse http response``
* ``INC-2291``          -> ``inc_2291 inc 2291``
* ``ERR_CART_STALE``    -> ``err_cart_stale err cart stale``
* ``9f3c2a1b7d...``     -> ``9f3c2a1b7d... 9f3c2a1`` (commit hashes keep a 7-char short form)
"""

from __future__ import annotations

import re
from functools import lru_cache

_RAW_TOKEN = re.compile(r"[A-Za-z0-9_]+(?:-[A-Za-z0-9_]+)*")
_CAMEL = re.compile(r"[A-Z]+(?=[A-Z][a-z])|[A-Z]?[a-z]+|[A-Z]+|\d+")
_HEX = re.compile(r"^[0-9a-f]{7,40}$")

STOPWORDS = frozenset(
    """a an and are as at be been being but by can could did do does doing done for from had has have having
    how i if in into is it its itself just me my no nor not of on once only or other our out over own same she
    should so some such than that the their them then there these they this those through to too under until up
    very was we were what when where which while who whom why will with would you your yours about above after
    again against all am any because before below between both down during each few further here more most off
    also get got make made use used using want wants need needs still ever even else every whose whats lets let
    happen happens happened""".split()
)

CODE_NOISE = frozenset(
    """self this def function func fn const let var return import from none null nil true false str int float
    bool string number void async await class pub use mut new end begin package type interface export default
    elif else self_ cls args kwargs""".split()
)

_SUFFIXES = (
    ("ational", "ate"),
    ("tional", "tion"),
    ("ization", "ize"),
    ("ations", "ate"),
    ("ation", "ate"),
    ("iveness", "ive"),
    ("fulness", "ful"),
    ("ousness", "ous"),
    ("ments", ""),
    ("ment", ""),
    ("ness", ""),
    ("ingly", ""),
    ("edly", ""),
    ("ings", ""),
    ("ing", ""),
    ("ies", "y"),
    ("ied", "y"),
    ("ers", ""),
    ("er", ""),
    ("ed", ""),
    ("es", ""),
    ("ly", ""),
    ("s", ""),
)


@lru_cache(maxsize=65536)
def stem(word: str) -> str:
    """Tiny deterministic suffix stripper (Porter-lite) used for dense features only."""
    if len(word) <= 3 or not word.isalpha():
        return word
    for suffix, repl in _SUFFIXES:
        if word.endswith(suffix) and len(word) - len(suffix) >= 3:
            if suffix == "s" and word.endswith("ss"):
                break
            word = word[: len(word) - len(suffix)] + repl
            break
    if len(word) > 4 and word[-1] in "ey":
        word = word[:-1]
    return word


@lru_cache(maxsize=65536)
def split_identifier(token: str) -> tuple[str, ...]:
    """Split ``snake_case`` / ``camelCase`` / ``kebab-case`` identifiers into lowercase parts."""
    parts: list[str] = []
    for chunk in re.split(r"[_\-]+", token):
        if not chunk:
            continue
        parts.extend(p.lower() for p in _CAMEL.findall(chunk))
    return tuple(parts)


@lru_cache(maxsize=65536)
def _expand(raw: str) -> tuple[str, ...]:
    lowered = raw.lower()
    if _HEX.match(lowered) and any(c.isdigit() for c in lowered) and any(c.isalpha() for c in lowered):
        return (lowered,) if len(lowered) == 7 else (lowered, lowered[:7])
    canonical = lowered.replace("-", "_").strip("_")
    parts = split_identifier(raw)
    out: list[str] = []
    if canonical and (len(parts) != 1 or parts[0] != canonical):
        out.append(canonical)
    out.extend(parts)
    return tuple(t for t in out if t)


def raw_tokens(text: str) -> list[str]:
    return _RAW_TOKEN.findall(text)


def analyze(text: str) -> list[str]:
    """Index-side term stream (no stopword removal; BM25 handles common terms)."""
    out: list[str] = []
    for raw in _RAW_TOKEN.findall(text):
        out.extend(_expand(raw))
    return out


def analyze_query(text: str) -> list[str]:
    """Query-side terms: expanded, de-duplicated, with natural-language stopwords removed."""
    seen: set[str] = set()
    out: list[str] = []
    for raw in _RAW_TOKEN.findall(text):
        for term in _expand(raw):
            if term in STOPWORDS or len(term) < 2 or term in seen:
                continue
            seen.add(term)
            out.append(term)
    return out


def is_rare_token(raw: str) -> bool:
    """Heuristic for 'exact-match' tokens: identifiers, error codes, ticket ids and hashes."""
    if len(raw) < 4:
        return False
    if "_" in raw.strip("_") or "-" in raw:
        return True
    if any(c.isdigit() for c in raw) and any(c.isalpha() for c in raw):
        return True
    if raw[0].islower() and any(c.isupper() for c in raw[1:]):
        return True
    if sum(c.isupper() for c in raw) >= 2 and any(c.islower() for c in raw):
        return True  # PascalCase compound such as CartService / RuntimeError
    if raw.isupper() and len(raw) >= 4:
        return True
    return False


def dense_terms(text: str) -> list[str]:
    """Feature stream for the hashing embedder: stemmed subwords plus intact compound identifiers."""
    out: list[str] = []
    for raw in _RAW_TOKEN.findall(text):
        expanded = _expand(raw)
        for term in expanded:
            if term in STOPWORDS or term in CODE_NOISE or len(term) < 2:
                continue
            if "_" in term:
                out.append(term)
            elif term.isdigit():
                if len(term) >= 3:
                    out.append(term)
            else:
                out.append(stem(term))
    return out
