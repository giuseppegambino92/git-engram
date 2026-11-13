"""RAG Context Firewall: neutralises prompt-injection payloads before retrieved text reaches an agent.

Threats handled (CamoLeak / PoisonedRAG style attacks hidden in third-party code, docs or git logs):

1. **Invisible-character smuggling** -- Unicode tag block (U+E0000..U+E007F, "ASCII smuggling"), zero-width
   characters, bidirectional overrides/isolates ("Trojan Source", CVE-2021-42574), variation selectors used
   for emoji smuggling, and other format (Cf) code points. All are stripped; tag-block payloads are decoded so
   the hidden instruction is reported verbatim.
2. **Instruction-override patterns** -- "ignore previous instructions", fake role/turn delimiters, tool-call
   coercion, secret exfiltration and markdown image/link beacons (the CamoLeak exfil channel). Matching spans
   are redacted in place.
3. **High-perplexity spans** -- adversarial suffixes (GCG-style) and encoded blobs. A hashed character-trigram
   language model trained on the indexed repository scores 64-char windows; windows above the calibrated
   99.5th-percentile perplexity of the corpus itself are flagged (reported, not redacted, since minified or
   generated code is also legitimately high-perplexity).
"""

from __future__ import annotations

import io
import math
import re
import unicodedata
import zlib
from dataclasses import dataclass, field

import numpy as np

from ..models import SecurityFinding

_TAG_BLOCK = (0xE0000, 0xE007F)
_ZERO_WIDTH = {0x200B, 0x200C, 0x200D, 0x2060, 0xFEFF, 0x180E, 0x2061, 0x2062, 0x2063, 0x2064, 0x034F, 0x00AD}
_BIDI = set(range(0x202A, 0x202F)) | set(range(0x2066, 0x206A)) | {0x200E, 0x200F, 0x061C}
_VARIATION = set(range(0xFE00, 0xFE10)) | set(range(0xE0100, 0xE01F0))

_INJECTION_RULES: list[tuple[str, str, re.Pattern[str]]] = [
    (
        "override.ignore_instructions",
        "high",
        re.compile(
            r"\b(ignore|disregard|forget|override|bypass)\b[^\n.]{0,40}\b(previous|prior|above|earlier|all|any|your)"
            r"\b[^\n.]{0,30}\b(instructions?|rules|prompts?|guidelines|directives|context|system)\b",
            re.I,
        ),
    ),
    (
        "override.new_instructions",
        "high",
        re.compile(
            r"\b(new|updated|real|actual|hidden)\s+(system\s+)?(instructions?|directives?|prompt)\s*[:\-]", re.I
        ),
    ),
    (
        "override.role_hijack",
        "high",
        re.compile(
            r"\byou\s+are\s+now\b|\bact\s+as\s+(an?\s+)?(unrestricted|jailbroken|dan|developer\s+mode)\b|"
            r"\b(enter|enable)\s+(developer|god|dan)\s+mode\b",
            re.I,
        ),
    ),
    (
        "delimiter.fake_turn",
        "high",
        re.compile(
            r"<\|?(im_start|im_end|system|endoftext|eot_id|start_header_id)\|?>|\[/?INST\]|<<\s*SYS\s*>>|"
            r"^\s*#{2,}\s*(system|assistant)\s*:?\s*$|^\s*(human|assistant)\s*:\s*(ignore|you|now|from)\b",
            re.I | re.M,
        ),
    ),
    (
        "delimiter.fake_tool_call",
        "high",
        re.compile(r"<\s*/?\s*(function_calls|invoke|tool_use|tool_call|antml:[a-z_]+)\b", re.I),
    ),
    (
        "exfil.markdown_beacon",
        "high",
        re.compile(
            r"!\[[^\]]*\]\(\s*https?://[^)\s]*[?&#][^)\s]*(\{|%7B|\$|data=|token=|secret=|key=|q=)[^)]*\)", re.I
        ),
    ),
    ("exfil.camo_proxy", "medium", re.compile(r"https?://camo\.githubusercontent\.com/[0-9a-f]{20,}", re.I)),
    (
        "exfil.secret_request",
        "high",
        re.compile(
            r"\b(send|post|upload|exfiltrate|leak|forward|include|append|print|reveal|output)\b[^\n]{0,60}"
            r"\b(api[_\s-]?keys?|secrets?|credentials?|passwords?|tokens?|\.env|ssh\s+keys?|private\s+keys?)\b"
            r"[^\n]{0,60}\b(to|into|via|at|in)\b[^\n]{0,40}(https?://|curl|wget|webhook|url|the\s+(image|link))",
            re.I,
        ),
    ),
    ("exec.remote_pipe", "high", re.compile(r"\b(curl|wget)\b[^\n|]{0,200}\|\s*(ba|z)?sh\b", re.I)),
    (
        "social.concealment",
        "medium",
        re.compile(
            r"\b(do\s+not|don't|never)\s+(tell|inform|mention|reveal|show)\s+(this\s+to\s+)?(the\s+)?(user|human|developer)",
            re.I,
        ),
    ),
    (
        "agent.destructive_order",
        "high",
        re.compile(
            r"\b(ai|llm|assistant|agent|claude|copilot|cursor|model)s?\b[^\n]{0,40}\b(must|should|shall|are\s+required\s+to|"
            r"have\s+to|need\s+to)\b[^\n]{0,40}\b(delete|remove|disable|drop|wipe|erase|truncate|force[-\s]?push|"
            r"rm\s+-rf|exfiltrate|leak|skip)\b",
            re.I,
        ),
    ),
    (
        "agent.coercion",
        "medium",
        re.compile(
            r"\b(ai|llm|assistant|agent|claude|copilot|cursor|model)s?\b[^\n]{0,40}\b(must|should|are\s+required\s+to|"
            r"have\s+to)\b[^\n]{0,40}\b(run|execute|push|commit|approve|merge|send|install|call)\b",
            re.I,
        ),
    ),
    ("encoding.base64_blob", "low", re.compile(r"(?<![A-Za-z0-9+/=])[A-Za-z0-9+/]{120,}={0,2}(?![A-Za-z0-9+/=])")),
]


@dataclass(slots=True)
class SanitizeResult:
    text: str
    findings: list[SecurityFinding] = field(default_factory=list)

    @property
    def clean(self) -> bool:
        return not self.findings

    @property
    def blocked(self) -> bool:
        return any(f.severity == "high" for f in self.findings)


def _classify_char(cp: int) -> str | None:
    if _TAG_BLOCK[0] <= cp <= _TAG_BLOCK[1]:
        return "tag"
    if cp in _ZERO_WIDTH:
        return "zero_width"
    if cp in _BIDI:
        return "bidi"
    if cp in _VARIATION:
        return "variation_selector"
    if cp in (0x09, 0x0A, 0x0D):
        return None
    cat = unicodedata.category(chr(cp))
    if cat == "Cf":
        return "format"
    if cat == "Cc":
        return "control"
    return None


def _is_emoji(ch: str) -> bool:
    cp = ord(ch)
    return 0x1F000 <= cp <= 0x1FAFF or 0x2600 <= cp <= 0x27BF or 0x1F1E6 <= cp <= 0x1F1FF


def strip_invisible(text: str) -> tuple[str, list[SecurityFinding]]:
    out: list[str] = []
    counts: dict[str, int] = {}
    tag_payload: list[str] = []
    n = len(text)
    for i, ch in enumerate(text):
        kind = _classify_char(ord(ch))
        if kind is None:
            out.append(ch)
            continue
        # Legitimate emoji sequences: ZWJ between two emoji, VS16 right after an emoji.
        if ch == "\u200d" and 0 < i < n - 1 and _is_emoji(text[i - 1]) and _is_emoji(text[i + 1]):
            out.append(ch)
            continue
        if ch == "\ufe0f" and i > 0 and _is_emoji(text[i - 1]):
            out.append(ch)
            continue
        counts[kind] = counts.get(kind, 0) + 1
        if kind == "tag":
            inner = ord(ch) - 0xE0000
            if 0x20 <= inner < 0x7F:
                tag_payload.append(chr(inner))
    findings: list[SecurityFinding] = []
    if counts.get("tag"):
        decoded = "".join(tag_payload).strip()
        findings.append(
            SecurityFinding(
                rule="invisible.unicode_tags",
                severity="high",
                detail=f"{counts['tag']} tag chars stripped; hidden payload: {decoded[:200]!r}",
            )
        )
    for kind, sev in (
        ("bidi", "high"),
        ("zero_width", "medium"),
        ("variation_selector", "medium"),
        ("format", "low"),
        ("control", "low"),
    ):
        if counts.get(kind):
            findings.append(
                SecurityFinding(
                    rule=f"invisible.{kind}",
                    severity=sev,
                    detail=f"{counts[kind]} {kind.replace('_', ' ')} code points stripped",
                )
            )
    if counts.get("tag") and tag_payload:
        hidden = "".join(tag_payload)
        for rule, _sev, pattern in _INJECTION_RULES:
            if pattern.search(hidden):
                findings.append(
                    SecurityFinding(
                        rule=f"smuggled.{rule}",
                        severity="high",
                        detail="injection pattern inside invisible tag payload",
                    )
                )
                break
    return "".join(out), findings


class CharTrigramLM:
    """Hashed character-trigram LM with add-k smoothing; perplexity is calibrated on the repo itself."""

    BUCKETS3 = 1 << 18
    BUCKETS2 = 1 << 16
    ALPHABET = 96
    WINDOW = 64

    def __init__(self) -> None:
        self.c3 = np.zeros(self.BUCKETS3, dtype=np.uint32)
        self.c2 = np.zeros(self.BUCKETS2, dtype=np.uint32)
        self.threshold = float("inf")
        self.k = 0.1

    @staticmethod
    def _norm(text: str) -> str:
        return unicodedata.normalize("NFKC", text)

    def _ids(self, text: str) -> tuple[np.ndarray, np.ndarray]:
        b = np.frombuffer(self._norm(text).encode("utf-8", "replace"), dtype=np.uint8).astype(np.uint32)
        if b.size < 3:
            return np.zeros(0, np.uint32), np.zeros(0, np.uint32)
        h2 = (b[:-2] * 257 + b[1:-1]) % self.BUCKETS2
        h3 = ((b[:-2] * 257 + b[1:-1]) * 263 + b[2:]) % self.BUCKETS3
        return h2, h3

    MIN_TRAINING_CHARS = 100_000

    def fit(self, texts: list[str], max_chars: int = 3_000_000) -> None:
        """Train on the repository; stays disabled (threshold = inf) if the corpus is too small to calibrate."""
        used = 0
        sample_windows: list[str] = []
        for t in texts:
            if used > max_chars:
                break
            used += len(t)
            h2, h3 = self._ids(t)
            np.add.at(self.c2, h2, 1)
            np.add.at(self.c3, h3, 1)
            for start in range(0, max(0, len(t) - self.WINDOW), self.WINDOW * 7):
                sample_windows.append(t[start : start + self.WINDOW])
        if sample_windows and used >= self.MIN_TRAINING_CHARS:
            ppl = np.array([self.perplexity(w) for w in sample_windows[:20000]])
            self.threshold = float(np.percentile(ppl, 99.5)) * 1.5

    def perplexity(self, text: str) -> float:
        h2, h3 = self._ids(text)
        if h3.size == 0:
            return 1.0
        p = (self.c3[h3].astype(np.float64) + self.k) / (self.c2[h2].astype(np.float64) + self.k * self.ALPHABET)
        return float(math.exp(-np.mean(np.log(np.clip(p, 1e-12, 1.0)))))

    def anomalous_windows(self, text: str) -> list[tuple[int, float]]:
        if not math.isfinite(self.threshold) or len(text) < self.WINDOW:
            return []
        out: list[tuple[int, float]] = []
        for start in range(0, len(text) - self.WINDOW + 1, self.WINDOW // 2):
            window = text[start : start + self.WINDOW]
            if sum(c.isalpha() for c in window) < self.WINDOW * 0.4:
                continue
            ppl = self.perplexity(window)
            if ppl > self.threshold:
                out.append((start, ppl))
        return out

    def state_bytes(self) -> bytes:
        buf = io.BytesIO()
        np.savez_compressed(buf, c3=self.c3, c2=self.c2, thr=np.array([self.threshold]))
        return buf.getvalue()

    @classmethod
    def from_state(cls, blob: bytes) -> CharTrigramLM:
        lm = cls()
        data = np.load(io.BytesIO(blob))
        lm.c3, lm.c2, lm.threshold = data["c3"], data["c2"], float(data["thr"][0])
        return lm


class ContextFirewall:
    def __init__(self, lm: CharTrigramLM | None = None, redact: bool = True):
        self.lm = lm
        self.redact = redact
        self._cache: dict[int, SanitizeResult] = {}

    def scan(self, text: str) -> SanitizeResult:
        key = zlib.crc32(text.encode("utf-8", "surrogatepass")) ^ len(text)
        cached = self._cache.get(key)
        if cached is not None:
            return cached
        cleaned, findings = strip_invisible(text)
        detection_view = unicodedata.normalize("NFKC", cleaned)
        spans: list[tuple[int, int, str]] = []
        for rule, severity, pattern in _INJECTION_RULES:
            for m in pattern.finditer(detection_view):
                findings.append(SecurityFinding(rule=rule, severity=severity, detail=m.group(0)[:160]))
                if severity == "high" and detection_view == cleaned:
                    spans.append((m.start(), m.end(), rule))
        if self.lm is not None:
            windows = self.lm.anomalous_windows(cleaned)
            if windows:
                worst = max(p for _, p in windows)
                findings.append(
                    SecurityFinding(
                        rule="perplexity.anomalous_span",
                        severity="low",
                        detail=f"{len(windows)} window(s) above calibrated perplexity "
                        f"{self.lm.threshold:.1f} (max {worst:.1f})",
                    )
                )
        if self.redact and spans:
            spans.sort()
            merged: list[list] = []
            for s, e, r in spans:
                if merged and s <= merged[-1][1]:
                    merged[-1][1] = max(merged[-1][1], e)
                else:
                    merged.append([s, e, r])
            parts, last = [], 0
            for s, e, r in merged:
                parts.append(cleaned[last:s])
                parts.append(f"[REDACTED by git-engram firewall: {r}]")
                last = e
            parts.append(cleaned[last:])
            cleaned = "".join(parts)
        elif self.redact and any(
            f.severity == "high" and not f.rule.startswith(("invisible", "smuggled")) for f in findings
        ):
            cleaned = "[REDACTED by git-engram firewall: injection detected in NFKC-normalised text]"
        result = SanitizeResult(cleaned, findings)
        if len(self._cache) > 4096:
            self._cache.clear()
        self._cache[key] = result
        return result

    def sanitize(self, text: str) -> str:
        return self.scan(text).text
