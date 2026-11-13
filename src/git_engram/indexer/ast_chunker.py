"""AST-aware structural chunking into semantic symbols (functions, classes, methods, types).

Strategy per language, best available first:

1. Python -> CPython's own ``ast`` (exact, zero dependencies).
2. Other languages -> tree-sitter, when ``tree-sitter-language-pack`` is installed *and* the grammar is
   already cached locally (or ``allow_download`` is set). Grammar downloads never happen implicitly.
3. Fallback -> a structural parser: declaration regexes per language family plus a lexer-aware brace
   matcher (skips strings, template literals and comments) to find each symbol's extent.

Every file additionally yields a ``<module>`` chunk containing the lines that belong to no symbol
(imports, constants, configuration) so that history attribution never loses a hunk.
"""

from __future__ import annotations

import ast
import os
import re
from collections.abc import Callable
from dataclasses import dataclass
from functools import lru_cache
from pathlib import PurePosixPath

from ..models import Symbol

EXTENSION_LANGUAGE = {
    ".py": "python",
    ".pyi": "python",
    ".ts": "typescript",
    ".mts": "typescript",
    ".cts": "typescript",
    ".tsx": "tsx",
    ".js": "javascript",
    ".mjs": "javascript",
    ".cjs": "javascript",
    ".jsx": "javascript",
    ".go": "go",
    ".rs": "rust",
    ".dart": "dart",
    ".java": "java",
    ".kt": "kotlin",
    ".swift": "swift",
    ".c": "c",
    ".h": "c",
    ".cc": "cpp",
    ".cpp": "cpp",
    ".hpp": "cpp",
    ".cs": "csharp",
    ".rb": "ruby",
    ".php": "php",
    ".scala": "scala",
    ".sql": "sql",
    ".yaml": "yaml",
    ".yml": "yaml",
    ".json": "json",
    ".toml": "toml",
    ".md": "markdown",
    ".rst": "text",
    ".txt": "text",
    ".html": "html",
    ".css": "css",
    ".scss": "css",
    ".sh": "shell",
    ".ini": "ini",
    ".cfg": "ini",
    ".proto": "proto",
    ".graphql": "graphql",
    ".tf": "hcl",
}
CODE_LANGUAGES = {
    "python",
    "typescript",
    "tsx",
    "javascript",
    "go",
    "rust",
    "dart",
    "java",
    "kotlin",
    "swift",
    "c",
    "cpp",
    "csharp",
    "ruby",
    "php",
    "scala",
}
MODULE = "<module>"


def detect_language(path: str) -> str:
    name = PurePosixPath(path).name
    if name in {"Dockerfile", "Makefile"}:
        return name.lower()
    return EXTENSION_LANGUAGE.get(PurePosixPath(path).suffix.lower(), "text")


@dataclass(slots=True)
class ChunkResult:
    symbols: list[Symbol]
    imports: list[str]
    parser: str


def _module_chunk(path: str, language: str, lines: list[str], symbols: list[Symbol], doc: str = "") -> Symbol | None:
    covered = bytearray(len(lines) + 2)
    for s in symbols:
        if s.parent is None:
            for ln in range(s.start_line, min(s.end_line, len(lines)) + 1):
                covered[ln] = 1
    rest = [lines[i - 1] for i in range(1, len(lines) + 1) if not covered[i]]
    body = "\n".join(rest).strip()
    if not body:
        return None
    return Symbol(
        path=path,
        qualname=MODULE,
        name=MODULE,
        kind="module",
        language=language,
        start_line=1,
        end_line=max(1, len(lines)),
        signature=f"module {path}",
        docstring=doc,
        content=body[:6000],
    )


# --------------------------------------------------------------------------- python
def _python_header(lines: list[str], node: ast.AST) -> str:
    start = node.lineno - 1
    header: list[str] = []
    for i in range(start, min(len(lines), start + 12)):
        header.append(lines[i].strip())
        code = lines[i].split("#", 1)[0].rstrip()
        if code.endswith(":"):
            break
    return " ".join(header)


def chunk_python(path: str, source: str) -> ChunkResult:
    lines = source.splitlines()
    tree = ast.parse(source)
    symbols: list[Symbol] = []
    imports: list[str] = []

    def visit(body: list[ast.stmt], prefix: str, parent: str | None, depth: int) -> None:
        for node in body:
            if isinstance(node, (ast.Import, ast.ImportFrom)) and depth == 0:
                if isinstance(node, ast.ImportFrom) and node.module:
                    imports.append(node.module)
                imports.extend(a.name for a in node.names)
            if not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
                continue
            start = min([d.lineno for d in node.decorator_list] + [node.lineno])
            end = node.end_lineno or node.lineno
            qual = f"{prefix}{node.name}"
            is_class = isinstance(node, ast.ClassDef)
            kind = "class" if is_class else ("method" if parent else "function")
            if is_class:
                member_lines: set[int] = set()
                for child in node.body:
                    if isinstance(child, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
                        cstart = min([d.lineno for d in child.decorator_list] + [child.lineno])
                        member_lines.update(range(cstart, (child.end_lineno or child.lineno) + 1))
                own = [lines[i - 1] for i in range(start, end + 1) if i not in member_lines and i - 1 < len(lines)]
                sigs = [
                    f"    {_python_header(lines, c)}"
                    for c in node.body
                    if isinstance(c, (ast.FunctionDef, ast.AsyncFunctionDef))
                ]
                content = "\n".join(own + sigs)
            else:
                content = "\n".join(lines[start - 1 : end])
            symbols.append(
                Symbol(
                    path=path,
                    qualname=qual,
                    name=node.name,
                    kind=kind,
                    language="python",
                    start_line=start,
                    end_line=end,
                    signature=_python_header(lines, node),
                    docstring=ast.get_docstring(node) or "",
                    content=content[:8000],
                    parent=parent,
                )
            )
            if is_class and depth < 3:
                visit(node.body, f"{qual}.", qual, depth + 1)

    visit(tree.body, "", None, 0)
    module = _module_chunk(path, "python", lines, symbols, ast.get_docstring(tree) or "")
    if module:
        symbols.insert(0, module)
    return ChunkResult(symbols, sorted(set(imports)), "python-ast")


# ----------------------------------------------------------------------- tree-sitter
_TS_DEFS: dict[str, dict[str, str]] = {
    "typescript": {
        "function_declaration": "function",
        "generator_function_declaration": "function",
        "class_declaration": "class",
        "abstract_class_declaration": "class",
        "method_definition": "method",
        "interface_declaration": "interface",
        "type_alias_declaration": "type",
        "enum_declaration": "enum",
    },
    "go": {"function_declaration": "function", "method_declaration": "method", "type_spec": "type"},
    "rust": {
        "function_item": "function",
        "struct_item": "struct",
        "enum_item": "enum",
        "trait_item": "trait",
        "impl_item": "impl",
    },
    "java": {
        "class_declaration": "class",
        "interface_declaration": "interface",
        "method_declaration": "method",
        "constructor_declaration": "method",
        "enum_declaration": "enum",
    },
}
_TS_DEFS["tsx"] = _TS_DEFS["typescript"]
_TS_DEFS["javascript"] = _TS_DEFS["typescript"]
_TS_CONTAINERS = {
    "class_declaration",
    "abstract_class_declaration",
    "impl_item",
    "trait_item",
    "class_body",
    "declaration_list",
    "interface_declaration",
}


@lru_cache(maxsize=32)
def _ts_parser(language: str, allow_download: bool):
    if os.environ.get("GIT_ENGRAM_NO_TREE_SITTER"):
        return None
    try:
        import tree_sitter_language_pack as tslp
    except ImportError:
        return None
    try:
        if not allow_download and language not in set(tslp.downloaded_languages()):
            return None
        return tslp.get_parser(language)
    except Exception:
        return None


def tree_sitter_available(language: str, allow_download: bool = False) -> bool:
    return language in _TS_DEFS and _ts_parser(language, allow_download) is not None


def _ts_name(node, src: bytes) -> str | None:
    named = node.child_by_field_name("name")
    if named is None and node.type == "impl_item":
        named = node.child_by_field_name("type")
    if named is None:
        return None
    return src[named.start_byte : named.end_byte].decode("utf-8", "replace")


def chunk_tree_sitter(path: str, source: str, language: str, allow_download: bool = False) -> ChunkResult | None:
    parser = _ts_parser(language, allow_download)
    if parser is None or language not in _TS_DEFS:
        return None
    src = source.encode("utf-8")
    tree = parser.parse(src)
    defs = _TS_DEFS[language]
    lines = source.splitlines()
    symbols: list[Symbol] = []

    def emit(node, name: str, kind: str, parent: str | None, start_node) -> str:
        qual = f"{parent}.{name}" if parent else name
        start = start_node.start_point[0] + 1
        end = node.end_point[0] + 1
        text = src[start_node.start_byte : node.end_byte].decode("utf-8", "replace")
        signature = text.split("{", 1)[0].strip().replace("\n", " ")[:300]
        if kind == "impl":
            kind = "class"
        if parent and kind == "function":
            kind = "method"
        symbols.append(
            Symbol(
                path=path,
                qualname=qual,
                name=name,
                kind=kind,
                language=language,
                start_line=start,
                end_line=end,
                signature=signature,
                docstring=_leading_comment(lines, start),
                content=text[:8000],
                parent=parent,
            )
        )
        return qual

    def walk(node, parent: str | None) -> None:
        for child in node.children:
            target, start_node = child, child
            if child.type == "export_statement":
                inner = [c for c in child.children if c.type in defs or c.type == "lexical_declaration"]
                if inner:
                    target = inner[0]
            if target.type in defs:
                name = _ts_name(target, src)
                owner = parent
                if target.type == "method_declaration" and language == "go":
                    recv = target.child_by_field_name("receiver")
                    m = (
                        re.search(
                            r"\*?\s*(\w+)\s*(?:\[[^\]]*\])?\s*\)\s*$",
                            src[recv.start_byte : recv.end_byte].decode("utf-8", "replace"),
                        )
                        if recv
                        else None
                    )
                    owner = m.group(1) if m else parent
                if name:
                    qual = emit(target, name, defs[target.type], owner, start_node)
                    if target.type in _TS_CONTAINERS or defs[target.type] in {"class", "impl", "trait"}:
                        walk(target, qual)
                    continue
            if target.type == "lexical_declaration" and language in {"typescript", "tsx", "javascript"}:
                for decl in target.children:
                    if decl.type != "variable_declarator":
                        continue
                    value = decl.child_by_field_name("value")
                    if value is not None and value.type in {"arrow_function", "function_expression", "function"}:
                        name = _ts_name(decl, src)
                        if name:
                            emit(target, name, "function", parent, start_node)
                continue
            if child.type in _TS_CONTAINERS or child.type in {"class_body", "declaration_list", "type_declaration"}:
                walk(child, parent)

    walk(tree.root_node, None)
    module = _module_chunk(path, language, lines, [s for s in symbols if s.parent is None])
    if module:
        symbols.insert(0, module)
    return ChunkResult(symbols, _regex_imports(source, language), f"tree-sitter:{language}")


# ------------------------------------------------------------------- structural fallback
_IDENT = r"[A-Za-z_$][\w$]*"
_FALLBACK_PATTERNS: dict[str, list[tuple[re.Pattern[str], str]]] = {
    "typescript": [
        (re.compile(rf"^\s*(?:export\s+)?(?:default\s+)?(?:declare\s+)?(?:abstract\s+)?class\s+({_IDENT})"), "class"),
        (re.compile(rf"^\s*(?:export\s+)?(?:declare\s+)?interface\s+({_IDENT})"), "interface"),
        (re.compile(rf"^\s*(?:export\s+)?(?:declare\s+)?(?:const\s+)?enum\s+({_IDENT})"), "enum"),
        (re.compile(rf"^\s*(?:export\s+)?(?:default\s+)?(?:async\s+)?function\s*\*?\s*({_IDENT})"), "function"),
        (
            re.compile(
                rf"^\s*(?:export\s+)?(?:const|let|var)\s+({_IDENT})\s*(?::[^=]+)?=\s*(?:async\s*)?"
                rf"(?:\([^)]*\)|{_IDENT})\s*(?::\s*[^=]+)?=>"
            ),
            "function",
        ),
        (re.compile(rf"^\s*(?:export\s+)?type\s+({_IDENT})\s*(?:<[^>]*>)?\s*=\s*\{{"), "type"),
    ],
    "go": [
        (re.compile(r"^func\s+\(\s*\w+\s+\*?(\w+)[^)]*\)\s*(\w+)"), "method"),
        (re.compile(r"^func\s+(\w+)"), "function"),
        (re.compile(r"^type\s+(\w+)\s+(?:struct|interface)\b"), "type"),
    ],
    "rust": [
        (re.compile(r"^\s*(?:pub(?:\([^)]*\))?\s+)?(?:async\s+)?(?:const\s+)?(?:unsafe\s+)?fn\s+(\w+)"), "function"),
        (re.compile(r"^\s*(?:pub(?:\([^)]*\))?\s+)?(?:struct|enum|trait)\s+(\w+)"), "type"),
        (re.compile(r"^\s*impl(?:<[^>]*>)?\s+(?:[\w:<>]+\s+for\s+)?(\w+)"), "class"),
    ],
    "dart": [
        (re.compile(r"^\s*(?:abstract\s+)?(?:class|mixin|extension)\s+(\w+)"), "class"),
        (
            re.compile(
                r"^\s*(?:static\s+)?(?:Future<[^>]*>|Stream<[^>]*>|[\w<>?,\[\] ]+?)\s+(\w+)\s*\([^;]*\)\s*"
                r"(?:async\*?|sync\*)?\s*\{"
            ),
            "function",
        ),
    ],
    "java": [
        (
            re.compile(
                r"^\s*(?:(?:public|private|protected|abstract|final|static|sealed)\s+)*(?:class|interface|enum|record)"
                r"\s+(\w+)"
            ),
            "class",
        ),
        (
            re.compile(
                r"^\s*(?:(?:public|private|protected|static|final|synchronized|abstract)\s+)+[\w<>\[\],\s]+\s+(\w+)"
                r"\s*\([^;]*\)\s*(?:throws [\w.,\s]+)?\{"
            ),
            "method",
        ),
    ],
}
for _alias in ("tsx", "javascript"):
    _FALLBACK_PATTERNS[_alias] = _FALLBACK_PATTERNS["typescript"]
for _alias in ("kotlin", "csharp", "scala", "swift", "cpp", "c", "php"):
    _FALLBACK_PATTERNS[_alias] = _FALLBACK_PATTERNS["java"]

_TS_METHOD = re.compile(
    rf"^\s+(?:(?:public|private|protected|static|readonly|async|override|get|set)\s+)*\*?({_IDENT})\s*"
    rf"(?:<[^>]*>)?\s*\([^;]*\)\s*(?::\s*[^{{;]+)?\{{"
)
_KEYWORDS = {"if", "for", "while", "switch", "catch", "return", "function", "constructor_", "else", "do", "with"}


def _strip_line_comment(line: str, language: str) -> str:
    return line.split("//", 1)[0] if language != "python" else line.split("#", 1)[0]


def find_block_end(lines: list[str], start_idx: int) -> int:
    """Index of the line closing the body ``{`` at/after ``start_idx`` (lexer-aware brace matching).

    Braces inside parentheses (destructured parameters, inline object types in signatures) are ignored, so
    ``function F({ a }: { a: T }) {`` matches the body block rather than the parameter pattern.
    """
    depth = 0
    parens = 0
    opened = False
    in_block_comment = False
    quote: str | None = None
    for i in range(start_idx, len(lines)):
        line = lines[i]
        j = 0
        while j < len(line):
            ch = line[j]
            nxt = line[j + 1] if j + 1 < len(line) else ""
            if in_block_comment:
                if ch == "*" and nxt == "/":
                    in_block_comment = False
                    j += 1
            elif quote:
                if ch == "\\":
                    j += 1
                elif ch == quote:
                    quote = None
            elif ch == "/" and nxt == "/":
                break
            elif ch == "/" and nxt == "*":
                in_block_comment = True
                j += 1
            elif ch in "\"'`":
                quote = ch
            elif ch == "(" and depth == 0:
                parens += 1
            elif ch == ")" and depth == 0 and parens > 0:
                parens -= 1
            elif parens > 0:
                pass  # inside a parameter list: braces/semicolons belong to parameter types
            elif ch == ";" and not opened:
                return i  # declaration ended before any block opened (e.g. one-line arrow function)
            elif ch == "{":
                depth += 1
                opened = True
            elif ch == "}":
                depth -= 1
                if opened and depth == 0:
                    return i
            j += 1
        if quote in {"'", '"'}:
            quote = None  # unterminated single-line string literal
        if not opened and i > start_idx + 6:
            return start_idx  # declaration without a body (e.g. abstract signature)
    return len(lines) - 1


def _leading_comment(lines: list[str], start_line: int) -> str:
    out: list[str] = []
    i = start_line - 2
    while i >= 0 and len(out) < 20:
        s = lines[i].strip()
        if s.startswith(("//", "*", "/*", "///", "#")):
            out.append(s.lstrip("/*# ").rstrip("*/ "))
            i -= 1
            continue
        break
    return "\n".join(reversed([o for o in out if o]))


def _regex_imports(source: str, language: str) -> list[str]:
    found: set[str] = set()
    patterns = [
        r"""^\s*import\s+(?:[^'"]*?from\s+)?['"]([^'"]+)['"]""",
        r"""require\(\s*['"]([^'"]+)['"]\s*\)""",
        r"""^\s*use\s+([\w:]+)""",
        r"""^\s*import\s+([\w.]+)""",
        r"""^\s*"([\w./-]+)"\s*$""",
    ]
    for pat in patterns:
        for m in re.finditer(pat, source, re.MULTILINE):
            found.add(m.group(1))
    return sorted(found)


def chunk_fallback(path: str, source: str, language: str) -> ChunkResult:
    lines = source.splitlines()
    patterns = _FALLBACK_PATTERNS.get(language, [])
    symbols: list[Symbol] = []
    i = 0
    stack: list[tuple[str, int]] = []  # (qualname, end_idx) of enclosing containers

    while i < len(lines):
        while stack and i > stack[-1][1]:
            stack.pop()
        line = _strip_line_comment(lines[i], language)
        matched: tuple[str, str, str | None] | None = None
        for pat, kind in patterns:
            m = pat.match(line)
            if m:
                if language == "go" and kind == "method":
                    matched = (m.group(2), kind, m.group(1))
                else:
                    matched = (m.group(1), kind, None)
                break
        if matched is None and stack and language in {"typescript", "tsx", "javascript"}:
            m = _TS_METHOD.match(line)
            if m and m.group(1) not in _KEYWORDS:
                matched = (m.group(1), "method", None)
        if matched is None:
            i += 1
            continue
        name, kind, receiver = matched
        if name in _KEYWORDS:
            i += 1
            continue
        end = find_block_end(lines, i)
        parent = receiver or (stack[-1][0] if stack else None)
        if parent and kind == "function":
            kind = "method"
        qual = f"{parent}.{name}" if parent else name
        text = "\n".join(lines[i : end + 1])
        symbols.append(
            Symbol(
                path=path,
                qualname=qual,
                name=name,
                kind=kind,
                language=language,
                start_line=i + 1,
                end_line=end + 1,
                signature=line.strip().rstrip("{").strip()[:300],
                docstring=_leading_comment(lines, i + 1),
                content=text[:8000],
                parent=parent,
            )
        )
        if kind in {"class", "interface", "type", "impl"} and end > i:
            stack.append((qual, end))
            i += 1
        else:
            i = end + 1
    top_level = [s for s in symbols if s.parent is None or language == "go"]
    module = _module_chunk(path, language, lines, top_level)
    if module:
        symbols.insert(0, module)
    return ChunkResult(symbols, _regex_imports(source, language), f"structural:{language}")


def chunk_text(path: str, source: str, language: str, window: int = 80) -> ChunkResult:
    """Non-code files (YAML, SQL, Markdown, HTML ...) become fixed line windows."""
    lines = source.splitlines()
    symbols: list[Symbol] = []
    for start in range(0, max(1, len(lines)), window):
        block = lines[start : start + window]
        if not "\n".join(block).strip():
            continue
        qual = MODULE if start == 0 else f"<lines {start + 1}-{start + len(block)}>"
        symbols.append(
            Symbol(
                path=path,
                qualname=qual,
                name=qual,
                kind="module" if start == 0 else "section",
                language=language,
                start_line=start + 1,
                end_line=start + len(block),
                signature=f"{language} file {path}",
                content="\n".join(block)[:8000],
            )
        )
    return ChunkResult(symbols, [], f"text:{language}")


def chunk_source(path: str, source: str, allow_download: bool = False, prefer_tree_sitter: bool = True) -> ChunkResult:
    language = detect_language(path)
    if language == "python":
        try:
            return chunk_python(path, source)
        except (SyntaxError, ValueError, RecursionError):
            return chunk_text(path, source, "python")
    if language in CODE_LANGUAGES:
        if prefer_tree_sitter:
            result = chunk_tree_sitter(path, source, language, allow_download)
            if result is not None:
                return result
        return chunk_fallback(path, source, language)
    return chunk_text(path, source, language)


def symbols_overlapping(symbols: list[Symbol], ranges: list[tuple[int, int]]) -> list[Symbol]:
    """Most specific symbols touched by the given 1-based inclusive line ranges.

    Symbols are painted onto a line map largest-first, so a line inside a method resolves to the method
    rather than its class; lines outside any named symbol resolve to the ``<module>`` chunk.
    """
    if not ranges:
        return []
    named = sorted((s for s in symbols if s.qualname != MODULE), key=lambda s: s.end_line - s.start_line, reverse=True)
    module = next((s for s in symbols if s.qualname == MODULE), None)
    n_lines = max([s.end_line for s in symbols] + [hi for _, hi in ranges] + [1])
    owner: list[int] = [-1] * (n_lines + 2)
    for idx, s in enumerate(named):
        for line in range(max(1, s.start_line), min(s.end_line, n_lines) + 1):
            owner[line] = idx
    hit: dict[str, Symbol] = {}
    for lo, hi in ranges:
        for line in range(max(1, lo), min(hi, n_lines) + 1):
            idx = owner[line]
            if idx >= 0:
                hit[named[idx].qualname] = named[idx]
            elif module is not None:
                hit[MODULE] = module
    return list(hit.values())


SymbolChunker = Callable[[str, str], ChunkResult]
