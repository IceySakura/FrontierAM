"""Graph-driven source editing — turn feature nodes into unified diffs.

Shared primitive used by both Mask (stub entries + delete helpers) and Eval (revert
a feature's tests). It rewrites repo source using each node's SCIP ``enclosing_range``
line span (no Python parse, so boundaries are approximate) and emits a unified diff
plus its exact inverse.

Depends only on the protobuf-free ``graph`` model + stdlib; it reads repo files (so
it lives outside ``graph``, which must stay free of source IO) and is stage-agnostic
(so it lives outside any single stage package).
"""

from __future__ import annotations

import difflib
import re

from collections import defaultdict
from pathlib import Path

import networkx as nx

from .graph import node_info, qualified_name


_HUNK_RE = re.compile(r'^@@ -(\S+) \+(\S+) @@')
_PYTHON_TEST_SUPPORT_FILES = frozenset({'__init__.py', 'conftest.py'})


def _hunk_side_count(range_spec: str) -> int | None:
    """Line count of one side of a hunk range: ``start`` (implicit 1) or ``start,count``.

    Returns ``None`` when *count* is not an int (e.g. agent prose that happens to look like a
    hunk header) — callers treat that as "not a real hunk" instead of raising.
    """
    _, _, count = range_spec.partition(',')
    if not count:
        return 1
    try:
        return int(count)
    except ValueError:
        return None


def _header_path(raw: str) -> str:
    """A ``--- ``/``+++ `` header value → repo-relative path (drop the ``a/``/``b/`` prefix + trailing tab)."""
    path = raw.split('\t', 1)[0].strip()
    if path.startswith(('a/', 'b/')):
        path = path[2:]
    return path


def patch_files(patch: str) -> set[str]:
    """Repo-relative paths a unified diff modifies.

    Reads file headers only — ``diff --git`` plus the ``--- ``/``+++ `` pair — and skips hunk BODY
    lines by counting each hunk's declared line counts (the way ``git apply`` does), so body content
    that happens to start with ``--- ``/``+++ `` is never mistaken for a header. Whole-file deletions
    (``+++ /dev/null``) are recorded via their ``--- a/x`` path, and whole-file creates
    (``--- /dev/null``) via the ``+++ b/x`` path — both of which ``git apply`` accepts headerless.
    A line that looks like ``@@ … @@`` but has a non-integer count is ignored (not a hunk), so
    agent-authored prose cannot raise out of this helper.
    """
    files: set[str] = set()
    old_left = new_left = 0  # counted lines still owed by the current hunk (0/0 => outside a hunk)
    prev_old = ''  # last `--- ` path, used to name a `+++ /dev/null` whole-file deletion
    for line in patch.splitlines():
        if old_left <= 0 and new_left <= 0:
            if line.startswith('diff --git '):
                _head, sep, tail = line.partition(' b/')
                if sep:
                    files.add(tail.strip())
            elif line.startswith('--- '):
                prev_old = _header_path(line[4:])
            elif line.startswith('+++ '):
                path = _header_path(line[4:])
                if path and path != '/dev/null':
                    files.add(path)
                elif path == '/dev/null' and prev_old and prev_old != '/dev/null':
                    files.add(prev_old)  # whole-file deletion: the file is the ``---`` side
            else:
                m = _HUNK_RE.match(line)
                if m:
                    old_c = _hunk_side_count(m.group(1))
                    new_c = _hunk_side_count(m.group(2))
                    if old_c is None or new_c is None:
                        continue  # prose, not a hunk — stay outside
                    old_left, new_left = old_c, new_c
        else:
            # Inside a hunk body: consume by leading char so the counts decide when the hunk ends.
            head = line[:1]
            if head == '-':
                old_left -= 1
            elif head == '+':
                new_left -= 1
            elif head != '\\':  # a '\ No newline' marker belongs to the prior line; don't count it
                old_left -= 1
                new_left -= 1
    return files


def patch_test_targets(patch: str) -> list[str]:
    """Executable test targets touched by *patch*, excluding Python support files.

    ``test_gold`` may add package markers or pytest configuration alongside real tests. Passing
    ``__init__.py`` as well as its child test module can collect the same tests twice on older
    pytest versions; passing ``conftest.py`` as a target has similarly framework-dependent meaning.
    Both files remain in the patch and are available during collection, but are not runner targets.
    """
    return sorted(path for path in patch_files(patch) if Path(path).name not in _PYTHON_TEST_SUPPORT_FILES)


STUB = 'stub'
"""Keep the function's signature header, replace its body with ``raise NotImplementedError``."""
DELETE = 'delete'
"""Remove the function (and any decorators above it) entirely."""


def _span(info: dict) -> tuple[int, int] | None:
    """0-based inclusive ``(start_line, end_line)`` from the node's ``enclosing_range``."""
    er = info.get('enclosing_range')
    if not er or len(er) != 4:
        return None
    return int(er[0]), int(er[2])


def _leading_ws(line: str) -> str:
    return line[: len(line) - len(line.lstrip())]


def _code_without_comment(line: str) -> str:
    """Logical code of a source line with a trailing ``#`` comment stripped (best-effort)."""
    return line.split('#', 1)[0].rstrip()


def _stub_replacement(lines: list[str], start: int, end: int) -> list[str]:
    """Keep the signature header, replace the body with ``raise NotImplementedError``.

    Handles multi-line defs (header ends with ``:``), one-liners (``def f(a): return a`` — keep
    only through the colon so the original body is not leaked), and a trailing-comment header
    (``def f(a):  # note``). Never returns the original body alongside the raise.
    """
    body_indent = _leading_ws(lines[start]) + '    '
    raise_line = f'{body_indent}raise NotImplementedError\n'
    last = min(end, len(lines) - 1)

    for i in range(start, last + 1):
        code = _code_without_comment(lines[i])
        if code.endswith(':'):
            # Multi-line (or ``def f():  # c``) — keep through this header line, drop the rest.
            return [*lines[start : i + 1], raise_line]
        if i == start and ':' in code:
            # One-liner ``def …: <body>`` — keep only the signature through the colon.
            before, colon, _after = lines[i].partition(':')
            return [before + colon + '\n', raise_line]

    # No usable ':' in the span — hide everything; keep a truncated first line if it has a colon,
    # otherwise replace the whole span with just the raise (degenerate SCIP span).
    first = lines[start]
    if ':' in first:
        before, colon, _after = first.partition(':')
        return [before + colon + '\n', raise_line]
    return [raise_line]


def _delete_start(lines: list[str], start: int) -> int:
    """Extend the delete span up over contiguous decorator lines above *start*."""
    while start - 1 >= 0 and lines[start - 1].lstrip().startswith('@'):
        start -= 1
    return start


def _unified(path: str, before: str, after: str) -> str:
    diff = list(
        difflib.unified_diff(
            before.splitlines(keepends=True),
            after.splitlines(keepends=True),
            fromfile=f'a/{path}',
            tofile=f'b/{path}',
        )
    )
    if not diff:
        return ''
    # difflib emits source lines verbatim: a file whose last line lacks a trailing
    # newline yields a hunk line without '\n' and no marker, which `git apply` rejects
    # (and which would run into the next per-file diff once ''.join-ed). Re-terminate
    # such a line and append git's ``\ No newline at end of file`` sentinel.
    body: list[str] = []
    for line in diff:
        if line.endswith('\n'):
            body.append(line)
        else:
            body.append(line + '\n')
            body.append('\\ No newline at end of file\n')
    return f'diff --git a/{path} b/{path}\n' + ''.join(body)


def build_edit_patch(repo_root: str | Path, graph: nx.DiGraph, edits: list[tuple[str, str]]) -> dict:
    """Apply per-symbol :data:`STUB` / :data:`DELETE` edits; return the diff + inverse.

    *edits* is a list of ``(symbol, kind)``. Returns ``{patch, reverse, files,
    warnings}``: ``patch`` turns the original source into the edited state and
    ``reverse`` restores it. Line-based off each node's ``enclosing_range`` (no Python
    parse); symbols without a file/span, spans past EOF, and unreadable files are
    skipped and recorded in ``warnings``. Overlapping / nested edits in a file are
    de-duplicated (outermost wins) and applied bottom-up so line numbers hold.
    """
    repo = Path(repo_root)
    # file -> list of (start0, end0, kind, symbol)
    per_file: dict[str, list[tuple[int, int, str, str]]] = defaultdict(list)
    warnings: list[str] = []
    for sym, kind in edits:
        info = node_info(graph, sym)
        file = info.get('file')
        span = _span(info)
        if not file or span is None:
            warnings.append(f'skip {qualified_name(graph, sym)}: no file / enclosing_range')
            continue
        per_file[file].append((span[0], span[1], kind, sym))

    patch_parts: list[str] = []
    reverse_parts: list[str] = []
    touched: list[str] = []
    for file in sorted(per_file):
        path = repo / file
        if not path.is_file():
            warnings.append(f'skip {file}: not found under {repo}')
            continue
        src = path.read_text(encoding='utf-8')
        lines = src.splitlines(keepends=True)

        file_edits: list[tuple[int, int, list[str]]] = []  # 1-based inclusive range + replacement
        for start, end, kind, sym in per_file[file]:
            if start >= len(lines):
                warnings.append(f'skip {qualified_name(graph, sym)}: span past EOF in {file}')
                continue
            end = min(end, len(lines) - 1)
            if kind == DELETE:
                file_edits.append((_delete_start(lines, start) + 1, end + 1, []))
            else:
                file_edits.append((start + 1, end + 1, _stub_replacement(lines, start, end)))

        # Drop overlapping / nested edits, then apply bottom-up so line numbers hold.
        kept: list[tuple[int, int, list[str]]] = []
        cur_end = 0
        for edit in sorted(file_edits, key=lambda e: e[0]):
            if edit[0] <= cur_end:
                continue
            kept.append(edit)
            cur_end = max(cur_end, edit[1])
        for s, e, repl in sorted(kept, key=lambda x: x[0], reverse=True):
            lines[s - 1 : e] = repl

        after = ''.join(lines)
        if after == src:
            continue
        patch_parts.append(_unified(file, src, after))
        reverse_parts.append(_unified(file, after, src))
        touched.append(file)

    return {
        'patch': ''.join(patch_parts),
        'reverse': ''.join(reverse_parts),
        'files': touched,
        'warnings': warnings,
    }


def delete_functions_patch(repo_root: str | Path, graph: nx.DiGraph, symbols: list[str]) -> dict:
    """Unified diff that deletes each function in *symbols* from the repo (+ inverse).

    Thin wrapper over :func:`build_edit_patch` with every symbol marked
    :data:`DELETE`. Returns ``{patch, reverse, files, warnings}``: ``patch`` removes
    the functions, ``reverse`` restores them. Eval reverts a feature's tests this way.
    """
    return build_edit_patch(repo_root, graph, [(s, DELETE) for s in symbols])
