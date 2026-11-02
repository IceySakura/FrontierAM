"""Python test classification (heuristic) over the entity graph.

scip-python (v0.6.x) almost never emits the SCIP Test role, so the ``is_test`` flag
from the graph build alone is unreliable. This module re-tags tests from pytest /
unittest file + name conventions:

- :func:`annotate_tests` re-classifies ``is_test`` on every internal node in place
  (a Parse enrichment step, run right after the graph is built);
- :func:`is_test_function` is a read-only predicate for "this node is itself a test
  case" (``test_*``), used by Eval to pick real test entry points.

Depends only on the protobuf-free ``graph`` model + stdlib, so it is shared by
``parse`` and ``eval`` without either importing the other. Demo scope: single-language
(Python), not pluggable.
"""

from __future__ import annotations

import re

from fnmatch import fnmatch

import networkx as nx

from .graph import node_info


KIND_CLASS = 'class'
KIND_FUNCTION = 'function'

TEST_FILE_GLOBS = ('test_*.py', '*_test.py', 'conftest.py')
TEST_DIR_PARTS = frozenset({'tests', 'test', 'testing'})
TEST_FUNC_RE = re.compile(r'^test[_A-Z0-9]')
TEST_CLASS_RE = re.compile(r'^Test([_A-Z0-9].*)?$')


def _is_test_file(path: str) -> bool:
    """True when *path* (repo-relative, POSIX) is a Python test file."""
    if not path:
        return False
    parts = path.split('/')
    if any(part in TEST_DIR_PARTS for part in parts[:-1]):
        return True
    return any(fnmatch(parts[-1], glob) for glob in TEST_FILE_GLOBS)


def _name_looks_like_test(name: str, kind: str) -> bool:
    if not name:
        return False
    if kind == KIND_FUNCTION:
        return bool(TEST_FUNC_RE.match(name))
    if kind == KIND_CLASS:
        return bool(TEST_CLASS_RE.match(name))
    return False


def is_test_function(graph: nx.DiGraph, symbol: str) -> bool:
    """True for a function that is itself a test case (name ``test_*``), not just
    test-support code that happens to live in a test file. Eval uses this to pick
    real test entry points out of the broader ``is_test`` pool."""
    if symbol not in graph:
        return False
    info = node_info(graph, symbol)
    return info.get('kind') == KIND_FUNCTION and bool(TEST_FUNC_RE.match(info.get('name') or ''))


def annotate_tests(graph: nx.DiGraph) -> dict[str, int]:
    """Re-classify ``is_test`` on internal nodes in place; return a small summary.

    Rule (strict): a node is test code **only when it lives in a test file** — every
    entity in a test module is test/support code, kept out of feature slices. A
    test-looking *name* (``test_*``) or scip's Test role are recorded as evidence but
    never flip ``is_test`` on their own: a function named ``test_connection`` sitting
    in application code is production code, not a test. Evidence is recorded under
    ``info['test_evidence']`` for provenance/audit. External nodes are left untouched
    (they have no source in this repo).
    """
    added = 0
    total = 0
    for symbol in graph.nodes:
        info = node_info(graph, symbol)
        if info.get('external'):
            continue
        in_test_path = _is_test_file(info.get('file') or '')
        evidence: list[str] = []
        if info.get('is_test'):
            evidence.append('scip_role')
        if in_test_path:
            evidence.append('path')
        if _name_looks_like_test(info.get('name') or '', info.get('kind') or ''):
            evidence.append('name')

        was_test = bool(info.get('is_test'))
        # Path is required: only a test-file location qualifies (name/scip are audit-only).
        is_test = in_test_path
        info['is_test'] = is_test
        info['test_evidence'] = evidence
        if is_test:
            total += 1
            if not was_test:
                added += 1
    return {'tests': total, 'added': added}
