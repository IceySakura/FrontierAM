"""Eval deterministic core — find the feature's native, runnable test cases.

Given the entity graph and a feature (function ids), :func:`find_candidate_tests`
collects the repo's own tests that already exercise it, restricted to **final test
functions**: a node that

- lives in a test file (``is_test`` — see ``utils.classify_tests``),
- is itself a test case (name ``test_*`` — :func:`is_test_function`), and
- is a *leaf*: not called by any other function (so pytest-style helpers, which are
  invoked by other tests, are excluded — we only want the tests the runner drives).

These ids (the long SCIP symbol strings, carrying file + qualified name) are the
candidate list shown to the agent, which selects the subset that actually covers the
feature and may add new non-conflicting tests. Turning the selected ids into a
mask/restore patch pair is the shared ``utils.source_edit.delete_functions_patch``
primitive, driven host/sandbox-side by the task — not here.

Coverage note: only *direct* ``reference`` edges are followed, so a test that reaches
the feature only through a fixture/helper indirection is missed.
"""

from __future__ import annotations

from collections import defaultdict

import networkx as nx

from ...utils.classify_tests import is_test_function
from ...utils.graph import callers, node_info, qualified_name


def find_candidate_tests(
    graph: nx.DiGraph, feature_ids: list[str], *, include_feature_tests: bool = True
) -> list[dict]:
    """Return the feature's final (leaf) native test cases, best first.

    Each entry: ``{id, name, file, covers}`` where ``covers`` lists the feature
    function ids the test directly references. Only real, runnable test cases are
    returned — test-file helper functions (which have callers) are excluded — so the
    agent selects from tests the runner would actually invoke. Ranked by how many
    feature functions each touches, then by name.

    *include_feature_tests*: Select runs with ``include_tests=true`` and may pull test
    functions into the feature. Mask never masks tests (it keeps them intact), so such
    a test still exercises the masked feature and is a valid oracle — kept by default.
    Set False to treat a feature-member test as part of the feature and drop it from
    the candidates (e.g. if masking is ever extended to tests).
    """
    feature_set = set(feature_ids)
    fs = [f for f in feature_ids if f in graph and node_info(graph, f).get('kind') == 'function']
    covers: dict[str, set[str]] = defaultdict(set)
    for f in fs:
        for t in callers(graph, f):
            info_t = node_info(graph, t)
            # A final test case: in a test file (is_test), named test_* (is_test_function),
            # and a leaf — no other function calls it, so helpers are filtered out.
            if not (info_t.get('is_test') and is_test_function(graph, t) and not callers(graph, t)):
                continue
            # A test Select pulled into the feature is still valid (mask leaves it intact)
            # unless the caller opts to treat feature-member tests as part of the feature.
            if not include_feature_tests and t in feature_set:
                continue
            covers[t].add(f)

    out: list[dict] = []
    for test, covered in covers.items():
        info = node_info(graph, test)
        out.append({
            'id': test,
            'name': qualified_name(graph, test),
            'file': info.get('file') or '',
            'covers': sorted(covered),
        })
    out.sort(key=lambda r: (-len(r['covers']), r['name']))
    return out
