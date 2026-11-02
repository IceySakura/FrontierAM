"""Pinned Claude Code prompt for the Select agentic pass.

One prompt per heuristic slice: the agent reads the slice's files read-only and
writes one JSON file to a pinned sandbox path, either emitting a coherent feature
or abandoning the slice. The host downloads, validates and archives that file.
"""

from __future__ import annotations

import json

from typing import Any

from ...utils.constants import DEFAULT_MAX_FINAL_NODES, DEFAULT_MIN_FINAL_NODES, UNBOUNDED_NODES


SELECT_PROMPT_TEMPLATE = """\
You are auditing whether a slice of a codebase forms ONE coherent feature.

Repository root (read-only): {working_dir}
Entity graph (read-only): {graph_path}

The entity graph is a FILTERED working view of the repository, not the raw index.
{graph_scope}Every id you emit must be copied verbatim from a node in that file
whose ``info.kind`` is ``"function"``. Never construct, guess or normalize an id
from what you read in the source code - a source-derived id will not match.

This check is all-or-nothing: ONE id that is not a function node in that file
throws away the whole result, and the slice is recorded as unresolved even
though the feature itself was right. So before you write the file, look up every
id in the graph again and confirm it appears there character for character.

Candidate entities from a heuristic slice (JSON). Prefer focusing on
``kind == "function"`` (methods/functions) for the final selection -
classes are context only:

{candidates_json}

Your job:
1. Start from the candidates and read their files/functions under the repository
   root. Read-only for source code - do not modify repository source files.
2. Judge whether naming, call relations, and docstrings/signatures point to the
   SAME feature (one cohesive capability), or whether the slice mixes unrelated
   concerns.
3. If the candidates DO point at a feature, produce the ``feature_node_ids`` list for it.
   Omit candidates that are inconsistent with it, and add the functions needed to
   complete THAT feature (same module / class / call cluster) by inspecting the
   entity graph and source files. Do not include a function merely because it is
   adjacent; include it only when it is semantically part of the same feature.
4. If the candidates do NOT point at any feature worth extracting - auto-generated
   boilerplate, unrelated fragments, trivial accessors - STOP immediately. Do not
   search the repository for some other feature to report: an abandoned slice is
   an expected, useful outcome. Take the abandon exit below instead.

The candidates are the subject of the audit, not a hint to start from. So check your
``feature_node_ids`` list against them before writing it: if it shares few or none of its
functions with the candidates, you have drifted onto a different feature, however coherent that
feature is. Drift is NOT a success - report it as unresolved via the abandon exit,
saying which feature you drifted onto.

Testability - a downstream step turns this feature into a "re-implement it" task graded by
running tests, so only include a function in ``feature_node_ids`` if its behaviour can be
verified by a test in THIS repository's environment: it is EITHER already exercised by an
existing test, OR a black-box test for it (through its own inputs/outputs) could be written and
actually RUN here. EXCLUDE a function whose only real work is to shell out to an external
binary / service / network that may be absent, load a large trained model, or otherwise cannot
be exercised offline - such code cannot be graded even though it may look central. Keep the
feature to the testable core; if dropping the untestable parts leaves no coherent, testable
feature, abandon the slice instead.
{size_rule}
When finished, WRITE exactly one JSON object to this path (create parent dirs if
needed; overwrite if present):

    {output_path}

The file must contain ONLY this JSON object (no markdown fences). There are
exactly two valid shapes.

Success - the slice yielded one complete, coherent feature:

{{"feature": <str>, "resolved": true, \
"feature_node_ids": [<function entity ids from the entity graph>], "abandoned": false}}

Abandon - nothing worth extracting here:

{{"feature": "", "resolved": false, "feature_node_ids": [], "abandoned": true, \
"abandon_reason": <one short sentence>}}

Do not invent a third shape: never report ``resolved: false`` with a non-empty
``feature_node_ids``. After writing the file, print a short confirmation to stdout. Do not
attempt network access or run the test suite.
"""


def _final_size_rule(min_final_nodes: int, max_final_nodes: int) -> str:
    """Render the ``feature_node_ids`` size requirement; empty when neither bound is active."""
    bounded_below = min_final_nodes > 1
    bounded_above = max_final_nodes < UNBOUNDED_NODES
    if bounded_below and bounded_above:
        demand = f'between {min_final_nodes} and {max_final_nodes} functions, inclusive'
        advice = (
            'Do not pad it with loosely related functions to reach the lower bound, and do\n'
            'not cut a feature in half to fit the upper bound'
        )
    elif bounded_below:
        demand = f'at least {min_final_nodes} functions'
        advice = 'Do not pad it with loosely related functions to reach that count'
    elif bounded_above:
        demand = f'at most {max_final_nodes} functions'
        advice = 'Do not cut a feature in half to fit under that count'
    else:
        return ''
    return (
        f'\nSize requirement: ``feature_node_ids`` must contain {demand}.\n'
        f'{advice} - if the feature you found\n'
        'cannot satisfy this, abandon the slice instead. The host rejects a list\n'
        'outside the allowed size.\n'
    )


def build_select_prompt(
    *,
    working_dir: str,
    graph_path: str,
    output_path: str,
    candidates: list[dict[str, Any]],
    include_external: bool = False,
    min_final_nodes: int = DEFAULT_MIN_FINAL_NODES,
    max_final_nodes: int = DEFAULT_MAX_FINAL_NODES,
) -> str:
    """Render the pinned Select consistency-check prompt with embedded candidates.

    The graph-scope sentence is rendered from the actual filter settings, so the
    agent is never told a symbol class is missing when it is present.
    """
    graph_scope = (
        'It includes external / third-party symbols.\n'
        if include_external
        else 'External / third-party symbols have been removed from it.\n'
    )
    # Strip long documentation to keep the prompt compact.
    compact = []
    for node in candidates:
        compact.append({
            'id': node.get('id'),
            'name': node.get('name'),
            'kind': node.get('kind'),
            'file': node.get('file'),
            'def_line': node.get('def_line'),
            'module': node.get('module'),
            'is_test': node.get('is_test'),
            'documentation': (node.get('documentation') or '')[:240],
        })
    return SELECT_PROMPT_TEMPLATE.format(
        working_dir=working_dir,
        graph_path=graph_path,
        graph_scope=graph_scope,
        output_path=output_path,
        candidates_json=json.dumps(compact, indent=2, ensure_ascii=False),
        size_rule=_final_size_rule(min_final_nodes, max_final_nodes),
    )
