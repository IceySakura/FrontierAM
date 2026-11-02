"""Claude Code prompt for the Mask refine pass.

The deterministic step (:mod:`.heuristics.apply_mask`) produces a DRAFT mask patch
from graph boundaries. The refine agent keeps those boundary signatures as a safe
minimum, promotes other important feature callables when the graph missed their
public role, and returns one clean, applyable mask patch.
"""

from __future__ import annotations

import json

from typing import Any


MASK_REFINE_TEMPLATE = """\
You are turning a code feature into a "re-implement it" exercise by masking it.

Repository root (read-only for source): {working_dir}

Selected feature (one-line summary, may be empty): {feature_desc}

Intended masking rules:
- The heuristic boundary entries below are a MINIMUM: KEEP every one of their signatures.
- Review every candidate helper below. If repository source, tests, documentation, or normal usage
  shows that a callable is an important interface for configuring, invoking, composing, serializing,
  or otherwise using THIS selected feature, promote it by KEEPING its signature too. The call graph
  can miss dynamic calls and external library users, so lack of an outside caller is not proof that
  a function is an internal helper.
- For every kept callable, preserve its ``def`` line(s), parameters, return annotation, decorators
  needed for calling it, and optionally its docstring; replace its body with one
  ``raise NotImplementedError``.
- DELETE only functions that are genuinely implementation-only helpers for this feature, including
  their decorators, signatures, and bodies. Do not keep a function merely because it is nearby or
  participates internally in the implementation.
- Do NOT touch anything else - especially test functions/files must stay intact.
- Every file you change must remain valid, importable Python.

Heuristic boundary entries (mandatory minimum; keep signature). JSON: id -> qualified name:

{entries_json}

Candidate helpers (review each; either promote and keep its signature, or delete entirely).
JSON: id -> qualified name:

{deletes_json}

A DRAFT unified diff was produced mechanically from line spans (approximate - its
signature/decorator/docstring boundaries may be slightly off) and written to:

    {heuristic_patch_path}

Your job: read that draft patch, inspect the listed functions and their user-facing usage under the
repository root, decide the final keep-vs-delete boundary, then produce the CORRECT unified diff that
applies cleanly with ``git apply`` and realises the rules above. The draft's helper classification is
only a starting point; its boundary-entry classification remains the mandatory safety floor.

SCOPE: modify ONLY the files that hold the entry/helper functions listed above.
Keep every hunk inside those files and the change minimal - edits that stray to
other files are recorded as out-of-scope and count against the patch's quality.

When finished, WRITE exactly one JSON object to this path (overwrite if present):

    {output_path}

The file must contain ONLY this object (no markdown fences):

{{"mask_patch": <unified diff string>}}
"""


def build_mask_refine_prompt(
    *,
    working_dir: str,
    feature_desc: str,
    output_path: str,
    heuristic_patch_path: str,
    entries: dict[str, str],
    deletes: dict[str, str],
) -> str:
    """Render the mask refine prompt.

    *entries* / *deletes* map the heuristic split to qualified names; the former is
    mandatory while the latter is open to agentic promotion. *heuristic_patch_path*
    is the sandbox path of the raw draft patch (``run_mask`` writes it there).
    """
    return MASK_REFINE_TEMPLATE.format(
        working_dir=working_dir,
        feature_desc=feature_desc or '(none given)',
        output_path=output_path,
        heuristic_patch_path=heuristic_patch_path,
        entries_json=json.dumps(entries, indent=2, ensure_ascii=False),
        deletes_json=json.dumps(deletes, indent=2, ensure_ascii=False),
    )


def name_map(graph: Any, ids: list[str]) -> dict[str, str]:
    """Helper for callers: build the ``id -> qualified name`` map the prompt wants."""
    from ...utils.graph import qualified_name

    return {i: qualified_name(graph, i) for i in ids}
