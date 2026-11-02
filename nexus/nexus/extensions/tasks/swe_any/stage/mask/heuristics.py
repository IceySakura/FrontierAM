"""Mask a selected feature into a to-be-implemented hole (deterministic draft).

Simplified rules for this round (a draft the agent then refines):

- **test functions** are never touched — kept whole (they are the oracle).
- **entries** = feature functions called from OUTSIDE the feature (by tests or by
  other repo code). Their signature is kept, the body becomes ``raise
  NotImplementedError`` — this is the API the solver re-implements against.
- **deletes** = every other (non-test) feature function. Removed entirely.

This split is dangle-safe for *function-body* references: anything referenced from
outside the delete set (external code, tests, or a kept entry *signature*) is itself
an entry, so a deleted function is only ever referenced by bodies that are gone. It
is NOT a full guarantee — the reference graph only sees calls inside a function's
enclosing_range, so a delete referenced at module/class-body level or reached only
indirectly can still dangle (broken import/collection). Those are not prevented here;
they surface when the mask/gold patches are execution-validated downstream.

The source edit is driven by the SCIP graph's ``enclosing_range`` (line spans),
not a Python parse — so it is crude (signature/decorator/docstring boundaries are
approximate). The resulting ``mask_patch`` is a DRAFT handed to an agent
(:mod:`.prompts`) that reads the repo and produces the clean, valid patch.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import networkx as nx

from ...utils.graph import callers, node_info
from ...utils.source_edit import DELETE, STUB, build_edit_patch


def _feature_functions(graph: nx.DiGraph, feature_ids: list[str]) -> list[str]:
    """Feature entities that are maskable functions — excludes classes and tests."""
    out: list[str] = []
    for f in feature_ids:
        if f not in graph:
            continue
        info = node_info(graph, f)
        if info.get('kind') == 'function' and not info.get('is_test'):
            out.append(f)
    return out


def _feature_classes(graph: nx.DiGraph, feature_ids: list[str]) -> list[str]:
    return [c for c in feature_ids if c in graph and node_info(graph, c).get('kind') == 'class']


def find_entry_points(graph: nx.DiGraph, feature_ids: list[str]) -> set[str]:
    """Feature functions called from OUTSIDE the feature (tests or other code).

    Their signature is kept so those callers still resolve; everything else in the
    feature is deleted. Tests count as outside the feature, so a tested function is
    always an entry.
    """
    fs = set(_feature_functions(graph, feature_ids))
    return {f for f in fs if callers(graph, f) - fs}


@dataclass
class MaskPlan:
    """What a mask keeps (signature) vs deletes for one feature."""

    nodes: list[str]
    entries: list[str]
    deletes: list[str]
    classes: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        # ``warnings`` is intentionally omitted: run_mask surfaces the classification warnings
        # merged with the apply-mask warnings in ONE top-level ``warnings`` field, so keeping a
        # second copy inside the plan would be redundant.
        return {
            'nodes': self.nodes,
            'entries': self.entries,
            'deletes': self.deletes,
            'classes': self.classes,
        }


def classify_feature(graph: nx.DiGraph, feature_ids: list[str]) -> MaskPlan:
    """Split a feature into signature-kept entries vs fully deleted functions."""
    fs = _feature_functions(graph, feature_ids)
    entries = set(find_entry_points(graph, feature_ids))
    deletes = [f for f in fs if f not in entries]
    warnings: list[str] = []
    if fs and not entries:
        warnings.append(
            'no entry points: every feature function is internal-only, so the mask '
            'deletes the whole feature and keeps no signatures for the solver to anchor on'
        )
    return MaskPlan(
        nodes=fs,
        entries=sorted(entries),
        deletes=sorted(deletes),
        classes=_feature_classes(graph, feature_ids),
        warnings=warnings,
    )


def apply_mask(repo_root: str | Path, graph: nx.DiGraph, plan: MaskPlan) -> dict:
    """Rewrite the repo source per *plan* using graph line spans; return patches.

    ``mask_patch`` turns the original repo into the masked (problem) state and
    ``gold_patch`` is its inverse (the reference solution). Thin wrapper over
    :func:`..source_edit.build_edit_patch`: entries are stubbed (signature kept, body
    → ``raise NotImplementedError``), the rest deleted. Demo scope: line edits from
    each node's ``enclosing_range`` (no Python parse), so boundaries are approximate —
    the patch is a DRAFT for the agent refine pass. Nodes without an ``enclosing_range``
    and unreadable files are skipped and recorded in ``warnings``.
    """
    edits = [(sym, STUB) for sym in plan.entries] + [(sym, DELETE) for sym in plan.deletes]
    out = build_edit_patch(repo_root, graph, edits)
    return {
        'mask_patch': out['patch'],
        'gold_patch': out['reverse'],
        'files': out['files'],
        'warnings': out['warnings'],
    }
