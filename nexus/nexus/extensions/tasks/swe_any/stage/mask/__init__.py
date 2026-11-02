"""SWE-Any Mask — turn a selected feature into a re-implement-it hole.

Mirrors the ``heuristics`` + ``prompts`` split of the other stages:

- :mod:`.heuristics` (deterministic): classify a feature into signature-kept
  entries (called from outside the feature) vs fully deleted functions
  (:func:`classify_feature`; tests are never masked), then rewrite the source from
  the graph's line spans (:func:`apply_mask`) into a DRAFT mask patch (problem) +
  gold patch (solution).
- :mod:`.prompts`: the Claude Code pass that treats graph entries as a safety floor,
  preserves additional important feature interfaces when needed, and refines the draft
  into a clean, applyable patch.

Because helpers are deleted outright, the result is only gradable behaviourally,
so Mask is paired with an execution test oracle from ``eval``.
"""

from __future__ import annotations

from .heuristics import MaskPlan, apply_mask, classify_feature, find_entry_points
from .runner import run_mask_feature
from .schema import MaskTaskInput, MaskTaskOutput


__all__ = [
    'MaskPlan',
    'MaskTaskInput',
    'MaskTaskOutput',
    'apply_mask',
    'classify_feature',
    'find_entry_points',
    'run_mask_feature',
]
