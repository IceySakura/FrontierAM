"""SWE-Any Eval — build the test oracle for a masked feature.

Mirrors the ``heuristics`` + ``prompts`` split of the other stages:

- :mod:`.heuristics` (deterministic): :func:`find_candidate_tests` mines the
  repo's own **final (leaf) test cases** that reference the feature — the runnable
  ``test_*`` functions in test files, excluding helpers. A HINT for the agent.
- :mod:`.prompts`: the Claude Code pass that authors TWO patches (NOT inverses) plus
  a grade-only structured ``test_runner.sh``:
  ``test_mask_patch`` deletes EVERY existing repo test that exercises the feature
  (hiding the oracle from the solver) and ``test_gold_patch`` is the FORWARD, generalized
  grade-time oracle (black-box tests covering every substantive observable capability in
  the feature patches + optional feature-independent P2P guards). The required runner adapts
  the repository's real test framework to stable per-test JSONL.
- a second independent Claude Code pass reads the authored oracle and classifies leakage,
  destructive scope, over-constraint, weakness, easy gaming, runner-integrity, and flaky-environment
  failures with a stable integer verdict.

When either agent fails or the verifier rejects, the feature is rejected.

At grade time the harness applies ``feature_mask + test_mask`` (solver-facing state) then
``feature_gold + test_gold`` (graded state) and derives FAIL_TO_PASS / PASS_TO_PASS by
execution, rejecting on any pass→fail regression.
"""

from __future__ import annotations

from .heuristics import find_candidate_tests
from .runner import run_eval_feature
from .schema import EvalTaskInput, EvalTaskOutput


__all__ = [
    'EvalTaskInput',
    'EvalTaskOutput',
    'find_candidate_tests',
    'run_eval_feature',
]
