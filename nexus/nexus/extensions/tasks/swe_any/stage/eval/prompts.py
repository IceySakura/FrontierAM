"""Claude Code prompt for the Eval test-oracle pass (mask all feature tests + author the gold oracle).

The deterministic step (:func:`heuristics.find_candidate_tests`) lists the feature's final (leaf)
native test cases that *directly* reference the feature - a HINT only (it misses tests that reach
the feature via a fixture/helper). This agentic pass shows that hint plus the feature's mask/gold
patches (by path) and asks the agent to produce TWO patches that are NOT inverses plus one grade-only
test runner:

1. ``test_mask_patch`` - delete/hide EVERY existing repo test that exercises the feature (may promote
   to whole test classes / files), so a later solver never sees the oracle. The agent must go beyond
   the candidate hint and find them all.
2. ``test_gold_patch`` - the FORWARD grade-time oracle: generalized / newly-authored black-box tests
   that cover every substantive observable capability removed by the feature mask, plus optional
   feature-INDEPENDENT tests kept as PASS_TO_PASS regression guards. It does NOT restore the masked
   originals verbatim.
3. ``test_runner.sh`` - a one-command adapter for this repository's actual environment/test framework.
   It writes per-test JSONL to ``$SWE_ANY_RESULTS_PATH`` so pytest, unittest, Django, and other native
   runners share one grading protocol.

At grade time the harness applies ``feature_mask + test_mask + test_gold`` (masked state -> some tests
FAIL) and then ``feature_gold`` (fixed state -> all tests PASS, no pass->fail). The agent is asked to
self-verify this in the sandbox. A second independent agent reviews the authored oracle before the
host accepts it.

The agent WRITES the two diffs as RAW unified-diff files plus the shell runner to three fixed sandbox
paths the host then downloads.
"""

from __future__ import annotations

import json

from typing import Any


TEST_ORACLE_TEMPLATE = """\
You are building the TEST ORACLE for ONE code feature. These tests decide whether a later
re-implementation of the feature is correct, so design them carefully.

Repository root (you may read AND run commands in it): {working_dir}

Reference patches for this feature. READ BOTH in full before deciding test scope:
- feature mask (the exact source changes that create the solver-facing hole): {feature_mask_path}
- feature gold (the reference implementation that restores that hole): {feature_gold_path}

These two patches, the repository's public usage, and its native tests define the feature boundary.
Do not reduce the feature to only the functions whose signatures happen to survive the mask. The mask
may delete important public callables entirely when static call-graph analysis missed their external
role. Inspect every changed callable, then group the patch into substantive caller-visible
capabilities. Shared private helpers need not be tested directly, but every observable capability
removed by the mask must be exercised through an appropriate public interface.

CANDIDATE native tests that DIRECTLY reference the feature (a HINT - NOT exhaustive; tests that reach
the feature only through a fixture/helper are missing). JSON list (MAY be empty), each with its SCIP
``id``, qualified ``name``, ``file`` and the feature functions it ``covers``:

{candidates_json}

DRAFT heuristic test patches (mechanical starting points over the candidate tests - read and improve
on them; they are hints, not independently exportable oracles):
- test_mask draft (deletes ALL candidate tests): {heuristic_test_mask_path}
- test_gold draft (restores ALL candidate tests verbatim): {heuristic_test_gold_path}

Produce TWO patches (they are NOT inverses of each other) and ONE test runner:

1. test_mask_patch - DELETE every EXISTING repo test that exercises this feature, so the solver can
   never read the oracle. Start from the draft, but you MUST find and delete ALL of them, including
   ones the candidate hint missed (search the repo). Promote to deleting a whole test class or whole
   test file when all of its tests exercise the feature. Only touch TEST files; NEVER touch the
   feature's own source files. Apply cleanly with ``git apply``.

2. test_gold_patch - the FORWARD grade-time oracle. It must:
   - COVER every substantive externally observable capability removed by feature_mask, including
     distinct behaviours and edge cases, through the repository's PUBLIC interface only. Cross-check
     the full feature patches and deleted originals so you do not silently narrow a large feature to
     one helper, one retained signature, or one subcomponent.
   - Be GENERALIZED, not word-by-word copies of the originals: assert observable behaviour, not
     implementation internals, so a reasonable alternative implementation also passes. (But it must
     still PASS on the reference/gold code - do not over-generalize past what gold does.)
   - Optionally ADD a few feature-INDEPENDENT, stable tests (on unrelated modules the feature does
     not touch) as PASS_TO_PASS regression guards.
   - Apply cleanly with ``git apply`` onto the CURRENT repo (new files/dirs allowed), and must RUN in
     the current environment (no external binaries/services/network that may be absent).

3. test_runner.sh - the GRADE-ONLY one-command runner for this repo. The harness invokes the SAME
   script in the masked/pregold and fixed/postgold trees with:
   - current directory and ``$SWE_ANY_REPO_DIR`` = repository root;
   - ``$SWE_ANY_TEST_FILES_JSON`` = JSON array of existing paths touched by test_gold;
   - ``$SWE_ANY_RESULTS_PATH`` = file your script MUST overwrite with structured JSONL.

   Choose the environment and native framework that actually work here (pytest, unittest, Django,
   or another repository runner). Run ONLY the supplied test files / their equivalent native test
   labels, deduplicated. Do not promote package markers (for example ``__init__.py``) or framework
   support files (for example ``conftest.py``) into broader package/directory test targets. If the
   repo's default test settings require an unavailable database/service, select a repository-supported
   hermetic test configuration instead of submitting a runner that reports an environment failure.
   Raw diagnostics may go to stdout/stderr. The result file must contain:
   - one line per observed test:
     ``{{"type":"test","id":"stable opaque id","status":"PASSED"}}``
     where status is PASSED, FAILED, ERROR, SKIPPED, XFAIL, or XPASS;
   - exactly one final summary:
     ``{{"type":"summary","status":"completed","count":N}}``.
   Test ids MUST be stable across pregold and postgold AND UNIQUE within each result stream. Prefer
   the framework's full node id, including module path and parameterization. If the framework reports
   call/setup/teardown outcomes separately for one logical test, merge them into one record using the
   most severe status (ERROR > FAILED > XPASS > SKIPPED > XFAIL > PASSED). A test failure is a normal
   completed run: still write all records + the completed summary and exit 0. If the
   runner/environment itself cannot execute, write only:
   ``{{"type":"summary","status":"runner_error","error_kind":"...","message":"..."}}``
   and exit 2. Do not fake test records from expected outcomes.
   Collection/import failures caused by the masked feature are test ``ERROR`` outcomes (or absent
   pregold ids when the framework cannot enumerate them), NOT runner_error; still finish normally.

   The script must NOT apply/reverse patches or edit implementation/tests: the harness prepares each
   tree. It may only perform environment setup and execute/observe the supplied oracle tests.

CONSISTENCY (mask x gold): grading runs every test that exists in the test files AFTER applying
test_mask then test_gold. So anything you want GRADED (feature tests -> FAIL_TO_PASS, guards ->
PASS_TO_PASS) must be present after mask+gold - if you reuse an existing test as a guard, do NOT
delete it in test_mask (or re-add it in test_gold). Anything you want HIDDEN must be deleted by
test_mask.

SELF-VERIFY (do this in the sandbox before writing your answer; run ONLY the relevant tests, never
the whole suite, to save time):
- Apply feature_mask + your test_mask + your test_gold, invoke your test_runner.sh: at least one
  emitted test MUST be non-PASS (there is a FAIL_TO_PASS), and any guard tests must PASS.
- Then apply feature_gold (the reference impl), invoke the SAME script: ALL emitted tests MUST PASS -
  no test may go from pass to fail (no P2F). If something fails on gold, fix the test or runner.
- Inspect both JSONL files: every test id is unique, the summary count equals the number of test
  records, and neither invocation reports ``runner_error``.

Hard rules:
- Black-box only: exercise caller-visible feature behaviour through public repository interfaces;
  never assert on private helpers or internal representation (the implementation will be rewritten).
- test_mask and test_gold may only touch TEST files; NEVER modify the feature's own source files.
- Do NOT leave any repo edits behind after self-verification.

WRITE all three artifacts to these exact paths, overwriting if present:

- test_mask_patch (deletes all feature tests; leave EMPTY if you delete no test) -> {mask_output_path}
- test_gold_patch (the forward oracle + optional guards)                          -> {gold_output_path}
- test_runner.sh (executable shell source; no markdown fence)                    -> {runner_output_path}
"""


TEST_ORACLE_VERIFIER_TEMPLATE = """\
You are the independent QUALITY GATE for a test oracle authored for ONE code feature.
Do not trust the author agent's self-review. Read the repository and every artifact below:

Repository root (read-only): {working_dir}
- feature mask (solver-facing implementation hole): {feature_mask_path}
- feature gold (reference implementation): {feature_gold_path}
- authored test mask: {test_mask_path}
- authored test gold: {test_gold_path}
- authored structured test runner: {runner_path}

Return ONE primary integer verdict from this stable taxonomy:

- -1 OTHER: reject for a material issue that does not fit a known category below.
-  0 CLEAN: no material issue; the oracle is safe and strong enough.
-  1 ORACLE_LEAKAGE: test_mask fails to hide relevant existing tests/assertions/fixtures, or another
     solver-visible artifact reveals the hidden oracle/reference answer.
-  2 OUT_OF_SCOPE_OR_DESTRUCTIVE: test patches modify implementation, unrelated tests, shared
     fixtures/configuration, or other scope in a way that can disturb existing functionality; or the
     runner grades unrelated/broader targets.
-  3 OVERCONSTRAINED: tests assert private internals, exact algorithms/representations, or behaviour
     not guaranteed by the public contract, so a correct compatible implementation may fail.
-  4 TOO_WEAK: tests miss substantive public capabilities or important edge cases, contain only
     smoke/existence/happy-path/tautological assertions, mock or bypass the feature, or can pass on
     the masked hole.
-  5 EASILY_GAMED: a solver can pass without implementing the feature by hard-coding a tiny fixed
     example set, detecting test-specific constants/paths, exploiting source/text inspection, or
     special-casing predictable oracle inputs.
-  6 RUNNER_INTEGRITY: the runner omits authored tests, fabricates statuses, swallows failures,
     emits unstable/colliding ids, edits the repo/applies patches, or otherwise makes results
     untrustworthy.
-  7 NON_HERMETIC_OR_FLAKY: tests depend on network/services, wall-clock timing, randomness, order,
     mutable global state, unavailable binaries, or other unstable environment assumptions.

If several issues exist, choose the most direct/exploitable material issue as ``verdict`` and include
ALL issues in ``findings``. Prefix each finding with its category, for example ``"[4] misses retry
semantics"``. Use ``-1`` only for a material issue genuinely outside 1..7.

This is a semantic review, not just a syntax/apply check. You may run read-only inspection commands
and relevant tests, but MUST NOT edit repository files or any supplied artifact. Be conservative but
evidence-based: do not reject merely because coverage is not exhaustive or because tests use normal
repository fixtures.

WRITE exactly one JSON object to this path, with no markdown fence:

{verdict_output_path}

The object must have exactly this shape:
{{
  "verdict": 0,
  "rationale": "Concise overall explanation grounded in specific feature/test evidence.",
  "findings": []
}}

For verdict 0, findings MUST be empty. For every non-zero verdict, findings MUST contain at least one
concrete issue. The host accepts only verdict 0; every other value is fail-closed.
"""


_NO_DRAFT = '(none - no usable mechanical draft; author from candidates and feature context)'


def build_test_oracle_prompt(
    *,
    working_dir: str,
    candidates: list[dict[str, Any]],
    mask_output_path: str,
    gold_output_path: str,
    runner_output_path: str,
    feature_mask_path: str,
    feature_gold_path: str,
    heuristic_test_mask_path: str,
    heuristic_test_gold_path: str,
) -> str:
    """Render the test-oracle prompt.

    *candidates* is the (possibly empty) hint list from :func:`heuristics.find_candidate_tests`.
    All patches are referenced BY PATH: the feature
    *feature_mask_path* / *feature_gold_path* (from Mask, read-only "what is the feature" context) and
    the two heuristic test drafts *heuristic_test_mask_path* (delete all candidates) /
    *heuristic_test_gold_path* (restore them verbatim) - mechanical starting-point hints. A draft
    path is empty when the feature has no native candidate tests or draft construction failed; it is
    shown as ``(none ...)`` and does not block the author/refine pass. The agent WRITES its two raw
    unified diffs to *mask_output_path* (test_mask) and *gold_output_path* (test_gold), plus the
    repository-specific structured runner to *runner_output_path*.
    """
    compact = [
        {'id': c.get('id'), 'name': c.get('name'), 'file': c.get('file'), 'covers': c.get('covers')} for c in candidates
    ]
    return TEST_ORACLE_TEMPLATE.format(
        working_dir=working_dir,
        candidates_json=json.dumps(compact, indent=2, ensure_ascii=False),
        feature_mask_path=feature_mask_path,
        feature_gold_path=feature_gold_path,
        heuristic_test_mask_path=heuristic_test_mask_path or _NO_DRAFT,
        heuristic_test_gold_path=heuristic_test_gold_path or _NO_DRAFT,
        mask_output_path=mask_output_path,
        gold_output_path=gold_output_path,
        runner_output_path=runner_output_path,
    )


def build_test_oracle_verifier_prompt(
    *,
    working_dir: str,
    feature_mask_path: str,
    feature_gold_path: str,
    test_mask_path: str,
    test_gold_path: str,
    runner_path: str,
    verdict_output_path: str,
) -> str:
    """Render the independent semantic quality-gate prompt for an authored oracle."""
    return TEST_ORACLE_VERIFIER_TEMPLATE.format(
        working_dir=working_dir,
        feature_mask_path=feature_mask_path,
        feature_gold_path=feature_gold_path,
        test_mask_path=test_mask_path,
        test_gold_path=test_gold_path,
        runner_path=runner_path,
        verdict_output_path=verdict_output_path,
    )
