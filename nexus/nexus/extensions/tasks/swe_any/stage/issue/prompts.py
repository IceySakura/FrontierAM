"""Claude Code prompt for the Issue stage (write a non-leaking problem statement).

The working tree the agent sees is the exact rollout state: Issue has already applied
``feature_mask_patch`` + ``test_mask_patch``, so the feature is stubbed and its tests are hidden.
The host uploads only the two gold patches:

- ``feature_gold_patch`` - one reference implementation, used only to understand intended outcomes.
- ``test_gold_patch`` - the hidden evaluation oracle, used only to calibrate observable behaviour.

The masked tree is the authority for the task state. Its surviving public signatures and types are
the callable boundary the solver can inspect; the issue may describe their end-to-end purpose and
externally observable behaviour, but must hide helpers, fields, internal classes, control flow, and
other decomposition visible only in a gold patch.
"""

from __future__ import annotations


ISSUE_TEMPLATE = """\
You are writing a GitHub ISSUE that asks a developer to implement ONE feature. A later solver will
read ONLY your issue (not this prompt) and must be able to implement the feature and pass a hidden
test suite.

Repository root (you may read it): {working_dir}
The two mask patches have ALREADY been applied to this working tree. Do not look for separate mask
patches: the current files are the authoritative task state.

Feature (one-line summary, may be empty): {feature_desc}

READ these two private reference files:
- feature gold (ONE reference implementation - a hint, NOT a spec): {feature_gold_patch_path}
- test gold (the ACCEPTANCE CRITERION - the tests that grade the solution): {test_gold_patch_path}

Understand these execution states before writing:

- ROLLOUT: the solver receives a fresh copy of exactly the CURRENT masked repository plus your issue.
  The solver does NOT receive either gold patch or the hidden tests.
- EVALUATION: evaluation starts from that same masked repository, applies the solver's patch, then
  privately applies test_gold and runs it. feature_gold is NOT applied to the solver's solution.

Write the issue strictly from the CURRENT masked repository's point of view. Inspect that tree to
identify the public callables and types a developer can actually implement against. Assign
requirements only to caller-facing interfaces visible there; never introduce a code identifier that
is visible only in a gold patch. Any callable you mention must be described as one whole capability,
in terms of what a caller can observe. Do not expose how that capability is split into internal
pieces.
You may mention public types, errors, configuration terms, or other concepts already visible in the
masked repository when they are necessary to state the caller-facing contract, but do not restate
surrounding APIs the solver can inspect directly.
The visible callable boundary is a ceiling, not a checklist or an outline: do not try to mention
every retained signature, and do not give each callable its own section. Organize the issue around
cohesive user-visible capabilities and scenarios, not around functions, classes, files, or modules.
Mention a callable name only when needed to make such a capability clear.

Use feature_gold only to understand intended outcomes and test_gold only to calibrate which observable
cases matter. Details that are visible only inside either gold patch are not automatically part of the
issue. Include them only when they change behaviour observable through a public interface visible in
the masked tree; otherwise omit them. Ignore feature-independent regression guards.
Never turn either patch into a method inventory, architecture description, state-machine recipe, or
ordered implementation plan.
Do not reproduce an exhaustive matrix of literal inputs, outputs, constants, or edge cases from the
tests. Generalize them into behavioural categories and include only representative detail needed to
make the contract unambiguous.
A white-box assertion does not make private state part of the public contract. Translate it into an
observable outcome through a public interface visible in the masked tree when possible; otherwise
omit it. Do not require a particular internal representation, intermediate state, algorithm,
complexity, or execution order.

Write a clear, self-contained problem statement, as a user or maintainer would file it:
- Describe the feature's overall DESIRED behaviour from the outside: inputs, outputs,
  externally visible side effects, errors, and edge cases where relevant.
- Motivate it briefly (what is broken / missing today).
- Give enough information to implement the feature, and no more.
- Prefer cohesive scenarios and invariants over an inventory of source elements.
- Do not add a catalogue of surrounding modules, types, or collaborators.
- Do not add an implementation-notes section or a repository tour.

MUST NOT (this is the point of the task - no leakage):
- Do NOT reproduce, paraphrase line-by-line, or embed the reference implementation.
- Do NOT mention, name, quote, or describe the specific tests, test files, fixtures, or assertions.
- Do NOT reveal names, structure, algorithms, control flow, or implementation choices learned only
  from the gold patches.
- Do NOT expand a public capability into the hidden functions or components used to implement it.
- Do NOT turn the issue into a function-by-function transcription of feature_gold or test_gold.
- Do NOT edit any repository files - only WRITE the single JSON object below.

Draft the issue privately, then revise it before writing the file:
1. Structure audit: remove headings or lists organized around individual callables or source units.
2. Identifier audit: remove code identifiers that are not public contract terms visible in the
   masked repository.
3. White-box audit: remove private state, intermediate steps, mechanisms, and implementation
   constraints, even when a hidden test or gold patch exposes them.
4. Test-detail audit: collapse exhaustive vectors and literals into behavioural categories.
5. Grounding audit: ensure every remaining requirement is an observable contract exposed by the
   masked repository, rather than the reference implementation's decomposition.

WRITE exactly one JSON object to this path (overwrite if present):

    {output_path}

The file must contain ONLY this object (no markdown fences):

{{"problem_statement": <the issue text, as a single markdown string>}}
"""


def build_issue_prompt(
    *,
    working_dir: str,
    feature_desc: str,
    feature_gold_patch_path: str,
    test_gold_patch_path: str,
    output_path: str,
) -> str:
    """Render the Issue prompt.

    The host has already applied both mask patches to the working tree and uploads only the two gold
    references. The agent derives the public boundary from that masked tree and writes
    ``{problem_statement}`` to *output_path*.
    """
    return ISSUE_TEMPLATE.format(
        working_dir=working_dir,
        feature_desc=feature_desc or '(none given)',
        feature_gold_patch_path=feature_gold_patch_path,
        test_gold_patch_path=test_gold_patch_path,
        output_path=output_path,
    )
