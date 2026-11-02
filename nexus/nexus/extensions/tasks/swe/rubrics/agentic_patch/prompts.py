"""Prompt templates for rubric generation agent and rubric evaluation LLM judge."""

# ============== Rubric Generation Agent Prompts ==============

GEN_SYSTEM_PROMPT = """\
You are an expert code reviewer that can understand issues and are well versed in the codebase. \
Your job is to write high-quality rubrics to grade the solution to a given issue.
You SHOULD NOT attempt to solve the task yourself, but rather understand the task, \
go through the codebase, and write rubrics only.
In each turn, include a brief summary of your thinking before any tool calls."""

GEN_INSTANCE_TEMPLATE = """\
<uploaded_files>
{working_dir}
</uploaded_files>

<gold_patch>
{gold_patch}
</gold_patch>

I've uploaded a code repository in the directory {working_dir}. Consider the following PR description:
<pr_description>
{problem_statement}
</pr_description>

Can you help me write high quality rubrics to grade the solution to the task described in the <pr_description>?
You have access to the <gold_patch>, which is the reference implementation for this task.
You can use the gold patch to ground your rubrics in the actual correct implementation path.

Follow these steps to write the rubrics:
1. As a first step, it might be a good idea to find and read code relevant to the <pr_description> by searching the codebase using search tools.
2. Then think of the approach to solve the task (functional requirements, non-functional requirements, etc.)
3. Also understand how the codebase is structured and how the code is organized, as well as coding style, etc.
4. Write a list of rubrics that can be used to grade the solution to the task described in the <pr_description>. Your rubrics should be along the axes described below.
5. Then finally, make a new file in the {working_dir} directory called `rubrics.yaml` with the rubrics you wrote. It should be a valid YAML file with the structure described later below.
6. This should be the only file you create in the {working_dir} directory. DO NOT create/modify/delete any other files or directories.
7. Finally, submit the task.

Atomicity:
  - Each rubric criterion should evaluate exactly one distinct aspect.
  - Avoid bundling multiple criteria into a single rubric. Most stacked criteria with the word "and" can be broken up into multiple pieces.

Self-containment & specificity (strict):
  - Do NOT write generic items; bind each item to exact paths/symbols/tokens seen in this instance.
  - Never rely on cross-item references; each item stands alone with its own identifiers and patterns.
  - The judge will only have access to the problem and patch and the current rubric under evaluation, so make sure the rubric can be evaluated without any other information.

Mutually Exclusive, Collectively Exhaustive (MECE):
  - The rubric set should be mutually exclusive and collectively exhaustive.

Style constraints (strict):
  - YAML only — no prose outside YAML.
  - Avoid backslash-heavy patterns; if you absolutely must include one, double any backslashes so the YAML stays valid.
  - Each rubric description starts with a third-person singular verb (e.g., Identifies, Implements, Validates, Confirms, Avoids, Cleans up, Plans).
  - Make descriptions concrete using tokens from PATCH/PR_DESCRIPTION.
  - Each rubric item includes: id (short), description (verb-first, instance-grounded), weight (int; 1=nice, 2=valuable, 3=must).
  - Avoid double-counting: do not re-score the same behavior under multiple items.

Pattern-writing guidelines (keep literal YAML-friendly text; no regex required):
  - Use plain path mentions like "diff --git a/path/to/file.py" or "+++ b/path/to/file.py".
  - Refer to symbols with straightforward phrases such as "def my_function(" instead of regex classes.
  - Describe value patterns in words (e.g., "string containing total") instead of complex expressions.
  - If you need to forbid something, just mention the exact string like '@pytest.mark.skip', etc.

Axes (execution-free):
  - file_change_rubrics (4-8): Scope, locality, and sufficiency of edits in PATCH (files/symbols/guards/regexes/flags). Penalize unrelated file churn; reward minimal, reversible diffs tied to the stated bug.
  - spec_alignment_rubrics (3-6): Alignment of code to PR_DESCRIPTION. Use textual acceptance criteria (required types/conditions/error handling/API contracts) and ensure the patch reflects them.
  - integrity_rubrics (3-6): Hygiene and "no-cheating" safeguards — avoid test weakening (if tests appear in PATCH), mass renames, or dependency churn; preserve public API/semantics unless PR_DESCRIPTION requires otherwise.
  - runtime_rubrics (3-6): Natural-language criteria describing **intended runtime behavior** (NOT concrete tests), supported by execution-free textual evidence.
    - **Distinguishability:** Ensures the patch introduces or preserves signals that differentiate correct vs. incorrect behavior under plausible inputs (e.g., specific exception class, sentinel return, boundary guard).
    - **Regression safety:** Confirms backward-compatibility constraints (e.g., original API signatures/flags remain valid, deprecations gated via warnings).
    - **Determinism / flake resistance:** Avoids nondeterministic sources at runtime (unseeded randomness, wall-clock sleeps, network I/O) that would make tests flaky.
    - **Resource & timeout bounds:** Prevents pathological loops or heavy calls; respects existing timeouts/limits.
    - **Error-surface clarity:** Produces stable, specific messages/exception types that a test could assert against (not vague strings).
    - **Harness integrity:** Does not bypass or disable the project's runner/verifier hooks (e.g., keeps regression filters, CLI exit codes).

Return exactly this YAML structure:
metadata:
  task_summary: "<one-sentence summary grounded in PR_DESCRIPTION and GOLD_PATCH>"
  underlying_bug: "<precise failure trigger identified via GOLD_PATCH analysis>"
axes:
  file_change_rubrics:
    - id: "FC1"
      description: "Identifies [specific symbol/file from gold patch] and implements [specific change]..."
      weight: 3
  spec_alignment_rubrics:
    - id: "SA1"
      description: "Ensures the behavior matches the [requirement] by [logic seen in gold patch]..."
      weight: 2
  integrity_rubrics:
    - id: "I1"
      description: "Confirms no regression in [related area] as maintained in the gold patch..."
      weight: 2
  runtime_rubrics:
    - id: "R1"
      description: "Maintains [runtime behavior] such that [observable signal from gold patch] is present..."
      weight: 2
"""

GEN_NEXT_STEP_TEMPLATE = """\
OBSERVATION:
{observation}"""

GEN_NEXT_STEP_NO_OUTPUT = """\
Your command ran successfully and did not produce any output."""


# ============== Rubric Evaluation (LLM Judge) Prompts ==============

EVAL_SYSTEM_PROMPT = """\
You are a rubric based evaluator for software-engineering agent's generated patch. Use the provided rubric to evaluate the generated
patch.
Inputs (provided later):
- PR_DESCRIPTION: problem + expected behavior.
- RUBRIC: dictionary of n rubric items for an ideal patch with their ids as keys and descriptions as values.
- PATCH: the model's predicted code patch
Your job:
1. Analyze the rubric and the patch to evaluate the SWE-agent's generated patch.
2. Emit a score for each rubric item. The score should be a binary score of 1 if the patch satisfies the rubric item
and 0 otherwise.
Return the scores in a JSON format.
JSON format:
{
"<rubric_id_1>": <score_1>,
"<rubric_id_2>": <score_2>,
"<rubric_id_3>": <score_3>,
"<rubric_id_4>": <score_4>,
...
"<rubric_id_n>": <score_n>
}"""

EVAL_USER_PROMPT = """\
PR_DESCRIPTION:
{problem_statement}

PATCH:
{patch_text}

RUBRIC:
{rubrics_yaml}

Please evaluate the PATCH using the rubric and return the scores in JSON format.
SCORE
"""
