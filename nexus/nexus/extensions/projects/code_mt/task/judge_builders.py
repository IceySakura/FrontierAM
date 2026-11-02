"""Built-in prompt builders for LLM judges whose input is the agent's runtime
output rather than an on-disk / reward.json field.

Wire a builder into a task via the `[other.llm_judge]` field:
    prompt_builder_import = "nexus.extensions.projects.code_mt.task.judge_builders::rubric_qa"

A builder has the signature
    (template, ctx, task_input, eval_result, judge_config) -> str
and returns the fully-rendered prompt string. It is only invoked when the
dataset sets `prompt_builder_import`, so judges that rely on the default
`{key}` render (e.g. builder anti_hack) are never affected.
"""

from __future__ import annotations

import contextlib

from pathlib import Path
from typing import Any

from nexus.extensions.projects.code_mt.task.scoring import (
    build_format_vars,
    extract_agent_trajectory,
    render_template,
)


# RepoQA / codeqa_rubrics upstream templates carry no reference answer; the
# rubric itself is the ground truth (see the origin parquet's ground_truth).
_NO_REFERENCE = 'No reference answer; refer to the rubrics for the judgement.'


def rubric_qa(
    template: str,
    ctx: Any,
    task_input: Any,
    eval_result: dict[str, Any],
    judge_config: Any,
) -> str:
    """Generic rubric/QA prompt builder.

    Fills the model's answer (and, for whole-trajectory rubrics, the full
    conversation) into an upstream template, supporting BOTH verbatim
    `[BRACKET]` placeholders (repoqa's judge_template.md) and `{brace}`
    placeholders. Any remaining `{key}` is then rendered against the standard
    format vars (reward.json + task.extra scalars) so mixed templates work.

    Placeholders (case as written):
      [QUESTION]/{question}     — the original task prompt (instruction.md)
      [REFERENCE]/{reference}   — reference answer, else the no-reference note
      [ANSWER]/{answer}         — the agent's final assistant message
      [TRAJECTORY]/{trajectory} — the whole conversation transcript
    """
    answer, transcript, _ = extract_agent_trajectory(ctx)

    extra = getattr(task_input, 'extra', {}) or {}
    # Question source, in order: explicit extra field → task message → the
    # instruction.md the agent was actually prompted with (repoqa keeps the
    # question there, not in task.toml, so message is usually empty).
    question = extra.get('question') or getattr(task_input, 'message', '') or ''
    if not question.strip():
        task_dir = getattr(task_input, 'task_dir', '') or ''
        if task_dir:
            with contextlib.suppress(OSError):
                question = (Path(task_dir) / 'instruction.md').read_text(encoding='utf-8')
    reference = extra.get('reference') or _NO_REFERENCE

    subs = {
        'QUESTION': question,
        'REFERENCE': reference,
        'ANSWER': answer,
        'TRAJECTORY': transcript,
    }
    for key, val in subs.items():
        template = template.replace(f'[{key}]', val)

    # Render any remaining {key} (including {question}/{reference}/{answer}/
    # {trajectory}) against the standard vars + these four.
    format_vars = build_format_vars(
        task_input,
        eval_result,
        extra_overrides={
            'question': question,
            'reference': reference,
            'answer': answer,
            'trajectory': transcript,
            'agent_answer': answer,
            'agent_trajectory': transcript,
        },
    )
    return render_template(template, format_vars)
