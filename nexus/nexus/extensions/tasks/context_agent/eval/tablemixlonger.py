"""TableMixLonger evaluator — checklist-rubric LLM judge.

Unlike the generic answer / rubric judges, TableMixLonger grades with the
dataset's own judge prompt (the verl reward template), kept here as
:data:`DEFAULT_TEMPLATE`. The prompt has three placeholders:
``[prompt_placeholder]`` (the question), ``[checklist_placeholder]`` (the
rubrics) and ``[answer_placeholder]`` (the model output).

Each task only needs to carry:

* ``rubrics`` — the per-task grading criteria (``list[str]``; this is the
  dataset's ``checklist``), rendered into ``[checklist_placeholder]``.

The template asks the judge to reason and then emit a final line
``【评分】：X分`` where ``X`` is ``0`` or ``1``; we parse that score back out.
If rubrics are absent (e.g. hand-built data) we fall back to an answer-derived
criterion. A task may still override the prompt via ``metadata.judge_template``,
but that is no longer part of the converted data.
"""

from __future__ import annotations

import logging
import re


logger = logging.getLogger(__name__)

MAX_RETRIES = 10

# Canonical judge prompt shipped with the dataset (extra_info.extra.template).
# This is the primary template; records no longer carry their own copy.
DEFAULT_TEMPLATE = (
    '你是一个严格的判分专家。请根据标准答案和判分标准，判断模型的回答是否正确。\n\n'
    '【问题】\n[prompt_placeholder]\n\n'
    '【评判标准】\n[checklist_placeholder]\n\n'
    '【模型回答】\n[answer_placeholder]\n\n'
    '请逐条核对判分标准，判断模型回答是否与标准答案等价。\n'
    '评分只有两档：1分表示完全正确，0分表示有任何错误或不符合判分标准。\n'
    '在回答的最后一行，必须单独输出分数，格式为：【评分】：X分'
)

# Try strict ("【评分】：1分") first, then looser variants, taking the LAST match
# so a trailing score line wins over any score mentioned mid-reasoning.
_SCORE_PATTERNS = (
    re.compile(r'【评分】\s*[：:]\s*([01])\s*分'),
    re.compile(r'【评分】\s*[：:]?\s*([01])'),
    re.compile(r'([01])\s*分\s*$'),
)


def _parse_score(text: str) -> int | None:
    """Extract the binary score from a judge reply, or ``None`` if absent."""
    if not text:
        return None
    for pattern in _SCORE_PATTERNS:
        matches = pattern.findall(text)
        if matches:
            return int(matches[-1])
    return None


def _checklist_text(task: dict) -> str:
    """Render the task's ``rubrics`` (== checklist) into numbered judge text."""
    rubrics = task.get('rubrics')
    if isinstance(rubrics, (list, tuple)) and rubrics:
        return '\n'.join(f'{i}. {str(r).strip()}' for i, r in enumerate(rubrics, 1))
    if isinstance(rubrics, str) and rubrics.strip():
        return rubrics.strip()
    # Fallbacks for hand-built / legacy records without a rubrics list.
    meta = task.get('metadata') or {}
    legacy = meta.get('checklist') or task.get('checklist')
    if legacy and str(legacy).strip():
        return str(legacy).strip()
    answer = task.get('answer', '')
    return f'1. 正确答案为 {answer}，核心信息必须与之一致' if answer else '（无明确判分标准，请判断回答是否合理且正确）'


def _build_prompt(model_output: str, task: dict) -> str:
    """Fill the dataset judge template with question / rubrics / answer."""
    meta = task.get('metadata') or {}
    template = meta.get('judge_template') or task.get('judge_template') or DEFAULT_TEMPLATE
    question = task.get('context_question') or meta.get('final_question') or ''
    checklist = _checklist_text(task)

    return (
        template
        .replace('[prompt_placeholder]', str(question))
        .replace('[checklist_placeholder]', checklist)
        .replace('[answer_placeholder]', model_output)
    )


async def evaluate(model_output: str, task: dict, llm) -> dict:
    """Evaluate *model_output* for a TableMixLonger task via the dataset judge."""
    if not model_output or not model_output.strip():
        return {
            'score': 0,
            'rationale': 'No model output',
            'requirement_status': [],
            'judge_prompt': [],
        }

    prompt = _build_prompt(model_output, task)
    messages = [{'role': 'user', 'content': prompt}]

    for attempt in range(MAX_RETRIES):
        try:
            raw = await llm.chat(messages)
            if isinstance(raw, dict):
                raw = raw.get('response', '')
            score = _parse_score(raw or '')
            if score is None:
                raise ValueError('no 【评分】 score line found in judge output')
            return {
                'score': score,
                'rationale': (raw or '').strip(),
                'requirement_status': [],
                'judge_prompt': messages,
            }
        except ValueError as exc:
            logger.warning(
                'TableMixLonger judge parse failed (attempt %s/%s): %s',
                attempt + 1,
                MAX_RETRIES,
                exc,
            )
        except Exception as exc:  # noqa: BLE001
            logger.error(
                'TableMixLonger judge call failed (attempt %s/%s): %s',
                attempt + 1,
                MAX_RETRIES,
                exc,
            )

    return {
        'score': 0,
        'rationale': f'Judge failed after {MAX_RETRIES} attempts',
        'requirement_status': [],
        'judge_prompt': messages,
    }
