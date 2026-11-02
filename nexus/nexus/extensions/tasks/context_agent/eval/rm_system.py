"""RewardService-backed rubric grading for ContextAgent RL rows.

The unified ContextAgent branch uses model-based benchmark evaluators by
default. RL keeps the established RewardService routes for outside_L3 and
CtxAgent1000 so their reward semantics remain comparable with prior runs.
Each rubric receives a 0/1 verdict and the service returns their mean.
"""

from __future__ import annotations

from typing import Any

from ..scoring import rm_scorer


DEFAULT_ROUTER_ID = 'T1-Omni-0'
JUDGE_FAILED_PREFIX = 'Judge failed'

RUBRIC_MEAN_TEMPLATE = """你是一名严谨的判卷老师。请根据题目和学生答案，对每个评分项分别给出0或1分。

规则：
1. 每个<RUBRIC_N>...</RUBRIC_N>是一个独立评分项，必须逐项判断。
2. 满足该项记1分，不满足记0分；不同评分项互不影响。
3. 输出分数数量必须与评分项数量完全一致，并保持相同顺序。
4. 空答案或无意义答案的所有评分项均为0分。
5. 必须严格使用以下两行作为输出结尾，不要使用其他字段名：
[grading rationale]: 简要说明逐项评分依据
[score]: [1, 0, 1]

<题目>：
[prompt_placeholder]

<评分项>：
[checklist_placeholder]

<学生答案>：
[answer_placeholder]
"""

T1_EXTRA_DEFAULTS: dict[str, Any] = {
    'answer_type': 'T1',
    'short_answer_pattern': '(.*)',
    'process_format_error': '1',
    'process_truncate': 'True',
    'rubric_aggregation': 'mean',
}


def build_checklist(rubrics: list[str]) -> str:
    return '\n'.join(
        f'<RUBRIC_{index}>\n{str(rubric).strip()}\n</RUBRIC_{index}>' for index, rubric in enumerate(rubrics, start=1)
    )


def build_extra(rubrics: list[str], overrides: dict | None = None) -> dict[str, Any]:
    rubric_items = [str(rubric).strip() for rubric in rubrics]
    extra: dict[str, Any] = {
        **T1_EXTRA_DEFAULTS,
        'template': RUBRIC_MEAN_TEMPLATE,
        'checklist': build_checklist(rubric_items),
        # ComplexInstruction-L3Reward validates these legacy fields before
        # invoking its judge. The item catalog carries rubrics but no original
        # RM ``extra`` payload, so reconstruct equal weights here.
        'checklist_items': rubric_items,
        'rubric_items': rubric_items,
        'weight': [1.0] * len(rubric_items),
        'ref_answer': 'None',
    }
    extra.update(overrides or {})
    return extra


def _prompt(task: dict[str, Any]) -> list[dict[str, str]]:
    reward_prompt = task.get('reward_prompt')
    if isinstance(reward_prompt, list) and reward_prompt:
        return [
            {'role': str(message.get('role') or 'user'), 'content': str(message.get('content') or '')}
            for message in reward_prompt
            if isinstance(message, dict)
        ]
    question = str(task.get('context_question') or '').strip()
    return [{'role': 'user', 'content': question}] if question else []


def _failed(rationale: str, *, reward_error: bool) -> dict[str, Any]:
    return {
        'score': 0.0,
        'rationale': f'{JUDGE_FAILED_PREFIX} (rm_system): {rationale}',
        'requirement_status': [],
        'reward_error': reward_error,
        'graded_on': 'reward_system',
    }


async def evaluate(model_output: str, task: dict[str, Any], llm=None) -> dict[str, Any]:
    del llm
    if not model_output or not str(model_output).strip():
        return {
            'score': 0.0,
            'rationale': 'No model output',
            'requirement_status': [],
            'reward_error': False,
            'graded_on': 'reward_system',
        }

    rubrics = [str(item).strip() for item in (task.get('rubrics') or []) if str(item).strip()]
    if not rubrics:
        return _failed('the row carries no rubrics', reward_error=True)
    prompt = _prompt(task)
    if not prompt:
        return _failed('the row carries no reward prompt', reward_error=True)

    router_id = str(task.get('reward_system_router_id') or DEFAULT_ROUTER_ID)
    result = await rm_scorer.score_answer(
        prompt,
        str(model_output),
        router_id=router_id,
        ref_answer=str(task.get('ref_answer') or 'None'),
        extra=build_extra(rubrics, task.get('rm_extra')),
    )
    final_score = result.get('final_score')
    if not result.get('ok'):
        return _failed(
            f'{result.get("error_msg") or "error"} (final_score={final_score}, router={router_id})',
            reward_error=bool(result.get('reward_error')),
        )
    return {
        'score': final_score,
        'rationale': f'reward_system final_score={final_score} over {len(rubrics)} rubric(s) via {router_id}',
        'requirement_status': [],
        'reward_error': False,
        'graded_on': 'reward_system',
        'rm_final_score': final_score,
        'rm_final_score_ori': result.get('final_score_ori'),
        'rm_request_ids': list(result.get('request_ids') or []),
    }


__all__ = [
    'DEFAULT_ROUTER_ID',
    'RUBRIC_MEAN_TEMPLATE',
    'build_checklist',
    'build_extra',
    'evaluate',
]
