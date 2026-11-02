"""ACEBench task — 使用官方 checker 评估（手动 agent 调用）"""

import importlib
import json
import logging
import os
import sys

from pathlib import Path
from typing import Any

from nexus.core import (
    TaskRunInput,
    TaskRunOutput,
    WithScore,
    agent_registry,
    task_registry,
)


logger = logging.getLogger(__name__)


def _setup_acebench_path() -> Path:
    candidates = []

    env_path = os.getenv('ACE_BENCH_PATH')
    if env_path:
        candidates.append(Path(env_path).expanduser())

    candidates.extend([
        Path(__file__).parent / 'ACEBench',
        Path(__file__).resolve().parents[4] / 'ACEBench-upstream',
    ])

    for ace_path in candidates:
        if (ace_path / 'eval_main.py').exists() and (ace_path / 'generate.py').exists():
            if str(ace_path) not in sys.path:
                sys.path.insert(0, str(ace_path))
            return ace_path

    searched = ', '.join(str(p) for p in candidates)
    raise RuntimeError(
        f'ACEBench source tree not found. Set ACE_BENCH_PATH or place the official repo at one of: {searched}'
    )


ACE_BENCH_PATH = _setup_acebench_path()

_checker = importlib.import_module('model_eval.checker')
agent_checker = _checker.agent_checker
normal_checker = _checker.normal_checker


# ============ Task Input/Output ============


class AceTaskInput(TaskRunInput):
    # class AceTaskInput(TaskRunInput, WithAgentRunner):
    """ACE task input"""

    category: str = 'normal'
    language: str = 'en'
    task_data: dict[str, Any] | None = None
    api_pool: list[dict[str, Any]] = []
    max_tokens: int = 4096
    temperature: float = 0.0


class AceTaskOutput(TaskRunOutput, WithScore):
    """ACE task output"""

    correct: bool = False
    model_response: str = ''
    tool_calls: list[dict[str, Any]] = []
    ground_truth: str = ''
    # ground_truth: dict[str, Any] | str | list | None = None  # ✅ 支持 list/dict/str
    category: str = ''
    task_id: str = ''
    termination_reason: str | None = None
    error: str | None = None
    checker_error: list | None = None
    checker_error_type: str | None = None


# ============ 评估函数 ============


def _evaluate_with_official_checker(
    model_tool_calls: list[dict[str, Any]],
    ground_truth: dict[str, Any] | str | list[dict[str, Any]],
    api_pool: list[dict[str, Any]],
    question: str,
    category: str,
    model_response: str = '',  # ✅ 添加 model_response 参数
) -> dict[str, Any]:
    """使用官方 Checker 评估"""

    # ✅ 空值检查
    if not category:
        logger.error('category is None or empty')
        return {'valid': False, 'error': ['category is None or empty'], 'error_type': 'invalid_input'}

    # ✅ Special 类别：检查文本响应，不检查函数调用
    if category.startswith('special_'):
        return _evaluate_special_category(category, model_response, ground_truth)

    # ✅ 处理 ground_truth 为字符串的情况（罕见）
    if isinstance(ground_truth, str):
        logger.warning(f'ground_truth is string (rare case): {ground_truth[:50]}...')
        if not model_tool_calls:
            return {'valid': True, 'error': [], 'error_type': None}

        return {
            'valid': False,
            'error': ['Model should not call API for this case'],
            'error_type': 'unexpected_tool_call',
        }

    if not model_tool_calls and not category.startswith('agent'):
        return {'valid': False, 'error': ['No tool calls in model response'], 'error_type': 'no_tool_calls'}

    try:
        if category.startswith('agent'):
            possible_answer_items = (
                ground_truth if isinstance(ground_truth, list) else [ground_truth] if ground_truth else []
            )
            raw_model_result = json.loads(model_response)
            model_result_items = raw_model_result if isinstance(raw_model_result, list) else [raw_model_result]

            valid = True
            errors: list[Any] = []
            error_type = ''

            if len(possible_answer_items) != len(model_result_items):
                valid = False
                error_type = 'wrong number of class'
            else:
                for index in range(len(possible_answer_items)):
                    possible_answer = possible_answer_items[index]
                    if not isinstance(possible_answer, dict):
                        continue

                    possible_keys = set(possible_answer.keys())
                    matched_dict: dict[str, Any] | None = None
                    for model_dict in model_result_items:
                        if not isinstance(model_dict, dict):
                            continue

                        model_keys = set(model_dict.keys())
                        if possible_keys == model_keys:
                            matched_dict = model_dict
                            break

                    if matched_dict:
                        checker_result = agent_checker(matched_dict, possible_answer)
                    else:
                        checker_result = {'valid': False, 'error': 'missing matched dict', 'error_type': 'missing_dict'}

                    if not checker_result['valid']:
                        valid = False
                        errors.append(checker_result['error'])
                        error_type = checker_result['error_type']

            return {'valid': valid, 'error': errors, 'error_type': error_type}

        # normal_checker(func_descriptions, model_output, possible_answers, question, test_category)
        return normal_checker(
            func_descriptions=api_pool,
            model_output=model_tool_calls,
            possible_answers=ground_truth,
            question=question,
            test_category=category,
        )

    except Exception as e:
        logger.error(f'Checker error: {e}')
        return {'valid': False, 'error': [str(e)], 'error_type': 'checker_exception'}


def _evaluate_special_category(
    category: str,
    model_response: str,
    ground_truth: dict[str, Any] | str | list[dict[str, Any]],
) -> dict[str, Any]:
    """尽量对齐官方 eval_main.py special_eval 的单样本逻辑。"""

    result = {'valid': True, 'error': [], 'error_type': ''}

    # 官方 special_eval 预期 ground_truth 是 dict；这里兼容 str/list 但仍保持官方关键词判定
    if isinstance(ground_truth, str):
        ground_truth = {'unknown': [ground_truth]}
    elif isinstance(ground_truth, list):
        ground_truth = {str(i): item for i, item in enumerate(ground_truth)}

    if 'incomplete' in category:
        for name, values in ground_truth.items():
            if 'Missing necessary parameters' not in model_response:
                result['valid'] = False
                result['error'] = [
                    f"The user's instruction is missing necessary parameters ({values}) for the ({name}), but the model failed to correctly point it out"
                ]
                result['error_type'] = 'error_detection'
                break
            if name not in model_response:
                result['valid'] = False
                result['error'] = [
                    f"The user's instruction is missing necessary parameters ({values}) for the ({name}), but the model failed to correctly point it out"
                ]
                result['error_type'] = 'error_correction'
                break

            for value in values:
                if value not in model_response:
                    result['valid'] = False
                    result['error'] = [
                        f"The user's instruction is missing necessary parameters ({value}) for the ({name}), but the model failed to correctly point it out"
                    ]
                    result['error_type'] = 'error_correction'
                    break
            if not result['valid']:
                break

    elif 'error' in category:
        for name, values in ground_truth.items():
            if 'There is incorrect value' not in model_response:
                result['valid'] = False
                result['error'] = [
                    f"The user's instruction contains incorrect values ({values}) of the parameters ({name}), but the model failed to correctly point it out"
                ]
                result['error_type'] = 'error_detection'
                break

            for value in values:
                if value not in model_response:
                    result['valid'] = False
                    result['error'] = [
                        f"The user's instruction contains incorrect values ({values}) of the parameters ({name}), but the model failed to correctly point it out"
                    ]
                    result['error_type'] = 'error_correction'
                    break
            if not result['valid']:
                break

    elif 'irrelevant' in category and 'the limitations of the function' not in model_response:
        result['valid'] = False
        result['error'] = ['The model cannot solve this problem, due to the limitations of the function']
        result['error_type'] = 'error_detection'

    return result


# ============ Task Function ============


@task_registry.register('ace')
async def run_ace_task(task_input: AceTaskInput) -> AceTaskOutput:
    """ACE task - 手动调用 agent"""

    logger.debug('ACE task started')

    if task_input.task_data is None:
        raise ValueError('task_data is required')

    task_id = task_input.task_data.get('id', 'unknown')
    category = task_input.category
    language = task_input.language
    ground_truth = task_input.task_data.get('ground_truth', {})
    question = task_input.task_data.get('question', '')

    logger.debug(f'Task: task_id={task_id}, category={category}')

    # 🔧 手动调用 agent（不用 WithAgentRunner）
    try:
        # 获取 agent 函数
        agent_func = agent_registry.get('ace_agent')
        if agent_func is None:
            raise RuntimeError("Agent 'ace_agent' not registered")

        # 构建 agent input
        from nexus.extensions.tasks.ace.agent import AceAgentInput

        agent_input = AceAgentInput(
            category=category,
            language=language,
            task_data=task_input.task_data,
            api_pool=task_input.api_pool,
            model_name=os.getenv('ACE_AGENT_MODEL', 'gpt-4o'),
            max_tokens=task_input.max_tokens,
            temperature=task_input.temperature,
        )

        # 调用 agent
        logger.debug('Calling agent...')
        agent_output = await agent_func(agent_input)

        # 提取输出
        model_response = getattr(agent_output, 'model_response', '')
        tool_calls = getattr(agent_output, 'tool_calls', [])
        termination_reason = getattr(agent_output, 'termination_reason', None)
        agent_error = ''

    except Exception as e:
        logger.error(f'Agent call failed: {e}')
        model_response = ''
        tool_calls = []
        termination_reason = 'ERROR'
        agent_error = str(e)

    if 'normal' in category:
        model_tool_calls_format = tool_calls
    else:
        model_tool_calls_format = []
        for tool_call in tool_calls:
            model_tool_calls_format.append(list(tool_call.values())[0])

    # 评估
    checker_result = _evaluate_with_official_checker(
        model_tool_calls=model_tool_calls_format,
        ground_truth=ground_truth,
        api_pool=task_input.api_pool,
        question=question,
        category=category,
        model_response=model_response,  # ✅ 传递 model_response
    )

    correct = checker_result.get('valid', False)
    score = 1.0 if correct else 0.0

    # 确保 checker_error 是列表
    checker_error = checker_result.get('error', [])
    if isinstance(checker_error, str):
        checker_error = [checker_error] if checker_error else []

    logger.info(f'Task result: score={score}, correct={correct}')

    return AceTaskOutput(
        score=score,
        correct=correct,
        model_response=model_response,
        tool_calls=tool_calls,
        ground_truth=json.dumps({'ground_truth': ground_truth, 'mile_stone': task_input.task_data['mile_stone']}),
        category=category,
        task_id=str(task_id),
        termination_reason=termination_reason,
        error=agent_error,
        checker_error=checker_error,
        checker_error_type=checker_result.get('error_type', None),
    )
