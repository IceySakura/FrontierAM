# Copyright (c) 2025 Bytedance Ltd. and/or its affiliates
# SPDX-License-Identifier: MIT

"""DeepSearch QA 评估模块。

使用官方 Set Answer prompt 格式，通过一次 LLM 调用获取所有匹配信息，
然后基于 Correctness Details 和 Excessive Answers 计算 F1 分数。

F1 计算逻辑：
- Recall = Correctness Details 中 true 的数量 / expected 总数
- Precision = 正确数量 / (正确数量 + excessive 数量)
- F1 = 2 * P * R / (P + R)
"""

import json
import logging
import re

from dataclasses import dataclass

from nexus.core import require_context
from nexus.extensions.agents.base.evaluation import EvaluationDetails
from nexus.extensions.tasks.search.deepsearch_qa_evaluation.prompt import (
    DEEPSEARCH_QA_PROMPT_TEMPLATE,
    DEEPSEARCH_QA_SYSTEM_PROMPT,
)


logger = logging.getLogger(__name__)


@dataclass
class DeepsearchQaDetails(EvaluationDetails):
    """Detailed evaluation metrics for DEEPSEARCH_QA mode."""

    precision: float = 0.0
    recall: float = 0.0
    f1: float = 0.0
    predicted_count: int = 0
    expected_count: int = 0
    matched_predicted: int = 0
    matched_expected: int = 0
    correctness_details: dict[str, bool] | None = None
    excessive_answers: list[str] | None = None


def _parse_judge_f1_json(content: str) -> dict:
    """从 LLM 返回的文本中解析 Answer Correctness JSON。

    支持以下格式：
    - 纯 JSON 文本
    - 包裹在 ```json ... ``` 代码块中的 JSON
    """
    # 尝试从 markdown 代码块中提取 JSON
    code_block_match = re.search(r'```(?:json)?\s*\n?(.*?)\n?```', content, re.DOTALL)
    json_str = code_block_match.group(1).strip() if code_block_match else content.strip()

    # 尝试提取最外层的 JSON 对象
    brace_match = re.search(r'\{.*\}', json_str, re.DOTALL)
    if brace_match:
        json_str = brace_match.group(0)

    return json.loads(json_str)


async def _call_judge_f1(
    question: str,
    predicted: str,
    expected: str,
    *,
    judge_prompt_template: str | None = None,
    judge_system_prompt: str | None = None,
    max_input_length: int = 65536,
) -> dict:
    """调用 LLM 一次性评估所有 predicted 与 expected 的匹配关系。

    直接将原始答案字符串传给 LLM，由 LLM 自行理解答案结构并拆分，
    避免在代码层面因分隔符歧义导致错误拆分。

    使用官方 Set Answer prompt 格式，LLM 返回 JSON 包含：
    - Correctness Details: 每个 expected 答案是否在 response 中被找到
    - Excessive Answers: response 中多余的答案

    Returns:
        解析后的 Answer Correctness 字典。
    """
    response_text = predicted
    answer_text = expected

    # 截断
    response_text = response_text[-max_input_length:] if len(response_text) > max_input_length else response_text
    answer_text = answer_text[-max_input_length:] if len(answer_text) > max_input_length else answer_text

    # 构建 system prompt
    system_prompt = judge_system_prompt if judge_system_prompt is not None else DEEPSEARCH_QA_SYSTEM_PROMPT

    # 构建 user prompt
    if judge_prompt_template is not None:
        user_content = judge_prompt_template.format(
            prompt=question,
            question=question,
            response=response_text,
            predicted=response_text,
            model_response=response_text,
            answer=answer_text,
            expected=answer_text,
            response_reference=answer_text,
            ref_answer=answer_text,
            prompt_type='Set Answer',
        )
    else:
        user_content = DEEPSEARCH_QA_PROMPT_TEMPLATE.format(
            prompt=question,
            response=response_text,
            answer=answer_text,
            prompt_type='Set Answer',
        )

    messages = [
        {'role': 'system', 'content': system_prompt},
        {'role': 'user', 'content': user_content},
    ]

    ctx = require_context()
    client = ctx.get_openai_client('judge')

    response = await client.create_completion(messages=messages)
    content = response.choices[0].message.content or ''

    parsed = _parse_judge_f1_json(content)
    return parsed.get('Answer Correctness', {})


async def deepsearch_qa_eval(
    question: str,
    predicted: str,
    expected: str,
    *,
    judge_prompt_template: str | None = None,
    judge_system_prompt: str | None = None,
    max_input_length: int = 65536,
) -> tuple[float, DeepsearchQaDetails]:
    """LLM-based semantic evaluation using F1 score.

    直接将原始答案字符串传给 LLM，由 LLM 自行理解答案结构并拆分，
    避免在代码层面因分隔符与答案内容冲突导致错误拆分。
    F1 完全基于 LLM 返回的 Correctness Details 和 Excessive Answers 计算。

    F1 calculation:
    - Recall = Correctness Details 中 true 的数量 / Correctness Details 总数
    - Precision = 正确数量 / (正确数量 + excessive 数量)
    - F1 = 2 * P * R / (P + R)

    Args:
        question: The question being evaluated.
        predicted: The predicted answer string (passed directly to LLM without splitting).
        expected: The expected answer string (passed directly to LLM without splitting).
        judge_prompt_template: Optional custom prompt template.
        judge_system_prompt: Optional custom system prompt.
        max_input_length: Maximum length for input strings.

    Returns:
        Tuple of (f1_score, DeepsearchQaDetails).
    """
    details = DeepsearchQaDetails()

    if not predicted.strip() or not expected.strip():
        details.msg = 'Empty predicted or expected string'
        return 0.0, details

    try:
        answer_correctness = await _call_judge_f1(
            question,
            predicted,
            expected,
            judge_prompt_template=judge_prompt_template,
            judge_system_prompt=judge_system_prompt,
            max_input_length=max_input_length,
        )
    except Exception as e:
        logger.warning(f'DeepSearch QA eval call failed ({e}), returning 0.0')
        details.msg = f'DeepSearch QA eval call failed: {e}'
        return 0.0, details

    # 解析 Correctness Details 和 Excessive Answers
    # LLM 自行理解答案结构，返回每个 expected 答案项的匹配结果
    correctness_details = answer_correctness.get('Correctness Details', {})
    excessive_answers = answer_correctness.get('Excessive Answers', [])

    details.correctness_details = correctness_details
    details.excessive_answers = excessive_answers

    # Recall = expected 中被覆盖到的比例（基于 LLM 返回的 Correctness Details key 数量）
    matched_expected = sum(1 for v in correctness_details.values() if v)
    total_expected = len(correctness_details)
    recall = matched_expected / total_expected if total_expected > 0 else 0.0

    # Precision = 正确数量 / (正确数量 + excessive 数量)
    matched_predicted = matched_expected  # 正确匹配的数量
    total_predicted = matched_predicted + len(excessive_answers)
    precision = matched_predicted / total_predicted if total_predicted > 0 else 0.0

    # F1 = 2 * P * R / (P + R)
    f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) > 0 else 0.0

    details.precision = precision
    details.recall = recall
    details.f1 = f1
    details.predicted_count = matched_predicted + len(excessive_answers)
    details.expected_count = total_expected
    details.matched_predicted = matched_predicted
    details.matched_expected = matched_expected

    logger.info(
        f'DeepSearch QA: precision={precision:.3f} ({matched_predicted}/{total_predicted}), '
        f'recall={recall:.3f} ({matched_expected}/{total_expected}), f1={f1:.3f}'
    )

    return f1, details
