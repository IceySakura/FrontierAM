# Copyright (c) 2025 Bytedance Ltd. and/or its affiliates
# SPDX-License-Identifier: MIT

"""DRACO Rubric 评估模块。

基于 DRACO 论文的评估方法论，使用 LLM-as-a-judge 协议对每个 criterion 进行
二元判定（MET/UNMET），然后通过加权求和计算归一化分数。

评分公式（来自 DRACO 论文）：
    raw_score = sum(v_i * w_i for all i)
    normalized_score = clamp(raw_score / sum(w_i for all i where w_i > 0), 0, 1) * 100%

其中 v_i 为二元判定（1=MET, 0=UNMET），w_i 为 criterion 权重。
正权重 criterion MET 时加分，负权重 criterion MET 时扣分（表示存在错误）。
"""

import asyncio
import json
import logging
import re

from dataclasses import dataclass, field
from typing import Literal

from nexus.core import require_context
from nexus.extensions.agents.base.evaluation import EvaluationDetails
from nexus.extensions.tasks.search.draco_rubric_evaluation.prompt import (
    DRACO_RUBRIC_ONESHOT_SYSTEM_PROMPT,
    DRACO_RUBRIC_ONESHOT_USER_PROMPT_TEMPLATE,
    DRACO_RUBRIC_SYSTEM_PROMPT,
    DRACO_RUBRIC_USER_PROMPT_TEMPLATE,
)


logger = logging.getLogger(__name__)


@dataclass
class CriterionResult:
    """单个 criterion 的评估结果。"""

    id: str
    section_id: str
    requirement: str
    weight: int
    verdict: Literal['MET', 'UNMET']
    explanation: str


@dataclass
class SectionScore:
    """单个 section（评估维度）的分数。"""

    id: str
    title: str
    score: float  # 归一化分数 0-1
    raw_score: float  # 原始加权分数
    total_positive_weight: float
    criteria_met: int
    criteria_total: int


@dataclass
class DracoRubricDetails(EvaluationDetails):
    """DRACO Rubric 评估的详细结果。"""

    # 总体分数
    normalized_score: float = 0.0
    raw_score: float = 0.0
    total_positive_weight: float = 0.0

    # 各维度分数
    section_scores: list[SectionScore] = field(default_factory=list)

    # 逐条 criterion 结果
    criterion_results: list[CriterionResult] = field(default_factory=list)

    # 统计信息
    criteria_met: int = 0
    criteria_total: int = 0
    positive_criteria_met: int = 0
    positive_criteria_total: int = 0
    negative_criteria_met: int = 0  # 负权重 criterion MET 表示存在错误
    negative_criteria_total: int = 0


def _parse_rubric_json(answer: str) -> dict:
    """解析 DRACO 数据集中 answer 字段的 rubric JSON。

    answer 字段是一个 JSON 编码的 rubric，包含 sections 和 criteria。
    """
    if isinstance(answer, dict):
        return answer
    return json.loads(answer)


def _flatten_criteria(rubric: dict) -> list[dict]:
    """将 rubric 中所有 sections 的 criteria 展平为一个列表，保留 section 信息。"""
    criteria = []
    for section in rubric.get('sections', []):
        section_id = section.get('id', '')
        section_title = section.get('title', '')
        for criterion in section.get('criteria', []):
            criteria.append({
                'id': criterion.get('id', ''),
                'section_id': section_id,
                'section_title': section_title,
                'weight': criterion.get('weight', 0),
                'requirement': criterion.get('requirement', ''),
            })
    return criteria


def _parse_criterion_response(content: str) -> dict:
    """解析 LLM 返回的单条 criterion 评估结果 JSON。"""
    # 尝试从 markdown 代码块中提取
    code_block_match = re.search(r'```(?:json)?\s*\n?(.*?)\n?```', content, re.DOTALL)
    json_str = code_block_match.group(1).strip() if code_block_match else content.strip()

    # 提取最外层 JSON 对象
    brace_match = re.search(r'\{.*\}', json_str, re.DOTALL)
    if brace_match:
        json_str = brace_match.group(0)

    return json.loads(json_str)


def _parse_oneshot_response(content: str) -> list[dict]:
    """解析 LLM 返回的批量 criterion 评估结果 JSON。"""
    code_block_match = re.search(r'```(?:json)?\s*\n?(.*?)\n?```', content, re.DOTALL)
    json_str = code_block_match.group(1).strip() if code_block_match else content.strip()

    brace_match = re.search(r'\{.*\}', json_str, re.DOTALL)
    if brace_match:
        json_str = brace_match.group(0)

    parsed = json.loads(json_str)
    return parsed.get('criteria_evaluations', [])


async def _judge_single_criterion(
    query: str,
    response: str,
    criterion: dict,
    *,
    system_prompt: str | None = None,
    max_response_length: int = 16384,
) -> CriterionResult:
    """对单个 criterion 进行 LLM 评估。

    Args:
        query: 原始研究问题。
        response: 被评估的回答文本。
        criterion: criterion 字典，包含 id, section_id, weight, requirement。
        system_prompt: 可选自定义 system prompt。
        max_response_length: response 最大截断长度。

    Returns:
        CriterionResult 评估结果。
    """
    # 截断 response
    response_text = response[-max_response_length:] if len(response) > max_response_length else response

    criterion_type = 'negative' if criterion['weight'] < 0 else 'positive'

    sys_prompt = system_prompt or DRACO_RUBRIC_SYSTEM_PROMPT
    user_content = DRACO_RUBRIC_USER_PROMPT_TEMPLATE.format(
        criterion_type=criterion_type,
        requirement=criterion['requirement'],
        query=query,
        response=response_text,
    )

    messages = [
        {'role': 'system', 'content': sys_prompt},
        {'role': 'user', 'content': user_content},
    ]

    ctx = require_context()
    client = ctx.get_openai_client('judge')

    try:
        llm_response = await client.create_completion(messages=messages)
        content = llm_response.choices[0].message.content or ''
        parsed = _parse_criterion_response(content)

        verdict = parsed.get('criterion_status', 'UNMET').upper()
        if verdict not in ('MET', 'UNMET'):
            verdict = 'UNMET'

        explanation = parsed.get('explanation', '')
    except Exception as e:
        logger.warning(f'Failed to judge criterion {criterion["id"]}: {e}')
        verdict = 'UNMET'
        explanation = f'Judge failed: {e}'

    return CriterionResult(
        id=criterion['id'],
        section_id=criterion['section_id'],
        requirement=criterion['requirement'],
        weight=criterion['weight'],
        verdict=verdict,
        explanation=explanation,
    )


async def _judge_criteria_batch(
    query: str,
    response: str,
    criteria: list[dict],
    *,
    system_prompt: str | None = None,
    max_response_length: int = 16384,
) -> list[CriterionResult]:
    """批量评估一组 criteria（one-shot 模式）。

    将一组 criteria 打包在一次 LLM 调用中评估，适用于 criteria 数量较少的 section。

    Args:
        query: 原始研究问题。
        response: 被评估的回答文本。
        criteria: criteria 列表。
        system_prompt: 可选自定义 system prompt。
        max_response_length: response 最大截断长度。

    Returns:
        CriterionResult 列表。
    """
    response_text = response[-max_response_length:] if len(response) > max_response_length else response

    # 构建 criteria 文本
    criteria_lines = []
    for idx, criterion in enumerate(criteria):
        criterion_type = (
            'NEGATIVE (status MET if error IS present, UNMET if error is NOT present)'
            if criterion['weight'] < 0
            else 'POSITIVE (status MET if requirement IS present, UNMET if requirement is NOT present)'
        )
        criteria_lines.append(f'{idx}. [{criterion_type}] (weight: {criterion["weight"]}) {criterion["requirement"]}')

    criteria_text = '\n'.join(criteria_lines)

    sys_prompt = system_prompt or DRACO_RUBRIC_ONESHOT_SYSTEM_PROMPT
    user_content = DRACO_RUBRIC_ONESHOT_USER_PROMPT_TEMPLATE.format(
        criteria_text=criteria_text,
        query=query,
        response=response_text,
    )

    messages = [
        {'role': 'system', 'content': sys_prompt},
        {'role': 'user', 'content': user_content},
    ]

    ctx = require_context()
    client = ctx.get_openai_client('judge')

    try:
        llm_response = await client.create_completion(messages=messages)
        content = llm_response.choices[0].message.content or ''
        evaluations = _parse_oneshot_response(content)

        # 构建 idx -> evaluation 映射
        eval_map = {item.get('criterion_idx', -1): item for item in evaluations}

        results = []
        for idx, criterion in enumerate(criteria):
            eval_item = eval_map.get(idx)
            if eval_item:
                verdict = eval_item.get('criterion_status', 'UNMET').upper()
                if verdict not in ('MET', 'UNMET'):
                    verdict = 'UNMET'
                explanation = eval_item.get('explanation', '')
            else:
                verdict = 'UNMET'
                explanation = 'Evaluation not found in LLM response'

            results.append(
                CriterionResult(
                    id=criterion['id'],
                    section_id=criterion['section_id'],
                    requirement=criterion['requirement'],
                    weight=criterion['weight'],
                    verdict=verdict,
                    explanation=explanation,
                )
            )
        return results

    except Exception as e:
        logger.warning(f'Batch judge failed: {e}, falling back to all UNMET')
        return [
            CriterionResult(
                id=c['id'],
                section_id=c['section_id'],
                requirement=c['requirement'],
                weight=c['weight'],
                verdict='UNMET',
                explanation=f'Batch judge failed: {e}',
            )
            for c in criteria
        ]


def _compute_scores(
    criterion_results: list[CriterionResult],
    rubric: dict,
) -> tuple[float, float, float, list[SectionScore]]:
    """根据 criterion 评估结果计算分数。

    使用 DRACO 论文的评分公式：
        raw_score = sum(v_i * w_i for all i)
        normalized_score = clamp(raw_score / sum(w_i for w_i > 0), 0, 1)

    Returns:
        (normalized_score, raw_score, total_positive_weight, section_scores)
    """
    # 总体分数
    raw_score = sum((1.0 if r.verdict == 'MET' else 0.0) * r.weight for r in criterion_results)
    total_positive_weight = sum(r.weight for r in criterion_results if r.weight > 0)

    normalized_score = max(0.0, min(1.0, raw_score / total_positive_weight)) if total_positive_weight > 0 else 0.0

    # 按 section 计算分数
    section_scores = []
    for section in rubric.get('sections', []):
        section_id = section.get('id', '')
        section_title = section.get('title', '')
        section_results = [r for r in criterion_results if r.section_id == section_id]

        if not section_results:
            continue

        section_raw = sum((1.0 if r.verdict == 'MET' else 0.0) * r.weight for r in section_results)
        section_positive_weight = sum(r.weight for r in section_results if r.weight > 0)
        section_negative_weight = sum(abs(r.weight) for r in section_results if r.weight < 0)

        if section_positive_weight > 0:
            section_norm = max(0.0, min(1.0, section_raw / section_positive_weight))
        elif section_negative_weight > 0:
            # 全负权重 section：无错误得 1.0，全错误得 0.0
            section_norm = max(0.0, min(1.0, 1.0 + section_raw / section_negative_weight))
        else:
            section_norm = 0.0

        section_scores.append(
            SectionScore(
                id=section_id,
                title=section_title,
                score=section_norm,
                raw_score=section_raw,
                total_positive_weight=section_positive_weight,
                criteria_met=sum(1 for r in section_results if r.verdict == 'MET'),
                criteria_total=len(section_results),
            )
        )

    return normalized_score, raw_score, total_positive_weight, section_scores


async def draco_rubric_eval(
    question: str,
    predicted: str,
    expected: str,
    *,
    judge_system_prompt: str | None = None,
    max_response_length: int = 16384,
    batch_size: int = 0,
    max_concurrency: int = 10,
) -> tuple[float, DracoRubricDetails]:
    """DRACO Rubric 评估主入口。

    解析 expected 中的 rubric JSON，对 predicted 的每个 criterion 进行 LLM 评估，
    然后计算加权归一化分数。

    Args:
        question: 原始研究问题。
        predicted: 被评估的回答文本。
        expected: DRACO 数据集中的 answer 字段（JSON 编码的 rubric）。
        judge_system_prompt: 可选自定义 system prompt。
        max_response_length: response 最大截断长度（默认 16384）。
        batch_size: 批量评估大小。0 表示逐条评估（并行），>0 表示按 batch 打包评估。
        max_concurrency: 最大并发数（逐条评估模式下的并发 LLM 调用数）。

    Returns:
        (normalized_score, DracoRubricDetails)
    """
    details = DracoRubricDetails()

    # 解析 rubric
    try:
        rubric = _parse_rubric_json(expected)
    except (json.JSONDecodeError, TypeError) as e:
        logger.error(f'Failed to parse rubric JSON: {e}')
        details.msg = f'Failed to parse rubric JSON: {e}'
        return 0.0, details

    # 展平所有 criteria
    criteria = _flatten_criteria(rubric)
    if not criteria:
        details.msg = 'No criteria found in rubric'
        return 0.0, details

    details.criteria_total = len(criteria)

    # 评估所有 criteria
    if batch_size > 0:
        # 批量模式：按 batch_size 分组，每组一次 LLM 调用
        all_results: list[CriterionResult] = []
        for i in range(0, len(criteria), batch_size):
            batch = criteria[i : i + batch_size]
            batch_results = await _judge_criteria_batch(
                question,
                predicted,
                batch,
                system_prompt=judge_system_prompt,
                max_response_length=max_response_length,
            )
            all_results.extend(batch_results)
    else:
        # 逐条并行模式：使用 semaphore 控制并发
        semaphore = asyncio.Semaphore(max_concurrency)

        async def _judge_with_semaphore(criterion: dict) -> CriterionResult:
            async with semaphore:
                return await _judge_single_criterion(
                    question,
                    predicted,
                    criterion,
                    system_prompt=judge_system_prompt,
                    max_response_length=max_response_length,
                )

        tasks = [_judge_with_semaphore(c) for c in criteria]
        all_results = list(await asyncio.gather(*tasks))

    # 计算分数
    normalized_score, raw_score, total_positive_weight, section_scores = _compute_scores(all_results, rubric)

    # 填充 details
    details.normalized_score = normalized_score
    details.raw_score = raw_score
    details.total_positive_weight = total_positive_weight
    details.section_scores = section_scores
    details.criterion_results = all_results
    details.criteria_met = sum(1 for r in all_results if r.verdict == 'MET')

    # 正/负权重统计
    positive_results = [r for r in all_results if r.weight > 0]
    negative_results = [r for r in all_results if r.weight < 0]
    details.positive_criteria_met = sum(1 for r in positive_results if r.verdict == 'MET')
    details.positive_criteria_total = len(positive_results)
    details.negative_criteria_met = sum(1 for r in negative_results if r.verdict == 'MET')
    details.negative_criteria_total = len(negative_results)

    logger.info(
        f'DRACO Rubric: normalized_score={normalized_score:.3f}, '
        f'raw_score={raw_score:.1f}/{total_positive_weight:.1f}, '
        f'criteria_met={details.criteria_met}/{details.criteria_total}, '
        f'sections={[f"{s.id}={s.score:.2f}" for s in section_scores]}'
    )

    return normalized_score, details
