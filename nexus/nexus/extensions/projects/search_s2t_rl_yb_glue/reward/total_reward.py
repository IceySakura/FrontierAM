"""总奖励计算模块（基于 examples/yuanbao_search 的 agts 奖励）。

整合多个评分组件计算最终奖励（``data_source`` 含 ``agts_closed_qa`` 时）：

1. 格式检查（rule_based_judge）：``planner_rule_check == -1`` ⇒ 最终分数 ``-1.0``；
2. 必须搜索门控：没有包含非空 query 的有效搜索 ⇒ 最终分数 ``0.0``，并跳过 LLM Judge；
3. 硬门控（gate_judge，与 LLM Judge 并行）：format / safety / length / natural / logic 五个门控中
   任一被判负 ⇒ 最终分数 ``0.0``（纯 pass/fail 开关，不参与语义分加权）；
4. LLM Judge 评分：``closed_qa_score ∈ {0.0, 0.5, 1.0}``；
5. 搜索不再打折：``base_score = closed_qa_score``；
6. 小幅子查询成本：每次搜索第 1 个子查询免费，额外子查询每个扣 ``0.01``，
   单条轨迹最多扣 ``0.05``；
7. 最终分数：``max(0.0, closed_qa_score - query_cost)``。

有效搜索次数使用结构化轨迹里的 ``extra_info['valid_search_count']``：只有搜索 Tool
返回且 ``ToolResponse.meta['subquery_count'] > 0`` 才计数，空白 query 不会解锁语义分。
"""

import asyncio
import logging

from typing import Any

from . import gate_judge, llm_judge, rule_based_judge


logger = logging.getLogger(__name__)

QUERY_COST_PER_EXTRA_SUBQUERY = 0.01
MAX_QUERY_COST = 0.05


def _count_tool_calls(solution_str: str, extra_info: dict | None = None, tool_name: str | None = None) -> int:
    """统计 tool call 的数量。

    优先使用结构化轨迹里 agent 统计好的 ``extra_info['tool_call_count']``；仅当缺失时才
    回退到扫描 DeepSeek 内联标记（对结构化输出而言通常为 0）。
    """
    extra_info = extra_info or {}
    if 'tool_call_count' in extra_info and extra_info['tool_call_count'] is not None:
        try:
            return int(extra_info['tool_call_count'])
        except (TypeError, ValueError):
            pass

    if tool_name is None:
        return solution_str.count('<｜tool▁call▁begin｜>')
    pattern = f'<｜tool▁call▁begin｜>{tool_name}<｜tool▁sep｜>'
    return solution_str.count(pattern)


def _count_valid_searches(extra_info: dict | None = None) -> int:
    """读取 Agent 统计的有效搜索次数，字段缺失或非法时 fail closed 为 0。"""
    extra_info = extra_info or {}
    try:
        return max(0, int(extra_info.get('valid_search_count', 0) or 0))
    except (TypeError, ValueError):
        return 0


def _compute_query_cost(extra_info: dict | None = None) -> tuple[int, int, float]:
    """Return ``(subquery_count, extra_subquery_count, query_cost)``.

    One subquery per accepted search call is free. Extra attempted subqueries
    carry a deliberately small additive cost, capped so semantic correctness
    remains the dominant reward component.
    """
    extra_info = extra_info or {}
    try:
        subquery_count = max(0, int(extra_info.get('subquery_count', 0) or 0))
    except (TypeError, ValueError):
        subquery_count = 0
    try:
        search_count = max(0, int(extra_info.get('search_count', 0) or 0))
    except (TypeError, ValueError):
        search_count = 0

    extra_subquery_count = max(0, subquery_count - search_count)
    query_cost = min(MAX_QUERY_COST, QUERY_COST_PER_EXTRA_SUBQUERY * extra_subquery_count)
    return subquery_count, extra_subquery_count, float(query_cost)


async def compute_score_async(
    solution_str: str,
    ground_truth: str = '无参考答案',
    extra_info: dict | None = None,
    data_source: str = '',
) -> dict[str, Any]:
    """异步计算总奖励分数。

    分流逻辑：
    - data_source 含 'agts_closed_qa'（本项目数据）：格式检查 + 必须搜索门控 + LLM Judge。
    - 其他：仅 rule 检查各项求和（保留 agts 的兜底逻辑）。
    """
    extra_info = extra_info or {}

    if 'agts_closed_qa' in data_source:
        # Step 1: 格式检查
        rule_score_dict = await rule_based_judge.compute_score(
            solution_str=solution_str,
            ground_truth=ground_truth,
            extra_info=extra_info,
        )

        planner_rule_check = rule_score_dict.get('planner_rule_check', 0)
        format_failed = planner_rule_check == -1
        tool_call_count = max(0, _count_tool_calls(solution_str, extra_info))
        valid_search_count = _count_valid_searches(extra_info)
        subquery_count, extra_subquery_count, query_cost = _compute_query_cost(extra_info)

        if format_failed:
            logger.warning(f'format check failed: planner_rule_check={planner_rule_check}')

            merged_score_dict = {}
            merged_score_dict.update(rule_score_dict)
            merged_score_dict['score'] = -1.0
            merged_score_dict['format_failed'] = True
            merged_score_dict['tool_call_count'] = tool_call_count
            merged_score_dict['valid_search_count'] = valid_search_count
            merged_score_dict['discount'] = 0.0
            merged_score_dict['search_performed'] = valid_search_count > 0
            merged_score_dict['judge_skipped'] = True
            merged_score_dict['subquery_count'] = subquery_count
            merged_score_dict['extra_subquery_count'] = extra_subquery_count
            merged_score_dict['query_cost'] = query_cost
            merged_score_dict['query_cost_applied'] = 0.0
            merged_score_dict.update({
                'closed_qa_score': 0,
                'judgment': '格式错误',
                'confidence': '高',
                'reason': 'format check failed, skipped closed_qa evaluation',
            })
            return merged_score_dict

        # Step 2: 必须搜索门控。ToolResponse 只有在携带至少一个清洗后的非空
        # subquery 时才算有效搜索；空白 query 虽返回 ToolResponse，也不能解锁语义分。
        if valid_search_count == 0:
            logger.info('required-search reward: no valid search performed, skip closed_qa evaluation and return 0')

            merged_score_dict = {}
            merged_score_dict.update(rule_score_dict)
            merged_score_dict.update({
                'score': 0.0,
                'format_failed': False,
                'tool_call_count': tool_call_count,
                'valid_search_count': 0,
                # Keep the historical field for downstream metrics. Search is no
                # longer discounted, so its neutral value is always 1.0.
                'discount': 1.0,
                'base_score': 0.0,
                'search_performed': False,
                'judge_skipped': True,
                'subquery_count': subquery_count,
                'extra_subquery_count': extra_subquery_count,
                'query_cost': query_cost,
                'query_cost_applied': 0.0,
                'closed_qa_score': 0,
                'acc': 0,
                'judgment': '未搜索',
                'confidence': '高',
                'reason': 'required valid search was not performed, skipped closed_qa evaluation',
            })
            return merged_score_dict

        # Step 3: LLM Judge 评分 + 五个硬门控并行发出（两次独立 GRM 请求，wall-clock 最短）。
        closed_qa_score_dict, gate_result = await asyncio.gather(
            llm_judge.compute_score(
                solution_str=solution_str,
                ground_truth=ground_truth,
                extra_info=extra_info,
            ),
            gate_judge.evaluate_gates(
                solution_str=solution_str,
                ground_truth=ground_truth,
                extra_info=extra_info,
            ),
        )
        closed_qa_score = closed_qa_score_dict.get('closed_qa_score', 0)
        gate_passed = gate_result.get('gate_passed', True)
        failed_gates = gate_result.get('failed_gates', [])
        gate_status = gate_result.get('gate_status', {})

        # 硬门控：format / safety / length / natural / logic 任一判负 ⇒ 整条样本归零。
        # 保留语义分等观测字段，仅把最终 score 置 0。
        if not gate_passed:
            logger.info(
                f'gate FAILED {failed_gates} -> reward 0.0 (closed_qa_score={closed_qa_score}, gates={gate_status})'
            )

            merged_score_dict = {}
            merged_score_dict.update(rule_score_dict)
            merged_score_dict.update(closed_qa_score_dict)
            merged_score_dict['score'] = 0.0
            merged_score_dict['format_failed'] = False
            merged_score_dict['tool_call_count'] = tool_call_count
            merged_score_dict['valid_search_count'] = valid_search_count
            merged_score_dict['discount'] = 1.0
            merged_score_dict['base_score'] = float(closed_qa_score)
            merged_score_dict['search_performed'] = True
            merged_score_dict['judge_skipped'] = False
            merged_score_dict['subquery_count'] = subquery_count
            merged_score_dict['extra_subquery_count'] = extra_subquery_count
            merged_score_dict['query_cost'] = query_cost
            merged_score_dict['query_cost_applied'] = 0.0
            merged_score_dict['gate_passed'] = False
            merged_score_dict['failed_gates'] = failed_gates
            merged_score_dict['gate_status'] = gate_status
            merged_score_dict['gate_rm_request_id'] = gate_result.get('gate_rm_request_id', '')
            return merged_score_dict

        # Step 4: 搜索是必需行为，不再使用 0.8^tool_call_count 惩罚搜索。
        discount = 1.0
        base_score = float(closed_qa_score)

        # Step 5: 保留轻量 query-packing 成本，但正常轨迹的最低分保持为 0，
        # 避免“搜索后答错”因 query 成本反而低于“不搜索”的 0 分。
        final_score = max(0.0, min(1.0, base_score - query_cost))
        query_cost_applied = min(query_cost, max(0.0, base_score))

        logger.info(
            f'required-search closed_qa reward: closed_qa_score={closed_qa_score}, '
            f'tool_calls={tool_call_count}, valid_searches={valid_search_count}, '
            f'subqueries={subquery_count}, query_cost={query_cost:.4f}, '
            f'final_score={final_score:.4f}'
        )

        merged_score_dict = {}
        merged_score_dict.update(rule_score_dict)
        merged_score_dict.update(closed_qa_score_dict)
        merged_score_dict['score'] = float(final_score)
        merged_score_dict['format_failed'] = False
        merged_score_dict['tool_call_count'] = tool_call_count
        merged_score_dict['valid_search_count'] = valid_search_count
        merged_score_dict['discount'] = float(discount)
        merged_score_dict['base_score'] = float(base_score)
        merged_score_dict['search_performed'] = True
        merged_score_dict['judge_skipped'] = False
        merged_score_dict['subquery_count'] = subquery_count
        merged_score_dict['extra_subquery_count'] = extra_subquery_count
        merged_score_dict['query_cost'] = query_cost
        merged_score_dict['query_cost_applied'] = float(query_cost_applied)
        merged_score_dict['gate_passed'] = True
        merged_score_dict['failed_gates'] = []
        merged_score_dict['gate_status'] = gate_status
        merged_score_dict['gate_rm_request_id'] = gate_result.get('gate_rm_request_id', '')

        return merged_score_dict

    # 默认逻辑：并行调用多个评分函数（仅 rule 检查）。
    reward_tasks = [
        rule_based_judge.compute_score(
            solution_str=solution_str,
            ground_truth=ground_truth,
            extra_info=extra_info,
        ),
    ]

    score_dicts = await asyncio.gather(*reward_tasks)

    merged_score_dict = {}
    for score_dict in score_dicts:
        merged_score_dict.update(score_dict)

    planner_rule_check = merged_score_dict.get('planner_rule_check', 0)
    answer_rule_check = merged_score_dict.get('answer_rule_check', 0)
    picture_rule_check = merged_score_dict.get('picture_rule_check', 0)
    video_rule_check = merged_score_dict.get('video_rule_check', 0)
    search_rule_check = merged_score_dict.get('search_rule_check', 0)

    total_score = planner_rule_check + answer_rule_check + picture_rule_check + video_rule_check + search_rule_check
    merged_score_dict['score'] = float(total_score)

    return merged_score_dict


def compute_score(
    solution_str: str,
    ground_truth: str = '无参考答案',
    extra_info: dict | None = None,
    data_source: str = '',
) -> dict[str, Any]:
    """同步接口的 compute_score，内部使用异步调用。"""
    return asyncio.run(
        compute_score_async(
            solution_str=solution_str,
            ground_truth=ground_truth,
            extra_info=extra_info,
            data_source=data_source,
        )
    )
