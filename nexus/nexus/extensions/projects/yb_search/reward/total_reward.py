"""
总奖励计算模块。

整合多个评分组件计算最终奖励：
1. 格式检查（rule_based_judge）
2. LLM Judge 评分 / Pairwise RM 评分
3. 工具调用折扣
"""

import asyncio
import logging
import os

from typing import Any

from . import (
    closed_qa_judge,
    closed_qa_rubric_judge,
    format_judge,
    grm_judge,
    hy_reward,
    logical_consistency_judge,
    pairwise_judge,
    pairwise_rubrics,
    relevance_judge,
    rubric_judge,
    rule_based_judge,
    # style_judge,  # style_score 未参与总分计算，暂时禁用以节省 LLM 资源
)
from .tool_call_utils import compute_media_bonus, compute_tool_call_stats, count_tool_calls, get_turn_count


logger = logging.getLogger(__name__)
TOTAL_TIMEOUT = int(os.environ.get('TAIJI_JUDGE_TOTAL_TIMEOUT', 600))
PAIRWISE_SCORE_WEIGHT = float(os.environ.get('PAIRWISE_SCORE_WEIGHT', 0.5))
PAIRWISE_RUBRIC_WEIGHT = float(os.environ.get('PAIRWISE_RUBRIC_WEIGHT', 0.5))


_DEFAULT_SCORES: dict[str, dict[str, Any]] = {
    'rule_based_judge': {
        'answer_rule_check': 0,
        'planner_rule_check': 0,
        'search_rule_check': 0,
        'picture_rule_check': 0,
        'video_rule_check': 0,
        'think_rule_check': 0,
    },
    'format_judge': {
        'format_score': 0,
        'language_consistency_score': 0,
    },
    'relevance_judge': {
        'relevance_score': 0,
        '基础指令遵循': '{}',
        '基础回复合理性': '{}',
        '多轮效果': '{}',
    },
    'logical_consistency_judge': {
        'logic_score': 0,
        '推理逻辑谬误': '{}',
        '前后一致性': '{}',
        '概念使用准确性': '{}',
        '论证完整性': '{}',
        '因果关系正确性': '{}',
        '回答实用性': '{}',
    },
    'style_judge': {
        'style_score': 0.5,
        'style_raw_score': 1,
        'style_dim_scores': {},
        'style_error': '',
    },
    'pairwise_judge': {
        'pairwise_score': 0.0,
        'pairwise_raw_score': 0.0,
        'pairwise_error': '',
    },
    'pairwise_rubrics': {
        'pairwise_rubric_score': 0.0,
        'pairwise_rubric_raw_score': 0.0,
        'pairwise_rubric_max_score': 0.0,
        'pairwise_rubric_fatal': False,
        'pairwise_rubric_must_pass_score': 0.0,
        'pairwise_rubric_hit_score': 0.0,
        'pairwise_rubric_bonus_score': 0.0,
        'pairwise_rubric_balance_bonus': 0.0,
        'pairwise_rubric_n_hit': 0,
        'pairwise_rubric_n_total': 0,
        'pairwise_rubric_results': [],
        'pairwise_rubric_overall_reason': '',
        'pairwise_rubric_error': '',
        'pairwise_rubric_error_type': '',
        'pairwise_rubric_skip': True,
    },
    'grm_judge': {
        'grm_score': 0.0,
        'grm_raw_score': 0.0,
        'grm_pair_scores': '[]',
        'grm_num_refs': 0,
        'grm_error': '',
    },
    'closed_qa_judge': {
        'closed_qa_score': 0.0,
        'judgment': '不匹配',
        'confidence': '低',
        'reason': '',
        'conciseness': '符合',
        'conciseness_reason': '',
    },
    'hy_reward': {
        'hy_score': 0,
        'hy_score_json': '{}',
    },
    'rubric_judge': {
        'rubric_score': 0.0,
        'rubric_raw_score': 0.0,
        'rubric_max_score': 0,
        'rubric_trap_penalty': 0.0,
        'rubric_details': '[]',
        'rubric_overall_comment': '',
        'rubric_overall_quality': '',
        'rubric_error': '',
    },
}

_ERROR_FIELDS = ('style_error', 'pairwise_error', 'pairwise_rubric_error', 'grm_error', 'rubric_error', 'reason')


def _canonical_reward_field_defaults() -> dict[str, Any]:
    """Guarantee stable schema across all branches.

    Used to canonicalize the per-branch return so that pairwise / closed_qa /
    closed_qa_rubric / hunyuan / default branches all return the SAME schema. This
    guarantees that downstream consumers see stable columns regardless of:
      - which branch was taken for a given sample
      - which data_sources happened to appear in a particular batch
    """
    return {
        # rule_based_judge
        'answer_rule_check': 0.0,
        'planner_rule_check': 0.0,
        'search_rule_check': 0.0,
        'picture_rule_check': 0.0,
        'video_rule_check': 0.0,
        'think_rule_check': 0.0,
        # format_judge
        'format_score': 0.0,
        'language_consistency_score': 0.0,
        # relevance_judge
        'relevance_score': 0.0,
        '基础指令遵循': '{}',
        '基础回复合理性': '{}',
        '多轮效果': '{}',
        # logical_consistency_judge
        'logic_score': 0.0,
        '推理逻辑谬误': '{}',
        '前后一致性': '{}',
        '概念使用准确性': '{}',
        '论证完整性': '{}',
        '因果关系正确性': '{}',
        '回答实用性': '{}',
        # pairwise_judge
        'pairwise_score': 0.0,
        'pairwise_raw_score': 0.0,
        'pairwise_error': '',
        # grm
        'grm_score': 0.0,
        'grm_raw_score': 0.0,
        'grm_pair_scores': '[]',
        'grm_num_refs': 0.0,
        'grm_error': '',
        # pairwise_rubrics
        'pairwise_rubric_score': 0.0,
        'pairwise_rubric_raw_score': 0.0,
        'pairwise_rubric_max_score': 0.0,
        'pairwise_rubric_fatal': False,
        'pairwise_rubric_must_pass_score': 0.0,
        'pairwise_rubric_hit_score': 0.0,
        'pairwise_rubric_bonus_score': 0.0,
        'pairwise_rubric_balance_bonus': 0.0,
        'pairwise_rubric_n_hit': 0,
        'pairwise_rubric_n_total': 0,
        'pairwise_rubric_results': [],
        'pairwise_rubric_overall_reason': '',
        'pairwise_rubric_error': '',
        'pairwise_rubric_error_type': '',
        'pairwise_rubric_skip': True,
        # closed_qa_judge / closed_qa_rubric_judge (shared output fields)
        'closed_qa_score': 0.0,
        'judgment': '不匹配',
        'confidence': '低',
        'reason': '',
        'conciseness': '符合',
        'conciseness_reason': '',
        # closed_qa_rubric_judge specific
        'rubric_verify_score': 0.0,
        'rubric_verify_must_score': 0.0,
        'rubric_verify_nice_score': None,
        'rubric_verify_results': [],
        'rubric_verify_overall_reason': '',
        'rubric_verify_must_weight': 0.7,
        'rubric_verify_nice_weight': 0.3,
        'rubric_verify_error': '',
        'rubric_verify_error_type': '',
        # hy_reward
        'hy_score': 0.0,
        'hy_score_json': '{}',
        # style fields
        'style_score': 0.0,
        'style_raw_score': 0.0,
        'style_dim_scores': {},
        'style_error': '',
        # tool call stats
        'tool_call_count': 0,
        'turn_count': 0,
        'total_queries': 0,
        'max_web_queries': 0,
        'max_picture_queries': 0,
        'max_video_queries': 0,
        'tool_call_discount': 0.0,
        'max_tool_calls_per_turn': 0,
        'turn_discount': 0.0,
        'discount': 0.0,
        # reward runtime stats
        'reward_timeout_count': 0,
        'reward_error_count': 0,
        'reward_judge_timeout_count': 0,
        'reward_judge_error_count': 0,
        'reward_rate_limit_wait_time': 0.0,
        'reward_error_sources': [],
        # rubric_judge (badcase)
        'rubric_score': 0.0,
        'rubric_raw_score': 0.0,
        'rubric_max_score': 0.0,
        'rubric_trap_penalty': 0.0,
        'rubric_details': '[]',
        'rubric_overall_comment': '',
        'rubric_overall_quality': '',
        'rubric_error': '',
        # media bonus
        'media_bonus': 0.0,
        'picture_media_bonus': 0.0,
        'video_media_bonus': 0.0,
        # branch flags
        'format_failed': False,
        'any_check_failed': False,
        'rule_check_failed': False,
        'format_judge_failed': False,
        # final aggregated score
        'score': 0.0,
    }


def _canonicalize_score_dict(merged: dict[str, Any]) -> dict[str, Any]:
    """Return a dict carrying the full canonical reward schema.

    Branch-specific values in `merged` always win; missing keys are filled with
    the canonical default.
    """
    canonical = _canonical_reward_field_defaults()
    for k, v in merged.items():
        canonical[k] = v
    return canonical


def _default_score_dict(judge_name: str, error_message: str = '') -> dict[str, Any]:
    template = _DEFAULT_SCORES.get(judge_name, {})
    result = {k: (dict(v) if isinstance(v, dict) else v) for k, v in template.items()}
    if error_message:
        for key in _ERROR_FIELDS:
            if key in result:
                result[key] = error_message
                break
    return result


def _init_reward_runtime_stats() -> dict[str, Any]:
    return {
        'reward_timeout_count': 0,
        'reward_error_count': 0,
        'reward_judge_timeout_count': 0,
        'reward_judge_error_count': 0,
        'reward_rate_limit_wait_time': 0.0,
        'reward_error_sources': [],
    }


def _merge_reward_runtime_stats(base: dict[str, Any], delta: dict[str, Any]) -> dict[str, Any]:
    base['reward_timeout_count'] = int(base.get('reward_timeout_count', 0)) + int(delta.get('reward_timeout_count', 0))
    base['reward_error_count'] = int(base.get('reward_error_count', 0)) + int(delta.get('reward_error_count', 0))
    base['reward_judge_timeout_count'] = int(base.get('reward_judge_timeout_count', 0)) + int(
        delta.get('reward_judge_timeout_count', 0)
    )
    base['reward_judge_error_count'] = int(base.get('reward_judge_error_count', 0)) + int(
        delta.get('reward_judge_error_count', 0)
    )
    base['reward_rate_limit_wait_time'] = float(base.get('reward_rate_limit_wait_time', 0.0) or 0.0) + float(
        delta.get('reward_rate_limit_wait_time', 0.0) or 0.0
    )
    base['reward_error_sources'].extend(delta.get('reward_error_sources', []))
    return base


async def _safe_compute_score(
    judge_name: str, coro: Any, traj_id: str = 'unknown'
) -> tuple[dict[str, Any], dict[str, Any]]:
    stats = _init_reward_runtime_stats()
    tag = f'[{judge_name}][{traj_id}]'
    try:
        result = await asyncio.wait_for(coro, timeout=TOTAL_TIMEOUT)
        if not isinstance(result, dict):
            logger.warning(f'{tag} invalid judge output type={type(result).__name__}, fallback to empty dict')
            stats['reward_error_count'] = 1
            stats['reward_judge_error_count'] = 1
            stats['reward_error_sources'].append(f'{judge_name}:InvalidResultType')
            return _default_score_dict(judge_name, 'InvalidResultType'), stats
        stats['reward_rate_limit_wait_time'] = float(result.pop('reward_rate_limit_wait_time', 0.0) or 0.0)
        return result, stats
    except asyncio.TimeoutError:
        logger.warning(f'{tag} timeout after {TOTAL_TIMEOUT}s')
        stats['reward_timeout_count'] = 1
        stats['reward_error_count'] = 1
        stats['reward_judge_timeout_count'] = 1
        stats['reward_judge_error_count'] = 1
        stats['reward_error_sources'].append(f'{judge_name}:TotalTimeout')
        return _default_score_dict(judge_name, f'Timeout after {TOTAL_TIMEOUT}s'), stats
    except Exception as e:
        is_timeout = isinstance(e, (asyncio.TimeoutError, TimeoutError)) or 'timeout' in type(e).__name__.lower()
        if is_timeout:
            stats['reward_timeout_count'] = 1
            stats['reward_judge_timeout_count'] = 1
        logger.warning(f'{tag} failed with {type(e).__name__}: {e}', exc_info=True)
        stats['reward_error_count'] = 1
        stats['reward_judge_error_count'] = 1
        stats['reward_error_sources'].append(f'{judge_name}:{type(e).__name__}')
        return _default_score_dict(judge_name, f'{type(e).__name__}: {e}'), stats


async def compute_score_async(
    solution_str: str,
    ground_truth: str = '无参考答案',
    extra_info: dict | None = None,
    data_source: str = '',
) -> dict[str, Any]:
    """
    异步计算总奖励分数。

    分流逻辑：
    - data_source 包含 'pairwise': 使用 pairwise reward model 评估
    - data_source 包含 'agts_closed_qa': 使用简化的 closed_qa 评估流程
    - data_source 包含 'hunyuan': 使用 HY Reward 平台 API
    - 其他: 使用规则检查

    Args:
        solution_str: 模型生成的完整回复
        ground_truth: 参考答案
        extra_info: 额外信息
        data_source: 数据来源标识

    Returns:
        包含各项评分的字典
    """
    extra_info = extra_info or {}
    traj_id = extra_info.get('traj_id', 'unknown')
    reward_runtime_stats = _init_reward_runtime_stats()

    # ========== pairwise 数据使用 pairwise reward model ==========
    # 对齐 baseline agts_total_reward.py 的 pairwise 分支
    # 1. 并行检测 rule_based_judge 和 format_judge
    # 2. 调用 pairwise 评分
    # 3. 如果任何检查失败，final_score = 0；否则 final_score = pairwise_score * discount
    if 'pairwise' in data_source:
        # Step 1: 并行检测 rule_based_judge 和 format_judge
        (rule_score_dict, rule_stats), (format_score_dict, format_stats) = await asyncio.gather(
            _safe_compute_score(
                'rule_based_judge',
                rule_based_judge.compute_score(
                    solution_str=solution_str,
                    ground_truth=ground_truth,
                    extra_info=extra_info,
                ),
                traj_id=traj_id,
            ),
            _safe_compute_score(
                'format_judge',
                format_judge.compute_score(
                    solution_str=solution_str,
                    ground_truth=ground_truth,
                    extra_info=extra_info,
                ),
                traj_id=traj_id,
            ),
        )
        _merge_reward_runtime_stats(reward_runtime_stats, rule_stats)
        _merge_reward_runtime_stats(reward_runtime_stats, format_stats)

        # 获取 rule_based_judge 所有检查项
        answer_rule_check = rule_score_dict.get('answer_rule_check', 0)
        planner_rule_check = rule_score_dict.get('planner_rule_check', 0)
        search_rule_check = rule_score_dict.get('search_rule_check', 0)
        picture_rule_check = rule_score_dict.get('picture_rule_check', 0)
        video_rule_check = rule_score_dict.get('video_rule_check', 0)
        think_rule_check = rule_score_dict.get('think_rule_check', 0)

        # 获取 format_judge 检查项
        format_score = format_score_dict.get('format_score', 0)
        language_consistency_score = format_score_dict.get('language_consistency_score', 0)

        # 检查是否有任何 rule_based 检查失败（任意一项为 -1）
        rule_failed = (
            answer_rule_check < 0
            or planner_rule_check < 0
            or search_rule_check < 0
            or picture_rule_check < 0
            or video_rule_check < 0
            or think_rule_check < 0
        )
        # 检查是否有任何 format_judge 失败
        format_judge_failed = format_score == -1 or language_consistency_score == -1
        reward_failed = int(reward_runtime_stats.get('reward_error_count', 0)) > 0
        any_check_failed = rule_failed or format_judge_failed or reward_failed

        # Step 2: 调用 pairwise 评分 (BT-RM 或 GRM 二选一)
        if 'GRM_refans1' in extra_info and extra_info['GRM_refans1']:
            (
                (pairwise_score_dict, pairwise_stats),
                (pairwise_rubric_score_dict, pairwise_rubric_stats),
            ) = await asyncio.gather(
                _safe_compute_score(
                    'grm_judge',
                    grm_judge.compute_score(
                        solution_str=solution_str,
                        ground_truth=ground_truth,
                        extra_info=extra_info,
                    ),
                    traj_id=traj_id,
                ),
                _safe_compute_score(
                    'pairwise_rubrics',
                    pairwise_rubrics.compute_score(
                        solution_str=solution_str,
                        ground_truth=ground_truth,
                        extra_info=extra_info,
                    ),
                    traj_id=traj_id,
                ),
            )
            pairwise_score = pairwise_score_dict.get('grm_score', 0)
        else:
            (
                (pairwise_score_dict, pairwise_stats),
                (pairwise_rubric_score_dict, pairwise_rubric_stats),
            ) = await asyncio.gather(
                _safe_compute_score(
                    'pairwise_judge',
                    pairwise_judge.compute_score(
                        solution_str=solution_str,
                        ground_truth=ground_truth,
                        extra_info=extra_info,
                    ),
                    traj_id=traj_id,
                ),
                _safe_compute_score(
                    'pairwise_rubrics',
                    pairwise_rubrics.compute_score(
                        solution_str=solution_str,
                        ground_truth=ground_truth,
                        extra_info=extra_info,
                    ),
                    traj_id=traj_id,
                ),
            )
            pairwise_score = pairwise_score_dict.get('pairwise_score', 0)
        pairwise_rubric_skip = pairwise_rubric_score_dict.get('pairwise_rubric_skip', True)
        pairwise_rubric_score = pairwise_rubric_score_dict.get('pairwise_rubric_score', 0)
        _merge_reward_runtime_stats(reward_runtime_stats, pairwise_stats)
        _merge_reward_runtime_stats(reward_runtime_stats, pairwise_rubric_stats)

        # Step 3: 计算 tool call 统计和 discount
        stats = compute_tool_call_stats(extra_info, solution_str)
        tool_call_discount = stats['tool_call_discount']
        media_details: dict[str, Any] = {'media_bonus': 0.0}

        # 如果任何检查失败，final_score = 0
        if any_check_failed:
            final_score = 0.0
            logger.warning(
                f'[pairwise] check failed: '
                f'rule_failed={rule_failed} (answer={answer_rule_check}, planner={planner_rule_check}, '
                f'search={search_rule_check}, picture={picture_rule_check}, video={video_rule_check}, '
                f'think={think_rule_check}), '
                f'format_judge_failed={format_judge_failed} (format={format_score}, lang={language_consistency_score}), '
                f'pairwise_score={pairwise_score:.4f} -> final_score=0'
            )
        else:
            # 计算 media bonus
            media_bonus, media_details = compute_media_bonus(extra_info, solution_str)
            if pairwise_rubric_skip:
                combined_score = pairwise_score
            elif pairwise_rubric_score == 0:
                combined_score = 0.0
            else:
                combined_score = PAIRWISE_SCORE_WEIGHT * pairwise_score + PAIRWISE_RUBRIC_WEIGHT * pairwise_rubric_score
            final_score = min(1.0, (combined_score + media_bonus) * tool_call_discount)
            logger.warning(
                f'[pairwise] all checks passed: pairwise_score={pairwise_score:.4f}, '
                f'pairwise_rubric_score={pairwise_rubric_score:.4f}, '
                f'media_bonus={media_bonus:.4f}, '
                f'queries(web={stats["max_web_queries"]}, pic={stats["max_picture_queries"]}, '
                f'vid={stats["max_video_queries"]}, total={stats["total_queries"]}), '
                f'tool_call_discount={tool_call_discount:.4f}, combined_score={combined_score:.4f}, '
                f'final_score={final_score:.4f}'
            )

        merged_score_dict: dict[str, Any] = {}
        merged_score_dict.update(rule_score_dict)
        merged_score_dict.update(format_score_dict)
        merged_score_dict.update(pairwise_score_dict)
        merged_score_dict.update(pairwise_rubric_score_dict)
        merged_score_dict.update(media_details)
        merged_score_dict['score'] = float(final_score)
        merged_score_dict['any_check_failed'] = any_check_failed
        merged_score_dict['rule_check_failed'] = rule_failed
        merged_score_dict['format_judge_failed'] = format_judge_failed
        merged_score_dict.update(stats)
        merged_score_dict.update(reward_runtime_stats)

        return _canonicalize_score_dict(merged_score_dict)

    # ========== badcase_rubric_search / badcase_rubric_main: rubric judge 分支 ==========
    # 必须放在 pairwise / agts_closed_qa 分支之前，否则会被截胡。
    if 'badcase_rubric_search' in data_source or 'badcase_rubric_main' in data_source:
        rubric_score_dict: dict[str, Any] = {}
        tool_call_discount = 1.0

        if 'badcase_rubric_search' in data_source:
            # badcase_rubric_search: 并行 rule_based + format 检查，通过后调用 rubric_judge
            # 构造 judge_extra_info（兜底 question 字段）
            judge_extra_info = dict(extra_info)
            if not judge_extra_info.get('current_prompt'):
                judge_extra_info['current_prompt'] = judge_extra_info.get('question', '')

            (rule_score_dict, rule_stats), (format_score_dict, format_stats) = await asyncio.gather(
                _safe_compute_score(
                    'rule_based_judge',
                    rule_based_judge.compute_score(
                        solution_str=solution_str,
                        ground_truth=ground_truth,
                        extra_info=extra_info,
                    ),
                    traj_id=traj_id,
                ),
                _safe_compute_score(
                    'format_judge',
                    format_judge.compute_score(
                        solution_str=solution_str,
                        ground_truth=ground_truth,
                        extra_info=extra_info,
                    ),
                    traj_id=traj_id,
                ),
            )
            _merge_reward_runtime_stats(reward_runtime_stats, rule_stats)
            _merge_reward_runtime_stats(reward_runtime_stats, format_stats)

            # 检查 rule_based
            answer_rule_check = rule_score_dict.get('answer_rule_check', 0)
            planner_rule_check = rule_score_dict.get('planner_rule_check', 0)
            search_rule_check = rule_score_dict.get('search_rule_check', 0)
            picture_rule_check = rule_score_dict.get('picture_rule_check', 0)
            video_rule_check = rule_score_dict.get('video_rule_check', 0)
            think_rule_check = rule_score_dict.get('think_rule_check', 0)

            format_score = format_score_dict.get('format_score', 0)
            language_consistency_score = format_score_dict.get('language_consistency_score', 0)

            rule_failed = (
                answer_rule_check == -1
                or planner_rule_check == -1
                or search_rule_check == -1
                or picture_rule_check == -1
                or video_rule_check == -1
                or think_rule_check == -1
            )
            format_judge_failed = format_score == -1 or language_consistency_score == -1
            reward_failed = int(reward_runtime_stats.get('reward_error_count', 0)) > 0
            any_check_failed = rule_failed or format_judge_failed or reward_failed

            if any_check_failed:
                logger.warning(
                    f'[badcase_rubric_search] check failed: '
                    f'rule_failed={rule_failed}, format_judge_failed={format_judge_failed}, '
                    f'reward_failed={reward_failed} -> final_score=0'
                )
                final_score = 0.0
                merged_score_dict = {}
                merged_score_dict.update(rule_score_dict)
                merged_score_dict.update(format_score_dict)
                merged_score_dict['score'] = float(final_score)
                merged_score_dict['any_check_failed'] = any_check_failed
                merged_score_dict['rule_check_failed'] = rule_failed
                merged_score_dict['format_judge_failed'] = format_judge_failed
                merged_score_dict.update(reward_runtime_stats)
                return _canonicalize_score_dict(merged_score_dict)

            # 通过检查，调用 rubric_judge
            rubric_score_dict, rubric_stats = await _safe_compute_score(
                'rubric_judge',
                rubric_judge.compute_score(
                    solution_str=solution_str,
                    ground_truth=ground_truth,
                    extra_info=judge_extra_info,
                ),
                traj_id=traj_id,
            )
            _merge_reward_runtime_stats(reward_runtime_stats, rubric_stats)

            rubric_score = rubric_score_dict.get('rubric_score', 0.0)

            # 计算 tool call discount
            stats = compute_tool_call_stats(extra_info, solution_str)
            tool_call_discount = stats['tool_call_discount']
            final_score = rubric_score * tool_call_discount

            logger.warning(
                f'[badcase_rubric_search] rubric_score={rubric_score:.4f}, '
                f'tool_call_discount={tool_call_discount:.4f}, '
                f'final_score={final_score:.4f}'
            )

            merged_score_dict = {}
            merged_score_dict.update(rule_score_dict)
            merged_score_dict.update(format_score_dict)
            merged_score_dict.update(rubric_score_dict)
            merged_score_dict['score'] = float(final_score)
            merged_score_dict['any_check_failed'] = False
            merged_score_dict.update(stats)
            merged_score_dict.update(reward_runtime_stats)
            return _canonicalize_score_dict(merged_score_dict)

        # badcase_rubric_main: 直接调用 rubric_judge，无 rule/format 前置检查
        rubric_score_dict, rubric_stats = await _safe_compute_score(
            'rubric_judge',
            rubric_judge.compute_score(
                solution_str=solution_str,
                ground_truth=ground_truth,
                extra_info=extra_info,
            ),
            traj_id=traj_id,
        )
        _merge_reward_runtime_stats(reward_runtime_stats, rubric_stats)

        rubric_score = rubric_score_dict.get('rubric_score', 0.0)
        final_score = rubric_score  # 无 discount

        logger.warning(f'[badcase_rubric_main] rubric_score={rubric_score:.4f}, final_score={final_score:.4f}')

        merged_score_dict = {}
        merged_score_dict.update(rubric_score_dict)
        merged_score_dict['score'] = float(final_score)
        merged_score_dict.update(reward_runtime_stats)
        return _canonicalize_score_dict(merged_score_dict)

    # ========== agts_closed_qa_rubric: rubric verifier 逻辑 ==========
    # 必须放在 agts_closed_qa 分支之前，否则会被 'agts_closed_qa' in data_source 截胡。
    if 'agts_closed_qa' in data_source and 'rubric' in data_source:
        # Step 1: 格式检查
        rule_score_dict, rule_stats = await _safe_compute_score(
            'rule_based_judge',
            rule_based_judge.compute_score(
                solution_str=solution_str,
                ground_truth=ground_truth,
                extra_info=extra_info,
            ),
            traj_id=traj_id,
        )
        _merge_reward_runtime_stats(reward_runtime_stats, rule_stats)
        # hack：closed_qa rubric 同样不检查 answer_rule_check
        rule_score_dict['answer_rule_check'] = 0.0

        planner_rule_check = rule_score_dict.get('planner_rule_check', 0)
        think_rule_check = rule_score_dict.get('think_rule_check', 0)
        reward_failed = int(reward_runtime_stats.get('reward_error_count', 0)) > 0

        format_failed = planner_rule_check == -1 or think_rule_check == -1

        if format_failed or reward_failed:
            logger.warning(
                f'[closed_qa_rubric] early exit: format_failed={format_failed} '
                f'(planner_rule_check={planner_rule_check}, think_rule_check={think_rule_check}), '
                f'reward_failed={reward_failed}'
            )
            merged_score_dict = {}
            merged_score_dict.update(rule_score_dict)
            merged_score_dict['score'] = -1.0
            merged_score_dict['format_failed'] = format_failed
            merged_score_dict['tool_call_count'] = count_tool_calls(solution_str)
            merged_score_dict['turn_count'] = get_turn_count(extra_info, solution_str)
            merged_score_dict['tool_call_discount'] = 0.0
            merged_score_dict['turn_discount'] = 0.0
            merged_score_dict['discount'] = 0.0
            merged_score_dict.update(reward_runtime_stats)
            merged_score_dict.update({
                'closed_qa_score': 0,
                'judgment': '格式错误' if format_failed else '评分异常',
                'confidence': '高',
                'reason': 'format check failed'
                if format_failed
                else 'rule_based_judge error, skipped rubric evaluation',
                'rubric_verify_error': 'format_failed' if format_failed else 'reward_failed',
                'rubric_verify_error_type': 'format_failed' if format_failed else 'reward_failed',
            })
            return _canonicalize_score_dict(merged_score_dict)

        # Step 2: rubric LLM Judge 评分
        rubric_score_dict, rubric_stats = await _safe_compute_score(
            'closed_qa_rubric_judge',
            closed_qa_rubric_judge.compute_score(
                solution_str=solution_str,
                ground_truth=ground_truth,
                extra_info=extra_info,
            ),
            traj_id=traj_id,
        )
        _merge_reward_runtime_stats(reward_runtime_stats, rubric_stats)
        closed_qa_score = rubric_score_dict.get('closed_qa_score', 0.0)

        # Step 3: 计算 tool call 统计和 discount
        stats = compute_tool_call_stats(extra_info, solution_str)
        tool_call_discount = stats['tool_call_discount']
        turn_discount = 1
        discount = tool_call_discount * turn_discount

        final_score = closed_qa_score * discount

        logger.warning(
            f'closed_qa_rubric reward: closed_qa_score={closed_qa_score:.4f}, '
            f'must={rubric_score_dict.get("rubric_verify_must_score")}, '
            f'nice={rubric_score_dict.get("rubric_verify_nice_score")}, '
            f'tool_calls={stats["tool_call_count"]}, '
            f'queries(web={stats["max_web_queries"]}, pic={stats["max_picture_queries"]}, '
            f'vid={stats["max_video_queries"]}, total={stats["total_queries"]}), '
            f'turns={stats["turn_count"]}, '
            f'tool_call_discount={tool_call_discount:.4f}, turn_discount={turn_discount:.4f}, '
            f'discount={discount:.4f}, final_score={final_score:.4f}'
        )

        merged_score_dict = {}
        merged_score_dict.update(rule_score_dict)
        merged_score_dict.update(rubric_score_dict)
        merged_score_dict['score'] = float(final_score)
        merged_score_dict['format_failed'] = False
        merged_score_dict.update(stats)
        merged_score_dict['max_tool_calls_per_turn'] = stats['total_queries']
        merged_score_dict['turn_discount'] = float(turn_discount)
        merged_score_dict['discount'] = float(discount)
        merged_score_dict.update(reward_runtime_stats)
        return _canonicalize_score_dict(merged_score_dict)

    # agts_closed_qa 数据使用改进的 reward 逻辑
    if 'agts_closed_qa' in data_source:
        # Step 1: 格式检查
        rule_score_dict, rule_stats = await _safe_compute_score(
            'rule_based_judge',
            rule_based_judge.compute_score(
                solution_str=solution_str,
                ground_truth=ground_truth,
                extra_info=extra_info,
            ),
            traj_id=traj_id,
        )
        _merge_reward_runtime_stats(reward_runtime_stats, rule_stats)
        # hack,closed_qa 不检查 answer_rule_check
        rule_score_dict['answer_rule_check'] = 0.0

        planner_rule_check = rule_score_dict.get('planner_rule_check', 0)
        think_rule_check = rule_score_dict.get('think_rule_check', 0)
        reward_failed = int(reward_runtime_stats.get('reward_error_count', 0)) > 0

        format_failed = planner_rule_check == -1 or think_rule_check == -1

        if format_failed or reward_failed:
            logger.warning(
                f'[closed_qa] early exit: format_failed={format_failed} '
                f'(planner_rule_check={planner_rule_check}, think_rule_check={think_rule_check}), '
                f'reward_failed={reward_failed}'
            )

            merged_score_dict = {}
            merged_score_dict.update(rule_score_dict)
            merged_score_dict['score'] = -1.0
            merged_score_dict['format_failed'] = format_failed
            merged_score_dict['tool_call_count'] = count_tool_calls(solution_str)
            merged_score_dict['turn_count'] = get_turn_count(extra_info, solution_str)
            merged_score_dict['tool_call_discount'] = 0.0
            merged_score_dict['turn_discount'] = 0.0
            merged_score_dict['discount'] = 0.0
            merged_score_dict.update(reward_runtime_stats)

            # 填充其他字段为默认值（保持兼容性）
            merged_score_dict.update({
                'closed_qa_score': 0,
            })
            return _canonicalize_score_dict(merged_score_dict)

        # Step 2: LLM Judge 评分
        closed_qa_score_dict, closed_qa_stats = await _safe_compute_score(
            'closed_qa_judge',
            closed_qa_judge.compute_score(
                solution_str=solution_str,
                ground_truth=ground_truth,
                extra_info=extra_info,
            ),
            traj_id=traj_id,
        )
        _merge_reward_runtime_stats(reward_runtime_stats, closed_qa_stats)
        closed_qa_score = closed_qa_score_dict.get('closed_qa_score', 0)

        # Step 3: 计算 tool call 统计和 discount
        stats = compute_tool_call_stats(extra_info, solution_str)
        tool_call_discount = stats['tool_call_discount']

        # turn discount (暂时禁用)
        turn_discount = 1
        discount = tool_call_discount * turn_discount

        # 最终分数
        final_score = closed_qa_score * discount

        logger.warning(
            f'closed_qa reward: closed_qa_score={closed_qa_score}, tool_calls={stats["tool_call_count"]}, '
            f'queries(web={stats["max_web_queries"]}, pic={stats["max_picture_queries"]}, '
            f'vid={stats["max_video_queries"]}, total={stats["total_queries"]}), '
            f'turns={stats["turn_count"]}, '
            f'tool_call_discount={tool_call_discount:.4f}, turn_discount={turn_discount:.4f}, '
            f'discount={discount:.4f}, final_score={final_score:.4f}'
        )

        # 合并所有字段
        merged_score_dict = {}
        merged_score_dict.update(rule_score_dict)
        merged_score_dict.update(closed_qa_score_dict)
        merged_score_dict['score'] = float(final_score)
        merged_score_dict['format_failed'] = False
        merged_score_dict.update(stats)
        merged_score_dict['max_tool_calls_per_turn'] = stats['total_queries']
        merged_score_dict['turn_discount'] = float(turn_discount)
        merged_score_dict['discount'] = float(discount)
        merged_score_dict.update(reward_runtime_stats)

        return _canonicalize_score_dict(merged_score_dict)

    # hunyuan 数据：直接调用 HY Reward 平台 API，不做规则检查和 discount
    # 对齐 baseline: agts_total_reward.py 中 if 'hunyuan' in data_source 分支
    if 'hunyuan' in data_source or 'reward_system' in data_source or 'zero_reward' in data_source:
        # think 格式检查（轻量规则，不依赖 LLM，所有分支都应执行）
        think_rule_check = rule_based_judge.check_think_format(solution_str, extra_info or {})

        hy_score_dict, hy_stats = await _safe_compute_score(
            'hy_reward',
            hy_reward.compute_score(
                solution_str=solution_str,
                ground_truth=ground_truth,
                extra_info=extra_info,
            ),
            traj_id=traj_id,
        )
        _merge_reward_runtime_stats(reward_runtime_stats, hy_stats)
        hy_score = hy_score_dict.get('hy_score', 0)

        # think 格式违规时，final_score 设为 0
        if think_rule_check == -1:
            final_score = 0.0
            logger.warning(f'[hunyuan] think_rule_check failed: hy_score={hy_score:.4f} -> final_score=0')
        else:
            final_score = float(hy_score)

        merged_score_dict: dict[str, Any] = {}
        merged_score_dict.update(hy_score_dict)
        merged_score_dict['score'] = final_score
        merged_score_dict['think_rule_check'] = think_rule_check
        merged_score_dict.update(reward_runtime_stats)

        return _canonicalize_score_dict(merged_score_dict)

    # 默认逻辑：并行调用多个评分函数
    # 对齐 baseline agts_total_reward.py 的 _async_compute_score（5 个 judge）
    reward_tasks = [
        _safe_compute_score(
            'rule_based_judge',
            rule_based_judge.compute_score(
                solution_str=solution_str,
                ground_truth=ground_truth,
                extra_info=extra_info,
            ),
            traj_id=traj_id,
        ),
        _safe_compute_score(
            'format_judge',
            format_judge.compute_score(
                solution_str=solution_str,
                ground_truth=ground_truth,
                extra_info=extra_info,
            ),
            traj_id=traj_id,
        ),
        _safe_compute_score(
            'relevance_judge',
            relevance_judge.compute_score(
                solution_str=solution_str,
                ground_truth=ground_truth,
                extra_info=extra_info,
            ),
            traj_id=traj_id,
        ),
        _safe_compute_score(
            'logical_consistency_judge',
            logical_consistency_judge.compute_score(
                solution_str=solution_str,
                ground_truth=ground_truth,
                extra_info=extra_info,
            ),
            traj_id=traj_id,
        ),
        # style_judge 暂时禁用：style_score 未参与总分计算，节省 LLM 资源
        # _safe_compute_score(
        #     'style_judge',
        #     style_judge.compute_score(
        #         solution_str=solution_str,
        #         ground_truth=ground_truth,
        #         extra_info=extra_info,
        #     ),
        #     traj_id=traj_id,
        # ),
        _safe_compute_score(
            'pairwise_judge',
            pairwise_judge.compute_score(
                solution_str=solution_str,
                ground_truth=ground_truth,
                extra_info=extra_info,
            ),
            traj_id=traj_id,
        ),
    ]

    score_results = await asyncio.gather(*reward_tasks)

    merged_score_dict = {}
    for score_dict, score_stats in score_results:
        _merge_reward_runtime_stats(reward_runtime_stats, score_stats)
        merged_score_dict.update(score_dict)

    # 计算总分
    format_score = merged_score_dict.get('format_score', 0)
    language_consistency_score = merged_score_dict.get('language_consistency_score', 0)
    logic_score = merged_score_dict.get('logic_score', 0)
    relevance_score = merged_score_dict.get('relevance_score', 0)
    planner_rule_check = merged_score_dict.get('planner_rule_check', 0)
    answer_rule_check = merged_score_dict.get('answer_rule_check', 0)
    picture_rule_check = merged_score_dict.get('picture_rule_check', 0)
    video_rule_check = merged_score_dict.get('video_rule_check', 0)
    search_rule_check = merged_score_dict.get('search_rule_check', 0)
    think_rule_check = merged_score_dict.get('think_rule_check', 0)
    # style_score = merged_score_dict.get('style_score', 0)  # 未参与总分计算
    pairwise_score = merged_score_dict.get('pairwise_score', 0)

    # 计算 tool call 统计和 discount
    stats = compute_tool_call_stats(extra_info, solution_str)
    tool_call_discount = stats['tool_call_discount']

    pos_score = logic_score + relevance_score + 0.5 * pairwise_score
    neg_score = (
        format_score
        + language_consistency_score
        + planner_rule_check
        + answer_rule_check
        + picture_rule_check
        + video_rule_check
        + search_rule_check
        + think_rule_check
    )

    # 应用 discount
    pos_discount_score = pos_score * tool_call_discount
    # 计算 media bonus
    media_bonus, media_details = compute_media_bonus(extra_info, solution_str)
    total_score = pos_discount_score + neg_score + media_bonus
    merged_score_dict['score'] = float(total_score)

    logger.warning(
        f'[default] pos_score={pos_score:.4f}, neg_score={neg_score} '
        f'media_bonus={media_bonus:.4f} '
        f'(think_rule_check={think_rule_check}), '
        f'queries(web={stats["max_web_queries"]}, pic={stats["max_picture_queries"]}, '
        f'vid={stats["max_video_queries"]}, total={stats["total_queries"]}), '
        f'tool_call_discount={tool_call_discount:.4f}, final_score={total_score:.4f}'
    )

    merged_score_dict.update({'hy_score': 0, 'hy_score_json': '{}'})
    merged_score_dict.update(media_details)
    merged_score_dict.update(stats)
    merged_score_dict.update(reward_runtime_stats)

    return _canonicalize_score_dict(merged_score_dict)


def compute_score(
    solution_str: str,
    ground_truth: str = '无参考答案',
    extra_info: dict | None = None,
    data_source: str = '',
) -> dict[str, Any]:
    """
    同步接口的 compute_score，内部使用异步调用。

    Args:
        solution_str: 模型生成的完整回复
        ground_truth: 参考答案
        extra_info: 额外信息
        data_source: 数据来源标识

    Returns:
        包含各项评分的字典
    """
    return asyncio.run(
        compute_score_async(
            solution_str=solution_str,
            ground_truth=ground_truth,
            extra_info=extra_info,
            data_source=data_source,
        )
    )
