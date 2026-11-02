"""
工具调用统计与折扣计算。

提供 tool call 数量统计、query 数量统计和 discount 计算等公共工具函数。
"""

import logging
import re

from typing import Any


logger = logging.getLogger(__name__)


def count_tool_calls(solution_str: str, tool_name: str | None = None) -> int:
    """
    统计 tool call 的数量，兼容 DeepSeek 和 Yb 两种协议。

    Args:
        solution_str: 模型生成的完整回复
        tool_name: 如果指定，只统计该工具的调用次数；否则统计所有 tool call

    Returns:
        tool call 的数量
    """
    if tool_name is None:
        ds_count = solution_str.count('<｜tool▁call▁begin｜>')
        yb_count = solution_str.count('<tool_call>')
        return ds_count + yb_count
    ds_pattern = f'<｜tool▁call▁begin｜>{tool_name}<｜tool▁sep｜>'
    yb_pattern = f'<tool_call>{tool_name}<tool_sep>'
    return solution_str.count(ds_pattern) + solution_str.count(yb_pattern)


def get_max_queries_per_turn(extra_info: dict, solution_str: str) -> dict[str, int]:
    """
    获取单轮中各类型 search 的最大 query 数量。

    对齐 baseline agts_total_reward.py 的 _get_max_queries_per_turn。
    tool call 格式中，多个 query 用 ｜｜ 分隔传给 search API。

    Args:
        extra_info: 包含 tool_call_list 的字典
        solution_str: 模型生成的完整回复（备用）

    Returns:
        dict: 各类型 search 的最大 query 数量
    """

    def _count_queries_in_text(text: str) -> dict[str, int]:
        result = {'web_search': 0, 'picture_search': 0, 'video_search': 0}
        for search_type in result:
            if search_type in text:
                # DS 协议: JSON 格式 "query": "q1｜｜q2"
                query_pattern = rf'{search_type}[^{{]*\{{\s*"query"\s*:\s*"([^"]*)"'
                matches = re.findall(query_pattern, text)
                for query_content in matches:
                    query_count = query_content.count('｜｜') + 1
                    result[search_type] = max(result[search_type], query_count)

                # Yb 协议: list 格式 "query": ["q1", "q2"] 或 <arg_value>["q1", "q2"]</arg_value>
                list_pattern = rf'{search_type}.*?"query"\s*:\s*\[(.*?)\]'
                list_matches = re.findall(list_pattern, text, re.DOTALL)
                for list_content in list_matches:
                    items = [s.strip() for s in list_content.split(',') if s.strip().strip('"').strip("'")]
                    result[search_type] = max(result[search_type], len(items))

                # Yb 协议: <arg_key>query</arg_key><arg_value>[...]</arg_value>
                xml_pattern = r'<arg_key>\s*query\s*</arg_key>\s*<arg_value>\s*\[(.*?)\]\s*</arg_value>'
                xml_matches = re.findall(xml_pattern, text, re.DOTALL)
                for xml_content in xml_matches:
                    items = [s.strip() for s in xml_content.split(',') if s.strip().strip('"').strip("'")]
                    result[search_type] = max(result[search_type], len(items))
        return result

    max_queries: dict[str, int] = {'web_search': 0, 'picture_search': 0, 'video_search': 0}

    tool_call_list = extra_info.get('tool_call_list')

    try:
        if tool_call_list is not None and len(tool_call_list) > 0:
            for turn_message in tool_call_list:
                if isinstance(turn_message, str):
                    turn_queries = _count_queries_in_text(turn_message)
                    for search_type in max_queries:
                        max_queries[search_type] = max(max_queries[search_type], turn_queries[search_type])

            if any(v > 0 for v in max_queries.values()):
                max_queries['max_total'] = max(max_queries.values())
                return max_queries
    except (TypeError, ValueError) as e:
        logger.warning(f'tool_call_list parsing error: {e}, type={type(tool_call_list)}')

    # Fallback: 直接从 solution_str 中统计
    fallback_queries = _count_queries_in_text(solution_str)
    for search_type in max_queries:
        max_queries[search_type] = max(max_queries[search_type], fallback_queries[search_type])

    max_queries['max_total'] = max(max_queries.values()) if any(v > 0 for v in max_queries.values()) else 0

    return max_queries


def get_turn_count(extra_info: dict, solution_str: str) -> int:
    """
    获取轮数。

    对齐 baseline agts_total_reward.py 的 _get_turn_count。
    优先使用 extra_info 中的 num_turns（来自 rollout 的实际统计），
    如果不存在，则通过 tool output 的数量推断。

    Args:
        extra_info: 包含 num_turns 的字典
        solution_str: 模型生成的完整回复（备用）

    Returns:
        轮数（assistant turns）
    """
    num_turns = extra_info.get('num_turns')
    if num_turns is not None:
        # num_turns = user_turns + assistant_turns + 1
        # assistant_turns ≈ num_turns // 2
        return max(int(num_turns) // 2, 1)

    # Fallback: 统计 tool output 的数量作为轮数（兼容两种协议）
    ds_turn_count = solution_str.count('<｜tool▁output▁begin｜>')
    yb_turn_count = solution_str.count('</tool_response>')
    turn_count = ds_turn_count + yb_turn_count
    return max(turn_count, 1)


def _compute_single_tool_penalty(count: int, base_limit: int = 3, soft_limit: int = 5) -> float:
    """
    计算单个工具的惩罚系数。

    对齐 baseline agts_total_reward.py 的 _compute_single_tool_penalty。

    Args:
        count: 该工具的调用次数
        base_limit: 免费额度（默认3）
        soft_limit: 轻度惩罚上限（默认5）

    Returns:
        float: 惩罚系数
    """
    if count <= base_limit:
        return 1.0
    if count <= soft_limit:
        # 超过 base_limit 但不超过 soft_limit，用 0.96 惩罚
        return 0.96 ** (count - base_limit)
    # 超过 soft_limit，用 0.8 惩罚
    return 0.8 ** (count - soft_limit)


def compute_tool_call_stats(
    extra_info: dict,
    solution_str: str,
    max_query_limit: int = 5,
) -> dict[str, Any]:
    """
    计算 tool call 统计信息和 discount。

    对齐 baseline agts_total_reward.py 的 _compute_tool_call_stats。
    分工具独立计算惩罚后相乘。

    Args:
        extra_info: 包含 tool_call_list 的字典
        solution_str: 模型生成的完整回复
        max_query_limit: query 数量阈值（未使用，保留兼容）

    Returns:
        dict: 包含统计信息和 discount 的字典
    """
    tool_call_count = count_tool_calls(solution_str)
    turn_count = get_turn_count(extra_info, solution_str)

    max_queries = get_max_queries_per_turn(extra_info, solution_str)
    max_web_queries = max_queries.get('web_search', 0)
    max_picture_queries = max_queries.get('picture_search', 0)
    max_video_queries = max_queries.get('video_search', 0)

    total_queries = max_web_queries + max_picture_queries + max_video_queries

    # 分工具计算惩罚，然后相乘
    web_penalty = _compute_single_tool_penalty(max_web_queries)
    picture_penalty = _compute_single_tool_penalty(max_picture_queries)
    video_penalty = _compute_single_tool_penalty(max_video_queries)

    tool_call_discount = web_penalty * picture_penalty * video_penalty

    return {
        'tool_call_count': tool_call_count,
        'turn_count': turn_count,
        'total_queries': total_queries,
        'max_web_queries': max_web_queries,
        'max_picture_queries': max_picture_queries,
        'max_video_queries': max_video_queries,
        'tool_call_discount': tool_call_discount,
    }


def compute_media_bonus(
    extra_info: dict | None,
    solution_str: str,
) -> tuple[float, dict[str, Any]]:
    """
    计算 media 正向奖励 bonus（Step 1：只加正奖励，不改惩罚）。

    规则：
    - 图片分类=搜图 且调用了 picture_search → +0.20
    - 视频等级=strong 且调用了 video_search   → +0.20
    - 其它情况                                → 0.0

    不解析 <media_info>，仅判定工具是否被调用。
    "调工具但漏嵌 media_info" 的情况会被现有 picture_rule_check / video_rule_check
    硬门控归零，bonus 不会生效。

    Args:
        extra_info: 额外信息字典（含 图片等级 / 视频等级）
        solution_str: 模型生成的完整回复

    Returns:
        (media_bonus, details_dict)
    """
    extra_info = extra_info or {}
    (extra_info.get('图片等级', '') or '').strip()
    video_level = (extra_info.get('视频等级', '') or '').strip()
    picture_class = (extra_info.get('图片分类', '') or '').strip()

    picture_called = count_tool_calls(solution_str, 'picture_search') > 0
    video_called = count_tool_calls(solution_str, 'video_search') > 0

    picture_bonus = 0.2 if picture_called and picture_class == '搜图' else 0.0

    video_bonus = 0.2 if video_called and video_level == 'strong' else 0.0

    media_bonus = picture_bonus + video_bonus  # ∈ [0, 0.40]

    details = {
        'picture_level': picture_class,
        'video_level': video_level,
        'picture_called': picture_called,
        'video_called': video_called,
        'picture_media_bonus': float(picture_bonus),
        'video_media_bonus': float(video_bonus),
        'media_bonus': float(media_bonus),
    }
    return float(media_bonus), details
