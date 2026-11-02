"""基于规则的格式检查模块（对齐 agts 奖励，适配结构化 function-calling 轨迹）。

沿用 agts 奖励的字段与门控语义：

    planner_rule_check == -1  ⇒  格式失败 ⇒ 最终分数 -1.0

与 agts 的差异：本项目的 agent 走结构化 OpenAI function-calling（tool-call 由
rollout 引擎的 tool_parser 解析，最终答案是干净文本），因此不再解析 DeepSeek/Qwen
的内联 tool-call 文本格式，而是：

* 若最终答案里*泄漏*了 tool-call 标记（DeepSeek/Qwen/Hermes 等），判 planner 失败；
* 若工具调用次数超过上限（默认 10，与 agts ``max_tool_calls`` 对齐），判 planner 失败。
* 若模型在执行硬限制关闭后仍请求工具，判 planner 失败。

媒体（图片/视频）相关检查完全沿用 agts 的 ``answer_check``，对纯搜索的 S2T 数据而言
它们通常保持为 0（图片等级/视频等级 = none 且答案无 media_info）。

答案包裹标签（``<result>`` / ``<answer>``）**不再作为要求**：对齐 sp 的元宝语音
人设后，模型输出是纯自然口语文本、不包裹 ``<result>``，因此本模块只在标签存在时用它
抽取答案内容（graceful fallback），缺失标签不再判 ``answer_rule_check = -1``。
"""

import json
import logging
import re

from typing import Any


logger = logging.getLogger(__name__)


DEFAULT_MAX_TOOL_CALLS = 10

# Leftover tool-call syntax that must NOT appear in a clean final answer.
_LEFTOVER_TOOL_MARKERS = (
    '<｜tool▁calls▁begin｜>',
    '<｜tool▁calls▁end｜>',
    '<｜tool▁call▁begin｜>',
    '<｜tool▁call▁end｜>',
    '<｜tool▁sep｜>',
    '<tool_call>',
    '</tool_call>',
)


def get_citations(text: str) -> list[int]:
    """提取文本中的 [citation:xx] 标记。"""
    citations = re.findall(r'\[citation:(\d+)\]', text)
    return [int(c) for c in citations]


def answer_check(answer: str, extra_info: dict) -> dict[str, int]:
    """检查答案格式（媒体 / 包裹标签），沿用 agts 语义。"""
    image_len = extra_info.get('image_len', 0)
    video_len = extra_info.get('video_len', 0)
    image_tag = extra_info.get('图片等级', '')
    video_tag = extra_info.get('视频等级', '')

    answer_check_dict = {
        'answer_rule_check': 0,
        'planner_rule_check': 0,
        'search_rule_check': 0,
        'picture_rule_check': 0,
        'video_rule_check': 0,
    }

    # sp.md 的元宝语音人设输出纯自然文本、不包裹 <result>/<answer>，因此不再把「缺少包裹
    # 标签」判为格式失败。若模型仍输出了包裹标签，则据其抽取答案内容做后续媒体检查
    # （graceful fallback）；否则整段文本即答案内容。
    result_match = re.search(r'<result>(.*?)</result>', answer, re.DOTALL)
    answer_match = re.search(r'<answer>(.*?)</answer>', answer, re.DOTALL)
    if result_match:
        answer_content = result_match.group(1)
    elif answer_match:
        answer_content = answer_match.group(1)
    else:
        answer_content = answer

    media_pattern = r'<media_info>(.*?)</media_info>'
    media_matches = list(re.finditer(media_pattern, answer_content, re.DOTALL))

    image_media = None
    video_media = None

    for media_match in media_matches:
        media_json_str = media_match.group(1).strip()
        media_start = media_match.start()
        media_end = media_match.end()

        try:
            media_data = json.loads(media_json_str)
        except Exception:
            answer_check_dict['answer_rule_check'] = -1
            continue

        required_keys = ['layout', 'type', 'ids']
        for key in required_keys:
            if key not in media_data:
                answer_check_dict['answer_rule_check'] = -1

        id_ = media_data.get('ids', '')
        if not isinstance(id_, list):
            answer_check_dict['answer_rule_check'] = -1
        if isinstance(id_, list) and len(id_) == 0:
            answer_check_dict['answer_rule_check'] = -1

        media_type = str(media_data.get('type', '')).lower()
        if media_type not in ['image', 'video']:
            answer_check_dict['answer_rule_check'] = -1

        if media_type == 'image':
            if image_tag == 'none':
                answer_check_dict['picture_rule_check'] = -1
            if image_media is not None:
                answer_check_dict['answer_rule_check'] = -1
            image_media = {'data': media_data, 'start': media_start, 'end': media_end}
            if 'ids' in media_data and isinstance(media_data['ids'], list):
                for media_id in media_data['ids']:
                    if not isinstance(media_id, int):
                        answer_check_dict['answer_rule_check'] = -1
                        continue
                    if media_id > image_len or media_id < 1:
                        answer_check_dict['answer_rule_check'] = -1
        elif media_type == 'video':
            if video_tag == 'none':
                answer_check_dict['video_rule_check'] = -1
            if video_media is not None:
                answer_check_dict['answer_rule_check'] = -1
            video_media = {'data': media_data, 'start': media_start, 'end': media_end}
            if 'ids' in media_data and isinstance(media_data['ids'], list):
                for media_id in media_data['ids']:
                    if not isinstance(media_id, int):
                        answer_check_dict['answer_rule_check'] = -1
                        continue
                    if media_id > video_len or media_id < 1:
                        answer_check_dict['answer_rule_check'] = -1

    if image_media:
        content_before_image = answer_content[: image_media['start']].strip()
        if content_before_image:
            answer_check_dict['answer_rule_check'] = -1
    else:
        if image_tag == 'strong':
            answer_check_dict['picture_rule_check'] = -1

    if video_media:
        content_after_video = answer_content[video_media['end'] :].strip()
        if content_after_video:
            answer_check_dict['answer_rule_check'] = -1
    else:
        if video_tag == 'strong':
            answer_check_dict['video_rule_check'] = -1

    return answer_check_dict


def check_structured_format(solution_str: str, extra_info: dict) -> dict[str, int]:
    """结构化轨迹的格式检查。

    ``solution_str`` 为 agent 抽取出的干净最终答案（最后一条 assistant 文本）。
    """
    answer = (solution_str or '').strip()
    score_dict = answer_check(answer, extra_info)

    web_len = extra_info.get('web_len', 0)
    is_search = extra_info.get('是否调用文本搜', '')

    # 搜索需求约束（本数据 is_search 通常为空 -> 不触发）。
    if is_search == '否' and web_len > 0:
        score_dict['search_rule_check'] = -1
    if is_search == '是' and web_len < 1:
        score_dict['search_rule_check'] = -1

    # citation 只能在有搜索结果时出现，且下标不能越界。
    if web_len == 0 and '[citation:' in answer:
        score_dict['answer_rule_check'] = -1
    for citation in get_citations(answer):
        if int(citation) > web_len:
            score_dict['answer_rule_check'] = -1

    # planner 门控：最终答案不得泄漏 tool-call 标记。
    if any(marker in answer for marker in _LEFTOVER_TOOL_MARKERS):
        score_dict['answer_rule_check'] = -1
        score_dict['planner_rule_check'] = -1

    # planner 门控：工具调用次数不得超过上限（对齐 agts max_tool_calls）。
    tool_call_count = int(extra_info.get('tool_call_count', 0) or 0)
    max_tool_calls = int(extra_info.get('max_tool_calls', DEFAULT_MAX_TOOL_CALLS) or DEFAULT_MAX_TOOL_CALLS)
    if tool_call_count > max_tool_calls:
        score_dict['planner_rule_check'] = -1

    # planner 门控：存在解析失败的工具调用（agent 侧标记）。
    if int(extra_info.get('malformed_tool_calls', 0) or 0) > 0:
        score_dict['planner_rule_check'] = -1

    # planner 门控：模型在最终回答轮或搜索/observation 预算耗尽后仍请求工具。
    if int(extra_info.get('rejected_tool_calls', 0) or 0) > 0:
        score_dict['planner_rule_check'] = -1

    return score_dict


async def compute_score(
    solution_str: str,
    ground_truth: str = '',
    extra_info: dict | None = None,
) -> dict[str, Any]:
    """计算格式检查分数（返回 planner/answer/search/picture/video rule checks）。"""
    extra_info = extra_info or {}
    format_score = check_structured_format(solution_str, extra_info)

    print(
        f'[NexusS2TYBGlue:RuleJudge] Result: {format_score}, '
        f'tool_call_count={extra_info.get("tool_call_count")}, '
        f'answer_head={str(solution_str)[:80]!r}',
    )

    return format_score
