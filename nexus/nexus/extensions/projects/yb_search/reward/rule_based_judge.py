"""
基于规则的格式检查模块。

检查 DeepSeek/Qwen 格式的 Agent 输出是否符合规范：
- tool call 格式是否正确
- 答案格式是否正确
- media 引用是否合法
"""

import json
import logging
import re

from typing import Any


logger = logging.getLogger(__name__)


# 支持的工具定义
# v1.4: timeliness 改为可选参数，因为新版本 tool call 可能不包含 timeliness
TOOLS = {
    'web_search': {'required': ['query'], 'optional': ['timeliness']},
    'picture_search': {'required': ['query'], 'optional': ['timeliness']},
    'video_search': {'required': ['query'], 'optional': ['timeliness']},
}

NO_SEARCH_QUESTIONS: set[str] = set()


def get_citations(text: str) -> list[int]:
    """提取文本中的 [citation:xx] 标记。"""
    citations = re.findall(r'\[citation:(\d+)\]', text)
    return [int(c) for c in citations]


def answer_check(answer: str, extra_info: dict) -> dict[str, int]:
    """
    检查答案格式。

    Args:
        answer: 模型生成的答案部分
        extra_info: 额外信息（包含 image_len, video_len 等）

    Returns:
        包含各项检查结果的字典
    """
    image_len = extra_info.get('image_len', 0)
    video_len = extra_info.get('video_len', 0)
    image_tag = extra_info.get('图片等级', '')
    image_class = extra_info.get('图片分类', '')
    video_tag = extra_info.get('视频等级', '')

    answer_check_dict = {
        'answer_rule_check': 0,
        'planner_rule_check': 0,
        'search_rule_check': 0,
        'picture_rule_check': 0,
        'video_rule_check': 0,
    }

    if not answer.startswith('<answer>') or '</answer>' not in answer:
        answer_check_dict['answer_rule_check'] = -1

    answer_pattern = r'<answer>(.*?)</answer>'
    answer_match = re.search(answer_pattern, answer, re.DOTALL)

    answer_content = answer if not answer_match else answer_match.group(1)

    media_pattern = r'<media_info>(.*?)</media_info>'
    media_matches = list(re.finditer(media_pattern, answer_content, re.DOTALL))

    image_media = None
    video_media = None

    for media_match in media_matches:
        media_json_str = media_match.group(1).strip()
        media_start = media_match.start()
        media_end = media_match.end()

        # 验证 JSON 格式
        try:
            media_data = json.loads(media_json_str)
        except Exception:
            answer_check_dict['answer_rule_check'] = -1
            continue

        # v1.4 新格式: list of {"type": "image_only"/"image"/"video", "id": int, "description": ""}
        if isinstance(media_data, list):
            if len(media_data) == 0:
                answer_check_dict['answer_rule_check'] = -1
                continue

            image_only_ids = []  # 搜图专用（纯图无文字）
            image_ids = []  # 有图有文字
            video_ids = []
            for item in media_data:
                if not isinstance(item, dict):
                    answer_check_dict['answer_rule_check'] = -1
                    continue
                item_type = str(item.get('type', '')).lower()
                item_id = item.get('id')

                if item_type not in ['image_only', 'image', 'video']:
                    answer_check_dict['answer_rule_check'] = -1
                    continue

                if not isinstance(item_id, int):
                    answer_check_dict['answer_rule_check'] = -1
                    continue

                if item_type == 'image_only':
                    image_only_ids.append(item_id)
                elif item_type == 'image':
                    image_ids.append(item_id)
                elif item_type == 'video':
                    video_ids.append(item_id)

            all_image_ids = image_only_ids + image_ids

            # image_only: 纯图无文字，仅搜图场景可用
            if image_only_ids:
                if image_tag == 'none':
                    answer_check_dict['picture_rule_check'] = -1
                if image_class != '搜图':
                    answer_check_dict['picture_rule_check'] = -1

            # image: 有图有文字，非搜图场景也可用，但 image_tag=none 时不能出
            if image_ids and image_tag == 'none':
                answer_check_dict['picture_rule_check'] = -1

            # 搜图场景必须用 image_only，用了 image 类型等于格式错误
            if image_class == '搜图' and all_image_ids and not image_only_ids:
                answer_check_dict['picture_rule_check'] = -1

            if all_image_ids:
                if image_media is not None:
                    answer_check_dict['answer_rule_check'] = -1
                image_media = {'start': media_start, 'end': media_end}
                for mid in all_image_ids:
                    if mid > image_len or mid < 1:
                        answer_check_dict['answer_rule_check'] = -1

            if video_ids:
                if video_tag == 'none':
                    answer_check_dict['video_rule_check'] = -1
                if video_media is not None:
                    answer_check_dict['answer_rule_check'] = -1
                video_media = {'start': media_start, 'end': media_end}
                for mid in video_ids:
                    if mid > video_len or mid < 1:
                        answer_check_dict['answer_rule_check'] = -1

        # 旧格式: {"layout": "matrix/horizontal", "type": "image/video", "ids": [...]}
        elif isinstance(media_data, dict):
            # 检查必需的 key
            required_keys = ['layout', 'type', 'ids']
            for key in required_keys:
                if key not in media_data:
                    answer_check_dict['answer_rule_check'] = -1

            id_ = media_data.get('ids', '')
            if not isinstance(id_, list):
                answer_check_dict['answer_rule_check'] = -1

            if isinstance(id_, list) and len(id_) == 0:
                answer_check_dict['answer_rule_check'] = -1

            # 判断媒体类型
            media_type = str(media_data.get('type', '')).lower()

            if media_type not in ['image', 'video']:
                answer_check_dict['answer_rule_check'] = -1

            if media_type == 'image':
                if image_tag == 'none':
                    answer_check_dict['picture_rule_check'] = -1

                # v1.4: layout=matrix 规则
                layout = media_data.get('layout', '')
                if layout == 'matrix' and image_class != '搜图':
                    answer_check_dict['picture_rule_check'] = -1
                if image_class == '搜图' and layout != 'matrix':
                    answer_check_dict['picture_rule_check'] = -1

                if image_media is not None:
                    answer_check_dict['answer_rule_check'] = -1
                image_media = {
                    'start': media_start,
                    'end': media_end,
                }
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
                video_media = {
                    'start': media_start,
                    'end': media_end,
                }
                if 'ids' in media_data and isinstance(media_data['ids'], list):
                    for media_id in media_data['ids']:
                        if not isinstance(media_id, int):
                            answer_check_dict['answer_rule_check'] = -1
                            continue
                        if media_id > video_len or media_id < 1:
                            answer_check_dict['answer_rule_check'] = -1
        else:
            answer_check_dict['answer_rule_check'] = -1

    if image_media:
        content_before_image = answer_content[: image_media['start']].strip()
        if content_before_image:
            answer_check_dict['answer_rule_check'] = -1
    else:
        # v1.4: 从 image_tag=='strong' 改为 image_class=='搜图'
        if image_class == '搜图':
            answer_check_dict['picture_rule_check'] = -1

    if video_media:
        content_after_video = answer_content[video_media['end'] :].strip()
        if content_after_video:
            answer_check_dict['answer_rule_check'] = -1
    else:
        if video_tag == 'strong':
            answer_check_dict['video_rule_check'] = -1

    return answer_check_dict


def check_ds_format(solution_str: str, extra_info: dict) -> dict[str, int]:
    """
    检查 DeepSeek 格式是否正确。

    Args:
        solution_str: 模型生成的完整响应
        extra_info: 额外信息

    Returns:
        包含各项检查结果的字典
    """
    # print("=========rule_check:",json.dumps({"solution_str":solution_str,"extra_info":extra_info},ensure_ascii=False))
    # 提取最终答案部分
    answer = solution_str
    if '<｜tool▁output▁end｜>' in answer:
        last_tool_output_idx = solution_str.rfind('<｜tool▁output▁end｜>')
        answer: str = solution_str[last_tool_output_idx + len('<｜tool▁output▁end｜>') :]
    if '<｜Assistant｜></think>' in answer:
        last_assistant_idx = solution_str.rfind('<｜Assistant｜></think>')
        answer = solution_str[last_assistant_idx + len('<｜Assistant｜></think>') :]
    if '<｜hy_Assistant｜><think></think>' in answer:
        last_assistant_idx = solution_str.rfind('<｜hy_Assistant｜><think></think>')
        answer = solution_str[last_assistant_idx + len('<｜hy_Assistant｜><think></think>') :]
    if '<think></think>' in answer:
        last_assistant_idx = solution_str.rfind('<think></think>')
        answer = solution_str[last_assistant_idx + len('<think></think>') :]

    answer = answer.replace('<｜end▁of▁sentence｜>', '')
    answer = answer.replace('<eos:6124c78e>', '')
    answer = answer.strip()

    score_dict = answer_check(answer, extra_info)
    web_len = extra_info.get('web_len', 0)
    is_search = extra_info.get('是否调用文本搜', '')
    image_tag = extra_info.get('图片等级', '')
    image_class = extra_info.get('图片分类', '')
    video_tag = extra_info.get('视频等级', '')
    # print(f"==========={web_len=}\t{is_search=}\t{image_tag=}\t{image_class=}\t{video_tag=}")

    if is_search == '否' and web_len > 0:
        score_dict['search_rule_check'] = -1

    if is_search == '是' and web_len < 1:
        score_dict['search_rule_check'] = -1

    if web_len == 0 and '[citation:' in answer:
        score_dict['answer_rule_check'] = -1

    cur_citations = get_citations(answer)
    for citation in cur_citations:
        if int(citation) < 1 or int(citation) > web_len:
            score_dict['answer_rule_check'] = -1

    # 检查答案中是否包含工具调用标记
    tool_markers = ['<｜tool▁calls▁begin｜>', '<｜tool▁calls▁end｜>', '<｜tool▁call▁begin｜>', '<｜tool▁call▁end｜>']
    if any(marker in answer for marker in tool_markers):
        score_dict['answer_rule_check'] = -1
        score_dict['planner_rule_check'] = -1

    # 检查不完整的 tool calls
    has_tool_calls_begin = '<｜tool▁calls▁begin｜>' in solution_str
    has_tool_calls_end = '<｜tool▁calls▁end｜>' in solution_str
    if has_tool_calls_begin and not has_tool_calls_end:
        score_dict['planner_rule_check'] = -1

    # 检查不完整的单个 tool call
    tool_call_begin_count = solution_str.count('<｜tool▁call▁begin｜>')
    tool_call_end_count = solution_str.count('<｜tool▁call▁end｜>')
    if tool_call_begin_count > tool_call_end_count:
        score_dict['planner_rule_check'] = -1

    # v1.4: 去掉 tool_call 数量硬限制（加了 discount 后不再需要）
    # max_tool_calls = 10
    # if tool_call_begin_count > max_tool_calls:
    #     score_dict['planner_rule_check'] = -1

    # 检查工具调用格式
    tool_call_list = extra_info.get('tool_call_list', [])

    # v1.4: 如果 tool_call_list 为空但 solution_str 中有 tool call，从 solution_str 中提取
    if len(tool_call_list) == 0 and tool_call_begin_count > 0:
        tool_calls_pattern = r'<｜tool▁calls▁begin｜>(.*?)<｜tool▁calls▁end｜>'
        tool_calls_matches = re.findall(tool_calls_pattern, solution_str, re.DOTALL)
        if tool_calls_matches:
            tool_call_list = tool_calls_matches

    has_image_fc = False
    has_video_fc = False

    for tool_call_str in tool_call_list:
        pattern = r'<｜tool▁call▁begin｜>(.*?)<｜tool▁sep｜>(.*?)<｜tool▁call▁end｜>'
        matches = re.findall(pattern, tool_call_str, re.DOTALL)

        if not matches:
            score_dict['planner_rule_check'] = -1
            continue

        for tool_name, args_str in matches:
            tool_name = tool_name.strip()

            if tool_name not in TOOLS:
                score_dict['planner_rule_check'] = -1
                continue

            if tool_name == 'picture_search':
                has_image_fc = True
            if tool_name == 'video_search':
                has_video_fc = True

            try:
                args = json.loads(args_str)
            except Exception:
                score_dict['planner_rule_check'] = -1
                continue

            for param in TOOLS[tool_name]['required']:
                if param not in args:
                    score_dict['planner_rule_check'] = -1

            if 'timeliness' in args and not isinstance(args['timeliness'], int):
                score_dict['planner_rule_check'] = -1

    # v1.4: image_tag=='strong' 改为 image_class=='搜图'
    if image_class == '搜图' and not has_image_fc:
        score_dict['picture_rule_check'] -= 1
    if image_tag == 'none' and has_image_fc:
        score_dict['picture_rule_check'] -= 1

    if video_tag == 'strong' and not has_video_fc:
        score_dict['video_rule_check'] -= 1
    if video_tag == 'none' and has_video_fc:
        score_dict['video_rule_check'] -= 1

    return score_dict


def check_qwen_format(solution_str: str, extra_info: dict) -> dict[str, int]:
    """
    检查 Qwen 格式是否正确。

    Args:
        solution_str: 模型生成的完整响应
        extra_info: 额外信息

    Returns:
        包含各项检查结果的字典
    """
    last_assistant_idx = solution_str.rfind('<|im_start|>assistant')
    answer = solution_str[last_assistant_idx + len('<|im_start|>assistant') :]

    answer = answer.replace('<|im_end|>', '')
    answer = answer.replace('<|endoftext|>', '')
    answer = answer.strip()

    score_dict = answer_check(answer, extra_info)
    web_len = extra_info.get('web_len', 0)
    is_search = extra_info.get('是否调用文本搜', '')

    if is_search == '否' and web_len > 0:
        score_dict['search_rule_check'] = -1

    if web_len == 0 and '[citation:' in answer:
        score_dict['answer_rule_check'] = -1

    cur_citations = get_citations(answer)
    for citation in cur_citations:
        if int(citation) < 1 or int(citation) > web_len:
            score_dict['answer_rule_check'] = -1

    if '<tool_call>' in answer or '</tool_call>' in answer:
        score_dict['answer_rule_check'] = -1

    tool_call_list = extra_info.get('tool_call_list', [])

    # v1.4: 如果 tool_call_list 为空但 solution_str 中有 tool call，从 solution_str 中提取
    if len(tool_call_list) == 0 and '<tool_call>' in solution_str:
        tool_calls_pattern = r'<tool_call>(.*?)</tool_call>'
        tool_calls_matches = re.findall(tool_calls_pattern, solution_str, re.DOTALL)
        if tool_calls_matches:
            tool_call_list = [f'<tool_call>{m}</tool_call>' for m in tool_calls_matches]

    for tool_call_str in tool_call_list:
        pattern = r'<tool_call>(.*?)</tool_call>'
        matches = re.findall(pattern, tool_call_str, re.DOTALL)

        if not matches:
            score_dict['planner_rule_check'] = -1
            continue

        for call_str in matches:
            call_str = call_str.strip()

            try:
                call_obj = json.loads(call_str)
            except Exception:
                score_dict['planner_rule_check'] = -1
                continue

            if 'name' not in call_obj:
                score_dict['planner_rule_check'] = -1
                continue

            tool_name = call_obj['name']

            if tool_name not in TOOLS:
                score_dict['planner_rule_check'] = -1
                continue

            if 'arguments' not in call_obj:
                score_dict['planner_rule_check'] = -1
                continue

            args = call_obj['arguments']

            if not isinstance(args, dict):
                score_dict['planner_rule_check'] = -1
                continue

            for param in TOOLS[tool_name]['required']:
                if param not in args:
                    score_dict['planner_rule_check'] = -1

            if 'timeliness' in args and not isinstance(args['timeliness'], int):
                score_dict['planner_rule_check'] = -1

    return score_dict


def check_yb_format(solution_str: str, extra_info: dict) -> dict[str, int]:
    """
    检查 Yb 新协议格式是否正确。

    新协议特征：
    - 工具调用: <tool_calls><tool_call>name<tool_sep>args</tool_call></tool_calls>
    - 思考: <think>...</think>
    - 参数: <arg_key>key</arg_key><arg_value>value</arg_value> 或 JSON
    - 回复: <think>思考</think>回复内容（无 <answer> 标签）
    """
    score_dict = {
        'answer_rule_check': 0,
        'planner_rule_check': 0,
        'search_rule_check': 0,
        'picture_rule_check': 0,
        'video_rule_check': 0,
    }

    # 清理 EOS token
    answer = solution_str.replace('<eos:6124c78e>', '').strip()

    # 提取最终答案：最后一个 </think> 之后、<tool_calls> 之前的内容
    # 如果有多轮 tool 交互，取最后一轮 assistant 输出中 </think> 后面的部分
    if '</think>' in answer:
        last_think_end = answer.rfind('</think>')
        answer = answer[last_think_end + len('</think>') :].strip()

    # 如果答案部分还包含 <tool_calls>，说明最终回复还没生成（只有 tool call 没有最终答案）
    if '<tool_calls>' in answer:
        # 最终答案 = tool_calls 之前的部分
        answer = answer[: answer.find('<tool_calls>')].strip()

    # 新协议不要求 <answer> 标签，直接用提取出的文本做 media/citation 检查
    image_len = extra_info.get('image_len', 0)
    video_len = extra_info.get('video_len', 0)
    image_tag = extra_info.get('图片等级', '')
    image_class = extra_info.get('图片分类', '')
    video_tag = extra_info.get('视频等级', '')
    query = extra_info.get('query', extra_info.get('question', ''))

    # media_info 检查（复用 answer_check 中的 media 逻辑）
    media_pattern = r'<media_info>(.*?)</media_info>'
    media_matches = list(re.finditer(media_pattern, answer, re.DOTALL))

    image_media = None
    video_media = None

    for media_match in media_matches:
        media_json_str = media_match.group(1).strip()
        media_start = media_match.start()
        media_end = media_match.end()

        try:
            media_data = json.loads(media_json_str)
        except Exception:
            score_dict['answer_rule_check'] = -1
            continue

        # v1.4 新格式: list of {"type": "image_only"/"image"/"video", "id": int}
        if isinstance(media_data, list):
            if len(media_data) == 0:
                score_dict['answer_rule_check'] = -1
                continue

            image_only_ids = []
            image_ids = []
            video_ids = []
            for item in media_data:
                if not isinstance(item, dict):
                    score_dict['answer_rule_check'] = -1
                    continue
                item_type = str(item.get('type', '')).lower()
                item_id = item.get('id')
                if item_type not in ['image_only', 'image', 'video']:
                    score_dict['answer_rule_check'] = -1
                    continue
                if not isinstance(item_id, int):
                    score_dict['answer_rule_check'] = -1
                    continue
                if item_type == 'image_only':
                    image_only_ids.append(item_id)
                elif item_type == 'image':
                    image_ids.append(item_id)
                elif item_type == 'video':
                    video_ids.append(item_id)

            all_image_ids = image_only_ids + image_ids
            if image_only_ids:
                if image_tag == 'none':
                    score_dict['picture_rule_check'] = -1
                if image_class != '搜图':
                    score_dict['picture_rule_check'] = -1
            if image_ids and image_tag == 'none':
                score_dict['picture_rule_check'] = -1
            if image_class == '搜图' and all_image_ids and not image_only_ids:
                score_dict['picture_rule_check'] = -1
            if all_image_ids:
                if image_media is not None:
                    score_dict['answer_rule_check'] = -1
                image_media = {'start': media_start, 'end': media_end}
                for mid in all_image_ids:
                    if mid > image_len or mid < 1:
                        score_dict['answer_rule_check'] = -1
            if video_ids:
                if video_tag == 'none':
                    score_dict['video_rule_check'] = -1
                if video_media is not None:
                    score_dict['answer_rule_check'] = -1
                video_media = {'start': media_start, 'end': media_end}
                for mid in video_ids:
                    if mid > video_len or mid < 1:
                        score_dict['answer_rule_check'] = -1

        # 旧格式: {"layout": "...", "type": "image/video", "ids": [...]}
        elif isinstance(media_data, dict):
            required_keys = ['layout', 'type', 'ids']
            for key in required_keys:
                if key not in media_data:
                    score_dict['answer_rule_check'] = -1

            id_ = media_data.get('ids', '')
            if not isinstance(id_, list) or len(id_) == 0:
                score_dict['answer_rule_check'] = -1

            media_type = str(media_data.get('type', '')).lower()
            if media_type not in ['image', 'video']:
                score_dict['answer_rule_check'] = -1

            if media_type == 'image':
                if image_tag == 'none':
                    score_dict['picture_rule_check'] = -1
                layout = media_data.get('layout', '')
                if layout == 'matrix' and image_class != '搜图':
                    score_dict['picture_rule_check'] = -1
                if image_class == '搜图' and layout != 'matrix':
                    score_dict['picture_rule_check'] = -1
                if image_media is not None:
                    score_dict['answer_rule_check'] = -1
                image_media = {'start': media_start, 'end': media_end}
                if 'ids' in media_data and isinstance(media_data['ids'], list):
                    for media_id in media_data['ids']:
                        if not isinstance(media_id, int) or media_id > image_len or media_id < 1:
                            score_dict['answer_rule_check'] = -1
            elif media_type == 'video':
                if video_tag == 'none':
                    score_dict['video_rule_check'] = -1
                if video_media is not None:
                    score_dict['answer_rule_check'] = -1
                video_media = {'start': media_start, 'end': media_end}
                if 'ids' in media_data and isinstance(media_data['ids'], list):
                    for media_id in media_data['ids']:
                        if not isinstance(media_id, int) or media_id > video_len or media_id < 1:
                            score_dict['answer_rule_check'] = -1
        else:
            score_dict['answer_rule_check'] = -1

    if image_media:
        content_before_image = answer[: image_media['start']].strip()
        if content_before_image:
            score_dict['answer_rule_check'] = -1
    else:
        if image_class == '搜图':
            score_dict['picture_rule_check'] = -1

    if video_media:
        content_after_video = answer[video_media['end'] :].strip()
        if content_after_video:
            score_dict['answer_rule_check'] = -1
    else:
        if video_tag == 'strong':
            score_dict['video_rule_check'] = -1

    # 搜索规则检查
    web_len = extra_info.get('web_len', 0)
    is_search = extra_info.get('是否调用文本搜', '')

    if is_search == '否' and web_len > 0:
        score_dict['search_rule_check'] = -1
    if is_search == '是' and web_len < 1:
        score_dict['search_rule_check'] = -1

    # citation 检查
    if web_len > 0 and query not in NO_SEARCH_QUESTIONS and '[citation:' not in answer:
        score_dict['answer_rule_check'] = -1
    if web_len == 0 and '[citation:' in answer:
        score_dict['answer_rule_check'] = -1
    for citation in get_citations(answer):
        if int(citation) < 1 or int(citation) > web_len:
            score_dict['answer_rule_check'] = -1

    # 答案中不应包含 tool call 标记
    new_tool_markers = ['<tool_calls>', '</tool_calls>', '<tool_call>', '</tool_call>', '<tool_sep>']
    if any(marker in answer for marker in new_tool_markers):
        score_dict['answer_rule_check'] = -1
        score_dict['planner_rule_check'] = -1

    # 检查不完整的 tool calls
    tc_begin_count = solution_str.count('<tool_call>')
    tc_end_count = solution_str.count('</tool_call>')
    tcs_begin = '<tool_calls>' in solution_str
    tcs_end = '</tool_calls>' in solution_str
    if tcs_begin and not tcs_end:
        score_dict['planner_rule_check'] = -1
    if tc_begin_count > tc_end_count:
        score_dict['planner_rule_check'] = -1

    # 解析和验证 tool call 内容
    tool_call_list = extra_info.get('tool_call_list', [])

    if len(tool_call_list) == 0 and tc_begin_count > 0:
        tcs_pattern = r'<tool_calls>(.*?)</tool_calls>'
        tcs_matches = re.findall(tcs_pattern, solution_str, re.DOTALL)
        if tcs_matches:
            tool_call_list = tcs_matches

    has_image_fc = False
    has_video_fc = False

    tc_pattern = r'<tool_call>(.*?)<tool_sep>(.*?)</tool_call>'
    arg_kv_pattern = r'<arg_key>(.*?)</arg_key>\s*<arg_value>(.*?)</arg_value>'

    for tool_call_str in tool_call_list:
        matches = re.findall(tc_pattern, tool_call_str, re.DOTALL)

        if not matches:
            score_dict['planner_rule_check'] = -1
            continue

        for tool_name, args_str in matches:
            tool_name = tool_name.strip()

            if tool_name not in TOOLS:
                score_dict['planner_rule_check'] = -1
                continue

            if tool_name == 'picture_search':
                has_image_fc = True
            if tool_name == 'video_search':
                has_video_fc = True

            # 尝试 <arg_key>/<arg_value> XML 格式
            kv_pairs = re.findall(arg_kv_pattern, args_str, re.DOTALL)
            if kv_pairs:
                args = {}
                for key, value in kv_pairs:
                    key = key.strip()
                    try:
                        args[key] = json.loads(value.strip())
                    except (json.JSONDecodeError, ValueError):
                        args[key] = value.strip()
            else:
                # 尝试 JSON 格式
                try:
                    args = json.loads(args_str.strip())
                except Exception:
                    score_dict['planner_rule_check'] = -1
                    continue

            if not isinstance(args, dict):
                score_dict['planner_rule_check'] = -1
                continue

            for param in TOOLS[tool_name]['required']:
                if param not in args:
                    score_dict['planner_rule_check'] = -1

            if 'timeliness' in args and not isinstance(args['timeliness'], int):
                score_dict['planner_rule_check'] = -1

    if image_class == '搜图' and not has_image_fc:
        score_dict['picture_rule_check'] -= 1
    if image_tag == 'none' and has_image_fc:
        score_dict['picture_rule_check'] -= 1
    if video_tag == 'strong' and not has_video_fc:
        score_dict['video_rule_check'] -= 1
    if video_tag == 'none' and has_video_fc:
        score_dict['video_rule_check'] -= 1

    return score_dict


def check_think_format(solution_str: str, extra_info: dict) -> int:
    """检查每一轮 assistant 输出的 <think> 格式是否符合 reasoning_effort 协议。

    chat template 行为：
    - no_think: prefill ``<think></think>``，模型不应有思考内容
    - high/low: prefill ``<think>``（开标签），模型需输出 ``</think>`` 关闭

    response_str 通过 ``tokenizer.decode(skip_special_tokens=True)`` 构建：
    - ``<think>``/``</think>`` 是普通 token，会保留
    - ``<｜hy_Assistant｜>`` 是 special token，会被跳过，无法用于分轮
    - 用 ``</tool_responses>`` 分割多轮：之前是第一轮，之后每段是后续轮

    Returns:
        0: 通过 / 无需检查
       -1: 任意一轮违规
    """
    reasoning_effort = extra_info.get('reasoning_effort', '')
    if not reasoning_effort:
        return 0  # 未指定 reasoning_effort，向后兼容，不检查

    # 按 </tool_responses> 分割出每一轮 assistant 输出
    separator = '</tool_responses>'
    parts = solution_str.split(separator)
    # parts[0] = 第一轮（prefill 在 prompt 中，不在 response_str 里）
    # parts[1:] = 第二轮+（每段开头有 chat template 注入的 <think> 前缀）

    for i, part in enumerate(parts):
        if not part.strip():
            continue

        if i == 0:
            # 第一轮：prefill 在 prompt 里，response_str 中不含 <think> 开标签
            if reasoning_effort == 'no_think':
                # prefill 是 <think></think>，模型输出不应有任何 think 标签
                if '<think>' in part or '</think>' in part:
                    return -1
            else:
                # high/low: prefill 是 <think>，模型输出需有且仅有一个 </think> 来关闭
                if part.count('</think>') != 1:
                    return -1
                if '<think>' in part:
                    return -1  # 不应有额外的 <think> 开标签
        else:
            # 第二轮+: chat template 在 response_str 中注入了完整的 <think>...</think> 前缀
            # 每轮应恰好一对 <think></think>，且 <think> 在 </think> 之前
            if part.count('<think>') != 1 or part.count('</think>') != 1:
                return -1
            if part.index('<think>') > part.index('</think>'):
                return -1

    return 0


async def compute_score(
    solution_str: str,
    ground_truth: str = '',
    extra_info: dict | None = None,
) -> dict[str, Any]:
    """
    计算格式检查分数。

    Args:
        solution_str: 模型生成的完整响应
        ground_truth: 参考答案（此处未使用）
        extra_info: 额外信息

    Returns:
        包含各项检查结果的字典
    """
    extra_info = extra_info or {}
    model_type = extra_info.get('model_type', 'ds')

    if model_type == 'ds':
        format_score = check_ds_format(solution_str, extra_info)
    elif model_type == 'yb':
        format_score = check_yb_format(solution_str, extra_info)
    else:
        format_score = check_qwen_format(solution_str, extra_info)

    # think 格式检查（需要 extra_info['reasoning_effort']，向后兼容）
    format_score['think_rule_check'] = check_think_format(solution_str, extra_info)

    print(
        f'[NexusAGTS:RuleJudge] Result: {format_score}, model_type={model_type}, \n'
        f'solution_str: {solution_str[:100]} ... {solution_str[-100:]} \n'
        f'{extra_info.get("tool_call_list")=}',
    )

    return format_score
