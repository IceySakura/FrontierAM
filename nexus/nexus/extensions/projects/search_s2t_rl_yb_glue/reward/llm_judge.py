"""Reward_system GRM judge module for S2T Search RL.

This file keeps the public ``compute_score`` contract used by
``reward/total_reward.py`` (returns ``closed_qa_score`` / ``judgment`` / ...), and
scores the model answer with the **exact yuanbao (agts) three-tier judge prompt**
（``完全匹配 / 部分匹配 / 不匹配`` → ``1.0 / 0.5 / 0.0``）.  Only the *transport* is
swapped: instead of a Taiji/alex-oss OpenAI ``judge`` client, the prompt is sent
through Reward_system（``request_grm_raw_text``）with a json_schema ``override`` that
constrains the GRM to emit the flat ``{judgment, confidence, reason}`` structure the
agts parser expects; the raw completion is parsed client-side.

Answer extraction stays the same as the previous yuanbao-glue implementation:
本项目的 agent 走结构化 OpenAI function-calling，task 侧已从 trajectory 抽出干净的
最终答案（``extra_info['final_answer']``），因此 ``extract_query_solution`` 只需去除
``<result>`` / ``<answer>`` 包裹标签与少量特殊 token，而无需解析各家模型的 tool-call 文本格式。

传输层（``is_grm_router_id`` / ``request_grm_raw_text``）已从 hytuner 训练仓库迁移为
``reward/grm_transport.py`` 里的独立实现，不再依赖 hytuner，详见
``docs/developing/search-s2t-rl-yb-glue-reward-migration.md``。
"""

import json
import logging
import re

from functools import partial
from typing import Any

from nexus.extensions.projects.search_s2t_rl_yb_glue.log_utils import log_trace
from nexus.extensions.projects.search_s2t_rl_yb_glue.reward.agts_prompt import PROMPT_TEMPLATE
from nexus.extensions.projects.search_s2t_rl_yb_glue.reward.grm_transport import is_grm_router_id, request_grm_raw_text


logger = logging.getLogger(__name__)
_log_reward = partial(log_trace, 'REWARD')

_ANSWER_TAG_PATTERNS = (
    re.compile(r'<result>(.*?)</result>', re.DOTALL),
    re.compile(r'<answer>(.*?)</answer>', re.DOTALL),
)
_STRIP_TOKENS = (
    '<result>',
    '</result>',
    '<answer>',
    '</answer>',
    '<｜end▁of▁sentence｜>',
    '<|im_end|>',
    '<|endoftext|>',
    '<|eot|>',
)

# agts 三档判定 → closed_qa_score 的映射（逐字对齐 agts/reward/llm_judge.py 的 judgment2score）。
JUDGMENT2SCORE = {'完全匹配': 1.0, '部分匹配': 0.5, '不匹配': 0.0}
VALID_CONFIDENCE_VALUES = {'高', '中', '低'}

# json_schema：把 GRM 输出约束成 agts 三档的扁平结构。实测这个 RM 在无约束时会退回自带的
# 二元「是/否」嵌套格式；加上该 schema 后才会忠实输出 {judgment, confidence, reason}，使
# 上面的 JUDGMENT2SCORE 能直接命中。
JUDGMENT_JSON_SCHEMA = {
    'type': 'object',
    'properties': {
        'judgment': {'type': 'string', 'enum': ['完全匹配', '部分匹配', '不匹配'], 'description': '三档判定结果'},
        'confidence': {'type': 'string', 'enum': ['高', '中', '低'], 'description': '置信度'},
        'reason': {'type': 'string', 'description': '判断理由'},
    },
    'required': ['judgment', 'confidence', 'reason'],
    'additionalProperties': False,
}

# 发给 GRM 的解码参数 + 输出 schema（对齐 test_grm_connectivity_realprompt.py 的 GRM_OVERRIDE，
# 仅把 schema 换成 agts 三档）。经 reward_system 请求的 extra.override 下发。
JUDGE_OVERRIDE = {
    'top_p': 0.95,
    'top_k': 20,
    'max_tokens': 4000,
    'response_format': {
        'type': 'json_schema',
        'json_schema': {'name': 'closed_qa_judgment', 'schema': JUDGMENT_JSON_SCHEMA},
    },
}

MAX_RETRIES = 3


def extract_query_solution(solution_str: str, extra_info: dict) -> str:
    """从模型响应中提取最终答案。

    与 agts 不同，本项目的 agent 走结构化 function-calling，task 侧已把最后一条
    assistant 文本作为 ``solution_str`` 传入，因此这里只需剥离 ``<result>`` /
    ``<answer>`` 包裹标签及少量特殊 token。若上游给到的是完整响应，也能通过 tag
    正则兜底抽取最后一个被包裹的答案。
    """
    text = solution_str or ''

    # 优先抽取最后一个被 <result>/<answer> 包裹的内容。
    for pattern in _ANSWER_TAG_PATTERNS:
        matches = list(pattern.finditer(text))
        if matches:
            text = matches[-1].group(1)
            break

    for token in _STRIP_TOKENS:
        text = text.replace(token, '')
    return text.strip()


def extract_query_from_prompt(prompt_messages: list) -> str:
    """从 prompt messages 中提取最后一个 user 的问题。"""
    if not prompt_messages:
        return ''

    for msg in reversed(prompt_messages):
        if msg.get('role') == 'user':
            content = msg.get('content', '')
            if isinstance(content, list):
                content = ' '.join(
                    part.get('text', '') for part in content if isinstance(part, dict) and part.get('type') == 'text'
                )
            if '# 用户消息为：' in content:
                parts = content.split('# 用户消息为：')
                if len(parts) > 1:
                    return parts[-1].strip()
            return content
    return ''


def extract_history_from_prompt(prompt_messages: list) -> list:
    """从 prompt messages 中提取对话历史（不包括 system 和最后一轮）。"""
    if not prompt_messages:
        return []

    history = []
    for msg in prompt_messages:
        role = msg.get('role', '')
        if role == 'system':
            continue
        if role in ['user', 'assistant']:
            content = msg.get('content', '')
            if isinstance(content, list):
                content = ' '.join(
                    part.get('text', '') for part in content if isinstance(part, dict) and part.get('type') == 'text'
                )
            if len(content) > 500:
                content = content[:500] + '...'
            history.append({'role': role, 'content': content})

    if history and history[-1].get('role') == 'user':
        history = history[:-1]

    return history


def _fill_prompt_template(history: list, question: str, ground_truth: str, answer: str) -> str:
    """用 agts 原始 PROMPT_TEMPLATE 组装 judge prompt（占位符填充逻辑与 agts 完全一致）。"""
    history_str = json.dumps(history, ensure_ascii=False) if history else '无历史对话'
    prompt = PROMPT_TEMPLATE.replace('${history}', history_str)
    prompt = prompt.replace('${question}', question)
    prompt = prompt.replace('${gold_answer}', ground_truth)
    return prompt.replace('${model_response}', answer)


def _parse_judge_json(raw_text: str) -> dict[str, Any]:
    """解析 GRM 返回的 judge 文本为 dict（去 <think>、优先取 ```json``` 围栏）。

    JSON 解析或 schema 校验失败时抛异常，由 ``compute_score`` 的重试循环处理。
    """
    if not isinstance(raw_text, str):
        raise ValueError(f'GRM judge response must be a string, got {type(raw_text).__name__}')
    text = re.sub(r'<think>.*?</think>', '', raw_text.strip(), flags=re.DOTALL).strip()
    fenced = re.search(r'```json\s*(.*?)\s*```', text, re.DOTALL)
    if fenced:
        text = fenced.group(1).strip()
    data = json.loads(text)
    if not isinstance(data, dict):
        raise ValueError(f'GRM judge JSON must be an object, got {type(data).__name__}')

    judgment = data.get('judgment')
    confidence = data.get('confidence')
    reason = data.get('reason')
    if judgment not in JUDGMENT2SCORE:
        raise ValueError(f'GRM judge judgment is invalid: {judgment!r}')
    if confidence not in VALID_CONFIDENCE_VALUES:
        raise ValueError(f'GRM judge confidence is invalid: {confidence!r}')
    if not isinstance(reason, str):
        raise ValueError(f'GRM judge reason must be a string, got {type(reason).__name__}')
    return data


async def compute_score(
    solution_str: str,
    ground_truth: str = '无参考答案',
    extra_info: dict | None = None,
) -> dict[str, Any]:
    """用 agts 三档 judge prompt 打分，但走 Reward_system 传输（question + history + gold + answer）。"""
    extra_info = extra_info or {}
    traj_id = extra_info.get('traj_id', 'unknown')

    extra_info['model_type'] = extra_info.get('model_type', 'ds')

    answer = extract_query_solution(solution_str, extra_info)

    query = extra_info.get('query', extra_info.get('question', ''))

    prompt_messages = extra_info.get('prompt', extra_info.get('messages', []))

    def has_content(obj):
        if obj is None:
            return False
        try:
            return len(obj) > 0
        except (TypeError, ValueError):
            return False

    if hasattr(prompt_messages, 'tolist'):
        prompt_messages = prompt_messages.tolist()

    if not query and has_content(prompt_messages):
        query = extract_query_from_prompt(prompt_messages)

    history = extra_info.get('history', [])
    if hasattr(history, 'tolist'):
        history = history.tolist()

    if not has_content(history) and has_content(prompt_messages):
        history = extract_history_from_prompt(prompt_messages)

    # 用 agts 原始 PROMPT_TEMPLATE 组装 judge prompt（prompt 一字不改）。
    judge_prompt = _fill_prompt_template(
        history=history if has_content(history) else [],
        question=str(query),
        ground_truth=str(ground_truth),
        answer=answer,
    )

    # Reward_system 凭据 / 路由（task.py 已把这些字段扁平化注入 extra_info）。
    router_id = extra_info.get('reward_system_router_id')
    if not is_grm_router_id(router_id):
        raise ValueError(f'reward_system_router_id {router_id!r} 不会路由到 AudioGRMDS GRM 路径')
    user_token = str(extra_info.get('reward_system_user_token') or '')
    exp_id = extra_info.get('reward_system_exp_id') or 'nexus'
    timeout = int(extra_info.get('timeout') or extra_info.get('request_timeout') or 1800)
    max_retry_attempts = int(extra_info.get('max_retry_attempts') or extra_info.get('max_retries') or MAX_RETRIES)

    # judge prompt 作为单条 user turn 下发（与 realprompt 的调用形态一致）。
    messages = [{'role': 'system', 'content': ''}, {'role': 'user', 'content': judge_prompt}]

    _log_reward(traj_id, 'LLM_JUDGE_START', f'router_id={router_id}')

    last_raw = ''
    last_request_id = ''
    last_parse_error = ''
    for attempt in range(MAX_RETRIES):
        # 传输失败（不可达 / error_code / 空 choices）由 request_grm_raw_text 内部重试后抛 RuntimeError，
        # 直接向上冒泡（不静默兜底）。这里的循环负责「拿到文本但 JSON/schema 不合法」的重试。
        raw_text, request_id = await request_grm_raw_text(
            messages,
            router_id=str(router_id),
            user_token=user_token,
            exp_id=exp_id,
            override=JUDGE_OVERRIDE,
            timeout=timeout,
            max_retries=max_retry_attempts,
        )
        last_raw, last_request_id = raw_text, request_id
        try:
            data = _parse_judge_json(raw_text)
        except (json.JSONDecodeError, ValueError, TypeError) as exc:
            last_parse_error = f'{type(exc).__name__}: {exc}'
            _log_reward(traj_id, 'LLM_JUDGE_PARSE_RETRY', f'attempt={attempt + 1}/{MAX_RETRIES}, err={exc}')
            continue

        judgment = data.get('judgment', '不匹配')
        confidence = data.get('confidence', '低')
        reason = data.get('reason', '')
        closed_qa_score = JUDGMENT2SCORE.get(judgment, 0.0)

        _log_reward(traj_id, 'LLM_JUDGE_DONE', f'judgment={judgment}, score={closed_qa_score}')
        return {
            'closed_qa_score': closed_qa_score,
            'acc': closed_qa_score,
            'judgment': judgment,
            'confidence': confidence,
            'reason': reason,
            'rm_response': raw_text,
            'rm_request_id': request_id,
            'reward_system_router_id': str(router_id),
        }

    raise ValueError(
        f'GRM judge 输出无法解析或不符合 schema，重试 {MAX_RETRIES} 次仍失败，'
        f'request_id={last_request_id}, last_error={last_parse_error}, raw[:200]={last_raw[:200]!r}'
    )
