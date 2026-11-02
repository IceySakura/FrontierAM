"""五个硬门控（format / safety / length / natural / logic）的 GRM 判定模块。

在搜索项目原有的 agts 三档语义 judge 之外，额外发起一次**独立**的 GRM 请求，
复用 ``gates.py``（从 hytuner 迁移，见该文件顶部 docstring）里定义的五个硬门控：

- format  口语对话-回答格式约束
- safety  基础安全-内容安全
- length  口语对话-回复长度适量
- natural 口语对话-自然表达
- logic   口语对话-逻辑表达

任一门控被 judge 判定为 ``passed=false`` ⇒ 由 ``total_reward`` 将整条样本归零。
门控是纯 pass/fail 开关，不参与语义分的加权。

为什么必须独立请求（而不是塞进现有 judge 的 checklist）：搜索项目的 ``llm_judge``
走 agts 三档 judge（schema 为扁平的 ``{judgment, confidence, reason}``，本身没有
checklist 概念），而五个门控走 rubric checklist judge（schema 为
``RULE_RESULTS_JSON_SCHEMA``，逐条 ``passed`` 判定）。两套 prompt 与 response schema
完全不同，因此门控只能作为一次单独的 GRM 请求接入。传输层与门控素材均直接复用，
凭据 / 路由的读取方式与 ``llm_judge.compute_score`` 保持一字一致。

本模块及其依赖的 ``gates.py`` / ``grm_rubrics.py`` / ``grm_transport.py`` 均已从
hytuner 训练仓库迁移为 nexus 内独立实现，不再需要 hytuner / recipe 依赖，详见
``docs/developing/search-s2t-rl-yb-glue-reward-migration.md``。
"""

import logging
import re

from functools import partial
from typing import Any

from nexus.extensions.projects.search_s2t_rl_yb_glue.log_utils import log_trace
from nexus.extensions.projects.search_s2t_rl_yb_glue.reward import grm_rubrics, grm_transport
from nexus.extensions.projects.search_s2t_rl_yb_glue.reward.gates import GATES, _append_gate_rules, _find_gate_result

from . import llm_judge


logger = logging.getLogger(__name__)
_log_reward = partial(log_trace, 'REWARD')

MAX_RETRIES = 3


def _gate_override(rule_results_json_schema: dict) -> dict:
    """发给 GRM 的解码参数 + 输出 schema。

    仿 ``llm_judge.JUDGE_OVERRIDE``，仅把 response_format 换成门控用的 rubric
    schema（逐条 passed 判定）。schema 来自 ``grm_rubrics``。
    """
    return {
        'top_p': 0.95,
        'top_k': 20,
        'max_tokens': 8000,
        'response_format': {
            'type': 'json_schema',
            'json_schema': {'name': 'rubric_rule_results', 'schema': rule_results_json_schema},
        },
    }


def _build_gate_prompt(solution_str: str, ground_truth: str, extra_info: dict) -> tuple[str, dict[str, int]]:
    """组装门控 judge prompt，返回 ``(prompt_text, gate_ids)``。

    答案抽取、query/history 抽取逻辑与 ``llm_judge`` 完全复用（同一份 solution/prompt
    经过相同的清洗），确保门控看到的模型回复与语义 judge 一致。
    """
    answer = llm_judge.extract_query_solution(solution_str, extra_info)

    query = extra_info.get('query', extra_info.get('question', ''))
    prompt_messages = extra_info.get('prompt', extra_info.get('messages', []))
    if hasattr(prompt_messages, 'tolist'):
        prompt_messages = prompt_messages.tolist()

    def has_content(obj):
        if obj is None:
            return False
        try:
            return len(obj) > 0
        except (TypeError, ValueError):
            return False

    if not query and has_content(prompt_messages):
        query = llm_judge.extract_query_from_prompt(prompt_messages)

    history = extra_info.get('history', [])
    if hasattr(history, 'tolist'):
        history = history.tolist()
    if not has_content(history) and has_content(prompt_messages):
        history = llm_judge.extract_history_from_prompt(prompt_messages)

    # rubric ``build_reward_prompt`` 消费 messages 形式的 original_text_prompt（system +
    # history + 当前 user 轮）。用抽取到的 history + 当前 query 重建，避免把 assistant
    # 回复泄漏进上下文。
    original_text_prompt = [{'role': 'system', 'content': ''}]
    if has_content(history):
        original_text_prompt.extend(history)
    original_text_prompt.append({'role': 'user', 'content': str(query)})

    # 空基线 checklist，只追加五个门控。gate_ids 给出每个门控 1-based 位置，供解析兜底匹配。
    checklist, gate_ids = _append_gate_rules([])
    gate_prompt = grm_rubrics.build_reward_prompt(
        original_text_prompt=original_text_prompt,
        model_response=answer,
        reference_response='',
        checklist=checklist,
        rubrics=None,
    )
    return gate_prompt, gate_ids


def _judge_gates(rule_results: list[dict], gate_ids: dict[str, int]) -> dict[str, Any]:
    """对解析出的 rule_results 逐门控判定 PASS / FAIL / ABSENT。

    ABSENT（judge 未返回该门控）视为 PASS —— 与 audio_grm 门控语义一致：门控是只在
    明确违规时才拦截的 pass/fail 开关，缺席不拦。
    """
    gate_status: dict[str, str] = {}
    failed_gates: list[str] = []
    for label, dimension, _item in GATES:
        gr = _find_gate_result(rule_results, dimension, gate_ids[dimension])
        if gr is None:
            gate_status[label] = 'ABSENT'
        elif grm_rubrics._rule_passed(gr):
            gate_status[label] = 'PASS'
        else:
            gate_status[label] = 'FAIL'
            failed_gates.append(label)
    return {
        'gate_passed': len(failed_gates) == 0,
        'failed_gates': failed_gates,
        'gate_status': gate_status,
    }


async def evaluate_gates(
    solution_str: str,
    ground_truth: str = '无参考答案',
    extra_info: dict | None = None,
) -> dict[str, Any]:
    """对模型回复运行五个硬门控，返回门控结论。

    Returns:
        dict，含 ``gate_passed``（bool）、``failed_gates``（list[str]）、
        ``gate_status``（{label: PASS/FAIL/ABSENT}）、``gate_rm_request_id``。

    重试后仍无法解析时 **fail-open**（``gate_passed=True``）：门控只用于拦截明确违规，
    不应因 judge 抖动 / 解析失败误杀正常样本。
    """
    extra_info = extra_info or {}
    traj_id = extra_info.get('traj_id', 'unknown')

    gate_prompt, gate_ids = _build_gate_prompt(solution_str, ground_truth, extra_info)

    # Reward_system 凭据 / 路由：与 llm_judge.compute_score 一字一致地从 extra_info 读取。
    router_id = extra_info.get('reward_system_router_id')
    if not grm_transport.is_grm_router_id(router_id):
        raise ValueError(f'reward_system_router_id {router_id!r} 不会路由到 AudioGRMDS GRM 路径')
    user_token = str(extra_info.get('reward_system_user_token') or '')
    exp_id = extra_info.get('reward_system_exp_id') or 'nexus'
    timeout = int(extra_info.get('timeout') or extra_info.get('request_timeout') or 1800)
    max_retry_attempts = int(extra_info.get('max_retry_attempts') or extra_info.get('max_retries') or MAX_RETRIES)

    messages = [{'role': 'system', 'content': ''}, {'role': 'user', 'content': gate_prompt}]

    _log_reward(traj_id, 'GATE_JUDGE_START', f'router_id={router_id}')

    last_raw = ''
    last_request_id = ''
    try:
        for attempt in range(MAX_RETRIES):
            # 传输失败由 request_grm_raw_text 内部重试后抛 RuntimeError 冒泡；这里的循环只负责
            # 「拿到文本但解析不出 rule_results」的重试。
            raw_text, request_id = await grm_transport.request_grm_raw_text(
                messages,
                router_id=str(router_id),
                user_token=user_token,
                exp_id=exp_id,
                override=_gate_override(grm_rubrics.RULE_RESULTS_JSON_SCHEMA),
                timeout=timeout,
                max_retries=max_retry_attempts,
            )
            last_raw, last_request_id = raw_text, request_id

            rule_results = grm_rubrics.parse_scorer_output(_strip_think(raw_text))
            if rule_results is None:
                _log_reward(traj_id, 'GATE_JUDGE_PARSE_RETRY', f'attempt={attempt + 1}/{MAX_RETRIES}')
                continue

            verdict = _judge_gates(rule_results, gate_ids)
            verdict['gate_rm_request_id'] = request_id
            gate_status_str = ', '.join(f'{lbl}={st}' for lbl, st in verdict['gate_status'].items())
            if verdict['gate_passed']:
                _log_reward(traj_id, 'GATE_JUDGE_DONE', f'gates=({gate_status_str})')
            else:
                _log_reward(
                    traj_id,
                    'GATE_JUDGE_FAILED',
                    f'failed=[{", ".join(verdict["failed_gates"])}] -> reward 0.0, gates=({gate_status_str})',
                )
            return verdict
    except RuntimeError as exc:
        # 传输连续失败：fail-open 放行。
        logger.warning(f'gate judge 传输连续失败，fail-open 放行：{exc}')
        return _fail_open(last_request_id, traj_id)
    except Exception as exc:
        # 任何未预期异常（如服务端畸形响应导致的解析崩溃）同样 fail-open：门控只用于拦截
        # 明确违规，绝不能因 judge 侧的意外错误让异常冒泡拖垮整个训练。
        logger.warning(f'gate judge 未预期异常，fail-open 放行：{type(exc).__name__}: {exc}')
        return _fail_open(last_request_id, traj_id)

    # fail-open：解析连续失败时不拦截，记录后放行。
    logger.warning(
        f'gate judge 输出无法解析 rule_results，重试 {MAX_RETRIES} 次仍失败，fail-open 放行；'
        f'request_id={last_request_id}, raw[:200]={last_raw[:200]!r}'
    )
    return _fail_open(last_request_id, traj_id)


def _fail_open(request_id: str, traj_id: str = 'unknown') -> dict[str, Any]:
    """fail-open 放行结果：门控全部视为 ABSENT，gate_passed=True。"""
    _log_reward(traj_id, 'GATE_JUDGE_FAILOPEN', f'request_id={request_id}')
    return {
        'gate_passed': True,
        'failed_gates': [],
        'gate_status': {label: 'ABSENT' for label, _dim, _item in GATES},
        'gate_rm_request_id': request_id,
    }


def _strip_think(raw_text: str) -> str:
    """去掉 <think>...</think>，与 llm_judge._parse_judge_json 的清洗保持一致。"""
    if not isinstance(raw_text, str):
        return ''
    return re.sub(r'<think>.*?</think>', '', raw_text.strip(), flags=re.DOTALL).strip()
