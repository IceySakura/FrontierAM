"""
Judge 模块公共工具函数。

提供各 judge 模块共享的工具：
- retry_backoff_sleep: 指数退避 + 随机 jitter 的重试等待
- extract_query_solution: 从模型响应中提取最终答案
- extract_query_from_prompt: 从 prompt messages 中提取用户问题
- extract_history_from_prompt: 从 prompt messages 中提取对话历史
"""

import asyncio
import random


_BACKOFF_BASE = 2.0
_BACKOFF_MAX = 60.0
_JITTER_MAX = 6.0


async def retry_backoff_sleep(attempt: int) -> None:
    """指数退避 + 随机 jitter，避免多 worker 同时重试导致惊群。

    delay = min(base * 2^attempt, max) + uniform(0, jitter)

    attempt 0: 2~8s,  1: 4~10s,  2: 8~14s,  3: 16~22s,  4: 32~38s,  5+: 60~66s
    """
    delay = min(_BACKOFF_BASE * (2**attempt), _BACKOFF_MAX) + random.uniform(0, _JITTER_MAX)
    await asyncio.sleep(delay)


def extract_query_solution(solution_str: str, extra_info: dict) -> str:
    """从模型响应中提取最终答案。"""
    model_type = extra_info.get('model_type', 'ds')

    if model_type == 'yb':
        answer = solution_str
        answer = answer.replace('<eos:6124c78e>', '')

        if '</think>' in answer:
            last_think_end = answer.rfind('</think>')
            answer = answer[last_think_end + len('</think>') :]

        if '<tool_calls>' in answer:
            answer = answer[: answer.find('<tool_calls>')]

        answer = answer.strip()
    elif model_type == 'ds':
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
    else:
        last_assistant_idx = solution_str.rfind('<|im_start|>assistant')
        answer = solution_str[last_assistant_idx + len('<|im_start|>assistant') :]

        answer = answer.replace('<|im_end|>', '')
        answer = answer.replace('<|endoftext|>', '')
        answer = answer.strip()

    answer = answer.replace('<answer>', '')
    answer = answer.replace('</answer>', '')
    return answer.strip()


def extract_query_from_prompt(prompt_messages: list) -> str:
    """从 prompt messages 中提取最后一个 user 的问题。"""
    if not prompt_messages:
        return ''

    for msg in reversed(prompt_messages):
        if msg.get('role') == 'user':
            content = msg.get('content', '')
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
            if len(content) > 500:
                content = content[:500] + '...'
            history.append({'role': role, 'content': content})

    if history and history[-1].get('role') == 'user':
        history = history[:-1]

    return history
