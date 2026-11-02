"""
Hunyuan Reward 评分模块。

通过外部 Reward 平台 API 计算 hunyuan 数据的奖励分数。
对齐 baseline: verl/utils/reward_score/agts_hy_reward.py
"""

import asyncio
import json
import logging
import os
import uuid

from functools import partial
from typing import Any

from nexus.extensions.projects.yb_search.log_utils import log_trace
from nexus.utils.rate_limiter import get_rate_limiter
from nexus.utils.timeout import async_timeout

from .judge_utils import extract_query_solution


logger = logging.getLogger(__name__)

MAX_RETRIES = int(os.environ.get('HY_JUDGE_MAX_RETRIES', 5))
SINGLE_ATTEMPT_TIMEOUT = int(os.environ.get('HY_JUDGE_TIMEOUT_SINGLE', 400))
RATE_LIMIT = int(os.environ.get('HY_JUDGE_RATE_LIMIT', 4))  # 最大并发数

REWARD_SERVICE_URL = (
    'http://api-reward-platform-api-reward-platform-354ba565.turbotke.production.polaris:10001/reward/RewardService'
)
REWARD_SERVICE_TOKEN = os.environ.get('REWARD_SERVICE_TOKEN')

_log_reward = partial(log_trace, 'REWARD')


async def _call_reward_service(data: dict) -> dict | None:
    """
    异步调用 Reward 平台 API。

    对齐 baseline cal_service() 函数：无单次超时，等待直到返回。
    """
    import aiohttp

    if not REWARD_SERVICE_TOKEN:
        raise ValueError('REWARD_SERVICE_TOKEN is not set')
    headers = {
        'Content-Type': 'application/json',
        'token': REWARD_SERVICE_TOKEN,
    }
    try:
        async with aiohttp.ClientSession() as session:
            async with session.post(
                REWARD_SERVICE_URL,
                data=json.dumps(data),
                headers=headers,
            ) as resp:
                return await resp.json()
    except Exception as e:
        logger.debug(f'Reward service call failed: {type(e).__name__}: {e}')
        return None


async def hy_judge(
    answer: str,
    ground_truth: str,
    extra_info: dict,
    traj_id: str = 'unknown',
) -> dict[str, Any]:
    """
    调用 Hunyuan Reward 平台进行评分。

    对齐 baseline llm_as_a_judge() 函数：
    - 无 rate_limiter
    - 无总超时
    - error_code/低分时直接 continue 不 sleep
    - 仅异常时 sleep(1)

    Args:
        answer: 提取后的模型回答
        ground_truth: 参考答案
        extra_info: 额外信息（包含 history, question, extra, reward_system_router_id 等）
        traj_id: 轨迹唯一标识

    Returns:
        包含 hy_score 的字典
    """
    score_dict: dict[str, Any] = {
        'hy_score': 0,
        'hy_score_json': '{}',
    }

    # 从 extra_info 中获取 prompt 和其他信息
    history_list = extra_info.get('history') or []
    if not isinstance(history_list, list):
        history_list = []
    prompt = extra_info.get('question') or ''
    messages = history_list + [{'role': 'user', 'content': prompt}, {'role': 'assistant', 'content': answer}]

    reward_extra = extra_info.get('extra') or {}
    reward_system_router_id = extra_info.get('reward_system_router_id', '')

    try:
        class1, class2, router_version = reward_system_router_id.split('-')
    except (ValueError, AttributeError):
        raise ValueError(f'Invalid reward_system_router_id: {reward_system_router_id}') from None

    _log_reward(traj_id, 'HY_JUDGE_START')

    return await _hy_judge_inner(
        score_dict, messages, ground_truth, class1, class2, router_version, reward_extra, traj_id
    )


async def _hy_judge_inner(
    score_dict: dict[str, Any],
    messages: list,
    ground_truth: str,
    class1: str,
    class2: str,
    router_version: str,
    reward_extra: dict,
    traj_id: str,
) -> dict[str, Any]:
    """hy_judge 的内部实现，被 wait_for 包裹以支持总超时。"""
    last_exception: Exception | None = None
    last_reason = 'unknown'
    rate_limiter = get_rate_limiter('hy_judge', max_concurrent=RATE_LIMIT)
    rate_limit_wait_time = 0.0

    for attempt in range(MAX_RETRIES):
        try:
            _log_reward(traj_id, 'HY_JUDGE_ATTEMPT', f'attempt={attempt + 1}/{MAX_RETRIES}')

            data = {
                'request_id': f'{uuid.uuid4()}',
                'exp_id': 'waterr_1234',
                'source': 'yb_rl_collobarate',
                'model_type': 'hy_turbo_v2.07',
                'messages': messages,
                'ref_answer': ground_truth,
                'class1': class1,
                'class2': class2,
                'router_version': int(router_version),
                'public_test': [],
                'extra': reward_extra,
            }

            rl_start = asyncio.get_event_loop().time()
            async with rate_limiter:
                rl_wait = asyncio.get_event_loop().time() - rl_start
                rate_limit_wait_time += rl_wait
                if rl_wait > 10:
                    _log_reward(traj_id, 'RATE_LIMIT_WAIT', f'{rl_wait:.2f}s')
                async with async_timeout(SINGLE_ATTEMPT_TIMEOUT):
                    res = await _call_reward_service(data)

            if res is None:
                last_reason = 'empty_response'
                continue

            if res['data']['reward']['error_code'] != 0:
                last_reason = f'error_code={res["data"]["reward"]["error_code"]}'
                continue

            score = res['data']['reward']['final_score']
            if score < -99:
                last_reason = f'invalid_score={score}'
                continue

            score_dict['hy_score'] = float(score)
            trimmed = {
                'status': res.get('status'),
                'msg': res.get('msg'),
                'data': {
                    'error_code': res['data'].get('error_code'),
                    'request_id': res['data'].get('request_id'),
                    'cost_time': res['data'].get('cost_time'),
                    'reward': res['data'].get('reward'),
                },
            }
            score_dict['hy_score_json'] = json.dumps(trimmed, ensure_ascii=False)
            score_dict['reward_rate_limit_wait_time'] = rate_limit_wait_time

            _log_reward(traj_id, 'HY_JUDGE_DONE', f'hy_score={score}')
            return score_dict

        except TimeoutError:
            logger.debug(
                f'[{traj_id}] HY judge attempt {attempt + 1}/{MAX_RETRIES} timed out after {SINGLE_ATTEMPT_TIMEOUT}s'
            )
            last_exception = TimeoutError(f'single attempt timeout after {SINGLE_ATTEMPT_TIMEOUT}s')
            await asyncio.sleep(1)
        except Exception as e:
            logger.debug(f'[{traj_id}] HY judge attempt {attempt + 1}/{MAX_RETRIES} exception: {type(e).__name__}: {e}')
            last_exception = e
            await asyncio.sleep(1)

    _log_reward(traj_id, 'HY_JUDGE_FAILED', f'max_retries={MAX_RETRIES}')
    logger.error(
        f'[{traj_id}] HY judge failed after {MAX_RETRIES} retries: '
        f'{type(last_exception).__name__ if last_exception else "RuntimeError"}: {last_exception}'
    )
    if last_exception is not None:
        raise last_exception
    raise RuntimeError(f'HY judge failed after {MAX_RETRIES} retries: {last_reason}')


async def compute_score(
    solution_str: str,
    ground_truth: str = '暂无',
    extra_info: dict | None = None,
) -> dict[str, Any]:
    """
    计算 Hunyuan Reward 评分。

    对齐 baseline agts_hy_reward.compute_score()。

    Args:
        solution_str: 模型生成的完整回复
        ground_truth: 参考答案
        extra_info: 额外信息

    Returns:
        包含 hy_score 的字典
    """
    extra_info = extra_info or {}
    traj_id = extra_info.get('traj_id', 'unknown')

    # 设置默认模型类型
    extra_info['model_type'] = extra_info.get('model_type', 'ds')

    # 提取最终回答（复用 closed_qa_judge 中的 extract_query_solution）
    answer = extract_query_solution(solution_str, extra_info)

    # 调用 HY Reward 平台评分
    return await hy_judge(answer, ground_truth, extra_info, traj_id=traj_id)
