"""
Pairwise Reward Model Judge 模块。

对齐 baseline: verl/utils/reward_score/pairwise_judge.py

pairwise 模型返回一个分数，通过 Polaris 服务发现找到 pairwise RM 服务进行评分。
"""

import asyncio
import logging
import os
import threading

from functools import partial
from typing import Any

import aiohttp

from nexus.extensions.projects.yb_search.log_utils import log_trace
from nexus.utils.rate_limiter import get_rate_limiter
from nexus.utils.timeout import async_timeout

from .judge_utils import retry_backoff_sleep


logger = logging.getLogger(__name__)

MAX_RETRIES = int(os.environ.get('PAIRWISE_JUDGE_MAX_RETRIES', 10))
SINGLE_ATTEMPT_TIMEOUT = int(os.environ.get('PAIRWISE_JUDGE_TIMEOUT_SINGLE', 200))
RATE_LIMIT = int(os.environ.get('PAIRWISE_JUDGE_RATE_LIMIT', 6))  # 最大并发数

_log_reward = partial(log_trace, 'REWARD')

DEFAULT_PAIRWISE_MODEL_NAME = os.environ.get('PAIRWISE_MODEL_NAME', 'rm_exp2.9_step45_sz')
DEFAULT_PAIRWISE_MODEL_NAME_APP = os.environ.get('PAIRWISE_MODEL_NAME_APP', 'BTRM_app_gpt52only_hy27based')
DEFAULT_PAIRWISE_AUTH_TOKEN = os.environ.get('PAIRWISE_AUTH_TOKEN')
DEFAULT_PAIRWISE_WSID = os.environ.get('PAIRWISE_WSID', '11644')

print(
    f'pairwise_judge initialized with {DEFAULT_PAIRWISE_MODEL_NAME=}, {RATE_LIMIT=}, {MAX_RETRIES=}, '
    f'single_attempt_timeout={SINGLE_ATTEMPT_TIMEOUT}'
)

_consumer_lock = threading.Lock()
_consumer_api = None


def _get_consumer_api():
    """获取缓存的 Polaris consumer 实例，避免高并发下反复创建导致线程资源耗尽。"""
    global _consumer_api
    if _consumer_api is None:
        with _consumer_lock:
            if _consumer_api is None:
                from polaris.api.consumer import create_consumer_by_config

                _consumer_api = create_consumer_by_config('')
                logger.debug('Polaris consumer created and cached')
    return _consumer_api


def has_content(obj) -> bool:
    """安全检查对象是否有内容，兼容 list 和 numpy array。"""
    if obj is None:
        return False
    try:
        return len(obj) > 0
    except (TypeError, ValueError):
        return False


def extract_final_answer(solution_str: str, extra_info: dict) -> str:
    """
    从 solution_str 中提取模型的最终回答。

    对齐 baseline pairwise_judge.py 的 extract_final_answer。
    """
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


def build_pairwise_messages_bt_rm(extra_info: dict, solution_str: str, ground_truth: str = '') -> list:
    """
    构建发送给 pairwise 模型的 messages。

    对齐 baseline pairwise_judge.py 的 build_pairwise_messages_BT_RM。

    格式: system + history + user_query + assistant_answer
    """
    # 提取模型最终回答
    extra_info.setdefault('model_type', 'ds')
    final_answer = extract_final_answer(solution_str, extra_info)

    # 获取 prompt messages（可能是 list 或 numpy array）
    prompt_messages = extra_info.get('prompt', extra_info.get('messages', []))
    if hasattr(prompt_messages, 'tolist'):
        prompt_messages = prompt_messages.tolist()

    # 提取 history（优先从 extra_info["history"]，否则从 prompt_messages 提取）
    history = extra_info.get('history', [])
    if hasattr(history, 'tolist'):
        history = history.tolist()

    # 如果 history 为空，从 prompt_messages 提取（除了最后一个 user）
    if not has_content(history) and has_content(prompt_messages):
        history = []
        for msg in prompt_messages:
            if not isinstance(msg, dict):
                continue
            role = msg.get('role', '')
            content = msg.get('content', '')
            # 跳过 system、tool 等角色
            if role in ['user', 'assistant']:
                # 跳过包含 tool_calls 的 assistant 消息
                if role == 'assistant' and msg.get('tool_calls'):
                    continue
                history.append({'role': role, 'content': content})
        # 移除最后一个 user 消息（这是当前问题）
        if history and history[-1].get('role') == 'user':
            history = history[:-1]

    # 提取当前问题（优先从 extra_info，否则从 prompt_messages 提取）
    query = extra_info.get('query', extra_info.get('question', ''))
    if not query and has_content(prompt_messages):
        # 从 prompt_messages 中找最后一个 user 消息
        for msg in reversed(prompt_messages):
            if isinstance(msg, dict) and msg.get('role') == 'user':
                query = msg.get('content', '')
                break

    current_datetime = extra_info.get('current_time', '')
    current_location = extra_info.get('current_location', '')

    reward_system_router_id = extra_info.get('reward_system_router_id', '')
    if 'need_rubrics' in reward_system_router_id:
        query = f'{query}\n{ground_truth}'

    system_message = [{'role': 'system', 'content': f'当前日期：{current_datetime}\n用户位置：{current_location}'}]
    return (
        system_message + history + [{'role': 'user', 'content': query}, {'role': 'assistant', 'content': final_answer}]
    )


async def pairwise_model_invoke_async(
    messages: list,
    model_name: str | None = None,
) -> float:
    """
    异步调用 pairwise 模型。

    对齐 baseline pairwise_judge.py 的 pairwise_model_invoke_async。

    通过 Polaris 服务发现获取实例 IP/端口，向 http://{ip}:8021/classify 发送 POST 请求。

    Args:
        messages: 消息列表
        model_name: 模型名称

    Returns:
        raw_score: 原始分数
    """
    from polaris.pkg.model.error import SDKError
    from polaris.pkg.model.service import GetOneInstanceRequest

    if model_name is None:
        model_name = DEFAULT_PAIRWISE_MODEL_NAME

    if not DEFAULT_PAIRWISE_AUTH_TOKEN:
        raise ValueError('PAIRWISE_AUTH_TOKEN is not set')
    headers = {
        'Content-Type': 'application/json',
        'Authorization': f'Bearer {DEFAULT_PAIRWISE_AUTH_TOKEN}',
        'wsid': DEFAULT_PAIRWISE_WSID,
    }

    namespace = 'Production'
    service = model_name + '_AIDE'

    consumer_api = _get_consumer_api()
    request = GetOneInstanceRequest(namespace=namespace, service=service)
    try:
        response = consumer_api.get_one_instance_resp(request)
    except SDKError as e:
        raise Exception(f'Polaris SDK error: {repr(e)}') from e

    if response.is_service_data_expired():
        logger.warning(f'service {namespace}/{service} data is expired')

    instance = response[0]
    ip = instance.get_host()
    instance.get_port()

    ss_url = f'http://{ip}:8021/classify'

    json_data = {
        'model': model_name,
        'messages': messages,
        'temperature': 0.7,
        'top_p': 1.0,
        'query_id': 'pairwise_judge_' + str(id(messages)),
        'openai_infer': True,
        'stream': False,
        'tools': [],
        'add_generation_prompt': True,
        'chat_template_kwargs': {
            'enable_thinking': False,
        },
    }

    logger.debug(f'[PAIRWISE_API] Request URL={ss_url}, Model={model_name}, Messages count={len(messages)}')

    async with aiohttp.ClientSession() as session:
        async with session.post(
            ss_url,
            headers=headers,
            json=json_data,
            timeout=aiohttp.ClientTimeout(total=60),
        ) as resp:
            if resp.status != 200:
                text = await resp.text()
                logger.error(f'[PAIRWISE_API] HTTP Error: {resp.status}, {text[:200]}')
                raise Exception(f'HTTP {resp.status}: {text[:200]}')

            res = await resp.json()
            raw_score = res['data'][0]['probs'][0]

            logger.info(f'[PAIRWISE_API] Response: raw_score={raw_score:.4f}')

            return raw_score


async def compute_score(
    solution_str: str,
    ground_truth: str = '无参考答案',
    extra_info: dict | None = None,
) -> dict[str, Any]:
    """
    Pairwise 数据的评分函数。

    对齐 baseline pairwise_judge.py 的 compute_score。

    Args:
        solution_str: 模型生成的完整回答
        ground_truth: 参考答案（pairwise 通常不需要）
        extra_info: 包含 prompt, messages 等信息的字典

    Returns:
        score_dict: 包含各项评分的字典
    """
    extra_info = extra_info or {}
    traj_id = extra_info.get('traj_id', 'unknown')

    logger.debug(f'[{traj_id}] [PAIRWISE_JUDGE] compute_score called, solution_str length={len(solution_str)}')

    score_dict: dict[str, Any] = {
        'pairwise_score': 0.0,
        'pairwise_raw_score': 0.0,
        'pairwise_error': '',
    }

    # 搜图或强视频场景跳过 pairwise 评分（对齐 baseline）
    image_class = extra_info.get('图片分类', '')
    video_tag = extra_info.get('视频等级', '')
    if image_class == '搜图' or video_tag == 'strong':
        return score_dict

    last_error = None

    # 构建 messages
    messages = build_pairwise_messages_bt_rm(extra_info, solution_str, ground_truth)

    if not messages:
        raise ValueError('pairwise_judge: No messages found in extra_info')

    _log_reward(traj_id, 'PAIRWISE_JUDGE_START')

    # APP 场景使用单独的 RM
    is_app = extra_info.get('data_origin') == 'app'
    model_name = DEFAULT_PAIRWISE_MODEL_NAME_APP if is_app else DEFAULT_PAIRWISE_MODEL_NAME

    # 按模型分别限流：APP RM 与通用 RM 是两套服务实例，独立并发池更贴近实际容量
    rate_limiter = get_rate_limiter(f'pairwise_rm:{model_name}', max_concurrent=RATE_LIMIT)
    rate_limit_wait_time = 0.0

    for attempt in range(MAX_RETRIES):
        try:
            rl_start = asyncio.get_event_loop().time()
            async with rate_limiter:
                rl_wait = asyncio.get_event_loop().time() - rl_start
                rate_limit_wait_time += rl_wait
                if rl_wait > 10:
                    _log_reward(traj_id, 'RATE_LIMIT_WAIT', f'{rl_wait:.2f}s')
                async with async_timeout(SINGLE_ATTEMPT_TIMEOUT):
                    raw_score = await pairwise_model_invoke_async(messages, model_name=model_name)

            normalized_score = raw_score

            score_dict['pairwise_raw_score'] = float(raw_score)
            score_dict['pairwise_score'] = float(normalized_score)
            score_dict['pairwise_error'] = ''
            score_dict['reward_rate_limit_wait_time'] = rate_limit_wait_time

            _log_reward(traj_id, 'PAIRWISE_JUDGE_DONE', f'raw_score={raw_score:.4f}')

            return score_dict

        except TimeoutError:
            logger.debug(
                f'[{traj_id}] Pairwise judge attempt {attempt + 1}/{MAX_RETRIES} timed out after {SINGLE_ATTEMPT_TIMEOUT}s'
            )
            last_error = f'single attempt timeout after {SINGLE_ATTEMPT_TIMEOUT}s'
            await retry_backoff_sleep(attempt)
        except Exception as e:
            logger.debug(
                f'[{traj_id}] Pairwise judge attempt {attempt + 1}/{MAX_RETRIES} exception: {type(e).__name__}: {e}'
            )
            last_error = str(e)
            await retry_backoff_sleep(attempt)

    _log_reward(traj_id, 'PAIRWISE_JUDGE_FAILED', f'max_retries={MAX_RETRIES}')
    logger.error(f'[{traj_id}] Pairwise model invoke failed after {MAX_RETRIES} retries: {last_error}')
    raise RuntimeError(last_error or f'Pairwise model invoke failed after {MAX_RETRIES} retries')
