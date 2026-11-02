"""CSIG 后端直连。

从 litellm_router/proxy_csig.py 内联而来。策略不变：用流式请求
（stream=True）发给 CSIG，在本地收集所有 chunk，拼装成标准 OpenAI 非流式
响应 dict 返回。这样避免后端非流式超时限制。失败时指数退避重试。
"""

from __future__ import annotations

import asyncio
import json
import logging

import httpx

from .config import (
    CSIG_API_KEY,
    CSIG_BASE,
    CSIG_INITIAL_BACKOFF,
    CSIG_MAX_RETRIES,
    CSIG_TIMEOUT,
)
from .router import RouteInfo


logger = logging.getLogger(__name__)

_REQUEST_PATH = '/v1/chat/completions'

_CACHE_MARK = {'type': 'ephemeral'}


def _mark_last_entry(entries: list) -> list | None:
    """在列表副本的最后一个 dict 上打 ``cache_control`` 标记并返回该副本。

    已打过标、列表为空、或末项不是 dict 时返回 ``None`` 表示无需改动
    （对应原先 ``setdefault`` 的"不覆盖既有标记"语义）。
    """
    if not entries or not isinstance(entries[-1], dict):
        return None
    if 'cache_control' in entries[-1]:
        return None
    marked = list(entries)
    marked[-1] = {**marked[-1], 'cache_control': dict(_CACHE_MARK)}
    return marked


def _text_blocks(text: str) -> list[dict]:
    """把纯文本包装成带 ``cache_control`` 标记的单个 text block。"""
    return [{'type': 'text', 'text': text, 'cache_control': dict(_CACHE_MARK)}]


def _inject_cache_control(payload: dict) -> None:
    """为 Anthropic 请求注入 cache_control 标记以启用 Prompt Caching。

    参考 litellm_router/proxy_csig.py 的策略：在不变的前缀上打 ``ephemeral`` 标记，
    让 Anthropic 缓存这些前缀、后续请求复用，减少重复计算开销：
      - system prompt 的最后一个 content block
      - tools 列表的最后一个 tool 定义
      - messages 中从后往前第一条可打标的非 system 消息的最后一个 content block
        （断点随多轮后移；content 为空则跳过继续往前）

    三步全部 copy-on-write：只复制被打标的那一层容器与 dict，不改动调用方传入的
    原始对象——``messages`` / ``tools`` 往往是 agent 循环里持有的同一份 history 与
    tool schema，原地 mutate 会让标记随对象引用跨轮次残留累积，超过 Anthropic 每
    请求最多 4 个 cache_control breakpoint 的限制触发 400；且会把 system content 由
    str 永久改写成 content block 列表，使 trace 里同一条 system 消息在首轮与后续轮
    形态不一致，导致 chat_tracker 无法按前缀去重、分叉出被截断的假轨迹。
    """
    # 1. system prompt
    if payload.get('system'):
        # Anthropic 原生格式：顶层 system 为 str 或 content block 列表
        sys_val = payload['system']
        if isinstance(sys_val, list):
            marked = _mark_last_entry(sys_val)
            if marked is not None:
                payload['system'] = marked
        elif isinstance(sys_val, str):
            payload['system'] = _text_blocks(sys_val)
    else:
        # OpenAI 兼容格式：首条 role=system 消息
        messages = payload.get('messages') or []
        for index, msg in enumerate(messages):
            if msg.get('role') != 'system':
                continue
            content = msg.get('content')
            new_content = None
            if isinstance(content, list):
                new_content = _mark_last_entry(content)
            elif isinstance(content, str) and content:
                new_content = _text_blocks(content)
            if new_content is not None:
                new_messages = list(messages)
                new_messages[index] = {**msg, 'content': new_content}
                payload['messages'] = new_messages
            break

    # 2. tools 最后一个定义
    tools = payload.get('tools')
    if isinstance(tools, list):
        marked_tools = _mark_last_entry(tools)
        if marked_tools is not None:
            payload['tools'] = marked_tools

    # 3. 对话末尾：从后往前找第一条能挂断点的非 system 消息，在副本上打标。
    messages = payload.get('messages')
    if messages:
        new_messages = list(messages)
        for i in range(len(new_messages) - 1, -1, -1):
            msg = new_messages[i]
            if msg.get('role') == 'system':
                continue
            content = msg.get('content')
            if isinstance(content, list) and content and isinstance(content[-1], dict):
                blocks = list(content)
                blocks[-1] = {**blocks[-1], 'cache_control': dict(_CACHE_MARK)}
                new_messages[i] = {**msg, 'content': blocks}
                break
            if isinstance(content, str) and content:
                new_messages[i] = {**msg, 'content': _text_blocks(content)}
                break
        payload['messages'] = new_messages


def _log_cache_stats(assembled: dict, route: RouteInfo) -> None:
    """打印 Anthropic Prompt Caching 命中情况（基于 usage）。"""
    try:
        usage = assembled.get('usage', {}) or {}
        cache_create = usage.get('cache_creation_input_tokens', 0)
        cache_read = usage.get('cache_read_input_tokens', 0)
        prompt_tokens = usage.get('prompt_tokens', 0)
        logger.debug(
            'Anthropic cache stats model=%s: prompt_tokens=%d, cache_creation=%d, cache_read=%d, cache_hit_rate=%.1f%%',
            route.model,
            prompt_tokens,
            cache_create,
            cache_read,
            (cache_read / max(prompt_tokens, 1)) * 100,
        )
    except Exception:
        pass


def assemble_stream_chunks(chunks: list[dict]) -> dict:
    """将流式 SSE chunk 列表拼装成标准 OpenAI 非流式响应 dict。"""
    if not chunks:
        return {'choices': [], 'usage': {}}

    collected: dict[int, dict] = {}

    last_chunk = chunks[-1]
    model = last_chunk.get('model', '')
    created = last_chunk.get('created', 0)
    chunk_id = last_chunk.get('id', '')

    for chunk in chunks:
        for choice in chunk.get('choices', []):
            idx = choice.get('index', 0)
            if idx not in collected:
                collected[idx] = {
                    'content': '',
                    'reasoning_content': '',
                    'thinking_blocks': {},
                    'tool_calls': {},
                    'finish_reason': None,
                    'role': 'assistant',
                }

            delta = choice.get('delta', {})

            if delta.get('role'):
                collected[idx]['role'] = delta['role']
            if delta.get('content'):
                collected[idx]['content'] += delta['content']
            # 扩展思考（Anthropic）：litellm 以 delta.reasoning_content 流式返回思考增量，
            # 若不在此拼装会被丢弃，导致 messages/trajectory 里看不到思考内容。
            if delta.get('reasoning_content'):
                collected[idx]['reasoning_content'] += delta['reasoning_content']
            # thinking_blocks 只累积 type/signature/data；thinking 文本由收尾阶段
            # 从 reasoning_content 取，避免与 reasoning_content 重复拼一遍。
            for tb_i, tb in enumerate(delta.get('thinking_blocks') or []):
                tb_acc = collected[idx]['thinking_blocks'].setdefault(
                    tb_i,
                    {'type': 'thinking', 'signature': None, 'data': ''},
                )
                if tb.get('type'):
                    tb_acc['type'] = tb['type']
                if tb.get('signature'):
                    tb_acc['signature'] = tb['signature']
                if tb.get('data'):
                    tb_acc['data'] += tb['data']

            for tc_delta in delta.get('tool_calls', []):
                tc_idx = tc_delta.get('index', 0)
                if tc_idx not in collected[idx]['tool_calls']:
                    collected[idx]['tool_calls'][tc_idx] = {
                        'id': '',
                        'type': 'function',
                        'function': {'name': '', 'arguments': ''},
                    }
                tc = collected[idx]['tool_calls'][tc_idx]
                if tc_delta.get('id'):
                    tc['id'] += tc_delta['id']
                func = tc_delta.get('function', {})
                if func.get('name'):
                    tc['function']['name'] += func['name']
                if func.get('arguments'):
                    tc['function']['arguments'] += func['arguments']

            if choice.get('finish_reason'):
                collected[idx]['finish_reason'] = choice['finish_reason']

    choices = []
    for idx in sorted(collected):
        c = collected[idx]
        message: dict = {'role': c['role'], 'content': c['content'] or None}
        if c.get('reasoning_content'):
            message['reasoning_content'] = c['reasoning_content']
        if c.get('thinking_blocks'):
            tbs = []
            reasoning_text = c.get('reasoning_content') or ''
            reasoning_used = False
            for i in sorted(c['thinking_blocks']):
                tb = c['thinking_blocks'][i]
                if tb.get('type') == 'redacted_thinking' and tb.get('data'):
                    tbs.append({'type': 'redacted_thinking', 'data': tb['data']})
                elif tb.get('signature') or (not reasoning_used and reasoning_text):
                    # 思考正文只挂在第一个 thinking block 上，后续块留空，
                    # 避免同一段 reasoning_content 被重复计入多个块。
                    tbs.append({
                        'type': 'thinking',
                        'thinking': reasoning_text if not reasoning_used else '',
                        'signature': tb.get('signature'),
                    })
                    reasoning_used = True
            if tbs:
                message['thinking_blocks'] = tbs
        if c['tool_calls']:
            message['tool_calls'] = [c['tool_calls'][i] for i in sorted(c['tool_calls'])]
        choices.append({
            'index': idx,
            'message': message,
            'finish_reason': c['finish_reason'],
            'logprobs': None,
        })

    usage = last_chunk.get('usage', {})
    if not usage:
        for chunk in reversed(chunks):
            u = chunk.get('usage')
            if u:
                usage = u
                break
    return {
        'id': chunk_id,
        'object': 'chat.completion',
        'created': created,
        'model': model,
        'choices': choices,
        'usage': usage,
    }


async def _stream_and_collect(
    client: httpx.AsyncClient,
    url: str,
    body: dict,
    headers: dict,
    timeout: float,
) -> dict:
    """用流式请求发送，收集所有 chunk 拼装成非流式响应 dict。

    非 200 时抛出 httpx.HTTPStatusError。
    """
    chunks: list[dict] = []

    async with client.stream(
        'POST',
        url,
        json=body,
        headers=headers,
        timeout=timeout,
    ) as resp:
        if resp.status_code != 200:
            await resp.aread()
            resp.raise_for_status()

        buffer = ''
        async for text in resp.aiter_text():
            buffer += text
            while '\n' in buffer:
                line, buffer = buffer.split('\n', 1)
                line = line.strip()
                if not line:
                    continue
                if line.startswith('data: '):
                    data_str = line[6:]
                    if data_str.strip() == '[DONE]':
                        continue
                    try:
                        chunks.append(json.loads(data_str))
                    except json.JSONDecodeError:
                        logger.warning('CSIG stream: bad JSON chunk: %s', data_str[:200])

        if buffer.strip():
            line = buffer.strip()
            if line.startswith('data: '):
                data_str = line[6:]
                if data_str.strip() != '[DONE]':
                    try:
                        chunks.append(json.loads(data_str))
                    except json.JSONDecodeError:
                        logger.warning('CSIG stream: bad JSON in trailing buffer: %s', data_str[:200])

    return assemble_stream_chunks(chunks)


async def forward(
    client: httpx.AsyncClient,
    body: dict,
    route: RouteInfo,
    strategy_type: str = '',
    session_id: str = '',
) -> dict:
    """转发请求到 CSIG，返回标准 OpenAI 非流式响应 dict。

    Prompt cache（命中需"建缓存"+"路由到同一实例"两者同时满足）：
      1. 对 anthropic 注入 ``cache_control`` 创建 prefix 缓存（建缓存）。
      2. 设置请求体 ``user`` 字段做一致性哈希路由——CSIG 上游 litellm proxy
         按 ``user`` 把同一 key 的请求粘连到同一推理实例，命中该实例上的缓存。
         （实测：固定 ``user`` 可达 7/8 命中；而 Taiji 式 ``StrategyType/SessionID``
         header 对 CSIG 无效 ≈ 随机路由，故仅作无害冗余保留。）

    重试策略：最多 CSIG_MAX_RETRIES 次，指数退避。
    """
    url = CSIG_BASE + _REQUEST_PATH
    headers = {
        'Authorization': f'Bearer {CSIG_API_KEY}',
        'Content-Type': 'application/json',
    }
    # Taiji 式一致性哈希 header：CSIG 上游不识别，仅作无害冗余保留。
    if strategy_type:
        headers['StrategyType'] = str(strategy_type)
    if session_id:
        headers['SessionID'] = str(session_id)

    payload = {**body, 'stream': True}
    # ── CSIG prompt cache 的关键：用 user 字段做粘连路由 ──
    # litellm proxy 按请求体 user 一致性哈希到固定实例，使同 session 的请求命中
    # 同一实例上由 cache_control 建立的 prefix 缓存。仅在调用方未显式指定 user 时注入。
    if session_id and not payload.get('user'):
        payload['user'] = str(session_id)
    # stream=True 默认不返回 usage，必须开 stream_options.include_usage
    so = dict(payload.get('stream_options') or {})  # type: ignore[no-matching-overload]
    so.setdefault('include_usage', True)
    payload['stream_options'] = so
    # Anthropic 不支持 top_p 与 temperature 同时传，自动去掉 top_p
    if route.provider == 'anthropic':
        payload.pop('top_p', None)
        # anthropic-beta 由 yaml request_args 传入，拼成逗号分隔的 HTTP header 发给上游。
        raw_betas = payload.pop('anthropic-beta', [])
        if isinstance(raw_betas, str):
            betas: list[str] = [raw_betas]
        elif isinstance(raw_betas, list):
            betas = [str(b) for b in raw_betas]
        else:
            betas = []
        # 交联思考（interleaved thinking）：每次工具结果后继续思考。
        # 触发条件：payload 里有 thinking dict 且 type == enabled。
        thinking = payload.get('thinking') or {}
        if (
            isinstance(thinking, dict)
            and thinking.get('type') == 'enabled'
            and 'interleaved-thinking-2025-05-14' not in betas
        ):
            betas.append('interleaved-thinking-2025-05-14')
        if betas:
            headers['anthropic-beta'] = ','.join(str(b) for b in betas)
        # ── Anthropic Prompt Caching ──
        # 在 system / tools / 最后一条 user 的前缀上打 cache_control(ephemeral)，
        # 让 Anthropic 缓存不变前缀，后续请求复用以降本提速。
        _inject_cache_control(payload)

    last_exc: Exception | None = None
    for attempt in range(CSIG_MAX_RETRIES):
        try:
            assembled = await _stream_and_collect(client, url, payload, headers, CSIG_TIMEOUT)
            if route.provider == 'anthropic':
                _log_cache_stats(assembled, route)
            return assembled
        except httpx.HTTPStatusError as exc:
            # 400 参数错误不重试
            if exc.response is not None and exc.response.status_code == 400:
                raise
            last_exc = exc
        except httpx.TransportError as exc:
            last_exc = exc

        if attempt < CSIG_MAX_RETRIES - 1:
            wait = CSIG_INITIAL_BACKOFF * (2**attempt)
            logger.warning(
                'CSIG retry %d/%d model=%s: %s. Wait %.0fs',
                attempt + 1,
                CSIG_MAX_RETRIES,
                route.model,
                last_exc,
                wait,
            )
            await asyncio.sleep(wait)

    raise last_exc  # type: ignore[misc]
