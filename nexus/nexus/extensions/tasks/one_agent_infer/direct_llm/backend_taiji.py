"""Taiji 后端直连。

从 litellm_router/proxy_taiji.py 内联而来。策略不变：用流式请求（stream=True）
发给太极，本地收集所有 chunk 拼装成标准 OpenAI 非流式响应 dict 返回，避免太极
非流式接口的 ~120s 超时限制。失败时指数退避重试。

Prompt cache：通过注入 StrategyType=ConsistentHash + 稳定 SessionID 这两个
header，让同一会话多轮请求落到同一推理实例，命中该实例的 prefix/KV cache。
StrategyType / SessionID 由 client 计算后传入（见 client.py）。

SBS 直连：当 ``ONE_AGENT_TAIJI_URL`` 命中
``stream-server-online-sbs-{wsid}.turbotke.production.polaris`` 时，自动把 wsid
写入 header。
"""

from __future__ import annotations

import asyncio
import json
import logging
import re
import uuid

import httpx

from .config import (
    TAIJI_API_KEY,
    TAIJI_INITIAL_BACKOFF,
    TAIJI_MAX_RETRIES,
    TAIJI_REASONING_EFFORT,
    TAIJI_TIMEOUT,
    TAIJI_URL,
)
from .router import RouteInfo


logger = logging.getLogger(__name__)

# SBS 直连 Polaris 域名：stream-server-online-sbs-{wsid}.turbotke.production.polaris
# 比通用 taiji-stream-server-online-openapi 网关更快；命中后必须带 wsid header。
_SBS_WSID_RE = re.compile(
    r'stream-server-online-sbs-(\d+)\.turbotke\.production\.polaris',
    re.IGNORECASE,
)


def extract_sbs_wsid(url: str) -> str:
    """从 SBS Polaris URL 解析 wsid；不匹配则返回空串。"""
    if not url:
        return ''
    match = _SBS_WSID_RE.search(url)
    return match.group(1) if match else ''


def build_headers(strategy_type: str = '', session_id: str = '') -> dict:
    """构造发往 Taiji 的 headers。

    StrategyType / SessionID 用于一致性哈希路由（prompt cache 命中）。
    注意：Taiji header 字段名大小写严格区分：StrategyType / SessionID。
    ``TAIJI_URL`` 命中 SBS pattern 时额外带 wsid（与 curl ``-H "wsid: 11644"`` 对齐）。
    """
    headers = {
        'Authorization': f'Bearer {TAIJI_API_KEY}',
        'Content-Type': 'application/json',
        'wsid': '11644',
    }
    if strategy_type:
        headers['StrategyType'] = str(strategy_type)
    if session_id:
        headers['SessionID'] = str(session_id)
    wsid = extract_sbs_wsid(TAIJI_URL)
    if wsid:
        headers['wsid'] = wsid
    return headers


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
                    'tool_calls': {},
                    'finish_reason': None,
                    'role': 'assistant',
                }

            delta = choice.get('delta', {})

            if delta.get('role'):
                collected[idx]['role'] = delta['role']
            if delta.get('content'):
                collected[idx]['content'] += delta['content']
            # 推理模型的思维链走 delta.reasoning_content，与 content 分开逐块吐出
            if delta.get('reasoning_content'):
                collected[idx]['reasoning_content'] += delta['reasoning_content']

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
        if c['reasoning_content']:
            message['reasoning_content'] = c['reasoning_content']
        if c['tool_calls']:
            message['tool_calls'] = [c['tool_calls'][i] for i in sorted(c['tool_calls'])]
        choices.append({
            'index': idx,
            'message': message,
            'finish_reason': c['finish_reason'],
            'logprobs': None,
        })

    usage = last_chunk.get('usage', {})
    return {
        'id': chunk_id,
        'object': 'chat.completion',
        'created': created,
        'model': model,
        'choices': choices,
        'usage': usage,
    }


# ── Tool call 解析（HunYuan 模型格式）────────────────────────────
#
# 兼容两种标签形态：
#   1. 旧格式（无后缀）：
#        <tool_calls>
#        <tool_call>name<tool_sep>
#        <arg_key>k</arg_key><arg_value>v</arg_value>
#        </tool_call>
#        </tool_calls>
#   2. 新格式（每个标签带 per-response hash 防伪后缀，如 :6124c78e）：
#        <tool_calls:6124c78e>
#        <tool_call:6124c78e>name<tool_sep:6124c78e>
#        <arg_key:6124c78e>k</arg_key:6124c78e><arg_value:6124c78e>v</arg_value:6124c78e>
#        </tool_call:6124c78e>
#        </tool_calls:6124c78e>
#      （HY3.0-dev0525+ 起，模型给每个结构标签追加 ``:<hash>`` 后缀防止被用户
#       内容伪造；同一条响应内 hash 固定。）
#
# 做法：在每个标签名后允许可选的 ``:<token>`` 后缀（``_TAG_SUFFIX``），两种
# 形态用同一套正则覆盖。开闭标签各自允许独立后缀以保持简单——同一响应内
# hash 必然一致，不会误匹配。

# 标签名后可选的 hash 后缀，例如 <tool_calls:6124c78e> 中的 ":6124c78e"
_TAG_SUFFIX = r'(?::[0-9A-Za-z_]+)?'

# 预编译：检测/拆解各标签（DOTALL 让 .*? 跨行）
_RE_HAS_TOOL_CALLS = re.compile(rf'<tool_calls{_TAG_SUFFIX}>')
_RE_TOOL_CALLS_BLOCK = re.compile(rf'<tool_calls{_TAG_SUFFIX}>(.*?)(?:</tool_calls{_TAG_SUFFIX}>|$)', re.DOTALL)
_RE_TOOL_CALL_BODY = re.compile(rf'<tool_call{_TAG_SUFFIX}>(.*?)(?:</tool_call{_TAG_SUFFIX}>|$)', re.DOTALL)
_RE_TOOL_SEP = re.compile(rf'<tool_sep{_TAG_SUFFIX}>')
_RE_ARG_KEY = re.compile(rf'<arg_key{_TAG_SUFFIX}>(.*?)</arg_key{_TAG_SUFFIX}>', re.DOTALL)
_RE_ARG_VALUE = re.compile(rf'<arg_value{_TAG_SUFFIX}>(.*?)</arg_value{_TAG_SUFFIX}>', re.DOTALL)
_RE_EMPTY_THINK = re.compile(rf'<think{_TAG_SUFFIX}>\s*</think{_TAG_SUFFIX}>\s*')

# ── 参数类型白名单 ─────────────────────────────────────────────────
# 白名单中的 (tool_name, param_name) 组合，其 <arg_value> 值应保留为原始
# string 不做 json.loads；不在白名单中的参数正常 json.loads 解析。
_STRING_PARAM_WHITELIST: set[tuple[str, str]] = {
    # bash / bash_v2
    ('bash', 'command'),
    ('bash', 'description'),
    # read / read_v2
    ('read', 'file_path'),
    ('read', 'description'),
    # edit / edit_v2
    ('edit', 'file_path'),
    ('edit', 'old_string'),
    ('edit', 'new_string'),
    ('edit', 'description'),
    # write / write_v2
    ('write', 'file_path'),
    ('write', 'content'),
    ('write', 'description'),
    # web_fetch
    ('web_fetch', 'url'),
    ('web_fetch', 'prompt'),
    # web_search
    ('web_search', 'description'),
    # picture_search
    ('picture_search', 'description'),
    # video_search
    ('video_search', 'description'),
    # load_skill
    ('load_skill', 'skill_name'),
    # image_generation/generate
    ('image_generation/generate', 'dst_intent'),
    ('image_generation/generate', 'prompt'),
    ('image_generation/generate', 'image_ratio'),
    # task_schedule
    ('task_schedule/create_task', 'instruction'),
    ('task_schedule/create_task', 'timezone'),
    ('task_schedule/create_task', 'task_kind'),
    ('task_schedule/create_task', 'run_at'),
    ('task_schedule/create_task', 'rrule'),
    ('task_schedule/create_task', 'title'),
    ('task_schedule/create_task', 'dtstart'),
    ('task_schedule/update_task', 'task_id'),
    ('task_schedule/update_task', 'title'),
    ('task_schedule/update_task', 'instruction'),
    ('task_schedule/update_task', 'timezone'),
    ('task_schedule/update_task', 'task_kind'),
    ('task_schedule/update_task', 'run_at'),
    ('task_schedule/update_task', 'rrule'),
    ('task_schedule/update_task', 'dtstart'),
    ('task_schedule/delete_task', 'task_id'),
}


def _parse_tool_calls_from_content(content: str) -> tuple[str, list[dict]]:
    """从 content 中解析 HunYuan 格式的 <tool_calls> 标签。

    兼容新旧两种标签形态（见上方说明）：标签名后可选 ``:<hash>`` 后缀。

    Returns (cleaned_content, tool_calls_list)。
    """
    if not content or not _RE_HAS_TOOL_CALLS.search(content):
        return content or '', []

    tc_match = _RE_TOOL_CALLS_BLOCK.search(content)
    if not tc_match:
        return content, []

    tc_block = tc_match.group(1)
    before = content[: tc_match.start()].strip()
    before = _RE_EMPTY_THINK.sub('', before).strip()

    tool_calls: list[dict] = []
    for tc_body_match in _RE_TOOL_CALL_BODY.finditer(tc_block):
        body = tc_body_match.group(1)
        sep_match = _RE_TOOL_SEP.search(body)
        if not sep_match:
            continue

        tool_name = body[: sep_match.start()].strip()
        if not tool_name:
            continue

        rest = body[sep_match.end() :]
        keys = _RE_ARG_KEY.findall(rest)
        values = _RE_ARG_VALUE.findall(rest)
        args = {}
        for k, v in zip(keys, values, strict=False):
            k = k.strip()
            if (tool_name, k) in _STRING_PARAM_WHITELIST:
                # 白名单参数：保留原始 string，不做 json.loads
                args[k] = v
            else:
                try:
                    args[k] = json.loads(v)
                except (json.JSONDecodeError, ValueError):
                    args[k] = v

        tool_calls.append({
            'id': f'chatcmpl-tool-{uuid.uuid4().hex[:16]}',
            'type': 'function',
            'function': {
                'name': tool_name,
                'arguments': json.dumps(args, ensure_ascii=False),
            },
        })

    return before, tool_calls


def _apply_tool_parsing(response: dict, had_tools: bool) -> dict:
    """当请求带 tools 时，把 content 中的 HunYuan tool_call 标签解析为标准 tool_calls。"""
    if not had_tools:
        return response

    for choice in response.get('choices', []):
        msg = choice.get('message', {})
        content = msg.get('content') or ''

        if msg.get('tool_calls'):
            continue

        cleaned, tool_calls = _parse_tool_calls_from_content(content)
        if tool_calls:
            msg['content'] = cleaned or None
            msg['tool_calls'] = tool_calls
            choice['finish_reason'] = 'tool_calls'

    return response


async def _stream_and_collect(
    client: httpx.AsyncClient,
    url: str,
    payload: dict,
    headers: dict,
    timeout: float,
    had_tools: bool = False,
) -> dict:
    """用流式请求发送，收集所有 chunk 拼装成非流式响应 dict。

    非 200 时抛出 httpx.HTTPStatusError。
    """
    chunks: list[dict] = []

    # 发送前移除内部控制字段（以 _ 开头）
    send_payload = {k: v for k, v in payload.items() if not k.startswith('_')}

    async with client.stream(
        'POST',
        url,
        json=send_payload,
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
                if line.startswith('data:'):
                    data_str = line[5:].lstrip()  # 兼容 "data:xxx" 和 "data: xxx"
                    if data_str.strip() == '[DONE]':
                        continue
                    try:
                        chunks.append(json.loads(data_str))
                    except json.JSONDecodeError:
                        logger.warning('Taiji stream: bad JSON chunk: %s', data_str[:200])

        if buffer.strip():
            line = buffer.strip()
            if line.startswith('data:'):
                data_str = line[5:].lstrip()
                if data_str.strip() != '[DONE]':
                    try:
                        chunks.append(json.loads(data_str))
                    except json.JSONDecodeError:
                        logger.warning('Taiji stream: bad JSON in trailing buffer: %s', data_str[:200])

    assembled = assemble_stream_chunks(chunks)
    # 自行解析 tool_call（已设置 tool_choice=none 绕过太极 parser）
    return _apply_tool_parsing(assembled, had_tools)


async def forward(
    client: httpx.AsyncClient,
    body: dict,
    route: RouteInfo,
    strategy_type: str = '',
    session_id: str = '',
) -> dict:
    """转发请求到 Taiji，返回标准 OpenAI 非流式响应 dict。

    注入 openai_infer / stream / query_id 字段；通过 tool_choice=none 绕过太极
    tool parser，自行解析 HunYuan tool_call 标签。

    Prompt cache：strategy_type/session_id 作为 StrategyType/SessionID header
    透传给太极，命中推理实例的 prefix/KV cache。

    重试策略：最多 TAIJI_MAX_RETRIES 次，指数退避；并检测 prompt_tokens=0 兜底响应。
    """
    headers = build_headers(strategy_type, session_id)

    payload = {
        **body,
        'openai_infer': True,
        'stream': True,
        'query_id': f'one_agent_infer_{uuid.uuid4()}',
    }

    had_tools = bool(payload.get('tools'))
    if had_tools:
        payload['tool_choice'] = 'none'

    # 慢思考：ONE_AGENT_TAIJI_REASONING_EFFORT 设置后才下发 reasoning_effort，
    # 未设置时不碰 payload（行为与开启前一致）。与 yaml request_args 里已有的
    # chat_template_kwargs 合并而非覆盖，且不改动调用方传入的原 dict。
    if TAIJI_REASONING_EFFORT:
        ctk = payload.get('chat_template_kwargs')
        ctk = dict(ctk) if isinstance(ctk, dict) else {}
        ctk['reasoning_effort'] = TAIJI_REASONING_EFFORT
        payload['chat_template_kwargs'] = ctk

    ctk = payload.get('chat_template_kwargs')
    reasoning_effort = ctk.get('reasoning_effort', '-') if isinstance(ctk, dict) else '-'
    logger.debug(
        'Taiji REQ model=%s msgs=%d tools=%s StrategyType=%s SessionID=%s wsid=%s reasoning_effort=%s',
        route.model,
        len(body.get('messages', [])),
        had_tools,
        headers.get('StrategyType', '-'),
        headers.get('SessionID', '-'),
        headers.get('wsid', '-'),
        reasoning_effort,
    )

    last_exc: Exception | None = None
    for attempt in range(TAIJI_MAX_RETRIES):
        try:
            if attempt > 0:
                payload['query_id'] = f'one_agent_infer_{uuid.uuid4()}'
            assembled = await _stream_and_collect(
                client,
                TAIJI_URL,
                payload,
                headers,
                TAIJI_TIMEOUT,
                had_tools=had_tools,
            )

            # 检测太极兜底响应：prompt_tokens=0 说明请求未到模型推理层
            usage = assembled.get('usage', {}) or {}
            prompt_tokens = usage.get('prompt_tokens', -1)
            choices = assembled.get('choices', [])
            first_msg = choices[0].get('message', {}) if choices else {}
            has_content = bool(first_msg.get('content'))
            has_tools = bool(first_msg.get('tool_calls'))

            if prompt_tokens == 0 and not has_content and not has_tools:
                if attempt < TAIJI_MAX_RETRIES - 1:
                    wait = TAIJI_INITIAL_BACKOFF * (2**attempt)
                    logger.warning(
                        'Taiji fallback detected (prompt_tokens=0, empty), retry %d/%d model=%s. Wait %.0fs',
                        attempt + 1,
                        TAIJI_MAX_RETRIES,
                        route.model,
                        wait,
                    )
                    await asyncio.sleep(wait)
                    continue
                logger.error(
                    'Taiji fallback persists after %d retries (prompt_tokens=0). model=%s.',
                    TAIJI_MAX_RETRIES,
                    route.model,
                )

            return assembled
        except httpx.HTTPStatusError as exc:
            if exc.response is not None and exc.response.status_code == 400:
                raise  # 参数错误，不重试
            last_exc = exc
        except httpx.TransportError as exc:
            last_exc = exc

        if attempt < TAIJI_MAX_RETRIES - 1:
            wait = TAIJI_INITIAL_BACKOFF * (2**attempt)
            logger.warning(
                'Taiji retry %d/%d model=%s: %s. Wait %.0fs',
                attempt + 1,
                TAIJI_MAX_RETRIES,
                route.model,
                last_exc,
                wait,
            )
            await asyncio.sleep(wait)

    raise last_exc  # type: ignore[misc]
