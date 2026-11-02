"""aidata rubric 评估的状态收集 + 摘要。

状态完全通过 R3 收集——每个 mock 暴露一个
``<name>_dump_state()`` MCP 工具，将其完整状态以带 rubric 标准键的 dict 形式返回。
``download_state`` 会发现活跃 MCP 会话上所有此类工具，逐一调用，并将结果合并到：

* 以 MCP server 名称作键的命名空间桶（``state['wechat']``），以及
* 遗留的扁平 rubric 切片（``state['messages']`` / ``state['emails']``）。

``build_summary`` 从已收集的 state dict 渲染出紧凑的每次运行快照，供 UI /
result.json 使用——不需要每个 mock 的专属知识。
"""

from __future__ import annotations

import asyncio
import copy
import json
import logging
import os

from typing import Any

from nexus.core import require_context


logger = logging.getLogger(__name__)


def _dump_state_max_attempts() -> int:
    raw = os.environ.get('AIDATA_DUMP_STATE_MAX_ATTEMPTS') or os.environ.get('AIDATA_DUMP_STATE_RETRIES')
    if raw is None:
        return 3
    try:
        return max(1, int(raw))
    except ValueError:
        logger.warning('[aidata] invalid AIDATA_DUMP_STATE_MAX_ATTEMPTS=%r; using 3', raw)
        return 3


def _dump_state_retry_backoff_s() -> float:
    raw = os.environ.get('AIDATA_DUMP_STATE_RETRY_BACKOFF_S')
    if raw is None:
        return 1.0
    try:
        return max(0.0, float(raw))
    except ValueError:
        logger.warning('[aidata] invalid AIDATA_DUMP_STATE_RETRY_BACKOFF_S=%r; using 1.0', raw)
        return 1.0


# ---------------------------------------------------------------------------
# 最终答案抽取
# ---------------------------------------------------------------------------


_FINISH_SENTINEL_NAMES: frozenset[str] = frozenset({'finish'})


def _build_tool_call_id_map(trajectory: list) -> dict[str, str]:
    """从 assistant 消息中构建 tool_call_id → tool 函数名的映射。"""
    id_to_name: dict[str, str] = {}
    for msg in trajectory:
        if msg.get('role') != 'assistant':
            continue
        for tc in msg.get('tool_calls') or []:
            tc_id = tc.get('id') or ''
            fn_name = (tc.get('function') or {}).get('name') or ''
            if tc_id and fn_name:
                id_to_name[tc_id] = fn_name
    return id_to_name


def extract_final_answer(state: dict[str, Any]) -> str:
    """返回 agent 的最终自然语言回复。

    解析顺序（首个非空者胜出）：
      1. 预填充键（agent_answer / final_answer / submit_response）
      2. 最后一条 finish sentinel 工具响应（ToolCallAgent 约定）
      3. 最后一条 submit_response 工具调用参数（HY3.0 约定）
      4. 最后一条 assistant 消息内容
    """
    for key in ('agent_answer', 'final_answer', 'submit_response'):
        val = state.get(key)
        if isinstance(val, str) and val.strip():
            return val

    trajectory = state.get('trajectory', []) or []

    # 构建 tool_call_id → 函数名的查找表，用于解析缺失 ``name`` 字段的
    # 工具响应（标准 OpenAI Chat Completions 格式仅携带 ``tool_call_id``，
    # 而不携带 ``name``）。
    tc_id_map = _build_tool_call_id_map(trajectory)

    for msg in reversed(trajectory):
        if msg.get('role') != 'tool':
            continue
        name = msg.get('name') or ''
        if not name:
            name = tc_id_map.get(msg.get('tool_call_id') or '', '')
        is_finish = name in _FINISH_SENTINEL_NAMES or any(name.endswith('_' + s) for s in _FINISH_SENTINEL_NAMES)
        if not is_finish:
            continue
        content = msg.get('content')
        if isinstance(content, str) and content.strip():
            return content
        if isinstance(content, list):
            parts = [c.get('text', '') for c in content if isinstance(c, dict) and c.get('type') in (None, 'text')]
            joined = ''.join(p for p in parts if isinstance(p, str))
            if joined.strip():
                return joined

    from_submit = ''
    from_finish_arg = ''
    tail_assistant = ''
    for msg in trajectory:
        if msg.get('role') != 'assistant':
            continue
        content = msg.get('content')
        if isinstance(content, str) and content.strip():
            tail_assistant = content
        for tc in msg.get('tool_calls') or []:
            fn = tc.get('function') or {}
            name = fn.get('name') or ''
            is_submit = name == 'submit_response' or name.endswith('_submit_response')
            is_finish = name in _FINISH_SENTINEL_NAMES or any(name.endswith('_' + s) for s in _FINISH_SENTINEL_NAMES)
            if not is_submit and not is_finish:
                continue
            raw = fn.get('arguments', '')
            if isinstance(raw, str):
                try:
                    parsed = json.loads(raw) if raw else {}
                except json.JSONDecodeError:
                    parsed = {}
            else:
                parsed = raw if isinstance(raw, dict) else {}
            if is_submit:
                text = parsed.get('text') or parsed.get('response') or ''
                if isinstance(text, str) and text.strip():
                    from_submit = text
            elif is_finish:
                text = parsed.get('message') or parsed.get('text') or ''
                if isinstance(text, str) and text.strip():
                    from_finish_arg = text
    return from_submit or from_finish_arg or tail_assistant


# ---------------------------------------------------------------------------
# 紧凑摘要（通用——仅读取 rubric 标准键）
# ---------------------------------------------------------------------------


def build_summary(state: dict[str, Any]) -> dict[str, Any]:
    """最终运行状态的紧凑、UI 友好快照。

    通用实现：迭代 state dict 的键，对 list 报告数量，对 dict 报告键数。
    不涉及每个 mock 的专属知识——只要新 mock 的 dump_state 返回标准键，
    就会自动出现在摘要中。
    """
    summary: dict[str, Any] = {}
    skip = {'trajectory', 'agent_answer'}
    for key, val in state.items():
        if key in skip:
            continue
        if isinstance(val, (list, dict)):
            summary[key] = len(val)
    return summary


# ---------------------------------------------------------------------------
# 状态下载——基于 R3 MCP（唯一路径）
# ---------------------------------------------------------------------------


async def download_state(
    runtime: Any,
    *,
    services: list | None = None,
    dump_state_args: dict[str, dict[str, Any]] | None = None,
) -> dict[str, Any]:
    """通过 R3 MCP dump_state 工具收集 mock 状态。

    发现活跃 MCP 会话上所有 ``*_dump_state`` 工具，逐一调用，
    并将返回的 rubric 标准键合并到单一结果 dict。 无文件系统访问，
    也没有针对每个 mock 的硬编码。

    顶层结果同时包含命名空间桶和遗留扁平切片。 新 rubric 应优先采用
    命名空间形式（``from: xhs`` + ``path: actions.0.type``）；扁平形式
    仅为兼容既有 case 而保留。

    ``dump_state_args`` 可选地将某个 MCP 命名空间（例如 ``pdd`` 或
    ``taobao``）映射到传入该 mock ``_dump_state`` 工具的参数。 未列出的
    命名空间保持 mock 默认行为，即以 ``_dump_state({})`` 调用。

    ``runtime`` 和 ``services`` 仅为 API 兼容而接受；状态本身通过 MCP
    获取。
    """
    result: dict[str, Any] = {}

    ctx = require_context()
    if ctx._mcp_manager is not None:
        await _collect_state_via_mcp(ctx, result, dump_state_args=dump_state_args)

    return result


async def _collect_state_via_mcp(
    ctx: Any,
    result: dict[str, Any],
    *,
    dump_state_args: dict[str, dict[str, Any]] | None = None,
) -> None:
    """调用每个 mock 的 _dump_state 工具并合并到 result。"""
    from nexus.core.tool.mcp.manager import MCPToolManager

    mgr: MCPToolManager = ctx._mcp_manager

    dump_tools = {name: tool for name, tool in mgr._tools.items() if name.endswith('_dump_state')}
    if not dump_tools:
        return

    logger.debug(
        '[aidata] R3 state readback: found %d dump_state tools: %s', len(dump_tools), sorted(dump_tools.keys())
    )

    reserved_namespaces = {tool.server_name for tool in dump_tools.values() if getattr(tool, 'server_name', None)}

    for full_name, _tool in dump_tools.items():
        namespace = getattr(_tool, 'server_name', '') or _namespace_from_tool_name(full_name)
        call_args = _dump_state_call_args(
            namespace=namespace,
            full_name=full_name,
            dump_state_args=dump_state_args,
        )
        try:
            data, attempts = await _call_dump_state_with_retries(mgr, full_name, call_args)
            bucket = result.setdefault(namespace, {})
            if isinstance(bucket, dict):
                _merge_dump_state(data, bucket)
            else:
                logger.warning(
                    '[aidata] R3 namespace collision for %s: existing value is %s',
                    namespace,
                    type(bucket).__name__,
                )

            # 为直接读取扁平切片的既有 rubric 保留后向兼容。
            # MCP server 名称已被保留，故当多个 mock 同时活跃时，
            # 扁平键也不会覆盖命名空间桶。
            _merge_dump_state(data, result, skip_keys=reserved_namespaces)
            logger.debug(
                '[aidata] R3 %s: merged %s under namespace=%s args=%s',
                full_name,
                sorted(data.keys()),
                namespace,
                sorted(call_args.keys()) if call_args else [],
            )
            if attempts > 1:
                logger.debug('[aidata] R3 %s succeeded after %d dump_state attempt(s)', full_name, attempts)
        except Exception as exc:
            error_info = _dump_state_error_info(exc)
            result.setdefault('_dump_state_errors', {})[namespace] = {
                'tool': full_name,
                'args': sorted(call_args.keys()) if call_args else [],
                'attempts': _dump_state_max_attempts(),
                **error_info,
            }
            logger.warning(
                '[aidata] R3 %s failed after retries: kind=%s error=%s hint=%s',
                full_name,
                error_info['kind'],
                error_info['error'],
                error_info['hint'],
            )


async def _call_dump_state_with_retries(
    mgr: Any,
    full_name: str,
    call_args: dict[str, Any],
) -> tuple[dict[str, Any], int]:
    """调用一个 dump_state 工具并解析其 JSON 载荷，带重试。

    R3 回读失败往往是传输层/载荷体积的抖动，而非业务状态失败。
    在此处重试可避免单次瞬时错误在后续 rubric SQL 物化时静默变成
    ``missing_slice``。
    """
    max_attempts = _dump_state_max_attempts()
    backoff_s = _dump_state_retry_backoff_s()
    last_exc: Exception | None = None
    for attempt in range(1, max_attempts + 1):
        try:
            resp = await mgr.call_tool(full_name, call_args)
            text = _dump_state_text_from_response(resp)
            if not text:
                raise ValueError('empty dump_state response')
            data = json.loads(text)
            if not isinstance(data, dict):
                raise ValueError(f'non-object dump_state payload: {type(data).__name__}')
            return data, attempt
        except Exception as exc:
            last_exc = exc
            if attempt >= max_attempts:
                break
            delay = backoff_s * (2 ** (attempt - 1))
            error_info = _dump_state_error_info(exc)
            logger.warning(
                '[aidata] R3 %s dump_state attempt %d/%d failed: kind=%s error=%s; retrying in %.2fs',
                full_name,
                attempt,
                max_attempts,
                error_info['kind'],
                error_info['error'],
                delay,
            )
            if delay:
                await asyncio.sleep(delay)
    assert last_exc is not None
    raise last_exc


def _dump_state_text_from_response(resp: Any) -> str:
    for part in getattr(resp, 'output', []) or []:
        if isinstance(part, dict) and part.get('type') == 'text':
            return part.get('text', '') or ''
        if hasattr(part, 'text'):
            return part.text or ''
    return ''


def _dump_state_error_info(exc: Exception) -> dict[str, str]:
    """对 dump_state 失败进行分类，用于 result.json 调试。

    大多数 MCP/客户端异常并未暴露稳定的公共错误层次，因此分类器结合了
    Python 异常类型与保守的消息匹配。 原始消息仍被完整保留。
    """
    error = str(exc) or exc.__class__.__name__
    text = error.lower()
    exc_name = exc.__class__.__name__
    if isinstance(exc, TimeoutError) or 'timeout' in text or 'timed out' in text:
        return {
            'kind': 'timeout',
            'error_type': exc_name,
            'error': error,
            'hint': 'dump_state call timed out; reduce requested tables or increase MCP/tool timeout if available',
        }
    if any(
        marker in text
        for marker in (
            'too large',
            'payload',
            'response size',
            'content length',
            'max content',
            'maximum context',
            'request entity too large',
            '413',
        )
    ):
        return {
            'kind': 'payload_too_large',
            'error_type': exc_name,
            'error': error,
            'hint': 'dump_state payload is likely too large; switch to order-scoped tables or fewer dump_state tables',
        }
    if isinstance(exc, json.JSONDecodeError) or 'json' in text or 'expecting value' in text:
        return {
            'kind': 'json_decode',
            'error_type': exc_name,
            'error': error,
            'hint': 'dump_state response was not valid JSON; inspect mock stderr/logs for partial or truncated output',
        }
    if 'empty dump_state response' in text:
        return {
            'kind': 'empty_response',
            'error_type': exc_name,
            'error': error,
            'hint': 'dump_state returned no text content; inspect MCP server readiness and tool response format',
        }
    if any(
        marker in text
        for marker in (
            'connection',
            'connect',
            'network',
            'transport',
            'broken pipe',
            'connection reset',
            'connection refused',
            'closed',
            'http',
            '503',
            '502',
            '504',
        )
    ):
        return {
            'kind': 'network',
            'error_type': exc_name,
            'error': error,
            'hint': 'dump_state failed in transport; retry may help, otherwise inspect sidecar/runtime network logs',
        }
    return {
        'kind': 'other',
        'error_type': exc_name,
        'error': error,
        'hint': 'inspect dump_state tool logs and requested dump_state args',
    }


def _dump_state_call_args(
    *,
    namespace: str,
    full_name: str,
    dump_state_args: dict[str, dict[str, Any]] | None,
) -> dict[str, Any]:
    """为一个工具解析可选的每 mock dump_state 参数。

    主键是 MCP server 命名空间（``pdd``）。 出于便捷，调用方也可通过完整工具名
    （``pdd__dump_state``）或从工具名尽力解析出的命名空间作为键。
    """
    if not dump_state_args:
        return {}
    for key in (namespace, full_name, _namespace_from_tool_name(full_name)):
        raw = dump_state_args.get(key)
        if isinstance(raw, dict):
            return dict(raw)
    return {}


def _dump_state_call_args(
    *,
    namespace: str,
    full_name: str,
    dump_state_args: dict[str, dict[str, Any]] | None,
) -> dict[str, Any]:
    """为一个工具解析可选的每 mock dump_state 参数。

    主键是 MCP server 命名空间（``pdd``）。 出于便捷，调用方也可通过完整工具名
    （``pdd__dump_state``）或从工具名尽力解析出的命名空间作为键。
    """
    if not dump_state_args:
        return {}
    for key in (namespace, full_name, _namespace_from_tool_name(full_name)):
        raw = dump_state_args.get(key)
        if isinstance(raw, dict):
            return dict(raw)
    return {}


def _namespace_from_tool_name(full_name: str) -> str:
    """针对非标准 ConnectedMCPTool stub 的尽力而为命名空间回退。"""
    suffix = '_dump_state'
    base = full_name[: -len(suffix)] if full_name.endswith(suffix) else full_name
    return base.rsplit('_', 1)[0] if '_' in base else base


def _merge_dump_state(data: dict, result: dict[str, Any], *, skip_keys: set[str] | None = None) -> None:
    """将一个 mock 的 dump_state 输出合并到 result。

    通用类型感知合并——首次遇到时创建键，后续遇到时扩展/更新。
    无需预先声明的键注册表。
    """
    skip_keys = skip_keys or set()
    for key, val in data.items():
        if key in skip_keys:
            continue
        if key not in result:
            result[key] = copy.deepcopy(val)
        elif isinstance(result[key], list) and isinstance(val, list):
            result[key].extend(val)
        elif isinstance(result[key], dict) and isinstance(val, dict):
            result[key].update(val)


async def collect_crontab_state(runtime: Any) -> list[dict[str, Any]]:
    """在沙箱中运行 ``crontab -l`` 并解析为结构化条目。

    返回一个 dict 列表，标准 5 字段 crontab 行会带上 ``schedule``、``command``
    和 ``raw`` 键。 非标准行仅保留 ``raw``。 当 crontab 为空、缺失或命令失败时
    返回 ``[]``。
    """
    try:
        res = await runtime.run_command('crontab -l 2>/dev/null', timeout=10)
    except Exception:
        return []
    stdout = getattr(res, 'stdout', '') or ''
    exit_code = getattr(res, 'return_code', None)
    if exit_code is None:
        exit_code = getattr(res, 'exit_code', None)
    if exit_code != 0 or not stdout.strip():
        return []
    entries: list[dict[str, Any]] = []
    for line in stdout.strip().splitlines():
        line = line.strip()
        if not line or line.startswith('#'):
            continue
        parts = line.split(None, 5)
        if len(parts) >= 6:
            entries.append({
                'schedule': ' '.join(parts[:5]),
                'command': parts[5],
                'raw': line,
            })
        else:
            entries.append({'raw': line})
    return entries


__all__ = ['build_summary', 'collect_crontab_state', 'download_state', 'extract_final_answer']
