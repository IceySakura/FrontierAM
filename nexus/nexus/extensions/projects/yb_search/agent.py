"""Yb OneAgent - OpenAI-compatible tool-calling agent.

Development rules: see AGENTS.md (read before modifying).

Input/output aligned with the OpenAI Chat Completion API:
- Input: messages, tools, tool_choice (standard OpenAI format)
- Output: non-streaming ChatCompletion-style dict

Tool calls are executed via the tool_proxy HTTP service using a lightweight
JSON client proxy (see tool_proxy_client.py).
"""

import json
import logging
import time
import uuid

from enum import Enum
from typing import Any, cast

from pydantic import Field

from nexus.core import (
    AgentRunInput,
    AgentRunOutput,
    WithMainTrajectory,
    agent_registry,
    require_context,
)
from nexus.extensions.agents.base.utils import extract_text_from_content
from nexus.extensions.projects.yb_search.tool_proxy_client import (
    ToolProxyConfig,
)
from nexus.extensions.projects.yb_search.tool_proxy_dispatcher import (
    ToolProxyDispatcher,
)
from nexus.extensions.projects.yb_search.tools.context import (
    ToolProxyContextIn,
)


logger = logging.getLogger(__name__)


def _log_phase(agent_id: str, phase: str, detail: str = ''):
    """统一的追踪日志格式（对齐 AGTS agent）"""
    from datetime import datetime

    timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    detail_str = f' | {detail}' if detail else ''
    print(f'[{timestamp}][YB-OneAgent][{agent_id}] {phase}{detail_str}', flush=True)


# ====== Input/Output (OpenAI-compatible) ======


class YbOneAgentInput(AgentRunInput):
    """Input for YbOneAgent (OpenAI-compatible).

    Core fields (OpenAI Chat Completion API):
        messages: OpenAI-format message list (required, system prompt should be included here)
        tools: OpenAI-format tool schemas
        tool_choice: Tool choice strategy (auto/none/required)

    Agent control:
        max_iterations: Maximum agent loop iterations

    Tool proxy configuration:
        tool_proxy_service + tool_proxy_namespace: Polaris service discovery
    """

    agent_id: str = Field(default='', description='Agent 唯一标识，用于日志追踪')

    messages: list[dict[str, Any]] = Field(
        description='OpenAI-format message list (e.g. [{"role": "system", "content": "..."}, {"role": "user", "content": "hello"}]).',
    )
    tools: list[dict[str, Any]] | None = Field(
        default=None,
        description='OpenAI-format tool schemas (e.g. [{"type": "function", "function": {...}}])',
    )
    tool_choice: str | None = Field(
        default=None,
        description='Tool choice strategy for the LLM (e.g. "auto", "none", "required")',
    )
    max_iterations: int = Field(default=30, description='Maximum iterations')
    max_parallel_calls: int = Field(default=10, description='Maximum tool calls to execute per iteration')
    stop_before_tool_exec_on_max_iterations: bool = Field(
        default=True,
        description='When True, stop before executing tool calls on the final iteration (AGTS baseline behavior).',
    )
    extra_info: dict[str, Any] = Field(default_factory=dict, description='Compatibility field for AGTS task inputs')
    format_type: str = Field(default='', description='Compatibility field for AGTS task inputs')
    tool_versions: dict[str, str] = Field(
        default_factory=dict,
        description='Explicit tool name→version mapping for tool_proxy dispatch (e.g. {"web_search": "v1.0"})',
    )

    # Tool proxy config
    tool_proxy_service: str | None = Field(
        default=None,
        description='Polaris service name for the tool proxy',
    )
    tool_proxy_namespace: str = Field(
        default='Production',
        description='Polaris namespace for service discovery',
    )
    tool_proxy_metadata: dict[str, str] = Field(
        default_factory=dict,
        description='Polaris source service metadata for route rule matching',
    )
    tool_proxy_env: str | None = Field(
        default=None,
        description='Polaris env route metadata, equivalent to transport.env in the standalone tool_proxy HTTP kit',
    )
    tool_proxy_src_service: str | None = Field(
        default=None,
        description='Polaris source service key as "namespace:service" for rule routing',
    )
    tool_proxy_timeout: int = Field(default=30, description='Timeout in seconds for tool proxy calls')
    tool_proxy_auth: dict[str, Any] = Field(
        default_factory=dict,
        description='Optional interface_auth overrides; defaults to TOOL_PROXY_AK / TOOL_PROXY_SK environment variables',
    )
    tool_proxy_caller: str = Field(
        default='',
        description='Legacy caller service name kept for config compatibility; HTTP transport ignores it',
    )
    tool_proxy_context_in: 'ToolProxyContextIn' = Field(
        default_factory=lambda: ToolProxyContextIn(),
        description='Static context_in fields (user / session ids / caller) to fill in every /batch_tool_call body',
    )

    # Per-turn generation limit
    max_tokens_per_turn: int | None = Field(
        default=None,
        description=(
            'Per-turn max generation tokens limit, independent of trajectory budget. '
            'When set, each completion call will use min(remaining_trajectory_budget, '
            'max_tokens_per_turn) as max_tokens. This limits single-turn LLM output '
            'length without affecting trajectory-level token budget tracking or break logic.'
        ),
    )

    # Multimodal (VLM) support
    enable_multimodal: bool = Field(
        default=False,
        description=(
            'Whether to pass images from tool results as multimodal content parts '
            '(OpenAI vision format). When True, for_model entries with type="image" '
            'are converted to image_url content parts for VLM models. When False, '
            'image entries are dropped and only text is kept.'
        ),
    )


YbOneAgentInput.model_rebuild()


class YbOneAgentOutput(AgentRunOutput, WithMainTrajectory):
    """Output for YbOneAgent (OpenAI ChatCompletion-compatible).

    Fields:
        response: Non-streaming OpenAI ChatCompletion-style response dict
        final_answer: The final text answer extracted from the last assistant message
        main_trajectory: Full conversation trajectory in OpenAI-compatible format

    AGTS reward alignment fields:
        response_str: Full response text decoded from token ids (for reward calculation)
        tool_call_list: Per-turn raw_response strings from LLM (for query counting)
        tool_result_list: Per-turn concatenated tool result strings
        web_len/image_len/video_len/mini_len: Resource counts from tool results
        num_turns: Total turn count (user_turns + assistant_turns + 1)
        tool_call_count: Total number of successful tool calls
    """

    response: dict[str, Any] = Field(default_factory=dict)
    exit_status: 'ExitStatus' = Field(description='How the agent exited')
    final_answer: str = ''

    # AGTS reward alignment
    response_str: str = Field(default='', description='Full response token decode for reward calculation')
    tool_call_list: list[str] = Field(default_factory=list, description='Per-turn raw LLM output strings')
    tool_result_list: list[str] = Field(default_factory=list, description='Per-turn concatenated tool results')
    web_len: int = Field(default=0, description='Web URL count from tool results')
    image_len: int = Field(default=0, description='Image URL count from tool results')
    video_len: int = Field(default=0, description='Video URL count from tool results')
    mini_len: int = Field(default=0, description='Mini program URL count from tool results')
    num_turns: int = Field(default=0, description='Turn count (user_turns + iterations + 1)')
    tool_call_count: int = Field(default=0, description='Total tool call count')
    max_queries_per_turn: dict[str, int] = Field(
        default_factory=lambda: {'web_search': 0, 'picture_search': 0, 'video_search': 0},
        description='Maximum query count per search tool type in a single turn',
    )
    tool_timeout_count: int = Field(default=0, description='Tool timeout count')
    tool_error_count: int = Field(default=0, description='Tool error count excluding format_error')
    tool_status_counts: dict[str, int] = Field(default_factory=dict, description='Tool call status category counts')
    empty_tool_output_reasons: list[dict[str, Any]] = Field(
        default_factory=list, description='Diagnostics for empty tool outputs'
    )
    observation_time: float = Field(default=0.0, description='Total tool execution time in seconds')
    tool_rate_limit_wait_time: float = Field(default=0.0, description='Toolproxy rate-limit wait time in seconds')
    tool_rpc_time: float = Field(
        default=0.0, description='Actual tool RPC time (observation_time - rate_limit_wait_time)'
    )


class ExitStatus(str, Enum):
    COMPLETED = 'completed'
    MAX_ITERATIONS = 'max_iterations'
    TOOL_ERROR = 'tool_error'
    CONFIG_ERROR = 'config_error'
    ERROR = 'error'


# ====== Helper: build ToolProxyDispatcher from input config ======


def _merge_user_context(
    context_in: 'ToolProxyContextIn',
    task_user_context: dict[str, Any] | None,
) -> 'ToolProxyContextIn':
    """Overlay ``task_config.user_context`` onto ``context_in.user``.

    Task-level fields (``location`` / ``wx_openid``) take precedence when
    present; empty / missing values fall back to whatever the YAML agent
    config already set.
    """
    if not task_user_context:
        return context_in
    merged_user = context_in.user.model_copy(
        update={
            k: v
            for k, v in {
                'location': task_user_context.get('location'),
                'wx_openid': task_user_context.get('wx_openid', ''),
            }.items()
            if v not in (None, '')
        }
    )
    return context_in.model_copy(update={'user': merged_user})


def _build_dispatcher(
    agent_input: YbOneAgentInput,
    task_config: dict[str, Any] | None = None,
) -> ToolProxyDispatcher:
    """Create a ToolProxyDispatcher from the agent input configuration.

    ``task_config['user_context']`` is merged into the static ``user``
    slice of ``tool_proxy_context_in`` before the dispatcher is
    instantiated — this is where per-task identity fields (``location`` /
    ``wx_openid``) are threaded in.
    """
    config = ToolProxyConfig(
        service=agent_input.tool_proxy_service,
        namespace=agent_input.tool_proxy_namespace,
        metadata=agent_input.tool_proxy_metadata,
        env=agent_input.tool_proxy_env,
        src_service=agent_input.tool_proxy_src_service,
        timeout=agent_input.tool_proxy_timeout,
        caller=agent_input.tool_proxy_caller,
        auth=agent_input.tool_proxy_auth,
    )
    user_ctx = (task_config or {}).get('user_context') if task_config else None
    context_in = _merge_user_context(agent_input.tool_proxy_context_in, user_ctx)
    return ToolProxyDispatcher(config, context_in)


def _build_chat_completion(
    completion_id: str,
    model: str,
    message: dict[str, Any],
    finish_reason: str = 'stop',
    usage: dict[str, int] | None = None,
) -> dict[str, Any]:
    """Build a non-streaming OpenAI ChatCompletion response dict."""
    response: dict[str, Any] = {
        'id': completion_id,
        'object': 'chat.completion',
        'created': int(time.time()),
        'model': model,
        'choices': [
            {
                'index': 0,
                'message': message,
                'finish_reason': finish_reason,
            },
        ],
    }
    if usage:
        response['usage'] = usage
    return response


def _extract_tool_versions(tools: list[dict[str, Any]] | None) -> dict[str, str]:
    """Extract ``{tool_name: version}`` from OpenAI tool schemas."""
    versions: dict[str, str] = {}
    if not tools:
        return versions

    for tool in tools:
        if not isinstance(tool, dict):
            continue
        func = tool.get('function')
        if not isinstance(func, dict):
            continue

        name = func.get('name')
        if not isinstance(name, str) or not name:
            continue

        version = func.get('version')
        if version is None:
            continue

        version_str = str(version).strip()
        if version_str:
            versions[name] = version_str

    return versions


def _sanitize_tools_for_llm(
    tools: list[dict[str, Any]] | None,
) -> list[dict[str, Any]] | None:
    """Drop non-OpenAI extension keys (e.g. ``function.version``) before LLM call."""
    if not tools:
        return tools

    sanitized: list[dict[str, Any]] = []
    for tool in tools:
        if not isinstance(tool, dict):
            continue

        tool_copy = dict(tool)
        func = tool_copy.get('function')
        if isinstance(func, dict):
            func_copy = dict(func)
            func_copy.pop('version', None)
            tool_copy['function'] = func_copy

        sanitized.append(tool_copy)

    return sanitized


# ====== Reward Alignment Helpers ======


def _extract_tool_result_text(tool_msg: dict[str, Any]) -> str:
    """Extract plain text from a tool message's content (str or content-parts list)."""
    content = tool_msg.get('content', '')
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts: list[str] = []
        for part in content:
            if isinstance(part, dict) and part.get('type') == 'text':
                parts.append(part.get('text', ''))
            elif isinstance(part, str):
                parts.append(part)
        return '\n'.join(parts)
    return str(content) if content else ''


def _count_resource_markers(text: str) -> tuple[int, int, int, int]:
    """Count resource markers in tool result text.

    Supports both AGTS DocProcessor format and Tool Proxy format:
      - AGTS: [web title], [available title], [image description], [video title], [miniprogram name]
      - Tool Proxy: [web_doc N begin], [image_doc N begin], [video_doc N begin]

    Returns:
        (web_len, image_len, video_len, mini_len)
    """
    import re

    # AGTS DocProcessor format
    web = text.count('[web title]') + text.count('[available title]')
    image = text.count('[image description]')
    video = text.count('[video title]')
    mini = text.count('[miniprogram name]')

    # Tool Proxy formats seen in for_model payloads.
    web += len(re.findall(r'\[web_doc \d+ begin\]', text))
    web += text.count('[web_doc_content begin]')
    image += len(re.findall(r'\[image_doc \d+ begin\]', text))
    image += text.count('[web_image_description]')
    video += len(re.findall(r'\[video_doc \d+ begin\]', text))
    video += text.count('[web_video_description]')

    return web, image, video, mini


def _normalize_tool_name(name: str) -> str:
    """Normalize optional namespaced tool names for reward statistics."""
    return name.split('/')[-1]


def _count_queries_in_tool_calls(raw_tool_calls: list[dict[str, Any]]) -> dict[str, int]:
    """Count model-emitted search queries in one turn for AGTS reward discount."""
    counts: dict[str, int] = {'web_search': 0, 'picture_search': 0, 'video_search': 0}
    for tc in raw_tool_calls:
        func = tc.get('function', {}) or {}
        name = _normalize_tool_name(str(func.get('name') or ''))
        if name not in counts:
            continue
        try:
            arguments = func.get('arguments', '{}')
            args = json.loads(arguments) if isinstance(arguments, str) else arguments
            query = args.get('query') or args.get('querys')
            if isinstance(query, list):
                counts[name] += len([q for q in query if str(q).strip()])
            elif isinstance(query, str) and query.strip():
                counts[name] += len([q for q in query.split('｜｜') if q.strip()])
        except (json.JSONDecodeError, ValueError, AttributeError):
            counts[name] += 1
    return counts


def _increment_status(counts: dict[str, int], category: str, amount: int = 1) -> None:
    counts[category] = counts.get(category, 0) + amount


def _build_response_str(client: Any) -> str:
    """Build response_str from HyTunerClient's accumulated tokens.

    This is the full response portion of the trajectory (all LLM generations +
    tool results + special tokens decoded as text), used by AGTS reward judges.

    For non-HyTuner clients (e.g. inference), returns empty string.
    """
    if not (hasattr(client, 'last_tokens') and hasattr(client, 'tokenizer') and client.last_tokens is not None):
        return ''
    initial_prompt_len = getattr(client, '_initial_prompt_len', 0) or 0
    last_tokens = cast(list[int], client.last_tokens)
    if initial_prompt_len > 0 and len(last_tokens) > initial_prompt_len:
        tokenizer = client.tokenizer
        response_ids = last_tokens[initial_prompt_len:]
        return tokenizer.decode(response_ids, skip_special_tokens=True)
    return ''


# ====== Agent Implementation ======


@agent_registry.register('yb_one_agent_toolproxy')
async def run_yb_one_agent_toolproxy(agent_input: YbOneAgentInput) -> YbOneAgentOutput:
    """OpenAI-compatible tool-calling agent.

    Accepts OpenAI-format messages/tools/tool_choice as input, runs a tool-calling
    loop, and produces a non-streaming ChatCompletion-style response as output.
    """
    ctx = require_context()
    client = ctx.openai_client

    agent_id = agent_input.agent_id
    completion_id = f'chatcmpl-{uuid.uuid4().hex[:24]}'
    model = getattr(client, 'model', 'unknown')

    # Build message history (messages already includes system prompt if needed)
    history: list[dict[str, Any]] = list(agent_input.messages)

    _log_phase(agent_id, 'AGENT_START', f'max_iterations={agent_input.max_iterations}, messages={len(history)}')

    # Create tool proxy dispatcher (fail-fast on invalid startup config).
    # ``ctx.task_config['user_context']`` is merged in so task-level
    # identity fields (``location`` / ``wx_openid``) are surfaced to
    # tool_proxy even though they're not in ``agent_config``.
    dispatcher = _build_dispatcher(agent_input, task_config=ctx.task_config)
    if not dispatcher.is_configured:
        raise ValueError('tool_proxy_service must be set for yb_one_agent_toolproxy')

    final_answer = ''
    last_response = None
    exit_status: ExitStatus | None = None

    # Artifact context cursors: track numbering across iterations
    # so that Tool Proxy assigns sequential IDs for doc/image/video/file.

    # Visited URLs: track URLs visited across iterations for deduplication.

    # AGTS reward alignment: tracking variables
    tool_call_list: list[str] = []
    tool_result_list: list[str] = []
    tool_call_count = 0
    tool_error_count = 0
    tool_timeout_count = 0
    tool_status_counts: dict[str, int] = {}
    empty_tool_output_reasons: list[dict[str, Any]] = []
    max_queries_per_turn: dict[str, int] = {'web_search': 0, 'picture_search': 0, 'video_search': 0}
    web_len = 0
    image_len = 0
    video_len = 0
    mini_len = 0
    user_turns = 0
    iterations = 0
    total_observation_time = 0.0
    total_tool_rate_limit_wait_time = 0.0

    # Extract raw_query from the last user message in the initial messages.
    raw_query = ''
    for msg in reversed(agent_input.messages):
        if msg.get('role') == 'user':
            content = msg.get('content', '')
            if isinstance(content, list):
                text_parts = [p.get('text', '') for p in content if isinstance(p, dict) and p.get('type') == 'text']
                raw_query = '\n'.join(text_parts)
            elif isinstance(content, str):
                raw_query = content
            break

    # Optional per-tool version mapping. Prefer explicit tool_versions so data can
    # keep LLM tools empty when tool schemas are already written in the system prompt.
    tool_versions = dict(agent_input.tool_versions or {})
    tool_versions.update(_extract_tool_versions(agent_input.tools))
    llm_tools = _sanitize_tools_for_llm(agent_input.tools)

    for iteration in range(agent_input.max_iterations):
        iterations += 1

        # Build extra kwargs for LLM call
        completion_kwargs: dict[str, Any] = {}
        if llm_tools:
            completion_kwargs['tools'] = llm_tools
        if agent_input.tool_choice:
            completion_kwargs['tool_choice'] = agent_input.tool_choice
        if agent_input.max_tokens_per_turn is not None:
            completion_kwargs['max_tokens_per_turn'] = agent_input.max_tokens_per_turn

        last_response = await client.create_completion(messages=history, **completion_kwargs)
        message = last_response.choices[0].message
        content_text = extract_text_from_content(message.content)

        # Capture raw_response for reward alignment (HyTunerClient sets this on ExtendedMessage).
        # model_dump(exclude_none=True) preserves raw_response when present.
        raw_response = getattr(message, 'raw_response', '') or ''
        history.append(message.model_dump(exclude_none=True))

        if content_text:
            final_answer = content_text

        # Debug: print LLM output preview
        _raw_preview = (raw_response or content_text or '')[:200]
        _tool_calls_preview = [
            {'name': tc.function.name, 'args': str(tc.function.arguments)[:100]} for tc in (message.tool_calls or [])
        ]
        print(
            f'[DBG-AGENT {agent_id} iter={iterations}/{agent_input.max_iterations}] llm_out={repr(_raw_preview)} tool_calls={_tool_calls_preview}',
            flush=True,
        )

        # No tool calls -> done
        if not message.tool_calls:
            exit_status = ExitStatus.COMPLETED

            # Count resource markers in tool results for the no-tool-call exit
            # (aligned with AGTS agent.py lines 436-443)
            for msg in history:
                if msg.get('role') == 'tool' and 'content' in msg:
                    text = _extract_tool_result_text(msg)
                    w, i, v, m = _count_resource_markers(text)
                    web_len += w
                    image_len += i
                    video_len += v
                    mini_len += m

            break

        # Record per-turn raw LLM output for reward query counting
        tool_call_list.append(raw_response)

        # Convert FunctionCall format to OpenAI tool_call format expected by tool_proxy_client
        raw_tool_calls = []
        for tc in message.tool_calls:
            tc_dict = tc.model_dump(exclude_none=True)
            # If already OpenAI format (has 'function' key), use as-is
            # Otherwise convert FunctionCall format {name, arguments} to {id, function: {name, arguments}}
            if 'function' not in tc_dict:
                tc_dict = {
                    'id': tc_dict.get('id', ''),
                    'function': {
                        'name': tc_dict.get('name', ''),
                        'arguments': tc_dict.get('arguments', '{}'),
                    },
                }
            raw_tool_calls.append(tc_dict)

        turn_query_counts = _count_queries_in_tool_calls(raw_tool_calls)
        for name, count in turn_query_counts.items():
            max_queries_per_turn[name] = max(max_queries_per_turn.get(name, 0), count)

        if agent_input.stop_before_tool_exec_on_max_iterations and iterations >= agent_input.max_iterations:
            exit_status = ExitStatus.MAX_ITERATIONS
            _log_phase(agent_id, 'MAX_ITERATIONS', f'reached max_iterations before tool exec, step={iterations}')
            break

        max_parallel_calls = agent_input.max_parallel_calls
        if max_parallel_calls is not None and max_parallel_calls > 0:
            truncated = len(raw_tool_calls) > max_parallel_calls
            raw_tool_calls_to_execute = raw_tool_calls[:max_parallel_calls] if truncated else raw_tool_calls
            raw_tool_calls_skipped = raw_tool_calls[max_parallel_calls:] if truncated else []
        else:
            raw_tool_calls_to_execute = raw_tool_calls
            raw_tool_calls_skipped = []
        if raw_tool_calls_skipped:
            _increment_status(tool_status_counts, 'skipped', len(raw_tool_calls_skipped))
            logger.warning(
                '[%s] too many tool calls: %d, executing first %d',
                agent_id,
                len(raw_tool_calls),
                max_parallel_calls,
            )

        # Execute tool calls via the tool proxy client
        _log_phase(agent_id, 'TOOL_EXEC_START', f'executing {len(raw_tool_calls_to_execute)} tool calls via proxy')
        tool_exec_start = time.monotonic()
        tool_messages: list[dict[str, Any]] = []
        if raw_tool_calls_to_execute:
            try:
                tool_messages, _stats = await dispatcher.execute_tool_calls(
                    raw_tool_calls_to_execute,
                    history=history,
                    raw_query=raw_query,
                    round_idx=iteration,
                    enable_multimodal=agent_input.enable_multimodal,
                    agent_id=agent_id,
                    tool_versions=tool_versions,
                )
                tool_call_count += len(raw_tool_calls_to_execute)
                total_tool_rate_limit_wait_time += _stats.rate_limit_wait_ms / 1000
                for call_status in _stats.call_statuses:
                    category = call_status.category
                    _increment_status(tool_status_counts, category)
                    if category == 'timeout':
                        tool_error_count += 1
                        tool_timeout_count += 1
                    elif category in {
                        'error',
                        'transport_error',
                        'config_error',
                        'tool_not_found',
                        'rate_limited',
                        'partial_success',
                    }:
                        tool_error_count += 1
                    if category == 'empty_result' or call_status.content_empty:
                        empty_tool_output_reasons.append({
                            'tool_id': call_status.call_id,
                            'tool_name': call_status.name,
                            'status_code': call_status.status_code,
                            'status_msg': call_status.status_msg,
                            'category': category,
                        })
                # Session cursors/visited_urls are owned by dispatcher.session_state.
            except Exception as e:
                elapsed = time.monotonic() - tool_exec_start
                is_timeout = 'timeout' in str(e).lower() or 'timeout' in type(e).__name__.lower()
                tag = 'TOOL_TIMEOUT' if is_timeout else 'TOOL_ERROR'
                category = 'timeout' if is_timeout else 'transport_error'
                _log_phase(agent_id, tag, f'iter={iterations}, elapsed={elapsed:.1f}s, {type(e).__name__}: {e}')
                exit_status = ExitStatus.TOOL_ERROR
                _increment_status(tool_status_counts, category, len(raw_tool_calls_to_execute))
                tool_error_count += len(raw_tool_calls_to_execute)
                if is_timeout:
                    tool_timeout_count += len(raw_tool_calls_to_execute)
                tool_messages = [
                    {
                        'role': 'tool',
                        'tool_call_id': tc.get('id', ''),
                        'content': f'Tool proxy call failed: {type(e).__name__}: {e}',
                    }
                    for tc in raw_tool_calls_to_execute
                ]
        total_observation_time += time.monotonic() - tool_exec_start

        for tc in raw_tool_calls_skipped:
            tool_messages.append({
                'role': 'tool',
                'tool_call_id': tc.get('id', ''),
                'content': f'工具调用数量超过{max_parallel_calls},跳过工具执行.',
            })

        # Collect per-turn tool result text for reward alignment
        turn_result_parts: list[str] = []
        for tool_msg in tool_messages:
            history.append(tool_msg)
            turn_result_parts.append(_extract_tool_result_text(tool_msg))
        tool_result_list.append('\n'.join(turn_result_parts))
        user_turns += 1

        # Debug: print tool result preview
        _result_preview = '\n'.join(turn_result_parts)[:300]
        _log_phase(
            agent_id,
            'TOOL_EXEC_DONE',
            f'executed {len(raw_tool_calls_to_execute)} tool calls, skipped {len(raw_tool_calls_skipped)}, result_preview={repr(_result_preview)}',
        )
    else:
        exit_status = ExitStatus.MAX_ITERATIONS
        _log_phase(
            agent_id,
            'MAX_ITERATIONS',
            f'reached max_iterations={agent_input.max_iterations}, tool_calls={tool_call_count}',
        )

        # Count resource markers when max_iterations exhausted (same as COMPLETED branch)
        for msg in history:
            if msg.get('role') == 'tool' and 'content' in msg:
                text = _extract_tool_result_text(msg)
                w, i, v, m = _count_resource_markers(text)
                web_len += w
                image_len += i
                video_len += v
                mini_len += m

    if exit_status is None:
        exit_status = ExitStatus.ERROR

    if web_len == 0 and image_len == 0 and video_len == 0 and mini_len == 0:
        for msg in history:
            if msg.get('role') == 'tool' and 'content' in msg:
                text = _extract_tool_result_text(msg)
                w, i, v, m = _count_resource_markers(text)
                web_len += w
                image_len += i
                video_len += v
                mini_len += m

    # Build final OpenAI ChatCompletion response
    # Use the last assistant message from history as the response message
    last_assistant_msg: dict[str, Any] = {'role': 'assistant', 'content': final_answer}
    finish_reason = 'stop'

    # Find the last assistant message in history for accurate output (may include tool_calls)
    for msg in reversed(history):
        if msg.get('role') == 'assistant':
            last_assistant_msg = msg
            if msg.get('tool_calls'):
                finish_reason = 'tool_calls'
            break

    # Build usage
    usage: dict[str, int] | None = None
    if last_response and hasattr(last_response, 'usage') and last_response.usage:
        usage = {
            'prompt_tokens': last_response.usage.prompt_tokens or 0,
            'completion_tokens': last_response.usage.completion_tokens or 0,
            'total_tokens': last_response.usage.total_tokens or 0,
        }

    response = _build_chat_completion(
        completion_id=completion_id,
        model=model,
        message=last_assistant_msg,
        finish_reason=finish_reason,
        usage=usage,
    )

    # Build response_str from HyTunerClient token accumulation (same logic as AGTS agent)
    response_str = _build_response_str(client)

    total_tool_rpc_time = total_observation_time - total_tool_rate_limit_wait_time

    _log_phase(
        agent_id,
        'AGENT_DONE',
        f'exit={exit_status.value}, iters={iterations}, tool_calls={tool_call_count}, '
        f'tool_errors={tool_error_count}, tool_status={tool_status_counts}, '
        f'obs_time={total_observation_time:.1f}s, rate_limit_wait={total_tool_rate_limit_wait_time:.1f}s, '
        f'tool_rpc_time={total_tool_rpc_time:.1f}s',
    )

    return YbOneAgentOutput(
        main_trajectory=history,
        response=response,
        exit_status=exit_status,
        final_answer=final_answer,
        response_str=response_str,
        tool_call_list=tool_call_list,
        tool_result_list=tool_result_list,
        web_len=web_len,
        image_len=image_len,
        video_len=video_len,
        mini_len=mini_len,
        num_turns=user_turns + iterations + 1,
        tool_call_count=tool_call_count,
        max_queries_per_turn=max_queries_per_turn,
        tool_timeout_count=tool_timeout_count,
        tool_error_count=tool_error_count,
        tool_status_counts=tool_status_counts,
        empty_tool_output_reasons=empty_tool_output_reasons,
        observation_time=total_observation_time,
        tool_rate_limit_wait_time=total_tool_rate_limit_wait_time,
        tool_rpc_time=total_tool_rpc_time,
    )
