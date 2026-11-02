"""one_agent_infer — 推理/评测 task（原 yb_one_agent_bench）。

处理 YB OneAgentBench 类型的数据集：带文件上传、多轮 session 与文件下载。

Supports both single-turn and multi-turn session evaluation:

Single-turn workflow (V0.2, CI_Query):
1. Upload input files → build doc_blocks → run agent → download files.

Multi-turn workflow (V0.3 sessions):
1. Upload turn-1 files → build doc_blocks → run agent → download files.
2. Extract answer from agent response.
3. Append (user_prompt, answer) to history.
4. Upload turn-2 files → build doc_blocks → run agent with history → download files.
5. Repeat for all turns — same sandbox throughout.

The generator groups same-session records into one BatchItem with a `turns` list.
"""

import base64
import json
import logging
import os
import re

from dataclasses import dataclass, field
from pathlib import Path

from nexus.core import (
    TaskRunInput,
    TaskRunOutput,
    WithAgentRunner,
    require_context,
    task_registry,
)
from nexus.extensions.agents.yb_one_agent import parse_cache
from nexus.extensions.agents.yb_one_agent.agent import is_sandbox_infrastructure_error
from nexus.extensions.agents.yb_one_agent.get_file_preview import (
    IMAGE_SUPPORTED,
    UPLOAD_MAX_BYTES,
    UPLOAD_SUPPORTED,
    build_doc_block,
    build_image_block,
    get_file_preview_from_bytes,
)
from nexus.extensions.agents.yb_one_agent.llm_errors import fatal_llm_exit_error, is_fatal_llm_exit, is_llm_http_error
from nexus.extensions.agents.yb_one_agent.prompt_builder import DEFAULT_TOOLS
from nexus.runtime.executors import FileItem


logger = logging.getLogger(__name__)

# ── Per-turn incremental output persistence ────────────────────────────
# Multi-turn sessions may be interrupted by an outer timeout (asyncio.wait_for).
# To avoid losing already-completed turns, we write trajectory and files to disk
# immediately after each turn finishes. This way, even if a later turn times out,
# the evaluation stage can find earlier turns' outputs in the filesystem.
_INCREMENTAL_OUTPUT_ENABLED = os.environ.get('NEXUS_INCREMENTAL_OUTPUT', '1') == '1'
_HANDOFF_MODES = frozenset({'disabled', 'offline', 'online'})


def _get_handoff_mode() -> str:
    """Return the configured hybrid routing protocol.

    ``disabled`` preserves the standard harness behavior and tool set.
    ``offline`` preserves the original benchmark behavior: stop the
    A3B session on handoff and let the outer evaluator fill this and later
    turns from an A20B baseline. ``online`` runs A20B inline for only
    the handed-off turn, on the same sandbox.
    """
    mode = os.environ.get('ONE_AGENT_HANDOFF_MODE', 'disabled').strip().lower()
    if mode not in _HANDOFF_MODES:
        choices = ', '.join(sorted(_HANDOFF_MODES))
        raise ValueError(f'Invalid ONE_AGENT_HANDOFF_MODE={mode!r}; expected one of: {choices}')
    return mode


def _a3b_tools_for_mode(handoff_mode: str) -> list[str] | None:
    """Return router tools for handoff modes, or None for harness defaults."""
    if handoff_mode == 'disabled':
        return None
    return sorted((set(DEFAULT_TOOLS) - {'delegate'}) | {'handoff'})


def _exit_status_value(agent_output) -> str | None:
    exit_status = getattr(agent_output, 'exit_status', None)
    return getattr(exit_status, 'value', exit_status)


def _persist_turn_output(
    turn_task_id: str,
    trajectory_msgs: list[dict],
    downloaded_files: dict[str, str],
    tools: list,
    output_base_dir: str | None,
    metadata: dict | None = None,
) -> None:
    """Write a completed turn's trajectory and files to disk immediately.

    Args:
        turn_task_id: e.g. "Doc-Synthesis-012" or "Doc-Synthesis-012_t2"
        trajectory_msgs: full cumulative trajectory (list of message dicts)
        downloaded_files: {filename: base64_content}
        tools: tool definitions list
        output_base_dir: base directory (parent of files/ and trajectory/)
        metadata: optional metadata dict to attach to trajectory
    """
    if not output_base_dir or not _INCREMENTAL_OUTPUT_ENABLED:
        return
    try:
        import json as _json

        # Write trajectory file
        traj_dir = os.path.join(output_base_dir, 'trajectory')
        os.makedirs(traj_dir, exist_ok=True)
        traj_data: dict = {'trajectory': trajectory_msgs, 'tools': tools}
        if metadata:
            traj_data['metadata'] = metadata
        traj_path = os.path.join(traj_dir, f'{turn_task_id}.jsonl')
        with open(traj_path, 'w', encoding='utf-8') as f:
            f.write(_json.dumps(traj_data, ensure_ascii=False) + '\n')

        # Write downloaded files
        if downloaded_files:
            import base64

            files_dir = os.path.join(output_base_dir, 'files', turn_task_id)
            os.makedirs(files_dir, exist_ok=True)
            for filename, b64content in downloaded_files.items():
                fpath = os.path.join(files_dir, filename)
                os.makedirs(os.path.dirname(fpath), exist_ok=True)
                with open(fpath, 'wb') as f:
                    f.write(base64.b64decode(b64content))

        logger.info(
            f'[{turn_task_id}] Persisted turn output: '
            f'trajectory={len(trajectory_msgs)} msgs, '
            f'files={len(downloaded_files)}'
        )
    except Exception as e:
        logger.warning(f'[{turn_task_id}] Failed to persist turn output: {e}')


# ════════════════════════════════════════════════════════════════════════
# Trajectory & answer extraction
# ════════════════════════════════════════════════════════════════════════


def _normalize_msg(msg) -> dict:
    if isinstance(msg, dict):
        return msg
    if hasattr(msg, 'model_dump'):
        return msg.model_dump()
    if hasattr(msg, 'dict'):
        return msg.dict()
    return dict(vars(msg)) if hasattr(msg, '__dict__') else {'_raw': str(msg)}


# search 类工具：其 tool 返回结果通常很长，多轮历史里可按需置空以省 token
SEARCH_TOOL_NAMES = frozenset({'web_search', 'picture_search', 'video_search'})


def _extract_turn_full_messages(agent_output, skip_count: int) -> list[dict]:
    """Extract the current turn's non-system messages (full structure).

    ``agent_output.main_trajectory`` includes messages from previous turns
    (passed in as history context). We skip the first ``skip_count`` non-system
    messages (previous turns) and keep only the current turn's messages,
    preserving ``content`` / ``tool_calls`` / ``tool_call_id``.
    """
    raw_history: list = []
    if hasattr(agent_output, 'main_trajectory') and agent_output.main_trajectory:
        raw_history = agent_output.main_trajectory
    current_turn_msgs: list[dict] = []
    non_system_count = 0
    for raw_msg in raw_history:
        msg = _normalize_msg(raw_msg) if not isinstance(raw_msg, dict) else raw_msg
        role = msg.get('role', '')
        if role == 'system':
            continue
        non_system_count += 1
        if non_system_count <= skip_count:
            continue
        entry: dict = {'role': role}
        content = msg.get('content')
        if content is not None:
            if isinstance(content, str):
                entry['content'] = content
            elif isinstance(content, list):
                entry['content'] = ''.join(
                    part.get('text', '') if isinstance(part, dict) else str(part) for part in content
                )
            else:
                entry['content'] = str(content)
        tool_calls = msg.get('tool_calls')
        if tool_calls:
            calls = []
            for tc in tool_calls:
                tc_dict = _normalize_msg(tc) if not isinstance(tc, dict) else tc
                func = tc_dict.get('function', {})
                if not isinstance(func, dict):
                    func = _normalize_msg(func) if hasattr(func, 'name') else {}
                calls.append({
                    'id': tc_dict.get('id', ''),
                    'type': tc_dict.get('type', 'function'),
                    'function': {
                        'name': func.get('name', ''),
                        'arguments': func.get('arguments', ''),
                    },
                })
            entry['tool_calls'] = calls
        # 保留扩展思考（含签名的 thinking_blocks），确保多轮 history / 落盘轨迹带思考。
        # Anthropic 多轮 tool use 要求把签名思考块原样回传，只留 reasoning_content 不够。
        reasoning_content = msg.get('reasoning_content')
        if reasoning_content:
            entry['reasoning_content'] = reasoning_content
        thinking_blocks = msg.get('thinking_blocks')
        if thinking_blocks:
            entry['thinking_blocks'] = [tb if isinstance(tb, dict) else _normalize_msg(tb) for tb in thinking_blocks]
        # For tool messages, preserve tool_call_id (+ name if present) so search
        # results can be identified later when blanking historical turns.
        if role == 'tool':
            tool_call_id = msg.get('tool_call_id')
            if tool_call_id:
                entry['tool_call_id'] = tool_call_id
            name = msg.get('name')
            if name:
                entry['name'] = name
        current_turn_msgs.append(entry)
    return current_turn_msgs


def _blank_search_results_in_history_turns(history: list[dict]) -> None:
    """In-place blank the content of search-tool results in ``history``.

    ``history`` here is ``accumulated_history`` — it only ever holds *completed*
    turns. The current turn's user prompt is passed to the agent separately, and
    the current turn's live search results are maintained inside the agent loop,
    NOT here. So every search result in ``accumulated_history`` is historical
    relative to whatever future turn will consume it, and can be blanked.

    For each ``role='tool'`` message whose originating tool is a search tool
    (:data:`SEARCH_TOOL_NAMES`), the ``content`` is set to ''. Idempotent:
    already-blanked messages stay ''.

    NOTE: do NOT try to protect "the current turn" (e.g. everything after the
    last ``role='user'`` message). That turn was just appended and is exactly
    the turn that becomes historical for the *next* agent call — protecting it
    leaves the immediately-preceding turn's search results un-blanked in the
    next turn's inference input.
    """
    # tool 消息本身不带工具名，需由 assistant 的 tool_calls 反查 tool_call_id → 工具名
    id_to_name: dict[str, str] = {}
    for msg in history:
        for tc in msg.get('tool_calls') or []:
            tc_id = tc.get('id', '')
            if tc_id:
                id_to_name[tc_id] = (tc.get('function') or {}).get('name', '')
    # 清空所有已完成轮次里 search 工具的 tool 返回（当前轮的实时结果不在此列）
    for msg in history:
        if msg.get('role') != 'tool':
            continue
        name = id_to_name.get(msg.get('tool_call_id', '') or '', '') or str(msg.get('name', '') or '')
        if name in SEARCH_TOOL_NAMES:
            msg['content'] = ''


def _blank_assistant_text_in_history_turns(history: list[dict]) -> None:
    """In-place blank the ``content`` of every assistant message in ``history``.

    ``history`` here is ``accumulated_history`` — it only ever holds *completed*
    turns. Each assistant message's ``content`` (the model's natural-language
    text) is set to ''. ``tool_calls`` are preserved intact so the tool-call /
    tool-result structure stays valid for the next agent call. Idempotent:
    already-blanked messages stay ''.
    """
    for msg in history:
        if msg.get('role') == 'assistant' and len(msg.get('tool_calls', [])) > 0:
            msg['content'] = ''


def _extract_trajectory(agent_output) -> str:
    """Extract the full trajectory as a JSON string (excluding system prompt).

    Includes history from previous turns (user + assistant only) plus the
    current turn's full execution (user, assistant, tool_calls, tool results).
    This captures the complete context the model saw during inference.
    """
    trajectory = []
    raw_history: list = []
    if hasattr(agent_output, 'main_trajectory') and agent_output.main_trajectory:
        raw_history = agent_output.main_trajectory

    for idx, raw_msg in enumerate(raw_history):
        try:
            msg = _normalize_msg(raw_msg)
        except Exception as e:
            logger.warning(f'_extract_trajectory: failed to normalize msg[{idx}]: {e}')
            continue

        role = msg.get('role', '')
        if role == 'system':
            continue

        entry: dict = {'role': role}
        content = msg.get('content')
        if content is not None:
            if isinstance(content, str):
                entry['content'] = content
            elif isinstance(content, list):
                entry['content'] = ''.join(
                    part.get('text', '') if isinstance(part, dict) else str(part) for part in content
                )

        tool_calls = msg.get('tool_calls')
        if tool_calls:
            calls = []
            for tc in tool_calls:
                tc_dict = _normalize_msg(tc) if not isinstance(tc, dict) else tc
                func = tc_dict.get('function', {})
                if not isinstance(func, dict):
                    func = _normalize_msg(func) if hasattr(func, 'name') else {}
                calls.append({
                    'id': tc_dict.get('id', ''),
                    'type': tc_dict.get('type', 'function'),
                    'function': {
                        'name': func.get('name', ''),
                        'arguments': func.get('arguments', ''),
                    },
                })
            entry['tool_calls'] = calls

        if role == 'tool':
            for key in ('tool_call_id', 'name'):
                if msg.get(key):
                    entry[key] = msg[key]

        # 保留扩展思考：reasoning_content 文本 + 带签名的 thinking_blocks，
        # 使 results.jsonl / turn_results 的 trajectory 字符串也含思考（与轨迹文件一致）。
        reasoning_content = msg.get('reasoning_content')
        if reasoning_content:
            entry['reasoning_content'] = reasoning_content
        thinking_blocks = msg.get('thinking_blocks')
        if thinking_blocks:
            entry['thinking_blocks'] = [tb if isinstance(tb, dict) else _normalize_msg(tb) for tb in thinking_blocks]

        trajectory.append(entry)

    if not trajectory:
        if raw_history:
            try:
                return json.dumps([_normalize_msg(m) for m in raw_history], ensure_ascii=False, default=str)
            except Exception:
                pass
        if hasattr(agent_output, 'final_response') and agent_output.final_response:
            return agent_output.final_response
        return str(agent_output)

    return json.dumps(trajectory, ensure_ascii=False, indent=None)


def _extract_answer_text(agent_output) -> str:
    """Extract the full content of the last assistant message.

    Returns the complete content including <answer> tags if present,
    preserving the original model output format.
    """
    history: list = []
    if hasattr(agent_output, 'main_trajectory') and agent_output.main_trajectory:
        history = agent_output.main_trajectory

    for msg in reversed(history):
        msg_dict = _normalize_msg(msg) if not isinstance(msg, dict) else msg
        if msg_dict.get('role') != 'assistant':
            continue
        content = msg_dict.get('content') or ''
        if not content:
            continue
        return content
    return ''


def _build_resume_history_from_output(agent_output) -> list[dict]:
    """Build clean history for an inline A20B continuation.

    The new agent invocation adds its own system prompt, so the previous
    system message is removed. The terminal A3B ``handoff`` call is also
    removed because it is a routing decision, not context A20B should answer.
    Tool calls and results produced before handoff remain available.
    """
    raw = getattr(agent_output, 'main_trajectory', None) or []
    messages = []
    for message in raw:
        normalized = _normalize_msg(message) if not isinstance(message, dict) else message
        if normalized.get('role') != 'system':
            messages.append(normalized)

    if messages:
        last = messages[-1]
        is_handoff = last.get('role') == 'assistant' and any(
            (tool_call.get('function') or tool_call).get('name') == 'handoff'
            for tool_call in (last.get('tool_calls') or [])
        )
        if is_handoff:
            messages.pop()
    return messages


async def _continue_with_live_a20b(task_input, a3b_output, turn_task_id: str):
    """Run the handed-off turn with A20B and always restore the A3B model."""
    a20b_model = os.environ.get('ONE_AGENT_A20B_MODEL', '').strip()
    if not a20b_model:
        raise RuntimeError('A3B handed off in online mode but ONE_AGENT_A20B_MODEL is not set.')

    resume_history = _build_resume_history_from_output(a3b_output)
    logger.info(
        f'[{turn_task_id}] Handoff → running live A20B ({a20b_model}) '
        f'continuation on same sandbox; resume_history={len(resume_history)} msgs.'
    )

    client = require_context().openai_client
    old_model = client.config.request_args.get('model')
    client.config.request_args['model'] = a20b_model
    a20b_tools = sorted((set(DEFAULT_TOOLS) - {'handoff'}) | {'delegate'})
    try:
        return await task_input.run_agent(
            user_prompt='',
            history=resume_history,
            resume_from_history=True,
            terminal_type=task_input.terminal_type,
            tools=a20b_tools,
        )
    finally:
        client.config.request_args['model'] = old_model


def _persist_routing_sidecar(
    output_base_dir: str,
    turn_task_id: str,
    session_id: str,
    turn_num: int,
    routed_to: str,
) -> None:
    """Persist live per-turn routing independently from result expansion."""
    try:
        routing_dir = os.path.join(output_base_dir, 'routing')
        os.makedirs(routing_dir, exist_ok=True)
        with open(
            os.path.join(routing_dir, f'{turn_task_id}.json'),
            'w',
            encoding='utf-8',
        ) as routing_file:
            json.dump(
                {
                    'task_id': turn_task_id,
                    'session_id': session_id,
                    'turn': turn_num,
                    'routed_to': routed_to,
                },
                routing_file,
                ensure_ascii=False,
            )
    except Exception as error:  # noqa: BLE001
        logger.warning(f'[{turn_task_id}] Failed to persist routing sidecar: {error}')


def _parse_media_info_paths(media_info_text: str) -> list[str]:
    """Parse a `<media_info>` body (JSON array) into file paths.

    Returns [] on any parse error / unexpected shape.
    """
    try:
        media_list = json.loads(media_info_text)
    except (json.JSONDecodeError, TypeError):
        return []
    if not isinstance(media_list, list):
        return []
    return [
        item['path']
        for item in media_list
        if isinstance(item, dict) and item.get('type') == 'file' and item.get('path')
    ]


def _extract_file_paths(agent_output) -> list[str]:
    """Extract file paths from the last assistant's `<media_info>`.

    Extraction mode is ``top_level``: match ``<media_info>`` directly in
    the message content. If no ``<media_info>`` found, return ``[]``.

    Always operates on the LAST assistant message's content. When multiple
    ``<media_info>`` blocks exist (rare; defensive), takes the last one.
    """
    history: list = []
    if hasattr(agent_output, 'main_trajectory') and agent_output.main_trajectory:
        history = agent_output.main_trajectory

    for msg in reversed(history):
        msg_dict = _normalize_msg(msg) if not isinstance(msg, dict) else msg
        if msg_dict.get('role') != 'assistant':
            continue
        content = msg_dict.get('content') or ''
        if not content:
            continue

        # top-level <media_info>
        media_matches = re.findall(
            r'<media_info>(.*?)</media_info>',
            content,
            re.DOTALL,
        )
        if not media_matches:
            return []
        return _parse_media_info_paths(media_matches[-1])

    return []


# ════════════════════════════════════════════════════════════════════════
# Input / Output models
# ════════════════════════════════════════════════════════════════════════


class OneAgentInferInput(TaskRunInput, WithAgentRunner):
    """Input for a YB OneAgentBench task (single or multi-turn session)."""

    task_id: str  # session's primary task_id (first turn)
    session_id: str = ''

    # Turns to execute sequentially. Each turn dict contains:
    #   turn, task_id, query, input_file_list, expected_output,
    #   L1, L2, difficulty, [pre_parsed_doc_blocks], [history]
    turns: list[dict]

    benchmark_root: str  # absolute path to benchmark root directory
    timeout: int = 600
    terminal_type: str = 'web'  # 'web' or 'app', passed to agent for prompt style

    # Legacy single-turn fields (for backward compat — ignored if turns is set)
    query: str = ''
    input_file_list: list[str] = []
    expected_output: str = ''
    pre_parsed_doc_blocks: list[str] | None = None
    history: list[dict] | None = None
    L1: str = ''
    L2: str = ''
    difficulty: str = ''


@dataclass
class TurnResult:
    """Result of a single turn within a session."""

    task_id: str
    trajectory: str
    answer: str
    downloaded_files: dict[str, str]
    error: str | None = field(default=None)


class OneAgentInferOutput(TaskRunOutput):
    """Output of a YB OneAgentBench task (may contain multiple turn results)."""

    task_id: str
    session_id: str = ''
    trajectory: str  # JSON: full trajectory of the last turn (backward compat)
    expected_output: str = ''
    downloaded_files: dict[str, str] = {}  # merged from all turns

    # Per-turn results for multi-turn sessions
    turn_results: list[dict] = []  # [{task_id, turn, trajectory, answer, downloaded_files, error}]

    error: str | None = None


# ════════════════════════════════════════════════════════════════════════
# File upload + doc_block building for a single turn
# ════════════════════════════════════════════════════════════════════════


async def _upload_and_build_doc_blocks(
    runtime,
    turn_info: dict,
    benchmark_root: Path,
    task_id: str,
    doc_index_start: int = 0,
) -> tuple[list[str], int]:
    """Upload turn's input files to sandbox and return doc_blocks.

    Returns:
        (doc_blocks, next_doc_index): The built doc blocks and the next
        available doc index for the session.
    """
    working_dir = '/data/workspace'
    inputs_dir = '/data/inputs'

    file_items: list[FileItem] = []
    doc_blocks: list[str] = []
    doc_index = doc_index_start

    # If pre-parsed doc blocks provided, use them directly
    pre_parsed = turn_info.get('pre_parsed_doc_blocks')
    use_pre_parsed = bool(pre_parsed)
    if use_pre_parsed:
        doc_blocks = list(pre_parsed)  # type: ignore[arg-type]
        logger.debug(f'[{task_id}] Using {len(doc_blocks)} pre-parsed doc blocks')

    for relative_path in turn_info.get('input_file_list', []):
        local_path = benchmark_root / relative_path
        if not local_path.exists():
            logger.warning(f'[{task_id}] Input file not found: {local_path}')
            continue

        file_content = local_path.read_bytes()
        encoded_content = base64.b64encode(file_content).decode('utf-8')
        filename = Path(relative_path).name
        suffix = Path(filename).suffix.lower()

        sandbox_path = f'{inputs_dir}/{filename}'
        file_items.append(FileItem(path=sandbox_path, content=encoded_content, encoding='base64'))
        file_items.append(FileItem(path=f'{working_dir}/{filename}', content=encoded_content, encoding='base64'))

        # Subdirectory paths referenced in query
        query = turn_info.get('query', '')
        pattern = re.compile(r'([\w./-]+/)' + re.escape(filename))
        for match in pattern.finditer(query):
            ref = match.group(0)
            if ref != filename:
                file_items.append(FileItem(path=f'{working_dir}/{ref}', content=encoded_content, encoding='base64'))

        if use_pre_parsed:
            continue

        doc_index += 1

        # 图片：直接构造 [user_image N] 块，不做内容解析
        if suffix in IMAGE_SUPPORTED:
            doc_blocks.append(build_image_block(doc_index, filename, sandbox_path))
            logger.debug(f'[{task_id}] Prepared image: {local_path} -> {sandbox_path}')
            continue

        # 文档：大小/类型校验 + 上传 COS → MCP parse_file 解析
        text_content: str | None = None
        parse_error: str | None = None

        if len(file_content) > UPLOAD_MAX_BYTES:
            parse_error = f'文件超过 {UPLOAD_MAX_BYTES // 1024 // 1024} MB，跳过解析'
        elif suffix not in UPLOAD_SUPPORTED:
            parse_error = f'不支持的文件类型 {suffix}，跳过解析（agent 仍可通过工具读取 {sandbox_path}）'
        else:
            # 内容级缓存：昂贵的 COS 上传 + parse_file 只对未见过的文件字节做一次，
            # 命中直接复用本地缓存（见 parse_cache 模块 / prebuild_parse_cache.py）。
            cache_dir = None
            sha = None
            if parse_cache.cache_enabled():
                cache_dir = parse_cache.resolve_cache_dir(benchmark_root)
                sha = parse_cache.content_sha256(file_content)
                text_content = parse_cache.load(cache_dir, sha)
                if text_content is not None:
                    logger.debug(f'[{task_id}] parse-cache HIT {filename} ({sha[:12]})')

            if text_content is None:
                try:
                    parse_info = await get_file_preview_from_bytes(filename, file_content)
                    text_content = parse_info.get('parse_result') or None
                    if cache_dir is not None and sha is not None and text_content:
                        try:
                            parse_cache.save(cache_dir, sha, filename, suffix, text_content)
                            logger.debug(f'[{task_id}] parse-cache STORE {filename} ({sha[:12]})')
                        except Exception as ce:  # 缓存写失败不影响主流程
                            logger.warning(f'[{task_id}] parse-cache save failed for {filename}: {ce}')
                except Exception as e:
                    parse_error = f'parse_file 失败：{e}'

        if parse_error:
            logger.info(f'[{task_id}] {filename}: {parse_error}')

        doc_blocks.append(build_doc_block(doc_index, filename, sandbox_path, text_content, parse_error))
        logger.debug(f'[{task_id}] Prepared: {local_path} -> {sandbox_path}')

    if file_items:
        upload_result = await runtime.upload_file(file_items, overwrite=True)
        for r in upload_result.results:
            logger.info(f'[{task_id}] Upload {r.path}: {"OK" if r.success else r.message}')

    return doc_blocks, doc_index


async def _upload_files_only(
    runtime,
    turn_info: dict,
    benchmark_root: Path,
    task_id: str,
) -> None:
    """Upload turn's input files to sandbox WITHOUT parsing or building doc_blocks.

    Used by ``mode='resume_trajectory'`` benches where ``history`` already
    contains pre-parsed ``[user_doc N]`` blocks (with ``[user_doc_content]``).
    The agent should still be able to ``read`` those files via tools, so we
    upload bytes to ``/data/inputs/<filename>`` and ``/data/workspace/<filename>``.
    """
    working_dir = '/data/workspace'
    inputs_dir = '/data/inputs'
    file_items: list[FileItem] = []

    for relative_path in turn_info.get('input_file_list', []):
        local_path = benchmark_root / relative_path
        if not local_path.exists():
            logger.warning(f'[{task_id}] (resume) Input file not found: {local_path}')
            continue

        file_content = local_path.read_bytes()
        encoded_content = base64.b64encode(file_content).decode('utf-8')
        filename = Path(relative_path).name

        sandbox_path = f'{inputs_dir}/{filename}'
        file_items.append(FileItem(path=sandbox_path, content=encoded_content, encoding='base64'))
        file_items.append(FileItem(path=f'{working_dir}/{filename}', content=encoded_content, encoding='base64'))

        # Subdirectory paths referenced in query (matches _upload_and_build_doc_blocks)
        query = turn_info.get('query', '')
        pattern = re.compile(r'([\w./-]+/)' + re.escape(filename))
        for match in pattern.finditer(query):
            ref = match.group(0)
            if ref != filename:
                file_items.append(FileItem(path=f'{working_dir}/{ref}', content=encoded_content, encoding='base64'))

    if file_items:
        upload_result = await runtime.upload_file(file_items, overwrite=True)
        for r in upload_result.results:
            logger.info(f'[{task_id}] (resume) Upload {r.path}: {"OK" if r.success else r.message}')


# ════════════════════════════════════════════════════════════════════════
# Download files from sandbox
# ════════════════════════════════════════════════════════════════════════


async def _download_files(runtime, file_paths: list[str], task_id: str) -> dict[str, str]:
    """Download files from sandbox, return {relative_path: base64_content}."""
    downloaded: dict[str, str] = {}
    for sandbox_path in file_paths:
        relative_path = sandbox_path.lstrip('/')
        try:
            result = await runtime.download_file(files=[FileItem(path=sandbox_path, encoding='base64')])
            if result.files and result.files[0].content:
                downloaded[relative_path] = result.files[0].content
                logger.debug(f'[{task_id}] Downloaded: {sandbox_path}')
        except Exception as e:
            logger.warning(f'[{task_id}] Failed to download {sandbox_path}: {e}')
    return downloaded


# ════════════════════════════════════════════════════════════════════════
# Main task
# ════════════════════════════════════════════════════════════════════════


@task_registry.register('one_agent_infer')
async def one_agent_infer_task(task_input: OneAgentInferInput) -> OneAgentInferOutput:
    """Execute a YB OneAgentBench evaluation task.

    Handles both single-turn and multi-turn sessions.
    Multi-turn sessions share one sandbox; history is passed between turns.
    """
    ctx = require_context()
    handoff_mode = _get_handoff_mode()
    logger.debug('[one_agent_infer] handoff_mode=%s', handoff_mode)

    # ── ask_user_question 执行策略注入 ──
    # yb_one_agent.tools.ask_user_question_tool 通过
    # ``context.tool_state['ask_user_question_strategy']`` 决定行为：
    #   - 'interactive'（默认）：返回 ``__ask_user_question_interactive__``
    #     占位符，等待真实用户在 UI 答题；批量推理场景下 agent 会把这个
    #     占位返回当作"问题已抛出"进入死循环或提前收尾，**不能用**。
    #   - 'random'：随机选择 option(s) 直接返回标准答案文本，让 agent
    #     无缝继续——这是离线 bench / SFT 蒸馏的正确模式。
    # 默认 'random'，可通过 ``ASK_USER_QUESTION_STRATEGY`` 环境变量覆盖
    # （取值同上：'interactive' / 'random'）。
    auq_strategy = os.getenv('ASK_USER_QUESTION_STRATEGY', 'random').strip() or 'random'
    if auq_strategy not in ('interactive', 'random'):
        logger.warning(
            'Unknown ASK_USER_QUESTION_STRATEGY=%r, falling back to "random".',
            auq_strategy,
        )
        auq_strategy = 'random'
    ctx.tool_state['ask_user_question_strategy'] = auq_strategy
    logger.debug(
        '[one_agent_infer] ask_user_question_strategy=%s',
        auq_strategy,
    )

    runtime = await ctx.get_runtime('main')
    session_id = task_input.session_id or task_input.task_id
    logger.debug(f'[{session_id}] Runtime initialized: {runtime}')

    benchmark_root = Path(task_input.benchmark_root)

    # ── Normalize turns ──
    # If turns is provided and non-empty, use it.
    # Otherwise fall back to legacy single-turn fields for backward compat.
    turns = task_input.turns
    if not turns:
        turns = [
            {
                'turn': 1,
                'task_id': task_input.task_id,
                'query': task_input.query,
                'input_file_list': task_input.input_file_list,
                'expected_output': task_input.expected_output,
                'pre_parsed_doc_blocks': task_input.pre_parsed_doc_blocks,
                'history': task_input.history,
                'L1': task_input.L1,
                'L2': task_input.L2,
                'difficulty': task_input.difficulty,
            }
        ]

    # ── Upload skills to sandbox ──
    skills_dir = Path('/data/skills')
    if skills_dir.exists():
        skill_files: list[FileItem] = []
        for skill_file in skills_dir.rglob('*'):
            if skill_file.is_file():
                rel = skill_file.relative_to(skills_dir)
                sandbox_path = f'/data/skills/{rel}'
                encoded = base64.b64encode(skill_file.read_bytes()).decode('utf-8')
                skill_files.append(FileItem(path=sandbox_path, content=encoded, encoding='base64'))
        if skill_files:
            upload_result = await runtime.upload_file(skill_files, overwrite=True)
            ok = sum(1 for r in upload_result.results if r.success)
            logger.debug(f'[{session_id}] Uploaded {ok}/{len(skill_files)} skill files to sandbox')

    # ── Authorize tdocs-cli inside the sandbox ──
    # The sandbox image ships tdocs-cli on PATH (all-in-one v41+), so only the
    # per-run credential has to be installed here. COS internal routing
    # (TDOCS_COS_INTERNAL) is injected as a native sandbox container env var via
    # runtime_configs.main.environment in the task yaml (docker/gongfeng
    # `--env`), so no wrapper is needed either.
    _tdocs_token = os.environ.get('TDOCS_CLI_TOKEN', '')
    if _tdocs_token:
        try:
            # The token travels in extra_env, so only the variable name ends up
            # in the command string that run_command logs.
            resp = await runtime.run_command(
                'tdocs-cli version && tdocs-cli auth login --token "$TDOCS_CLI_TOKEN"',
                cwd='/data/workspace',
                timeout=60,
                extra_env={'TDOCS_CLI_TOKEN': _tdocs_token},
            )
            logger.info(
                f'[{session_id}] tdocs-cli authorized: status={resp.status} '
                f'stdout={(resp.stdout or "")[:200]!r} stderr={(resp.stderr or "")[:200]!r}'
            )
        except Exception as e:
            logger.warning(f'[{session_id}] tdocs-cli authorization failed: {e}')
    else:
        logger.warning(
            f'[{session_id}] TDOCS_CLI_TOKEN not set; sandbox tdocs-cli stays unauthenticated — '
            'Tencent Docs calls will fail with an auth error'
        )

    # ── Execute turns sequentially ──
    accumulated_history: list[dict] = []  # (user, assistant) pairs across turns
    all_downloaded_files: dict[str, str] = {}
    turn_results: list[dict] = []
    last_trajectory = ''
    last_expected_output = ''
    session_doc_index = 0  # doc id counter across turns within the session
    # CI-style records ship history with already-built [user_doc N] blocks; the
    # turn's new doc ids must continue from where history left off to avoid
    # collisions (e.g. history having doc 1/2 and the current turn also starting
    # from 1). First turn's session_doc_index_start carries that offset.
    if turns:
        session_doc_index = int(turns[0].get('session_doc_index_start') or 0)  # type: ignore[arg-type]

    for turn_info in turns:
        raw_turn = turn_info.get('turn', 1)
        turn_num = int(raw_turn) if isinstance(raw_turn, (int, str)) else 1
        turn_task_id = str(turn_info.get('task_id', session_id))
        query = str(turn_info.get('query', ''))
        last_expected_output = str(turn_info.get('expected_output', ''))

        logger.debug(f'[{session_id}] ── Turn {turn_num}: {turn_task_id} ──')

        # ── Resume-trajectory mode ─────────────────────────────────────
        # When ``mode == 'resume_trajectory'`` the bench supplies a partial
        # trajectory in ``history`` ending in a ``tool`` message (e.g. an
        # ``ask_user_question`` call closed by simulated user answers); the
        # agent should continue from there with **no new user message**.
        # In this mode:
        #   - ``input_file_list`` files are STILL uploaded to sandbox (so the
        #     agent can ``read`` them via tools) but **not** parsed into new
        #     doc_blocks (history already carries pre-parsed blocks).
        #   - Step 2 (user_prompt construction) is skipped; user_prompt = ''.
        #   - Step 4 calls run_agent with ``resume_from_history=True``.
        resume_mode = turn_info.get('mode') == 'resume_trajectory'

        if resume_mode:
            # Upload files only (no parsing, no doc_blocks)
            await _upload_files_only(
                runtime,
                turn_info,
                benchmark_root,
                turn_task_id,
            )
            user_prompt = ''
            doc_blocks: list[str] = []
        else:
            # Step 1: Upload files + build doc_blocks for this turn
            doc_blocks, session_doc_index = await _upload_and_build_doc_blocks(
                runtime,
                turn_info,
                benchmark_root,
                turn_task_id,
                doc_index_start=session_doc_index,
            )

            # Step 2: Build user_prompt
            # 与 v1.1.0 训练数据格式对齐（R16 规则）：
            #   - 相邻 attachment block 之间用 ``\n\n`` 分隔
            #   - 最后一个 ``[user_*  N end]`` 与用户提问之间用单 ``\n``
            #   - 用户提问前加 ``[用户问题] `` 前缀
            # 若 query 已自带 ``[用户问题]`` 前缀（极少见），不重复加。
            user_prompt = query
            if doc_blocks:
                joined_docs = '\n\n'.join(doc_blocks)
                query_part = query.lstrip()
                if query_part.startswith('[用户问题]'):
                    user_prompt = joined_docs + '\n' + query_part
                else:
                    user_prompt = joined_docs + '\n[用户问题] ' + query_part

        # Step 3: Build history for this turn
        # Priority: turn_info.history (from CI data) > accumulated_history (from prev turns)
        turn_history = turn_info.get('history')
        if turn_history:
            # CI data with pre-built history — use it directly, plus accumulated
            history = turn_history + accumulated_history  # type: ignore[operator]
        elif accumulated_history:
            # Multi-turn session — use accumulated (query + answer) pairs
            history = accumulated_history.copy()
        else:
            history = None

        # Step 4: Run agent
        try:
            agent_kwargs: dict = {
                'user_prompt': user_prompt,
                'terminal_type': task_input.terminal_type,
            }
            if history:
                agent_kwargs['history'] = history
            if resume_mode:
                agent_kwargs['resume_from_history'] = True
            a3b_tools = _a3b_tools_for_mode(handoff_mode)
            if a3b_tools is not None:
                agent_kwargs['tools'] = a3b_tools
            agent_output = await task_input.run_agent(**agent_kwargs)
            trajectory_str = _extract_trajectory(agent_output)
            answer_text = _extract_answer_text(agent_output)
        except Exception as e:
            logger.error(f'[{turn_task_id}] Agent failed: {e}')
            if is_sandbox_infrastructure_error(e) or is_llm_http_error(e):
                # Dead sandbox / Gongfeng, or LLM 5xx (e.g. Taiji 503) — fail the
                # run so results.db is not marked success after the agent aborted.
                raise
            turn_results.append({
                'task_id': turn_task_id,
                'turn': turn_num,
                'trajectory': '',
                'answer': '',
                'downloaded_files': {},
                'error': str(e),
            })
            break  # stop session on error
        else:
            exit_status = getattr(agent_output, 'exit_status', None)
            if is_fatal_llm_exit(exit_status):
                raise fatal_llm_exit_error(turn_task_id, exit_status)

        a3b_handed_off = _exit_status_value(agent_output) == 'handoff'
        routed_to = 'a3b'
        if a3b_handed_off and handoff_mode == 'online':
            agent_output = await _continue_with_live_a20b(
                task_input,
                agent_output,
                turn_task_id,
            )
            trajectory_str = _extract_trajectory(agent_output)
            answer_text = _extract_answer_text(agent_output)
            routed_to = 'a20b'

        last_trajectory = trajectory_str

        # Step 5: Download output files from <media_info>
        # Extraction: look for <media_info> at top level in message content
        file_paths = _extract_file_paths(agent_output)
        logger.debug(f'[{turn_task_id}] Files from <media_info>: {file_paths}')
        downloaded = await _download_files(runtime, file_paths, turn_task_id)
        all_downloaded_files.update(downloaded)

        turn_result = {
            'task_id': turn_task_id,
            'turn': turn_num,
            'trajectory': trajectory_str,
            'answer': answer_text,
            'downloaded_files': downloaded,
            'error': None,
        }
        if handoff_mode == 'online':
            turn_result['routed_to'] = routed_to
        turn_results.append(turn_result)

        # Persist this turn's trajectory and files to disk immediately.
        # If a later turn times out, evaluation can still find earlier outputs.
        output_base_dir = os.environ.get('NEXUS_OUTPUT_DIR', '')
        if output_base_dir:
            # Build full trajectory (with system prompt) for this turn
            full_traj_msgs: list[dict] = []
            if hasattr(agent_output, 'main_trajectory') and agent_output.main_trajectory:
                for raw_msg in agent_output.main_trajectory:  # type: ignore[union-attr]
                    full_traj_msgs.append(_normalize_msg(raw_msg) if not isinstance(raw_msg, dict) else raw_msg)
            metadata = {
                'session_id': session_id,
                'task_id': turn_task_id,
                'n_turns': len(turns),
                'turn': turn_num,
            }
            if handoff_mode == 'online':
                metadata['routed_to'] = routed_to
            _persist_turn_output(
                turn_task_id=turn_task_id,
                trajectory_msgs=full_traj_msgs,
                downloaded_files=downloaded,
                tools=[],  # tools are attached by post-processing
                output_base_dir=output_base_dir,
                metadata=metadata,
            )
            if handoff_mode == 'online':
                _persist_routing_sidecar(
                    output_base_dir,
                    turn_task_id,
                    session_id,
                    turn_num,
                    routed_to,
                )

        if a3b_handed_off and handoff_mode == 'offline':
            logger.info(f'[{turn_task_id}] Handoff requested; stopping A3B session for offline baseline composition.')
            break

        # Step 6: Accumulate history for next turn
        # HISTORY_MODE controls how history is built across turns:
        #   - "qa_only" (default): only keep user query + assistant answer
        #   - "full": keep all messages (user, assistant with tool_calls, tool responses)
        #   - "full-empty-search": same as "full", but blank out the content of
        #     search tool results (web_search / picture_search / video_search) that
        #     belong to *historical* turns (the current turn is kept intact). Turns
        #     are delimited by role=user messages.
        #   - "full-empty-assistant-text": same as "full", but blank out the
        #     content (natural-language text) of every assistant message in the
        #     accumulated history (tool_calls are preserved).
        history_mode = os.getenv('HISTORY_MODE', 'qa_only').strip().lower()
        if history_mode == 'full':
            # Keep every non-system message of the current turn (content + tools)
            current_turn_msgs = _extract_turn_full_messages(agent_output, len(accumulated_history))
            accumulated_history.extend(current_turn_msgs)
        elif history_mode == 'full-empty-assistant-text':
            # Same accumulation as "full"…
            current_turn_msgs = _extract_turn_full_messages(agent_output, len(accumulated_history))
            accumulated_history.extend(current_turn_msgs)
            # …then blank the assistant text across all accumulated (completed)
            # turns. accumulated_history holds only finished turns, so every
            # assistant message in it is historical for the next agent call.
            # Idempotent, so running it each turn is fine.
            _blank_assistant_text_in_history_turns(accumulated_history)
        elif history_mode == 'full-empty-search':
            # Same accumulation as "full"…
            current_turn_msgs = _extract_turn_full_messages(agent_output, len(accumulated_history))
            accumulated_history.extend(current_turn_msgs)
            # …then blank search results across all accumulated (completed) turns.
            # accumulated_history holds only finished turns, so every search
            # result in it is historical for the next agent call. Idempotent, so
            # running it each turn is fine.
            _blank_search_results_in_history_turns(accumulated_history)
        else:
            # Default: qa_only — Only keep query + answer (not tool calls / tool results)
            accumulated_history.append({'role': 'user', 'content': user_prompt})
            accumulated_history.append({'role': 'assistant', 'content': answer_text})

        logger.info(
            f'[{turn_task_id}] Turn {turn_num} done. '
            f'Trajectory: {len(trajectory_str)} chars, '
            f'Files: {list(downloaded.keys())}'
        )

    logger.info(f'[{session_id}] Session done. {len(turn_results)} turns completed.')

    return OneAgentInferOutput(
        task_id=task_input.task_id,
        session_id=session_id,
        trajectory=last_trajectory,
        expected_output=last_expected_output,
        downloaded_files=all_downloaded_files,
        turn_results=turn_results,
    )
