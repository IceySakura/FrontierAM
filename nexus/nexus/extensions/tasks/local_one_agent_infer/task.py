"""local_one_agent_infer — 本地模式推理/评测 task。

改编自 ``one_agent_infer``：核心逻辑（多轮 session、doc_block 构造、trajectory /
``<media_info>`` 抽取、按轮增量落盘、``resume_trajectory``）保持一致，差别在于**执行环境**：

- ``one_agent_infer`` 驱动 ``yb_one_agent``，跑在 Docker/gongfeng 沙箱里，
  输入文件靠 ``runtime.upload_file`` 传进 ``/data/inputs`` + ``/data/workspace``，
  产物靠 ``runtime.download_file`` 拉回。
- ``local_one_agent_infer`` 驱动 ``local_one_agent``（``LocalToolExecutor`` 执行工具）。
  依赖文件（``input_file_list``）与产物回收按 ``local_context.runtime.os`` 分叉：
  macos/linux → 本地 ``shutil.copy`` / ``Path.read_bytes``；
  windows/WAA → ``WaaBackend.upload_file`` / ``read_bytes`` 把字节传到远端 VM workspace。

多轮历史通过 ``local_one_agent`` 的 ``history`` / ``resume_from_history`` 入参传递
（与 ``one_agent_infer`` → ``yb_one_agent`` 对齐）。
"""

import base64
import hashlib
import json
import logging
import ntpath
import os
import re
import shutil

from pathlib import Path
from typing import Any

from nexus.core import (
    TaskRunInput,
    TaskRunOutput,
    WithAgentRunner,
    task_registry,
)
from nexus.extensions.agents.local_one_agent.context import resolve_workspace_path
from nexus.extensions.agents.yb_one_agent.llm_errors import fatal_llm_exit_error, is_fatal_llm_exit, is_llm_http_error
from nexus.runtime.executors.models.command import CommandRunStatus


logger = logging.getLogger(__name__)

# 注：``yb_one_agent.get_file_preview`` 会连带 import COS/MCP（fastmcp、requests
# 等）重依赖。本地模式常见路径（图片、pre_parsed_doc_blocks、或模型自己 read 文件）
# 并不需要它，故把该 import 延迟到真正解析文档时（``_seed_and_build_doc_blocks``），
# 让本模块在缺这些可选依赖的环境里也能被 import / 复用抽取逻辑。

# ── Per-turn incremental output persistence ────────────────────────────
# 多轮 session 可能被外层 timeout 打断。为避免丢失已完成轮次，每轮结束立即把
# trajectory / 产物落盘，即使后续轮次超时，评测阶段也能在文件系统里找到早期产物。
_INCREMENTAL_OUTPUT_ENABLED = os.environ.get('NEXUS_INCREMENTAL_OUTPUT', '1') == '1'


def _traj_is_complete(msgs: list[dict]) -> bool:
    """轨迹是否以「最终回答」收尾：末条为 assistant 且不带 tool_calls。

    截断轨迹（如 chat_tracker 树里断在中间工具调用的分支）末条会是带 ``tool_calls``
    的 assistant，据此可判定是否被截断。空轨迹按不完整处理。
    """
    if not msgs:
        return False
    last = msgs[-1]
    return last.get('role') == 'assistant' and not last.get('tool_calls')


def _persist_turn_output(
    turn_task_id: str,
    trajectory_msgs: list[dict],
    downloaded_files: dict[str, str],
    tools: list,
    output_base_dir: str | None,
    metadata: dict | None = None,
    source_metadata: dict | None = None,
) -> None:
    """把一轮已完成的 trajectory 与产物立即落盘。

    轨迹源恒为 agent 的 ``main_trajectory``（= agent 真实完整消息历史），是与
    ``results.db`` 一致的权威来源，**不**取自 ``processor_results.chat`` 的多叶子树
    （那里可能含断在中间工具调用的截断分支）。

    Args:
        turn_task_id: e.g. "Doc-Synthesis-012" or "Doc-Synthesis-012_t2"
        trajectory_msgs: full cumulative trajectory (list of message dicts)
        downloaded_files: {filename: base64_content}
        tools: tool definitions list
        output_base_dir: base directory (parent of files/ and trajectory/)
        metadata: optional metadata dict to attach to trajectory
        source_metadata: optional benchmark source_metadata to attach
    """
    if not output_base_dir or not _INCREMENTAL_OUTPUT_ENABLED:
        return
    try:
        import json as _json

        # 完整性校验：main_trajectory 理论上恒完整（== agent history）；若末条仍是带
        # tool_calls 的 assistant，说明拿到的不是最终态，告警但仍按 main_trajectory 写盘
        # （绝不写截断分支）。
        if not _traj_is_complete(trajectory_msgs):
            logger.warning(
                f'[{turn_task_id}] Persisted trajectory does not end on a final answer '
                f'(last role={trajectory_msgs[-1].get("role") if trajectory_msgs else None}); '
                f'writing main_trajectory as-is.'
            )

        # Write trajectory file
        traj_dir = os.path.join(output_base_dir, 'trajectory')
        os.makedirs(traj_dir, exist_ok=True)
        traj_data: dict = {'trajectory': trajectory_msgs, 'tools': tools}
        if metadata:
            traj_data['metadata'] = metadata
        if source_metadata:
            traj_data['source_metadata'] = source_metadata
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
# Prompt-version-aware <media_info> extraction policy
# ════════════════════════════════════════════════════════════════════════
# 不同 prompt 版本 <media_info> 的包裹方式不同：
#   - v0.2 / v0.3                 : <media_info> 必须在 <answer>...</answer> 内
#   - v1.0+ / latest (live)       : <media_info> 顶层出现（无 <answer>）
#
# 路由严格：按 LOCAL_PROMPT_VERSION 选唯一策略；若期望包裹未命中返回 []，**不回退**
# 到另一策略（否则会掩盖 prompt 格式 bug）。
_PROMPT_VERSIONS_REQUIRING_ANSWER_WRAPPER = {'0.2', '0.3'}


def _media_extract_mode() -> str:
    """Return 'in_answer' or 'top_level' based on LOCAL_PROMPT_VERSION env.

    Mapping:
        '0.2', '0.3'                              → 'in_answer'
        '1.0', 'latest', '', unset, anything else → 'top_level'

    默认 'top_level'，因为本地模式的线上 SP（SPT-Local-Mode）与 v1.0+ 一样已去掉
    <answer> 包裹。
    """
    version = (os.getenv('LOCAL_PROMPT_VERSION', '') or '').strip().lower()
    if version in _PROMPT_VERSIONS_REQUIRING_ANSWER_WRAPPER:
        return 'in_answer'
    return 'top_level'


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

    ``agent_output.main_trajectory`` 含前面轮次（作为 history 上下文传入）的消息。
    跳过前 ``skip_count`` 个 non-system 消息（历史轮次），只保留当前轮，保留
    ``content`` / ``tool_calls`` / ``tool_call_id``。
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
    """In-place blank the content of search-tool results.

    For ``role='tool'`` messages whose originating tool is a search tool
    (:data:`SEARCH_TOOL_NAMES`), the ``content`` is set to ''.
    """
    # tool 消息本身不带工具名，需由 assistant 的 tool_calls 反查 tool_call_id → 工具名
    id_to_name: dict[str, str] = {}
    for msg in history:
        for tc in msg.get('tool_calls') or []:
            tc_id = tc.get('id', '')
            if tc_id:
                id_to_name[tc_id] = (tc.get('function') or {}).get('name', '')
    for msg in history:
        if msg.get('role') != 'tool':
            continue
        name = id_to_name.get(msg.get('tool_call_id', '') or '', '') or str(msg.get('name', '') or '')
        if name in SEARCH_TOOL_NAMES:
            msg['content'] = ''


def _blank_assistant_text_in_history_turns(history: list[dict]) -> None:
    """In-place blank the ``content`` of every assistant message with tool_calls.

    ``history`` here is ``accumulated_history`` — it only ever holds *completed*
    turns. Each such assistant message's ``content`` (natural-language text) is
    set to ''. ``tool_calls`` are preserved so the tool-call / tool-result
    structure stays valid for the next agent call. Idempotent.
    """
    for msg in history:
        if msg.get('role') == 'assistant' and len(msg.get('tool_calls', []) or []) > 0:
            msg['content'] = ''


def _extract_trajectory(agent_output) -> str:
    """Extract the full trajectory as a JSON string (excluding system prompt)."""
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
                calls.append({'name': func.get('name', ''), 'arguments': func.get('arguments', '')})
            entry['tool_calls'] = calls

        trajectory.append(entry)

    if not trajectory:
        if raw_history:
            try:
                return json.dumps([_normalize_msg(m) for m in raw_history], ensure_ascii=False, default=str)
            except Exception:
                pass
        if hasattr(agent_output, 'final_answer') and agent_output.final_answer:
            return agent_output.final_answer
        return str(agent_output)

    return json.dumps(trajectory, ensure_ascii=False, indent=None)


def _extract_answer_text(agent_output) -> str:
    """Extract the full content of the last assistant message."""
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


def _parse_media_info_specs(media_info_text: str) -> list[dict]:
    """Parse a `<media_info>` body (JSON array) into media specs.

    每项形如 ``{"type": "file"|"directory", "path": ...}``；只保留受支持的 type 且 path 非空。

      * ``file``: 单文件（原语义）。
      * ``directory``: 回收该目录下**所有文件（递归）**，落盘保留目录名 + 内部层级。

    返回 ``[{'type': ..., 'path': ...}, ...]``，忽略未知 type 与缺 path 的条目。
    """
    try:
        media_list = json.loads(media_info_text)
    except (json.JSONDecodeError, TypeError):
        return []
    if not isinstance(media_list, list):
        return []
    specs: list[dict] = []
    for item in media_list:
        if not isinstance(item, dict):
            continue
        stype, path = item.get('type'), item.get('path')
        if stype in ('file', 'directory') and path:
            specs.append({'type': stype, 'path': path})
    return specs


def _extract_media_specs(agent_output) -> list[dict]:
    """Extract media specs from the last assistant's `<media_info>`.

    抽取策略由 ``LOCAL_PROMPT_VERSION``（:func:`_media_extract_mode`）严格路由：

      * ``in_answer`` (v0.2 / v0.3): 要求 ``<media_info>`` 在 ``<answer>...</answer>``
        内；无 ``<answer>`` 或内无 ``<media_info>`` 返回 ``[]`` — **无回退**。
      * ``top_level`` (v1.0+ / latest): 直接匹配 ``<media_info>``，忽略 ``<answer>``；
        无 ``<media_info>`` 返回 ``[]`` — **无回退**。

    总是作用于最后一条 assistant 消息；多个 ``<media_info>`` 时取最后一个。
    返回值为 media specs（见 :func:`_parse_media_info_specs`），含 ``type`` 以区分 file/directory。
    """
    history: list = []
    if hasattr(agent_output, 'main_trajectory') and agent_output.main_trajectory:
        history = agent_output.main_trajectory

    mode = _media_extract_mode()

    for msg in reversed(history):
        msg_dict = _normalize_msg(msg) if not isinstance(msg, dict) else msg
        if msg_dict.get('role') != 'assistant':
            continue
        content = msg_dict.get('content') or ''
        if not content:
            continue

        if mode == 'in_answer':
            # ── v0.x: <answer><media_info>...</media_info></answer> ──
            answer_match = re.search(r'<answer>(.*?)</answer>', content, re.DOTALL)
            if not answer_match:
                return []
            media_matches = re.findall(
                r'<media_info>(.*?)</media_info>',
                answer_match.group(1),
                re.DOTALL,
            )
            if not media_matches:
                return []
            return _parse_media_info_specs(media_matches[-1])

        # ── v1.0+ / latest: top-level <media_info> ──
        media_matches = re.findall(
            r'<media_info>(.*?)</media_info>',
            content,
            re.DOTALL,
        )
        if not media_matches:
            return []
        return _parse_media_info_specs(media_matches[-1])

    return []


# ════════════════════════════════════════════════════════════════════════
# Input / Output models
# ════════════════════════════════════════════════════════════════════════


class LocalOneAgentInferInput(TaskRunInput, WithAgentRunner):
    """Input for a local-mode OneAgentBench task (single or multi-turn session)."""

    task_id: str  # session's primary task_id (first turn)
    session_id: str = ''

    # Turns to execute sequentially. Each turn dict contains:
    #   turn, task_id, query, input_file_list, expected_output,
    #   L1, L2, difficulty, [pre_parsed_doc_blocks], [history], [mode]
    turns: list[dict]

    benchmark_root: str  # absolute path to benchmark root directory

    # local_one_agent 运行时上下文；工作区路径见 local_context.workspace.path（单一配置源）。
    local_context: dict = {}

    timeout: int = 600

    # 遗留字段：仅当 local_context.workspace.path 未配置时使用。
    workspace_dir: str | None = None

    # Legacy single-turn fields (for backward compat — ignored if turns is set)
    query: str = ''
    input_file_list: list[str] = []
    expected_output: str = ''
    pre_parsed_doc_blocks: list[str] | None = None
    history: list[dict] | None = None
    L1: str = ''
    L2: str = ''
    difficulty: str = ''

    # Session 级溯源信息（generator 从 benchmark record / BatchItem 透传），仅用于把
    # trajectory 落盘时的 metadata / source_metadata 补全到与既有导出格式一致。
    # 形如 {'index': int, 'session_id': str, 'source_metadata': dict}。
    trace_meta: dict = {}


class LocalOneAgentInferOutput(TaskRunOutput):
    """Output of a local-mode OneAgentBench task (may contain multiple turn results)."""

    task_id: str
    session_id: str = ''
    trajectory: str  # JSON: full trajectory of the last turn (backward compat)
    expected_output: str = ''
    downloaded_files: dict[str, str] = {}  # merged from all turns

    # Per-turn results for multi-turn sessions
    turn_results: list[dict] = []  # [{task_id, turn, trajectory, answer, downloaded_files, error}]

    error: str | None = None


# ════════════════════════════════════════════════════════════════════════
# 本地文件系统：铺输入 + 建 doc_block（对应 reference 的 upload_file）
# ════════════════════════════════════════════════════════════════════════


async def _seed_and_build_doc_blocks(
    turn_info: dict,
    benchmark_root: Path,
    workspace_dir: Path,
    task_id: str,
    doc_index_start: int = 0,
) -> tuple[list[str], int]:
    """把本轮输入文件拷进本地 workspace 并返回 doc_blocks。

    对应 reference 的 ``_upload_and_build_doc_blocks``，差别：不走 ``runtime.upload_file``，
    直接 ``shutil.copy`` 到本地 workspace（``local_one_agent`` 的工具在这里读文件）。

    Returns:
        (doc_blocks, next_doc_index)
    """
    # 延迟 import：仅在真正需要建 doc_block / 解析文档时才拉 COS/MCP 重依赖。
    from nexus.extensions.agents.yb_one_agent.get_file_preview import (
        IMAGE_SUPPORTED,
        UPLOAD_MAX_BYTES,
        UPLOAD_SUPPORTED,
        build_doc_block,
        build_image_block,
        get_file_preview_from_bytes,
    )

    file_items: list[Path] = []  # 仅用于日志计数
    doc_blocks: list[str] = []
    doc_index = doc_index_start

    # If pre-parsed doc blocks provided, use them directly
    pre_parsed = turn_info.get('pre_parsed_doc_blocks')
    if isinstance(pre_parsed, list) and pre_parsed:
        doc_blocks = [str(block) for block in pre_parsed]
        logger.debug(f'[{task_id}] Using {len(doc_blocks)} pre-parsed doc blocks')
        use_pre_parsed = True
    else:
        use_pre_parsed = False

    for relative_path in turn_info.get('input_file_list', []):
        local_path = benchmark_root / relative_path
        if not local_path.exists():
            logger.warning(f'[{task_id}] Input file not found: {local_path}')
            continue

        file_content = local_path.read_bytes()
        rel = _strip_leading_dirs(relative_path)
        filename = rel.name
        suffix = Path(filename).suffix.lower()

        # 剥掉前两级（files/<task_id>/）后拼到 workspace，保留更深层级；模型自己
        # ls/read 找。本地模式无 /data/inputs 概念，workspace 即用户工作区。
        dest_path = workspace_dir / rel
        _local_copy(local_path, dest_path)
        file_items.append(dest_path)

        # query 里若引用了子目录路径，同步铺一份（对齐 reference 行为）
        query = turn_info.get('query', '')
        pattern = re.compile(r'([\w./-]+/)' + re.escape(filename))
        for match in pattern.finditer(query):
            ref = match.group(0)
            if ref != filename:
                _local_copy(local_path, workspace_dir / ref)

        # doc_block 里的路径 = agent 在本地看到的绝对路径
        sandbox_path = str(dest_path)

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
            try:
                parse_info = await get_file_preview_from_bytes(filename, file_content)
                text_content = parse_info.get('parse_result') or None
            except Exception as e:
                parse_error = f'parse_file 失败：{e}'

        if parse_error:
            logger.info(f'[{task_id}] {filename}: {parse_error}')

        doc_blocks.append(build_doc_block(doc_index, filename, sandbox_path, text_content, parse_error))
        logger.debug(f'[{task_id}] Prepared: {local_path} -> {sandbox_path}')

    if file_items:
        logger.debug(f'[{task_id}] Seeded {len(file_items)} file(s) into workspace {workspace_dir}')

    return doc_blocks, doc_index


def _local_copy(src: Path, dest: Path) -> None:
    """本地文件拷贝（建目录 + overwrite），失败只告警。"""
    try:
        dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy(src, dest)
        logger.debug(f'Seeded {src} -> {dest}')
    except Exception as e:
        logger.warning(f'Failed to seed {src} -> {dest}: {e}')


def _strip_leading_dirs(relative_path: str, n: int = 2) -> Path:
    """去掉相对路径的前 ``n`` 级目录，返回剩余部分（保留更深层级）。

    ``input_file_list`` 条目形如 ``files/<task_id>/...``，前两级（``files/`` 与
    ``<task_id>/``）是 benchmark 的组织约定，铺进 workspace 时应剥掉，只保留
    ``<task_id>`` 之下的真实层级（可能是多层递归）。例如 ``n=2``：

        files/Doc-Creative-004/a.html         → a.html
        files/Doc-Creative-004/sub/dir/b.pdf  → sub/dir/b.pdf

    绝对路径、或层级不足 ``n+1`` 时退回纯文件名，避免拼接后逃出 workspace。
    """
    p = Path(relative_path)
    if p.is_absolute():
        return Path(p.name)
    parts = p.parts
    if len(parts) > n:
        return Path(*parts[n:])
    return Path(p.name)


# ════════════════════════════════════════════════════════════════════════
# OS 感知的 workspace 文件 I/O：macos/linux 走本地文件系统；windows 走 WAA VM
# ════════════════════════════════════════════════════════════════════════
# local_one_agent 有两种执行环境：
#   - macos/linux：workspace 就是 nexus 所在机器的本地目录，输入靠 shutil.copy 铺进去、
#     产物直接从本地读回。
#   - windows/WAA：workspace 是远端 Windows VM 的目录，本地 copy 碰不到 VM，必须经
#     WaaBackend（upload_file / read_bytes）把字节上传 / 读回。
# 用同一套 seed/collect 逻辑（含 _strip_leading_dirs 剥层语义），仅在「落地方式」上按 os 分叉。


def _task_os(task_input: LocalOneAgentInferInput) -> str:
    """从 local_context 读用户系统类型（缺省按 macos——本地文件系统语义）。"""
    local_context = task_input.local_context or {}
    runtime = local_context.get('runtime') if isinstance(local_context, dict) else {}
    return ((runtime or {}).get('os')) or 'macos'


async def _acquire_waa_backend() -> Any:
    """windows/WAA：用当前 ExecutionContext 的 runtime 构造 WaaBackend。

    ``ctx.get_runtime()`` 有缓存，与 agent 侧 provisioning 复用同一个 VM 实例（同名 'main'）。
    仅在 windows 分支调用；macos/linux 本地模式没有 runtime provider，不能调。
    """
    from nexus.core import require_context
    from nexus.extensions.agents.local_one_agent.waa_backend import WaaBackend

    ctx = require_context()
    runtime = await ctx.get_runtime()
    return WaaBackend(runtime)


def _vm_join(workspace_path: str, rel: str) -> str:
    """把相对路径拼成 VM 绝对路径（ntpath 语义，统一用反斜杠分隔）。"""
    parts = rel.replace('\\', '/').split('/')
    return ntpath.join(workspace_path.replace('/', '\\'), *parts)


async def _place_seed_file(
    local_path: Path,
    workspace: Path | str,
    rel: Path,
    task_id: str,
    backend: Any,
) -> None:
    """把单个输入文件落到 workspace：本地 ``shutil.copy`` 或 WAA ``upload_file``。"""
    if backend is None:
        _local_copy(local_path, Path(workspace) / rel)
        return
    vm_target = _vm_join(str(workspace), str(rel))
    try:
        await backend.upload_file(vm_target, local_path.read_bytes())
    except Exception as e:  # noqa: BLE001  单文件失败不阻断其余
        logger.warning(f'[{task_id}] Failed to seed (WAA) {local_path} -> {vm_target}: {e}')
    else:
        logger.debug(f'[{task_id}] Seeded (WAA) {local_path} -> {vm_target}')


async def _seed_files_only(
    turn_info: dict,
    benchmark_root: Path,
    workspace: Path | str,
    task_id: str,
    backend: Any = None,
) -> None:
    """把本轮输入文件铺进 workspace，不解析、不构建 doc_blocks。

    对应 reference 的 ``_upload_files_only``。用于 ``mode='resume_trajectory'``：
    ``history`` 里已有预解析的 ``[user_doc N]`` 块，仍需把文件铺进 workspace，
    便于 agent 通过工具 ``read`` 读到。

    ``input_file_list`` 条目形如 ``files/<task_id>/...``，剥掉前两级
    （``files/`` 与 ``<task_id>/``）后拼到 workspace（见 :func:`_strip_leading_dirs`），
    保留 ``<task_id>`` 之下的层级。例如 ``files/Data-001/sub/a.pdf`` →
    ``{workspace}/sub/a.pdf``。

    OS 分叉：``backend is None``（macos/linux）本地 copy；``backend``（windows/WAA）
    上传字节到远端 VM workspace。剥层语义两端一致。
    """
    if backend is not None:
        # windows/WAA：workspace 是远端 VM 目录，先确保存在（upload 也会兜底建父目录）。
        try:
            await backend.create_folder(str(workspace).replace('/', '\\'))
        except Exception as e:  # noqa: BLE001  建目录失败不阻断，upload 会再试建父目录
            logger.warning(f'[{task_id}] Failed to ensure workspace folder (WAA) {workspace}: {e}')

    file_count = 0
    for relative_path in turn_info.get('input_file_list', []):
        local_path = benchmark_root / relative_path
        if not local_path.exists():
            logger.warning(f'[{task_id}] (resume) Input file not found: {local_path}')
            continue

        # 剥掉前两级（files/<task_id>/）后拼到 workspace，保留更深层级。
        rel = _strip_leading_dirs(relative_path)
        await _place_seed_file(local_path, workspace, rel, task_id, backend)
        file_count += 1

        # query 里若引用了其他子目录路径，同步铺一份（对齐 reference 行为）
        filename = rel.name
        query = turn_info.get('query', '')
        pattern = re.compile(r'([\w./-]+/)' + re.escape(filename))
        for match in pattern.finditer(query):
            ref = Path(match.group(0))
            if ref != rel:
                await _place_seed_file(local_path, workspace, ref, task_id, backend)

        logger.debug(f'[{task_id}] (resume) Seeded: {local_path} -> {rel}')

    if file_count:
        logger.debug(f'[{task_id}] (resume) Seeded {file_count} file(s) into workspace {workspace}')


# ════════════════════════════════════════════════════════════════════════
# 本地文件系统：从 workspace 读回产物（对应 reference 的 download_file）
# ════════════════════════════════════════════════════════════════════════


def _collect_one_local(p: Path, workspace_dir: Path, task_id: str, collected: dict[str, str]) -> None:
    """收单个本地产物文件到 ``collected``：相对 workspace 归一，否则退回文件名。"""
    try:
        if not p.is_file():
            logger.warning(f'[{task_id}] Output file not found: {p}')
            return
        try:
            rel = str(p.relative_to(workspace_dir))
        except ValueError:
            rel = p.name
        collected[rel] = base64.b64encode(p.read_bytes()).decode('utf-8')
        logger.debug(f'[{task_id}] Collected: {p}')
    except Exception as e:
        logger.warning(f'[{task_id}] Failed to collect {p}: {e}')


def _collect_dir_local(dir_path: Path, workspace_dir: Path, task_id: str, collected: dict[str, str]) -> None:
    """收本地目录下**所有文件（递归）**到 ``collected``。

    落盘相对路径保留目录名 + 内部层级（相对目录父级），例如 ``reports/`` 下的
    ``a.docx`` / ``sub/b.docx`` → ``reports/a.docx`` / ``reports/sub/b.docx``。
    目录不存在只告警，单文件失败不阻断其余。
    """
    if not dir_path.is_dir():
        logger.warning(f'[{task_id}] Output directory not found: {dir_path}')
        return
    anchor = dir_path.parent  # 相对目录父级 → 保留目录名
    count = 0
    for f in sorted(dir_path.rglob('*')):
        if not f.is_file():
            continue
        try:
            rel = str(f.relative_to(anchor))
        except ValueError:
            rel = f.name
        try:
            collected[rel] = base64.b64encode(f.read_bytes()).decode('utf-8')
            count += 1
        except Exception as e:  # noqa: BLE001  单文件失败不阻断其余
            logger.warning(f'[{task_id}] Failed to collect {f}: {e}')
    logger.debug(f'[{task_id}] Collected directory {dir_path}: {count} file(s)')


def _collect_files(specs: list[dict], workspace_dir: Path, task_id: str) -> dict[str, str]:
    """从本地读回产物，返回 {relative_path: base64_content}。

    对应 reference 的 ``_download_file``，差别：不走 ``runtime.download_file``，
    直接读本地文件系统。按 spec 的 ``type`` 分派：``file`` 收单文件，``directory``
    递归收目录下所有文件。绝对路径尽量相对 workspace 归一化，否则退回按文件名。
    """
    collected: dict[str, str] = {}
    for spec in specs:
        path_str = spec.get('path')
        if not path_str:
            continue
        p = Path(path_str)
        if not p.is_absolute():
            p = workspace_dir / path_str
        if spec.get('type') == 'directory':
            _collect_dir_local(p, workspace_dir, task_id, collected)
        else:
            _collect_one_local(p, workspace_dir, task_id, collected)
    return collected


async def _collect_one_windows(
    path_str: str, workspace_path: str, ws: str, task_id: str, backend: Any, collected: dict[str, str]
) -> None:
    """收单个 VM 产物文件到 ``collected``：``read_bytes`` 取字节，相对 workspace 归一。"""
    raw = path_str.replace('/', '\\')
    vm_path = raw if ntpath.isabs(raw) else _vm_join(workspace_path, path_str)
    try:
        data = await backend.read_bytes(vm_path)
    except Exception as e:  # noqa: BLE001  单文件失败不阻断其余
        logger.warning(f'[{task_id}] Failed to collect (WAA) {vm_path}: {e}')
        return
    prefix = ws.lower() + '\\'
    rel = vm_path[len(ws) + 1 :] if vm_path.lower().startswith(prefix) else ntpath.basename(vm_path)
    rel = rel.replace('\\', '/')  # VM 反斜杠 → posix 分隔符，落盘时才会正确建子目录（见 _persist_turn_output）
    collected[rel] = base64.b64encode(data).decode('utf-8')
    logger.debug(f'[{task_id}] Collected (WAA): {vm_path}')


async def _collect_dir_windows(dir_str: str, task_id: str, backend: Any, collected: dict[str, str]) -> None:
    """收 VM 目录下**所有文件（递归）**到 ``collected``。

    WAA 无列目录 executor，走 PowerShell ``Get-ChildItem -Recurse -File`` 列出全路径，
    再逐个 ``read_bytes``。落盘相对路径保留目录名 + 内部层级（相对目录父级），与本地版
    :func:`_collect_dir_local` 对称。目录不存在 / 命令失败只告警。
    """
    vm_dir = dir_str.replace('/', '\\').rstrip('\\')
    safe = vm_dir.replace("'", "''")  # PowerShell 单引号字符串转义
    ps = f"Get-ChildItem -LiteralPath '{safe}' -Recurse -File | ForEach-Object {{ $_.FullName }}"
    resp = await backend.run_command(ps)
    if resp.status != CommandRunStatus.Finished:
        logger.warning(f'[{task_id}] Failed to list dir (WAA) {vm_dir}: {resp.stderr}')
        return
    anchor = ntpath.dirname(vm_dir)  # 目录父级 → 保留目录名
    count = 0
    for vm_path in (line.strip() for line in (resp.stdout or '').splitlines() if line.strip()):
        try:
            data = await backend.read_bytes(vm_path)
        except Exception as e:  # noqa: BLE001  单文件失败不阻断其余
            logger.warning(f'[{task_id}] Failed to collect (WAA) {vm_path}: {e}')
            continue
        if anchor and vm_path.lower().startswith(anchor.lower() + '\\'):
            rel = vm_path[len(anchor) + 1 :]
        else:
            rel = ntpath.basename(vm_path)
        rel = rel.replace('\\', '/')  # VM 反斜杠 → posix 分隔符，落盘时才会正确建子目录（见 _persist_turn_output）
        collected[rel] = base64.b64encode(data).decode('utf-8')
        count += 1
    logger.debug(f'[{task_id}] Collected directory (WAA) {vm_dir}: {count} file(s)')


async def _collect_files_windows(specs: list[dict], workspace_path: str, task_id: str, backend: Any) -> dict[str, str]:
    """windows/WAA：从 VM 读回产物，返回 {relative_path: base64_content}。

    与本地版对称，差别是走 ``backend.read_bytes``（get_file 取字节，二进制安全）。
    按 spec 的 ``type`` 分派：``file`` 收单文件，``directory`` 递归收目录下所有文件。
    绝对 VM 路径若在 workspace 下则归一成相对路径，否则退回按文件名。
    """
    collected: dict[str, str] = {}
    ws = workspace_path.replace('/', '\\').rstrip('\\')
    for spec in specs:
        path_str = spec.get('path')
        if not path_str:
            continue
        if spec.get('type') == 'directory':
            await _collect_dir_windows(path_str, task_id, backend, collected)
        else:
            await _collect_one_windows(path_str, workspace_path, ws, task_id, backend, collected)
    return collected


async def _collect_output_files(
    specs: list[dict], workspace_path: str, task_id: str, backend: Any = None
) -> dict[str, str]:
    """OS 感知产物回收：windows 从 VM 读回，其余走本地文件系统。"""
    if backend is not None:
        return await _collect_files_windows(specs, workspace_path, task_id, backend)
    return _collect_files(specs, Path(workspace_path), task_id)


# ════════════════════════════════════════════════════════════════════════
# MD5-based workspace diff：在 agent 执行前/后各 snapshot，产出回收新增/修改文件
# ════════════════════════════════════════════════════════════════════════


def _md5_of_bytes(data: bytes) -> str:
    return hashlib.md5(data).hexdigest()  # noqa: S324  (non-crypto usage, checksum only)


def _snapshot_workspace_local(workspace_dir: Path) -> dict[str, str]:
    """递归枚举 workspace 下所有文件，返回 {相对路径: md5hex}。"""
    snapshot: dict[str, str] = {}
    if not workspace_dir.is_dir():
        return snapshot
    for f in sorted(workspace_dir.rglob('*')):
        if not f.is_file():
            continue
        try:
            rel = str(f.relative_to(workspace_dir))
            snapshot[rel] = _md5_of_bytes(f.read_bytes())
        except Exception as e:  # noqa: BLE001
            logger.warning(f'[snapshot] Cannot hash {f}: {e}')
    return snapshot


async def _snapshot_workspace_windows(workspace_path: str, task_id: str, backend: Any) -> dict[str, str]:
    """WAA：用 PowerShell 列出所有文件 + 取 MD5，返回 {相对路径: md5hex}。

    PowerShell 内置 ``Get-FileHash -Algorithm MD5`` 可批量计算：一次命令返回所有结果，
    再逐条 split 解析，避免逐文件 read_bytes 的往返开销（快照阶段不需要内容本身）。
    """
    ws = workspace_path.replace('/', '\\').rstrip('\\')
    safe = ws.replace("'", "''")
    ps = (
        f"Get-ChildItem -LiteralPath '{safe}' -Recurse -File | "
        f'Get-FileHash -Algorithm MD5 | '
        f"ForEach-Object {{ $_.Hash + '|' + $_.Path }}"
    )
    snapshot: dict[str, str] = {}
    try:
        resp = await backend.run_command(ps)
    except Exception as e:  # noqa: BLE001
        logger.warning(f'[{task_id}] _snapshot_workspace_windows failed: {e}')
        return snapshot
    from nexus.runtime.executors.models.command import CommandRunStatus

    if resp.status != CommandRunStatus.Finished:
        logger.warning(f'[{task_id}] _snapshot_workspace_windows: command failed: {resp.stderr}')
        return snapshot
    prefix = ws.lower() + '\\'
    for line in (resp.stdout or '').splitlines():
        line = line.strip()
        if '|' not in line:
            continue
        md5hex, vm_path = line.split('|', 1)
        vm_path = vm_path.strip()
        if vm_path.lower().startswith(prefix):
            rel = vm_path[len(ws) + 1 :].replace('\\', '/')
        else:
            rel = ntpath.basename(vm_path)
        snapshot[rel] = md5hex.strip().lower()
    return snapshot


def _collect_changed_local(
    before: dict[str, str],
    after: dict[str, str],
    workspace_dir: Path,
    task_id: str,
) -> dict[str, str]:
    """对比前后快照，读回新增或内容变化的文件（base64 编码）。"""
    changed = {rel for rel, md5 in after.items() if before.get(rel) != md5}
    collected: dict[str, str] = {}
    for rel in sorted(changed):
        p = workspace_dir / rel
        _collect_one_local(p, workspace_dir, task_id, collected)
    logger.debug(f'[{task_id}] MD5 diff: {len(changed)} changed file(s): {sorted(changed)}')
    return collected


async def _collect_changed_windows(
    before: dict[str, str],
    after: dict[str, str],
    workspace_path: str,
    task_id: str,
    backend: Any,
) -> dict[str, str]:
    """对比前后快照，从 WAA VM 读回新增或内容变化的文件（base64 编码）。"""
    changed = {rel for rel, md5 in after.items() if before.get(rel) != md5}
    collected: dict[str, str] = {}
    ws = workspace_path.replace('/', '\\').rstrip('\\')
    for rel in sorted(changed):
        vm_path = _vm_join(ws, rel.replace('/', '\\'))
        await _collect_one_windows(vm_path, ws, ws, task_id, backend, collected)
    logger.debug(f'[{task_id}] MD5 diff (WAA): {len(changed)} changed file(s): {sorted(changed)}')
    return collected


async def _snapshot_workspace(workspace_path: str, task_id: str, backend: Any = None) -> dict[str, str]:
    """OS 感知快照：windows 走 WAA PowerShell，其余走本地文件系统。"""
    if backend is not None:
        return await _snapshot_workspace_windows(workspace_path, task_id, backend)
    return _snapshot_workspace_local(Path(workspace_path))


async def _collect_changed_files(
    before: dict[str, str],
    after: dict[str, str],
    workspace_path: str,
    task_id: str,
    backend: Any = None,
) -> dict[str, str]:
    """OS 感知产物回收（基于 MD5 diff）：windows 走 WAA，其余走本地文件系统。"""
    if backend is not None:
        return await _collect_changed_windows(before, after, workspace_path, task_id, backend)
    return _collect_changed_local(before, after, Path(workspace_path), task_id)


# ════════════════════════════════════════════════════════════════════════
# Main task
# ════════════════════════════════════════════════════════════════════════


@task_registry.register('local_one_agent_infer')
async def local_one_agent_infer_task(
    task_input: LocalOneAgentInferInput,
) -> LocalOneAgentInferOutput:
    """Execute a local-mode OneAgentBench evaluation task.

    单轮 / 多轮均支持。多轮 session 共享同一个本地 workspace；历史通过
    ``history`` / ``resume_from_history`` 传给 ``local_one_agent``。
    """
    # 本地模式无 runtime：工具由 LocalToolExecutor 直接在 workspace 上执行，
    # OpenAI client 由 run_agent 内部从 ExecutionContext 取，task 层无需 ctx。
    session_id = task_input.session_id or task_input.task_id
    workspace_path = resolve_workspace_path(task_input.model_dump())

    # OS 感知：windows/WAA 的 workspace 是远端 VM 目录，不能本地 mkdir，取 backend 走上传/读回；
    # macos/linux 直接在本地文件系统建目录。seed/collect 均按此分叉。
    is_windows = _task_os(task_input) == 'windows'
    waa_backend: Any = None
    workspace: Path | str
    if is_windows:
        waa_backend = await _acquire_waa_backend()
        workspace = workspace_path
    else:
        workspace = Path(workspace_path)
        workspace.mkdir(parents=True, exist_ok=True)
    logger.debug(f'[{session_id}] Workspace ({"windows/WAA" if is_windows else "local"}): {workspace_path}')

    benchmark_root = Path(task_input.benchmark_root)

    # ── Normalize turns ──
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

    # ── Execute turns sequentially ──
    accumulated_history: list[dict] = []  # (user, assistant) pairs across turns
    all_downloaded_files: dict[str, str] = {}
    turn_results: list[dict] = []
    last_trajectory = ''
    last_expected_output = ''
    # Reserved for _seed_and_build_doc_blocks when doc seeding is re-enabled.
    session_doc_index = 0  # noqa: F841
    if turns:
        raw_doc_start = turns[0].get('session_doc_index_start') or 0
        if isinstance(raw_doc_start, int):
            session_doc_index = raw_doc_start  # noqa: F841
        elif isinstance(raw_doc_start, str) and raw_doc_start.isdigit():
            session_doc_index = int(raw_doc_start)  # noqa: F841

    for turn_info in turns:
        raw_turn = turn_info.get('turn', 1)
        turn_num = raw_turn if isinstance(raw_turn, int) else 1
        turn_task_id = str(turn_info.get('task_id') or session_id)
        query = str(turn_info.get('query') or '')
        last_expected_output = str(turn_info.get('expected_output') or '')

        logger.debug(f'[{session_id}] ── Turn {turn_num}: {turn_task_id} ──')

        # ── Resume-trajectory mode ─────────────────────────────────────
        # When ``mode == 'resume_trajectory'`` the bench supplies a partial
        # trajectory in ``history`` ending in a ``tool`` message (e.g. an
        # ``ask_user_question`` call closed by simulated user answers); the
        # agent should continue from there with **no new user message**.
        # In this mode:
        #   - ``input_file_list`` files are STILL seeded to workspace (so the
        #     agent can ``read`` them via tools) but **not** parsed into new
        #     doc_blocks (history already carries pre-parsed blocks).
        #   - Step 2 (user_prompt construction) is skipped; user_prompt = ''.
        #   - Step 4 calls run_agent with ``resume_from_history=True``.
        resume_mode = turn_info.get('mode') == 'resume_trajectory'

        if resume_mode:
            await _seed_files_only(
                turn_info,
                benchmark_root,
                workspace,
                turn_task_id,
                backend=waa_backend,
            )
            user_prompt = ''
            doc_blocks: list[str] = []
        else:
            # Step 1: 把输入文件铺进 workspace（local agent 自行通过工具读取，不构建 doc_blocks）
            await _seed_files_only(
                turn_info,
                benchmark_root,
                workspace,
                turn_task_id,
                backend=waa_backend,
            )
            doc_blocks = []

            # Step 2: Build user_prompt（对齐训练数据格式：doc block 之间 \n\n，
            # 最后一个 block 与提问之间单 \n，提问前加 [用户问题] 前缀）
            user_prompt = query
            if doc_blocks:
                joined_docs = '\n\n'.join(doc_blocks)
                query_part = query.lstrip()
                if query_part.startswith('[用户问题]'):
                    user_prompt = joined_docs + '\n' + query_part
                else:
                    user_prompt = joined_docs + '\n[用户问题] ' + query_part

        # Step 2.5: 在 agent 执行前快照 workspace（用于 MD5 diff 产物回收）
        pre_snapshot = await _snapshot_workspace(workspace_path, turn_task_id, backend=waa_backend)

        # Step 3: Build history for this turn
        # Priority: turn_info.history (from CI data) > accumulated_history (from prev turns)
        turn_history = turn_info.get('history')
        history: list[dict] | None
        if isinstance(turn_history, list) and turn_history:
            history = list(turn_history) + accumulated_history
        elif accumulated_history:
            history = accumulated_history.copy()
        else:
            history = None

        # Step 4: Run agent
        try:
            agent_kwargs: dict = {'user_prompt': user_prompt}
            if history:
                agent_kwargs['history'] = history
            if resume_mode:
                agent_kwargs['resume_from_history'] = True
            agent_output = await task_input.run_agent(**agent_kwargs)
            trajectory_str = _extract_trajectory(agent_output)
            answer_text = _extract_answer_text(agent_output)
        except Exception as e:
            import traceback

            logger.error(f'[{turn_task_id}] Agent failed: {e}\n{traceback.format_exc()}')
            if is_llm_http_error(e):
                # LLM 5xx (e.g. Taiji 503) — fail the run so results.db is not
                # marked success after the agent aborted.
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

        last_trajectory = trajectory_str

        # Step 5: Collect output files via MD5 workspace diff (new/modified files)
        post_snapshot = await _snapshot_workspace(workspace_path, turn_task_id, backend=waa_backend)
        downloaded = await _collect_changed_files(
            pre_snapshot, post_snapshot, workspace_path, turn_task_id, backend=waa_backend
        )
        all_downloaded_files.update(downloaded)

        turn_results.append({
            'task_id': turn_task_id,
            'turn': turn_num,
            'trajectory': trajectory_str,
            'answer': answer_text,
            'downloaded_files': downloaded,
            'error': None,
        })

        # Persist this turn's trajectory and files to disk immediately.
        output_base_dir = os.environ.get('NEXUS_OUTPUT_DIR', '')
        if output_base_dir:
            full_traj_msgs: list[dict] = []
            raw_traj = getattr(agent_output, 'main_trajectory', None)
            if isinstance(raw_traj, list):
                for raw_msg in raw_traj:
                    full_traj_msgs.append(_normalize_msg(raw_msg) if not isinstance(raw_msg, dict) else raw_msg)
            _persist_turn_output(
                turn_task_id=turn_task_id,
                trajectory_msgs=full_traj_msgs,
                downloaded_files=downloaded,
                tools=[],
                output_base_dir=output_base_dir,
                metadata={
                    'index': task_input.trace_meta.get('index'),
                    'session_id': session_id,
                    'task_id': turn_task_id,
                    'n_turns': len(turns),
                    'L1': turn_info.get('L1', task_input.L1),
                    'L2': turn_info.get('L2', task_input.L2),
                    'difficulty': turn_info.get('difficulty', task_input.difficulty),
                    'turn': turn_num,
                },
                source_metadata=(turn_info.get('source_metadata') or task_input.trace_meta.get('source_metadata')),  # type: ignore[invalid-argument-type]
            )

        # Step 6: Accumulate history for next turn.
        # HISTORY_MODE 语义同 reference（qa_only / full / full-empty-search /
        # full-empty-assistant-text）。
        history_mode = os.getenv('HISTORY_MODE', 'qa_only').strip().lower()
        logger.debug(f'[{turn_task_id}] HISTORY_MODE: {history_mode}')

        if history_mode == 'full':
            current_turn_msgs = _extract_turn_full_messages(agent_output, len(accumulated_history))
            accumulated_history.extend(current_turn_msgs)
        elif history_mode == 'full-empty-assistant-text':
            current_turn_msgs = _extract_turn_full_messages(agent_output, len(accumulated_history))
            accumulated_history.extend(current_turn_msgs)
            _blank_assistant_text_in_history_turns(accumulated_history)
        elif history_mode == 'full-empty-search':
            current_turn_msgs = _extract_turn_full_messages(agent_output, len(accumulated_history))
            accumulated_history.extend(current_turn_msgs)
            _blank_search_results_in_history_turns(accumulated_history)
        else:
            accumulated_history.append({'role': 'user', 'content': user_prompt})
            accumulated_history.append({'role': 'assistant', 'content': answer_text})

        logger.info(
            f'[{turn_task_id}] Turn {turn_num} done. '
            f'Trajectory: {len(trajectory_str)} chars, '
            f'Files: {list(downloaded.keys())}'
        )

    logger.info(f'[{session_id}] Session done. {len(turn_results)} turns completed.')

    return LocalOneAgentInferOutput(
        task_id=task_input.task_id,
        session_id=session_id,
        trajectory=last_trajectory,
        expected_output=last_expected_output,
        downloaded_files=all_downloaded_files,
        turn_results=turn_results,
    )
