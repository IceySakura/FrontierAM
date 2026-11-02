"""local_one_agent_infer AGS 模式 — 远端 runtime 推理/评测 task。

改编自 ``local_one_agent_infer.task``：多轮 session、trajectory / ``<media_info>`` 抽取、
按轮增量落盘、``resume_trajectory`` 与本地版一致。核心差别（对应 ``one_agent_infer``）：

1. **必须初始化 runtime**（``ctx.get_runtime('main')``），工具经 AGS/WAA 在远端 VM 执行。
2. **输入文件 upload 到镜像/VM**，不走本地 ``shutil.copy``。AGS WAA 无 FileItem
   ``runtime.upload_file`` executor，故经 ``WaaBackend.upload_file``（``setup_upload``）
   把字节铺进 ``local_context.workspace.path``；产物经 ``read_bytes`` 拉回。

当配置 ``runtime_provider_name: ags`` 时，generator 会把 ``task_name`` 路由到本模块
注册的 ``local_one_agent_infer_ags``（见 ``generator.py`` / ``config/ags_base.yaml``）。
"""

import logging
import os
import re

from pathlib import Path
from typing import Any

from nexus.core import require_context, task_registry
from nexus.extensions.agents.local_one_agent.context import resolve_workspace_path
from nexus.extensions.agents.local_one_agent.waa_backend import WaaBackend
from nexus.extensions.agents.yb_one_agent.llm_errors import fatal_llm_exit_error, is_fatal_llm_exit, is_llm_http_error
from nexus.extensions.tasks.local_one_agent_infer.task import (
    LocalOneAgentInferInput,
    LocalOneAgentInferOutput,
    _blank_assistant_text_in_history_turns,
    _blank_search_results_in_history_turns,
    _collect_files_windows,
    _extract_answer_text,
    _extract_media_specs,
    _extract_trajectory,
    _extract_turn_full_messages,
    _media_extract_mode,
    _normalize_msg,
    _persist_turn_output,
    _strip_leading_dirs,
    _vm_join,
)


logger = logging.getLogger(__name__)


# ════════════════════════════════════════════════════════════════════════
# AGS/WAA：upload 输入 + download 产物（对应 one_agent_infer 的 upload/download）
# ════════════════════════════════════════════════════════════════════════


async def _upload_files_only(
    runtime: Any,
    turn_info: dict,
    benchmark_root: Path,
    workspace_path: str,
    task_id: str,
) -> None:
    """把本轮输入文件 upload 到 AGS/WAA VM workspace，不解析、不构建 doc_blocks。

    对应 ``one_agent_infer._upload_files_only`` / 本地版 ``_seed_files_only``。
    AGS WAA 走 ``WaaBackend.upload_file``（底层 ``setup_upload``），目标路径为
    ``{workspace}/{剥层后相对路径}``。

    ``input_file_list`` 条目形如 ``files/<task_id>/...``，剥掉前两级后拼到 workspace。
    """
    backend = WaaBackend(runtime)
    try:
        await backend.create_folder(str(workspace_path).replace('/', '\\'))
    except Exception as e:  # noqa: BLE001  建目录失败不阻断，upload 会再试建父目录
        logger.warning(f'[{task_id}] Failed to ensure workspace folder (AGS) {workspace_path}: {e}')

    file_count = 0
    for relative_path in turn_info.get('input_file_list', []):
        local_path = benchmark_root / relative_path
        if not local_path.exists():
            logger.warning(f'[{task_id}] (upload) Input file not found: {local_path}')
            continue

        rel = _strip_leading_dirs(relative_path)
        await _upload_one(backend, local_path, workspace_path, rel, task_id)
        file_count += 1

        # query 里若引用了其他子目录路径，同步 upload 一份（对齐 one_agent_infer）
        filename = rel.name
        query = turn_info.get('query', '')
        pattern = re.compile(r'([\w./-]+/)' + re.escape(filename))
        for match in pattern.finditer(query):
            ref = Path(match.group(0))
            if ref != rel:
                await _upload_one(backend, local_path, workspace_path, ref, task_id)

        logger.debug(f'[{task_id}] (upload) Seeded: {local_path} -> {rel}')

    if file_count:
        logger.debug(f'[{task_id}] (upload) Uploaded {file_count} file(s) into AGS workspace {workspace_path}')


async def _upload_one(
    backend: WaaBackend,
    local_path: Path,
    workspace_path: str,
    rel: Path,
    task_id: str,
) -> None:
    """单文件 upload 到 VM（对应 one_agent_infer 往 FileItem 列表追加一条）。"""
    vm_target = _vm_join(workspace_path, str(rel))
    try:
        await backend.upload_file(vm_target, local_path.read_bytes())
    except Exception as e:  # noqa: BLE001  单文件失败不阻断其余
        logger.warning(f'[{task_id}] Failed to upload (AGS) {local_path} -> {vm_target}: {e}')
    else:
        logger.debug(f'[{task_id}] Upload (AGS) {local_path} -> {vm_target}: OK')


async def _download_files(
    runtime: Any,
    specs: list[dict],
    workspace_path: str,
    task_id: str,
) -> dict[str, str]:
    """从 AGS/WAA VM 拉回产物，返回 {relative_path: base64_content}。

    对应 ``one_agent_infer._download_files``；WAA 无 ``runtime.download_file`` executor，
    经 ``WaaBackend.read_bytes`` 取字节。specs 含 ``type`` 以区分 file/directory。
    """
    backend = WaaBackend(runtime)
    return await _collect_files_windows(specs, workspace_path, task_id, backend)


# ════════════════════════════════════════════════════════════════════════
# Main task
# ════════════════════════════════════════════════════════════════════════


@task_registry.register('local_one_agent_infer_ags')
async def local_one_agent_infer_ags_task(
    task_input: LocalOneAgentInferInput,
) -> LocalOneAgentInferOutput:
    """Execute a local_one_agent bench task on AGS remote runtime (WAA).

    单轮 / 多轮均支持。多轮 session 共享同一个远端 workspace；历史通过
    ``history`` / ``resume_from_history`` 传给 ``local_one_agent``。
    """
    # ── 初始化 runtime（与 one_agent_infer 对齐）──
    # 必须先 get_runtime，后续工具执行与文件 upload 才走远端 VM。
    ctx = require_context()
    runtime = await ctx.get_runtime('main')
    session_id = task_input.session_id or task_input.task_id
    logger.debug(f'[{session_id}] Runtime initialized: {runtime}')

    workspace_path = resolve_workspace_path(task_input.model_dump())
    logger.debug(f'[{session_id}] AGS workspace: {workspace_path}')

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
    accumulated_history: list[dict] = []
    all_downloaded_files: dict[str, str] = {}
    turn_results: list[dict] = []
    last_trajectory = ''
    last_expected_output = ''

    for turn_info in turns:
        raw_turn = turn_info.get('turn', 1)
        turn_num = raw_turn if isinstance(raw_turn, int) else 1
        turn_task_id = str(turn_info.get('task_id') or session_id)
        query = str(turn_info.get('query') or '')
        last_expected_output = str(turn_info.get('expected_output') or '')

        logger.debug(f'[{session_id}] ── Turn {turn_num}: {turn_task_id} ──')

        resume_mode = turn_info.get('mode') == 'resume_trajectory'

        # Step 1: upload 输入文件到 AGS 镜像/VM（resume / 正常模式都做）
        await _upload_files_only(
            runtime,
            turn_info,
            benchmark_root,
            workspace_path,
            turn_task_id,
        )

        # DEBUG STEP 3: 用 WAA server 自己的 API 排查 upload 落盘位置（与 upload 同一 API 族）。
        # try:
        #     _win_ws = str(workspace_path).replace('/', '\\')
        #     _raw = runtime  # AGSRawWindowsAgentArenaRuntime
        #     _out = [f'workspace={_win_ws}']
        #     # 1) 新建一个 ASCII 命名的探针文件夹 + 文件，抓 setup 返回文本
        #     _probe_dir = 'C:\\Users\\a-chen\\Desktop\\dbgprobe'
        #     try:
        #         _r_cf = await _raw.setup_create_folder(_probe_dir)
        #         _out.append(f'setup_create_folder({_probe_dir}) -> {_r_cf!r}')
        #     except Exception as _e:  # noqa: BLE001
        #         _out.append(f'setup_create_folder err: {_e!r}')
        #     try:
        #         _r_up = await _raw.setup_upload(_probe_dir + '\\hello.txt', b'hello-waa')
        #         _out.append(f'setup_upload(hello.txt) -> {_r_up!r}')
        #     except Exception as _e:  # noqa: BLE001
        #         _out.append(f'setup_upload err: {_e!r}')
        #     # 2) server 侧 list_directory 各处
        #     for _p in ['C:\\', 'C:\\Users', _probe_dir, _win_ws, _win_ws.rsplit('\\', 1)[0]]:
        #         try:
        #             _ld = await _raw.list_directory(_p)
        #             _out.append(f'list_directory({_p}) -> {str(_ld)[:600]}')
        #         except Exception as _e:  # noqa: BLE001
        #             _out.append(f'list_directory({_p}) err: {_e!r}')
        #     # 3) file_exists 针对刚上传的真实业务文件
        #     try:
        #         _fe = await _raw.file_exists('电网调度模块成绩.xls', _win_ws)
        #         _out.append(f'file_exists(电网调度模块成绩.xls, workspace) -> {_fe!r}')
        #     except Exception as _e:  # noqa: BLE001
        #         _out.append(f'file_exists err: {_e!r}')
        #     print('[DEBUG-STEP3]\n' + '\n'.join(_out), flush=True)
        # except Exception as _dbg_e:  # noqa: BLE001
        #     import traceback as _tb
        #     print(f'[DEBUG-STEP3] failed: {_dbg_e!r}\n{_tb.format_exc()}', flush=True)
        # raise ValueError('debug exit: STEP3 files uploaded to VM workspace')

        if resume_mode:
            user_prompt = ''
            doc_blocks: list[str] = []
        else:
            doc_blocks = []
            user_prompt = query
            if doc_blocks:
                joined_docs = '\n\n'.join(doc_blocks)
                query_part = query.lstrip()
                if query_part.startswith('[用户问题]'):
                    user_prompt = joined_docs + '\n' + query_part
                else:
                    user_prompt = joined_docs + '\n[用户问题] ' + query_part

        # Step 3: Build history for this turn
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
                raise
            turn_results.append({
                'task_id': turn_task_id,
                'turn': turn_num,
                'trajectory': '',
                'answer': '',
                'downloaded_files': {},
                'error': str(e),
            })
            break
        else:
            exit_status = getattr(agent_output, 'exit_status', None)
            if is_fatal_llm_exit(exit_status):
                raise fatal_llm_exit_error(turn_task_id, exit_status)

        last_trajectory = trajectory_str

        # Step 5: Collect output files from <media_info>
        media_specs = _extract_media_specs(agent_output)
        logger.debug(f'[{turn_task_id}] Media specs from <media_info> (mode={_media_extract_mode()}): {media_specs}')
        downloaded = await _download_files(runtime, media_specs, workspace_path, turn_task_id)
        all_downloaded_files.update(downloaded)

        turn_results.append({
            'task_id': turn_task_id,
            'turn': turn_num,
            'trajectory': trajectory_str,
            'answer': answer_text,
            'downloaded_files': downloaded,
            'error': None,
        })

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

        # Step 6: Accumulate history for next turn
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
