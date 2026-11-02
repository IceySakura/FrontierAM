"""Toolathlon-Gym task: single-task RL training with Toolathlon benchmark.

This task mirrors the Toolathlon path in mcp_benchmark but is simplified
for RL training scenarios. MCP servers run inside a gongfeng (or docker)
sandbox and are exposed to the agent via nexus's MCPToolManager.
"""

import asyncio
import base64
import contextlib
import copy
import io
import json
import logging
import os
import shlex
import tarfile

from datetime import datetime
from pathlib import Path, PurePosixPath
from typing import Any

# Import for side effect: registers the local-* tools (local-claim_done,
# local-python_execute). The task references 'local-claim_done' by name below,
# so this registration must happen at import time. Previously this was an
# accidental side effect of importing tools.python_execute, which commit
# 378f92d7 ("Disable python execute") removed — silently de-registering
# claim_done and breaking every run (agent crashes before the first inference).
import nexus.extensions.tasks.mcp_benchmark.tools  # noqa: F401

from nexus.core import require_context
from nexus.core.task import task_registry
from nexus.extensions.tasks.mcp_benchmark.evaluators.toolathlon import ToolathlonRuntimeEvaluator
from nexus.extensions.tasks.mcp_benchmark.tools.python_execute import PYTHON_EXECUTE_WORKSPACE_KEY
from nexus.extensions.tasks.mcp_benchmark.tools.write_file import WRITE_FILE_WORKSPACE_KEY
from nexus.runtime.executors.command import CommandRunStatus
from nexus.runtime.executors.file import FileItem
from nexus.trace import get_current_span_id, update_span

from .binding import (
    BACKEND_MOCK_ISOLATION,
    MCP_PLACEMENT_BACKEND,
    MCP_PLACEMENT_HYBRID,
    MCP_PLACEMENT_MAIN,
    SPLIT_WORKSPACE_ISOLATION,
    verify_task_binding,
)
from .constants import (
    BENCHMARK_LOCAL_SERVERS,
    BENCHMARK_WORK_ROOT,
    TOOLATHLON_IMAGE_ROOT,
    TOOLATHLON_IMAGE_TASKS_ROOT,
)
from .types import ToolathlonGymTaskInput, ToolathlonGymTaskOutput


logger = logging.getLogger(__name__)


DEFAULT_SYSTEM_PROMPT = (
    'You are operating inside a benchmark sandbox and must work autonomously. '
    'Use the available tools to inspect the environment, local files, repositories, and data before asking for help. '
    'Do not ask the user to provide local paths, repository locations, or environment details that can be discovered via tools. '
    'If the task appears to involve local data, proactively inspect accessible directories such as /data, /workspace, and the current workspace. '
    'When the task depends on exact facts, include the exact identifiers in the final answer as well, such as hashes, IDs, dates, counts, filenames, and quoted titles or messages when available. '
    'Base your final answer only on information you obtained from tool calls or directly from the prompt.'
)


def _case_root(task_input: ToolathlonGymTaskInput) -> str:
    """Derive case_root matching the path convention used by mcp_benchmark.

    The bootstrap / evaluator commands baked into each BenchmarkCase already
    reference paths under /workspace/mcp-benchmark/toolathlon/{server_mode}/{case_id}.
    We must use the same layout so those commands work unchanged.
    """
    return str(PurePosixPath(BENCHMARK_WORK_ROOT) / 'toolathlon' / task_input.server_mode.value / task_input.case_id)


def _agent_workspace(case_root: str) -> str:
    return str(PurePosixPath(case_root) / 'agent_workspace')


def _launch_time() -> str:
    return datetime.now().strftime('%Y-%m-%d %H:%M:%S %A')


def _append_cli_arg(command: str, flag: str, value: str) -> str:
    if flag in command:
        return command
    return f'{command} {flag} {shlex.quote(value)}'


async def _run_checked_command(
    runtime, *, command: str, cwd: str | None = None, env: dict[str, str] | None = None, timeout: float = 600.0
) -> None:
    result = await runtime.run_command(command=command, cwd=cwd, extra_env=env or {}, timeout=timeout)
    if result.status != CommandRunStatus.Finished or result.return_code != 0:
        raise RuntimeError(
            f'command failed: {command}\nstatus={result.status.value}\nreturn_code={result.return_code}\n'
            f'stdout={(result.stdout or "")[:4000]}\nstderr={(result.stderr or "")[:4000]}'
        )


async def _write_eval_log(runtime, *, case_root: str, agent_output: Any, final_answer: str, launch_time: str) -> str:
    log_path = str(PurePosixPath(case_root) / 'res_log.json')
    payload = {
        'status': 'success',
        'config': {'launch_time': launch_time},
        'messages': getattr(agent_output, 'main_trajectory', []),
        'final_answer': final_answer,
    }
    result = await runtime.upload_file([
        FileItem(path=log_path, content=json.dumps(payload, ensure_ascii=False), encoding='utf8')
    ])
    upload_result = result.results[0]
    if not upload_result.success:
        raise RuntimeError(f'failed to upload eval log: {upload_result.message}')
    return log_path


async def _upload_local_dir_to_runtime(
    runtime,
    *,
    local_dir: str,
    remote_dir: str,
    exclude_dirs: set[str] | None = None,
) -> None:
    """Pack a local directory into a tar.gz, upload via base64, and extract on the runtime."""
    local_path = Path(local_dir)
    if not local_path.is_dir():
        raise RuntimeError(f'local_case_source_dir does not exist: {local_dir}')

    exclude_dirs = exclude_dirs or set()

    buf = io.BytesIO()
    with tarfile.open(fileobj=buf, mode='w:gz') as tar:
        for item in local_path.rglob('*'):
            arcname = str(item.relative_to(local_path))
            if any(arcname == d or arcname.startswith(f'{d}/') for d in exclude_dirs):
                continue
            tar.add(str(item), arcname=arcname)
    tar_b64 = base64.b64encode(buf.getvalue()).decode('ascii')

    tar_remote_path = f'{remote_dir}/__case_data.tar.gz'
    result = await runtime.upload_file([FileItem(path=tar_remote_path, content=tar_b64, encoding='base64')])
    upload_result = result.results[0]
    if not upload_result.success:
        raise RuntimeError(f'failed to upload case data tar: {upload_result.message}')

    await _run_checked_command(
        runtime,
        command=f'tar -xzf {shlex.quote(tar_remote_path)} -C {shlex.quote(remote_dir)} && rm -f {shlex.quote(tar_remote_path)}',
        timeout=120.0,
    )


async def _upload_tar_b64_to_runtime(runtime, *, tar_b64: str, remote_dir: str) -> None:
    """Upload a pre-encoded tar.gz (base64) to the runtime and extract it."""
    tar_remote_path = f'{remote_dir}/__inject.tar.gz'
    result = await runtime.upload_file([FileItem(path=tar_remote_path, content=tar_b64, encoding='base64')])
    upload_result = result.results[0]
    if not upload_result.success:
        raise RuntimeError(f'failed to upload tar: {upload_result.message}')

    await _run_checked_command(
        runtime,
        command=f'tar -xzf {shlex.quote(tar_remote_path)} -C {shlex.quote(remote_dir)} && rm -f {shlex.quote(tar_remote_path)}',
        timeout=120.0,
    )


_EVAL_HIDDEN_DIRS = {'evaluation', 'groundtruth_workspace'}

# ── MCP-server isolation (dual-container) ──────────────────────────────────
# When TOOLATHLON_GYM_ISOLATE_MCP_SERVER is truthy, the trusted backend — the
# MCP servers plus everything they need locally (data.db + sqlcipher key, PG,
# the case's mcp_server/ dir) AND the eval pipeline — runs in a SEPARATE runtime
# container ('mcp_backend'), reachable only through the MCP HTTP proxy. The agent
# and its code tools (local-python_execute / local-write_file) stay in 'main',
# which then holds no data.db, no key, no PG and no MCP server process — closing
# the /proc/<pid>/mem sqlcipher-key recovery bypass. Because the boundary lives
# at the MCP server (via each spec's runtime_name → mcp_start_proxy) rather than
# in nexus-private tool routing, the isolation travels with the MCP servers and
# any consuming agent inherits it. The agent_workspace is synced 'main'↔'backend'
# so the isolated evaluator can still read the agent's outputs. Default off —
# existing runs are unaffected.
MCP_BACKEND_RUNTIME_NAME = 'mcp_backend'


def _binding_isolation(binding: dict[str, Any] | None) -> str | None:
    """Return the row-selected isolation mode.

    Missing bindings intentionally mean the historical single-sandbox mode.
    Environment variables never opt a legacy parquet row into isolation.
    """
    return None if binding is None else binding.get('isolation')


async def _sync_workspace_dir(
    src_runtime,
    dst_runtime,
    *,
    remote_dir: str,
    clean_destination: bool = False,
) -> None:
    """Copy a directory's contents from one runtime container to another.

    Used to seed the agent_workspace into 'main' before the agent runs, and to
    copy the agent's outputs back to the isolated backend for evaluation.

    The source archive is fully validated before the destination is touched.
    ``clean_destination`` wipes the destination, so a pack failure that fell
    through would delete the very workspace it was asked to mirror.
    """
    quoted = shlex.quote(remote_dir)
    pack = await src_runtime.run_command(
        command=f'mkdir -p {quoted} && cd {quoted} && tar czf - . | base64 -w0',
        timeout=120.0,
    )
    if pack.status != CommandRunStatus.Finished or pack.return_code != 0:
        raise RuntimeError(
            f'failed to pack workspace {remote_dir}: status={pack.status.value} '
            f'return_code={pack.return_code}\n{(pack.stderr or pack.stdout or "")[:2000]}'
        )
    tar_b64 = (pack.stdout or '').strip()
    if not tar_b64:
        # `tar czf - .` emits a valid archive even for an empty directory, so an
        # empty payload always means the pack failed (timeout/OOM), never that
        # there was nothing to copy.
        raise RuntimeError(f'workspace archive for {remote_dir} is unexpectedly empty')
    prepare_destination = f'mkdir -p {quoted}'
    if clean_destination:
        prepare_destination += f' && find {quoted} -mindepth 1 -maxdepth 1 -exec rm -rf -- {{}} +'
    await _run_checked_command(dst_runtime, command=prepare_destination, timeout=30.0)
    tar_remote_path = f'{remote_dir}/__ws_sync.tar.gz'
    result = await dst_runtime.upload_file([FileItem(path=tar_remote_path, content=tar_b64, encoding='base64')])
    upload_result = result.results[0]
    if not upload_result.success:
        raise RuntimeError(f'failed to upload workspace sync tar: {upload_result.message}')
    await _run_checked_command(
        dst_runtime,
        command=f'tar -xzf {shlex.quote(tar_remote_path)} -C {quoted} && rm -f {shlex.quote(tar_remote_path)}',
        timeout=120.0,
    )


async def _sync_workspace_artifacts(
    src_runtime,
    dst_runtime,
    *,
    remote_dir: str,
    artifacts: list[str],
) -> None:
    """Copy the bound agent artifacts that exist from main to the backend evaluator.

    A missing artifact means the agent never produced it, which is a rollout
    outcome the evaluator must score — not an infrastructure failure. Treating it
    as fatal turned every "agent forgot to write result.json" run into an
    UNHANDLED_ERROR with no reward, discarding the checks the agent did satisfy.
    Genuine pack/transfer failures still raise.
    """
    from .binding import validate_output_artifacts

    safe_artifacts = validate_output_artifacts(artifacts)
    quoted_dir = shlex.quote(remote_dir)
    quoted_files = ' '.join(shlex.quote(path) for path in safe_artifacts)
    pack = await src_runtime.run_command(
        command=(
            f'cd {quoted_dir} && present=""; missing=""; '
            f'for f in {quoted_files}; do '
            f'if [ -f "$f" ]; then present="$present $f"; else missing="$missing $f"; fi; done; '
            f'if [ -n "$missing" ]; then echo "missing output artifacts:$missing" >&2; fi; '
            f'if [ -z "$present" ]; then exit 0; fi; '
            f'tar czf - -- $present | base64 -w0'
        ),
        timeout=120.0,
    )
    if pack.status != CommandRunStatus.Finished or pack.return_code != 0:
        raise RuntimeError(f'failed to pack bound output artifacts: {(pack.stderr or pack.stdout or "")[:2000]}')
    missing_note = (pack.stderr or '').strip()
    if missing_note:
        logger.warning('toolathlon_gym: %s in %s; evaluator will score without them', missing_note, remote_dir)
    tar_b64 = (pack.stdout or '').strip()
    if not tar_b64:
        # No declared artifact was produced at all. Let the evaluator run and
        # report the resulting low score instead of failing the whole task.
        return
    await _run_checked_command(dst_runtime, command=f'mkdir -p {quoted_dir}', timeout=30.0)
    tar_remote_path = f'{remote_dir}/__artifact_sync.tar.gz'
    result = await dst_runtime.upload_file([FileItem(path=tar_remote_path, content=tar_b64, encoding='base64')])
    upload_result = result.results[0]
    if not upload_result.success:
        raise RuntimeError(f'failed to upload artifact sync tar: {upload_result.message}')
    await _run_checked_command(
        dst_runtime,
        command=f'tar -xzf {shlex.quote(tar_remote_path)} -C {quoted_dir} && rm -f {shlex.quote(tar_remote_path)}',
        timeout=120.0,
    )


async def _prepare_backend_mock_agent_workspace(runtime, *, case_root: str, agent_workspace: str) -> None:
    """Create an output-only main sandbox for a bound backend-mock task.

    Some runtime images also contain a source copy of the Toolathlon task tree.
    Removing only the active ``case_root`` would still let local Python inspect
    another copy of ``data.db``/evaluation assets under the image dataset root.
    Bound backend-mock tasks need none of that tree in main, so remove it before
    the agent starts and expose only the empty output workspace.
    """
    await _run_checked_command(
        runtime,
        command=(
            f'rm -rf {shlex.quote(case_root)} {shlex.quote(TOOLATHLON_IMAGE_TASKS_ROOT)} && '
            f'mkdir -p {shlex.quote(agent_workspace)}'
        ),
        timeout=60.0,
    )


async def _prepare_split_main_workspace(runtime, *, case_root: str, agent_workspace: str) -> None:
    """Prepare main as the sole workspace owner without exposing task sources."""
    await _run_checked_command(
        runtime,
        command=(
            f'rm -rf {shlex.quote(case_root)} {shlex.quote(TOOLATHLON_IMAGE_TASKS_ROOT)} && '
            f'mkdir -p {shlex.quote(agent_workspace)}'
        ),
        timeout=60.0,
    )


def _apply_bound_server_placements(
    servers: list[dict[str, Any]],
    *,
    placements: dict[str, str],
) -> set[str]:
    """Apply a verified split binding and return hybrid server names."""
    hybrid_servers: set[str] = set()
    for server in servers:
        name = server.get('name')
        if not isinstance(name, str):
            raise ValueError(f'MCP server name must be a string: {name!r}')
        placement = placements.get(name)
        if placement == MCP_PLACEMENT_MAIN:
            server['runtime_name'] = 'main'
        elif placement in {MCP_PLACEMENT_BACKEND, MCP_PLACEMENT_HYBRID}:
            server['runtime_name'] = MCP_BACKEND_RUNTIME_NAME
            if placement == MCP_PLACEMENT_HYBRID:
                hybrid_servers.add(name)
        else:
            raise ValueError(f'missing or invalid MCP placement for {name!r}: {placement!r}')
    return hybrid_servers


def _install_hybrid_workspace_bridge(
    ctx,
    *,
    hybrid_servers: set[str],
    agent_runtime,
    backend_runtime,
    agent_workspace: str,
) -> None:
    """Serialize hybrid calls around an exact main↔backend workspace mirror."""
    if not hybrid_servers:
        return
    manager = ctx._mcp_manager
    original_call_tool = manager.call_tool
    lock = asyncio.Lock()

    async def bridged_call_tool(full_name: str, arguments: dict[str, Any] | None = None):
        connected = manager.tools.get(full_name)
        if connected is None or connected.server_name not in hybrid_servers:
            return await original_call_tool(full_name, arguments)
        async with lock:
            await _sync_workspace_dir(
                agent_runtime,
                backend_runtime,
                remote_dir=agent_workspace,
                clean_destination=True,
            )
            try:
                return await original_call_tool(full_name, arguments)
            finally:
                await _sync_workspace_dir(
                    backend_runtime,
                    agent_runtime,
                    remote_dir=agent_workspace,
                    clean_destination=True,
                )

    manager.call_tool = bridged_call_tool


async def _prepare_case(runtime, task_input: ToolathlonGymTaskInput) -> 'tuple[str, str]':
    """Create case directory, copy source files, and set up symlinks.

    Evaluation and groundtruth data are deliberately EXCLUDED from the case
    directory during agent execution to prevent reward hacking.  They are
    injected later by ``_inject_eval_data`` right before scoring.

    If ``case_data_tar_b64`` is provided (self-contained parquet mode), it is
    used directly and both ``case_source_dir`` and ``local_case_source_dir``
    are ignored.
    """
    case_root = _case_root(task_input)
    agent_workspace = _agent_workspace(case_root)

    quoted_case_root = shlex.quote(case_root)
    quoted_agent_workspace = shlex.quote(agent_workspace)
    commands = [f'rm -rf {quoted_case_root}', f'mkdir -p {quoted_case_root} {quoted_agent_workspace}']

    if task_input.case_data_tar_b64:
        # Self-contained mode: case data (excl eval/gt) stored in parquet
        await _run_checked_command(runtime, command=' && '.join(commands), timeout=600.0)
        commands = []
        await _upload_tar_b64_to_runtime(
            runtime,
            tar_b64=task_input.case_data_tar_b64,
            remote_dir=case_root,
        )
    elif task_input.case_source_dir:
        quoted_source = shlex.quote(task_input.case_source_dir)
        # Check if the source dir exists on the runtime; if not, upload from local
        check_result = await runtime.run_command(
            command=f'test -d {quoted_source} && echo EXISTS || echo MISSING',
            timeout=10.0,
        )
        source_exists = 'EXISTS' in (check_result.stdout or '')

        if source_exists:
            commands.append(f'cp -a {quoted_source}/. {quoted_case_root}/')
        elif task_input.local_case_source_dir:
            # Create case_root first, then upload local data into it
            await _run_checked_command(runtime, command=' && '.join(commands), timeout=600.0)
            commands = []  # already executed
            await _upload_local_dir_to_runtime(
                runtime,
                local_dir=task_input.local_case_source_dir,
                remote_dir=case_root,
                exclude_dirs=_EVAL_HIDDEN_DIRS,
            )
        else:
            # No local fallback, proceed with cp -a (will fail if truly missing)
            commands.append(f'cp -a {quoted_source}/. {quoted_case_root}/')

    # Set up Toolathlon configs and local servers
    commands.extend([
        f'mkdir -p {quoted_case_root}/configs',
        f'cp -a {shlex.quote(f"{TOOLATHLON_IMAGE_ROOT}/configs")}/. {quoted_case_root}/configs/',
        f'ln -sfn {shlex.quote(f"{TOOLATHLON_IMAGE_ROOT}/deployment")} {quoted_case_root}/deployment',
        f'ln -sfn {shlex.quote(f"{TOOLATHLON_IMAGE_ROOT}/local_binary")} {quoted_case_root}/local_binary',
        f'ln -sfn {shlex.quote(BENCHMARK_LOCAL_SERVERS)} {quoted_case_root}/local_servers',
    ])

    # Copy initial workspace if specified
    if task_input.initial_workspace_dir:
        initial_workspace = shlex.quote(str(PurePosixPath(case_root) / task_input.initial_workspace_dir))
        commands.append(
            f'if [ -d {initial_workspace} ]; then cp -a {initial_workspace}/. {quoted_agent_workspace}/; fi'
        )

    if commands:
        await _run_checked_command(runtime, command=' && '.join(commands), timeout=600.0)

    # Patch server.py to inject stdlib modules into dynamically loaded tools.py.
    # tools.py files are generated without import statements and expect stdlib modules
    # to be pre-injected into the module namespace (matching the original unencrypted server.py).
    _patch_server_script = (
        'import sys; p=sys.argv[1]; '
        'lines=open(p).readlines(); '
        'out=[]; '
        '[out.extend(['
        "'    import json as _json\\n',"
        "'    import re as _re\\n',"
        "'    import math as _math\\n',"
        "'    import datetime as _datetime\\n',"
        "'    try:\\n',"
        "'        import sqlcipher3 as _sqlite3\\n',"
        "'    except ImportError:\\n',"
        "'        import sqlite3 as _sqlite3\\n',"
        "'    import uuid as _uuid\\n',"
        "'    module.json = _json\\n',"
        "'    module.re = _re\\n',"
        "'    module.math = _math\\n',"
        "'    module.datetime = _datetime\\n',"
        "'    module.sqlite3 = _sqlite3\\n',"
        "'    module.uuid = _uuid\\n',"
        "l]) if 'spec.loader.exec_module(module)' in l else out.append(l) for l in lines]; "
        "open(p,'w').writelines(out)"
    )
    _patch_cmd = (
        f'python3 -c {shlex.quote(_patch_server_script)} {quoted_case_root}/mcp_server/server.py 2>/dev/null || true'
    )
    await _run_checked_command(runtime, command=_patch_cmd, timeout=10.0)

    # Remove evaluation/groundtruth from case_root AND all image source paths
    # to prevent agent from accessing them via python_execute or filesystem tools.
    # We remove from the entire finalpool directory (not just this case) because
    # the agent could potentially read other cases' evaluation scripts too.
    rm_commands = [
        f'rm -rf {quoted_case_root}/evaluation {quoted_case_root}/groundtruth_workspace',
        f'find {shlex.quote(TOOLATHLON_IMAGE_ROOT)} -type d \\( -name evaluation -o -name groundtruth_workspace \\) -exec rm -rf {{}} + 2>/dev/null || true',
    ]
    await _run_checked_command(runtime, command=' && '.join(rm_commands), timeout=60.0)

    # Upload local credentials (API keys, service accounts, etc.) into the
    # case's configs/ directory so that preprocess scripts and MCP servers
    # inside the sandbox can find them.
    if task_input.local_credentials_dir:
        creds_path = Path(task_input.local_credentials_dir)
        if creds_path.is_dir():
            remote_configs = f'{case_root}/configs'
            await _upload_local_dir_to_runtime(
                runtime,
                local_dir=str(creds_path),
                remote_dir=remote_configs,
            )

    return case_root, agent_workspace


def _resolve_system_prompt(task_input: ToolathlonGymTaskInput, agent_workspace: str) -> str | list:
    prompt = task_input.system_prompt if task_input.system_prompt is not None else DEFAULT_SYSTEM_PROMPT
    if isinstance(prompt, str):
        prompt = prompt.replace('!!<<<<||||workspace_dir||||>>>>!!', agent_workspace)
    return prompt


async def _inject_eval_data(runtime, task_input: ToolathlonGymTaskInput, case_root: str) -> None:
    """Inject evaluation and groundtruth data into the container AFTER agent finishes.

    This is the core anti-reward-hacking mechanism: the agent never sees these
    files during execution because they are only uploaded right before scoring.
    """
    if task_input.eval_tar_b64:
        await _upload_tar_b64_to_runtime(
            runtime,
            tar_b64=task_input.eval_tar_b64,
            remote_dir=case_root,
        )
    if task_input.groundtruth_tar_b64:
        await _upload_tar_b64_to_runtime(
            runtime,
            tar_b64=task_input.groundtruth_tar_b64,
            remote_dir=case_root,
        )


async def _decrypt_db_for_eval(runtime, *, case_root: str, db_key: str | None) -> None:
    """Decrypt data.db in-place so evaluation scripts (using stdlib sqlite3) can read it."""
    if not db_key:
        logger.warning('_decrypt_db_for_eval: no db_key provided, skipping decrypt')
        return
    key_hex = db_key.encode().hex()
    decrypt_script = (
        'import os, sys\n'
        'import sqlcipher3\n'
        f"db_path = os.path.join({case_root!r}, 'data.db')\n"
        'if not os.path.exists(db_path):\n'
        '    sys.exit(0)\n'
        f"key = bytes.fromhex('{key_hex}').decode()\n"
        "tmp_path = db_path + '.plain'\n"
        'src = sqlcipher3.connect(db_path)\n'
        'src.execute(f"PRAGMA key = \'{key}\'")\n'
        "src.execute('SELECT count(*) FROM sqlite_master')\n"
        "# Export to plaintext using sqlcipher's ATTACH + sqlcipher_export\n"
        "src.execute(f\"ATTACH DATABASE '{tmp_path}' AS plaintext KEY ''\")\n"
        'src.execute("SELECT sqlcipher_export(\'plaintext\')")\n'
        'src.execute("DETACH DATABASE plaintext")\n'
        'src.close()\n'
        'os.replace(tmp_path, db_path)\n'
    )
    try:
        await _run_checked_command(
            runtime,
            command=f'/opt/mcp-benchmarks/venv/bin/python3 -c {shlex.quote(decrypt_script)}',
            timeout=30.0,
        )
    except Exception as exc:
        logger.error('_decrypt_db_for_eval failed for %s: %s', case_root, exc)


async def _restore_pg_access_for_eval(runtime) -> None:
    """Switch PG auth to trust so evaluation scripts can connect regardless of password."""
    cmd = (
        'PG_DATA=/var/lib/postgresql/15/toolathlon_gym && '
        'if [ -f "$PG_DATA/pg_hba.conf" ]; then '
        'sudo -u postgres sed -i \'s/md5$/trust/g\' "$PG_DATA/pg_hba.conf" && '
        'sudo -u postgres /usr/lib/postgresql/15/bin/pg_ctl -D "$PG_DATA" reload -s; '
        'fi'
    )
    with contextlib.suppress(Exception):
        await _run_checked_command(runtime, command=cmd, timeout=10.0)


async def _lock_pg_access(runtime) -> None:
    """Switch PG auth back to md5 so agent cannot access database via python_execute."""
    cmd = (
        'PG_DATA=/var/lib/postgresql/15/toolathlon_gym && '
        'if [ -f "$PG_DATA/pg_hba.conf" ]; then '
        'sudo -u postgres sed -i \'s/trust$/md5/g\' "$PG_DATA/pg_hba.conf" && '
        'sudo -u postgres /usr/lib/postgresql/15/bin/pg_ctl -D "$PG_DATA" reload -s; '
        'fi'
    )
    with contextlib.suppress(Exception):
        await _run_checked_command(runtime, command=cmd, timeout=10.0)


async def _evaluate(
    task_input: ToolathlonGymTaskInput,
    *,
    runtime,
    case_root: str,
    agent_workspace: str,
    launch_time: str,
    res_log_file: str | None,
    db_key: str | None = None,
    continuous_reward_allowed: bool = False,
) -> Any:
    if task_input.evaluator.type == 'toolathlon_runtime' and not continuous_reward_allowed:
        raise RuntimeError('toolathlon_runtime continuous reward requires an isolated backend_mock data binding')

    # Inject eval/groundtruth data now that the agent has finished
    await _inject_eval_data(runtime, task_input, case_root)

    # Decrypt data.db so evaluation scripts using stdlib sqlite3 can access it.
    await _decrypt_db_for_eval(runtime, case_root=case_root, db_key=db_key)

    # Switch PG to trust auth so evaluation scripts with hardcoded passwords can connect.
    await _restore_pg_access_for_eval(runtime)

    evaluator_config = dict(task_input.evaluator.config)
    command = evaluator_config['command']
    if launch_time:
        command = _append_cli_arg(command, '--agent_workspace', agent_workspace)
        command = _append_cli_arg(command, '--launch_time', launch_time)
    if res_log_file:
        command = _append_cli_arg(command, '--res_log_file', res_log_file)
    return await ToolathlonRuntimeEvaluator().evaluate(
        runtime=runtime,
        command=command,
        cwd=evaluator_config.get('cwd'),
        env=dict(evaluator_config.get('env', {})),
        timeout_s=float(evaluator_config.get('timeout_s', 600.0)),
        reward_json_path=(
            '/logs/verifier/reward.json'
            if task_input.evaluator.type == 'toolathlon_runtime' and continuous_reward_allowed
            else None
        ),
    )


@task_registry.register('toolathlon_gym')
async def toolathlon_gym_task(task_input: ToolathlonGymTaskInput) -> ToolathlonGymTaskOutput:
    """Execute a single Toolathlon task for RL training."""
    ctx = require_context()
    runtime = await ctx.get_runtime(task_input.runtime_name)

    # Record runtime_id in span for tracing/debugging
    if hasattr(runtime, 'config') and hasattr(runtime.config, 'base_url'):
        base_url = str(runtime.config.base_url)
        # Extract nexus ID: e.g. "http://nexus-abc123.xxx.hyrl-sandbox..." -> "nexus-abc123"
        url = base_url.removeprefix('http://').removeprefix('https://')
        runtime_id = url.split('.')[0] if '.' in url else url
        span_id = get_current_span_id()
        if span_id:
            update_span(span_id, attributes={'hytuner.runtime_id': runtime_id})

    task_config = task_input.model_dump(mode='json', exclude={'run_agent'}, exclude_unset=True)
    binding = verify_task_binding(task_config)
    if binding is not None:
        logger.debug(
            'toolathlon_gym/%s: verified data binding dataset=%s isolation=%s mock_state=%s artifacts=%s',
            task_input.case_id,
            binding.get('dataset'),
            binding.get('isolation'),
            binding.get('mock_state_access'),
            binding.get('output_artifacts'),
        )

    # ── MCP-server isolation setup ─────────────────────────────────────────
    # Provision the trusted backend BEFORE case prep. When isolation is on, the
    # case (data.db + sqlcipher key + PG + mcp_server/) and the eval pipeline
    # live in 'mcp_backend', while the agent and its code tools run in 'main'.
    # Off → backend_runtime is 'main' and behavior is identical to before.
    isolation_mode = _binding_isolation(binding)
    backend_mock_binding = isolation_mode == BACKEND_MOCK_ISOLATION
    split_workspace_binding = isolation_mode == SPLIT_WORKSPACE_ISOLATION
    # Forward compatibility: a row without data_binding always keeps the
    # historical single-sandbox topology. Isolation is selected by row data,
    # never implicitly by the process environment.
    isolate_mcp = backend_mock_binding or split_workspace_binding
    agent_runtime = runtime
    backend_runtime = runtime
    if isolate_mcp:
        # 'mcp_backend' isn't declared in runtime_configs, so it inherits no
        # image. Reuse 'main's image/working_dir (env var fallback) — same
        # uv/python toolchain and DB deps — but this fresh container is where
        # DB/key/PG/MCP servers live, away from the agent's code execution.
        main_cfg = ctx.runtime_configs.get('main') or {}
        backend_params: dict[str, Any] = {}
        backend_image = main_cfg.get('image') or os.environ.get('MCP_BENCHMARK_MAIN_IMAGE')
        if backend_image:
            backend_params['image'] = backend_image
        if main_cfg.get('working_dir'):
            backend_params['working_dir'] = main_cfg['working_dir']
        backend_runtime = await ctx.get_runtime(MCP_BACKEND_RUNTIME_NAME, **backend_params)
        logger.debug(
            'toolathlon_gym/%s: acquired backend runtime %r for isolation=%s',
            task_input.case_id,
            MCP_BACKEND_RUNTIME_NAME,
            isolation_mode,
        )

    case_root, agent_workspace = await _prepare_case(backend_runtime, task_input)
    resolved_mcp_servers = copy.deepcopy(task_input.mcp_servers)
    hybrid_servers: set[str] = set()

    if backend_mock_binding:
        # The main sandbox owns only the output workspace.  Mock databases,
        # state dumps, server code and evaluator assets remain backend-only.
        await _prepare_backend_mock_agent_workspace(
            agent_runtime,
            case_root=case_root,
            agent_workspace=agent_workspace,
        )
    elif split_workspace_binding:
        await _prepare_split_main_workspace(
            agent_runtime,
            case_root=case_root,
            agent_workspace=agent_workspace,
        )

    # When isolated, relocate the MCP servers to the backend container: setting
    # each spec's runtime_name makes MCPToolManager start them there via
    # mcp_start_proxy — beside data.db/PG and away from the agent's code
    # execution in 'main'. The proxy is the only path the agent has to the DB.
    if backend_mock_binding:
        for server in resolved_mcp_servers:
            if isinstance(server, dict):
                server['runtime_name'] = MCP_BACKEND_RUNTIME_NAME
    elif split_workspace_binding:
        assert binding is not None
        hybrid_servers = _apply_bound_server_placements(
            resolved_mcp_servers,
            placements=binding['server_placements'],
        )

    # Read .db_key before MCP server startup (server.py deletes it on read).
    # Kept in memory to decrypt data.db for evaluation after the agent finishes.
    db_key: str | None = None
    try:
        db_key_path = f'{case_root}/.db_key'
        key_result = await backend_runtime.run_command(
            command=f'cat {shlex.quote(db_key_path)} 2>/dev/null || true',
            timeout=5.0,
        )
        if key_result.stdout and key_result.stdout.strip():
            db_key = key_result.stdout.strip()
    except Exception:
        pass

    final_answer = ''
    eval_result = None
    agent_output = None
    tool_names: list[str] = []
    task_error: Exception | None = None
    launch_time = _launch_time()
    res_log_file: str | None = None

    # Ensure /opt/toolathlon_gym_db/secure_pg.sh exists in the container.
    # Older parquets reference this script in bootstraps but it may not be baked
    # into the Docker image. Creating it here is idempotent and harmless if it
    # already exists.
    _secure_pg_script = (
        "cat > /opt/toolathlon_gym_db/secure_pg.sh << 'EOFSCRIPT'\n"
        '#!/bin/bash\n'
        'PW="${1:-camel}"\n'
        'PG_DATA=/var/lib/postgresql/15/toolathlon_gym\n'
        '[ -f "$PG_DATA/pg_hba.conf" ] || exit 0\n'
        'for i in $(seq 1 30); do\n'
        '  sudo -u postgres /usr/lib/postgresql/15/bin/pg_isready -q && break\n'
        '  sleep 0.5\n'
        'done\n'
        'if sudo -u postgres /usr/lib/postgresql/15/bin/psql -c "ALTER USER eigent PASSWORD \'$PW\'" 2>/dev/null; then\n'
        '  sudo -u postgres sed -i \'s/trust$/md5/g\' "$PG_DATA/pg_hba.conf"\n'
        '  sudo -u postgres /usr/lib/postgresql/15/bin/pg_ctl -D "$PG_DATA" reload -s 2>/dev/null || true\n'
        '  echo -n "$PW" > /tmp/.pg_cred\n'
        'fi\n'
        'EOFSCRIPT\n'
        'chmod +x /opt/toolathlon_gym_db/secure_pg.sh'
    )
    with contextlib.suppress(Exception):
        await _run_checked_command(backend_runtime, command=_secure_pg_script, timeout=10.0)

    try:
        # Run bootstraps (e.g. toolathlon_prepare_case.py, mock servers)
        for bootstrap in task_input.bootstraps:
            await _run_checked_command(
                backend_runtime,
                command=bootstrap.command,
                cwd=bootstrap.cwd,
                env=bootstrap.env,
                timeout=bootstrap.timeout_s,
            )

        # Switch PG to trust auth before preprocess so scripts with hardcoded passwords can connect.
        await _restore_pg_access_for_eval(backend_runtime)

        # Run preprocess (e.g. preprocess/main.py to inject initial state)
        if task_input.preprocess is not None:
            preprocess_command = task_input.preprocess.command
            preprocess_command = _append_cli_arg(preprocess_command, '--agent_workspace', agent_workspace)
            preprocess_command = _append_cli_arg(preprocess_command, '--launch_time', launch_time)
            await _run_checked_command(
                backend_runtime,
                command=preprocess_command,
                cwd=task_input.preprocess.cwd,
                env=task_input.preprocess.env,
                timeout=task_input.preprocess.timeout_s,
            )

        # Inject PG password from /tmp/.pg_cred into MCP server envs.
        # Node's pg library doesn't read .pgpass — it needs PGPASSWORD in the env.
        try:
            cred_result = await backend_runtime.run_command(
                command='cat /tmp/.pg_cred 2>/dev/null || true', timeout=5.0
            )
            pg_password = (cred_result.stdout or '').strip()
        except Exception:
            pg_password = ''
        if pg_password:
            for server in resolved_mcp_servers:
                client = server.get('client', {}) if isinstance(server, dict) else server.client
                env = client.get('env', {}) if isinstance(client, dict) else client.env
                if env and ('PG_USER' in env or 'PGUSER' in env):
                    env['PG_PASSWORD'] = pg_password
                    env['PGPASSWORD'] = pg_password

        # Ensure per-call timeout is set on each MCP server to prevent indefinite hangs.
        for server in resolved_mcp_servers:
            client = server.get('client', server)
            client.setdefault('timeout_s', 300.0)

        # Load MCP tools into the execution context
        loaded_tools = await ctx.load_mcp_tools(resolved_mcp_servers, clear_existing=True)
        loaded_tool_names = sorted(loaded_tools)
        logger.debug('Loaded MCP tools for toolathlon_gym/%s: %s', task_input.case_id, loaded_tool_names)
        if split_workspace_binding:
            _install_hybrid_workspace_bridge(
                ctx,
                hybrid_servers=hybrid_servers,
                agent_runtime=agent_runtime,
                backend_runtime=backend_runtime,
                agent_workspace=agent_workspace,
            )

        if task_input.enabled_tools:
            allowed = set(task_input.enabled_tools)
            tool_names = [name for name in loaded_tool_names if name in allowed]
        else:
            tool_names = loaded_tool_names

        if not tool_names:
            raise RuntimeError(f'No tools enabled for toolathlon_gym/{task_input.case_id}')

        # Lock down database access before the agent runs:
        # 1. Switch PG auth back to md5 (MCP servers already have password in env)
        # 2. Remove credential files so agent can't read them via python_execute
        # Only lock if password was set — otherwise MCP servers have no credentials
        # and would fail to authenticate.
        if pg_password:
            await _lock_pg_access(backend_runtime)
        await _run_checked_command(
            backend_runtime,
            command='rm -f /root/.pgpass /tmp/.pg_cred && rm -f ' + shlex.quote(f'{case_root}/.db_key'),
            timeout=10.0,
        )

        # Inject local-claim_done so agent can signal task completion
        tool_names.append('local-claim_done')

        # Inject local-write_file so agent can save result.json
        ctx.tool_state[WRITE_FILE_WORKSPACE_KEY] = agent_workspace
        tool_names.append('local-write_file')

        # Inject local-python_execute so agent can run Python code
        ctx.tool_state[PYTHON_EXECUTE_WORKSPACE_KEY] = agent_workspace
        tool_names.append('local-python_execute')

        # When isolated, the code tools stay in 'main' (default runtime) while the
        # MCP servers/DB live in 'mcp_backend'. Seed 'main' with the workspace that
        # prepare/preprocess produced in the backend so the agent starts from the
        # same initial state.
        if split_workspace_binding:
            await _sync_workspace_dir(
                backend_runtime,
                agent_runtime,
                remote_dir=agent_workspace,
                clean_destination=True,
            )
        elif isolate_mcp and not backend_mock_binding:
            await _sync_workspace_dir(backend_runtime, agent_runtime, remote_dir=agent_workspace)

        # Run the agent
        agent_kwargs: dict[str, Any] = {
            'user_prompt': task_input.prompt,
            'tools': tool_names,
            'max_iterations': task_input.max_iterations,
            'system_prompt': _resolve_system_prompt(task_input, agent_workspace),
            'observation_time_timeout': task_input.observation_time_timeout or 600,
        }
        if task_input.tool_output_truncation is not None:
            agent_kwargs['tool_output_truncation'] = task_input.tool_output_truncation

        agent_output = await task_input.run_agent(**agent_kwargs)
        final_answer = getattr(agent_output, 'final_answer', '') or ''

        # Copy the agent's workspace outputs (e.g. result.json) from 'main' back
        # to the isolated backend so the evaluator, which runs there against
        # data.db/PG, can read them.
        if backend_mock_binding:
            assert binding is not None
            await _sync_workspace_artifacts(
                agent_runtime,
                backend_runtime,
                remote_dir=agent_workspace,
                artifacts=binding['output_artifacts'],
            )
        elif split_workspace_binding:
            await _sync_workspace_dir(
                agent_runtime,
                backend_runtime,
                remote_dir=agent_workspace,
                clean_destination=True,
            )
        elif isolate_mcp:
            await _sync_workspace_dir(agent_runtime, backend_runtime, remote_dir=agent_workspace)

        # Write evaluation log
        res_log_file = await _write_eval_log(
            backend_runtime,
            case_root=case_root,
            agent_output=agent_output,
            final_answer=final_answer,
            launch_time=launch_time,
        )

        # Evaluate
        eval_result = await _evaluate(
            task_input,
            runtime=backend_runtime,
            case_root=case_root,
            agent_workspace=agent_workspace,
            launch_time=launch_time,
            res_log_file=res_log_file,
            db_key=db_key,
            continuous_reward_allowed=backend_mock_binding,
        )
    except Exception as exc:
        task_error = exc

    if task_error is not None:
        raise task_error
    assert eval_result is not None
    assert agent_output is not None

    # Binary-reward mode for RL training: collapse a continuous evaluator score to
    # pass/fail so the trainer's reward (task_output.score; see runner/hytuner.py)
    # is 1.0 only on a perfect score and 0.0 otherwise. Opt-in via NEXUS_BINARY_REWARD.
    # The native exit-code path is already 0/1, so this is a no-op there; it only
    # affects the continuous reward.json path. passed stays consistent (score>=1.0).
    score = eval_result.score
    if os.getenv('NEXUS_BINARY_REWARD', '').strip().lower() in {'1', 'true', 'yes', 'on'}:
        score = 1.0 if score >= 1.0 else 0.0

    return ToolathlonGymTaskOutput(
        case_id=task_input.case_id,
        server_mode=task_input.server_mode,
        score=score,
        passed=eval_result.passed,
        eval_details=dict(eval_result.details),
        main_trajectory=getattr(agent_output, 'main_trajectory', []),
        agent_final_answer=final_answer,
        loaded_tools=tool_names,
        agent_workspace=agent_workspace,
    )
