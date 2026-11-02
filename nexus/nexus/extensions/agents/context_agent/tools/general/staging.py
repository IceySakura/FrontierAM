"""Host-side staging of local files into a path-native general-context sandbox.

HTTP URIs stay as URLs: the sandbox ``curl`` tool downloads them. Local URIs
are uploaded with ``runtime.upload_file`` into an isolated workspace.
``task-files://`` URIs are materialized through Nexus TaskFiles directly into
the same Runtime workspace. The seven RUNTIME tools therefore never see host
filesystem paths or Dataset object-store coordinates.

Staging records ``working_dir`` / ``declared_files`` / ``knobs`` on
``ctx.tool_state`` so sandbox-side :mod:`.sessions` can rebuild
:class:`GeneralState`.
"""

from __future__ import annotations

import base64
import logging

from pathlib import Path

from nexus.core import ExecutionContext
from nexus.runtime.executors import FileItem

from .state import DeclaredFile, allocate_staged_name
from .workspace import Workspace


logger = logging.getLogger(__name__)


def knobs_payload(
    *,
    file_search_top_k: int,
    file_search_chunk_lines: int,
    python_timeout: int,
    curl_timeout: int,
    curl_max_bytes: int,
) -> dict[str, int]:
    return {
        'file_search_top_k': int(file_search_top_k),
        'file_search_chunk_lines': int(file_search_chunk_lines),
        'python_timeout': int(python_timeout),
        'curl_timeout': int(curl_timeout),
        'curl_max_bytes': int(curl_max_bytes),
    }


def serialize_declared_files(declared_files: list[DeclaredFile]) -> list[dict[str, str | bool | None]]:
    return [
        {
            'name': declared.name,
            'uri': declared.uri,
            'local_path': str(declared.local_path) if declared.local_path is not None else None,
            'runtime_preloaded': declared.runtime_preloaded,
        }
        for declared in declared_files
    ]


def record_tool_state(
    ctx: ExecutionContext,
    *,
    working_dir: str,
    declared_files: list[DeclaredFile],
    knobs: dict | None = None,
) -> None:
    ctx.tool_state['working_dir'] = working_dir
    ctx.tool_state['workspace_root'] = working_dir
    ctx.tool_state['cwd'] = working_dir
    ctx.tool_state['docs_dir'] = f'{working_dir}/{Workspace.DOCS_SUBDIR}'
    ctx.tool_state['declared_files'] = serialize_declared_files(declared_files)
    if knobs:
        ctx.tool_state['knobs'] = dict(knobs)


async def stage_local_files(
    ctx: ExecutionContext,
    declared_files: list[DeclaredFile],
    knobs: dict | None = None,
    *,
    working_dir: str,
    runtime_name: str = 'main',
) -> tuple[list[DeclaredFile], list[dict[str, str]], str]:
    """Materialize task files and upload local files into the sandbox workspace.

    Returns ``(declared_files, staged, working_dir)``. Staged files advertise
    ``./workspace/<name>``; HTTP(S) URLs are unchanged.
    """
    existing_dir = ctx.tool_state.get('working_dir')
    if existing_dir:
        if str(existing_dir) != working_dir:
            raise RuntimeError(
                'context_agent_general: runtime workspace already initialized at '
                f'{existing_dir!r}, cannot switch to {working_dir!r}'
            )
        logger.debug('context_agent_general: workspace already staged at %s; reusing.', existing_dir)
        return declared_files, [], str(existing_dir)

    runtime = await ctx.get_runtime(runtime_name)
    working_dir = str(working_dir).rstrip('/')
    if not working_dir:
        raise ValueError('context_agent_general: working_dir must not be empty')
    docs_dir = f'{working_dir}/{Workspace.DOCS_SUBDIR}'

    file_items: list[FileItem] = []
    task_file_downloads: list[tuple[str, str, str]] = []
    out: list[DeclaredFile] = []
    staged: list[dict[str, str]] = []
    used: set[str] = set()
    has_runtime_preloaded = False

    for declared in declared_files:
        if declared.runtime_preloaded:
            if declared.local_path is None or not declared.local_path.is_absolute():
                raise ValueError(
                    'context_agent_general: Runtime-preloaded files require an absolute runtime local_path'
                )
            has_runtime_preloaded = True
            out.append(declared)
            continue

        if declared.is_task_file:
            task_file_path = declared.task_file_path
            if not task_file_path:
                raise ValueError(f'context_agent_general: empty task-files URI: {declared.uri!r}')
            if ctx.task_files is None:
                raise RuntimeError(
                    f'context_agent_general: {declared.uri!r} requires RunInput.task_files, '
                    'but this run has no TaskFiles source'
                )
            name = allocate_staged_name(declared, used)
            dest = f'{docs_dir}/{name}'
            task_file_downloads.append((task_file_path, dest, name))
            out.append(
                DeclaredFile(
                    name=declared.name or name,
                    uri=f'./{Workspace.DOCS_SUBDIR}/{name}',
                    local_path=Path(dest),
                )
            )
            staged.append({'name': name, 'source': declared.uri})
            continue

        source = declared.local_path
        if declared.is_url or source is None or not source.is_file():
            out.append(declared)
            continue

        name = allocate_staged_name(declared, used)
        dest = f'{docs_dir}/{name}'
        content_b64 = base64.b64encode(source.read_bytes()).decode('ascii')
        file_items.append(FileItem(path=dest, content=content_b64, encoding='base64'))
        out.append(
            DeclaredFile(
                name=declared.name or name,
                uri=f'./{Workspace.DOCS_SUBDIR}/{name}',
                local_path=Path(dest),
            )
        )
        staged.append({'name': name, 'source': str(source)})

    if not file_items and not task_file_downloads and not has_runtime_preloaded:
        file_items.append(FileItem(path=f'{docs_dir}/.keep', content='', encoding='utf8'))

    if file_items:
        await runtime.upload_file(file_items, overwrite=True)
    for task_file_path, dest, _name in task_file_downloads:
        assert ctx.task_files is not None
        # LocalRuntime executes in this process and has no RuntimeControl-backed
        # TaskFiles accessor. Materialize directly to its local filesystem.
        # Remote runtimes use the in-runtime downloader so remote ZIP entries can
        # be range-read from object storage without routing their bytes through
        # the Runner.
        from nexus.runtime.client.local import LocalRuntime

        if isinstance(runtime, LocalRuntime):
            await ctx.task_files.download(task_file_path, dest)
        else:
            await ctx.task_files.download(task_file_path, dest, runtime=runtime)
    record_tool_state(ctx, working_dir=working_dir, declared_files=out, knobs=knobs)
    logger.debug(
        'context_agent_general: staged %d file(s) into sandbox %s (%d local upload(s), %d task file(s))',
        len(staged),
        working_dir,
        len(file_items),
        len(task_file_downloads),
    )
    return out, staged, working_dir


async def cleanup_runtime_session(ctx: ExecutionContext, *, runtime_name: str = 'main') -> None:
    """Best-effort teardown of the sandbox tool session (closes the REPL)."""
    working_dir = ctx.tool_state.get('working_dir')
    if not working_dir:
        return
    try:
        from nexus.core import ToolContext
        from nexus.core.tool import tool_registry
        from nexus.utils.timeout import async_timeout

        from .executor import execution_tool_state
        from .runtime_tools import CLEANUP_TOOL_NAME, registry_id

        entry = tool_registry.get_entry(registry_id(CLEANUP_TOOL_NAME))
        if entry.import_path is None:
            return
        async with async_timeout(10):
            runtime = await ctx.get_runtime(runtime_name)
            tool_context = ToolContext(
                run_id=ctx.run_id,
                tool_state=execution_tool_state(ctx.tool_state),
                trace_id=ctx.trace_id,
            )
            await runtime.call_tool(entry.import_path, {}, tool_context)
    except Exception as exc:  # noqa: BLE001
        logger.debug('context_agent_general: cleanup_runtime_session ignored error: %s', exc)
