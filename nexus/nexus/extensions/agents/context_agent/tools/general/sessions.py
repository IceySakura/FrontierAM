"""Per-rollout tool session registry for path-native general tools (sandbox-side).

Every RUNTIME tool call arrives at the sandbox as a separate ``call_tool``
request, but a rollout needs shared state across those calls: the file cache,
the lazily-built BM25 index, msearch pointers, and the persistent Python REPL.
The sandbox runtime server is a long-lived process, so that state lives in a
process-global dict keyed by ``run_id`` + ``working_dir``.

The host agent stages local files into the sandbox workspace and records
``working_dir`` / ``declared_files`` / ``knobs`` on ``ToolContext.tool_state``
before any model-visible tool runs. HTTP URIs are left as URLs for sandbox
``curl`` to fetch.
"""

from __future__ import annotations

import logging
import threading

from pathlib import Path

from nexus.core import ToolContext

from .state import DeclaredFile, GeneralState, Knobs
from .workspace import Workspace


logger = logging.getLogger(__name__)

_SESSIONS: dict[str, GeneralState] = {}
_LOCK = threading.Lock()


def _session_key(context: ToolContext) -> str:
    working_dir = context.tool_state.get('working_dir') or ''
    key = f'{context.run_id or "?"}::{working_dir}'
    identity = context.tool_state.get('execution_identity')
    if identity is not None:
        key += f'::uid={identity["uid"]}:gid={identity["gid"]}'
    return key


def _knobs_from_context(context: ToolContext) -> Knobs:
    raw = context.tool_state.get('knobs') or {}
    if not isinstance(raw, dict):
        raw = {}
    defaults = Knobs()
    return Knobs(
        file_search_top_k=int(raw.get('file_search_top_k', defaults.file_search_top_k)),
        file_search_chunk_lines=int(raw.get('file_search_chunk_lines', defaults.file_search_chunk_lines)),
        python_timeout=int(raw.get('python_timeout', defaults.python_timeout)),
        curl_timeout=int(raw.get('curl_timeout', defaults.curl_timeout)),
        curl_max_bytes=int(raw.get('curl_max_bytes', defaults.curl_max_bytes)),
    )


def _declared_files_from_context(context: ToolContext) -> list[DeclaredFile]:
    raw = context.tool_state.get('declared_files') or []
    if not isinstance(raw, list):
        return []
    out: list[DeclaredFile] = []
    for item in raw:
        if not isinstance(item, dict):
            continue
        uri = str(item.get('uri') or '').strip()
        if not uri:
            continue
        local = item.get('local_path')
        out.append(
            DeclaredFile(
                name=str(item.get('name') or ''),
                uri=uri,
                local_path=Path(local) if local else None,
                runtime_preloaded=bool(item.get('runtime_preloaded', False)),
            )
        )
    return out


def _build_session(context: ToolContext) -> GeneralState:
    working_dir = context.tool_state.get('working_dir')
    if not working_dir:
        raise RuntimeError(
            "context_agent_general tool session: 'working_dir' is missing from "
            'tool_state. The agent must stage the workspace into the sandbox '
            'and set tool_state["working_dir"] before calling tools.'
        )
    workspace_root = context.tool_state.get('workspace_root') or working_dir
    cwd = context.tool_state.get('cwd') or workspace_root
    docs_dir = context.tool_state.get('docs_dir') or str(Path(workspace_root) / Workspace.DOCS_SUBDIR)
    workspace = Workspace(root=Path(workspace_root), docs_dir=docs_dir, cwd=cwd)
    return GeneralState(
        workspace=workspace,
        knobs=_knobs_from_context(context),
        declared_files=_declared_files_from_context(context),
    )


def get_session(context: ToolContext) -> GeneralState:
    """Return (creating if needed) the shared :class:`GeneralState` for a rollout."""
    key = _session_key(context)
    existing = _SESSIONS.get(key)
    if existing is not None:
        return existing
    with _LOCK:
        existing = _SESSIONS.get(key)
        if existing is not None:
            return existing
        state = _build_session(context)
        _SESSIONS[key] = state
        logger.debug(
            'context_agent_general: built tool session %s (%d declared file(s))',
            key,
            len(state.declared_files),
        )
        return state


def cleanup_session(context: ToolContext) -> bool:
    """Tear down a rollout's session (close the REPL subprocess). Idempotent."""
    key = _session_key(context)
    with _LOCK:
        state = _SESSIONS.pop(key, None)
    if state is None:
        return False
    state.close()
    logger.debug('context_agent_general: closed tool session %s', key)
    return True
