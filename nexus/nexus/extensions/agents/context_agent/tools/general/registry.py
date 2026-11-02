"""Tool dispatch for HyGeneralToolSystem.

``dispatch_tool`` routes a single tool call (by name) to its handler, threading
the per-rollout :class:`GeneralState`. All handlers return a string (the
content shown back to the model).
"""

from __future__ import annotations

import asyncio
import logging

from typing import Any

from .state import GeneralState
from .tools.curl import curl as _curl
from .tools.file_search import mclick as _mclick
from .tools.file_search import msearch as _msearch
from .tools.grep import grep as _grep
from .tools.python_repl import PythonREPL
from .tools.read_lines import read_lines as _read_lines
from .tools.shell import bash as _bash


logger = logging.getLogger(__name__)


def _get_repl(state: GeneralState) -> PythonREPL:
    if state._python_repl is None:
        state._python_repl = PythonREPL(cwd=state.workspace.cwd, timeout=state.knobs.python_timeout)
    return state._python_repl  # type: ignore[return-value]


async def dispatch_tool(
    *,
    name: str,
    arguments: dict[str, Any],
    state: GeneralState,
) -> str:
    """Dispatch a single tool call by name and return its string result."""
    if name == 'curl':
        return await _curl(
            state,
            str(arguments.get('url', '')),
            arguments.get('output'),
        )

    if name == 'bash':
        return await _bash(
            str(arguments.get('command', '')),
            cwd=state.workspace.cwd,
        )

    if name == 'python':
        repl = _get_repl(state)
        return await repl.exec(str(arguments.get('code', '')))

    if name == 'file_search_msearch':
        queries = arguments.get('queries', [])
        if isinstance(queries, str):
            queries = [queries]
        top_k = arguments.get('top_k')
        return await asyncio.to_thread(
            _msearch,
            state,
            list(queries),
            int(top_k) if top_k is not None else None,
            arguments.get('files'),
        )

    if name == 'file_search_mclick':
        pointers = arguments.get('pointers', [])
        if isinstance(pointers, str):
            pointers = [pointers]
        return await asyncio.to_thread(_mclick, state, list(pointers))

    if name == 'grep':
        before = arguments.get('before')
        after = arguments.get('after')
        head_limit = arguments.get('head_limit')
        return await asyncio.to_thread(
            _grep,
            state,
            str(arguments.get('pattern', '')),
            path=arguments.get('path'),
            use_regex=bool(arguments.get('use_regex', False)),
            context_lines=int(arguments.get('context_lines', 1)),
            before=int(before) if before is not None else None,
            after=int(after) if after is not None else None,
            output_mode=str(arguments.get('output_mode') or 'content'),
            head_limit=int(head_limit) if head_limit is not None else None,
            offset=int(arguments.get('offset') or 0),
        )

    if name == 'read_lines':
        return await asyncio.to_thread(
            _read_lines,
            state,
            str(arguments.get('path', '')),
            int(arguments.get('start', 0)),
            int(arguments.get('end', -1)),
        )

    return f'[Error] Unknown tool: {name}'
