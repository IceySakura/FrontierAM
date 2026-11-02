"""The user-memory file a worker harness reads on startup.

Where it goes is the harness's to declare (``SessionManager.memory_path``); what
goes in it is the run's, passed through ``worker_config`` as ``user_memory``. A run
given no text, or a harness declaring no path, writes nothing.

The write happens sandbox-side, so the path resolves against the worker's own home
rather than one the host assumed.
"""

import logging
import os

from collections.abc import Mapping
from pathlib import Path

from nexus.extensions.tasks.hitl.utils import coerce
from nexus.runtime import Runtime


logger = logging.getLogger(__name__)


CONFIG_KEY = 'user_memory'


def memory_text(config: Mapping[str, object]) -> str:
    """The user memory this run was configured with, or '' for none.

    Whitespace-only counts as none: an all-blank file is one nobody finished
    writing, not a request to place an empty file.
    """
    text = coerce.optional_string(config.get(CONFIG_KEY)) or ''
    return text if text.strip() else ''


def write_user_memory(*, text: str, memory_path: str) -> dict[str, str]:
    """Sandbox-side: write ``text`` at ``memory_path`` under the worker's home.

    An absolute ``memory_path`` is taken as given. Runs in the sandbox so ``HOME``
    is the worker's own, matching how the hook settings are placed
    (``print_claude_code/runtime.py``). ``upload_file`` cannot do this: it resolves
    a relative path against the runtime's cwd and never expands ``~``
    (``nexus/runtime/executors/file.py:34-44``).
    """
    target = Path(os.environ.get('HOME', '/root')) / memory_path
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(text, encoding='utf-8')
    return {'path': str(target), 'chars': str(len(text))}


async def place_user_memory(runtime: Runtime, config: Mapping[str, object], *, memory_path: str) -> None:
    """Write this run's user memory into the sandbox, if it has any.

    Must run before the harness starts: Claude Code reads the file when a session
    starts and does not re-read it, and a PTY session lives for the whole run.

    Raises ``RuntimeError`` when the write fails, since a worker that silently never
    saw its instructions produces a run that looks fine and is not.
    """
    text = memory_text(config)
    if not text or not memory_path:
        return

    try:
        result = await runtime.invoke(write_user_memory, text=text, memory_path=memory_path)
    except Exception as exc:
        raise RuntimeError(f'could not place the user memory file in the sandbox: {exc}') from exc
    placed = result.get('path') if isinstance(result, Mapping) else None
    logger.info('placed %d chars of user memory at %s', len(text), placed or '(unknown path)')


__all__ = [
    'CONFIG_KEY',
    'memory_text',
    'place_user_memory',
    'write_user_memory',
]
