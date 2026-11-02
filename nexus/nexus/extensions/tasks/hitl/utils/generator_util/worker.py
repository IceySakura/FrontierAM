"""What a run configures its worker with, independent of the harness.

The pieces here hold for any backend: the user memory and skill directories a run
supplies, and the env-derived request-arg kwargs its client profile takes. What only
Claude Code accepts lives in :mod:`claude_code`.
"""

import logging
import os

from collections.abc import Sequence
from pathlib import Path
from typing import Any

from nexus.extensions.tasks.hitl.user_agent import mcp_config
from nexus.extensions.tasks.hitl.utils.generator_util.env import _env_bool, _env_int


logger = logging.getLogger(__name__)


def _worker_version(default: str) -> str:
    """The harness release to install, for whichever harness the backend names."""
    return os.environ.get('HITL_WORKER_VERSION', '').strip() or default


def _worker_turn_timeout(default: int) -> int:
    """Seconds one worker turn may run. ``default`` is the harness's own."""
    return _env_int('HITL_WORKER_TURN_TIMEOUT', default=default)


def _worker_context_window(*, default: int = 0) -> int:
    """The context window the harness sizes its auto-compaction against.

    Claude Code compacts at ``HITL_CC_AUTOCOMPACT_PCT`` of it, Codex at
    ``min(auto_compact_token_limit, this * 90%)``.
    """
    return _env_int('HITL_WORKER_CONTEXT_WINDOW', default=default)


def _worker_reasoning_effort() -> str:
    """The effort this run asks for, in the profile's own terms. '' leaves the default.

    Read by the host client's profile, by the Codex config (``model_reasoning_effort``),
    and by the run metadata.
    """
    return os.environ.get('HITL_WORKER_REASONING_EFFORT', '').strip()


def _worker_profile_kwargs(*, session_id: str = '') -> dict[str, Any]:
    """Collect env-derived values to pass to the selected profile build function.

    Only non-empty values are included, so each profile's own defaults apply unless
    explicitly overridden. ``session_id`` (a per-run uuid) is threaded in for profiles
    that route by it, e.g. HY3 consistent-hash.
    """
    kwargs: dict[str, Any] = {}
    if session_id:
        kwargs['session_id'] = session_id
    # thinking_preset is absent: it is the proxy's wire encoding, resolved by
    # claude_code._worker_thinking_preset, rather than a profile request-arg input.
    str_env = {
        'reasoning_mode': 'HITL_WORKER_REASONING_MODE',
        'thinking_type': 'HITL_WORKER_THINKING_TYPE',
        'cache_strategy': 'HITL_WORKER_CACHE_STRATEGY',
    }
    for arg, env_key in str_env.items():
        value = os.environ.get(env_key, '').strip()
        if value:
            kwargs[arg] = value
    if effort := _worker_reasoning_effort():
        kwargs['reasoning_effort'] = effort
    return kwargs


def _read_user_memory() -> str:
    """The text of the file ``HITL_WORKER_MEMORY_FILE`` points at.

    Returns '' when the variable is unset or the file is empty, which the managers read
    as "this run was given no user memory".

    The path may be anywhere: ``~`` and relative paths are expanded, so a run can point
    at a file outside the repo. A path that names no file raises rather than returning
    '', since a typo would otherwise run the whole batch with no preferences and
    nothing said.
    """
    raw = os.environ.get('HITL_WORKER_MEMORY_FILE', '').strip()
    if not raw:
        return ''
    path = Path(raw).expanduser().resolve()
    if not path.is_file():
        raise ValueError(f'HITL_WORKER_MEMORY_FILE names no file: {path}')
    text = path.read_text(encoding='utf-8')
    if not text.strip():
        logger.info('[hitl/generator] %s is empty; the worker is given no user memory', path)
        return ''
    return text


def with_worker_skills(
    worker_config: dict[str, Any],
    skill_dirs: Sequence[str],
    *,
    env_name: str = 'HITL_WORKER_SKILL_DIRS',
) -> dict[str, Any]:
    """Name skill directories for the worker and record what they install as.

    The directories are read here, where they exist, and validated as skills. Placing
    them is the backend's: it declares where its harness reads them and uploads them
    there (``actor/skills.py``), the same split the user memory file follows. A
    directory's own name is the name a ``/<name>`` message has to use.

    Writes ``worker_commands`` in the same call, which the manager reads to decide what
    it accepts and the user simulator is told it has. One call, so a run cannot install
    a skill it does not offer or offer one it did not install.

    ``env_name`` appears in a refusal, so a generator reading a differently named
    variable reports the one its own run script sets.
    """
    from nexus.extensions.tasks.hitl.actor.input_spec import WORKER_COMMANDS_KEY
    from nexus.extensions.tasks.hitl.actor.skills import CONFIG_KEY
    from nexus.extensions.tasks.hitl.user_agent.skills.loader import read_skill

    skills = []
    resolved = []
    for entry in skill_dirs:
        path = Path(entry).expanduser()
        if not path.is_dir():
            raise ValueError(f'{env_name} names {entry!r}, which is not a directory')
        skills.append(read_skill(path, origin='worker'))
        resolved.append(str(path))

    return {
        **worker_config,
        CONFIG_KEY: resolved,
        WORKER_COMMANDS_KEY: {skill.name: skill.description for skill in skills},
    }


def _search_server() -> dict[str, Any]:
    """The built-in search MCP server, as ``claude mcp add`` takes it.

    ``python`` rather than a path: the sandbox replaces a bare interpreter with the one
    it is running (``actor/mcp.py``), which is where ``mcp`` and ``httpx`` are. Claude
    Code reports a failed registration but not a failed launch, so an interpreter
    without them reads as a worker that never used the tool.
    """
    return {
        'name': 'search',
        'command': 'python',
        'args': ['-m', 'nexus.extensions.tasks.hitl.mcp.search_server'],
    }


def worker_mcp_servers(config: dict[str, Any]) -> list[dict[str, Any]]:
    """The MCP servers this run declares, plus the built-in search one when asked.

    Writes ``SEARCH_API_KEY`` into ``config['extra_env']`` for the search server, and
    ``SEARCH_API_URL`` when that is set too.
    """
    mcp_servers = mcp_config.worker_servers()
    if not _env_bool('HITL_WORKER_MCP_SEARCH'):
        return mcp_servers

    search_api_key = os.environ.get('SEARCH_API_KEY', '')
    if not search_api_key:
        raise ValueError('HITL_WORKER_MCP_SEARCH=1 requires SEARCH_API_KEY')
    # Through the same check a declared server gets.
    mcp_servers.append(mcp_config.check_worker_entry(_search_server(), source='the built-in search server'))
    config.setdefault('extra_env', {})['SEARCH_API_KEY'] = search_api_key
    search_api_url = os.environ.get('SEARCH_API_URL', '')
    if search_api_url:
        config['extra_env']['SEARCH_API_URL'] = search_api_url
    return mcp_servers


__all__ = [
    '_read_user_memory',
    '_search_server',
    '_worker_context_window',
    '_worker_profile_kwargs',
    '_worker_reasoning_effort',
    '_worker_turn_timeout',
    '_worker_version',
    'with_worker_skills',
    'worker_mcp_servers',
]
