"""What a print-mode Codex worker runs under: its paths, commands and env.

``worker_env`` is this backend's ``SessionManager.harness_env``; ``actor.envs``
adds the host credentials around it. The rest is what the manager declares as its
class attributes.
"""

import json

from collections.abc import Mapping, Sequence
from types import MappingProxyType


MEMORY_PATH = '.codex/AGENTS.md'
"""Where Codex reads its user-level instructions, relative to the sandbox home.

Codex resolves the global layer under ``$CODEX_HOME``, which ``runtime`` sets to
``~/.codex``, taking the first non-empty of ``AGENTS.override.md`` and ``AGENTS.md``.

The project layer -- ``AGENTS.md`` at each level from the git root down to the cwd --
is not written: ``capture_git_patch`` stages the whole working tree
(``utils/git.py:118``), so a file under the repo would land in the extracted patch.
"""

SKILLS_PATH = '.agents/skills'
"""Where Codex reads user-level skills, relative to the sandbox home.

``.agents/skills`` is the current convention for both user and project level;
``~/.codex/skills`` is still read for Codex's own built-ins. One directory per
skill, named as the skill is.
"""

COMMAND_PREFIX = '/'
"""What a message must start with to run one of this run's installed skills."""

BUILTIN_COMMANDS: Mapping[str, str] = MappingProxyType({})
"""The harness's own commands this backend offers. None declared yet."""

DENIED_COMMANDS: frozenset[str] = frozenset()
"""Commands that discard, detach, or relocate the session this backend holds.

None declared yet; each name here has to be one Codex carries in ``exec`` mode.
"""

UNSUPPORTED_COMMANDS: frozenset[str] = frozenset()
"""Commands that produce no assistant turn, so nothing signals completion.

None declared yet; see :data:`DENIED_COMMANDS`.
"""


def worker_env(config: Mapping[str, object]) -> dict[str, str]:
    """The Codex settings this backend's worker process runs under.

    Empty so far: what this backend configures reaches Codex through the
    ``config.toml`` and the command line the sandbox side writes, and
    ``CODEX_HOME`` names the per-session directory holding them.
    """
    del config
    return {}


def mcp_overrides(servers: Sequence[Mapping[str, object]]) -> list[str]:
    """``-c`` arguments registering ``servers`` for one invocation.

    Takes the ``{name, command, args, env}`` entries ``mcp_config.worker_servers``
    builds. Every turn passes these again, including a ``resume``: a thread carries
    its history, while the MCP servers belong to the invocation that declares them.

    An entry is skipped when it names no command, which a remote server does --
    ``--url`` transport is not built here.
    """
    overrides: list[str] = []
    for server in servers:
        name = str(server.get('name') or '')
        command = str(server.get('command') or '')
        if not name or not command:
            continue
        overrides += ['-c', f'mcp_servers.{name}.command={json.dumps(command)}']
        args = server.get('args')
        if isinstance(args, Sequence) and not isinstance(args, (str, bytes)):
            overrides += ['-c', f'mcp_servers.{name}.args={json.dumps([str(arg) for arg in args])}']
        env = server.get('env')
        if isinstance(env, Mapping) and env:
            pairs = ', '.join(f'{key} = {json.dumps(str(value))}' for key, value in env.items())
            overrides += ['-c', f'mcp_servers.{name}.env={{{pairs}}}']
    return overrides


__all__ = [
    'BUILTIN_COMMANDS',
    'COMMAND_PREFIX',
    'DENIED_COMMANDS',
    'MEMORY_PATH',
    'SKILLS_PATH',
    'UNSUPPORTED_COMMANDS',
    'mcp_overrides',
    'worker_env',
]
