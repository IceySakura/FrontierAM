"""What a PTY Claude Code worker runs under: its paths, commands and env.

``worker_env`` is this backend's ``SessionManager.harness_env``; ``actor.envs``
adds the host credentials around it. The rest is what the manager declares as its
class attributes.
"""

import os

from collections.abc import Mapping
from types import MappingProxyType

from nexus.extensions.tasks.hitl.utils import coerce


MEMORY_PATH = '.claude/CLAUDE.md'
"""Where Claude Code reads its user memory, relative to the sandbox home.

User-level rather than in the repository under test: ``capture_git_patch`` stages
the whole working tree (``utils/git.py:118``), so a project-root file would land in
the extracted patch.
"""

SKILLS_PATH = '.claude/skills'
"""Where Claude Code reads installed skills, relative to the sandbox home."""

COMMAND_PREFIX = '/'
"""What a message must start with to name one of Claude Code's own commands."""

BUILTIN_COMMANDS = MappingProxyType({
    'simplify': 'Look over the change for duplication and needless complexity, and clean it up.',
})
"""Built-in Claude Code commands a run may offer, name -> what it does.

Offered only where ``worker_config['builtin_commands']`` names one. A command
belongs here only when it produces a reply every time, which is what signals turn
completion.
"""

DENIED_COMMANDS = frozenset({
    'clear',
    'reset',
    'new',
    'resume',
    'continue',
    'rewind',
    'checkpoint',
    'undo',
    'branch',
    'fork',
    'exit',
    'quit',
    'background',
    'bg',
    'stop',
    'teleport',
    'desktop',
    'mobile',
    'remote-control',
    'rc',
})
"""Commands that discard, detach, or relocate the session this backend holds."""

UNSUPPORTED_COMMANDS = frozenset({
    'compact',
    'context',
    'cost',
    'usage',
    'stats',
    'status',
    'doctor',
    'checkup',
    'help',
    'skills',
    'hooks',
    'goal',
    'loop',
    'proactive',
    'model',
    'effort',
    'fast',
    'config',
    'settings',
    'permissions',
    'allowed-tools',
    'sandbox',
    'cd',
    'add-dir',
    'theme',
    'color',
    'focus',
    'tui',
    'voice',
    'keybindings',
    'statusline',
})
"""Commands that produce no assistant turn, so nothing signals completion.

``goal`` holds state across turns: clearing it once reached is itself turn-less, and
a second one opens a confirmation screen waiting on a keypress.
"""


# Forwarded from the host when present: one turn's output cap, including thinking.
_PASSTHROUGH = ('CLAUDE_CODE_MAX_OUTPUT_TOKENS',)

DEFAULT_CC_MODEL = 'claude-opus-4-8'

DEFAULT_AUTO_COMPACT_WINDOW = '950000'


def with_extended_context(model: str) -> str:
    """Append the 1M-context ``[1m]`` suffix unless the name already carries it."""
    return model if model.endswith('[1m]') else f'{model}[1m]'


def worker_env(config: Mapping[str, object]) -> dict[str, str]:
    """The Claude Code settings this backend's worker process runs under.

    ``ANTHROPIC_MODEL`` names what Claude Code believes it is: the host proxy
    substitutes the real model on every call, so this governs only its self-report
    and its context accounting, where the ``[1m]`` suffix enables the extended
    window.

    ``config`` entries override the defaults.
    """
    env = {key: value for key in _PASSTHROUGH if (value := os.environ.get(key))}
    env.setdefault(
        'CLAUDE_CODE_AUTO_COMPACT_WINDOW',
        os.environ.get('CLAUDE_CODE_AUTO_COMPACT_WINDOW', DEFAULT_AUTO_COMPACT_WINDOW),
    )
    env.setdefault('CLAUDE_CODE_DISABLE_NONESSENTIAL_TRAFFIC', '1')
    env.setdefault('CLAUDE_CODE_DISABLE_EXPERIMENTAL_BETAS', '1')
    env.setdefault('CLAUDE_CODE_ATTRIBUTION_HEADER', '0')
    env.setdefault('CLAUDE_CODE_DISABLE_GIT_INSTRUCTIONS', '1')

    cc_model = coerce.optional_string(config.get('cc_model')) or DEFAULT_CC_MODEL
    env.setdefault('ANTHROPIC_MODEL', with_extended_context(cc_model))

    auto_compact_window = coerce.optional_int(config.get('auto_compact_window'))
    if auto_compact_window is not None:
        env['CLAUDE_CODE_AUTO_COMPACT_WINDOW'] = str(auto_compact_window)

    autocompact_pct_override = coerce.optional_int(config.get('autocompact_pct_override'))
    if autocompact_pct_override is not None:
        env['CLAUDE_AUTOCOMPACT_PCT_OVERRIDE'] = str(autocompact_pct_override)

    # Tool search is a beta, so this also clears the blanket beta switch above.
    if coerce.optional_bool(config.get('enable_tool_search'), default=False):
        env['ENABLE_TOOL_SEARCH'] = coerce.optional_string(config.get('tool_search_mode')) or 'true'
        env['CLAUDE_CODE_DISABLE_EXPERIMENTAL_BETAS'] = '0'
    return env


__all__ = [
    'BUILTIN_COMMANDS',
    'COMMAND_PREFIX',
    'DEFAULT_AUTO_COMPACT_WINDOW',
    'DEFAULT_CC_MODEL',
    'DENIED_COMMANDS',
    'MEMORY_PATH',
    'SKILLS_PATH',
    'UNSUPPORTED_COMMANDS',
    'with_extended_context',
    'worker_env',
]
