"""Harbor-owned unattended interaction hooks for Claude Code."""

from __future__ import annotations

import json
import os
import shlex
import stat
import sys
import tempfile

from pathlib import Path
from typing import Any


ASKUSER_HOOK_SCRIPT_PATH = Path('/tmp/_nexus_askuser_hook.py')
CLAUDE_SETTINGS_PATH = Path('~/.claude/settings.json')

_EXITPLAN_HOOK_CMD = (
    'echo \'{"hookSpecificOutput":{"hookEventName":"PreToolUse","permissionDecision":"allow","updatedInput":{}}}\''
)
_ASKUSER_HOOK_SCRIPT = r"""
import json
import sys

data = json.load(sys.stdin)
tool_input = data.get('tool_input') or data.get('toolInput') or {}
questions = tool_input.get('questions', [])
parts = []
for question in questions:
    options = question.get('options', [])
    selected = options[0].get('label', 'first option') if options else 'first option'
    parts.append(f'"{question.get("question", "?")}"="{selected}"')
reason = 'User has answered your questions: ' + ', '.join(parts) + '. You can now continue with the user\'s answers in mind.'
output = {'hookSpecificOutput': {'hookEventName': 'PreToolUse', 'permissionDecision': 'deny', 'permissionDecisionReason': reason}}
print(json.dumps(output))
""".lstrip()


def _required_pre_tool_use_hooks(hook_script_path: Path) -> list[dict[str, Any]]:
    return [
        {
            'matcher': 'ExitPlanMode',
            'hooks': [{'type': 'command', 'command': _EXITPLAN_HOOK_CMD}],
        },
        {
            'matcher': 'AskUserQuestion',
            'hooks': [
                {
                    'type': 'command',
                    'command': f'{shlex.quote(sys.executable)} {shlex.quote(str(hook_script_path))}',
                }
            ],
        },
    ]


def install_unattended_interaction_hooks(
    *,
    settings_path: Path | str = CLAUDE_SETTINGS_PATH,
    hook_script_path: Path | str = ASKUSER_HOOK_SCRIPT_PATH,
) -> None:
    """Append Harbor's unattended hooks without replacing caller-owned hooks."""
    settings = Path(settings_path).expanduser()
    hook_script = Path(hook_script_path).expanduser()
    hook_script.parent.mkdir(parents=True, exist_ok=True)
    hook_script.write_text(_ASKUSER_HOOK_SCRIPT, encoding='utf-8')

    payload: dict[str, Any] = {}
    existing_mode: int | None = None
    if settings.exists():
        loaded = json.loads(settings.read_text(encoding='utf-8') or '{}')
        if not isinstance(loaded, dict):
            raise ValueError(f'Claude settings must contain a JSON object: {settings}')
        payload = loaded
        existing_mode = stat.S_IMODE(settings.stat().st_mode)

    hooks = payload.setdefault('hooks', {})
    if not isinstance(hooks, dict):
        raise ValueError(f'Claude settings hooks must be an object: {settings}')
    pre_tool_use = hooks.setdefault('PreToolUse', [])
    if not isinstance(pre_tool_use, list):
        raise ValueError(f'Claude settings hooks.PreToolUse must be a list: {settings}')
    for required in _required_pre_tool_use_hooks(hook_script):
        if required not in pre_tool_use:
            pre_tool_use.append(required)

    settings.parent.mkdir(parents=True, exist_ok=True)
    fd, raw_tmp_path = tempfile.mkstemp(prefix=f'.{settings.name}.', suffix='.tmp', dir=settings.parent)
    tmp_path = Path(raw_tmp_path)
    try:
        with os.fdopen(fd, 'w', encoding='utf-8') as tmp_file:
            json.dump(payload, tmp_file, indent=2, ensure_ascii=False)
            tmp_file.write('\n')
        if existing_mode is not None:
            tmp_path.chmod(existing_mode)
        os.replace(tmp_path, settings)
    finally:
        tmp_path.unlink(missing_ok=True)


if __name__ == '__main__':
    install_unattended_interaction_hooks()


__all__ = ['install_unattended_interaction_hooks']
