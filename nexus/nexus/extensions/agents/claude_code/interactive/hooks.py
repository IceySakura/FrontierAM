"""File-backed bridge for Claude Code PreToolUse interactions.

The command-hook process and the long-lived PTY runtime share a per-session
directory.  Hooks publish one request file and poll for the matching response;
the runtime turns those requests into Nexus ``UserInputRequest`` objects.

This module has no Nexus runtime dependency so it can be executed directly
by Claude Code's command-hook subprocess (``python -m ...hooks``).
"""

from __future__ import annotations

import asyncio
import contextlib
import hashlib
import json
import math
import os
import pathlib
import shutil
import sys
import time
import uuid

from dataclasses import dataclass
from typing import Any, Protocol


INTERACTION_DIR_ENV = 'NEXUS_PTY_INTERACTION_DIR'
SESSION_ID_ENV = 'NEXUS_PTY_SESSION_ID'
TURN_FILE_ENV = 'NEXUS_PTY_TURN_FILE'
HOOK_WAIT_TIMEOUT_ENV = 'NEXUS_PTY_HOOK_WAIT_TIMEOUT'

DEFAULT_HOOK_WAIT_TIMEOUT = 700.0
DEFAULT_HOOK_COMMAND_TIMEOUT = 750.0
POLL_INTERVAL = 0.05
CLAUDE_PLAN_DIR = pathlib.Path('.claude') / 'plans'

_UNATTENDED_ASK_REASON = (
    'This is an unattended task and no user is available. Infer the best decision from the issue, '
    'existing code, tests, and backward-compatibility requirements. Continue implementing and validating '
    'without asking the user again.'
)


@dataclass(frozen=True)
class InteractionRequest:
    """One pending interaction read from a request file."""

    version: int
    session_id: str
    turn_id: int
    interaction_id: str
    tool_name: str
    tool_input: dict[str, Any]
    created_at: float
    timeout_seconds: float = DEFAULT_HOOK_WAIT_TIMEOUT

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> InteractionRequest:
        tool_input = d.get('tool_input')
        return cls(
            version=int(d.get('version', 1)),
            session_id=str(d.get('session_id') or ''),
            turn_id=int(d.get('turn_id', 0)),
            interaction_id=str(d.get('interaction_id') or ''),
            tool_name=str(d.get('tool_name') or ''),
            tool_input=tool_input if isinstance(tool_input, dict) else {},
            created_at=float(d.get('created_at', 0.0)),
            timeout_seconds=float(d.get('timeout_seconds', DEFAULT_HOOK_WAIT_TIMEOUT)),
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            'version': self.version,
            'session_id': self.session_id,
            'turn_id': self.turn_id,
            'interaction_id': self.interaction_id,
            'tool_name': self.tool_name,
            'tool_input': self.tool_input,
            'created_at': self.created_at,
            'timeout_seconds': self.timeout_seconds,
        }


class HookInteractionBridge(Protocol):
    """Replaceable PTY-to-Hook transport used by ``ClaudeInteractiveSession``."""

    async def wait_request(self, *, timeout: float = 0.0) -> InteractionRequest | None: ...

    async def respond(self, interaction_id: str, answer: str | None) -> dict[str, Any]: ...

    async def close(self) -> None: ...


def interaction_key(interaction_id: str) -> str:
    return hashlib.sha256(interaction_id.encode('utf-8')).hexdigest()[:32]


def _paths(directory: pathlib.Path, interaction_id: str) -> tuple[pathlib.Path, pathlib.Path]:
    key = interaction_key(interaction_id)
    return directory / f'{key}.request.json', directory / f'{key}.response.json'


def _atomic_write_json(path: pathlib.Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
    tmp = path.with_name(f'.{path.name}.{os.getpid()}.tmp')
    payload = json.dumps(value, ensure_ascii=False, separators=(',', ':')).encode('utf-8')
    fd = os.open(tmp, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    try:
        with os.fdopen(fd, 'wb') as f:
            f.write(payload)
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp, path)
    except Exception:
        with contextlib.suppress(OSError):
            tmp.unlink()
        raise


def _read_json(path: pathlib.Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding='utf-8'))
    if not isinstance(value, dict):
        raise ValueError(f'{path.name} must contain a JSON object')
    return value


def _normalise_answers(answer: str, tool_input: dict[str, Any]) -> dict[str, str] | None:
    """Map user answer to AskUserQuestion's answers dict.

    Supports two input forms:
    - JSON object: {"question text": "answer", "multi": ["a", "b"]}
    - Plain string: assigned to the single question if there's exactly one.
    """
    questions = tool_input.get('questions')
    items = [q for q in questions if isinstance(q, dict)] if isinstance(questions, list) else []

    with contextlib.suppress(json.JSONDecodeError, TypeError):
        parsed = json.loads(answer)
        if isinstance(parsed, dict):
            return {
                str(k): ', '.join(str(i) for i in v) if isinstance(v, list) else str(v)
                for k, v in parsed.items()
                if v is not None
            } or None

    if len(items) == 1:
        q = str(items[0].get('question') or 'answer')
        return {q: answer}
    return None


def build_hook_output(tool_name: str, tool_input: dict[str, Any], answer: str | None) -> dict[str, Any]:
    """Translate a host answer into Claude Code's PreToolUse hook response."""
    text = answer.strip() if isinstance(answer, str) else ''

    if tool_name == 'AskUserQuestion':
        if text:
            answers = _normalise_answers(text, tool_input)
            if answers:
                return {
                    'hookSpecificOutput': {
                        'hookEventName': 'PreToolUse',
                        'permissionDecision': 'allow',
                        'updatedInput': {**tool_input, 'answers': answers},
                    }
                }
        return {
            'hookSpecificOutput': {
                'hookEventName': 'PreToolUse',
                'permissionDecision': 'deny',
                'permissionDecisionReason': _UNATTENDED_ASK_REASON,
            }
        }

    if tool_name == 'ExitPlanMode':
        if not text or text.casefold() in {'approve', 'allow', 'yes'}:
            return {
                'hookSpecificOutput': {
                    'hookEventName': 'PreToolUse',
                    'permissionDecision': 'allow',
                    'updatedInput': dict(tool_input),
                }
            }
        feedback = text
        if text.casefold() == 'reject':
            feedback = 'The user rejected this plan. Revise it before continuing.'
        elif text.casefold().startswith('reject:'):
            feedback = text.split(':', 1)[1].strip() or 'The user rejected this plan.'
        return {
            'hookSpecificOutput': {
                'hookEventName': 'PreToolUse',
                'permissionDecision': 'deny',
                'permissionDecisionReason': feedback,
            }
        }

    return {
        'hookSpecificOutput': {
            'hookEventName': 'PreToolUse',
            'permissionDecision': 'deny',
            'permissionDecisionReason': f'Unsupported interactive tool: {tool_name}',
        }
    }


# --------------------------------------------------------------------------- #
# Runtime-side bridge (async, used by ClaudeInteractiveSession)
# --------------------------------------------------------------------------- #


class FileHookInteractionBridge:
    """Runtime-side view of one session's request/response directory."""

    def __init__(self, directory: str | os.PathLike[str]) -> None:
        self.directory = pathlib.Path(directory)
        self.directory.mkdir(mode=0o700, parents=True, exist_ok=True)
        self._delivered: set[str] = set()

    def _next_request(self) -> InteractionRequest | None:
        for req_path in self.directory.glob('*.request.json'):
            stem = req_path.name.removesuffix('.request.json')
            if stem in self._delivered:
                continue
            resp_path = self.directory / f'{stem}.response.json'
            if resp_path.exists():
                continue
            try:
                request = InteractionRequest.from_dict(_read_json(req_path))
            except (OSError, ValueError, json.JSONDecodeError):
                continue
            if interaction_key(request.interaction_id) != stem:
                continue
            self._delivered.add(stem)
            return request
        return None

    async def wait_request(self, *, timeout: float = 0.0) -> InteractionRequest | None:
        deadline = time.monotonic() + max(timeout, 0.0)
        while True:
            request = self._next_request()
            if request is not None:
                return request
            if time.monotonic() >= deadline:
                return None
            await asyncio.sleep(min(POLL_INTERVAL, max(deadline - time.monotonic(), 0.0)))

    async def respond(self, interaction_id: str, answer: str | None) -> dict[str, Any]:
        req_path, resp_path = _paths(self.directory, interaction_id)
        if not req_path.exists():
            status = 'expired' if interaction_key(interaction_id) in self._delivered else 'unknown'
            return {'status': status, 'interaction_id': interaction_id}
        try:
            request = InteractionRequest.from_dict(_read_json(req_path))
        except (OSError, ValueError, json.JSONDecodeError):
            request = None
        if request is not None and time.time() >= request.created_at + request.timeout_seconds:
            return {'status': 'expired', 'interaction_id': interaction_id}
        if not resp_path.exists():
            _atomic_write_json(resp_path, {'interaction_id': interaction_id, 'answer': answer})
        # The Hook may have crossed its deadline and cleaned the request between
        # the checks above and the atomic response write. In that case the write
        # is still truthfully "submitted", but nothing will consume the file;
        # remove the orphan without claiming whether the Hook accepted it.
        if not req_path.exists():
            with contextlib.suppress(OSError):
                resp_path.unlink()
        return {'status': 'written', 'interaction_id': interaction_id}

    async def close(self) -> None:
        await asyncio.to_thread(shutil.rmtree, self.directory, True)


# --------------------------------------------------------------------------- #
# Hook-side entry point (synchronous, run by Claude Code's command hook)
# --------------------------------------------------------------------------- #


def _read_turn_id() -> int:
    turn_file = os.environ.get(TURN_FILE_ENV)
    if not turn_file:
        return 0
    try:
        return int((pathlib.Path(turn_file).read_text(encoding='utf-8') or '0').strip() or 0)
    except (OSError, ValueError):
        return 0


def _tool_input_from_stdin() -> tuple[str, dict[str, Any]]:
    try:
        payload = json.load(sys.stdin)
    except (json.JSONDecodeError, OSError):
        payload = {}
    if not isinstance(payload, dict):
        payload = {}
    tool_input = payload.get('tool_input') or payload.get('toolInput') or payload.get('input') or {}
    if not isinstance(tool_input, dict):
        tool_input = {}
    iid = payload.get('tool_use_id') or payload.get('toolUseId') or payload.get('id') or f'hook-{uuid.uuid4()}'
    return str(iid), tool_input


def _latest_plan() -> tuple[pathlib.Path, str] | None:
    """Return Claude Code's most recently updated plan, if one exists."""
    plan_dir = pathlib.Path(os.environ.get('HOME', '/root')) / CLAUDE_PLAN_DIR
    try:
        candidates = [path for path in plan_dir.glob('*.md') if path.is_file()]
        latest = max(candidates, key=lambda path: path.stat().st_mtime_ns)
        content = latest.read_text(encoding='utf-8', errors='replace')
    except (OSError, ValueError):
        return None
    return (latest, content) if content.strip() else None


def _context_tool_input(tool_name: str, tool_input: dict[str, Any]) -> dict[str, Any]:
    """Enrich Hook context without changing the input returned to Claude Code."""
    context_input = dict(tool_input)
    if tool_name != 'ExitPlanMode' or context_input.get('plan'):
        return context_input
    latest_plan = _latest_plan()
    if latest_plan is None:
        return context_input
    plan_path, plan = latest_plan
    context_input['plan'] = plan
    context_input['plan_path'] = str(plan_path)
    return context_input


def run_hook(tool_name: str, *, wait_timeout: float | None = None) -> dict[str, Any]:
    """Command-hook entry point: publish request, poll response, clean up."""
    iid, tool_input = _tool_input_from_stdin()
    directory_value = os.environ.get(INTERACTION_DIR_ENV)
    if not directory_value:
        return build_hook_output(tool_name, tool_input, None)

    directory = pathlib.Path(directory_value)
    directory.mkdir(mode=0o700, parents=True, exist_ok=True)
    req_path, resp_path = _paths(directory, iid)

    if wait_timeout is None:
        try:
            wait_timeout = float(os.environ.get(HOOK_WAIT_TIMEOUT_ENV, DEFAULT_HOOK_WAIT_TIMEOUT))
        except ValueError:
            wait_timeout = DEFAULT_HOOK_WAIT_TIMEOUT
    if not math.isfinite(wait_timeout) or wait_timeout < 0:
        wait_timeout = DEFAULT_HOOK_WAIT_TIMEOUT

    _atomic_write_json(
        req_path,
        {
            'version': 1,
            'session_id': os.environ.get(SESSION_ID_ENV, ''),
            'turn_id': _read_turn_id(),
            'interaction_id': iid,
            'tool_name': tool_name,
            'tool_input': _context_tool_input(tool_name, tool_input),
            'created_at': time.time(),
            'timeout_seconds': wait_timeout,
        },
    )

    deadline = time.monotonic() + wait_timeout
    answer: str | None = None
    while time.monotonic() < deadline:
        if resp_path.exists():
            try:
                resp = _read_json(resp_path)
                answer = resp.get('answer')
                if answer is not None and not isinstance(answer, str):
                    answer = None
            except (OSError, ValueError, json.JSONDecodeError):
                pass
            break
        time.sleep(POLL_INTERVAL)

    # Clean up both files after consuming.
    with contextlib.suppress(OSError):
        req_path.unlink()
    with contextlib.suppress(OSError):
        resp_path.unlink()

    return build_hook_output(tool_name, tool_input, answer)


def main() -> int:
    tool_name = sys.argv[1] if len(sys.argv) > 1 else ''
    if not tool_name:
        sys.stderr.write('Usage: python -m nexus.extensions.agents.claude_code.interactive.hooks <tool_name>\n')
        return 1
    print(json.dumps(run_hook(tool_name), ensure_ascii=False))
    return 0


__all__ = [
    'DEFAULT_HOOK_WAIT_TIMEOUT',
    'DEFAULT_HOOK_COMMAND_TIMEOUT',
    'FileHookInteractionBridge',
    'HOOK_WAIT_TIMEOUT_ENV',
    'HookInteractionBridge',
    'INTERACTION_DIR_ENV',
    'InteractionRequest',
    'POLL_INTERVAL',
    'SESSION_ID_ENV',
    'TURN_FILE_ENV',
    'build_hook_output',
    'interaction_key',
    'main',
    'run_hook',
]


if __name__ == '__main__':
    raise SystemExit(main())
