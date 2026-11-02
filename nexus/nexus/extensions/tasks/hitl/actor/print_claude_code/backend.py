"""Host-side backend for the labelled Claude Code print mode.

Owns the sandbox lifecycle for one manager: install once, open a proxy per
session, run turns, inject answers for deferred AskUserQuestions, tear down.

Rewritten rather than reused from the legacy print backend because the session
lifecycle differs -- a session now has an explicit open/close, since it owns a
proxy -- but the *behaviour* outside that is deliberately unchanged. In
particular these were all learned from failures and are reproduced exactly:

* ``resume`` is decided by whether this backend has already run a turn for the
  session, not by anything the sandbox reports;
* an answer-injection turn sends **no prompt** -- the answer file is what CC
  resumes on, and a prompt as well would start a fresh exchange;
* a deferred tool makes the turn ``success=False`` even though the process
  exited 0, because the turn is not finished until the question is answered;
* a missing exit code counts as success.

Trace is not this class's concern: the label travels with the request from the
sandbox proxy, so nothing here installs or clears processors.
"""

import json
import logging

from typing import Any

from nexus.extensions.tasks.hitl.actor.memory import place_user_memory
from nexus.extensions.tasks.hitl.actor.print_claude_code import runtime as sandbox
from nexus.extensions.tasks.hitl.actor.skills import place_worker_skills
from nexus.runtime import Runtime
from nexus.runtime.models import InvokeRequest


logger = logging.getLogger(__name__)

DEFAULT_CC_TURN_TIMEOUT_SECONDS = 7200

# Config keys forwarded to the sandbox's one-time environment setup. Listed
# explicitly: the sandbox call is a remote invoke, so an unexpected key becomes a
# signature error inside the container.
#
# ``working_dir`` is not among them: it comes from the session being opened, not from
# the worker config, which never carried it.
_SETUP_KEYS = (
    'version',
    'npm_registry',
    'binary_url',
    'binary_checksum',
    'mcp_servers',
    'extra_env',
    'config_file_patches',
    'pre_start_scripts',
)


class LabelledPrintBackend:
    """Sandbox orchestration for labelled print/resume turns."""

    def __init__(self, config: dict[str, Any], *, memory_path: str = '', skills_path: str = '') -> None:
        self._config = config
        self._memory_path = memory_path
        self._skills_path = skills_path
        self._env_ready = False
        # Sessions that have completed at least one turn, and therefore must be
        # resumed rather than started. Tracked host-side because the sandbox
        # cannot distinguish "never ran" from "ran and produced nothing".
        self._started_sessions: set[str] = set()
        self._open_sessions: dict[str, str] = {}

    @property
    def open_sessions(self) -> dict[str, str]:
        """session_id -> label, for every session with a live proxy."""
        return dict(self._open_sessions)

    async def ensure_env(self, runtime: Runtime, *, working_dir: str) -> None:
        """Install Claude Code, deploy hooks, place the user memory and the skills. Idempotent.

        ``working_dir`` is where a pre-start script runs, so it is the session's own
        rather than a default: a run whose code sits in ``/testbed`` would otherwise
        have every script start in a directory nothing created.
        """
        if self._env_ready:
            return
        if not working_dir.startswith('/'):
            raise ValueError(f'working_dir must be an absolute path, got {working_dir!r}')
        kwargs = {key: self._config[key] for key in _SETUP_KEYS if key in self._config}
        kwargs.setdefault('version', 'latest')
        await runtime.invoke(sandbox.setup_env, working_dir=working_dir, **kwargs)
        # Before any turn: each turn is its own Claude process, and every one of
        # them reads the file and the skill directory on startup.
        await place_user_memory(runtime, self._config, memory_path=self._memory_path)
        await place_worker_skills(runtime, self._config, skills_path=self._skills_path)
        self._env_ready = True

    async def open_session(self, runtime: Runtime, *, session_id: str, label: str, working_dir: str) -> int:
        """Start this session's labelled proxy, returning its port.

        Called before the first turn rather than lazily inside it, so a
        misconfigured label fails at session start rather than mid-turn.
        """
        await self.ensure_env(runtime, working_dir=working_dir)
        result = await runtime.invoke(
            sandbox.open_session,
            session_id=session_id,
            label=label,
            timeout=float(self._config.get('timeout', DEFAULT_CC_TURN_TIMEOUT_SECONDS)),
            conversion_log_level=str(self._config.get('conversion_log_level', 'INFO')),
            thinking_preset=str(self._config.get('thinking_preset', 'hunyuan')),
        )
        port = int(result.get('proxy_port', 0)) if isinstance(result, dict) else 0
        self._open_sessions[session_id] = label
        return port

    async def send_turn(
        self,
        runtime: Runtime,
        *,
        session_id: str,
        prompt: str,
        working_dir: str,
        timeout: float,
        extra_env: dict[str, str] | None = None,
        extra_cli_args: list[str] | None = None,
        pending_tool_use_id: str | None = None,
        pending_answers: dict[str, str] | None = None,
    ) -> dict[str, Any]:
        """Run one turn, answering a pending question first when there is one."""
        if session_id not in self._open_sessions:
            raise RuntimeError(f'session {session_id} was never opened; call open_session() first')

        is_answer_resume = False
        if pending_tool_use_id and pending_answers:
            await runtime.invoke(
                sandbox.write_answer,
                session_id=session_id,
                tool_use_id=pending_tool_use_id,
                answers=pending_answers,
            )
            is_answer_resume = True

        resume = session_id in self._started_sessions
        # ``disable_invoke_wait_timeout``: the sandbox runs the turn under ``timeout``,
        # which is a whole agent loop rather than one RPC. Left on, the host stops
        # waiting after the runtime client's own request timeout and the turn is
        # reported lost while it is still running.
        raw = await runtime.invoke(
            InvokeRequest.build(
                sandbox.run_turn,
                session_id=session_id,
                # No prompt on an answer resume: CC continues from the answer file.
                prompt='' if is_answer_resume else prompt,
                resume=resume,
                working_dir=working_dir,
                timeout=timeout,
                extra_env=extra_env,
                extra_cli_args=extra_cli_args,
                venv_id=runtime.default_venv_id,
                disable_invoke_wait_timeout=True,
            )
        )
        self._started_sessions.add(session_id)

        return _interpret_turn(raw, session_id=session_id)

    async def close_session(self, runtime: Runtime, *, session_id: str) -> None:
        """Stop one session's proxy and forget its resume state."""
        if session_id not in self._open_sessions:
            return
        try:
            await runtime.invoke(sandbox.close_session, session_id=session_id)
        except Exception as exc:  # noqa: BLE001 - teardown must not mask the real error
            logger.warning('Failed to close print session %s: %s', session_id, exc)
        self._open_sessions.pop(session_id, None)
        self._started_sessions.discard(session_id)

    async def shutdown(self, runtime: Runtime) -> None:
        """Stop every proxy. Safe to call when setup never happened."""
        if self._env_ready:
            try:
                await runtime.invoke(sandbox.teardown_env)
            except Exception as exc:  # noqa: BLE001
                logger.warning('LabelledPrintBackend teardown failed: %s', exc)
            self._env_ready = False
        self._open_sessions.clear()
        self._started_sessions.clear()


def _interpret_turn(raw: Any, *, session_id: str) -> dict[str, Any]:
    """Turn the sandbox's raw result into the manager's turn dict."""
    raw_dict = raw if isinstance(raw, dict) else {}
    stdout = str(raw_dict.get('output', ''))

    exit_code_value = raw_dict.get('exit_code')
    exit_code = int(exit_code_value) if exit_code_value is not None else 0
    process_ok = exit_code == 0

    assistant_messages = extract_assistant_messages(stdout)
    output_text = ''.join(assistant_messages) or _extract_result_text(stdout)
    # Only look for a pending question on a clean exit: a crashed turn's stdout
    # is truncated, so an unanswered tool_use there means "we stopped", not
    # "Claude is waiting".
    deferred_tool = parse_deferred_tools(stdout) if process_ok else None
    result_event = _extract_result_event(stdout)

    return {
        # Not finished while a question is outstanding, even though CC exited 0.
        'success': process_ok and deferred_tool is None,
        'session_id': session_id,
        'output': output_text,
        'assistant_messages': assistant_messages,
        'steps': extract_steps(stdout),
        'stderr': str(raw_dict.get('stderr', '')),
        'exit_code': exit_code,
        'execution_time': raw_dict.get('execution_time', 0),
        'events': [],
        'result_event': result_event or {},
        'deferred_tool': deferred_tool,
    }


def _iter_events(stdout: str):
    """Yield parsed events from CC's stream-json output, skipping junk lines."""
    for line in stdout.strip().splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            event = json.loads(line)
        except (json.JSONDecodeError, ValueError):
            continue
        if isinstance(event, dict):
            yield event


def parse_deferred_tools(stdout: str) -> dict[str, Any] | None:
    """Return the last unanswered ``AskUserQuestion``, or None.

    "Unanswered" is decided by pairing tool_use ids against tool_result ids in
    the same stream: the hook defers by not answering, so an id with no result
    is a question waiting on the orchestrator.
    """
    tool_uses: dict[str, tuple[str, dict[str, Any]]] = {}
    answered_ids: set[str] = set()

    for event in _iter_events(stdout):
        etype = event.get('type')

        if etype == 'assistant':
            content = (event.get('message') or {}).get('content')
            if isinstance(content, list):
                for block in content:
                    if not isinstance(block, dict):
                        continue
                    if block.get('type') == 'tool_use' and block.get('name') == 'AskUserQuestion':
                        tool_id = block.get('id', '')
                        if tool_id:
                            tool_uses[tool_id] = (block['name'], block.get('input', {}))

        elif etype in ('tool', 'tool_result', 'user'):
            content = event.get('content', (event.get('message') or {}).get('content', ''))
            if isinstance(content, list):
                for block in content:
                    if isinstance(block, dict) and block.get('type') == 'tool_result':
                        answered_ids.add(block.get('tool_use_id', ''))
            tool_use_id = event.get('tool_use_id', '')
            if tool_use_id:
                answered_ids.add(tool_use_id)

    # Latest first: with several questions in one turn, the last one is the one
    # Claude is actually blocked on.
    for tool_id, (name, tool_input) in reversed(list(tool_uses.items())):
        if tool_id not in answered_ids:
            return {'tool_use_id': tool_id, 'name': name, 'input': tool_input}

    return None


def extract_steps(stdout: str) -> list[dict[str, Any]]:
    """The turn's speech and tool calls, in the order Claude Code emitted them.

    An ``assistant`` event's ``content`` interleaves ``text`` and ``tool_use``
    blocks, so walking them in order is what places a run of tool calls between the
    two messages it happened between. Serialised as dicts rather than ``Step``
    values because this crosses the sandbox boundary.
    """
    steps: list[dict[str, Any]] = []
    for event in _iter_events(stdout):
        if event.get('type') != 'assistant':
            continue
        content = (event.get('message') or {}).get('content')
        if not isinstance(content, list):
            continue
        for block in content:
            if not isinstance(block, dict):
                continue
            kind = block.get('type')
            if kind == 'text' and (text := str(block.get('text') or '')).strip():
                steps.append({'type': 'text', 'text': text})
            elif kind == 'tool_use' and (name := str(block.get('name') or '')):
                raw = block.get('input')
                steps.append({'type': 'tool_use', 'name': name, 'input': raw if isinstance(raw, dict) else {}})
    return steps


def extract_assistant_messages(stdout: str) -> list[str]:
    """One string per assistant event that carried text.

    Tool-only events are skipped, so the last item is the worker's final spoken
    message -- usually its summary, which is what a recipe wants to show.
    """
    messages: list[str] = []
    for event in _iter_events(stdout):
        if event.get('type') != 'assistant':
            continue
        content = (event.get('message') or {}).get('content')
        if not isinstance(content, list):
            continue
        text = ''.join(
            block['text']
            for block in content
            if isinstance(block, dict) and block.get('type') == 'text' and block.get('text')
        )
        if text:
            messages.append(text)
    return messages


def _extract_result_event(stdout: str) -> dict[str, Any] | None:
    """The last ``result`` event, which carries the turn's summary and usage."""
    result_event = None
    for event in _iter_events(stdout):
        if event.get('type') == 'result':
            result_event = event
    return result_event


def _extract_result_text(stdout: str) -> str:
    """Fallback output when no assistant text was emitted."""
    result_event = _extract_result_event(stdout)
    return str(result_event.get('result', '')) if result_event else ''


__all__ = [
    'DEFAULT_CC_TURN_TIMEOUT_SECONDS',
    'LabelledPrintBackend',
    'extract_assistant_messages',
    'extract_steps',
    'parse_deferred_tools',
]
