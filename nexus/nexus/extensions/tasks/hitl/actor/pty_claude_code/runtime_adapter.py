"""Thin adapter over the upstream interactive PTY runtime.

The only module that imports
``nexus.extensions.agents.claude_code.interactive.runtime``; everything the upstream
exposes is reached through here.

It also concentrates five upstream behaviours that fail *silently* if handled
wrong. Each is annotated at its site below; in summary:

1. The pending-input status string is ``needs_user_input`` — comparing against
   anything else simply never matches, and the interaction branch never fires.
2. ``enable_input_bridge`` defaults to ``False``, and without it AskUserQuestion
   is denied unattended: the worker just never asks anything.
3. A turn that neither completes nor asks for input destroys the runtime-side
   session, so the next send silently starts a *fresh* Claude with no history.
4. A pending-input result carries ``success=True`` with empty ``output``, so
   callers must branch on ``UserInputRequest.source`` before ``success``.
5. An answer to more than one question must be JSON keyed by question text, or
   the hook denies the tool as unattended and the worker stops asking.
"""

import json
import logging
import math

from collections.abc import Mapping, Sequence
from typing import Any

from nexus.core import UserInputRequest
from nexus.extensions.agents.claude_code.interactive.runtime import (
    close_all_claude_code_interactive_sessions,
    close_claude_code_interactive_session,
    respond_and_wait_claude_code_interactive_session,
    run_claude_code_interactive_session_turn,
)
from nexus.extensions.agents.installed_common.multiturn import build_user_input_request
from nexus.extensions.tasks.hitl.actor.pty_claude_code import labelled_runtime
from nexus.runtime import Runtime
from nexus.runtime.models import InvokeRequest


logger = logging.getLogger(__name__)

# UPSTREAM FACT 1: the exact literal used by ClaudeInteractiveSession. Defining it
# once here keeps the string out of the manager and the recipe loop entirely —
# they branch on UserInputRequest.source instead.
STATUS_NEEDS_INPUT = 'needs_user_input'

AGENT_LABEL = 'Claude Code'

# The one hook tool whose answer has a shape. ExitPlanMode reads a bare verb.
ASK_USER_QUESTION = 'AskUserQuestion'

# An interactive turn runs the worker's whole agent loop, so it needs far longer
# than the upstream primitive's own 3600s default. Applied unless the worker
# config sets ``timeout`` explicitly.
DEFAULT_TURN_TIMEOUT_SECONDS = 7200

# How long the sandbox hook waits for an answer before taking its unattended
# fallback, and how long Claude Code lets that hook command run. Upstream defaults
# to 700/750; a user simulator that reads the repository between turns takes
# longer than that. The command timeout must exceed the wait, or the hook is
# killed while still waiting (``_coerce_hook_timeouts`` refuses the reverse).
DEFAULT_HOOK_WAIT_SECONDS = 1200.0
DEFAULT_HOOK_COMMAND_SECONDS = 1250.0

# Config keys the upstream per-turn primitive accepts. Anything else in the
# worker config dict (the host-side LLM handler's settings, the sandbox env) is not
# its business.
RUNTIME_CONFIG_KEYS = frozenset({
    'timeout',
    'version',
    'npm_registry',
    'binary_url',
    'binary_checksum',
    'mcp_servers',
    'extra_cli_args',
    'config_file_patches',
    'pre_start_scripts',
    'conversion_log_level',
    'thinking_preset',
    'rows',
    'cols',
    'prompt_settle_seconds',
    'log_frames',
    'hook_wait_timeout',
    'hook_command_timeout',
})


def runtime_kwargs(config: Mapping[str, object]) -> dict[str, object]:
    """Select the upstream-accepted subset of a worker config dict.

    UPSTREAM FACT 2: ``enable_input_bridge`` is forced on and deliberately not
    configurable. It gates the whole hook bridge (``runtime.py`` builds a
    ``FileHookInteractionBridge`` only when it is true); with it off, the PTY
    hook denies AskUserQuestion as unattended and no interaction ever reaches
    the host. Nothing raises — the worker merely stops asking questions, which
    reads like a model quirk rather than a wiring bug. This backend exists to
    carry those interactions, so the flag is part of the contract.
    """
    kwargs = {key: value for key, value in config.items() if key in RUNTIME_CONFIG_KEYS}
    kwargs.setdefault('timeout', DEFAULT_TURN_TIMEOUT_SECONDS)
    kwargs.setdefault('hook_wait_timeout', DEFAULT_HOOK_WAIT_SECONDS)
    kwargs.setdefault('hook_command_timeout', DEFAULT_HOOK_COMMAND_SECONDS)
    kwargs['enable_input_bridge'] = True
    return kwargs


async def open_session(
    runtime: Runtime,
    *,
    session_id: str,
    label: str,
    working_dir: str,
    config: Mapping[str, object],
    extra_env: Mapping[str, str] | None = None,
) -> dict[str, Any]:
    """Open a PTY session up front so its proxy carries ``label``.

    Registers the session under ``session_id``, which :func:`run_turn` then
    reuses. Claude itself is still spawned lazily on the first turn.
    """
    # runtime_kwargs returns dict[str, object], so the unpacked values read as object
    # against invoke's typed parameters. RUNTIME_CONFIG_KEYS is what keeps them right.
    return await runtime.invoke(  # ty: ignore[no-matching-overload]
        labelled_runtime.open_session,
        session_id=session_id,
        label=label,
        working_dir=working_dir,
        extra_env=dict(extra_env or {}),
        **runtime_kwargs(config),
    )


async def session_label(runtime: Runtime, *, session_id: str) -> dict[str, Any]:
    """Report the label on a session's proxy, for assertions and diagnostics."""
    return await runtime.invoke(labelled_runtime.session_label, session_id=session_id)


async def run_turn(
    runtime: Runtime,
    *,
    session_id: str,
    prompt: str,
    working_dir: str,
    config: Mapping[str, object],
    extra_env: Mapping[str, str] | None = None,
) -> dict[str, Any]:
    """Start or reuse the runtime-side PTY session and run one user turn.

    ``disable_invoke_wait_timeout`` hands the deadline to the sandbox, where
    ``send_turn`` holds one built from ``timeout``. Without it the host stops waiting
    after the runtime client's own ``request_timeout``, which is set for a single RPC
    rather than for a whole agent loop, and the turn is reported lost while it is still
    running.

    ``venv_id`` is filled here. ``Runtime.invoke`` supplies its default only for a
    callable, and a ready-made request reaches the sandbox as written. A turn landing
    outside the venv finds none of the sessions ``open_session`` registered.
    """
    return await runtime.invoke(
        InvokeRequest.build(
            run_claude_code_interactive_session_turn,
            session_id=session_id,
            prompt=prompt,
            working_dir=working_dir,
            extra_env=dict(extra_env or {}),
            venv_id=runtime.default_venv_id,
            disable_invoke_wait_timeout=True,
            **runtime_kwargs(config),  # ty: ignore[invalid-argument-type]
        )
    )


async def respond(
    runtime: Runtime,
    *,
    session_id: str,
    interaction_id: str,
    answer: str,
) -> dict[str, Any]:
    """Answer a pending hook and keep waiting inside the *same* PTY turn.

    The upstream keeps the original ``send_turn`` deadline authoritative and does
    not increment ``turn_id`` — answering a question is not a new turn. That deadline
    is the one this waits on, so ``disable_invoke_wait_timeout`` applies here as it
    does in :func:`run_turn`.
    """
    return await runtime.invoke(
        InvokeRequest.build(
            respond_and_wait_claude_code_interactive_session,
            session_id=session_id,
            interaction_id=interaction_id,
            answer=answer,
            venv_id=runtime.default_venv_id,
            disable_invoke_wait_timeout=True,
        )
    )


async def close(runtime: Runtime, *, session_id: str) -> dict[str, Any]:
    """Close one runtime-side PTY session."""
    return await runtime.invoke(close_claude_code_interactive_session, session_id=session_id)


async def close_all(runtime: Runtime) -> dict[str, Any]:
    """Close every runtime-side PTY session."""
    return await runtime.invoke(close_all_claude_code_interactive_sessions)


def pending_interaction_id(raw: Mapping[str, Any]) -> str | None:
    """Return the interaction id when the turn is blocked on a hook, else None."""
    if str(raw.get('status') or '') != STATUS_NEEDS_INPUT:
        return None
    interaction = raw.get('interaction')
    if not isinstance(interaction, dict):
        # Upstream raises rather than emitting this shape, so a mismatch here
        # means the contract moved; surface it instead of silently continuing.
        logger.warning('PTY reported %s without an interaction payload', STATUS_NEEDS_INPUT)
        return None
    candidate = interaction.get('interaction_id')
    return candidate if isinstance(candidate, str) and candidate else None


def pending_wait_seconds(raw: Mapping[str, Any]) -> float | None:
    """How long the sandbox hook will wait for this answer, per upstream.

    Read from the interaction rather than from our own config so the budget
    follows what the hook was actually started with. ``None`` when the payload
    does not carry it, which leaves the answer unbudgeted rather than guessed.
    """
    interaction = raw.get('interaction')
    if not isinstance(interaction, dict):
        return None
    value = interaction.get('timeout_seconds')
    if not isinstance(value, (int, float)) or isinstance(value, bool):
        return None
    seconds = float(value)
    return seconds if math.isfinite(seconds) and seconds > 0 else None


def answer_landed(raw: Mapping[str, Any]) -> bool:
    """Whether the answer we sent was the one the worker acted on.

    Upstream reports ``written`` / ``expired`` / ``unknown`` and treats the last
    two as non-fatal, keeping the session alive. Read here because an answer that
    was discarded otherwise reports the same shape as one that arrived: the worker
    took the hook's unattended fallback and the run reads as normal.
    """
    status = raw.get('interaction_response_status')
    if status is None:
        return True
    return str(status) == 'written'


def pending_questions(raw: Mapping[str, Any]) -> list[str]:
    """The questions an ``AskUserQuestion`` hook is blocked on, in order.

    Empty for every other tool and for a turn that is not blocked.
    """
    interaction = raw.get('interaction')
    if not isinstance(interaction, dict) or interaction.get('tool_name') != ASK_USER_QUESTION:
        return []
    tool_input = interaction.get('tool_input')
    if not isinstance(tool_input, dict):
        return []
    questions = tool_input.get('questions')
    if not isinstance(questions, list):
        return []
    return [text for item in questions if isinstance(item, dict) and (text := str(item.get('question') or ''))]


def hook_answer(answer: str, questions: Sequence[str]) -> str:
    """Shape one answer into what the upstream hook can parse.

    UPSTREAM FACT 5: ``_normalise_answers`` (``interactive/hooks.py:124``) accepts a JSON
    object keyed by question text, or plain prose *only* when there is exactly one
    question. Anything else returns None and the hook denies the tool with "no user is
    available" — the worker keeps working and stops asking, so nothing raises.

    One question stays prose. With several, only the first carries the reply and the rest
    point back to it: the user answers the batch in one paragraph, and repeating it
    against every question shows the worker the same text once per question.
    """
    if len(questions) < 2:
        return answer
    answers = {text: answer if index == 0 else '(answered above)' for index, text in enumerate(questions)}
    return json.dumps(answers, ensure_ascii=False)


def session_died(raw: Mapping[str, Any]) -> bool:
    """Whether this result tore down the runtime-side session.

    UPSTREAM FACT 3: a turn that is neither ``needs_user_input`` nor completed
    leaves the PTY in an unknown state, so the upstream closes it and drops it
    from its registry. The next send on the same ``session_id`` therefore starts
    a *brand new* Claude with no history — and reports success while doing it.
    Callers must treat this as terminal rather than retrying.

    Read off ``success``, which is the field the upstream sets from its own
    ``completed`` (``interactive/runtime.py:477``). ``completed`` itself appears
    only on the event inside ``events``, not at the top level, so testing for it
    there reports every finished turn as a dead session.
    """
    status = str(raw.get('status') or '')
    if status == STATUS_NEEDS_INPUT:
        return False
    return not raw.get('success')


async def build_hook_request(
    raw: Mapping[str, Any],
    *,
    session_id: str,
    turn: int,
) -> UserInputRequest | None:
    """Build the ``source='hook'`` request for a pending-input result.

    Returns ``None`` for every other status. The upstream helper also knows how
    to describe a completed turn (``source='turn'``), but this backend never asks
    for that: a non-None request on an ordinary turn would make every turn look
    like it carried a pending question to any caller that checks presence rather
    than ``source``.

    UPSTREAM FACT 4: the pending-input result has ``success=True`` and an empty
    ``output``, so this request is the *only* signal distinguishing "blocked on a
    question" from "answered normally".
    """
    if pending_interaction_id(raw) is None:
        return None
    # The upstream helper already understands this shape: it maps
    # raw['interaction'] to a deferred-tool payload, sets source='hook', builds a
    # human-readable hint for AskUserQuestion and ExitPlanMode, and carries the
    # perception fields through metadata. It is async — it optionally writes a
    # debug dump when NEXUS_PTY_CONTEXT_DUMP_PATH is set.
    return await build_user_input_request(
        agent_label=AGENT_LABEL,
        output='',
        result=dict(raw),
        session_id=session_id,
        turn=turn,
    )


def perception(raw: Mapping[str, Any]) -> tuple[str, list[dict[str, Any]], dict[str, Any]]:
    """Extract ``(latest_screen, recent_events, event_cursor)`` defensively."""
    latest_screen = raw.get('latest_screen')
    raw_events = raw.get('recent_events')
    cursor = raw.get('event_cursor')
    return (
        latest_screen if isinstance(latest_screen, str) else '',
        [event for event in raw_events if isinstance(event, dict)] if isinstance(raw_events, list) else [],
        dict(cursor) if isinstance(cursor, dict) else {},
    )


__all__ = [
    'AGENT_LABEL',
    'DEFAULT_HOOK_COMMAND_SECONDS',
    'DEFAULT_HOOK_WAIT_SECONDS',
    'DEFAULT_TURN_TIMEOUT_SECONDS',
    'RUNTIME_CONFIG_KEYS',
    'STATUS_NEEDS_INPUT',
    'answer_landed',
    'build_hook_request',
    'close',
    'close_all',
    'open_session',
    'pending_interaction_id',
    'pending_wait_seconds',
    'perception',
    'respond',
    'run_turn',
    'runtime_kwargs',
    'session_died',
    'session_label',
]
