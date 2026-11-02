"""Run a filesystem task, starting a fresh session after an unsuccessful execution."""

import logging

from collections.abc import Awaitable, Callable
from typing import Any, TypeVar, cast

from nexus.extensions.tasks.hitl.actor.base import Session, SessionManager, SessionMessage, TurnResult
from nexus.runtime import Runtime
from nexus.runtime.client.base import RuntimeUnhealthyError


logger = logging.getLogger(__name__)
DEFAULT_SESSION_ATTEMPTS = 3
DEFAULT_OUTPUT_ATTEMPTS = 3
OUTPUT_ERRORS = (ValueError, KeyError, TypeError, FileNotFoundError)
CORRECTION_MESSAGE = 'Correct the output using the original evidence: {error}'
RESTART_NAME = '{name}_retry_{attempt}'
ResultT = TypeVar('ResultT')


class SessionFailedError(RuntimeError):
    """The task did not complete in its current session."""


class SessionAttemptsExhaustedError(RuntimeError):
    """The task failed in every allowed session."""


def turn_failure(result: TurnResult) -> str | None:
    """Read execution failure details from the manager's existing turn result."""
    if result.awaiting_answer or result.kind == 'refused':
        return None
    raw = result.raw
    if raw.get('failure'):
        return str(raw['failure'])
    if raw.get('timed_out'):
        return str(raw.get('reason') or 'Worker turn timed out.')
    event = cast(dict[str, Any], raw.get('result_event') or {})
    if event.get('is_error') or str(event.get('subtype') or '').startswith('error'):
        return str(event.get('result') or event.get('subtype') or 'Worker turn failed.')
    if result.session_lost or raw.get('success') is False or raw.get('exit_code') not in (None, 0):
        return str(raw.get('reason') or raw.get('stderr') or raw.get('status') or 'Worker turn failed.')
    return None


async def close_session(manager: SessionManager, session: Session | None = None) -> str | None:
    """Close a session or its manager, preserving the task result if cleanup fails."""
    try:
        if session is None:
            await manager.shutdown()
        else:
            await manager.close(session)
    except Exception as error:
        logger.warning('Session cleanup failed; runtime cleanup remains with the Nexus context', exc_info=True)
        return str(error)
    return None


async def run_session(
    manager: SessionManager,
    runtime: Runtime,
    name: str,
    prompt: str,
    read_result: Callable[[], Awaitable[ResultT]],
    *,
    workspace: str,
    session_attempts: int = DEFAULT_SESSION_ATTEMPTS,
    output_attempts: int = DEFAULT_OUTPUT_ATTEMPTS,
    correction_message: str = CORRECTION_MESSAGE,
) -> ResultT:
    """Execute a task and read its output after a successful turn.

    Each failed execution closes its session before another starts with the original
    prompt in the same workspace. Only successful turns with malformed output receive
    correction messages in their current session. ``output_attempts`` includes the
    first read; ``session_attempts`` includes the first session.

    The caller sets up the manager and owns the workspace and runtime. Execution errors
    consume the session allowance. Cancellation and an unhealthy runtime end the task.
    """
    if session_attempts < 1 or output_attempts < 1:
        raise ValueError('Session and output attempts must be positive')
    attempt = 0
    while True:
        session = None
        try:
            session_name = name if attempt == 0 else RESTART_NAME.format(name=name, attempt=attempt)
            session = await manager.start(name=session_name, workspace=workspace)
            message = prompt
            for correction in range(output_attempts):
                result = await manager.send(session, SessionMessage(content=message))
                failure = turn_failure(result)
                if failure or result.kind != 'completed':
                    raise SessionFailedError(f'{name} failed: {failure or result.kind}')
                try:
                    return await read_result()
                except OUTPUT_ERRORS as error:
                    if correction + 1 == output_attempts:
                        raise SessionFailedError(f'{name} output remained invalid: {error}') from error
                    message = correction_message.format(error=error)
        except Exception as error:
            if runtime.is_unhealthy:
                raise RuntimeUnhealthyError(runtime.unhealthy_reason or 'Runtime is unavailable') from error
            if attempt + 1 == session_attempts:
                raise SessionAttemptsExhaustedError(f'{name} failed in {session_attempts} sessions: {error}') from error
            logger.warning('%s failed; starting a new session after cleanup: %s', name, error)
            attempt += 1
        finally:
            if session is not None:
                await close_session(manager, session)
