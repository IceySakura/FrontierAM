"""Sandbox-side session opening for the labelled PTY backend.

Invoked through ``runtime.invoke()``, so this runs inside the sandbox process and
shares module state with the upstream PTY runtime it delegates to.

Opens a PTY session, labels its proxy, and registers it in the upstream registry
under the session id the host will use, by calling ``_prepare_runtime_session``
(which starts the proxy without spawning Claude) and writing to ``_PTY_SESSIONS``.
Upstream's per-turn entry point reuses a registered session and creates one only
when the id is absent, so the turn path, the hook bridge and teardown stay
upstream's.
"""

import inspect
import logging

from typing import Any


logger = logging.getLogger(__name__)


# Arguments of the public turn entry point that ``_prepare_runtime_session`` does
# not take: they describe a turn, not a session.
_TURN_ONLY = frozenset({'prompt', 'parent_tool_use_id'})


def _prepare_defaults() -> dict[str, Any]:
    """Defaults for the arguments ``_prepare_runtime_session`` requires.

    Its eighteen keyword-only arguments have no defaults of their own; upstream's
    public turn entry point declares one for each and passes them down. They are
    read off that signature rather than restated, so a new argument or a changed
    value arrives with upstream instead of raising a TypeError in the sandbox.

    Only the public function is inspected. ``_prepare_runtime_session`` is what a
    test replaces to keep a session out of a sandbox, and reading a stub's
    signature would describe the stub.
    """
    from nexus.extensions.agents.claude_code.interactive.runtime import (
        run_claude_code_interactive_session_turn,
    )

    return {
        name: parameter.default
        for name, parameter in inspect.signature(run_claude_code_interactive_session_turn).parameters.items()
        if parameter.default is not inspect.Parameter.empty and name not in _TURN_ONLY
    }


async def open_session(
    *,
    session_id: str,
    label: str,
    working_dir: str = '/workspace',
    **prepare_kwargs: Any,
) -> dict[str, Any]:
    """Open a PTY session whose LLM requests carry ``label``.

    Returns ``{'session_id', 'label', 'proxy_port', 'reused'}``. ``reused`` is
    True when the id was already registered, in which case the existing label is
    kept and the call is a no-op.

    Raises ``ValueError`` if ``label`` is empty, and ``RuntimeError`` if the
    session was registered without a label, since either would export
    trajectories that no session claims.
    """
    from nexus.extensions.agents.claude_code.interactive.runtime import (
        _CREATE_LOCK,
        _PTY_SESSIONS,
        _prepare_runtime_session,
    )
    from nexus.extensions.tasks.hitl.actor.labelling.proxy import _LabellingBridge, attach_label
    from nexus.extensions.tasks.hitl.actor.mcp import resolve_interpreters

    if not label:
        raise ValueError('a labelled PTY session requires a non-empty label')

    async with _CREATE_LOCK:
        existing = _PTY_SESSIONS.get(session_id)
        if existing is not None:
            bridge = existing.proxy.bridge
            if not isinstance(bridge, _LabellingBridge):
                raise RuntimeError(
                    f'PTY session {session_id} is already open with an unlabelled proxy; '
                    'its trajectories cannot be attributed. Close it before opening a '
                    'labelled session on the same id.'
                )
            return {
                'session_id': session_id,
                'label': bridge.label,
                'proxy_port': existing.proxy.port,
                'reused': True,
            }

        prepared: dict[str, Any] = {**_prepare_defaults(), **prepare_kwargs, 'working_dir': working_dir}
        # Upstream registers these with ``claude mcp add``, which records a command
        # and runs it later, so a bare interpreter is resolved here where this
        # process's own is known.
        prepared['mcp_servers'] = resolve_interpreters(prepared.get('mcp_servers'))
        session = await _prepare_runtime_session(session_id, **prepared)
        try:
            attach_label(session.proxy, label)
        except BaseException:
            # The session is not registered yet, so nothing else will close it.
            await session.close()
            raise

        _PTY_SESSIONS[session_id] = session

    logger.info('[pty %s] proxy labelling requests as %s', session_id, label)
    return {
        'session_id': session_id,
        'label': label,
        'proxy_port': session.proxy.port,
        'reused': False,
    }


async def session_label(*, session_id: str) -> dict[str, Any]:
    """The label on a registered session's proxy, or empty if it has none."""
    from nexus.extensions.agents.claude_code.interactive.runtime import _PTY_SESSIONS
    from nexus.extensions.tasks.hitl.actor.labelling.proxy import _LabellingBridge

    session = _PTY_SESSIONS.get(session_id)
    if session is None:
        return {'session_id': session_id, 'label': '', 'open': False}
    bridge = session.proxy.bridge
    return {
        'session_id': session_id,
        'label': bridge.label if isinstance(bridge, _LabellingBridge) else '',
        'open': True,
    }


__all__ = ['open_session', 'session_label']
