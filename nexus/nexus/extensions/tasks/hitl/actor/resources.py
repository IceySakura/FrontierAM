"""Per-runtime shared resources for HITL worker managers.

``llm_request`` is a broadcast event: ``Runtime.on()`` appends to a per-instance
listener list and dispatch walks all of them, so two managers that each register
their own handler on one runtime make every LLM request serve twice -- two model
calls and two ``llm_response`` emits that overwrite each other, with nothing
raised.

This module makes the handler a property of the runtime rather than of whichever
manager started first. A manager asks for the handler instead of constructing one,
so a second manager on the same runtime gets the same object. The map is weak-keyed
on the runtime, tying the handler's lifetime to the runtime's.
"""

import logging

from typing import Any, Protocol, TypeVar, cast, runtime_checkable
from weakref import WeakKeyDictionary

from nexus.runtime import Runtime


logger = logging.getLogger(__name__)

LLM_REQUEST_EVENT = 'llm_request'


@runtime_checkable
class RegisterableHandler(Protocol):
    """The part of a handler this module drives."""

    def register(self) -> None: ...

    def unregister(self) -> None: ...


# Keyed weakly so a finished run's handler is collectable with the runtime.
_HANDLERS: WeakKeyDictionary[Runtime, RegisterableHandler] = WeakKeyDictionary()

# How many callers hold each runtime's handler.
_HOLDERS: WeakKeyDictionary[Runtime, int] = WeakKeyDictionary()

_HandlerT = TypeVar('_HandlerT', bound=RegisterableHandler)


def foreign_listener_count(runtime: Runtime) -> int:
    """How many ``llm_request`` listeners are installed on ``runtime``.

    Reads ``_event_listeners``, which the concrete runtimes each build in
    ``__init__`` and the ``Runtime`` base class does not have; the base exposes only
    ``on`` / ``off`` / ``emit``, so there is no public way to count listeners. A
    runtime keeping none, or an upstream rename, reads as 0 -- which is what
    ``acquire_llm_handler`` checks before registering, so the double-registration it
    refuses would go through instead.
    """
    listeners = getattr(runtime, '_event_listeners', None)
    if isinstance(listeners, dict):
        return len(listeners.get(LLM_REQUEST_EVENT, ()))
    return 0


def acquire_llm_handler(
    runtime: Runtime,
    *,
    openai_client: Any,
    factory: type[_HandlerT] | None = None,
    no_thinking_request_args: dict[str, Any] | None = None,
    haiku_model: str | None = None,
    preserve_thinking: bool = True,
) -> Any:
    """Return the one handler for ``runtime``, creating and registering it once.

    Later callers get the same instance and their construction arguments are
    ignored, so the first caller's model routing wins.

    ``factory`` selects the handler class, defaulting to the labelled handler. A
    caller that names a different class and finds this runtime already holding
    another is refused rather than served a handler that attributes requests
    differently.

    Returns ``Any`` rather than :class:`RegisterableHandler`: callers read
    ``unattributed_requests`` off the concrete handler, which the protocol does not
    declare.

    Raises ``RuntimeError`` if a handler this module does not own is already
    listening, or if ``factory`` disagrees with the class already registered.
    """
    if factory is None:
        from nexus.extensions.tasks.hitl.actor.labelling.handler import LabelledLLMRequestHandler

        factory = cast(Any, LabelledLLMRequestHandler)

    existing = _HANDLERS.get(runtime)
    if existing is not None:
        if not isinstance(existing, cast(type, factory)):
            raise RuntimeError(
                f'This runtime already has a {type(existing).__name__}, but a '
                f'{cast(type, factory).__name__} was requested. The handler class decides how '
                'requests are attributed, so sharing one across both styles would silently '
                'mis-attribute: give this manager its own runtime (ctx.get_runtime("<name>")), '
                'or use managers that agree on a handler.'
            )
        _HOLDERS[runtime] = _HOLDERS.get(runtime, 0) + 1
        return existing

    already_listening = foreign_listener_count(runtime)
    if already_listening:
        raise RuntimeError(
            f'{already_listening} LLM handler(s) are already registered on this runtime. '
            f'{LLM_REQUEST_EVENT!r} is a broadcast event, so every request would be served '
            'by each of them: the model is called once per handler and their responses '
            'overwrite one another. Give this manager its own runtime '
            '(ctx.get_runtime("<name>")), or let both managers share one handler by '
            'constructing them through this module.'
        )

    handler = cast(Any, factory)(
        runtime=runtime,
        openai_client=openai_client,
        no_thinking_request_args=no_thinking_request_args,
        haiku_model=haiku_model,
        preserve_thinking=preserve_thinking,
    )
    handler.register()
    _HANDLERS[runtime] = handler
    _HOLDERS[runtime] = 1
    logger.debug('Registered shared HITL LLM handler on runtime %s', type(runtime).__name__)
    return handler


def release_llm_handler(runtime: Runtime) -> None:
    """Give up one hold on ``runtime``'s handler, unregistering it on the last.

    Reference-counted because acquire hands the same object to every caller: two
    managers on one runtime both hold it, and unregistering on the first release
    would leave the second serving requests through a listener that is gone. The
    symptom is an ``llm_request`` reaching nobody, which stalls the turn rather than
    raising.

    Idempotent: releasing more times than acquired, or without acquiring, is a no-op,
    because shutdown paths must not raise. A failing ``unregister`` is logged for the
    same reason; the handler is dropped either way, so a later run on this runtime
    builds its own rather than inheriting a half-released one.
    """
    handler = _HANDLERS.get(runtime)
    if handler is None:
        return

    holders = _HOLDERS.get(runtime, 1) - 1
    if holders > 0:
        _HOLDERS[runtime] = holders
        return

    _HANDLERS.pop(runtime, None)
    _HOLDERS.pop(runtime, None)
    try:
        handler.unregister()
    except Exception:
        logger.warning(
            'Failed to unregister the HITL LLM handler from runtime %s; it may still be '
            'listening, in which case the next acquire on this runtime is refused.',
            type(runtime).__name__,
            exc_info=True,
        )


def holder_count(runtime: Runtime) -> int:
    """How many callers currently hold ``runtime``'s handler."""
    return _HOLDERS.get(runtime, 0)


def current_llm_handler(runtime: Runtime) -> RegisterableHandler | None:
    """The handler owned for ``runtime``, if any."""
    return _HANDLERS.get(runtime)


__all__ = [
    'LLM_REQUEST_EVENT',
    'RegisterableHandler',
    'acquire_llm_handler',
    'current_llm_handler',
    'foreign_listener_count',
    'holder_count',
    'release_llm_handler',
]
