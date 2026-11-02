"""The hooks a topic supplies to add its own logic to an agent's run.

An agent fn defines its own event classes and fires them; this module registers hooks,
resolves names to them, and refuses a subscription the agent never fires. An event is a
subclass of :class:`HookEvent`, so a hook subscribes by class and reads that class's own
fields after an ``isinstance``.

A hook answers two things: what to do at an event, and what its own failure means.
``on_failure`` returning False records the failure and lets the run carry on; True ends
the run.

``agent_config['hooks']`` names them, each by a :data:`run_hook_registry` entry or by a
``pkg.mod::Class`` import path. Writing one: see ``docs/develop.md`` §5.
"""

import logging

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any, ClassVar, Protocol, runtime_checkable

from nexus.core.registry import Registry, import_from_path


logger = logging.getLogger(__name__)


class HookEvent:
    """One position in a run. The agent firing it subclasses this with the fields it
    hands over there."""

    name: ClassVar[str]
    """What the position is called, prefixed with the name of the agent that fires it."""


@dataclass(frozen=True, kw_only=True)
class HookOutcome:
    """What a hook asks the run to record."""

    results: Mapping[str, Any]
    """Merged into the run's ``hook_results``, which reaches the agent's output."""


@runtime_checkable
class RunHook(Protocol):
    """What a hook declares: the event classes it wants, and the two methods answering
    them -- one for the event, one for its own failure."""

    events: tuple[type[HookEvent], ...]

    async def on_event(self, event: HookEvent) -> HookOutcome | None: ...

    async def on_failure(self, event: HookEvent, error: Exception) -> bool:
        """Whether this failure should end the run. ``return False`` to carry on."""
        ...


class HookError(ValueError):
    """A hook is named or declared in a way that cannot be used."""


class RequiredHookError(RuntimeError):
    """A hook's ``on_failure`` asked for the run to end."""

    def __init__(self, event: str, cause: BaseException) -> None:
        super().__init__(f'{event}: {type(cause).__name__}: {cause}')
        self.event = event
        self.cause = cause


run_hook_registry = Registry[type]('hitl_run_hook')


def resolve_hooks(configs: Sequence[Mapping[str, Any]] | None, *, events: Sequence[type[HookEvent]]) -> list[RunHook]:
    """Build the hooks ``configs`` names, in the order it names them.

    ``events`` is what the calling agent fires. Raises :class:`HookError` when a name
    resolves to nothing, when what it resolves to has no ``on_event``, and when its
    ``events`` is empty or holds a class outside ``events``.
    """
    return [_resolve_one(config, events=events) for config in configs or ()]


def _resolve_one(config: Mapping[str, Any], *, events: Sequence[type[HookEvent]]) -> RunHook:
    name = str(config.get('name') or '')
    if not name:
        raise HookError(f'hook config carries no name: {dict(config)}')
    args = config.get('args') or {}
    if not isinstance(args, Mapping):
        raise HookError(f'hook {name!r} args must be a mapping, got {type(args).__name__}')

    hook = (_import_hook(name) if '::' in name else _registered_hook(name))(**dict(args))
    fires = ', '.join(sorted(event.__name__ for event in events))
    missing = [method for method in ('on_event', 'on_failure') if not callable(getattr(hook, method, None))]
    if missing:
        raise HookError(f'hook {name!r} defines no {" or ".join(missing)}')
    if not isinstance(hook, RunHook) or not hook.events:
        raise HookError(f'hook {name!r} subscribes to no event. This agent fires: {fires}')
    unknown = [event.__name__ for event in hook.events if event not in set(events)]
    if unknown:
        raise HookError(f'hook {name!r} subscribes to {unknown}, which this agent never fires. It fires: {fires}')
    logger.info('hook %s subscribes to %s', name, [event.__name__ for event in hook.events])
    return hook


def _import_hook(path: str) -> type:
    try:
        return import_from_path(path)
    except Exception as exc:
        raise HookError(f'hook {path!r} could not be imported: {type(exc).__name__}: {exc}') from exc


def _registered_hook(name: str) -> type:
    try:
        return run_hook_registry.get(name)
    except Exception as exc:
        available = ', '.join(sorted(run_hook_registry.get_all())) or '(none)'
        raise HookError(
            f'unknown hook {name!r}. Registered: {available}. A hook that another module '
            "registers needs that module in the run's imports; otherwise name it as "
            "'pkg.mod::Class'."
        ) from exc


__all__ = [
    'HookError',
    'HookEvent',
    'HookOutcome',
    'RequiredHookError',
    'RunHook',
    'resolve_hooks',
    'run_hook_registry',
]
