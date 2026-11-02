"""Registry for HITL worker managers.

Registration checks the conventions a class definition cannot. The first is that a
manager takes the runtime's shared ``llm_handler`` instead of registering its own:
``llm_request`` is a broadcast event, so a second listener on one runtime serves
every request twice -- two model calls, two ``llm_response`` emits overwriting each
other, and nothing raised. The others are the two names a manager must declare,
``backend`` and ``product``, neither of which has a usable default.

Everything else a manager must satisfy is enforced by :class:`SessionManager` being
abstract: a missing method fails at construction.
"""

import inspect

from typing import Any

from nexus.core.registry import Registry
from nexus.extensions.tasks.hitl.actor.base import SessionManager


_INJECTION_PARAM = 'llm_handler'


def validate_worker_manager(obj: Any, name: str, meta: dict) -> None:
    """Refuse a manager that would claim the runtime's llm_request channel."""
    del meta
    if not isinstance(obj, type):
        raise TypeError(f'Worker manager {name!r} must be a class, got {type(obj).__name__}')

    if not issubclass(obj, SessionManager):
        raise TypeError(
            f'Worker manager {name!r} must subclass SessionManager, which is abstract: '
            'subclassing is what makes a missing method fail at construction rather than '
            'return None at the call site.'
        )

    if not obj.backend:
        raise TypeError(f'Worker manager {name!r} must define a non-empty `backend` attribute')

    if not obj.product:
        raise TypeError(
            f'Worker manager {name!r} must define a non-empty `product` attribute: the name a '
            "person would call this tool. It reaches the user simulator's prompt, and `backend` "
            'cannot stand in for it -- that one names an implementation.'
        )

    parameters = inspect.signature(obj.__init__).parameters
    accepts_injection = _INJECTION_PARAM in parameters or any(
        parameter.kind is inspect.Parameter.VAR_KEYWORD for parameter in parameters.values()
    )
    if not accepts_injection:
        raise TypeError(
            f'Worker manager {name!r} must accept an injected {_INJECTION_PARAM!r} argument. '
            "A manager that builds its own handler claims the runtime's llm_request "
            'channel exclusively; a second such manager on the same runtime makes every '
            'request run twice. Take the handler from '
            'nexus.extensions.tasks.hitl.actor.resources.acquire_llm_handler instead.'
        )


worker_manager_registry = Registry[type[SessionManager]]('hitl_worker_manager', validate_worker_manager)


__all__ = ['validate_worker_manager', 'worker_manager_registry']
