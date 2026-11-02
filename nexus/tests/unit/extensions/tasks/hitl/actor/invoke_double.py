"""Reading an ``InvokeRequest`` the way the real runtime client does.

A test double for ``Runtime.invoke`` records what it was called with. The real client
takes either a callable plus kwargs or a ready-made ``InvokeRequest``, and a worker turn
arrives as the latter so it can declare that the sandbox owns the deadline. A double
that only reads the callable form records nothing for those calls.

:func:`read_invoke` is for a double that returns a canned result.
:func:`resolve_invoke` is for one that runs the real runtime function.
"""

import importlib

from typing import Any, Callable


def read_invoke(target: Any, kwargs: dict[str, Any]) -> tuple[str, dict[str, Any]]:
    """The target's name and its kwargs, from either form ``invoke`` accepts.

    An ``InvokeRequest`` has its kwargs pickled inside. Its ``venv_id`` and
    ``disable_invoke_wait_timeout`` join them so a test can assert on either.
    """
    if not hasattr(target, 'unpack_payload'):
        return getattr(target, '__name__', str(target)), kwargs
    _, packed = target.unpack_payload()
    return (
        target.import_path.rsplit('::', 1)[-1],
        {
            **packed,
            'venv_id': target.venv_id,
            'disable_invoke_wait_timeout': target.disable_invoke_wait_timeout,
        },
    )


def resolve_invoke(target: Any, kwargs: dict[str, Any]) -> tuple[Callable[..., Any], dict[str, Any]]:
    """The callable to run and the kwargs to run it with.

    ``disable_invoke_wait_timeout`` is the client's own and is not passed on, which is
    what the client does with it.
    """
    if not hasattr(target, 'unpack_payload'):
        return target, kwargs
    module_path, _, name = target.import_path.partition('::')
    _, packed = target.unpack_payload()
    return getattr(importlib.import_module(module_path), name), packed


__all__ = ['read_invoke', 'resolve_invoke']
