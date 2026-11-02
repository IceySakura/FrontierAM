"""What registration enforces, and what the class definition enforces instead.

Registration keeps the rules a class definition cannot. The substantive one is that a
manager must accept an injected ``llm_handler``: ``llm_request`` is a broadcast event,
so a manager that builds its own claims the channel exclusively and a second one on
the same runtime makes every request run twice, with nothing raised. The other two are
the names with no usable default, ``backend`` and ``product``.

The rules that used to live here -- "implements the SessionManager methods", "has a
backend name" -- were vacuous: ``SessionManager`` was a Protocol used as a base
class, so ``callable(getattr(obj, method))`` was true for a class that implemented
nothing. They are now enforced by the class being abstract, which fails earlier and
cannot be talked around.
"""

from typing import Any

import pytest

from nexus.extensions.tasks.hitl.actor.base import (
    Session,
    SessionManager,
    SessionMessage,
    TurnResult,
)
from nexus.extensions.tasks.hitl.actor.registry import validate_worker_manager, worker_manager_registry
from nexus.extensions.tasks.hitl.actor.rendering import DEFAULT_DETAIL, Detail


class WellFormedManager(SessionManager):
    """A manager that satisfies every rule."""

    backend = 'well_formed'
    product = 'Well Formed'

    def __init__(self, *, llm_handler: Any = None, **_kwargs: Any) -> None:
        self._llm_handler = llm_handler

    async def setup(self) -> None: ...
    async def start(self, *, name: str, **_kwargs: Any) -> Session: ...  # type: ignore[empty-body]
    async def send(self, session: Session, message: SessionMessage) -> TurnResult: ...  # type: ignore[empty-body]
    def session_of(self, session_id: str) -> Session | None: ...  # type: ignore[empty-body]
    def report(self, result: TurnResult, *, detail: Detail = DEFAULT_DETAIL) -> str: ...  # type: ignore[empty-body]
    def traces(self, session: Session) -> dict[str, object]: ...  # type: ignore[empty-body]
    async def close(self, session: Session) -> None: ...
    async def shutdown(self) -> None: ...


def test_a_well_formed_manager_is_accepted() -> None:
    validate_worker_manager(WellFormedManager, 'well_formed', {})


def test_a_manager_that_cannot_take_an_injected_handler_is_rejected() -> None:
    """The one substantive rule: no injection means it must build its own handler."""

    class SelfSufficient(WellFormedManager):
        def __init__(self, *, default_working_dir: str = '/workspace') -> None:
            self._dir = default_working_dir

    with pytest.raises(TypeError, match='llm_handler'):
        validate_worker_manager(SelfSufficient, 'self_sufficient', {})


def test_var_keyword_counts_as_accepting_injection() -> None:
    class Flexible(WellFormedManager):
        def __init__(self, **kwargs: Any) -> None:
            self._kwargs = kwargs

    validate_worker_manager(Flexible, 'flexible', {})


def test_a_manager_outside_the_hierarchy_is_rejected() -> None:
    """Structural conformance is not enough: the abstract base is what enforces it.

    This class satisfies every method by name, which is exactly what the old
    ``callable(getattr(...))`` check accepted while the bodies returned None.
    """

    class LooksRight:
        backend = 'looks_right'
        product = 'Looks Right'

        def __init__(self, *, llm_handler: Any = None) -> None: ...
        async def setup(self) -> None: ...
        async def start(self, *, name: str, **_kwargs: Any) -> Any: ...
        async def send(self, session: Any, message: Any) -> Any: ...
        def report(self, result: Any, *, detail: Any = None) -> str: ...  # type: ignore[empty-body]
        def traces(self, session: Any) -> dict[str, object]: ...  # type: ignore[empty-body]
        async def close(self, session: Any) -> None: ...
        async def shutdown(self) -> None: ...

    with pytest.raises(TypeError, match='must subclass SessionManager'):
        validate_worker_manager(LooksRight, 'looks_right', {})


def test_a_missing_backend_attribute_is_rejected() -> None:
    class Anonymous(WellFormedManager):
        backend = ''

    with pytest.raises(TypeError, match='backend'):
        validate_worker_manager(Anonymous, 'anonymous', {})


def test_a_missing_product_attribute_is_rejected() -> None:
    """The name shown to a requester has no default worth falling back to.

    ``backend`` cannot stand in for it: ``pty_claude_code`` names an implementation
    and how it is driven, and the user simulator is told what it is working with.
    """

    class Unnamed(WellFormedManager):
        product = ''

    with pytest.raises(TypeError, match='product'):
        validate_worker_manager(Unnamed, 'unnamed', {})


def test_every_registered_backend_names_its_product() -> None:
    for backend, manager in worker_manager_registry.get_all().items():
        assert manager.product, backend


def test_an_instance_instead_of_a_class_is_rejected() -> None:
    with pytest.raises(TypeError, match='must be a class'):
        validate_worker_manager(WellFormedManager(), 'instance', {})
