"""Worker SessionManager factory.

Selects a worker backend by name and forwards the shared construction args. All
backend/model-specific configuration flows through ``default_config`` (an opaque dict
the generator assembles) — the factory never inspects it, so adding a new backend or a
new model knob touches only the generator and the target manager, never this factory
or the other backends.

``backend`` is an explicit argument; callers resolve it from their own config (e.g.
``worker_config.get('backend')``) before calling.
"""

from collections.abc import Mapping

from nexus.extensions.tasks.hitl.actor.base import SessionManager
from nexus.extensions.tasks.hitl.actor.registry import worker_manager_registry


# Imported for their registration side effect; the registry is the list of backends.
import nexus.extensions.tasks.hitl.actor.print_claude_code  # noqa: E402, F401  isort:skip
import nexus.extensions.tasks.hitl.actor.print_codex  # noqa: E402, F401  isort:skip
import nexus.extensions.tasks.hitl.actor.pty_claude_code  # noqa: E402, F401  isort:skip


def backends() -> list[str]:
    """Names a backend can be built by."""
    return sorted(worker_manager_registry.get_all())


def backend_class(backend: str) -> type[SessionManager]:
    """The manager class registered under ``backend``.

    For a generator reading a class attribute before any manager exists, e.g.
    ``skills_path`` to install a run's skills where that harness reads them.

    Raises ValueError on an unknown backend.
    """
    registered = worker_manager_registry.get_all()
    manager_cls = registered.get(backend)
    if manager_cls is None:
        raise ValueError(f'Unknown worker backend: {backend!r} (have: {sorted(registered)})')
    return manager_cls


def make_worker_manager(
    backend: str,
    *,
    default_working_dir: str = '/workspace',
    default_config: Mapping[str, object] | None = None,
    openai_client_name: str = 'main',
    runtime_name: str = 'main',
) -> SessionManager:
    """Construct the worker SessionManager for ``backend``.

    ``runtime_name`` selects the context runtime independently of the client.

    Every backend attributes traces by the label its own sandbox proxy stamps on each
    request, so none of them takes tracker or usage configuration: the run-level
    processors read both off the spans.

    Raises ValueError on an unknown backend.
    """
    manager_cls = backend_class(backend)

    return manager_cls(
        default_working_dir=default_working_dir,
        default_config=default_config,
        openai_client_name=openai_client_name,
        runtime_name=runtime_name,
    )


__all__ = ['backend_class', 'backends', 'make_worker_manager']
