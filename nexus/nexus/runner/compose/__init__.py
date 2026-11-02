"""Compose ordered run parts into existing Nexus runner inputs."""

from typing import TYPE_CHECKING, Any

from nexus.runner.compose.merge import deep_merge, set_path
from nexus.runner.compose.model import RunComposition, RunPart
from nexus.runner.compose.registry import run_part, run_part_registry


if TYPE_CHECKING:
    from nexus.runner.compose.compiler import RunComposer
    from nexus.runner.compose.compose_hooks import compose_hook, compose_hook_registry


__all__ = [
    'RunPart',
    'RunComposer',
    'RunComposition',
    'compile_batch',
    'compile_batch_file',
    'compile_single',
    'compile_single_file',
    'deep_merge',
    'compose_hook',
    'compose_hook_registry',
    'parse_ordered_parts',
    'run_part',
    'run_part_registry',
    'set_path',
]


def compile_batch(*args: Any, **kwargs: Any) -> Any:
    from nexus.runner.compose.compiler import compile_batch as _compile_batch

    return _compile_batch(*args, **kwargs)


def compile_batch_file(*args: Any, **kwargs: Any) -> Any:
    from nexus.runner.compose.compiler import compile_batch_file as _compile_batch_file

    return _compile_batch_file(*args, **kwargs)


def compile_single(*args: Any, **kwargs: Any) -> Any:
    from nexus.runner.compose.compiler import compile_single as _compile_single

    return _compile_single(*args, **kwargs)


def compile_single_file(*args: Any, **kwargs: Any) -> Any:
    from nexus.runner.compose.compiler import compile_single_file as _compile_single_file

    return _compile_single_file(*args, **kwargs)


def parse_ordered_parts(*args: Any, **kwargs: Any) -> Any:
    from nexus.runner.compose.parser import parse_ordered_parts as _parse_ordered_parts

    return _parse_ordered_parts(*args, **kwargs)


def __getattr__(name: str) -> Any:
    if name == 'RunComposer':
        from nexus.runner.compose.compiler import RunComposer

        return RunComposer
    if name in {'compose_hook', 'compose_hook_registry'}:
        from nexus.runner.compose import compose_hooks

        return getattr(compose_hooks, name)
    raise AttributeError(f'module {__name__!r} has no attribute {name!r}')
