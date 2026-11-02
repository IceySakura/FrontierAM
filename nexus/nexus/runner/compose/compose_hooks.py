"""Compose hooks: per-item steps applied to batch items after static run parts are merged."""

from __future__ import annotations

import hashlib
import importlib.util
import sys

from collections.abc import Callable, Iterable, Iterator, Mapping, Sequence
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any

from nexus.core.registry import Registry
from nexus.runner.compose.parser import parse_use_spec


if TYPE_CHECKING:
    from nexus.runner.batch.batch_config import BatchItem
    from nexus.runner.compose.model import RunPart


ComposeHookFactory = Callable[[dict[str, Any]], Any]

_FILE_FACTORY = 'create_compose_hook'


def _validate_compose_hook(obj: Any, name: str, _meta: dict) -> None:
    if not callable(obj):
        raise TypeError(f'Compose hook {name!r} must be a callable factory, got {type(obj).__name__}')


compose_hook_registry = Registry[ComposeHookFactory]('compose_hook', _validate_compose_hook)
compose_hook = compose_hook_registry.register


class ComposeHookError(ValueError):
    """An compose hook rejected an item or could not run."""


_HOST_ENV: ContextVar[Mapping[str, str] | None] = ContextVar('compose_hook_host_env', default=None)


def compose_hook_host_env() -> Mapping[str, str]:
    """The ``--host-env`` values of the submission whose hooks are running; empty outside ``nexus submit``."""
    return _HOST_ENV.get() or {}


@contextmanager
def submission_host_env(values: Mapping[str, str]) -> Iterator[None]:
    """Expose a submission's ``--host-env`` to the compose hooks run inside this context."""
    token = _HOST_ENV.set(dict(values))
    try:
        yield
    finally:
        _HOST_ENV.reset(token)


@dataclass(frozen=True)
class ComposeHookSpec:
    """An unresolved ``--hook`` value: a ``.py`` file or a registered compose hook, with its parameters."""

    name: str
    params: dict[str, Any]
    source: str


def parse_compose_hooks(values: Sequence[str] | None) -> list[ComposeHookSpec]:
    """Parse repeated ``--hook NAME[:key=value,...]`` values, kept in command-line order."""
    specs = []
    for value in values or ():
        spec = parse_use_spec(value)
        if spec.name == 'data':
            raise ValueError(f'--hook {value}: data:<uri> is a dataset; pass it with --data')
        specs.append(ComposeHookSpec(name=spec.name, params=dict(spec.params), source=f'--hook {value}'))
    return specs


@dataclass(frozen=True)
class _LoadedHook:
    source: str
    process: Callable[[BatchItem], BatchItem]
    finish: Callable[[], None] | None


def _load_factory(spec: ComposeHookSpec) -> ComposeHookFactory:
    if not spec.name.endswith('.py'):
        return compose_hook_registry.get(spec.name)
    path = Path(spec.name).expanduser().resolve()
    if not path.is_file():
        raise FileNotFoundError(f'compose hook file not found: {path}')
    module_name = f'nexus_compose_hook_{hashlib.sha256(str(path).encode()).hexdigest()[:16]}'
    module_spec = importlib.util.spec_from_file_location(module_name, path)
    if module_spec is None or module_spec.loader is None:
        raise ImportError(f'cannot load compose hook file: {path}')
    module = importlib.util.module_from_spec(module_spec)
    sys.modules[module_name] = module
    module_spec.loader.exec_module(module)
    factory = getattr(module, _FILE_FACTORY, None)
    if not callable(factory):
        raise TypeError(f'{path} must define {_FILE_FACTORY}(params)')
    return factory


def _instantiate(spec: ComposeHookSpec) -> _LoadedHook:
    try:
        hook = _load_factory(spec)(dict(spec.params))
    except Exception as exc:
        raise ComposeHookError(f'Compose hook {spec.source} could not be created: {exc}') from exc
    process = getattr(hook, 'process', None)
    if callable(process):
        finish = getattr(hook, 'finish', None)
        return _LoadedHook(source=spec.source, process=process, finish=finish if callable(finish) else None)
    if callable(hook):
        return _LoadedHook(source=spec.source, process=hook, finish=None)
    raise ComposeHookError(f'Compose hook {spec.source} must return a callable or an object with process()')


def _validated(result: Any, *, source: str, index: int) -> BatchItem:
    from nexus.runner.batch.batch_config import BatchItem
    from nexus.runner.single import RunInput

    if not isinstance(result, BatchItem):
        raise ComposeHookError(
            f'Compose hook {source} must return a BatchItem for item #{index}, got {type(result).__name__}'
        )
    try:
        config = RunInput.model_validate(result.config.model_dump(mode='python'))
        return BatchItem(config=config, run_id=result.run_id, metadata=result.metadata)
    except Exception as exc:
        raise ComposeHookError(f'Compose hook {source} returned an invalid item #{index}: {exc}') from exc


class ComposeHookChain:
    """Hooks for one generation; their state is shared across every item of that generation."""

    def __init__(self, specs: Sequence[ComposeHookSpec]) -> None:
        self._hooks = [_instantiate(spec) for spec in specs]
        self._count = 0
        self._finished = False

    def apply(self, item: BatchItem) -> BatchItem:
        index = self._count
        self._count += 1
        for hook in self._hooks:
            try:
                result = hook.process(item)
            except Exception as exc:
                raise ComposeHookError(
                    f'Compose hook {hook.source} rejected item #{index} (run_id={item.run_id!r}): {exc}'
                ) from exc
            item = _validated(result, source=hook.source, index=index)
        return item

    def finish(self) -> None:
        if self._finished:
            return
        self._finished = True
        for hook in self._hooks:
            if hook.finish is None:
                continue
            try:
                hook.finish()
            except Exception as exc:
                raise ComposeHookError(f'Compose hook {hook.source} failed after {self._count} items: {exc}') from exc


def apply_generator_overrides(item: BatchItem, overrides: Sequence[RunPart]) -> BatchItem:
    """Merge static run parts into a generated item's config, keeping its run ID and metadata."""
    from nexus.runner.compose.merge import deep_merge
    from nexus.runner.single import RunInput

    original = item.config.model_dump(mode='python')
    config = original
    for part in overrides:
        config = deep_merge(config, part.body, part.merge)
    # Resolve changed sections using Compose semantics without expanding
    # unrelated environment references into the persisted generator config.
    changed = {key: value for key, value in config.items() if key not in original or value != original[key]}
    if changed:
        resolved = RunInput.from_dict({'task_name': item.config.task_name, **changed})
        for key in changed:
            if key in RunInput.model_fields:
                config[key] = getattr(resolved, key)
    return item.model_copy(update={'config': RunInput.model_validate(config)})


def postprocess_generated_items(
    items: Iterable[BatchItem],
    *,
    overrides: Sequence[RunPart] | None = None,
    hooks: Sequence[ComposeHookSpec] | None = None,
) -> Iterator[BatchItem]:
    """Apply static overrides then compose hooks to generated items; ``finish()`` runs after the last one."""
    chain = ComposeHookChain(hooks) if hooks else None
    for item in items:
        if overrides:
            item = apply_generator_overrides(item, overrides)
        if chain is not None:
            item = chain.apply(item)
        yield item
    if chain is not None:
        chain.finish()


__all__ = [
    'ComposeHookChain',
    'ComposeHookError',
    'ComposeHookSpec',
    'apply_generator_overrides',
    'compose_hook',
    'compose_hook_host_env',
    'compose_hook_registry',
    'parse_compose_hooks',
    'postprocess_generated_items',
    'submission_host_env',
]
