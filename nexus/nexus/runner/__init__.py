"""Runner module for nexus framework.

Provides task orchestration and execution:
- run(): Run a single task
- run_batch(): Run multiple tasks (TODO)
- hytuner: HyTuner RL training integration
"""

from __future__ import annotations

import importlib

from typing import TYPE_CHECKING, Any

from nexus.runner.single import RunHooks, RunInput, RunOutput, run


if TYPE_CHECKING:
    from types import ModuleType

    from nexus.runner.hytuner import HyTunerRunInput, HyTunerRunOutput
    from nexus.runner.hytuner import run as run_hytuner

    hytuner: ModuleType


def __getattr__(name: str) -> Any:
    if name == 'hytuner':
        module = importlib.import_module('nexus.runner.hytuner')
        globals()[name] = module
        return module

    if name in {'HyTunerRunInput', 'HyTunerRunOutput', 'run_hytuner'}:
        module = importlib.import_module('nexus.runner.hytuner')
        value = module.run if name == 'run_hytuner' else getattr(module, name)
        globals()[name] = value
        return value

    raise AttributeError(f'module {__name__!r} has no attribute {name!r}')


__all__ = [
    'RunInput',
    'RunHooks',
    'RunOutput',
    'run',
    'hytuner',
    'HyTunerRunInput',
    'HyTunerRunOutput',
    'run_hytuner',
]
