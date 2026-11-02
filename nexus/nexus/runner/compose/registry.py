"""Registry for reusable run-compose parts."""

from __future__ import annotations

from collections.abc import Callable
from copy import deepcopy
from functools import wraps
from typing import Any

from nexus.core.registry import ProcessorResult, Registry
from nexus.runner.compose.model import RunPart


RunPartProvider = Callable[[dict[str, Any]], RunPart | dict[str, Any]]
ResolvedRunPartProvider = Callable[[dict[str, Any]], RunPart]


def validate_run_part(obj: Any, name: str, _meta: dict) -> ProcessorResult:
    """Validate and normalize a registered run part provider."""

    if not callable(obj):
        raise TypeError(f'Run part must be callable, got {type(obj)}')

    @wraps(obj)
    def wrapped(params: dict[str, Any]) -> RunPart:
        result = obj(params)
        if isinstance(result, RunPart):
            return RunPart(source=result.source, body=deepcopy(result.body), merge=dict(result.merge))
        if isinstance(result, dict):
            return RunPart(source=name, body=deepcopy(result))
        raise TypeError(f'Run part {name!r} returned {type(result).__name__}, expected RunPart or dict')

    return ProcessorResult(wrapped_obj=wrapped)


run_part_registry = Registry[ResolvedRunPartProvider]('run_part', validate_run_part)
run_part = run_part_registry.register
