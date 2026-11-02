"""Single-task adapter for the native batch generator; no environment overlay."""

from __future__ import annotations

from typing import Any

from nexus.extensions.tasks.os_world.generators.generator_native import single_config
from nexus.runner.compose import RunPart, run_part


@run_part('osworld_native')
def osworld_native(params: dict[str, Any]) -> RunPart:
    unexpected = sorted(set(params) - {'task_id', 'model_family', 'harness'})
    if unexpected:
        raise ValueError(f'osworld_native does not accept parameters: {unexpected}')
    if not params.get('task_id') or any(not isinstance(v, str) or not v.strip() for v in params.values()):
        raise ValueError('osworld_native requires task_id and non-empty string parameters')
    config = single_config(**params)
    return RunPart(body=config.model_dump(), source='osworld_native')
