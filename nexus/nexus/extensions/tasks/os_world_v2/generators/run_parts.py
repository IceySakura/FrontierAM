"""V2-only run-part; never dispatches through osworld_native/V1."""

from nexus.extensions.tasks.os_world_v2.generators.generator_native import single_config
from nexus.runner.compose import RunPart, run_part


@run_part('osworld_v2')
def osworld_v2(params):
    if set(params) - {'task_id', 'model_family', 'harness'}:
        raise ValueError('Unsupported osworld_v2 parameter')
    if not params.get('task_id') or any(not isinstance(value, str) or not value.strip() for value in params.values()):
        raise ValueError('osworld_v2 requires nonempty string parameters; use task_id=task_020')
    return RunPart(body=single_config(**params).model_dump(), source='osworld_v2')
