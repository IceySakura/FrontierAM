"""Run a visual task with the spatial tools added to the agent's tool list."""

from pydantic import Field

from nexus.core import task_registry
from nexus.extensions.tasks.visual.task import VisualTaskInput, VisualTaskOutput, visual_task
from nexus.extensions.tasks.visual_spatial.tools import SPATIAL_TOOLS


class VisualSpatialTaskInput(VisualTaskInput):
    """Visual task data plus the spatial tool selection."""

    spatial_tools: list[str] = Field(
        default_factory=lambda: list(SPATIAL_TOOLS),
        description='Spatial tools to enable; defaults to all four.',
    )
    base_tools: list[str] = Field(
        default_factory=list,
        description='Extra non-spatial tools, such as jupyter_with_image; none by default.',
    )


@task_registry.register('visual_spatial')
async def visual_spatial_task(task_input: VisualSpatialTaskInput) -> VisualTaskOutput:
    """Delegate to the visual task with the spatial tools appended to the agent's tools."""
    if unknown := sorted(set(task_input.spatial_tools) - set(SPATIAL_TOOLS)):
        raise ValueError(f'Unknown spatial tools {unknown}; known: {list(SPATIAL_TOOLS)}')
    tools = list(dict.fromkeys([*task_input.base_tools, *task_input.spatial_tools]))
    base_run_agent = task_input.run_agent

    async def run_agent(**kwargs):
        return await base_run_agent(**{**kwargs, 'tools': tools})

    return await visual_task(task_input.model_copy(update={'run_agent': run_agent}))


__all__ = ['VisualSpatialTaskInput', 'visual_spatial_task']
