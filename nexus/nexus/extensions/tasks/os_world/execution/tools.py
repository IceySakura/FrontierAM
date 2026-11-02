"""OSWorld task helper tools."""

from typing import Literal

from pydantic import BaseModel, Field

from nexus.core import ToolContext, ToolResponse, ToolType, tool_registry


FIXED_MARKER_TOOL_DESCRIPTION = """\
Return one of the official OSWorld fixed markers through the Nexus tool interface: WAIT, DONE, or FAIL.
Use WAIT when the desktop needs more time, DONE when the task is complete, and FAIL when the task cannot be completed.
"""


class FixedMarkerInput(BaseModel):
    """Input payload for OSWorld fixed marker actions."""

    marker: Literal['WAIT', 'DONE', 'FAIL'] = Field(
        ...,
        description=(
            'Official OSWorld fixed marker. WAIT means wait for more desktop progress, '
            'DONE means the task is complete, and FAIL means the task cannot be completed.'
        ),
    )


@tool_registry.register(
    'desktop/fixed_marker',
    meta={
        'name': 'fixed_marker',
        'description': FIXED_MARKER_TOOL_DESCRIPTION,
        'tool_type': ToolType.LOCAL,
    },
)
async def fixed_marker_tool(params: FixedMarkerInput, context: ToolContext) -> ToolResponse:
    """Return a fixed OSWorld marker for task control."""
    return ToolResponse(
        output=[{'type': 'text', 'text': params.marker}],
        meta={
            'success': True,
            'marker': params.marker,
            'termination_requested': params.marker in {'DONE', 'FAIL'},
        },
        context_updates={
            'os_world_last_action': params.marker,
            'desktop_last_tool': 'desktop/fixed_marker',
        },
    )
