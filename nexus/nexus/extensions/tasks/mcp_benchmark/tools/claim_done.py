"""
local-claim_done tool: signals task completion.

Matches the official Toolathlon behavior where the model calls this tool
to indicate it has finished the task. Returns a success message.
"""

from nexus.core import ToolContext, ToolResponse, ToolType, tool_registry
from nexus.core.types import RobustBaseModel


class ClaimDoneInput(RobustBaseModel):
    pass


@tool_registry.register(
    'local-claim_done',
    meta={
        'name': 'local-claim_done',
        'description': 'claim the task is done',
        'tool_type': ToolType.LOCAL,
    },
)
async def claim_done_tool(params: ClaimDoneInput, context: ToolContext) -> ToolResponse:
    return ToolResponse(
        output=[{'type': 'text', 'text': 'you have claimed the task is done!'}],
        meta={'success': True, 'termination_requested': True},
    )
