"""BrowserGym agent specific tools.

These tools are used by the BrowserGym agent but are not browser-specific.
"""

from pydantic import BaseModel, Field

from nexus.core import ToolContext, ToolResponse, ToolType, tool_registry


class SubmitResultInput(BaseModel):
    """Submit the final answer for the task."""

    message: str = Field(..., description='The direct answer to the task')


@tool_registry.register(
    'browser_gym/submit_result',
    meta={
        'name': 'submit_result',
        'description': 'Submit the final answer when task is completed',
        'tool_type': ToolType.LOCAL,
    },
)
async def submit_result_tool(params: SubmitResultInput, context: ToolContext) -> ToolResponse:
    """Submit the final answer for the task."""
    return ToolResponse(
        output=[{'type': 'text', 'text': params.message}],
        meta={'success': True, 'message_sent': True},
    )
