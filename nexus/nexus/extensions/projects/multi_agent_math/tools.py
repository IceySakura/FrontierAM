"""Arithmetic tools reserved for multi-agent math subagents."""

from __future__ import annotations

from pydantic import BaseModel, Field

from nexus.core import ToolContext, ToolResponse, ToolType, tool_registry


class BinaryOperatorInput(BaseModel):
    """Define two integer operands for a hidden binary operator."""

    a: int = Field(description='The first integer operand.')
    b: int = Field(description='The second integer operand.')


def evaluate_dollar(a: int, b: int) -> int:
    """Evaluate a $ b."""
    return (a + 1) * (b - 1)


def evaluate_at(a: int, b: int) -> int:
    """Evaluate a @ b."""
    return 3 * a - 2 * b


@tool_registry.register(
    'multi_agent_math/compute_dollar',
    meta={
        'name': 'compute_dollar',
        'description': 'Compute a $ b for two integers, where a $ b = (a + 1) * (b - 1).',
        'tool_type': ToolType.LOCAL,
    },
)
async def compute_dollar(params: BinaryOperatorInput, context: ToolContext) -> ToolResponse:
    """Compute one dollar operation."""
    del context
    result = evaluate_dollar(params.a, params.b)
    return ToolResponse(
        output=[{'type': 'text', 'text': f'{params.a} $ {params.b} = {result}'}],
        meta={'result': result},
    )


@tool_registry.register(
    'multi_agent_math/compute_at',
    meta={
        'name': 'compute_at',
        'description': 'Compute a @ b for two integers, where a @ b = 3 * a - 2 * b.',
        'tool_type': ToolType.LOCAL,
    },
)
async def compute_at(params: BinaryOperatorInput, context: ToolContext) -> ToolResponse:
    """Compute one at operation."""
    del context
    result = evaluate_at(params.a, params.b)
    return ToolResponse(
        output=[{'type': 'text', 'text': f'{params.a} @ {params.b} = {result}'}],
        meta={'result': result},
    )
