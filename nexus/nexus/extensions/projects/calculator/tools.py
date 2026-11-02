from typing import Literal

from pydantic import BaseModel, Field

from nexus.core import ToolContext, ToolResponse, ToolType, tool_registry


class CalculateInput(BaseModel):
    a: int = Field(description='The first operand')
    b: int = Field(description='The second operand')
    operand: Literal['+', '-', '*', '/', '@'] = Field(description="The operator: '+' or '-' or '*' or '/' or '@'")


@tool_registry.register(
    'calculator/calculate',
    meta={
        'name': 'calculate',
        'description': 'Compute the result using operand with two integers. '
        "Supported operands: '+', '-', '*', '/', '@'.",
        'tool_type': ToolType.LOCAL,
    },
)
async def calculate_tool(params: CalculateInput, context: ToolContext) -> ToolResponse:
    """Compute the result using operand with two integers.

    The @ operator is defined as: a @ b = 3 * a - 2 * b
    """
    a, b, operand = params.a, params.b, params.operand

    if operand == '@':
        result = 3 * a - 2 * b
    elif operand == '+':
        result = a + b
    elif operand == '-':
        result = a - b
    elif operand == '*':
        result = a * b
    elif operand == '/':
        if b == 0:
            return ToolResponse(
                output=[{'type': 'text', 'text': 'Error: Division by zero'}],
                meta={'error': True},
            )
        result = a / b
    else:
        return ToolResponse(
            output=[{'type': 'text', 'text': f'Unknown operand: {operand}'}],
            meta={'error': True},
        )

    return ToolResponse(
        output=[{'type': 'text', 'text': str(result)}],
        meta={'result': result, 'reward': 1.0},
    )
