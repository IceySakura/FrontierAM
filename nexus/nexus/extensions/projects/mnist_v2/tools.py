"""Expose grid selection and integer arithmetic without modifying shared tools."""

from __future__ import annotations

import asyncio

from dataclasses import replace
from typing import Annotated

from pydantic import BaseModel, ConfigDict, Field, StrictInt

from nexus.core import ToolContext, ToolResponse, ToolType, tool_registry
from nexus.extensions.projects.mnist_v2.arithmetic import calculate_at_chain
from nexus.extensions.projects.mnist_v2.protocol import (
    STATE_KEY,
    CompositePhase,
    CompositeTaskState,
    build_digit_parts,
)


class DissectInput(BaseModel):
    """Request digit images from the current indexed composite in the initial message."""

    model_config = ConfigDict(extra='forbid')
    image_index: Annotated[StrictInt, Field(ge=0)]
    count: Annotated[StrictInt, Field(ge=1, le=100)]


class CalculatorInput(BaseModel):
    """Evaluate the private per-image or final operation through the calculator."""

    model_config = ConfigDict(extra='forbid')
    numbers: list[StrictInt] = Field(min_length=1)


@tool_registry.register(
    'mnist_v2/dissect',
    meta={
        'name': 'dissect',
        'tool_type': ToolType.LOCAL,
        'description': 'Return count individual digit images from the indexed composite in the initial message. '
        'Use the required image_index and count. Returns all selected digit images at once. '
        'Call this tool before calculator for each image. Make exactly one tool call per turn.',
    },
)
async def dissect_tool(params: DissectInput, context: ToolContext) -> ToolResponse:
    """Return exactly N digit images and wait for the next model observation before calculating."""
    state = context.tool_state.get(STATE_KEY)
    if not isinstance(state, CompositeTaskState):
        return _error('No composite task is active.')
    if state.phase == CompositePhase.SUM:
        return _error(
            f'Image {state.image_index} has already been dissected. Read the returned digit images and call '
            'calculator with numbers containing those digits in order. '
            'Do not call dissect again for this image.'
        )
    if state.phase != CompositePhase.DISSECT or params.image_index != state.image_index:
        return _error(f'Expected phase={state.phase.value}, current image_index={state.image_index}.')
    if params.count != state.composites[state.image_index].selection_count:
        return _error(
            f'Expected count={state.composites[state.image_index].selection_count}; do not change the workload.'
        )
    if state.image is None:
        raise RuntimeError('Current composite pixels are missing from task state')
    parts = await asyncio.to_thread(
        build_digit_parts,
        state.image,
        params.count,
        state.tool_image_long_side,
        image_index=params.image_index,
    )
    return ToolResponse(
        output=[
            {
                'type': 'text',
                'text': f'Image {params.image_index}: read these {params.count} digits, then call '
                'calculator with numbers containing those digits in order.',
            },
            *parts,
        ],
        meta={
            'success': True,
            'image_index': params.image_index,
            'image_count': params.count,
            'image_long_side': state.tool_image_long_side,
        },
        context_updates={STATE_KEY: replace(state, phase=CompositePhase.DIGITS_PENDING)},
    )


@tool_registry.register(
    'mnist_v2/calculator',
    meta={
        'name': 'calculator',
        'tool_type': ToolType.LOCAL,
        'description': 'Calculate a result for the complete numbers list in one call. '
        'After dissect, provide the digits in returned order. '
        'After all images, provide all returned image results in image-index order.',
    },
)
async def calculator_tool(params: CalculatorInput, context: ToolContext) -> ToolResponse:
    """Commit valid arithmetic steps atomically; reject skipped images and fabricated final operands."""
    state = context.tool_state.get(STATE_KEY)
    if not isinstance(state, CompositeTaskState):
        return _error('No composite task is active.')
    if state.phase == CompositePhase.COMPLETE:
        if state.final_result is None:
            raise RuntimeError('Completed task is missing its final calculation result')
        return _error(
            f'All calculations are complete. Final result: {state.final_result}. '
            f'Do not call any more tools. Return \\boxed{{{state.final_result}}} as your final answer.'
        )
    if state.phase == CompositePhase.FINAL:
        if tuple(params.numbers) != state.image_results:
            return _error('Final numbers must equal the previously returned image results in their original order.')
        result = calculate_at_chain(params.numbers)
        updated = replace(state, phase=CompositePhase.COMPLETE, final_result=result)
        text = f'Final result: {result}. Return this integer inside \\boxed{{}}.'
    else:
        if state.phase != CompositePhase.SUM:
            return _error(
                f'Call dissect for the current image first; phase={state.phase.value}, image_index={state.image_index}.'
            )
        count = state.composites[state.image_index].selection_count
        if len(params.numbers) != count or any(number < 0 or number > 9 for number in params.numbers):
            return _error(f'Expected exactly {count} digit integers in [0, 9].')
        result = calculate_at_chain(params.numbers)
        results = (*state.image_results, result)
        digit_lists = (*state.image_digits, tuple(params.numbers))
        next_index = state.image_index + 1
        if next_index < len(state.composites):
            updated = replace(
                state,
                phase=CompositePhase.IMAGE_PENDING,
                image_index=next_index,
                image=state.composite_images[next_index],
                image_results=results,
                image_digits=digit_lists,
            )
            text = f'Image {state.image_index} result: {result}. Next, call dissect with '
            text += f'image_index={next_index} and count={state.composites[next_index].selection_count}. '
            text += 'Use the corresponding composite already provided in the initial message.'
        else:
            updated = replace(
                state, phase=CompositePhase.FINAL_PENDING, image=None, image_results=results, image_digits=digit_lists
            )
            text = f'Image {state.image_index} result: {result}. All images are complete. '
            text += f'Call calculator with numbers={list(results)}.'
    return ToolResponse(
        output=[{'type': 'text', 'text': text}],
        meta={'success': True, 'result': result},
        context_updates={STATE_KEY: updated},
    )


def _error(message: str) -> ToolResponse:
    return ToolResponse(output=[{'type': 'text', 'text': f'Error: {message}'}], meta={'success': False})
