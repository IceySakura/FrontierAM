"""Run the fixed-grid task through main_agent with task-local image preparation."""

from __future__ import annotations

import asyncio
import re

from dataclasses import replace
from typing import Annotated, Any, Literal

from pydantic import ConfigDict, Field, StrictInt, model_validator

from nexus.core import TaskRunInput, TaskRunOutput, WithAgentRunner, WithScore, require_context, task_registry
from nexus.core.context import ExecutionContext
from nexus.extensions.agents.main.agent import MainAgentOutput, PostIterationHookContext, PostIterationHookResult
from nexus.extensions.projects.mnist.task import extract_boxed_answer
from nexus.extensions.projects.mnist_v2.arithmetic import calculate_at_chain
from nexus.extensions.projects.mnist_v2.protocol import (
    SELECTION_ORDER,
    STATE_KEY,
    TOOL_NAMES,
    CompositeImageConfig,
    CompositePhase,
    CompositeTaskState,
    ImageLongSide,
    build_initial_observation,
)
from nexus.utils.multimodal import IMAGE_PLACEHOLDER


HOOK_PATH: str = 'nexus.extensions.projects.mnist_v2.task::advance_composite'


class MnistV2TaskInput(TaskRunInput, WithAgentRunner):
    """Configure the generated grid task; sizes are pixel long sides, not processor token caps.

    Args:
        composites: Consecutively indexed external 10x10 grids.
        user_prompt: Dataset instructions with one <image> placeholder per composite, excluding private labels.
        ground_truth: Expected calculator integer result, used only for scoring.
        reward_mode: Binary final reward, or equal-weight digit accuracy and final success.
        expected_digits: Private ordered digit labels per image; never forwarded to the agent.
        input_image_long_side: Big-image display resolution; defaults to 1024.
        tool_image_long_side: Returned digit resolution; defaults to 1024.
        max_iterations: Optional main_agent budget; defaults to 2 * images + 6.
        schema_version: Supported generator schema version.
        selection_order: Versioned down-first spiral ordering.
    """

    model_config = ConfigDict(coerce_numbers_to_str=True)
    composites: list[CompositeImageConfig] = Field(min_length=1)
    user_prompt: str
    ground_truth: str
    reward_mode: Literal['final', 'digit_and_final'] = 'final'
    expected_digits: list[list[Annotated[StrictInt, Field(ge=0, le=9)]]] | None = Field(default=None, repr=False)
    input_image_long_side: ImageLongSide = 1024
    tool_image_long_side: ImageLongSide = 1024
    max_iterations: int | None = Field(default=None, strict=True, gt=0)
    schema_version: Literal[1] = 1
    calculator_operation: Literal['at_parity_unit_coeff_v1'] = 'at_parity_unit_coeff_v1'
    selection_order: Literal['spiral_down_right_up_left_v1'] = SELECTION_ORDER

    @model_validator(mode='after')
    def validate_task(self) -> MnistV2TaskInput:
        """Reject ambiguous image ordering and noninteger scoring targets."""
        if [image.image_index for image in self.composites] != list(range(len(self.composites))):
            raise ValueError('composites must have consecutive image_index values starting at zero')
        if re.fullmatch(r'[+-]?[0-9]+', self.ground_truth) is None:
            raise ValueError('ground_truth must be an integer or integer string')
        if self.user_prompt.count(IMAGE_PLACEHOLDER) != len(self.composites):
            raise ValueError('Expected exactly one <image> placeholder per composite')
        if self.reward_mode == 'digit_and_final' and self.expected_digits is None:
            raise ValueError('digit_and_final reward requires private expected_digits')
        if self.expected_digits is not None:
            if len(self.expected_digits) != len(self.composites) or any(
                len(digits) != image.selection_count
                for digits, image in zip(self.expected_digits, self.composites, strict=True)
            ):
                raise ValueError('expected_digits must match every image and selection_count')
            expected_answer = calculate_at_chain([calculate_at_chain(digits) for digits in self.expected_digits])
            if expected_answer != int(self.ground_truth):
                raise ValueError('expected_digits and ground_truth disagree under the configured calculator')
        return self


class MnistV2TaskOutput(TaskRunOutput, WithScore):
    """Separate numeric correctness from completion of the required tool protocol."""

    answer: str | None = None
    answer_correct: bool
    process_complete: bool
    completed_images: int
    digit_acc: float | None = None
    final_success: bool
    extra_fields: dict[str, Any] = Field(default_factory=dict)


async def advance_composite(
    context: ExecutionContext, observation: PostIterationHookContext
) -> PostIterationHookResult:
    """Publish prepared observations without I/O; dependent tools require a new model turn."""
    state = context.tool_state[STATE_KEY]
    assistant = observation.history[-1]
    if len(assistant.get('tool_calls', [])) != 1:
        context.tool_state[STATE_KEY] = replace(state, phase=CompositePhase.INVALID)
        return PostIterationHookResult(
            extra_messages=[
                {
                    'role': 'user',
                    'content': 'Error: exactly one tool call per assistant turn is required. Task invalidated.',
                }
            ]
        )
    if state.phase == CompositePhase.DIGITS_PENDING:
        context.tool_state[STATE_KEY] = replace(state, phase=CompositePhase.SUM)
    elif state.phase == CompositePhase.IMAGE_PENDING:
        context.tool_state[STATE_KEY] = replace(state, phase=CompositePhase.DISSECT)
    elif state.phase == CompositePhase.FINAL_PENDING:
        context.tool_state[STATE_KEY] = replace(state, phase=CompositePhase.FINAL)
    return PostIterationHookResult()


@task_registry.register('mnist_v2_task')
async def run_mnist_v2_task(task_input: MnistV2TaskInput) -> MnistV2TaskOutput:
    """Present every composite initially and score completed, correctly answered episodes."""
    context = require_context()
    if STATE_KEY in context.tool_state:
        raise RuntimeError('A composite task is already active in this execution context')
    images, user_prompt = await asyncio.to_thread(
        build_initial_observation, task_input.user_prompt, task_input.composites, task_input.input_image_long_side
    )
    context.tool_state[STATE_KEY] = CompositeTaskState(
        composites=tuple(task_input.composites),
        input_image_long_side=task_input.input_image_long_side,
        tool_image_long_side=task_input.tool_image_long_side,
        composite_images=images,
        image=images[0],
    )
    try:
        output = await task_input.run_agent(
            user_prompt=user_prompt,
            tools=TOOL_NAMES,
            parallel_tool_calls=False,
            multimodal_tool_output=True,
            image_history_limit=None,
            post_iteration_hook=HOOK_PATH,
            max_iterations=task_input.max_iterations or 2 * len(task_input.composites) + 6,
            tool_error_continue=True,
            multi_agent=None,
            tool_overrides=None,
            tool_schema_overrides=None,
            history=None,
            max_tool_response_tokens=None,
            tool_output_truncation=None,
        )
        if not isinstance(output, MainAgentOutput):
            raise TypeError('mnist_v2_task requires main_agent and its MainAgentOutput')
        state = context.tool_state[STATE_KEY]
        answer = extract_boxed_answer(output.final_answer or '')
        if answer is not None:
            answer = answer.strip()
        answer_correct = bool(
            answer is not None and re.fullmatch(r'[+-]?[0-9]+', answer) and int(answer) == int(task_input.ground_truth)
        )
        process_complete = (
            state.phase == CompositePhase.COMPLETE
            and len(state.image_results) == len(task_input.composites)
            and output.exit_status == 'finished'
            and all(
                len(message.get('tool_calls', [])) <= 1
                for message in output.main_trajectory
                if message['role'] == 'assistant'
            )
        )
        final_success = process_complete and answer_correct and state.final_result == int(task_input.ground_truth)
        digit_acc = None
        if task_input.expected_digits is not None:
            total_digits = sum(len(digits) for digits in task_input.expected_digits)
            correct_digits = sum(
                predicted == expected
                for predicted_digits, expected_digits in zip(
                    state.image_digits, task_input.expected_digits, strict=False
                )
                for predicted, expected in zip(predicted_digits, expected_digits, strict=True)
            )
            digit_acc = correct_digits / total_digits
        score = float(final_success)
        if task_input.reward_mode == 'digit_and_final':
            assert digit_acc is not None
            score = 0.5 * digit_acc + 0.5 * float(final_success) if process_complete else 0.0
        reward_metrics = {
            'mnist_v2/final_success': float(final_success),
            'mnist_v2/reward': score,
            'mnist_v2/process_complete': float(process_complete),
        }
        if digit_acc is not None:
            reward_metrics['mnist_v2/digit_acc'] = digit_acc
        return MnistV2TaskOutput(
            score=score,
            answer=answer,
            answer_correct=answer_correct,
            process_complete=process_complete,
            completed_images=len(state.image_results),
            digit_acc=digit_acc,
            final_success=bool(final_success),
            extra_fields={'reward_extra_info': {'extra_reward_metrics': reward_metrics}},
        )
    finally:
        context.tool_state.pop(STATE_KEY, None)
