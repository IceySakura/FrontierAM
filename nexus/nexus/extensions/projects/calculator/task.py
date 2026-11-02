import logging

from nexus.core import (
    TaskRunInput,
    TaskRunOutput,
    WithAgentRunner,
    WithScore,
    WithUserPrompt,
    task_registry,
)
from nexus.core.agent import WithMainTrajectory


logger = logging.getLogger(__name__)


class CalculatorTaskInput(TaskRunInput, WithAgentRunner, WithUserPrompt):
    ground_truth: str
    """The expected result."""


class CalculatorTaskOutput(TaskRunOutput, WithScore):
    answer: str | None = None
    """The answer extracted from the agent's response."""


@task_registry.register('calculator_task')
async def run_calculator_task(task_input: CalculatorTaskInput) -> CalculatorTaskOutput:
    """Calculator task that requires the agent to compute expressions.

    The agent must use the calculate tool to evaluate expressions like:
    (7 @ 4 @ 9) @ 4 @ 3

    Where @ is a custom operator: a @ b = 3 * a - 2 * b
    """
    logger.debug(f'Running calculator task with prompt: {task_input.user_prompt}')

    # Run the agent with the user prompt directly
    agent_output = await task_input.run_agent(user_prompt=task_input.user_prompt)

    # Extract answer from the trajectory
    if not isinstance(agent_output, WithMainTrajectory):
        raise RuntimeError('Calculator task requires agent output to be a subclass of WithMainTrajectory')

    last_assistant_content = next(
        (
            msg['content']
            for msg in reversed(agent_output.main_trajectory)
            if msg['role'] == 'assistant' and msg.get('content')
        ),
        None,
    )

    answer = None
    if last_assistant_content:
        extracted = extract_boxed_answer(last_assistant_content)
        if extracted is not None:
            answer = extracted.strip()

    # Score: 1.0 if answer matches ground truth, 0.0 otherwise
    score = 1.0 if answer == task_input.ground_truth else 0.0

    logger.info(f'Answer: {answer}, Ground truth: {task_input.ground_truth}, Score: {score}')

    return CalculatorTaskOutput(
        score=score,
        answer=answer,
    )


def last_boxed_only_string(string: str) -> str | None:
    """Extract the last \\boxed{...} content from the string.

    Handles nested braces correctly.
    Adapted from: https://github.com/EleutherAI/lm-evaluation-harness
    """
    idx = string.rfind('\\boxed')
    if '\\boxed ' in string:
        return '\\boxed ' + string.split('\\boxed ')[-1].split('$')[0]
    if idx < 0:
        idx = string.rfind('\\fbox')
        if idx < 0:
            return None

    i = idx
    right_brace_idx = None
    num_left_braces_open = 0
    while i < len(string):
        if string[i] == '{':
            num_left_braces_open += 1
        if string[i] == '}':
            num_left_braces_open -= 1
            if num_left_braces_open == 0:
                right_brace_idx = i
                break
        i += 1

    return None if right_brace_idx is None else string[idx : right_brace_idx + 1]


def remove_boxed(s: str) -> str:
    """Remove \\boxed{} wrapper and return the inner content."""
    if '\\boxed ' in s:
        left = '\\boxed '
        assert s[: len(left)] == left
        return s[len(left) :]

    left = '\\boxed{'
    assert s[: len(left)] == left
    assert s[-1] == '}'
    return s[len(left) : -1]


def extract_boxed_answer(text: str) -> str | None:
    """Extract answer from \\boxed{...} in the text."""
    boxed_str = last_boxed_only_string(text)
    if boxed_str is None:
        return None
    try:
        return remove_boxed(boxed_str)
    except AssertionError:
        return None
