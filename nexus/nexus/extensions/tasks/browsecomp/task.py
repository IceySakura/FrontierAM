"""BrowseComp benchmark task implementation.

BrowseComp is a benchmark for evaluating AI agents' ability to browse
and comprehend web content to answer complex questions.
"""

import logging

from typing import Literal

from pydantic import Field

from nexus.core import (
    TaskRunInput,
    TaskRunOutput,
    WithAgentRunner,
    WithScore,
    task_registry,
)
from nexus.core.agent import WithMainTrajectory

from .evaluator import evaluate_answer


logger = logging.getLogger(__name__)


class BrowseCompTaskInput(TaskRunInput, WithAgentRunner):
    """BrowseComp task input.

    Attributes:
        task_id: Unique identifier for the BrowseComp question
        question: The question text to answer (requires web browsing)
        problem_topic: Task topic/category (e.g., Art, History, Sports)
        answer: Ground truth answer for evaluation
        evaluation_type: Type of evaluation (exact, fuzzy)
        metadata: Additional metadata
        runtime_name: Runtime name to use
        workdir: Working directory in runtime
        timeout: Timeout in seconds
    """

    task_id: str
    question: str
    problem_topic: str = 'General'

    # Evaluation
    answer: str  # Ground truth answer
    evaluation_type: Literal['exact', 'fuzzy'] = 'fuzzy'
    metadata: dict = Field(default_factory=dict)

    # Runtime configuration
    runtime_name: str = 'main'
    workdir: str = '/workspace'
    timeout: int = 60  # 5 minutes per question (browsing takes time)


class BrowseCompTaskOutput(TaskRunOutput, WithScore):
    """BrowseComp task output.

    Attributes:
        task_id: Task identifier
        score: Score between 0.0 and 1.0
        predicted_answer: The answer extracted from agent's response
        ground_truth: The expected answer
        is_correct: Whether the answer is considered correct
        error_message: Error message if task failed
        trajectory_length: Number of agent iterations
        evaluation_details: Additional evaluation information
    """

    task_id: str
    score: float
    predicted_answer: str | None = None
    ground_truth: str
    is_correct: bool = False
    error_message: str | None = None
    trajectory_length: int = 0
    evaluation_details: dict = Field(default_factory=dict)


@task_registry.register('browsecomp')
async def run_browsecomp_task(task_input: BrowseCompTaskInput) -> BrowseCompTaskOutput:
    """Run BrowseComp benchmark task.

    Workflow:
    1. Configure runtime with appropriate image
    2. Get runtime environment
    3. Run agent with the question (agent should use web browsing tools)
    4. Extract final answer from agent's response
    5. Evaluate against ground truth
    """
    logger.debug(f'Running BrowseComp task {task_input.task_id} (Topic: {task_input.problem_topic})')

    try:
        # Construct prompt
        prompt = _build_prompt(
            question=task_input.question,
            problem_topic=task_input.problem_topic,
        )

        logger.debug(f'Running agent with prompt: {prompt[:200]}...')

        # Run agent
        agent_output = await task_input.run_agent(
            user_prompt=prompt,
            timeout=task_input.timeout,
        )

        # Extract answer from trajectory
        if not isinstance(agent_output, WithMainTrajectory):
            raise RuntimeError('BrowseComp task requires agent output to be a subclass of WithMainTrajectory')

        predicted_answer = _extract_answer(agent_output.main_trajectory)
        trajectory_length = len(agent_output.main_trajectory)

        logger.debug(f'Predicted answer: {predicted_answer}')
        logger.debug(f'Ground truth: {task_input.answer}')

        # Evaluate
        is_correct, score, eval_details = evaluate_answer(
            predicted=predicted_answer,
            ground_truth=task_input.answer,
            evaluation_type=task_input.evaluation_type,
            metadata=task_input.metadata,
        )

        logger.info(f'Task {task_input.task_id} - Score: {score:.3f}, Correct: {is_correct}')

        return BrowseCompTaskOutput(
            task_id=task_input.task_id,
            score=score,
            predicted_answer=predicted_answer,
            ground_truth=task_input.answer,
            is_correct=is_correct,
            trajectory_length=trajectory_length,
            evaluation_details=eval_details,
        )

    except Exception as e:
        logger.error(f'Task {task_input.task_id} failed: {e}', exc_info=True)
        return BrowseCompTaskOutput(
            task_id=task_input.task_id,
            score=0.0,
            predicted_answer=None,
            ground_truth=task_input.answer,
            is_correct=False,
            error_message=str(e),
        )


def _build_prompt(
    question: str,
    problem_topic: str,
) -> str:
    """Build agent prompt for BrowseComp question."""
    prompt_parts = [
        'You are a helpful AI assistant with web browsing capabilities. '
        'Please use web search and browsing to find accurate information to answer the following question.',
        '',
        f'Topic: {problem_topic}',
        f'Question: {question}',
        '',
        'Instructions:',
        '1. Use web search to find relevant information',
        '2. Browse and analyze web pages carefully',
        '3. Extract and verify the answer from reliable sources',
        '4. Provide a concise, accurate answer',
        '',
        'IMPORTANT: When you have determined the final answer, wrap it in \\boxed{...} format.',
        'For example: \\boxed{1988-96} or \\boxed{Paris} or \\boxed{John Smith}',
        '',
        'Please provide your answer now.',
    ]

    return '\n'.join(prompt_parts)


def _extract_answer(trajectory: list[dict]) -> str | None:
    """Extract the final answer from agent's trajectory.

    Looks for \\boxed{...} in the last assistant message.
    """
    # Find last assistant message
    last_content = None
    for msg in reversed(trajectory):
        if msg.get('role') == 'assistant' and msg.get('content'):
            last_content = msg['content']
            break

    if not last_content:
        return None

    # Extract boxed answer
    return _extract_boxed(last_content)


def _extract_boxed(text: str) -> str | None:
    """Extract content from \\boxed{...} format."""
    idx = text.rfind('\\boxed{')
    if idx < 0:
        return None

    i = idx + len('\\boxed{')
    brace_count = 1
    result = []

    while i < len(text) and brace_count > 0:
        if text[i] == '{':
            brace_count += 1
        elif text[i] == '}':
            brace_count -= 1
            if brace_count == 0:
                break
        result.append(text[i])
        i += 1

    return ''.join(result).strip() if brace_count == 0 else None
