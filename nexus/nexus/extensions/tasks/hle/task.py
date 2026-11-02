"""HLE benchmark task implementation.

HLE (Human-Like Evaluation) is a benchmark for evaluating AI assistants
on tasks that require human-like reasoning, understanding, and problem-solving.
"""

import logging

from typing import Literal

from pydantic import Field

from nexus.core import (
    TaskRunInput,
    TaskRunOutput,
    WithAgentRunner,
    WithScore,
    require_context,
    task_registry,
)
from nexus.core.agent import WithMainTrajectory

from .evaluator import evaluate_answer


logger = logging.getLogger(__name__)


class HLETaskInput(TaskRunInput, WithAgentRunner):
    """HLE task input.

    Attributes:
        task_id: Unique identifier for the HLE question
        question: The question text to answer
        category: Task category (e.g., reasoning, knowledge, coding)
        difficulty: Difficulty level (easy, medium, hard)
        file_names: Optional list of files that accompany the question
        file_paths: Local paths to the files (will be copied to runtime)
        ground_truth: Expected answer for evaluation
        evaluation_type: Type of evaluation (exact, fuzzy, semantic)
        metadata: Additional metadata
    """

    task_id: str
    question: str
    category: str

    # File handling
    file_names: list[str] | None = None
    file_paths: list[str] | None = None

    # Evaluation
    ground_truth: str | list[str]  # Can be single answer or multiple acceptable answers
    evaluation_type: Literal['exact', 'fuzzy', 'semantic'] = 'fuzzy'
    metadata: dict = Field(default_factory=dict)

    # Runtime configuration
    runtime_name: str = 'main'
    workdir: str = '/workspace'
    timeout: int = 60  # 1 minutes per question


class HLETaskOutput(TaskRunOutput, WithScore):
    """HLE task output.

    Attributes:
        score: Score between 0.0 and 1.0
        predicted_answer: The answer extracted from agent's response
        ground_truth: The expected answer(s)
        is_correct: Whether the answer is considered correct
        error_message: Error message if task failed
        trajectory_length: Number of agent iterations
        evaluation_details: Additional evaluation information
    """

    task_id: str
    score: float
    predicted_answer: str | None = None
    ground_truth: str | list[str]
    is_correct: bool = False
    error_message: str | None = None
    trajectory_length: int = 0
    evaluation_details: dict = Field(default_factory=dict)


@task_registry.register('hle')
async def run_hle_task(task_input: HLETaskInput) -> HLETaskOutput:
    """Run HLE benchmark task.

    Workflow:
    1. Configure runtime with appropriate image
    2. Copy question files to runtime workspace (if exists)
    3. Run agent with the question
    4. Extract final answer from agent's response
    5. Evaluate against ground truth
    """
    logger.debug(f'Running HLE task {task_input.task_id} (Category: {task_input.category})')

    ctx = require_context()

    # # Configure runtime
    # if not ctx.runtime_provider:
    #     raise RuntimeError('HLE task requires a runtime provider')

    # # Use Python image with common tools
    # ctx.runtime_provider.configure(
    #     'main',  # 固定使用 'main' 作为 runtime 名称
    #     image='python:3.11-slim',
    # )

    # Get runtime
    runtime = await ctx.get_runtime('main')

    try:
        # Prepare workspace with files
        if task_input.file_paths and task_input.file_names:
            logger.debug(f'Copying {len(task_input.file_names)} file(s) to runtime')

            import base64

            from nexus.runtime.executors import FileItem

            file_items = []
            for file_path, file_name in zip(task_input.file_paths, task_input.file_names, strict=False):
                # Read file from host
                with open(file_path, 'rb') as f:
                    file_content = f.read()

                target_path = f'{task_input.workdir}/{file_name}'
                file_item = FileItem(
                    path=target_path, content=base64.b64encode(file_content).decode('utf-8'), encoding='base64'
                )
                file_items.append(file_item)

            await runtime.upload_file(file_items)

        # Construct prompt
        prompt = _build_prompt(
            question=task_input.question,
            category=task_input.category,
            file_names=task_input.file_names,
            workdir=task_input.workdir,
        )

        logger.debug(f'Running agent with prompt: {prompt[:200]}...')

        # Run agent
        agent_output = await task_input.run_agent(
            user_prompt=prompt,
            timeout=task_input.timeout,
        )

        # Extract answer from trajectory
        if not isinstance(agent_output, WithMainTrajectory):
            raise RuntimeError('HLE task requires agent output to be a subclass of WithMainTrajectory')

        predicted_answer = _extract_answer(agent_output.main_trajectory)
        trajectory_length = len(agent_output.main_trajectory)

        logger.debug(f'Predicted answer: {predicted_answer}')
        logger.debug(f'Ground truth: {task_input.ground_truth}')

        # Evaluate
        is_correct, score, eval_details = evaluate_answer(
            predicted=predicted_answer,
            ground_truth=task_input.ground_truth,
            evaluation_type=task_input.evaluation_type,
            metadata=task_input.metadata,
        )

        logger.info(f'Task {task_input.task_id} - Score: {score:.3f}, Correct: {is_correct}')

        return HLETaskOutput(
            task_id=task_input.task_id,
            score=score,
            predicted_answer=predicted_answer,
            ground_truth=task_input.ground_truth,
            is_correct=is_correct,
            trajectory_length=trajectory_length,
            evaluation_details=eval_details,
        )

    except Exception as e:
        logger.error(f'Task {task_input.task_id} failed: {e}', exc_info=True)
        return HLETaskOutput(
            task_id=task_input.task_id,
            score=0.0,
            predicted_answer=None,
            ground_truth=task_input.ground_truth,
            is_correct=False,
            error_message=str(e),
        )


def _build_prompt(question: str, category: str, file_names: list[str] | None, workdir: str) -> str:
    """Build agent prompt for HLE question."""
    prompt_parts = [
        'You are a helpful AI assistant. Please answer the following question accurately and thoroughly.',
        '',
        f'Category: {category}',
        f'Question: {question}',
    ]

    if file_names:
        prompt_parts.extend([
            '',
            'Provided files:',
        ])
        for file_name in file_names:
            prompt_parts.append(f'  - {workdir}/{file_name}')
        prompt_parts.append('You can read and analyze these files to help answer the question.')

    prompt_parts.extend([
        '',
        'IMPORTANT: When you have determined the final answer, wrap it in \\boxed{...} format.',
        'For example: \\boxed{42} or \\boxed{Paris} or \\boxed{The answer is...}',
        '',
        'Please provide your answer now.',
    ])

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
