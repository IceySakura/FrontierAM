"""GAIA benchmark task implementation.

GAIA (General AI Assistants) is a benchmark for evaluating AI assistants
on real-world questions requiring multi-step reasoning and tool use.
"""

import logging

from pathlib import Path
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


class GAIATaskInput(TaskRunInput, WithAgentRunner):
    """GAIA task input.

    Attributes:
        task_id: Unique identifier for the GAIA question
        question: The question text to answer
        level: Difficulty level (1, 2, or 3)
        file_name: Optional file that accompanies the question
        file_path: Local path to the file (will be copied to runtime)
        ground_truth: Expected answer for evaluation
        annotator_metadata: Additional metadata from annotators
    """

    task_id: str
    question: str
    level: Literal[1, 2, 3]

    # File handling
    file_name: str | None = None
    file_path: str | None = None

    # Evaluation
    ground_truth: str
    annotator_metadata: dict = Field(default_factory=dict)

    # Runtime configuration
    runtime_name: str = 'main'  # ← 新增字段
    workdir: str = '/workspace'
    timeout: int = 60  # 10 minutes per question


class GAIATaskOutput(TaskRunOutput, WithScore):
    """GAIA task output.

    Attributes:
        score: 1.0 if correct, 0.0 if incorrect
        predicted_answer: The answer extracted from agent's response
        ground_truth: The expected answer
        is_correct: Whether the answer matches ground truth
        error_message: Error message if task failed
        trajectory_length: Number of agent iterations
    """

    task_id: str
    score: float
    predicted_answer: str | None = None
    ground_truth: str
    is_correct: bool = False
    error_message: str | None = None
    trajectory_length: int = 0


@task_registry.register('gaia')
async def run_gaia_task(task_input: GAIATaskInput) -> GAIATaskOutput:
    """Run GAIA benchmark task.

    Workflow:
    1. Configure runtime with appropriate image
    2. Copy question file to runtime workspace (if exists)
    3. Run agent with the question
    4. Extract final answer from agent's response
    5. Evaluate against ground truth
    """
    logger.debug(f'Running GAIA task {task_input.task_id} (Level {task_input.level})')

    ctx = require_context()

    # Configure runtime
    if not ctx.runtime_provider:
        raise RuntimeError('GAIA task requires a runtime provider')

    # Use Python image with common tools
    ctx.runtime_provider.configure(
        'main',
        image='python:3.11-slim',
    )

    # 使用指定的 runtime 名称
    runtime = await ctx.get_runtime('main')

    try:
        # Prepare workspace with files
        if task_input.file_path and task_input.file_name:
            logger.debug(f'Copying file {task_input.file_name} to runtime')

            # Read file from host
            import base64

            with open(task_input.file_path, 'rb') as f:
                file_content = f.read()

            # Use upload_file instead of invoke
            from nexus.runtime.executors import FileItem

            target_path = f'{task_input.workdir}/{task_input.file_name}'
            file_item = FileItem(
                path=target_path, content=base64.b64encode(file_content).decode('utf-8'), encoding='base64'
            )

            await runtime.upload_file([file_item])

        # Construct prompt
        prompt = _build_prompt(
            question=task_input.question,
            file_name=task_input.file_name,
            workdir=task_input.workdir,
        )

        logger.debug(f'Running agent with prompt: {prompt[:200]}...')

        # Run agent
        agent_output = await task_input.run_agent(
            user_prompt=prompt,
            timeout=task_input.timeout,
        )
        # agent_output = await task_input.run_agent(user_prompt=task_input.user_prompt)
        # Extract answer from trajectory
        if not isinstance(agent_output, WithMainTrajectory):
            raise RuntimeError('GAIA task requires agent output to be a subclass of WithMainTrajectory')

        predicted_answer = _extract_answer(agent_output.main_trajectory)
        trajectory_length = len(agent_output.main_trajectory)

        logger.debug(f'Predicted answer: {predicted_answer}')
        logger.debug(f'Ground truth: {task_input.ground_truth}')

        # Evaluate
        is_correct = evaluate_answer(
            predicted=predicted_answer,
            ground_truth=task_input.ground_truth,
            metadata=task_input.annotator_metadata,
        )

        score = 1.0 if is_correct else 0.0

        logger.info(f'Task {task_input.task_id} - Score: {score}')

        return GAIATaskOutput(
            task_id=task_input.task_id,
            score=score,
            predicted_answer=predicted_answer,
            ground_truth=task_input.ground_truth,
            is_correct=is_correct,
            trajectory_length=trajectory_length,
        )

    except Exception as e:
        logger.error(f'Task {task_input.task_id} failed: {e}', exc_info=True)
        return GAIATaskOutput(
            task_id=task_input.task_id,
            score=0.0,
            predicted_answer=None,
            ground_truth=task_input.ground_truth,
            is_correct=False,
            error_message=str(e),
        )


def _write_file(path: str, content: bytes) -> None:
    """Helper to write file in runtime."""
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_bytes(content)


def _build_prompt(question: str, file_name: str | None, workdir: str) -> str:
    """Build agent prompt for GAIA question."""
    prompt_parts = [
        'You are a helpful AI assistant. Please answer the following question accurately.',
        '',
        f'Question: {question}',
    ]

    if file_name:
        prompt_parts.extend([
            '',
            f'A file is provided at: {workdir}/{file_name}',
            'You can read and analyze this file to help answer the question.',
        ])

    prompt_parts.extend([
        '',
        'IMPORTANT: When you have determined the final answer, wrap it in \\boxed{...} format.',
        'For example: \\boxed{42} or \\boxed{Paris}',
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

    # Extract boxed answer (reuse logic from calculator task)
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
