"""PinchBench task implementation for Nexus framework.

PinchBench is a benchmark for evaluating AI agents on real-world coding and
tool-use tasks. Each task is defined in a markdown file with:
- YAML frontmatter for metadata
- Prompt section for the task description
- Expected behavior section
- Grading criteria
- Optional automated checks (Python code)
- Optional LLM judge rubric

This module provides:
- Task and TaskLoader classes for parsing task files
- PinchBenchTaskInput/Output for Nexus integration
- run_pinchbench_task function registered with task_registry
"""

import logging
import re

from pathlib import Path
from typing import Any, Literal

import yaml

from pydantic import Field

from nexus.core import (
    TaskRunInput,
    TaskRunOutput,
    WithAgentRunner,
    WithScore,
    require_context,
    task_registry,
)

from .evaluator import GradeResult, grade_automated, grade_hybrid, grade_llm_judge


logger = logging.getLogger(__name__)


# ============== Task Model ==============


class Task:
    """Represents a single PinchBench benchmark task."""

    def __init__(
        self,
        task_id: str,
        name: str,
        category: str,
        grading_type: Literal['automated', 'llm_judge', 'hybrid'],
        timeout_seconds: int,
        workspace_files: list[dict[str, str]],
        prompt: str,
        expected_behavior: str,
        grading_criteria: list[str],
        automated_checks: str | None = None,
        llm_judge_rubric: str | None = None,
        grading_weights: dict[str, float] | None = None,
        file_path: Path | None = None,
        frontmatter: dict[str, Any] | None = None,
    ) -> None:
        self.task_id = task_id
        self.name = name
        self.category = category
        self.grading_type = grading_type
        self.timeout_seconds = timeout_seconds
        self.workspace_files = workspace_files
        self.prompt = prompt
        self.expected_behavior = expected_behavior
        self.grading_criteria = grading_criteria
        self.automated_checks = automated_checks
        self.llm_judge_rubric = llm_judge_rubric
        self.grading_weights = grading_weights
        self.file_path = file_path
        self.frontmatter = frontmatter or {}

    def __repr__(self) -> str:
        return f'Task(id={self.task_id}, name={self.name}, category={self.category})'

    def to_dict(self) -> dict[str, Any]:
        """Convert task to dictionary representation."""
        return {
            'task_id': self.task_id,
            'name': self.name,
            'category': self.category,
            'grading_type': self.grading_type,
            'timeout_seconds': self.timeout_seconds,
            'workspace_files': self.workspace_files,
            'prompt': self.prompt,
            'expected_behavior': self.expected_behavior,
            'grading_criteria': self.grading_criteria,
            'has_automated_checks': self.automated_checks is not None,
            'has_llm_judge_rubric': self.llm_judge_rubric is not None,
            'grading_weights': self.grading_weights,
            'frontmatter': self.frontmatter,
        }


class TaskLoader:
    """Loads and parses PinchBench task files from a directory."""

    def __init__(self, tasks_dir: Path) -> None:
        self.tasks_dir = tasks_dir
        logger.debug(f'Initialized TaskLoader with directory: {tasks_dir}')

    def load_all_tasks(self) -> list[Task]:
        """Load all task files from the tasks directory."""
        tasks = []
        task_files = sorted(self.tasks_dir.glob('task_*.md'))

        logger.debug(f'Found {len(task_files)} task files')

        for task_file in task_files:
            try:
                task = self.load_task(task_file)
                tasks.append(task)
                logger.debug(f'Successfully loaded task: {task.task_id}')
            except Exception as e:
                logger.error(f'Failed to load task from {task_file}: {e}', exc_info=True)

        logger.debug(f'Successfully loaded {len(tasks)} tasks')
        return tasks

    def load_task(self, task_file: Path) -> Task:
        """Load and parse a single task file."""
        logger.debug(f'Loading task from: {task_file}')

        content = task_file.read_text(encoding='utf-8')

        # Extract YAML frontmatter
        frontmatter_match = re.match(r'^---\s*\n(.*?)\n---\s*\n(.*)$', content, re.DOTALL)
        if not frontmatter_match:
            raise ValueError(f'No YAML frontmatter found in {task_file}')

        frontmatter_text = frontmatter_match.group(1)
        body_text = frontmatter_match.group(2)

        # Parse YAML frontmatter
        try:
            metadata = yaml.safe_load(frontmatter_text)
        except yaml.YAMLError as e:
            raise ValueError(f'Invalid YAML frontmatter in {task_file}: {e}') from e

        # Extract sections from body
        sections = self._parse_sections(body_text)

        # Extract grading criteria
        grading_criteria = self._extract_grading_criteria(sections.get('Grading Criteria', ''))

        # Create Task object
        return Task(
            task_id=metadata.get('id', ''),
            name=metadata.get('name', ''),
            category=metadata.get('category', ''),
            grading_type=metadata.get('grading_type', 'automated'),
            timeout_seconds=metadata.get('timeout_seconds', 120),
            workspace_files=metadata.get('workspace_files', []),
            prompt=sections.get('Prompt', '').strip(),
            expected_behavior=sections.get('Expected Behavior', '').strip(),
            grading_criteria=grading_criteria,
            automated_checks=sections.get('Automated Checks', None),
            llm_judge_rubric=sections.get('LLM Judge Rubric', None),
            grading_weights=metadata.get('grading_weights', None),
            file_path=task_file,
            frontmatter=metadata,
        )

    def _parse_sections(self, body: str) -> dict[str, str]:
        """Parse markdown sections from task body."""
        sections: dict[str, str] = {}
        current_section: str | None = None
        current_content: list[str] = []

        for line in body.split('\n'):
            # Check for section headers (## Header)
            header_match = re.match(r'^##\s+(.+)$', line)
            if header_match:
                # Save previous section
                if current_section:
                    sections[current_section] = '\n'.join(current_content).strip()

                # Start new section
                current_section = header_match.group(1)
                current_content = []
            else:
                if current_section:
                    current_content.append(line)

        # Save last section
        if current_section:
            sections[current_section] = '\n'.join(current_content).strip()

        return sections

    def _extract_grading_criteria(self, criteria_text: str) -> list[str]:
        """Extract grading criteria from checklist format."""
        criteria: list[str] = []
        for line in criteria_text.split('\n'):
            # Match checklist items: - [ ] or - [x]
            match = re.match(r'^-\s+\[[ x]\]\s+(.+)$', line.strip())
            if match:
                criteria.append(match.group(1))
        return criteria


# ============== Nexus Task Input/Output ==============


class PinchBenchTaskInput(TaskRunInput, WithAgentRunner):
    """Input for a PinchBench task.

    Attributes:
        task_file: Path to the PinchBench task markdown file
        workspace_path: Directory inside runtime where agent writes output
        assets_path: Path to assets directory (for workspace file sources)
        runtime_name: Name of the runtime to use
        workdir: Working directory in the runtime
        timeout: Timeout for agent execution in seconds
    """

    task_file: str
    """Path to the PinchBench task markdown file (absolute or relative to CWD)."""

    workspace_path: str = '/workspace'
    """Directory inside the runtime where the agent writes its output files."""

    assets_path: str | None = None
    """Path to assets directory for workspace file sources."""

    runtime_name: str = 'main'
    """Name of the runtime to use."""

    workdir: str = '/workspace'
    """Working directory in the runtime."""

    timeout: int = 300
    """Timeout for agent execution in seconds."""


class PinchBenchTaskOutput(TaskRunOutput, WithScore):
    """Output for a PinchBench task.

    Attributes:
        task_id: Unique task identifier
        score: Overall score (0.0 to 1.0)
        max_score: Maximum possible score
        grading_type: Type of grading used
        breakdown: Per-criterion scores
        notes: Grading notes or justifications
        error_message: Error message if task failed
    """

    task_id: str = ''
    score: float = 0.0
    max_score: float = 1.0
    grading_type: str = ''
    breakdown: dict[str, float] = Field(default_factory=dict)
    notes: str = ''
    error_message: str | None = None
    trajectory_length: int = 0


# ============== Task Function ==============


@task_registry.register('pinchbench')
async def run_pinchbench_task(task_input: PinchBenchTaskInput) -> PinchBenchTaskOutput:
    """Run a single PinchBench task and grade the result.

    Workflow:
    1. Load and parse the task markdown file
    2. Configure runtime with appropriate image
    3. Copy workspace files to runtime (if specified)
    4. Run agent with the task prompt
    5. Grade based on grading_type (automated, llm_judge, or hybrid)
    6. Return result with score and breakdown

    Args:
        task_input: PinchBenchTaskInput with task file path and settings

    Returns:
        PinchBenchTaskOutput with score and breakdown
    """
    logger.debug(f'Running PinchBench task from: {task_input.task_file}')

    # Load task
    task_path = Path(task_input.task_file)
    if not task_path.is_absolute():
        task_path = Path.cwd() / task_path

    loader = TaskLoader(task_path.parent)
    try:
        task = loader.load_task(task_path)
    except Exception as e:
        logger.error(f'Failed to load task: {e}')
        return PinchBenchTaskOutput(
            task_id='',
            score=0.0,
            error_message=f'Failed to load task: {e}',
        )

    logger.debug(f'Task: {task.task_id} - {task.name}')
    logger.debug(f'Category: {task.category}, Grading: {task.grading_type}')

    # Get context and configure runtime
    ctx = require_context()
    runtime = await ctx.get_runtime(task_input.runtime_name)

    try:
        # Prepare workspace with files
        await _prepare_workspace(
            task=task,
            runtime=runtime,
            assets_path=task_input.assets_path,
            workspace_path=task_input.workspace_path,
        )

        # Build prompt
        prompt = _build_prompt(task)

        logger.debug(f'Running agent with prompt (first 200 chars): {prompt[:200]}...')

        # Run agent
        agent_output = await task_input.run_agent(
            user_prompt=prompt,
            timeout=task.timeout_seconds,
        )

        # Extract trajectory info
        trajectory_length = 0
        transcript_data = getattr(agent_output, 'transcript', None)
        if transcript_data is not None:
            trajectory_length = len(transcript_data)

        # Grade the result
        grade_result = await _grade_task(
            task=task,
            agent_output=agent_output,
            workspace_path=task_input.workspace_path,
        )

        logger.info(f'Task {task.task_id} - Score: {grade_result.score}/{grade_result.max_score}')

        return PinchBenchTaskOutput(
            task_id=task.task_id,
            score=grade_result.score,
            max_score=grade_result.max_score,
            grading_type=grade_result.grading_type,
            breakdown=grade_result.breakdown,
            notes=grade_result.notes,
            trajectory_length=trajectory_length,
        )

    except Exception as e:
        logger.error(f'Task {task.task_id} failed: {e}', exc_info=True)
        return PinchBenchTaskOutput(
            task_id=task.task_id,
            score=0.0,
            error_message=str(e),
        )


# ============== Helper Functions ==============


async def _prepare_workspace(
    *,
    task: Task,
    runtime: Any,
    assets_path: str | None,
    workspace_path: str,
) -> None:
    """Copy workspace files to the runtime."""
    import base64

    from nexus.runtime.executors import FileItem

    for file_spec in task.workspace_files:
        if 'content' in file_spec:
            # Inline content
            dest = f'{workspace_path}/{file_spec["path"]}'
            file_item = FileItem(
                path=dest,
                content=base64.b64encode(file_spec['content'].encode()).decode(),
                encoding='base64',
            )
            await runtime.upload_file([file_item])
            logger.debug(f'Created inline workspace file: {dest}')
            continue

        # File from assets directory
        if not assets_path:
            logger.warning(f'No assets_path provided but file_spec has source: {file_spec}')
            continue

        source_path = Path(assets_path) / file_spec['source']
        dest = file_spec.get('dest', file_spec['source'])
        dest_path = f'{workspace_path}/{dest}'

        with open(source_path, 'rb') as f:
            content = f.read()

        file_item = FileItem(
            path=dest_path,
            content=base64.b64encode(content).decode(),
            encoding='base64',
        )
        await runtime.upload_file([file_item])
        logger.debug(f'Copied workspace file: {source_path} -> {dest_path}')


def _build_prompt(task: Task) -> str:
    """Build the agent prompt for the task."""
    return task.prompt


async def _grade_task(
    *,
    task: Task,
    agent_output: Any,
    workspace_path: str,
) -> GradeResult:
    """Grade the task based on grading_type."""
    if task.grading_type == 'automated':
        # Get transcript from agent output
        transcript: list[dict] = getattr(agent_output, 'transcript', None) or []

        return await grade_automated(
            task_id=task.task_id,
            grading_code=task.automated_checks or '',
            transcript=transcript,
            workspace_path=workspace_path,
        )

    if task.grading_type == 'llm_judge':
        return await grade_llm_judge(
            task_id=task.task_id,
            prompt=task.prompt,
            expected_behavior=task.expected_behavior,
            grading_criteria=task.grading_criteria,
            llm_judge_rubric=task.llm_judge_rubric,
            agent_output=agent_output,
        )

    if task.grading_type == 'hybrid':
        # Get transcript for automated grading
        transcript: list[dict] = getattr(agent_output, 'transcript', None) or []

        auto_result = await grade_automated(
            task_id=task.task_id,
            grading_code=task.automated_checks or '',
            transcript=transcript,
            workspace_path=workspace_path,
        )

        llm_result = await grade_llm_judge(
            task_id=task.task_id,
            prompt=task.prompt,
            expected_behavior=task.expected_behavior,
            grading_criteria=task.grading_criteria,
            llm_judge_rubric=task.llm_judge_rubric,
            agent_output=agent_output,
        )

        return grade_hybrid(
            task_id=task.task_id,
            auto_result=auto_result,
            llm_result=llm_result,
            weights=task.grading_weights,
        )

    raise ValueError(f'Unknown grading type: {task.grading_type}')


__all__ = [
    'Task',
    'TaskLoader',
    'PinchBenchTaskInput',
    'PinchBenchTaskOutput',
    'run_pinchbench_task',
]
