"""Frontend generation task.

Runs an agent in an empty working directory to generate a complete frontend
project from a one-line prompt.  The output is a tar.gz archive of the entire
working directory (including binary assets like images, databases, etc.) rather
than a git diff.
"""

import contextlib
import json
import logging

from pathlib import Path
from typing import Any

import aiofiles
import aiofiles.os

from nexus.core import (
    TaskRunInput,
    TaskRunOutput,
    WithAgentRunner,
    require_context,
    task_registry,
)
from nexus.runtime import Runtime
from nexus.trace import get_current_span_id, update_span
from nexus.trace.phases import phase
from nexus.utils.timeout import async_timeout


logger = logging.getLogger(__name__)


class FrontendTaskInput(TaskRunInput, WithAgentRunner):
    dataset_path: str = ''
    instance_id: str = ''
    prompt: str = ''
    working_dir: str = '/workspace'
    runtime_name: str = 'main'
    image_override: str | None = None
    timeout_seconds: int = 10000
    run_oracle: bool = False
    skip_infer_patch_eval: bool = False
    # Evaluation support: when task_dir is set, run tests/test.sh after agent completes
    task_dir: str = ''
    test_timeout: int = 600
    run_eval: bool = True
    eval_env: dict[str, str] = {}  # env vars injected when running test.sh (e.g. JUDGE_MODEL)


class FrontendTaskOutput(TaskRunOutput):
    completed: bool
    archive_b64: str = ''
    file_list: list[str] = []
    error: str | None = None
    termination_reason: str | None = None
    score: float | None = None


async def _load_prompt_from_dataset(dataset_path: str, instance_id: str) -> str:
    """Load prompt from a JSONL dataset by instance_id."""
    import hashlib

    path = Path(dataset_path)
    if not await aiofiles.os.path.exists(path):
        raise FileNotFoundError(f'Dataset file not found: {dataset_path}')

    async with aiofiles.open(path, encoding='utf-8', errors='replace') as f:
        async for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                data = json.loads(line)
            except json.JSONDecodeError:
                continue
            # Match by instance_id field directly
            if data.get('instance_id') == instance_id:
                return str(data.get('problem_statement', '') or data.get('prompt', '') or data.get('text', ''))
            # Fallback: match by MD5 of prompt when instance_id is missing
            if not data.get('instance_id'):
                prompt = data.get('prompt', '')
                if prompt and hashlib.md5(prompt.encode('utf-8')).hexdigest() == instance_id:
                    return str(data.get('problem_statement', '') or prompt or data.get('text', ''))

    raise ValueError(f'Instance {instance_id} not found in dataset {dataset_path}')


async def collect_project_archive(
    runtime: Runtime,
    working_dir: str = '/workspace',
) -> tuple[str, list[str]]:
    """Tar the working directory inside the runtime and return (base64, file_list)."""
    return await runtime.invoke(
        _tar_working_dir,
        working_dir,
    )


def _tar_working_dir(working_dir: str = '/workspace') -> tuple[str, list[str]]:
    """Runtime-side: tar.gz the working directory and return as base64."""
    import base64
    import io
    import os
    import tarfile

    buf = io.BytesIO()
    file_list: list[str] = []
    excluded_dirs = {
        'node_modules',
        '.git',
        '.gitignore',
        '__pycache__',
        '.next',
        'dist',
        'build',
        'out',
        'coverage',
        '.nuxt',
        '.output',
        '.svelte-kit',
        '.astro',
        '.angular',
        'storybook-static',
        '.cache',
        '.vite',
        '.parcel-cache',
        '.turbo',
        '.vercel',
        '.netlify',
    }

    with tarfile.open(fileobj=buf, mode='w:gz') as tar:
        for root, dirs, files in os.walk(working_dir):
            dirs[:] = [d for d in dirs if d not in excluded_dirs]
            for f in files:
                full = os.path.join(root, f)
                rel = os.path.relpath(full, working_dir)
                try:
                    tar.add(full, arcname=rel)
                    file_list.append(rel)
                except Exception:
                    pass

    return base64.b64encode(buf.getvalue()).decode(), file_list


async def _run_eval(
    runtime: Runtime, task_dir: str, test_timeout: int, eval_env: dict[str, str] | None = None
) -> float | None:
    """Upload tests/ and run test.sh, return score from reward.txt."""
    import shlex

    from nexus.extensions.tasks.terminal.utils import read_local_directory_to_file_items, read_local_file_to_file_item

    task_path = Path(task_dir)
    tests_dir = task_path / 'tests'
    if not tests_dir.is_dir():
        logger.warning('No tests/ directory in task_dir=%s, skipping eval', task_dir)
        return None

    # Upload tests/
    await runtime.run_command(command='mkdir -p /tests', timeout=60)
    file_items = read_local_directory_to_file_items(tests_dir, '/tests')
    logger.debug('Uploading %d test files to /tests', len(file_items))
    await runtime.upload_file(files=file_items, overwrite=True)

    # Upload instruction.md to /instruction.md (reward_request_agent.py reads from there)
    instruction_file = task_path / 'instruction.md'
    if instruction_file.is_file():
        item = read_local_file_to_file_item(instruction_file, '/instruction.md')
        await runtime.upload_file(files=[item], overwrite=True)

    # Run test.sh with eval env vars injected
    await runtime.run_command(command='mkdir -p /logs/verifier', timeout=60)
    env_prefix = ''
    if eval_env:
        env_prefix = ' '.join(f'{k}={shlex.quote(v)}' for k, v in eval_env.items() if v) + ' '
    command = f'{env_prefix}bash /tests/test.sh 2>&1 | tee /logs/verifier/test.txt'
    result = await runtime.invoke(
        'run_command',
        {'command': command, 'timeout': test_timeout},
        request_timeout=test_timeout + 120,
    )
    logger.info('Test execution result: %s', result)

    # Read reward
    reward_result = await runtime.run_command(command='cat /logs/verifier/reward.txt', timeout=60)
    score = None
    if reward_result.stdout and reward_result.stdout.strip():
        try:
            score = float(reward_result.stdout.strip())
            logger.info('Eval score: %s', score)
        except ValueError:
            logger.warning('Failed to parse reward.txt content: %r', reward_result.stdout)
    else:
        logger.warning('reward.txt is empty or missing')

    return score


@task_registry.register('frontend_task')
async def run_frontend_task(task_input: FrontendTaskInput) -> FrontendTaskOutput:
    ctx = require_context()
    if not ctx.runtime_provider:
        raise RuntimeError('Frontend task requires a runtime provider.')

    # Resolve prompt: explicit prompt > load from dataset
    prompt = task_input.prompt
    if not prompt and task_input.dataset_path and task_input.instance_id:
        prompt = await _load_prompt_from_dataset(task_input.dataset_path, task_input.instance_id)
    if not prompt:
        raise ValueError('No prompt provided and could not load from dataset.')

    runtime_name = task_input.runtime_name
    working_dir = task_input.working_dir
    term_reason = 'ENV_DONE'

    configure_params: dict[str, Any] = {}
    if task_input.image_override:
        configure_params['image'] = task_input.image_override

    try:
        if configure_params:
            ctx.runtime_provider.configure(runtime_name, **configure_params)

        phase('task.init')
        with phase('task.init.get_runtime'):
            runtime = await ctx.get_runtime(runtime_name)

        if hasattr(runtime, 'config'):
            span_id = get_current_span_id()
            runtime_id = ctx.get_runtime_id(runtime_name)
            if span_id and runtime_id:
                update_span(span_id, attributes={'hytuner.runtime_id': runtime_id})

        ctx.task_config['problem_statement'] = prompt
        ctx.task_config['working_dir'] = working_dir
        ctx.tool_state.setdefault('working_dir', working_dir)

        phase('task.agent')
        agent_output = await task_input.run_agent(
            user_prompt=prompt,
            working_dir=working_dir,
        )
        logger.info('Agent finished with exit status: %s', getattr(agent_output, 'exit_status', 'unknown'))

        # Evaluation phase (optional, triggered by task_dir)
        score = None
        if task_input.task_dir and task_input.run_eval:
            with phase('task.eval'):
                score = await _run_eval(runtime, task_input.task_dir, task_input.test_timeout, task_input.eval_env)

        with phase('task.collect'):
            async with async_timeout(300):
                archive_b64, file_list = await collect_project_archive(runtime, working_dir)

        logger.info('Collected %d files, archive size: %.1f KB', len(file_list), len(archive_b64) * 3 / 4 / 1024)

        await ctx.runtime_provider.release(name=runtime_name)

        phase()
        return FrontendTaskOutput(
            completed=True,
            archive_b64=archive_b64,
            file_list=file_list,
            score=score,
        )

    except Exception as e:
        err_str = str(e)
        if isinstance(e, TimeoutError) or 'TimeoutError' in err_str:
            term_reason = 'TASK_TIMEOUT'
        elif isinstance(e, RuntimeError) or 'RuntimeError' in err_str:
            term_reason = 'INITIALIZATION_ERROR'
        else:
            term_reason = 'UNHANDLED_ERROR'

        error_msg = f'{type(e).__name__}: {e}'
        logger.error('Frontend task failed: %s', error_msg)

        span_id = get_current_span_id()
        if span_id:
            update_span(span_id, attributes={'hytuner.termination_reason': term_reason})

        with contextlib.suppress(Exception):
            await ctx.runtime_provider.release(name=runtime_name)

        return FrontendTaskOutput(
            completed=False,
            error=error_msg,
            termination_reason=term_reason,
        )


__all__ = [
    'FrontendTaskInput',
    'FrontendTaskOutput',
    'run_frontend_task',
]
