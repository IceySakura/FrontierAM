from __future__ import annotations

import json
import logging

from pathlib import Path
from typing import TYPE_CHECKING

from nexus.core import (
    require_context,
)
from nexus.extensions.tasks.terminal.utils import read_local_directory_to_file_items
from nexus.trace import get_current_span_id, update_span


if TYPE_CHECKING:
    from nexus.extensions.tasks.terminal.task import TerminalTaskInput

logger = logging.getLogger(__name__)


async def run_terminal_tests_in_runtime(task_input: TerminalTaskInput) -> tuple[float, dict]:
    """
    Run terminal tests in runtime and return the reward and ctrf data.
    """
    ctx = require_context()
    runtime = await ctx.get_runtime()
    span_id = get_current_span_id()
    eval_runtime_id = ctx.get_runtime_id(runtime)
    if span_id and eval_runtime_id:
        update_span(span_id, attributes={'hytuner.eval_runtime_id': eval_runtime_id})

    try:
        probe = await runtime.run_command(command='echo ok', timeout=10)
        if probe.return_code != 0:
            logger.error(f'Runtime health probe failed (return_code={probe.return_code}): {probe.stderr}')
            raise RuntimeError(f'Runtime health probe failed (return_code={probe.return_code}): {probe.stderr}')
        logger.debug('Runtime health probe passed')
    except Exception as e:
        raise RuntimeError(f'Runtime is unreachable, cannot run tests: {e}') from e

    mkdir_result = await runtime.run_command(command='mkdir -p /tests', timeout=120)
    logger.debug(f'Mkdir /tests result: {mkdir_result}')

    if ctx.task_files is not None:
        logger.debug('Downloading terminal test files from task_files: pattern=tests/** target=/tests')
        await ctx.task_files.download('tests/**', '/tests', runtime=runtime)
    else:
        if not task_input.task_dir:
            raise ValueError('Terminal verifier requires task_files or task_config.task_dir')
        file_items = read_local_directory_to_file_items(
            local_dir=Path(task_input.task_dir) / 'tests', container_base_path='/tests'
        )

        logger.debug(f'Uploading {len(file_items)} test files to runtime...')
        upload_result = await runtime.upload_file(files=file_items, overwrite=True)
        logger.debug(f'Upload test files result: {upload_result}')

    container_test_path = '/tests/test.sh'

    mkdir_result = await runtime.run_command(command='mkdir -p /logs/verifier', timeout=120)
    logger.debug(f'Mkdir /logs/verifier result: {mkdir_result}')
    command = ' '.join([
        'bash',
        str(container_test_path),
        '|',
        'tee',
        '/logs/verifier/test.txt',
        '2>&1',
    ])
    result = await runtime.invoke(
        'run_command',
        {'command': command, 'timeout': task_input.test_timeout},
        request_timeout=task_input.test_timeout + 120,
    )
    logger.debug(f'Run tests result: {result}')

    # 读取容器内的/logs/verifier/reward.txt文件和ctrf.json文件
    reward_result = await runtime.run_command(command='cat /logs/verifier/reward.txt', timeout=120)
    logger.debug(f'Read reward result: {reward_result}')
    if reward_result.stdout:
        rewards = float(reward_result.stdout)
    else:
        raise RuntimeError(f'Read reward result: {reward_result}')

    ctrf_result = await runtime.run_command(command='cat /logs/verifier/ctrf.json', timeout=120)
    logger.debug(f'Read ctrf result: {ctrf_result}')
    ctrf_data = json.loads(ctrf_result.stdout) if ctrf_result.stdout else {}

    return rewards, ctrf_data
