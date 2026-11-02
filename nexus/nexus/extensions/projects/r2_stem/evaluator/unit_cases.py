"""Unit-cases evaluator for R2 STEM tasks."""

import json
import logging
import shlex
import time

from pathlib import Path
from typing import Any

from nexus.core import require_context
from nexus.extensions.projects.r2_stem.adapter import (
    TESTS_DIR,
    VERIFIER_LOG_DIR,
    command_error,
    reset_verifier_workspace,
)
from nexus.extensions.projects.r2_stem.evaluator.common import (
    build_verifier_test_results,
    collect_verifier_outputs,
    resolve_verifier_env,
)
from nexus.extensions.tasks.terminal.utils import read_local_directory_to_file_items
from nexus.trace import get_current_span_id, update_span


logger = logging.getLogger(__name__)


async def _upload_unit_case_tests(runtime: Any, task_dir: str) -> None:
    ctx = require_context()
    if ctx.task_files is not None:
        if not await ctx.task_files.exists('tests/test.sh'):
            raise FileNotFoundError('Unit-cases verifier requires task_files/tests/test.sh')
        logger.debug('Uploading unit-cases tests after agent completion: pattern=tests/** target=/tests')
        await ctx.task_files.download('tests/**', TESTS_DIR, runtime=runtime)
        return

    if not task_dir:
        raise ValueError('Unit-cases verifier requires task_files or task_config.task_dir')

    tests_dir = Path(task_dir) / 'tests'
    if not (tests_dir / 'test.sh').is_file():
        raise FileNotFoundError(f'Unit-cases verifier requires {tests_dir}/test.sh')

    file_items = read_local_directory_to_file_items(local_dir=tests_dir, container_base_path=TESTS_DIR)
    logger.debug('Uploading %s unit-cases test files after agent completion', len(file_items))
    upload_result = await runtime.upload_file(files=file_items, overwrite=True)
    failed = [item for item in upload_result.results if not item.success]
    if failed:
        raise RuntimeError(f'Failed to upload unit-cases tests: {failed}')


async def unit_cases_evaluate(
    *,
    runtime_name: str = 'main',
    task_dir: str = '',
    test_timeout: int = 3600,
    verifier_env: dict[str, str] | None = None,
    verifier_env_refs: dict[str, str] | None = None,
) -> tuple[float, dict[str, Any]]:
    """Upload and run task-owned tests only after the agent has returned."""
    ctx = require_context()
    runtime = await ctx.get_runtime(runtime_name)
    span_id = get_current_span_id()
    eval_runtime_id = ctx.get_runtime_id(runtime)
    if span_id and eval_runtime_id:
        update_span(span_id, attributes={'hytuner.eval_runtime_id': eval_runtime_id})

    probe = await runtime.run_command(command='echo ok', timeout=10)
    if probe.return_code != 0:
        raise command_error('Runtime health probe', probe)

    # Reset again after the agent so an agent-created /tests tree cannot shadow
    # or influence the trusted verifier files uploaded below.
    await reset_verifier_workspace(runtime)
    await _upload_unit_case_tests(runtime, task_dir)

    test_path = f'{TESTS_DIR}/test.sh'
    check = await runtime.run_command(command=f'test -f {shlex.quote(test_path)}', timeout=120)
    if check.return_code != 0:
        raise FileNotFoundError(f'Unit-cases verifier was not uploaded to {test_path}')

    inner_command = f'bash {shlex.quote(test_path)} 2>&1 | tee {VERIFIER_LOG_DIR}/test.txt'
    resolved_env = resolve_verifier_env(verifier_env, verifier_env_refs)
    started_at = time.monotonic()
    result = await runtime.invoke(
        'run_command',
        {
            'command': f'bash -o pipefail -c {shlex.quote(inner_command)}',
            'timeout': test_timeout,
            'extra_env': resolved_env,
        },
        request_timeout=test_timeout + 120,
    )
    duration_seconds = time.monotonic() - started_at
    logger.info('Unit-cases verifier finished: %s', result)

    outputs = await collect_verifier_outputs(runtime)
    if not outputs.reward_text:
        raise FileNotFoundError(f'Unit-cases verifier wrote no reward to {VERIFIER_LOG_DIR}/reward.txt')
    reward = float(outputs.reward_text)

    return reward, build_verifier_test_results(
        json.loads(outputs.ctrf_text) if outputs.ctrf_text else {},
        reward=reward,
        return_code=getattr(result, 'return_code', None),
        status=getattr(result, 'status', None),
        duration_seconds=duration_seconds,
        log_tail=outputs.log_tail,
        log_truncated=outputs.log_truncated,
        verifier_result=outputs.result,
    )


__all__ = ['unit_cases_evaluate']
