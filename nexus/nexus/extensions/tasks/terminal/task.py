"""Echo task example using ExecutionContext.

This task demonstrates:
- Accessing runtime via require_context().get_runtime()
- Using WithAgentRunner for agent execution
"""

import asyncio
import logging
import shlex
import time

from pathlib import Path
from typing import Any

from nexus.core import (
    TaskRunInput,
    TaskRunOutput,
    WithAgentRunner,
    WithScore,
    require_context,
    task_registry,
)
from nexus.extensions.tasks.terminal.eval import run_terminal_tests_in_runtime
from nexus.extensions.tasks.terminal.utils import read_local_directory_to_file_items
from nexus.trace import get_current_span_id, get_processor, update_span
from nexus.trace.phases import phase
from nexus.trace.processors.chat_tracker import ChatTrackerProcessor


logger = logging.getLogger(__name__)


class TerminalTaskInput(TaskRunInput, WithAgentRunner):
    """Terminal task input - business parameters only."""

    message: str
    run_oracle: bool = False
    task_dir: str = ''
    test_timeout: int = 3600
    agent_timeout: int = 3600  # Agent 执行超时时间（秒），默认 1 小时
    docker_timeout: int = 3600
    # Output quality monitoring (controlled by hytuner via task_config).
    # When False the monitor block is skipped entirely.
    enable_output_monitoring: bool = False
    # Penalty coefficients — injected by hytuner from agent_config.
    # All default to 0/False so the penalty path is a no-op when unconfigured.
    codeswitch_penalty: float = 0.0
    codeswitch_penalty_reward_threshold: float = 1.0
    output_format_bad_pattern_penalty: float = 0.0
    output_format_bad_pattern_reward_threshold: float = 1.0
    let_me_loop_penalty: float = 0.0
    let_me_loop_reward_threshold: float = 1.0
    tbench_json_fix: bool = False


class TerminalTaskOutput(TaskRunOutput, WithScore):
    """Terminal task output with score."""

    test_results: dict | None
    extra_fields: dict[str, Any] = {}


async def run_oracle(task_input: TerminalTaskInput) -> None:
    ctx = require_context()
    runtime = await ctx.get_runtime()
    mkdir_result = await runtime.run_command(command='mkdir -p /solution', timeout=60)
    logger.debug(f'Mkdir /solution result: {mkdir_result}')

    if ctx.task_files is not None:
        logger.debug('Downloading terminal oracle files from task_files: pattern=solution/** target=/solution')
        await ctx.task_files.download('solution/**', '/solution', runtime=runtime)
    else:
        if not task_input.task_dir:
            raise ValueError('Terminal oracle requires task_files or task_config.task_dir')
        file_items = read_local_directory_to_file_items(
            local_dir=Path(task_input.task_dir) / 'solution', container_base_path='/solution'
        )
        logger.debug(f'Uploading {len(file_items)} solution files to runtime...')
        upload_result = await runtime.upload_file(files=file_items, overwrite=True)
        logger.debug(f'Upload solution files result: {upload_result}')

    container_solve_path = '/solution/solve.sh'

    # 构建执行命令
    # set -o pipefail: 确保管道中任何命令失败都返回非零退出码
    # shlex.quote: 安全转义路径，防止命令注入
    # 2>&1: 将 stderr 重定向到 stdout
    # tee: 同时输出到终端并写入日志文件
    # 最终命令例如: set -o pipefail; bash '/solution/solve.sh' 2>&1 | tee /logs/agent/oracle.txt
    mkdir_result = await runtime.run_command(command='mkdir -p /logs/agent', timeout=60)
    logger.debug(f'Mkdir /logs/agent result: {mkdir_result}')
    # command = (
    #     "set -o pipefail; "
    #     f"bash {shlex.quote(str(container_solve_path))} 2>&1 | tee /logs/agent/oracle.txt"
    # )
    command = f'bash {shlex.quote(str(container_solve_path))}'
    result = await runtime.run_command(command=command, timeout=9999)
    logger.debug(f'Run oracle result: {result}')
    # 检查脚本执行结果，非零返回码表示执行失败
    if result.return_code != 0:
        raise RuntimeError(f'Oracle solution exited with result: {result}')


@task_registry.register('terminal_task')
async def terminal_task(task_input: TerminalTaskInput) -> TerminalTaskOutput:
    """Terminal task that requires the agent to run a terminal task."""
    logger.debug(f'Running terminal task with message {task_input.message}')

    ctx = require_context()
    if not ctx.runtime_provider:
        raise RuntimeError('Terminal task requires a runtime provider to be configured')

    phase('task.init')
    with phase('task.init.get_runtime'):
        runtime = await ctx.get_runtime()
    init_result = await runtime.run_command(command='mkdir -p /app', timeout=60)
    logger.debug(f'Init result: {init_result}')

    runtime_id = ctx.get_runtime_id(runtime)
    span_id = get_current_span_id()
    if span_id and runtime_id:
        update_span(span_id, attributes={'hytuner.runtime_id': runtime_id})

    phase('task.agent')
    if task_input.run_oracle:
        await run_oracle(task_input)
    else:
        try:
            await asyncio.wait_for(
                task_input.run_agent(user_prompt=task_input.message), timeout=task_input.agent_timeout
            )
        except asyncio.TimeoutError:
            logger.warning(
                f'Agent execution timed out after {task_input.agent_timeout} seconds, '
                f'continuing with test execution to check if work was completed'
            )
            # 不抛出异常，继续执行测试
            # wait_for 取消了 agent，它自己的 exit status 上报因此不会执行。这道墙由 task
            # 设置，也只能由 task 上报；否则轨迹带着空的 termination_reason 到 hytuner，
            # 回落成 ENV_DONE，与 agent 自行结束的样本无法区分。
            span_id = get_current_span_id()
            if span_id:
                update_span(span_id, attributes={'hytuner.termination_reason': 'TASK_TIMEOUT'})

    # 第4步: 执行测试
    phase('task.scoring')
    reward_start = time.monotonic()
    reward, ctrf_data = await run_terminal_tests_in_runtime(task_input)
    reward_time = time.monotonic() - reward_start

    # 返回结果
    span_id = get_current_span_id()
    if span_id and reward_time > 0:
        update_span(span_id, attributes={'hytuner.reward_time': reward_time})

    phase()  # 关闭剩余的 phase span
    score = int(reward) == 1

    # Output quality monitoring and penalty application.
    extra_fields: dict[str, Any] = {}
    if task_input.enable_output_monitoring:
        try:
            from nexus.extensions.tasks.terminal.reward_hacking.monitors import run_terminal_monitors

            processor = get_processor(ChatTrackerProcessor)
            if processor is not None:
                snapshot = processor.snapshot_result() or {}
                msgs: list[dict[str, Any]] = []
                for traj in snapshot.get('trajectories') or []:
                    if isinstance(traj, dict):
                        msgs.extend(traj.get('messages') or [])
                    elif isinstance(traj, list):
                        msgs.extend(traj)
                if msgs:
                    extra_fields.update(run_terminal_monitors(msgs))
        except Exception as exc:
            logger.warning('output monitoring skipped (fail-open): %s', exc)

        # Apply penalties based on detection results.
        try:
            from nexus.extensions.tasks.terminal.reward_hacking.penalties import apply_terminal_penalties

            penalized_score = apply_terminal_penalties(
                float(score),
                extra_fields,
                codeswitch_penalty=task_input.codeswitch_penalty,
                codeswitch_penalty_reward_threshold=task_input.codeswitch_penalty_reward_threshold,
                output_format_bad_pattern_penalty=task_input.output_format_bad_pattern_penalty,
                output_format_bad_pattern_reward_threshold=task_input.output_format_bad_pattern_reward_threshold,
                let_me_loop_penalty=task_input.let_me_loop_penalty,
                let_me_loop_reward_threshold=task_input.let_me_loop_reward_threshold,
                tbench_json_fix=task_input.tbench_json_fix,
            )
            if penalized_score is not None:
                score = penalized_score
        except Exception as exc:
            logger.warning('penalty application skipped (fail-open): %s', exc)

    return TerminalTaskOutput(
        score=score,
        test_results=ctrf_data,
        extra_fields=extra_fields,
    )
