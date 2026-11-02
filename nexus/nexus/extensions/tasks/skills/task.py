"""SkillsBench task — 与 terminal_task 评测流程完全一致。

SkillsBench 的 skills 已通过 Dockerfile 打包进镜像（COPY skills /root/.claude/skills 等），
评测层面无需额外处理，直接复用 terminal 的 eval 逻辑即可。
"""

import asyncio
import json
import logging
import shlex

from pathlib import Path

from nexus.core import (
    TaskRunInput,
    TaskRunOutput,
    WithAgentRunner,
    WithScore,
    require_context,
    task_registry,
)
from nexus.extensions.tasks.skills.utils import read_local_directory_to_file_items, read_local_file_to_file_item


logger = logging.getLogger(__name__)


class SkillsTaskInput(TaskRunInput, WithAgentRunner):
    """SkillsBench task input."""

    message: str
    run_oracle: bool = False
    task_dir: str = ''
    test_timeout: int = 3600
    agent_timeout: int = 3600
    docker_timeout: int = 3600
    verifier_env: dict[str, str] = {}
    solution_env: dict[str, str] = {}
    solution_files: list[dict] = []
    verifier_files: list[dict] = []


class SkillsTaskOutput(TaskRunOutput, WithScore):
    """SkillsBench task output with score."""

    test_results: dict | None


async def _upload_task_files(file_configs: list[dict], task_dir: str) -> None:
    """上传 task.toml 中配置的文件到容器。

    每个 file_config 包含:
      - path: 容器内的目标路径（绝对路径）
      - local_path: 本地文件路径（相对于 task_dir 或绝对路径）

    通过 runtime 的 upload_file API 上传。
    """
    if not file_configs:
        return

    ctx = require_context()
    runtime = await ctx.get_runtime()

    file_items = []
    task_dir_path = Path(task_dir)

    for fc in file_configs:
        container_path = fc.get('path', '')
        local_path_str = fc.get('local_path', '')

        if not container_path or not local_path_str:
            logger.warning(f'跳过无效的文件配置: {fc}')
            continue

        local_path = Path(local_path_str)
        if not local_path.is_absolute():
            local_path = task_dir_path / local_path

        if not local_path.exists():
            logger.warning(f'本地文件不存在，跳过: {local_path}')
            continue

        file_item = read_local_file_to_file_item(local_path, container_path)
        file_items.append(file_item)
        logger.debug(f'准备上传文件: {local_path} -> {container_path}')

    if file_items:
        logger.debug(f'上传 {len(file_items)} 个配置文件到容器...')
        await runtime.upload_file(files=file_items, overwrite=True)


async def _run_tests(task_input: SkillsTaskInput) -> tuple[float, dict]:
    ctx = require_context()
    runtime = await ctx.get_runtime()

    await runtime.run_command(command='mkdir -p /tests', timeout=120)
    file_items = read_local_directory_to_file_items(
        local_dir=Path(task_input.task_dir) / 'tests', container_base_path='/tests'
    )
    logger.debug(f'Uploading {len(file_items)} test files to runtime...')
    await runtime.upload_file(files=file_items, overwrite=True)

    # 上传 verifier 配置的文件
    await _upload_task_files(task_input.verifier_files, task_input.task_dir)

    await runtime.run_command(command='mkdir -p /logs/verifier', timeout=120)
    command = 'bash /tests/test.sh | tee /logs/verifier/test.txt 2>&1'
    result = await runtime.run_command(
        command=command,
        timeout=task_input.test_timeout,
        extra_env=task_input.verifier_env or {},
    )
    logger.debug(f'Run tests result: {result}')

    reward_result = await runtime.run_command(command='cat /logs/verifier/reward.txt', timeout=120)
    logger.debug(f'Read reward result: {reward_result}')
    if reward_result.stdout:
        rewards = float(reward_result.stdout)
    else:
        raise RuntimeError(f'Read reward result: {reward_result}')

    ctrf_result = await runtime.run_command(command='cat /logs/verifier/ctrf.json', timeout=120)
    logger.debug(f'Read ctrf result: {ctrf_result}')
    try:
        ctrf_data = json.loads(ctrf_result.stdout) if ctrf_result.stdout else {}
    except json.JSONDecodeError:
        logger.warning(f'无法解析 ctrf.json: {ctrf_result.stdout[:200] if ctrf_result.stdout else "empty"}')
        ctrf_data = {}

    return rewards, ctrf_data


async def _run_oracle(task_input: SkillsTaskInput) -> None:
    ctx = require_context()
    runtime = await ctx.get_runtime()

    await runtime.run_command(command='mkdir -p /solution', timeout=60)

    file_items = read_local_directory_to_file_items(
        local_dir=Path(task_input.task_dir) / 'solution', container_base_path='/solution'
    )
    logger.debug(f'Uploading {len(file_items)} solution files to runtime...')
    await runtime.upload_file(files=file_items, overwrite=True)

    # 上传 solution 配置的文件
    await _upload_task_files(task_input.solution_files, task_input.task_dir)

    await runtime.run_command(command='mkdir -p /logs/agent', timeout=60)
    command = f'bash {shlex.quote("/solution/solve.sh")}'
    result = await runtime.run_command(
        command=command, timeout=task_input.agent_timeout, extra_env=task_input.solution_env or {}
    )
    logger.debug(f'Run oracle result: {result}')
    if result.return_code != 0:
        raise RuntimeError(f'Oracle solution exited with result: {result}')


@task_registry.register('skills_task')
async def skills_task(task_input: SkillsTaskInput) -> SkillsTaskOutput:
    """SkillsBench task that runs the agent against a skills-equipped environment."""
    logger.debug(f'Running skills task with message: {task_input.message[:80]}...')

    ctx = require_context()
    if not ctx.runtime_provider:
        raise RuntimeError('Skills task requires a runtime provider to be configured')

    runtime = await ctx.get_runtime()
    await runtime.run_command(command='mkdir -p /app', timeout=60)

    if task_input.run_oracle:
        await _run_oracle(task_input)
    else:
        # agent 模式下也需要上传 solution_files（如 Google auth 文件）
        await _upload_task_files(task_input.solution_files, task_input.task_dir)

        try:
            await asyncio.wait_for(
                task_input.run_agent(user_prompt=task_input.message),
                timeout=task_input.agent_timeout,
            )
        except asyncio.TimeoutError:
            logger.warning(
                f'Agent execution timed out after {task_input.agent_timeout}s, continuing with test execution'
            )

    reward, ctrf_data = await _run_tests(task_input)

    score = reward == 1.0
    return SkillsTaskOutput(
        score=score,
        test_results=ctrf_data,
    )
