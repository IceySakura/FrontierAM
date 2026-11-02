"""评估阶段逻辑 — 初始化 eval_runtime、从 Host 本地加载 zip、执行 pytest、解析结果。"""

from __future__ import annotations

import base64
import contextlib
import logging
import os
import re
import shutil

from typing import TYPE_CHECKING

from nexus.extensions.tasks.nl2repo.task import NL2RepoTaskOutput, PytestResults
from nexus.runtime.executors.file import FileItem


if TYPE_CHECKING:
    from nexus.core import ExecutionContext
    from nexus.extensions.tasks.nl2repo.task import NL2RepoTaskInput

logger = logging.getLogger(__name__)


def parse_pytest_output(output: str, total_test_cases: int) -> PytestResults:
    """解析 pytest 输出，提取 passed、failed、errors 数量。

    Args:
        output: pytest 命令的完整输出
        total_test_cases: 测试用例总数（来自 JSONL 数据集）

    Returns:
        PytestResults 对象
    """
    passed = 0
    failed = 0
    errors = 0

    for line in output.split('\n'):
        passed_match = re.search(r'(\d+) passed', line)
        if passed_match:
            passed += int(passed_match.group(1))

        failed_match = re.search(r'(\d+) failed', line)
        if failed_match:
            failed += int(failed_match.group(1))

        error_match = re.search(r'(\d+) error', line)
        if error_match:
            errors += int(error_match.group(1))

    real_total = passed + failed + errors
    success_rate = 0.0
    if real_total > 0:
        success_rate = round(passed / real_total, 2)

    return PytestResults(
        passed=passed,
        failed=failed,
        errors=errors,
        total=total_test_cases,
        real_total=real_total,
        success_rate=success_rate,
    )


async def run_eval_phase(
    *,
    task_input: NL2RepoTaskInput,
    ctx: ExecutionContext,
    local_zip_path: str,
) -> NL2RepoTaskOutput:
    """执行评估阶段：初始化 eval_runtime → 从 Host 本地读取 zip → 解压 → 执行 pytest → 解析结果。

    Args:
        task_input: 任务输入
        ctx: 执行上下文
        local_zip_path: Host 本地的 zip 文件路径

    Returns:
        NL2RepoTaskOutput 任务输出

    Raises:
        Exception: 任何阶段失败时抛出
    """
    if not ctx.runtime_provider:
        raise RuntimeError('NL2Repo task requires a runtime provider.')

    eval_runtime_name = task_input.eval_runtime_name
    workspace_dir = task_input.workdir

    # 1. 配置并获取 eval_runtime（使用项目基础镜像）
    logger.debug('配置 eval_runtime, image=%s', task_input.eval_image)
    ctx.runtime_provider.configure(eval_runtime_name, image=task_input.eval_image)
    eval_runtime = await ctx.get_runtime(eval_runtime_name)
    logger.debug('eval_runtime 已就绪')

    # 2. 从 Host 本地读取 zip 文件
    logger.debug('从本地读取 zip, path=%s', local_zip_path)
    with open(local_zip_path, 'rb') as f:
        zip_bytes = f.read()
    zip_base64 = base64.b64encode(zip_bytes).decode('utf-8')
    logger.debug('zip 读取成功, size=%d bytes', len(zip_bytes))

    # 3. 通过 upload_file 上传 zip 到 eval_runtime
    zip_remote_path = '/tmp/workspace.zip'
    logger.debug('上传 zip 到 eval_runtime: %s', zip_remote_path)
    upload_resp = await eval_runtime.upload_file(
        files=[FileItem(path=zip_remote_path, content=zip_base64, encoding='base64')],
    )
    for result in upload_resp.results:
        if not result.success:
            raise RuntimeError(f'上传 zip 到 eval_runtime 失败: {result.message}')

    # 4. 使用 Python zipfile 解压 zip 到 workspace（不依赖系统 unzip 命令）
    from nexus.extensions.tasks.nl2repo.post_process import extract_zip

    logger.debug('解压 zip 到 %s', workspace_dir)
    await eval_runtime.invoke(
        extract_zip,
        zip_remote_path,
        workspace_dir,
    )

    # 5. 按顺序执行 test_commands，收集所有 pytest 命令的输出
    logger.debug('开始执行测试命令, 共 %d 条', len(task_input.test_commands))
    command_results: list[dict[str, str]] = []
    for i, cmd in enumerate(task_input.test_commands):
        env_cmd = f'export PYTHONPATH={workspace_dir}:$PYTHONPATH && {cmd}'
        logger.debug('执行命令 %d/%d: %s', i + 1, len(task_input.test_commands), cmd)
        exit_code = 'N/A'
        try:
            result = await eval_runtime.call_executor(
                'run_command',
                {
                    'command': env_cmd,
                    'cwd': workspace_dir,
                    'timeout': task_input.eval_timeout_seconds,
                },
            )
            if isinstance(result, dict):
                stdout = result.get('stdout', '')
                stderr = result.get('stderr', '')
                exit_code = result.get('exit_code', -1)
                output = f'{stdout}\n{stderr}'
                logger.info('命令执行完成, exit_code=%s', exit_code)
            else:
                output = str(result)
                logger.debug('命令执行完成')
        except Exception as e:
            logger.warning('命令执行异常: %s', e)
            output = str(e)
        command_results.append({'command': cmd, 'output': output})

    # 6. 解析 pytest 结果（仅解析命令中包含 pytest 的命令输出，与原始 NL2RepoBench 对齐）
    logger.debug('解析 pytest 结果')
    pytest_output_parts: list[str] = []
    for cr in command_results:
        if 'pytest' in cr['command'].lower():
            pytest_output_parts.append(cr['output'])
    combined_pytest_output = '\n'.join(pytest_output_parts) if pytest_output_parts else ''
    pytest_results = parse_pytest_output(combined_pytest_output, task_input.test_case_count)
    score = pytest_results.success_rate
    logger.info(
        'pytest 结果: passed=%d, failed=%d, errors=%d, total=%d, score=%.2f',
        pytest_results.passed,
        pytest_results.failed,
        pytest_results.errors,
        pytest_results.total,
        score,
    )

    # 7. 释放 eval_runtime
    logger.debug('释放 eval_runtime')
    with contextlib.suppress(Exception):
        await ctx.runtime_provider.release(name=eval_runtime_name)

    # 8. 清理 Host 本地 zip 文件
    local_dir = os.path.dirname(local_zip_path)
    if local_dir.startswith('/tmp/nl2repo/'):
        shutil.rmtree(local_dir, ignore_errors=True)
        logger.debug('已清理本地临时目录: %s', local_dir)

    return NL2RepoTaskOutput(
        score=score,
        completed=True,
        pytest_results=pytest_results,
        local_zip_path=local_zip_path,
    )


__all__ = ['parse_pytest_output', 'run_eval_phase']
