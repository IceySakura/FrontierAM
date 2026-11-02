"""NL2RepoBench task — Nexus 框架实现。

该任务实现 NL2RepoBench 的推理 + 评估流程：
1. 推理阶段：使用 CodeAct Agent 根据 start.md 生成完整代码仓库
2. 评估阶段：在干净的项目基础镜像中执行 pytest 并计算得分
"""

import contextlib
import logging

from pydantic import BaseModel

from nexus.core import (
    TaskRunInput,
    TaskRunOutput,
    WithAgentRunner,
    WithScore,
    require_context,
    task_registry,
)


logger = logging.getLogger(__name__)


# ============== 数据模型 ==============


class NL2RepoTaskInput(TaskRunInput, WithAgentRunner):
    """NL2Repo 任务输入。"""

    instance_id: str
    start_md_cos_key: str
    test_files: list[str]
    test_commands: list[str]
    test_case_count: int
    eval_image: str
    infer_runtime_name: str = 'main'
    eval_runtime_name: str = 'eval'
    infer_image: str = 'mirrors.tencent.com/hunyuan_yanguan/nl2repo-python:3.12.10-bookworm'
    eval_timeout_seconds: int = 1800
    max_iterations: int = 500
    dataset_namespace: str = 'swe-data'
    dataset_name: str = 'NL2RepoBench'
    workdir: str = '/workspace'


class PytestResults(BaseModel):
    """pytest 测试结果详情。"""

    passed: int = 0
    failed: int = 0
    errors: int = 0
    total: int = 0
    real_total: int = 0
    success_rate: float = 0.0


class NL2RepoTaskOutput(TaskRunOutput, WithScore):
    """NL2Repo 任务输出。"""

    completed: bool = False
    pytest_results: PytestResults | None = None
    local_zip_path: str | None = None
    error: str | None = None


# ============== 任务主入口 ==============


@task_registry.register('nl2repo_task')
async def run_nl2repo_task(task_input: NL2RepoTaskInput) -> NL2RepoTaskOutput:
    """运行单个 NL2RepoBench 推理 + 评估任务。"""
    from nexus.extensions.tasks.nl2repo.eval import run_eval_phase
    from nexus.extensions.tasks.nl2repo.infer import run_infer_phase

    ctx = require_context()
    if not ctx.runtime_provider:
        raise RuntimeError('NL2Repo task requires a runtime provider.')

    infer_runtime_name = task_input.infer_runtime_name
    eval_runtime_name = task_input.eval_runtime_name

    # ---- 推理阶段 ----
    local_zip_path: str | None = None
    try:
        local_zip_path = await run_infer_phase(
            task_input=task_input,
            ctx=ctx,
        )
    except Exception as e:
        error_msg = f'推理阶段异常: {type(e).__name__}: {e}'
        logger.error(error_msg)
        with contextlib.suppress(Exception):
            await ctx.runtime_provider.release(name=infer_runtime_name)
        raise RuntimeError(error_msg) from e

    # ---- 评估阶段 ----
    try:
        return await run_eval_phase(
            task_input=task_input,
            ctx=ctx,
            local_zip_path=local_zip_path,
        )
    except Exception as e:
        error_msg = f'评估阶段异常: {type(e).__name__}: {e}'
        logger.error(error_msg)
        with contextlib.suppress(Exception):
            await ctx.runtime_provider.release(name=eval_runtime_name)
        raise RuntimeError(error_msg) from e


__all__ = [
    'NL2RepoTaskInput',
    'NL2RepoTaskOutput',
    'PytestResults',
    'run_nl2repo_task',
]
