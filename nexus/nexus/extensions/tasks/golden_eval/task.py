"""golden_eval：多轮人工交互评测任务，会话结束时归档产物。

用于盲测竞技场：评测员把同一句开场 prompt 发给多个 run，分别观察各自沙箱里的
产物渲染，再针对性地给每个 run 发后续引导 prompt，最后保存最终产物与轨迹。

轮次由谁驱动，这里不做决定。任务会传一个基于 user_inputs 的脚本化输入源，
但 runner 合并参数时 agent_input_overrides 排在最后，live runner 正是用它注入
自己的回调，所以有人接管时人优先、无人接管时走脚本，不需要在这里分支。

代价是任务观察不到单轮边界，每轮快照只能由调用方在 run 处于 idle 时经运行时
代理自己抓。任务只负责最后那一次归档。
"""

import contextlib
import logging
import posixpath

from typing import Any, Literal

from pydantic import BaseModel

from nexus.core import (
    TaskRunInput,
    TaskRunOutput,
    WithAgentRunner,
    require_context,
    task_registry,
)
from nexus.trace import get_current_span_id, update_span
from nexus.trace.phases import phase
from nexus.utils.timeout import async_timeout


logger = logging.getLogger(__name__)

# 依赖目录与构建产物不进归档：它们体积大、可重建，且会把 base64 撑爆
DEFAULT_ARCHIVE_EXCLUDE = [
    '.git',
    'node_modules',
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
    '.venv',
    'venv',
]


class InputFile(BaseModel):
    """评测员随开场 prompt 一起提供的素材。"""

    path: str
    """相对路径按 working_dir 解析，也可直接给绝对路径。"""

    content: str
    encoding: Literal['utf8', 'base64'] = 'utf8'
    """文本用 utf8，图片等二进制先 base64。"""


class GoldenEvalInput(TaskRunInput, WithAgentRunner):
    prompt: str = ''
    """开场 prompt。留空则改由 agent_config.user_prompt 提供。"""

    working_dir: str = '/workspace'
    runtime_name: str = 'main'

    input_files: list[InputFile] = []
    """开场素材。在 agent 启动前写入沙箱，所以第一轮 prompt 就能引用它们。
    这些文件同样会进最终归档，不想要就把所在目录加进 archive_exclude。"""

    user_inputs: list[str] = []
    """脚本化的后续轮次。调用方注入自己的输入源时，这里会被覆盖。"""

    archive_exclude: list[str] = DEFAULT_ARCHIVE_EXCLUDE
    max_archive_mb: int = 64
    """超过此大小只回传文件清单，避免 RPC 塞入过大的 base64。"""

    collect_timeout: int = 300
    release_runtime: bool = False
    """默认不提前释放：ExecutionContext 退出时会先 drain_trace 再释放，
    提前释放会让这次 flush 打在已销毁的沙箱上拿到 502。"""


class GoldenEvalOutput(TaskRunOutput):
    completed: bool
    archive_b64: str = ''
    file_list: list[str] = []
    archive_bytes: int = 0
    archive_skipped: bool = False
    turns: int = 0
    ended_by_user: bool = False
    """True 表示由空 turn 正常收尾，False 多半是撞上了 max_turns。"""
    reached_max_turns: bool = False
    session_id: str = ''
    error: str | None = None
    termination_reason: str | None = None


def _tar_working_dir(
    working_dir: str = '/workspace',
    exclude_dirs: list[str] | None = None,
    max_bytes: int = 64 * 1024 * 1024,
) -> tuple[str, list[str], int, bool]:
    """在沙箱内执行：把工作目录打成 tar.gz 并 base64 返回。

    通过 runtime.invoke 调用，传的是本函数的导入路径而非函数体，
    所以沙箱侧必须能 import 到本模块（依赖 worker_venv 上传的 wheel）。
    """
    import base64
    import io
    import os
    import tarfile

    excluded = set(exclude_dirs or [])
    file_list: list[str] = []
    buf = io.BytesIO()

    with tarfile.open(fileobj=buf, mode='w:gz') as tar:
        for root, dirs, files in os.walk(working_dir):
            # 原地改 dirs 才能让 os.walk 跳过整棵子树
            dirs[:] = [d for d in dirs if d not in excluded]
            for name in files:
                full = os.path.join(root, name)
                rel = os.path.relpath(full, working_dir)
                try:
                    tar.add(full, arcname=rel)
                    file_list.append(rel)
                except Exception:
                    # 单个文件失败（软链断裂、权限等）不该让整次归档作废
                    pass

    raw = buf.getvalue()
    if len(raw) > max_bytes:
        return '', file_list, len(raw), True
    return base64.b64encode(raw).decode(), file_list, len(raw), False


@task_registry.register('golden_eval')
async def run_golden_eval(task_input: GoldenEvalInput) -> GoldenEvalOutput:
    ctx = require_context()
    if not ctx.runtime_provider:
        raise RuntimeError('golden_eval 需要 runtime provider。')

    working_dir = task_input.working_dir
    pending = list(task_input.user_inputs)

    async def scripted_user_input(_request: Any) -> str | None:
        """脚本化兜底：返回 None 即结束多轮循环。被调用方注入的回调覆盖。"""
        return pending.pop(0) if pending else None

    try:
        phase('golden_eval.init')
        with phase('golden_eval.init.get_runtime'):
            runtime = await ctx.get_runtime(task_input.runtime_name)

        # 把沙箱 id 挂到当前 span，便于事后按 trace 回查是哪个沙箱
        span_id = get_current_span_id()
        runtime_id = ctx.get_runtime_id(task_input.runtime_name)
        if span_id and runtime_id:
            update_span(span_id, attributes={'hytuner.runtime_id': runtime_id})

        ctx.task_config['working_dir'] = working_dir
        ctx.tool_state.setdefault('working_dir', working_dir)

        # 素材必须赶在 agent 启动前落盘，否则第一轮 prompt 引用不到
        if task_input.input_files:
            with phase('golden_eval.upload'):
                from nexus.runtime.executors.models.file import FileItem

                items = [
                    FileItem(
                        path=f.path if posixpath.isabs(f.path) else posixpath.join(working_dir, f.path),
                        content=f.content,
                        encoding=f.encoding,
                    )
                    for f in task_input.input_files
                ]
                result = await runtime.upload_file(files=items, overwrite=True)
                failed = [r.path for r in result.results if not r.success]
                if failed:
                    raise RuntimeError(f'开场素材上传失败：{failed}')
                logger.debug('已写入 %d 个开场素材', len(items))

        phase('golden_eval.agent')
        agent_kwargs: dict[str, Any] = {
            'working_dir': working_dir,
            'get_user_input': scripted_user_input,
        }
        # prompt 为空时不要传，否则会把 agent_config.user_prompt 覆盖成空串
        if task_input.prompt:
            agent_kwargs['user_prompt'] = task_input.prompt
        # 这一行内部就是整个多轮循环，直到输入源返回 None 或撞上 max_turns 才返回
        agent_output = await task_input.run_agent(**agent_kwargs)

        turns = int(getattr(agent_output, 'turns', 0) or 0)
        logger.info('golden_eval 完成 %d 轮对话', turns)

        with phase('golden_eval.collect'):
            async with async_timeout(task_input.collect_timeout):
                archive_b64, file_list, archive_bytes, skipped = await runtime.invoke(
                    _tar_working_dir,
                    working_dir,
                    task_input.archive_exclude,
                    task_input.max_archive_mb * 1024 * 1024,
                )

        if skipped:
            logger.warning(
                '%s 归档 %.1f MB，超过上限 %d MB，只回传文件清单',
                working_dir,
                archive_bytes / 1024 / 1024,
                task_input.max_archive_mb,
            )
        else:
            logger.info('归档 %d 个文件，共 %.1f KB', len(file_list), archive_bytes / 1024)

        if task_input.release_runtime:
            await ctx.runtime_provider.release(name=task_input.runtime_name)

        phase()
        return GoldenEvalOutput(
            completed=True,
            archive_b64=archive_b64,
            file_list=file_list,
            archive_bytes=archive_bytes,
            archive_skipped=skipped,
            turns=turns,
            ended_by_user=bool(getattr(agent_output, 'ended_by_user', False)),
            reached_max_turns=bool(getattr(agent_output, 'reached_max_turns', False)),
            session_id=str(getattr(agent_output, 'session_id', '') or ''),
        )

    except Exception as exc:
        # 异常一律转成 completed=False 的正常返回，避免一条失败拖垮整批；
        # 代价是调用方不能看批次汇总，必须查 completed 字段
        if isinstance(exc, TimeoutError):
            term_reason = 'TASK_TIMEOUT'
        elif isinstance(exc, RuntimeError):
            term_reason = 'INITIALIZATION_ERROR'
        else:
            term_reason = 'UNHANDLED_ERROR'

        error_msg = f'{type(exc).__name__}: {exc}'
        logger.error('golden_eval 失败：%s', error_msg)

        span_id = get_current_span_id()
        if span_id:
            update_span(span_id, attributes={'hytuner.termination_reason': term_reason})

        with contextlib.suppress(Exception):
            if task_input.release_runtime:
                await ctx.runtime_provider.release(name=task_input.runtime_name)

        return GoldenEvalOutput(
            completed=False,
            error=error_msg,
            termination_reason=term_reason,
        )


__all__ = [
    'GoldenEvalInput',
    'GoldenEvalOutput',
    'run_golden_eval',
]
