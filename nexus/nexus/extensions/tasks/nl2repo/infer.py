"""推理阶段逻辑 — 初始化 infer_runtime、运行 Agent、后处理、保存 zip 到 Host 本地。"""

from __future__ import annotations

import base64
import logging
import os

from typing import TYPE_CHECKING

from nexus.runtime.executors.file import FileItem


if TYPE_CHECKING:
    from nexus.core import ExecutionContext
    from nexus.extensions.tasks.nl2repo.task import NL2RepoTaskInput

logger = logging.getLogger(__name__)

# Agent 固定提示词
AGENT_PROMPT = (
    'According to the start.md in the workspace, implement the entire project as per the requirements '
    'specified in the document, ensuring that the final product can be directly run in the current directory. '
    'The running requirements should comply with the <API Usage Guide> section of the document. '
    'Please complete this task step by step.'
)


async def _download_start_md(task_input: NL2RepoTaskInput) -> bytes:
    """通过 DatasetClient 签名 URL 下载 start.md。"""
    import aiohttp

    from nexus.utils.legacy_dataset_client import DatasetClient

    client = DatasetClient()
    try:
        urls = await client.sign_urls(
            task_input.dataset_namespace,
            task_input.dataset_name,
            [task_input.start_md_cos_key],
        )
        signed_url = urls[task_input.start_md_cos_key]
        logger.debug('已获取 start.md 签名 URL, file=%s', task_input.start_md_cos_key)

        async with aiohttp.ClientSession() as session:
            async with session.get(signed_url) as resp:
                resp.raise_for_status()
                data = await resp.read()
                logger.debug('start.md 下载成功, size=%d bytes', len(data))
                return data
    finally:
        await client.close()


async def run_infer_phase(
    *,
    task_input: NL2RepoTaskInput,
    ctx: ExecutionContext,
) -> str:
    """执行推理阶段：初始化 Runtime → 下载并上传 start.md → 运行 Agent → 后处理 → 保存 zip 到 Host 本地。

    Args:
        task_input: 任务输入
        ctx: 执行上下文

    Returns:
        Host 本地的 zip 文件路径

    Raises:
        Exception: 任何阶段失败时抛出
    """
    from nexus.extensions.tasks.nl2repo.post_process import PACKAGE_FILES, post_process_and_zip

    if not ctx.runtime_provider:
        raise RuntimeError('NL2Repo task requires a runtime provider.')

    infer_runtime_name = task_input.infer_runtime_name
    workspace_dir = task_input.workdir

    # 1. 配置并获取 infer_runtime
    logger.debug('配置 infer_runtime, image=%s', task_input.infer_image)
    ctx.runtime_provider.configure(infer_runtime_name, image=task_input.infer_image)
    infer_runtime = await ctx.get_runtime(infer_runtime_name)
    logger.debug('infer_runtime 已就绪')

    # 2. 通过签名 URL 下载 start.md
    logger.debug('下载 start.md, file=%s', task_input.start_md_cos_key)
    start_md_bytes = await _download_start_md(task_input)
    start_md_content = base64.b64encode(start_md_bytes).decode('utf-8')

    # 3. 通过 upload_file 上传 start.md 到 Runtime
    start_md_path = f'{workspace_dir}/start.md'
    logger.debug('上传 start.md 到 Runtime: %s', start_md_path)
    upload_resp = await infer_runtime.upload_file(
        files=[FileItem(path=start_md_path, content=start_md_content, encoding='base64')],
    )
    for result in upload_resp.results:
        if not result.success:
            raise RuntimeError(f'上传 start.md 失败: {result.message}')
    logger.debug('start.md 上传成功')

    # 4. 运行 CodeAct Agent
    logger.debug('启动 Agent 推理, max_iterations=%d', task_input.max_iterations)
    agent_output = await task_input.run_agent(
        user_prompt=AGENT_PROMPT,
    )
    exit_status = getattr(agent_output, 'execution_status', None)
    logger.info('Agent 推理完成, exit_status=%s', exit_status)

    status_value = exit_status.value if exit_status is not None and hasattr(exit_status, 'value') else str(exit_status)
    if status_value == 'error':
        raise RuntimeError(f'Agent 推理失败, exit_status={status_value}')

    # 5. 在 infer_runtime 中执行后处理
    logger.debug('执行后处理: 删除包管理文件和测试文件, 打包 workspace')
    zip_path = await infer_runtime.invoke(
        post_process_and_zip,
        workspace_dir,
        task_input.test_files,
        PACKAGE_FILES,
    )
    logger.info('后处理完成, zip_path=%s', zip_path)

    # 6. 通过 download_file 将 zip 从 infer_runtime 下载到 Host 内存
    logger.debug('从 infer_runtime 下载 zip 文件')
    download_resp = await infer_runtime.download_file(
        files=[FileItem(path=zip_path, encoding='base64')],
    )
    if download_resp.errors:
        error_msgs = [f'{e.path}: {e.message}' for e in download_resp.errors]
        raise RuntimeError(f'下载 zip 文件失败: {"; ".join(error_msgs)}')
    if not download_resp.files:
        raise RuntimeError('下载 zip 文件失败: 返回结果为空')

    zip_base64 = download_resp.files[0].content
    if not zip_base64:
        raise RuntimeError('下载 zip 文件失败: 文件内容为空')
    zip_bytes = base64.b64decode(zip_base64)
    logger.debug('zip 文件下载成功, size=%d bytes', len(zip_bytes))

    # 7. 保存 zip 到 Host 本地（以 run_id 隔离，支持并发运行）
    run_id = ctx.run_id or task_input.instance_id
    local_dir = f'/tmp/nl2repo/{run_id}'
    os.makedirs(local_dir, exist_ok=True)
    local_zip_path = f'{local_dir}/workspace.zip'
    with open(local_zip_path, 'wb') as f:
        f.write(zip_bytes)
    logger.info('zip 已保存到本地: %s', local_zip_path)

    # 8. 释放 infer_runtime
    logger.debug('释放 infer_runtime')
    await ctx.runtime_provider.release(name=infer_runtime_name)

    return local_zip_path


__all__ = ['run_infer_phase']
