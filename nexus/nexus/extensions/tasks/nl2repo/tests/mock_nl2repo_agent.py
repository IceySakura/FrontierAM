"""Mock NL2Repo agent — 直接 clone gold_git_repo 作为正确答案，跳过模型调用。"""

import logging

from nexus.core import agent_registry, require_context
from nexus.extensions.agents.codeact.agent import CodeActAgentInput, CodeActAgentOutput, ExecutionStatus


logger = logging.getLogger(__name__)


def _clone_gold_repo(gold_git_repo: str, workspace_dir: str, gold_git_commit: str = '') -> dict:
    """在 Runtime 中 clone gold_git_repo 到 workspace 目录。

    通过 invoke 在 Runtime 中执行。
    如果提供了 gold_git_commit，则 checkout 到指定 commit 以确保版本一致。
    """
    import os
    import shutil
    import subprocess

    tmp_dir = '/tmp/gold_repo'
    if os.path.exists(tmp_dir):
        shutil.rmtree(tmp_dir)

    # 如果有指定 commit，需要完整 clone（不能用 --depth=1）
    clone_cmd = ['git', 'clone', gold_git_repo, tmp_dir]
    if not gold_git_commit:
        clone_cmd = ['git', 'clone', '--depth=1', gold_git_repo, tmp_dir]

    result = subprocess.run(
        clone_cmd,
        capture_output=True,
        text=True,
        timeout=600,
    )
    if result.returncode != 0:
        return {
            'success': False,
            'error': f'git clone failed: {result.stderr}',
        }

    # 如果指定了 commit，checkout 到该版本
    checked_out_commit = ''
    if gold_git_commit:
        result = subprocess.run(
            ['git', 'checkout', gold_git_commit],
            capture_output=True,
            text=True,
            cwd=tmp_dir,
            timeout=60,
        )
        if result.returncode != 0:
            return {
                'success': False,
                'error': f'git checkout {gold_git_commit} failed: {result.stderr}',
            }
        checked_out_commit = gold_git_commit

    os.makedirs(workspace_dir, exist_ok=True)

    for item in os.listdir(tmp_dir):
        if item == '.git':
            continue
        src = os.path.join(tmp_dir, item)
        dst = os.path.join(workspace_dir, item)
        if os.path.exists(dst):
            if os.path.isdir(dst):
                shutil.rmtree(dst)
            else:
                os.remove(dst)
        shutil.move(src, dst)

    shutil.rmtree(tmp_dir, ignore_errors=True)

    files = os.listdir(workspace_dir)
    return {
        'success': True,
        'files': files[:50],
        'total_files': len(files),
        'checked_out_commit': checked_out_commit,
    }


@agent_registry.register('mock_nl2repo_agent')
async def run_mock_nl2repo_agent(agent_input: CodeActAgentInput) -> CodeActAgentOutput:
    """直接 clone gold_git_repo 到 workspace，模拟完美的 agent 输出。"""
    del agent_input  # mock agent 不需要 agent_input

    ctx = require_context()
    runtime = await ctx.get_runtime()

    gold_git_repo = ctx.task_config.get('gold_git_repo', '')
    gold_git_commit = ctx.task_config.get('gold_git_commit', '')
    workspace_dir = ctx.task_config.get('workdir', '/workspace')

    if not gold_git_repo:
        logger.error('gold_git_repo 未配置，无法 mock')
        return CodeActAgentOutput(execution_status=ExecutionStatus.ERROR)

    logger.info(
        'Mock NL2Repo Agent: clone %s -> %s (commit=%s)',
        gold_git_repo,
        workspace_dir,
        gold_git_commit or 'HEAD',
    )

    if gold_git_commit:
        result = await runtime.invoke(
            _clone_gold_repo,
            gold_git_repo,
            workspace_dir,
            gold_git_commit,
        )
    else:
        result = await runtime.invoke(
            _clone_gold_repo,
            gold_git_repo,
            workspace_dir,
        )

    logger.info('Clone 结果: %s', result)

    if not result or not result.get('success'):
        error_msg = result.get('error', 'unknown error') if result else 'invoke returned None'
        logger.error('Mock agent clone 失败: %s', error_msg)
        return CodeActAgentOutput(execution_status=ExecutionStatus.ERROR)

    logger.info(
        'Mock agent 成功: workspace 包含 %d 个文件/目录',
        result.get('total_files', 0),
    )
    return CodeActAgentOutput(execution_status=ExecutionStatus.FINISHED)
