"""Agent judge: let claude-code execute test rubrics, then read reward.txt for each verdict.

Each rubric is an independent test prompt run in its own claude-code process inside its own
runtime container.  All rubrics run concurrently via asyncio.gather.

Score = number of passed rubrics / total rubrics.
"""

from __future__ import annotations

import logging
import uuid

from typing import TYPE_CHECKING, Any, Literal

from nexus.extensions.tasks.artifact.judges.base import BaseJudge, JudgeConfig, JudgeResult, RubricResultItem


if TYPE_CHECKING:
    from nexus.core.openai_client import OpenAICompatibleClient


logger = logging.getLogger(__name__)

DEFAULT_REWARD_FILE = '/tmp/reward.txt'

_REWARD_FILE_INSTRUCTION = """
严格按照上述 rubric 的要求进行验证，不要做超出 rubric 范围的额外检查。
如果 rubric 只要求检查文件内容，则只需读取文件验证，不要尝试安装依赖或构建项目。
如果 rubric 要求启动服务验证，而源代码问题导致服务无法启动，则直接判为0分。
验证通过记为1分，否则记为0分。
将最终得分写入 {reward_file}：
文件中只能包含单个字符 "1" 或 "0"，不要有其他内容。
"""


def _read_reward_file(reward_file: str = DEFAULT_REWARD_FILE) -> str:
    """Read reward.txt inside the runtime and return '1' or '0'.

    Executed via runtime.invoke (runs inside the sandbox).
    This function must be self-contained (no module-level references) so it
    can be invoked without requiring nexus to be installed in the runtime.
    Returns '0' on any error so the judge degrades gracefully.
    """
    from pathlib import Path

    try:
        content = Path(reward_file).read_text(encoding='utf-8').strip()
        if content not in ('0', '1'):
            return '0'
        return content
    except Exception:
        return '0'


class RubricResult(dict):
    """Per-rubric result stored in JudgeResult.evidence."""


class AgentJudgeConfig(JudgeConfig):
    """Config for the agent judge."""

    name: Literal['agent'] = 'agent'

    rubrics: list[str]
    """List of independent test prompts, one per functional requirement.

    Each rubric is run in a separate claude-code process inside its own runtime container.
    All rubrics execute concurrently.  Score = passed_count / total_count.

    Example:
        [
          "启动 /testbed 的服务，用 playwright 验证首页是否有搜索框。",
          "启动 /testbed 的服务，用 playwright 验证切换页面按钮是否正常。"
        ]
    """

    working_dir: str = '/testbed'
    """Working directory for claude-code."""

    timeout: float = 300
    """Timeout per rubric in seconds."""

    model: str | None = None
    """Claude model to use. Defaults to claude-code default."""

    subagent_model: str | None = None
    """Model for claude-code subagents."""

    binary_url: str | None = None
    """Direct URL to the claude-code binary (optional, bypasses GCS bucket)."""

    binary_checksum: str | None = None
    """Optional SHA-256 checksum for binary verification."""

    reward_file: str = DEFAULT_REWARD_FILE
    """Path inside the runtime where claude-code writes the verdict."""

    version: str = 'latest'
    """Claude Code version."""

    mcp_servers: list[dict[str, Any]] | None = None
    """MCP servers to register before running claude-code.

    Each dict should have:
    - name: str — server name (e.g. 'playwright')
    - command: str — the command to run (e.g. 'npx')
    - args: list[str] — command arguments (e.g. ['-y', '@playwright/mcp@latest'])

    Example:
        [{"name": "playwright", "command": "npx", "args": ["-y", "@playwright/mcp@latest"]}]
    """

    openai_client_name: str = 'main'
    """Name of the OpenAI client in ExecutionContext to use for this judge.

    Allows the judge to use a different model than the infer agent.
    The named client must be configured in openai_clients of the RunInput.

    Example:
        'openai_clients': {
            'main':  {..., 'model': 'claude-sonnet-4-6'},   # infer agent
            'judge': {..., 'model': 'claude-opus-4-6'},     # agent judge
        }
    Then set: {"name": "agent", "openai_client_name": "judge", ...}
    """


class AgentJudge(BaseJudge):
    """Run each rubric as an independent claude-code process in its own runtime; score = passed / total."""

    async def run(
        self,
        config: JudgeConfig,
        image: str,
        patch_text: str,
        working_dir: str,
        openai_client: OpenAICompatibleClient | None = None,
        **kwargs: Any,
    ) -> JudgeResult:
        assert isinstance(config, AgentJudgeConfig)
        import asyncio

        from nexus.core.context import require_context
        from nexus.runtime.server.services.llm_handler import LLMRequestHandler

        if not config.rubrics:
            return JudgeResult(
                judge_name='agent',
                score=0.0,
                error='rubrics list is empty',
            )

        # Resolve openai client: prefer named client from context, fall back to passed-in
        resolved_client = openai_client
        try:
            ctx = require_context()
            resolved_client = ctx.get_openai_client(config.openai_client_name)
        except Exception:
            pass

        if resolved_client is None:
            return JudgeResult(
                judge_name='agent',
                score=0.0,
                error='AgentJudge requires an openai_client but none was provided',
            )

        # Unique job ID so parallel judges don't collide on runtime names
        job_id = uuid.uuid4().hex[:8]

        async def _run_one_rubric(idx: int, rubric: str) -> dict[str, Any]:
            """Run a single rubric in its own isolated runtime container."""
            from nexus.trace import processor_context
            from nexus.trace.processors.chat_tracker import ChatTrackerProcessor

            rubric_label = f'rubric[{idx}]'
            runtime_name = f'eval_rubric_{job_id}_{idx}'
            logger.debug('AgentJudge: acquiring runtime %s for %s', runtime_name, rubric_label)

            runtime = None
            cc_output = ''
            cc_stderr = ''
            run_error: str | None = None
            rubric_trajectories: dict[str, Any] = {}

            rubric_chat_tracker = ChatTrackerProcessor()
            with processor_context([rubric_chat_tracker], finalize_on_exit=True):
                try:
                    ctx_inner = require_context()

                    # Configure and acquire a dedicated runtime for this rubric
                    if image and ctx_inner.runtime_provider:
                        ctx_inner.runtime_provider.configure(runtime_name, image=image)
                    runtime = await ctx_inner.get_runtime(runtime_name)

                    # Apply patch if provided
                    if patch_text:
                        await _apply_patch(runtime, patch_text, config.working_dir)

                    # Establish Socket.IO + LLM proxy for this runtime
                    await runtime.ws_connect()
                    handler = LLMRequestHandler(runtime=runtime, openai_client=resolved_client)
                    handler.register()

                    try:
                        # Clean up any stale reward.txt
                        await runtime.call_executor(
                            'run_command',
                            {
                                'command': f'rm -f {config.reward_file}',
                                'timeout': 10,
                            },
                        )

                        # Build full prompt with reward-file instruction
                        instruction = _REWARD_FILE_INSTRUCTION.format(reward_file=config.reward_file)
                        full_prompt = rubric.rstrip() + '\n' + instruction

                        try:
                            from nexus.extensions.agents.claude_code.runtime_setup import setup_and_run_claude_code

                            result = await runtime.invoke(
                                setup_and_run_claude_code,
                                task=full_prompt,
                                working_dir=config.working_dir,
                                timeout=config.timeout,
                                version=config.version,
                                binary_url=config.binary_url,
                                binary_checksum=config.binary_checksum,
                                mcp_servers=config.mcp_servers,
                            )
                            cc_output = result.get('output', '')
                            cc_stderr = result.get('stderr', '')
                            if not result.get('success', False):
                                run_error = f'claude-code exit_code={result.get("exit_code", -1)}'
                        except Exception as e:
                            run_error = f'{type(e).__name__}: {e}'
                            logger.exception('AgentJudge: %s raised an error', rubric_label)

                        # Read verdict via shell command
                        try:
                            read_result = await runtime.call_executor(
                                'run_command',
                                {
                                    'command': f'cat {config.reward_file} 2>/dev/null || echo "0"',
                                    'timeout': 10,
                                },
                            )
                            raw = (read_result.get('stdout', '') or '').strip()
                            verdict = raw if raw in ('0', '1') else '0'
                        except Exception:
                            verdict = '0'

                    finally:
                        handler.unregister()

                except Exception as e:
                    run_error = f'{type(e).__name__}: {e}'
                    verdict = '0'
                    logger.exception('AgentJudge: failed to run %s', rubric_label)
                finally:
                    # Always release the rubric's dedicated runtime
                    if runtime is not None:
                        try:
                            provider = require_context().runtime_provider
                            if provider:
                                await provider.release(name=runtime_name)
                            logger.debug('AgentJudge: released runtime %s', runtime_name)
                        except Exception as rel_err:
                            logger.warning('AgentJudge: failed to release runtime %s: %s', runtime_name, rel_err)

            rubric_trajectories = rubric_chat_tracker.get_result()

            passed = verdict == '1'
            logger.info('AgentJudge: %s → %s', rubric_label, 'PASS' if passed else 'FAIL')
            return {
                'index': idx,
                'rubric': rubric,
                'passed': passed,
                'verdict': verdict,
                'run_error': run_error,
                'stdout': cc_output,
                'stderr': cc_stderr,
                'trajectories': rubric_trajectories,
            }

        # Run all rubrics concurrently
        rubric_results: list[dict[str, Any]] = await asyncio.gather(*[
            _run_one_rubric(idx, rubric) for idx, rubric in enumerate(config.rubrics)
        ])

        passed_count = sum(1 for r in rubric_results if r['passed'])
        total = len(config.rubrics)
        score = round(passed_count / total, 4)
        evidence = f'passed={passed_count}/{total}'

        # Summarise per-rubric results for easy inspection
        raw_output_lines = []
        for r in rubric_results:
            status = 'PASS' if r['passed'] else 'FAIL'
            raw_output_lines.append(f'[{r["index"]}] {status} | {r["rubric"][:80]}')
            if r['run_error']:
                raw_output_lines.append(f'    error: {r["run_error"]}')

        # Build structured per-rubric results
        structured_rubric_results = [
            RubricResultItem(
                index=r['index'],
                rubric=r['rubric'],
                passed=r['passed'],
                verdict=r.get('verdict', ''),
                error=r.get('run_error'),
                trajectories=r.get('trajectories', {}),
            )
            for r in rubric_results
        ]

        return JudgeResult(
            judge_name='agent',
            score=score,
            passed=passed_count == total,
            evidence=evidence,
            raw_output='\n'.join(raw_output_lines),
            rubric_results=structured_rubric_results,
            # Store full per-rubric detail in error field only on partial failure
            error=None if passed_count == total else f'{total - passed_count} rubric(s) failed',
        )


async def _apply_patch(runtime: Any, patch_text: str, working_dir: str) -> None:
    """Apply a git patch inside a rubric runtime by uploading the patch file directly."""
    from nexus.runtime.executors.file import FileItem

    # Upload patch file directly to runtime (no shell ARG_MAX limit)
    await runtime.upload_file([FileItem(path='/tmp/artifact_patch.diff', content=patch_text)])

    # Try applying patch with multiple strategies (same order as SWE eval)
    for cmd in [
        f'cd {working_dir} && git apply --verbose /tmp/artifact_patch.diff',
        f'cd {working_dir} && git apply --verbose --reject /tmp/artifact_patch.diff',
        f'cd {working_dir} && patch --batch --fuzz=5 -p1 -i /tmp/artifact_patch.diff',
    ]:
        try:
            result = await runtime.call_executor(
                'run_command',
                {
                    'command': cmd,
                    'timeout': 120,
                },
            )
            if result.get('return_code', -1) == 0:
                return
        except Exception:
            continue
    logger.warning('AgentJudge: failed to apply patch in rubric runtime, continuing anyway')


__all__ = ['AgentJudgeConfig', 'AgentJudge']
