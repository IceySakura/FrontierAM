"""Pytest judge: run a shell/pytest script inside the eval runtime and parse results."""

from __future__ import annotations

import logging

from typing import TYPE_CHECKING, Any, Literal

from nexus.extensions.tasks.artifact.judges.base import BaseJudge, JudgeConfig, JudgeResult


if TYPE_CHECKING:
    from nexus.core.openai_client import OpenAICompatibleClient


logger = logging.getLogger(__name__)

# Default score when all tests pass
_SCORE_PASS = 1.0
_SCORE_FAIL = 0.0


class PytestJudgeConfig(JudgeConfig):
    """Config for the pytest judge."""

    name: Literal['pytest'] = 'pytest'

    eval_script: str = ''
    """Shell/pytest command to run inside the runtime.

    Example:
        'cd /testbed && python -m pytest tests/ -v --tb=short'
    """

    working_dir: str = '/testbed'
    """Working directory for the eval script."""

    timeout: int = 1800
    """Timeout for the eval script in seconds."""

    partial_scoring: bool = False
    """If True, score = passed / (passed + failed) instead of binary 0/1."""


def _run_pytest_script(
    eval_script: str,
    working_dir: str,
    timeout: int,
) -> dict[str, Any]:
    """Execute eval script inside the runtime and return raw results.

    Executed via runtime.invoke inside the sandbox.
    Must be self-contained with no nexus imports.
    """
    import os
    import subprocess

    from pathlib import Path

    os_envs = {k: v for k, v in os.environ.items() if k != 'LD_LIBRARY_PATH'}
    os_envs['BASH_ENV'] = '~/.bashrc'

    nix_bin = '/nix/runtime/nix-env/bin'
    current_path = os_envs.get('PATH', '')
    if nix_bin not in current_path:
        os_envs['PATH'] = f'{nix_bin}:{current_path}' if current_path else nix_bin

    # Find bash
    executable_bash = '/bin/bash'
    for candidate in ['/nix/runtime/nix-env/bin/bash', '/bin/bash']:
        if Path(candidate).exists():
            executable_bash = candidate
            break

    script_path = Path('/tmp/artifact_eval.sh')
    script_path.write_text(eval_script)

    try:
        proc = subprocess.run(
            [executable_bash, str(script_path)],
            cwd=working_dir,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            encoding='utf-8',
            errors='replace',
            timeout=timeout,
        )
        return {
            'completed': True,
            'return_code': proc.returncode,
            'output': proc.stdout or '',
        }
    except subprocess.TimeoutExpired:
        return {
            'completed': False,
            'return_code': -1,
            'output': f'Eval script timed out after {timeout}s',
            'error': 'timeout',
        }
    except Exception as e:
        return {
            'completed': False,
            'return_code': -1,
            'output': str(e),
            'error': str(e),
        }


def _parse_pytest_output(output: str) -> tuple[int, int]:
    """Parse pytest stdout to extract passed/failed counts.

    Returns:
        (passed, failed) counts.  Falls back to (0, 1) if unparseable.
    """
    import re

    # pytest summary line: "5 passed, 2 failed, 1 error in 3.14s"
    pattern = re.compile(
        r'(?:(\d+) passed)?.*?(?:(\d+) failed)?.*?(?:(\d+) error)?',
        re.IGNORECASE,
    )
    for line in reversed(output.splitlines()):
        m = pattern.search(line)
        if m and (m.group(1) or m.group(2) or m.group(3)):
            passed = int(m.group(1) or 0)
            failed = int(m.group(2) or 0) + int(m.group(3) or 0)
            return passed, failed
    return 0, 0


class PytestJudge(BaseJudge):
    """Run a shell/pytest script in the runtime and grade the result."""

    async def run(
        self,
        config: JudgeConfig,
        image: str,
        patch_text: str,
        working_dir: str,
        openai_client: OpenAICompatibleClient | None = None,
        **kwargs: Any,
    ) -> JudgeResult:
        assert isinstance(config, PytestJudgeConfig)
        import uuid

        from nexus.core.context import require_context

        if not config.eval_script:
            return JudgeResult(
                judge_name='pytest',
                score=0.0,
                error='eval_script is empty',
            )

        ctx = require_context()
        runtime_name = f'eval_pytest_{uuid.uuid4().hex[:8]}'
        if image and ctx.runtime_provider:
            ctx.runtime_provider.configure(runtime_name, image=image)
        runtime = await ctx.get_runtime(runtime_name)

        try:
            if patch_text:
                await _apply_patch(runtime, patch_text, config.working_dir)

            result = await runtime.invoke(
                _run_pytest_script,
                config.eval_script,
                config.working_dir,
                config.timeout,
            )
        finally:
            try:
                if ctx.runtime_provider:
                    await ctx.runtime_provider.release(name=runtime_name)
            except Exception as rel_err:
                logger.warning('PytestJudge: failed to release runtime %s: %s', runtime_name, rel_err)

        output: str = result.get('output', '')
        return_code: int = result.get('return_code', -1)
        completed: bool = result.get('completed', False)
        error: str | None = result.get('error')

        if not completed or error == 'timeout':
            return JudgeResult(
                judge_name='pytest',
                score=0.0,
                passed=False,
                raw_output=output,
                error=error or 'script did not complete',
            )

        if not config.partial_scoring:
            # Binary: return_code 0 = pass
            passed = return_code == 0
            score = _SCORE_PASS if passed else _SCORE_FAIL
            evidence = f'exit_code={return_code}'
        else:
            # Partial: parse pytest pass/fail counts
            n_passed, n_failed = _parse_pytest_output(output)
            total = n_passed + n_failed
            score = (n_passed / total) if total > 0 else (_SCORE_PASS if return_code == 0 else _SCORE_FAIL)
            passed = n_failed == 0 and return_code == 0
            evidence = f'passed={n_passed}, failed={n_failed}, exit_code={return_code}'

        return JudgeResult(
            judge_name='pytest',
            score=round(score, 4),
            passed=passed,
            evidence=evidence,
            raw_output=output,
        )


def _apply(patch: str, repo_dir: str) -> dict[str, Any]:
    """Apply a git patch inside the runtime (module-level for invoke compatibility)."""
    import os
    import subprocess

    from pathlib import Path

    os_envs = {k: v for k, v in os.environ.items() if k != 'LD_LIBRARY_PATH'}
    nix_bin = '/nix/runtime/nix-env/bin'
    if nix_bin not in os_envs.get('PATH', ''):
        os_envs['PATH'] = f'{nix_bin}:{os_envs.get("PATH", "")}'

    patch_file = Path('/tmp/artifact_patch.diff')
    patch_file.write_text(patch)

    for cmd in ['git apply --verbose', 'git apply --verbose --reject', 'patch --batch --fuzz=5 -p1 -i']:
        full_cmd = f'{cmd} {patch_file}'
        proc = subprocess.run(
            full_cmd, shell=True, cwd=repo_dir, capture_output=True, text=True, timeout=120, env=os_envs
        )
        if proc.returncode == 0:
            return {'applied': True}
    return {'applied': False}


async def _apply_patch(runtime: Any, patch_text: str, working_dir: str) -> None:
    """Apply a git patch inside the runtime."""
    result = await runtime.invoke(
        _apply,
        patch_text,
        working_dir,
    )
    if not result.get('applied'):
        logger.warning('PytestJudge: failed to apply patch, continuing anyway')


__all__ = ['PytestJudgeConfig', 'PytestJudge']
