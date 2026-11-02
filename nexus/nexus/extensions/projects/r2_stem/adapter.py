"""Agent-bound post-execution adapters for R2 STEM tasks."""

import logging
import shlex

from typing import Any

from nexus.runtime.executors.models.file import FileItem


logger = logging.getLogger(__name__)

TESTS_DIR = '/tests'
VERIFIER_LOG_DIR = '/logs/verifier'
_ANSWER_CAPTURE_PATH = f'{VERIFIER_LOG_DIR}/answer.txt'
_ANSWER_CAPTURE_SCRIPT = """#!/usr/bin/env bash
set -u

answer_path="${1:-}"
mkdir -p /logs/verifier

if [[ -n "$answer_path" && -f "$answer_path" ]]; then
    cat -- "$answer_path" > /logs/verifier/answer.txt
else
    : > /logs/verifier/answer.txt
fi
"""


def _clear_directory_contents_command(path: str, *, variable: str) -> str:
    """Build a fail-closed command that preserves a protected top-level directory."""
    quoted_path = shlex.quote(path)
    return (
        f'if [ -L {quoted_path} ] || {{ [ -e {quoted_path} ] && [ ! -d {quoted_path} ]; }}; then '
        f'echo "Unsafe verifier workspace type" >&2; exit 1; fi; '
        f'if [ -d {quoted_path} ]; then '
        f'if ! find {quoted_path} -mindepth 1 -delete >/dev/null 2>&1; then '
        f'echo "Failed to clear verifier workspace" >&2; exit 1; fi; '
        f'{variable}="$(find {quoted_path} -mindepth 1 -print -quit 2>/dev/null)" || '
        f'{{ echo "Failed to inspect verifier workspace" >&2; exit 1; }}; '
        f'if [ -n "${{{variable}}}" ]; then echo "Verifier workspace is not empty" >&2; exit 1; fi; '
        f'fi'
    )


def _clear_verifier_workspace_command(*, recreate: bool) -> str:
    commands = [
        _clear_directory_contents_command(TESTS_DIR, variable='tests_remaining'),
        _clear_directory_contents_command(VERIFIER_LOG_DIR, variable='logs_remaining'),
    ]
    if recreate:
        commands.append(f'mkdir -p {TESTS_DIR} {VERIFIER_LOG_DIR}')
        commands.append(
            f'[ -d {TESTS_DIR} ] && [ -w {TESTS_DIR} ] && '
            f'[ -d {VERIFIER_LOG_DIR} ] && [ -w {VERIFIER_LOG_DIR} ] || '
            f'{{ echo "Verifier workspace is not writable" >&2; exit 1; }}'
        )
    return '; '.join(commands)


def command_error(action: str, result: Any) -> RuntimeError:
    return RuntimeError(
        f'{action} failed (return_code={getattr(result, "return_code", None)}): '
        f'{getattr(result, "stderr", "") or getattr(result, "stdout", "")}'
    )


async def hide_verifier_from_agent(runtime: Any) -> None:
    """Remove verifier contents before the agent starts, preserving protected roots."""
    result = await runtime.run_command(
        command=_clear_verifier_workspace_command(recreate=False),
        timeout=120,
    )
    if result.return_code != 0:
        raise command_error('Clear verifier artifacts before agent execution', result)


async def reset_verifier_workspace(runtime: Any) -> None:
    """Create an empty verifier workspace after the agent has stopped."""
    result = await runtime.run_command(
        command=_clear_verifier_workspace_command(recreate=True),
        timeout=120,
    )
    if result.return_code != 0:
        raise command_error('Reset verifier workspace', result)


async def capture_answer_file(
    *,
    runtime: Any,
    answer_path: str,
    test_timeout: int = 120,
) -> str:
    """Inject a post-agent ``test.sh`` and return a Terminus file answer."""
    await reset_verifier_workspace(runtime)
    upload_result = await runtime.upload_file(
        files=[FileItem(path=f'{TESTS_DIR}/test.sh', content=_ANSWER_CAPTURE_SCRIPT, encoding='utf8')],
        overwrite=True,
    )
    failed = [item for item in upload_result.results if not item.success]
    if failed:
        raise RuntimeError(f'Failed to upload answer-capture test.sh: {failed}')

    result = await runtime.run_command(
        command=f'bash {TESTS_DIR}/test.sh {shlex.quote(answer_path)}',
        timeout=test_timeout,
    )
    if result.return_code != 0:
        raise command_error('Run answer-capture test.sh', result)

    download = await runtime.download_file([FileItem(path=_ANSWER_CAPTURE_PATH, encoding='utf8')])
    if download.errors or not download.files:
        logger.info('Answer-capture output is unavailable: errors=%s', download.errors)
        return ''
    content = download.files[0].content
    return content.strip() if isinstance(content, str) else ''


__all__ = [
    'TESTS_DIR',
    'VERIFIER_LOG_DIR',
    'capture_answer_file',
    'command_error',
    'hide_verifier_from_agent',
    'reset_verifier_workspace',
]
