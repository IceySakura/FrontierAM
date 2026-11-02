"""``bash`` — general-purpose main-image shell access in the working directory.

Deliberately not framed as a calculation helper. In this chain bash is the
agent's primary instrument for the meta-first workflow the system prompt asks
for (``ls -la``, ``wc -l``, ``wc -c``, ``file``, ``head -c``) as well as for
anything the other six tools do not cover: format conversion, batch processing,
moving files around.

cwd is the rollout log directory, so downloads are reachable at
``./workspace/<name>`` and anything the command writes lands alongside the
trajectory.
"""

from __future__ import annotations

import subprocess
import threading

from ..processes import run_cancellable, run_command_group
from ..subprocess_env import main_image_env, sandbox_command, subprocess_identity_args


_DEFAULT_TIMEOUT = 120
_DEFAULT_MAX_OUTPUT_CHARS = 200_000


def _run_command_blocking(
    command: str,
    timeout: int,
    max_output_chars: int,
    cwd: str | None = None,
    *,
    cancelled: threading.Event | None = None,
) -> str:
    try:
        result = run_command_group(
            sandbox_command(['/bin/sh', '-c', command]) if subprocess_identity_args() else command,
            shell=not bool(subprocess_identity_args()),
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            cancelled=cancelled if cancelled is not None else threading.Event(),
            text=True,
            encoding='utf-8',
            errors='replace',
            timeout=timeout,
            cwd=cwd,
            # Inheriting os.environ would put the endpoint credentials one `env`
            # call away from the model, and into the trace with them.
            env=main_image_env(),
            **subprocess_identity_args(),
        )
    except subprocess.TimeoutExpired:
        return f'Error: Command timed out after {timeout}s'
    except Exception as exc:  # noqa: BLE001
        return f'Error: {exc}'

    parts: list[str] = []
    if result.stdout:
        stdout = result.stdout.rstrip()
        if len(stdout) > max_output_chars:
            full_len = len(stdout)
            stdout = stdout[:max_output_chars] + (
                f'\n\n... (output truncated at {max_output_chars} chars; full '
                f'output was {full_len} chars. Do not cat whole documents — use '
                f'read_lines for line ranges, or write the result to a file.)'
            )
        parts.append(stdout)
    if result.stderr:
        parts.append(f'[stderr]\n{result.stderr.rstrip()}')
    if result.returncode != 0:
        parts.append(f'[exit code: {result.returncode}]')
    return '\n'.join(parts) if parts else '(no output)'


async def bash(command: str, *, cwd: str | None = None, timeout: int = _DEFAULT_TIMEOUT) -> str:
    if not command or not command.strip():
        return "Error: 'command' argument is required and must not be empty. Provide a valid bash command to execute."
    return await run_cancellable(_run_command_blocking, command, timeout, _DEFAULT_MAX_OUTPUT_CHARS, cwd)
