"""Sandbox-side execution for the SWE-Cosmos bugmaker pipeline.

This module is invoked via ``runtime.invoke`` inside the root SWE-Cosmos sandbox. It:

1. (optionally) refreshes the baked-in SWE-Cosmos checkout by overwrite-extracting a host-uploaded source tarball (``git_archive_path``).
2. Starts an in-sandbox OpenAI-compatible proxy bound to ``127.0.0.1`` so that the pipeline's ``litellm`` calls are forwarded back to the host LLM client.
3. Runs ``scripts/test_bugmaker_pipeline.sh`` as a subprocess, teeing combined stdout/stderr to a log file inside the sandbox.
4. Reads the pipeline's ``done`` marker and count files and returns them, plus the sandbox-side paths of the artifacts the host should download.

The whole run happens inside one long ``invoke`` job; nexus' runtime server keeps the sandbox's activity timestamp fresh for the duration via ``keep_activity_alive``, so the sandbox is not idle-reclaimed.
"""

from __future__ import annotations

import logging
import os
import shlex
import subprocess
import time

from pathlib import Path
from typing import Any


logger = logging.getLogger(__name__)

# SWE-Cosmos' harness (swecosmos/harness/validate.py) logs this when a repo's reference validation has zero PASSED tests and therefore every patch is skipped. The pipeline can still exit 0 in that case, but it produced nothing usable, so we treat it as a failure.
_REFERENCE_VALIDATION_FAILED_MARKER = (
    'No PASSED tests in reference validation, skipping all patches for this repository'
)


def _read_count(path: Path) -> int:
    """Read an integer count from *path*, returning 0 when missing/unparsable.

    The pipeline writes a single integer to each count file (unvalidated /
    validated / resolved / issued). A missing file simply means the pipeline
    exited before reaching that stage.
    """
    try:
        return int(path.read_text(encoding='utf-8').strip() or '0')
    except (OSError, ValueError):
        return 0


def _update_return_code_from_log(return_code: int | None, log_path: Path) -> int | None:
    """Escalate an apparent success to a failure (``return_code=1``) based on the pipeline log.

    SWE-Cosmos' reference-validation stage can find no PASSED tests, skip every patch for the repo, and still let the pipeline exit ``0`` (see :data:`_REFERENCE_VALIDATION_FAILED_MARKER`). Such a run yields no usable instances, so when the log carries that marker we override a clean ``0`` to ``1``. Already-failed (non-zero) or never-produced (``None``) codes are left untouched.
    """
    if return_code != 0:
        return return_code
    try:
        log_text = log_path.read_text(encoding='utf-8', errors='ignore')
    except OSError:
        return return_code
    if _REFERENCE_VALIDATION_FAILED_MARKER in log_text:
        logger.warning(
            'Reference validation produced no PASSED tests (all patches skipped); overriding return_code 0 -> 1.'
        )
        return 1
    return return_code


def _extract_repo_archive(repo_dir: str, archive_path: str) -> str:
    """Overwrite-extract a prebuilt source tarball over the baked checkout, then delete the tarball."""
    lines: list[str] = []
    ok = True
    try:
        extracted = subprocess.run(
            ['tar', '-xzf', archive_path, '-C', repo_dir],
            check=True,
            capture_output=True,
            text=True,
        )
        out = (extracted.stdout or '').strip()
        lines.append(f'extracted {archive_path} -> {repo_dir}' + (f': {out}' if out else ''))
    except (OSError, subprocess.CalledProcessError) as exc:
        ok = False
        detail = ((getattr(exc, 'stdout', '') or '') + '\n' + (getattr(exc, 'stderr', '') or '')).strip() or str(exc)
        lines.append(f'extract failed: {detail}')

    # Always try to remove the uploaded tarball, even when extraction failed.
    try:
        os.remove(archive_path)
        lines.append(f'removed {archive_path}')
    except OSError as exc:
        lines.append(f'failed to remove {archive_path}: {exc}')

    head = 'repo archive applied' if ok else 'repo archive update FAILED (kept baked checkout)'
    msg = f'{head} in {repo_dir}:\n' + '\n'.join(lines)
    (logger.info if ok else logger.warning)(msg)
    return msg


async def run_pipeline(
    *,
    repo_name: str,
    repo_dict_path: str,
    pipeline_env: dict[str, str],
    swe_cosmos_dir: str = '/root/workspace/SWE-Cosmos',
    output_dir: str = '/root/workspace/SWE-Cosmos/outputs',
    bugmaker_strategy: str = 'function_rewrite',
    timeout: float = 86400.0,
    git_archive_path: str | None = None,
    bugmaker_model_name: str = '',
    issuegen_model_name: str = '',
) -> dict[str, Any]:
    """Run the SWE-Cosmos bugmaker pipeline for a single repo inside the sandbox.

    Returns a JSON-serialisable dict consumed by the host task. ``done`` is the presence of the pipeline's completion marker; a falsy ``done`` together with a ``TimeLimitExceeded``/missing status signals an abnormal termination (timeout or hard sandbox kill) rather than a clean pipeline finish.
    """
    from nexus.extensions.agents.installed_common import DEFAULT_PROXY_API_KEY, find_free_port
    from nexus.runtime.context import require_runtime_control
    from nexus.runtime.executors.command import CommandRunStatus, RunCommandRequest, RunCommandResponse, run_command
    from nexus.runtime.server.services.openai_proxy import OpenAIProxy

    rt = require_runtime_control()
    start_time = time.time()

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    log_path = output_path / 'run_bugmaker.log'
    done_path = output_path / 'done'
    git_log = _extract_repo_archive(swe_cosmos_dir, git_archive_path) if git_archive_path else ''

    # Each stage tags its LLM calls with ``openai/<model-id>``: the ``openai/`` prefix makes litellm route through the loopback proxy, and the bare ``<model-id>`` selects the host client. Using the real model id (not an opaque stage name) lets the in-sandbox SWE-Cosmos get_model_config recognise the backend model. All stages share the same proxy (same BASE_URL / API_KEY).
    if not bugmaker_model_name or not issuegen_model_name:
        raise ValueError('run_pipeline requires non-empty bugmaker_model_name and issuegen_model_name.')
    proxy_port = find_free_port()
    stage_models = list(dict.fromkeys([bugmaker_model_name, issuegen_model_name]))
    proxy = OpenAIProxy(
        sio=rt.sio,
        listen_port=proxy_port,
        listen_host='127.0.0.1',
        request_timeout=timeout,
        models=stage_models,
    )
    await proxy.start()

    extra_env: dict[str, str] = {
        **pipeline_env,
        'BASE_URL': f'http://127.0.0.1:{proxy_port}/v1',
        'API_KEY': DEFAULT_PROXY_API_KEY,
        'REPO_NAME': repo_name,
        'REPO_PATH': repo_dict_path,
        'OUTPUT_PATH': output_dir,
        'BUGMAKER_STRATEGY': bugmaker_strategy,
        # Tag the two LLM-using stages so the host routes each to its own client. Their BASE_URL / API_KEY fall back to the shared proxy values above.
        'BUGMAKER_MODEL': f'openai/{bugmaker_model_name}',
        'ISSUEGEN_MODEL': f'openai/{issuegen_model_name}',
    }
    # Never let an inherited proxy hijack the loopback LLM endpoint.
    no_proxy = ','.join(
        dict.fromkeys(
            filter(
                None,
                [os.environ.get('no_proxy'), os.environ.get('NO_PROXY'), 'localhost', '127.0.0.1'],
            )
        )
    )
    extra_env['no_proxy'] = no_proxy
    extra_env['NO_PROXY'] = no_proxy

    script = f'{swe_cosmos_dir.rstrip(os.sep)}/scripts/test_bugmaker_pipeline.sh'
    # pipefail so the script's exit status (not tee's) is what run_command sees.
    bugmaker_pipeline_command = (
        f'set -o pipefail; '
        f'bash {shlex.quote(script)} '
        f'{shlex.quote(repo_name)} {shlex.quote(repo_dict_path)} {shlex.quote(output_dir)} '
        f'2>&1 | tee {shlex.quote(str(log_path))}'
    )

    try:
        result: RunCommandResponse = await run_command(
            RunCommandRequest(
                command=bugmaker_pipeline_command,
                cwd=swe_cosmos_dir,
                timeout=timeout,
                extra_env=extra_env,
            )
        )
    finally:
        await proxy.stop()

    status_name = result.status.value
    timed_out = result.status == CommandRunStatus.TimeLimitExceeded
    return_code = getattr(result, 'return_code', None)
    return_code = _update_return_code_from_log(return_code, log_path)

    instances_path = output_path / 'issuegen' / f'{repo_name}.{bugmaker_strategy}_instances.jsonl'

    return {
        'done': done_path.exists(),
        'repo_name': repo_name,
        'status': status_name,
        'timed_out': timed_out,
        'return_code': return_code,
        'counts': {
            'max_bugs': int(pipeline_env.get('MAX_BUGS', 0)),
            'exhausted': _read_count(output_path / 'exhausted'),
            'unvalidated': _read_count(output_path / 'unvalidated'),
            'validated': _read_count(output_path / 'validated'),
            'resolved': _read_count(output_path / 'resolved'),
            'issued': _read_count(output_path / 'issued'),
        },
        'sandbox_instances_path': str(instances_path) if instances_path.exists() else None,
        'sandbox_log_path': str(log_path) if log_path.exists() else None,
        'git_log': git_log,
        'execution_time': time.time() - start_time,
    }
