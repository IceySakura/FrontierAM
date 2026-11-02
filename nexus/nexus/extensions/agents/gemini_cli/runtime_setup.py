"""Runtime-side setup and execution for Gemini CLI via a local proxy."""

from __future__ import annotations

import json
import logging
import shlex
import time

from dataclasses import dataclass
from pathlib import Path
from typing import Any

from nexus.extensions.agents.installed_common import (
    DEFAULT_AGENT_INSTALL_ROOT,
    DEFAULT_NPM_REGISTRY,
    build_runtime_path,
    find_free_port,
    find_runtime_binary,
    unpack_command_result,
)

from .installer import (
    DEFAULT_GEMINI_CLI_VERSION,
    DEFAULT_INSTALL_ROOT,
    GeminiCliInstallation,
    install_gemini_cli,
)


logger = logging.getLogger(__name__)


DEFAULT_WORKING_DIR = '/workspace'
DEFAULT_HOME_ROOT = f'{DEFAULT_AGENT_INSTALL_ROOT}/gemini-cli-home'
DEFAULT_UPSTREAM_BASE_URL = 'http://trpc-gpt-eval.production.polaris:8080'
DEFAULT_MODEL = 'gemini-2.5-flash'
DEFAULT_OUTPUT_FORMAT = 'json'
DEFAULT_APPROVAL_MODE = 'yolo'
DEFAULT_API_TIMEOUT = 60

_GEMINI_LAUNCHER_SOURCE = """import crypto from 'node:crypto';
import process from 'node:process';

import { AuthType } from '@google/gemini-cli-core';
import { loadCliConfig, parseArguments } from '@google/gemini-cli/dist/src/config/config.js';
import { loadSettings } from '@google/gemini-cli/dist/src/config/settings.js';
import { runNonInteractive } from '@google/gemini-cli/dist/src/nonInteractiveCli.js';
import { runExitCleanup } from '@google/gemini-cli/dist/src/utils/cleanup.js';

async function main() {
  const input = process.env.NEXUS_GEMINI_PROMPT;
  const cwd = process.env.NEXUS_GEMINI_WORKDIR || process.cwd();
  const model = process.env.NEXUS_GEMINI_MODEL || 'gemini-2.5-flash';
  const approvalMode = process.env.NEXUS_GEMINI_APPROVAL_MODE || 'yolo';
  const outputFormat = process.env.NEXUS_GEMINI_OUTPUT_FORMAT || 'json';
  const baseUrl = process.env.NEXUS_GEMINI_BASE_URL;
  const dummyApiKey = process.env.NEXUS_GEMINI_DUMMY_API_KEY || 'nexus-proxy-dummy-key';

  if (!input) {
    throw new Error('NEXUS_GEMINI_PROMPT is required');
  }
  if (!baseUrl) {
    throw new Error('NEXUS_GEMINI_BASE_URL is required');
  }

  process.chdir(cwd);
  process.env.GOOGLE_API_KEY = process.env.GOOGLE_API_KEY || dummyApiKey;
  process.env.GOOGLE_GENAI_API_VERSION = process.env.GOOGLE_GENAI_API_VERSION || 'v1';

  process.argv = [
    process.execPath,
    'gemini',
    '--prompt',
    input,
    '--model',
    model,
    '--approval-mode',
    approvalMode,
    '--output-format',
    outputFormat,
    '--sandbox=false',
  ];

  const settings = loadSettings(cwd);
  const argv = await parseArguments(settings.merged);
  const config = await loadCliConfig(settings.merged, `nexus-${crypto.randomUUID()}`, argv, { cwd });

  await config.initialize();
  await config.refreshAuth(AuthType.USE_VERTEX_AI, undefined, baseUrl);
  await runNonInteractive({
    config,
    settings,
    input,
    prompt_id: crypto.randomUUID().replace(/-/g, ''),
  });
}

main()
  .then(async () => {
    await runExitCleanup();
  })
  .catch(async (error) => {
    try {
      await runExitCleanup();
    } catch {}
    process.stderr.write(`${error instanceof Error ? error.stack || error.message : String(error)}\\n`);
    process.exitCode = 1;
  });
"""


@dataclass(slots=True)
class GeminiCliResult:
    """Result of running Gemini CLI."""

    success: bool
    output: str
    stderr: str
    exit_code: int
    execution_time: float
    response: str | None = None


def _parse_cli_json(stdout: str) -> tuple[str | None, dict[str, Any] | None]:
    payload = stdout.strip()
    if not payload:
        return None, None
    candidate_payloads = [payload]
    brace_positions = [index for index, char in enumerate(payload) if char == '{']
    for index in reversed(brace_positions):
        if index == 0:
            continue
        candidate_payloads.append(payload[index:])

    seen: set[str] = set()
    for candidate in candidate_payloads:
        if candidate in seen:
            continue
        seen.add(candidate)
        try:
            data = json.loads(candidate)
        except json.JSONDecodeError:
            continue
        if not isinstance(data, dict):
            continue
        response = data.get('response')
        return (response if isinstance(response, str) else None), data
    return None, None


def _ensure_cli_home(home_root: str) -> Path:
    home_dir = Path(home_root)
    gemini_dir = home_dir / '.gemini'
    home_dir.mkdir(parents=True, exist_ok=True)
    gemini_dir.mkdir(parents=True, exist_ok=True)
    return home_dir


def _ensure_launcher(installation: GeminiCliInstallation) -> Path:
    installation.launcher_path.write_text(_GEMINI_LAUNCHER_SOURCE)
    return installation.launcher_path


def _error_result(message: str, start_time: float) -> dict[str, Any]:
    logger.error(message)
    return {
        'success': False,
        'output': '',
        'stderr': message,
        'exit_code': -1,
        'execution_time': time.time() - start_time,
        'response': None,
    }


async def setup_and_run_gemini_cli(
    task: str,
    working_dir: str = DEFAULT_WORKING_DIR,
    timeout: float = 3600,
    version: str = DEFAULT_GEMINI_CLI_VERSION,
    model: str = DEFAULT_MODEL,
    npm_registry: str | None = DEFAULT_NPM_REGISTRY,
    approval_mode: str = DEFAULT_APPROVAL_MODE,
    output_format: str = DEFAULT_OUTPUT_FORMAT,
    install_root: str = DEFAULT_INSTALL_ROOT,
    home_root: str = DEFAULT_HOME_ROOT,
) -> dict[str, Any]:
    """Install, proxy, and run Gemini CLI on the runtime side."""
    from nexus.runtime.context import require_runtime_control
    from nexus.runtime.executors.command import RunCommandRequest, run_command
    from nexus.runtime.server.services.gemini_proxy import GeminiProxy

    rt = require_runtime_control()

    start_time = time.time()

    if not task.strip():
        return _error_result('Gemini CLI task prompt is empty', start_time)

    work_path = Path(working_dir)
    work_path.mkdir(parents=True, exist_ok=True)
    home_dir = _ensure_cli_home(home_root)

    install_env = {
        'HOME': str(home_dir),
        'GEMINI_CLI_HOME': str(home_dir),
    }
    try:
        installation = await install_gemini_cli(
            version=version,
            install_root=install_root,
            timeout=min(timeout, 900),
            npm_registry=npm_registry,
            extra_env=install_env,
        )
    except Exception as exc:
        return _error_result(f'Failed to install Gemini CLI: {exc}', start_time)

    launcher_path = _ensure_launcher(installation)

    proxy_port = find_free_port()
    proxy = GeminiProxy(
        sio=rt.sio,
        listen_host='127.0.0.1',
        listen_port=proxy_port,
        request_timeout=timeout,
    )

    try:
        await proxy.start()
    except Exception as exc:
        return _error_result(f'Failed to start Gemini proxy: {exc}', start_time)

    try:
        node_path = find_runtime_binary('node') or 'node'
        cli_env = {
            'HOME': str(home_dir),
            'GEMINI_CLI_HOME': str(home_dir),
            'GOOGLE_GENAI_API_VERSION': 'v1',
            'NEXUS_GEMINI_PROMPT': task,
            'NEXUS_GEMINI_WORKDIR': str(work_path),
            'NEXUS_GEMINI_MODEL': model,
            'NEXUS_GEMINI_APPROVAL_MODE': approval_mode,
            'NEXUS_GEMINI_OUTPUT_FORMAT': output_format,
            'NEXUS_GEMINI_BASE_URL': f'http://127.0.0.1:{proxy_port}',
            'NEXUS_GEMINI_DUMMY_API_KEY': 'nexus-proxy-dummy-key',
            'PATH': build_runtime_path(installation.install_dir / 'node_modules' / '.bin'),
        }
        command = ' '.join(shlex.quote(part) for part in [node_path, str(launcher_path)])
        result = await run_command(
            RunCommandRequest(
                command=command,
                cwd=str(work_path),
                timeout=timeout,
                extra_env=cli_env,
            )
        )

        stdout, stderr, exit_code = unpack_command_result(
            result,
            timeout_message='Gemini CLI execution timed out',
            error_message='Gemini CLI command execution failed',
        )

        response, _payload = _parse_cli_json(stdout)
        execution_time = time.time() - start_time

        logger.info('Gemini CLI finished with exit code %s in %.2fs', exit_code, execution_time)
        return {
            'success': exit_code == 0,
            'output': stdout,
            'stderr': stderr,
            'exit_code': exit_code,
            'execution_time': execution_time,
            'response': response,
        }
    finally:
        await proxy.stop()


__all__ = [
    'DEFAULT_API_TIMEOUT',
    'DEFAULT_APPROVAL_MODE',
    'DEFAULT_HOME_ROOT',
    'DEFAULT_MODEL',
    'DEFAULT_OUTPUT_FORMAT',
    'DEFAULT_UPSTREAM_BASE_URL',
    'DEFAULT_WORKING_DIR',
    'GeminiCliResult',
    'setup_and_run_gemini_cli',
]
