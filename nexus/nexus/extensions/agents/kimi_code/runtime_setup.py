"""Run Kimi Code prompt mode through the Nexus OpenAI proxy."""

from __future__ import annotations

import asyncio
import hashlib
import json
import logging
import tempfile
import time

from pathlib import Path
from typing import Any

from nexus.extensions.agents.installed_common import DEFAULT_PROXY_API_KEY, find_free_port
from nexus.utils.shell import get_clean_sandbox_env

from .installer import DEFAULT_INSTALL_ROOT, DEFAULT_VERSION, install_kimi_code
from .output import parse_output
from .process import run_process


logger = logging.getLogger(__name__)
_PROTECTED = {'KIMI_CODE_HOME', 'KIMI_CODE_NO_AUTO_UPDATE', 'KIMI_DISABLE_TELEMETRY', 'NODE_OPTIONS'}


def build_config(
    *,
    model_id: str,
    proxy_url: str,
    context_window: int,
    max_tokens: int,
    thinking_effort: str | None,
    max_steps: int | None,
    api_key: str = DEFAULT_PROXY_API_KEY,
) -> str:
    """Generate TOML without importing a runtime-only serializer."""
    quote = json.dumps
    lines = [
        'default_model = "nexus"',
        'telemetry = false',
        '',
        '[providers.nexus]',
        'type = "openai"',
        f'base_url = {quote(proxy_url)}',
        f'api_key = {quote(api_key)}',
        '',
        '[models.nexus]',
        'provider = "nexus"',
        f'model = {quote(model_id)}',
        f'max_context_size = {context_window}',
        f'max_output_size = {max_tokens}',
    ]
    if thinking_effort is not None:
        lines.extend([
            'capabilities = ["thinking"]',
            'reasoning_key = "reasoning_content"',
            f'support_efforts = [{quote(thinking_effort)}]',
            f'default_effort = {quote(thinking_effort)}',
            '',
            '[thinking]',
            'enabled = true',
            f'effort = {quote(thinking_effort)}',
            'keep = "all"',
        ])
    lines.extend(['', '[model_catalog]', 'refresh_on_start = false'])
    if max_steps is not None:
        lines.extend(['', '[loop_control]', f'max_steps_per_turn = {max_steps}'])
    return '\n'.join(lines) + '\n'


def build_env(home: Path, extra_env: dict[str, str]) -> dict[str, str]:
    env = get_clean_sandbox_env()
    for key in list(env):
        if key.startswith(('KIMI_', 'ANTHROPIC_', 'OPENAI_')) or key == 'NODE_OPTIONS':
            env.pop(key)
    for key, value in extra_env.items():
        if key in _PROTECTED or key.startswith(('KIMI_MODEL_', 'KIMI_SECONDARY_MODEL')):
            logger.warning('Ignoring protected Kimi Code environment key: %s', key)
        else:
            env[key] = value
    env.update(KIMI_CODE_HOME=str(home), KIMI_CODE_NO_AUTO_UPDATE='1', KIMI_DISABLE_TELEMETRY='1', NO_COLOR='1', CI='1')
    return env


async def setup_and_run_kimi_code(
    *,
    task: str,
    working_dir: str,
    timeout: float,
    version: str = DEFAULT_VERSION,
    npm_registry: str | None = None,
    model_id: str = 'nexus-model',
    context_window: int = 262144,
    max_tokens: int = 16384,
    thinking_effort: str | None = None,
    max_steps: int | None = None,
    agent_profile: str | None = None,
    extra_env: dict[str, str] | None = None,
) -> dict[str, Any]:
    from nexus.runtime.context import require_runtime_control
    from nexus.runtime.server.services.openai_proxy import OpenAIProxy

    started = time.monotonic()
    run_root = Path(tempfile.mkdtemp(prefix='kimi-code-run-'))
    home = run_root / 'home'
    home.mkdir()
    Path(working_dir).mkdir(parents=True, exist_ok=True)
    transcript = run_root / 'transcript.jsonl'
    proxy = None
    stage = 'install'
    try:
        node, entry = await install_kimi_code(
            version=version, npm_registry=npm_registry, install_root=DEFAULT_INSTALL_ROOT
        )
        env = build_env(home, extra_env or {})
        actual_version = await run_process([str(node), str(entry), '--version'], cwd=working_dir, env=env, timeout=30)
        if actual_version.exit_code != 0 or actual_version.stdout.strip() != version:
            raise RuntimeError('Kimi Code executable version does not match the installed package')
        (run_root / 'installation.json').write_text(
            json.dumps(
                {
                    'package': '@moonshot-ai/kimi-code',
                    'version': version,
                    'node': str(node),
                    'entry': str(entry),
                    'entry_sha256': hashlib.sha256(entry.read_bytes()).hexdigest(),
                },
                indent=2,
            )
        )
        stage = 'proxy'
        proxy_port = find_free_port()
        rt = require_runtime_control()
        proxy = OpenAIProxy(
            sio=rt.sio, listen_port=proxy_port, listen_host='127.0.0.1', request_timeout=timeout, models=[model_id]
        )
        await proxy.start()
        (home / 'config.toml').write_text(
            build_config(
                model_id=model_id,
                proxy_url=f'http://127.0.0.1:{proxy_port}/v1',
                context_window=context_window,
                max_tokens=max_tokens,
                thinking_effort=thinking_effort,
                max_steps=max_steps,
            )
        )
        prompt_path = run_root / 'prompt.txt'
        prompt_path.write_bytes(task.encode('utf-8'))
        prompt_path.chmod(0o600)
        launcher = Path(__file__).with_name('_prompt_launcher.mjs')
        argv = [
            str(node),
            str(launcher),
            str(entry),
            str(prompt_path),
            '--model',
            'nexus',
            '--output-format',
            'stream-json',
        ]
        if agent_profile:
            argv.extend(['--agent', agent_profile])
        # The launcher supplies --prompt in-process, retaining the CLI's auto permission mode.
        stage = 'process'
        result = await run_process(argv, cwd=working_dir, env=env, timeout=timeout, transcript=transcript)
        parsed = parse_output(result.stdout, result.exit_code, timed_out=result.timed_out)
        detail = parsed.pop('detail')
        stderr = '\n'.join(value for value in (result.stderr.strip(), detail) if value)
        return {
            **parsed,
            'output': result.stdout,
            'stderr': stderr,
            'exit_code': result.exit_code,
            'execution_time': time.monotonic() - started,
            'transcript_path': str(transcript),
            'artifact_dir': str(run_root),
        }
    except Exception as exc:
        return {
            'success': False,
            'output': '',
            'stderr': f'{type(exc).__name__}: {exc}',
            'exit_code': -1,
            'exit_status': 'error',
            'failure_kind': 'infrastructure' if stage != 'process' else 'agent',
            'failure_stage': stage,
            'execution_time': time.monotonic() - started,
            'artifact_dir': str(run_root),
        }
    finally:
        if proxy is not None:
            stop = asyncio.create_task(proxy.stop())
            try:
                await asyncio.shield(stop)
            except asyncio.CancelledError:
                await stop
                raise
