"""Configure the unmodified IDE CLI to use Nexus's local OpenAI bridge."""

from __future__ import annotations

import json
import logging
import shlex
import tempfile
import time

from pathlib import Path
from typing import Any

from nexus.extensions.agents.installed_common import (
    DEFAULT_NPM_REGISTRY,
    DEFAULT_PROXY_API_KEY,
    build_runtime_path,
    find_free_port,
    find_runtime_binary,
)
from nexus.utils.shell import get_clean_sandbox_env

from .installer import DEFAULT_VERSION, ensure_system_dependencies, install_cb_insiders
from .output import parse_output
from .process import run_cli, run_pre_start_command, virtual_display


logger = logging.getLogger(__name__)
DEFAULT_MODEL_ID = 'nexus'

# These namespaces can redirect the IDE to a desktop session, replace its
# product, or expose the host's actual model credentials to a sandbox process.
_ENV_PREFIXES = (
    'VSCODE_',
    'ELECTRON_',
    'ACC_',
    'CODEBUDDY_',
    'CB_INSIDERS_',
    'WORKBUDDY_',
    'OPENAI_',
    'ANTHROPIC_',
    'LANGFUSE_',
)
_PROTECTED_ENV = frozenset({
    'HOME',
    'XDG_CONFIG_HOME',
    'XDG_DATA_HOME',
    'PATH',
    'NODE_OPTIONS',
    'NODE_PATH',
    'DISABLE_TELEMETRY',
    'DONT_PROMPT_WSL_INSTALL',
})
_MANAGED_FLAGS = frozenset({
    '-p',
    '--print',
    '--output-format',
    '--input-format',
    '--model',
    '--effort',
    '--permission-mode',
    '-y',
    '--dangerously-skip-permissions',
    '--cli',
    '--user-data-dir',
    '--product-file',
})


def validate_extra_args(args: list[str]) -> None:
    for value in args:
        if value.split('=', 1)[0] in _MANAGED_FLAGS or value == '--':
            raise ValueError(f'{value} is managed by the cb-insiders agent configuration')


def build_argv(
    app: Path,
    task: str,
    working_dir: str,
    user_data_dir: str,
    *,
    model_id: str = DEFAULT_MODEL_ID,
    effort: str | None = 'high',
    permission_mode: str = 'bypassPermissions',
    extra_cli_args: list[str] | None = None,
) -> list[str]:
    extra_cli_args = extra_cli_args or []
    validate_extra_args(extra_cli_args)
    cmd = [
        str(app / 'bin/buddycn-insiders'),
        '--no-sandbox',
        '--password-store=basic',
        '--use-inmemory-secretstorage',
        '--disable-gpu',
        '--disable-dev-shm-usage',
        f'--user-data-dir={user_data_dir}',
        '--cli',
        '--',
        '-p',
        '--output-format',
        'stream-json',
        '--model',
        model_id,
        '--permission-mode',
        permission_mode,
    ]
    if effort:
        cmd += ['--effort', effort]
    # A second delimiter belongs to the IDE CLI parser. Prompts beginning with
    # '-' remain data and can never become extra flags.
    cmd += [*extra_cli_args, '--', task, working_dir]
    return cmd


def build_models_config(
    proxy_url: str, model_id: str, *, max_input_tokens: int, max_output_tokens: int
) -> dict[str, Any]:
    return {
        'models': [
            {
                'id': model_id,
                'name': model_id,
                'vendor': 'nexus',
                'apiKey': DEFAULT_PROXY_API_KEY,
                'url': proxy_url,
                'maxInputTokens': max_input_tokens,
                'maxOutputTokens': max_output_tokens,
                'supportsToolCall': True,
                'supportsImages': True,
                'isDefault': True,
                'relatedModels': {'lite': model_id, 'reasoning': model_id},
            }
        ],
        'availableModels': [model_id],
    }


def build_env(app: Path, run_dir: Path, proxy_base: str, extra_env: dict[str, str] | None = None) -> dict[str, str]:
    def allowed(key: str) -> bool:
        return key not in _PROTECTED_ENV and not key.upper().startswith(_ENV_PREFIXES)

    inherited = get_clean_sandbox_env()
    base = {key: value for key, value in inherited.items() if allowed(key)}
    for key, value in (extra_env or {}).items():
        if allowed(key):
            base[key] = value
        else:
            logger.warning('Ignoring protected cb-insiders environment override: %s', key)
    # Keep the task image's HOME and language caches. Relocating HOME hides
    # preinstalled Go/Rust/npm dependencies and changes the SWE environment.
    home = inherited.get('HOME') or str(Path.home())
    config = run_dir / 'config'
    config.mkdir(parents=True, exist_ok=True)
    product = {
        'endpoint': proxy_base,
        'authentication': {
            'id': f'nexus-local-proxy-{run_dir.name}',
            'type': 'custom-token',
            'label': 'Nexus local proxy',
            'attributes': {'token': DEFAULT_PROXY_API_KEY},
        },
        'config': {'disabled': True},
        'productFeatures': {'CustomModelsJSON': True, 'CustomModelIdPrefix': False},
        'telemetry': {'enabled': False},
    }
    # This is the IDE's supported environment product overlay. Its prompts,
    # craft marketplace, tools and agents continue to come from the package.
    base.update({
        'HOME': home,
        'PATH': build_runtime_path(app / 'bin'),
        'CODEBUDDY_CONFIG_DIR': str(config),
        'ACC_PRODUCT_CONFIG_V3': json.dumps(product),
        'CODEBUDDY_AUTH_TOKEN': DEFAULT_PROXY_API_KEY,
        'ACC_USER_ID': 'nexus-local-proxy',
        'CODEBUDDY_HEADLESS_CLI_IDLE_TIMEOUT_MS': '1000',
        'CODEBUDDY_DISABLE_TELEMETRY': '1',
        'CODEBUDDY_REMOTE_CONFIG_DISABLED': '1',
        'DISABLE_TELEMETRY': '1',
        'DONT_PROMPT_WSL_INSTALL': '1',
    })
    for key in ('XDG_CONFIG_HOME', 'XDG_DATA_HOME'):
        if key in inherited:
            base[key] = inherited[key]
    for key in ('NO_PROXY', 'no_proxy'):
        base[key] = ','.join(filter(None, [base.get(key), '127.0.0.1', 'localhost']))
    return base


def write_config(
    run_dir: Path, proxy_base: str, model_id: str, *, max_input_tokens: int, max_output_tokens: int
) -> None:
    config_dir = run_dir / 'config'
    config_dir.mkdir(parents=True, exist_ok=True)
    config = build_models_config(
        f'{proxy_base}/v1/chat/completions',
        model_id,
        max_input_tokens=max_input_tokens,
        max_output_tokens=max_output_tokens,
    )
    (config_dir / 'models.json').write_text(json.dumps(config, ensure_ascii=False), encoding='utf-8')
    settings_dir = run_dir / 'user-data/User'
    settings_dir.mkdir(parents=True, exist_ok=True)
    (settings_dir / 'settings.json').write_text(
        json.dumps({
            'security.workspace.trust.enabled': False,
            'extensions.autoUpdate': False,
            'extensions.autoCheckUpdates': False,
            'update.mode': 'none',
            'telemetry.telemetryLevel': 'off',
        }),
        encoding='utf-8',
    )


async def run_pre_start_script(script: dict[str, Any], working_dir: str) -> None:
    """Prepare task dependencies without the worker's Nix binary and library paths."""
    from nexus.runtime.context import get_execution_user
    from nexus.runtime.executors.command import _command_for_execution_user

    interpreter = script.get('interpreter', 'bash')
    if interpreter == 'bash':
        executable = find_runtime_binary('bash') or 'bash'
    elif interpreter == 'python':
        executable = find_runtime_binary('python3') or find_runtime_binary('python') or 'python3'
    else:
        raise ValueError(f'Unsupported pre-start interpreter: {interpreter}')
    command = _command_for_execution_user(shlex.join([executable, '-c', script['code']]), get_execution_user())
    # Preserve profile loading for both Bash and Python preparation scripts.
    env = {**get_clean_sandbox_env(), 'BASH_ENV': '~/.bashrc', **(script.get('env') or {})}
    code, stdout, stderr, timed_out = await run_pre_start_command(
        [find_runtime_binary('bash') or 'bash', '-c', command],
        cwd=script.get('cwd') or working_dir,
        env=env,
        timeout=script.get('timeout', 300.0),
    )
    if timed_out or code != 0:
        status = 'timeout' if timed_out else 'error'
        raise RuntimeError(f'Pre-start script failed (status={status}, rc={code}):\n{stdout}\n{stderr}')
    if stdout:
        logger.info('[cb-insiders pre-start]\n%s', stdout)


async def setup_and_run_cb_insiders(
    task: str,
    working_dir: str = '/workspace',
    timeout: float = 3600,
    *,
    version: str = DEFAULT_VERSION,
    npm_registry: str | None = DEFAULT_NPM_REGISTRY,
    npm_username: str | None = None,
    npm_token: str | None = None,
    package_path: str | None = None,
    auto_install_system_deps: bool = True,
    model_id: str = DEFAULT_MODEL_ID,
    effort: str | None = 'high',
    permission_mode: str = 'bypassPermissions',
    max_input_tokens: int = 200000,
    max_output_tokens: int = 16384,
    extra_env: dict[str, str] | None = None,
    extra_cli_args: list[str] | None = None,
    pre_start_scripts: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    from nexus.runtime.context import require_runtime_control
    from nexus.runtime.server.services.openai_proxy import OpenAIProxy

    started = time.monotonic()
    proxy = None
    stage = 'pre_start'
    run_dir: Path | None = None
    try:
        for script in pre_start_scripts or []:
            await run_pre_start_script(script, working_dir)
        stage = 'install'
        app = await install_cb_insiders(
            version=version,
            npm_registry=npm_registry,
            npm_username=npm_username,
            npm_token=npm_token,
            package_path=package_path,
        )
        stage = 'system_dependencies'
        await ensure_system_dependencies(app, auto_install=auto_install_system_deps)
        stage = 'proxy'
        port = find_free_port()
        proxy = OpenAIProxy(
            sio=require_runtime_control().sio,
            listen_host='127.0.0.1',
            listen_port=port,
            request_timeout=timeout,
            models=[model_id],
        )
        await proxy.start()
        stage = 'configuration'
        run_dir = Path(tempfile.mkdtemp(prefix='nexus-cb-insiders-'))
        proxy_base = f'http://127.0.0.1:{port}'
        env = build_env(app, run_dir, proxy_base, extra_env)
        write_config(
            run_dir, proxy_base, model_id, max_input_tokens=max_input_tokens, max_output_tokens=max_output_tokens
        )
        Path(working_dir).mkdir(parents=True, exist_ok=True)
        argv = build_argv(
            app,
            task,
            working_dir,
            str(run_dir / 'user-data'),
            model_id=model_id,
            effort=effort,
            permission_mode=permission_mode,
            extra_cli_args=extra_cli_args,
        )
        stage = 'process'
        transcript = run_dir / 'stdout.jsonl'
        async with virtual_display(env) as display:
            code, stdout, stderr, timed_out = await run_cli(
                argv,
                cwd=working_dir,
                env={**env, **display},
                timeout=timeout,
                user_data_dir=str(run_dir / 'user-data'),
                transcript=transcript,
            )
        parsed = parse_output(stdout, code, timed_out=timed_out)
        if parsed.get('detail'):
            stderr = '\n'.join(filter(None, [stderr, parsed.pop('detail')]))
        else:
            parsed.pop('detail', None)
        return {
            **parsed,
            'output': stdout,
            'stderr': stderr,
            'exit_code': code,
            'execution_time': time.monotonic() - started,
            'transcript_path': str(transcript),
        }
    except Exception as exc:
        logger.exception('cb-insiders failed during %s', stage)
        return {
            'success': False,
            'status': 'error',
            'output': '',
            'stderr': str(exc),
            'exit_code': -1,
            'execution_time': time.monotonic() - started,
            'failure_kind': 'infrastructure',
            'failure_stage': stage,
        }
    finally:
        if proxy is not None:
            await proxy.stop()
        if run_dir:
            # The model config holds only the local proxy's placeholder key.
            # Preserve transcripts and diagnostics for inspection.
            (run_dir / 'config/models.json').unlink(missing_ok=True)
