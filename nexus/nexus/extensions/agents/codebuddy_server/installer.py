"""@tencent-ai/agent-server installer.

Installs ``@tencent-ai/agent-server`` via npm inside the runtime.
"""

from __future__ import annotations

import json
import logging
import shlex

from pathlib import Path

from nexus.extensions.agents.installed_common import (
    DEFAULT_NPM_REGISTRY,
    build_npm_env,
    find_runtime_binary,
    resolve_npm_registry,
)


logger = logging.getLogger(__name__)


# Pinned for SWE-bench experiment reproducibility. To upgrade, change this
# constant AND update SCAFFOLD_VERSION_PIN in ceph_configs/swe_generator_nj.py.
DEFAULT_AGENT_SERVER_VERSION = '0.0.23-beta'
NPM_PACKAGE = '@tencent-ai/agent-server'


def get_installed_version(install_dir: str) -> str | None:
    """Return the installed agent-server version if present."""
    package_json = Path(install_dir) / 'node_modules' / '@tencent-ai' / 'agent-server' / 'package.json'
    if not package_json.exists():
        return None
    try:
        data = json.loads(package_json.read_text())
    except Exception:
        return None
    version = data.get('version')
    return version if isinstance(version, str) else None


def _resolve_server_executable(install_dir: str) -> Path | None:
    """Return the agent-server executable path, if present."""
    bin_path = Path(install_dir) / 'node_modules' / '.bin' / 'agent-server'
    if bin_path.exists():
        return bin_path
    return None


async def install_agent_server(
    install_dir: str,
    version: str = DEFAULT_AGENT_SERVER_VERSION,
    npm_registry: str | None = DEFAULT_NPM_REGISTRY,
) -> Path:
    """Install @tencent-ai/agent-server via npm and return the executable path."""
    from nexus.runtime.executors.command import CommandRunStatus, RunCommandRequest, run_command

    install_path = Path(install_dir)
    install_path.mkdir(parents=True, exist_ok=True)

    existing = _resolve_server_executable(install_dir)
    installed_version = get_installed_version(install_dir)
    if existing is not None and version != 'latest' and installed_version == version:
        logger.debug(f'agent-server already installed at {existing} (version {installed_version})')
        return existing

    package_spec = NPM_PACKAGE if version == 'latest' else f'{NPM_PACKAGE}@{version}'
    npm_executable = find_runtime_binary('npm') or 'npm'
    registry = resolve_npm_registry(npm_registry)
    command = (
        f'{shlex.quote(npm_executable)} install '
        f'--prefix {shlex.quote(str(install_path))} --no-audit --no-fund --strict-ssl=false '
        f'--registry {shlex.quote(registry)} {shlex.quote(package_spec)}'
    )
    logger.debug(f'Installing agent-server: {package_spec} via registry {registry}')

    result = await run_command(
        RunCommandRequest(
            command=command,
            cwd=str(install_path),
            timeout=1800,
            extra_env=build_npm_env(npm_registry=registry),
        )
    )

    if result.status != CommandRunStatus.Finished or result.return_code != 0:
        stderr = result.stderr or ''
        stdout = result.stdout or ''
        raise RuntimeError(f'Failed to install agent-server via npm: {stderr or stdout or result.status}')

    server_bin = _resolve_server_executable(install_dir)
    if server_bin is None:
        raise RuntimeError(
            f'agent-server install succeeded but binary not found under {install_path}/node_modules/.bin/'
        )

    logger.debug(f'agent-server installed at {server_bin} (version {get_installed_version(install_dir)})')
    return server_bin


_BACKEND_JS_REL = 'node_modules/@tencent-ai/agent-server/.genie/dist/backend/index.js'


def _get_backend_js(install_dir: str) -> Path | None:
    p = Path(install_dir) / _BACKEND_JS_REL
    return p if p.exists() else None


def patch_agent_limits(
    install_dir: str,
    *,
    max_step_limit: int = 0,
    max_token_threshold_ratio: float = 0.99,
    max_input_tokens_fallback: int = 1_000_000,
) -> bool:
    """Remove / relax all hard limits in agent-server that cause premature termination.

    Patches applied (all idempotent):
      1. ``maxStepLimit||100``  →  ``maxStepLimit||<max_step_limit>``
         0 means effectively unlimited (use 999999).
      2. ``maxTokenThreshold||.8*(…maxInputTokens||128e3)``
         →  ``maxTokenThreshold||<ratio>*(…maxInputTokens||<fallback>)``
         Defaults raise the token ceiling so context-length breaks rarely trigger.

    Returns True if the JS file was found and patched.
    """
    js_path = _get_backend_js(install_dir)
    if js_path is None:
        logger.warning('Cannot patch agent-server limits: backend JS not found under %s', install_dir)
        return False

    content = js_path.read_text()
    original = content

    # --- 1. maxStepLimit ---
    effective_step = max_step_limit if max_step_limit > 0 else 999999
    old_step = 'maxStepLimit||100'
    new_step = f'maxStepLimit||{effective_step}'
    if old_step in content:
        content = content.replace(old_step, new_step)
        logger.debug('Patched maxStepLimit: 100 -> %d', effective_step)
    elif new_step in content:
        logger.debug('maxStepLimit already patched to %d', effective_step)
    else:
        logger.warning('maxStepLimit pattern not found; skipping')

    # --- 2. maxTokenThreshold ---
    old_token = '.8*(this.options.model.maxInputTokens||128e3)'
    fallback_str = f'{max_input_tokens_fallback:g}'  # e.g. "1e+06" -> "1e+06"; 1000000 -> "1e+06"
    if max_input_tokens_fallback >= 1000:
        # Prefer JS-friendly scientific notation: 1e6, 2e5, etc.
        import math

        exp = int(math.log10(max_input_tokens_fallback))
        mantissa = max_input_tokens_fallback / (10**exp)
        fallback_str = f'{int(mantissa)}e{exp}' if mantissa == int(mantissa) else str(max_input_tokens_fallback)
    new_token = f'{max_token_threshold_ratio}*(this.options.model.maxInputTokens||{fallback_str})'
    if old_token in content:
        content = content.replace(old_token, new_token)
        logger.debug('Patched maxTokenThreshold: 0.8*128k -> %s*%s', max_token_threshold_ratio, fallback_str)
    elif new_token in content:
        logger.debug('maxTokenThreshold already patched')
    else:
        logger.warning('maxTokenThreshold pattern not found; skipping')

    if content != original:
        js_path.write_text(content)
    return True


__all__ = [
    'DEFAULT_AGENT_SERVER_VERSION',
    'NPM_PACKAGE',
    'get_installed_version',
    'install_agent_server',
    'patch_agent_limits',
]
