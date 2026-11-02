"""HyCode DynamicAgent harness: ``@tencent/hycode-dynamicagent`` with code-arena presets."""

from __future__ import annotations

from typing import Any

from nexus.extensions.tasks.artifact import config
from nexus.extensions.tasks.artifact.agent_setup.claude_config import (
    CLAUDE_CODE_COMMON_ENV,
    STAR_PROXY_ENV,
)
from nexus.extensions.tasks.artifact.agent_setup.harness.base import HarnessSpec


_NATIVE_DEPS_SCRIPT: dict[str, Any] = {
    'interpreter': 'bash',
    'timeout': 300,
    'fail_on_error': False,
    'code': 'set -euo pipefail\nif command -v apk >/dev/null 2>&1; then\n  apk add --no-cache libstdc++ libgcc\nfi\n',
}


def _bootstrap_scripts(preset: str, working_dir: str, enabled: bool) -> list[dict[str, Any]]:
    """Return preset-specific scaffolding scripts (react/nextjs template + npm install)."""
    if not enabled:
        return []
    from nexus.extensions.agents.hycode_dynamicagent.generator import _bootstrap_scripts_for

    # working_dir must be threaded: the scripts cd into it, and the agent's own default is
    # /testbed while artifact runs in /workspace.
    return _bootstrap_scripts_for(preset, working_dir)


def _service_env(da: config.DynamicAgentCfg, working_dir: str) -> dict[str, str]:
    """Env vars for the bundled code-arena tools (image gen, serper)."""
    return {
        'CODE_ARENA_ROOT': working_dir,
        'CLAUDE_CODE_USE_NATIVE_FILE_SEARCH': '1',
        'CLAUDE_CODE_ATTRIBUTION_HEADER': '0',
        'CLAUDE_CODE_DISABLE_NONESSENTIAL_TRAFFIC': '1',
        'CLAUDE_CODE_DISABLE_EXPERIMENTAL_BETAS': '1',
        'IMAGE_GEN_APP_ID': da.image_gen_app_id,
        'IMAGE_GEN_APP_KEY': da.image_gen_app_key,
        'SERPER_APP_ID': da.image_gen_app_id,
        'SERPER_APP_KEY': da.image_gen_app_key,
        'SERPER_BASE_URL': da.serper_base_url,
        'IMAGE_GEN_SERPER_BASE_URL': da.serper_base_url,
        'IMAGE_GEN_NACI_URL': da.image_gen_naci_url,
        'IMAGE_GEN_NACI_TIMEOUT_SECONDS': str(da.image_gen_naci_timeout_seconds),
    }


def build_agent_config(spec: HarnessSpec) -> dict[str, Any]:
    """Build the hycode_dynamicagent infer agent's ``agent_config``."""
    cfg = config.get()
    da = cfg.dynamicagent
    preset = spec.codearena_preset or da.default_preset

    extra_cli_args = [
        '--print',
        '--input-format',
        'text',
        '--output-format',
        'stream-json',
        '--verbose',
        '--no-session-persistence',
        '--permission-mode',
        'bypassPermissions',
    ]
    # Escape hatch, kept open unconditionally: denying the exit traps the agent in plan mode.
    if not da.allow_native_web:
        extra_cli_args += ['--disallowedTools', 'WebSearch', 'WebFetch']
    extra_cli_args += ['--allowedTools', 'ExitPlanMode']
    extra_cli_args += ['--agent', preset]
    extra_cli_args += list(da.extra_cli_args)

    agent_config: dict[str, Any] = {
        'working_dir': spec.working_dir,
        'timeout': spec.timeout,
        'version': da.version,
        'npm_registry': da.npm_registry,
        'conversion_log_level': cfg.runtime.infer_conversion_log_level,
        'thinking_preset': da.thinking_preset or spec.thinking_preset,
        'extra_cli_args': extra_cli_args,
        'extra_env': {
            **STAR_PROXY_ENV,
            **CLAUDE_CODE_COMMON_ENV,
            **_service_env(da, spec.working_dir),
            **spec.env_dict(),
        },
        'pre_start_scripts': [
            _NATIVE_DEPS_SCRIPT,
            *_bootstrap_scripts(preset, spec.working_dir, da.bootstrap_templates),
            *spec.extra_pre_start_scripts,
        ],
    }
    if da.tee_log_dir:
        agent_config['tee_log_dir'] = da.tee_log_dir
    if cfg.models.llm_router_enable:
        agent_config['no_thinking_request_args'] = {
            'model': cfg.models.no_thinking_model or 'bedrock/anthropic.claude-opus-4-8'
        }
        if cfg.models.haiku_model:
            agent_config['haiku_model'] = cfg.models.haiku_model
    return agent_config
