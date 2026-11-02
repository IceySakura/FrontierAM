"""claude_code harness: Claude Code CLI launch config for the infer agent.

The infer agent has to write project files, so unlike the judge config in
``claude_config.default_judge_claude_input`` this one does not pass
``--allowedTools ExitPlanMode`` (which would block Write/Edit and yield an empty artifact) and
does not cap ``--max-turns`` (generation length is not known up front).
"""

from __future__ import annotations

from typing import Any

from nexus.extensions.tasks.artifact import config
from nexus.extensions.tasks.artifact.agent_setup import claude_hooks
from nexus.extensions.tasks.artifact.agent_setup.claude_config import (
    CLAUDE_CODE_COMMON_ENV,
    DEFAULT_CLAUDE_CODE_VERSION,
    PLAYWRIGHT_MCP_CONFIG_PATH,
    STAR_PROXY_ENV,
    build_playwright_mcp_config_pre_start,
)
from nexus.extensions.tasks.artifact.agent_setup.harness.base import HarnessSpec


def build_agent_config(spec: HarnessSpec) -> dict[str, Any]:
    """Build the claude_code infer agent's ``agent_config``.

    Global knobs (CLI version, log level, scaffolding, LLM router) are read from
    :func:`config.get` here rather than passed in, so each one is declared once in
    ``config.py``. *spec* carries only what varies per row.

    The playwright-mcp config script comes first in ``pre_start_scripts``: it writes the
    config file the MCP server reads at startup, so anything ordered before it would run
    without a browser available.

    Scaffolding (hooks / plugins / skills) is opt-in and off by default; each piece is
    appended only when the corresponding config field is set.

    Args:
        spec: Per-row inputs; see :class:`HarnessSpec`.

    Returns:
        A JSON-serializable ``agent_config`` for the claude_code agent.
    """
    cfg = config.get()
    cc = cfg.claude_code
    agent_config: dict[str, Any] = {
        'working_dir': spec.working_dir,
        'timeout': spec.timeout,
        'version': cc.version or DEFAULT_CLAUDE_CODE_VERSION,
        'mcp_servers': list(spec.mcp_servers),
        'conversion_log_level': cfg.runtime.infer_conversion_log_level,
        'thinking_preset': spec.thinking_preset,
        'extra_cli_args': ['--print', '--output-format', 'json', '--verbose'],
        # CLAUDE_CODE_* traffic/auth tuning from the single source of truth
        # (CLAUDE_CODE_COMMON_ENV, shared with the judge side) + star proxy + playwright config.
        'extra_env': {
            **STAR_PROXY_ENV,
            **CLAUDE_CODE_COMMON_ENV,
            'PLAYWRIGHT_MCP_CONFIG': PLAYWRIGHT_MCP_CONFIG_PATH,
            **spec.env_dict(),
        },
    }
    if cc.binary_url:
        agent_config['binary_url'] = cc.binary_url
    # LLM router: send no-thinking auxiliary calls (title / commit message / Task-Explore
    # subagents) to a cheaper model so they do not consume the main one (e.g. opus max-effort).
    if cfg.models.llm_router_enable:
        agent_config['no_thinking_request_args'] = {
            'model': cfg.models.no_thinking_model or 'bedrock/anthropic.claude-opus-4-8'
        }
        if cfg.models.haiku_model:
            agent_config['haiku_model'] = cfg.models.haiku_model

    scripts = [build_playwright_mcp_config_pre_start(), *spec.extra_pre_start_scripts]
    patches: list[dict[str, Any]] = []
    if cc.hooks_enabled:
        patches += claude_hooks.build_hook_patches()
        scripts += [claude_hooks.build_askuser_hook_writer()]
        # The hook scripts run inside the container and read their knobs from os.environ, so
        # the switches only reach them through extra_env. Without this the stop hook stays a
        # no-op no matter what the config says. setdefault, so an explicit per-row
        # spec.env entry still wins.
        for key, value in claude_hooks.build_hook_env().items():
            agent_config['extra_env'].setdefault(key, value)
    if cc.plugins:
        scripts += [claude_hooks.build_plugin_install_script(list(cc.plugins))]
    if cc.skills_dirs:
        scripts += [claude_hooks.build_skills_install_script(claude_hooks.parse_skills_dirs(','.join(cc.skills_dirs)))]
    agent_config['pre_start_scripts'] = scripts
    if patches:
        agent_config['config_file_patches'] = patches
    return agent_config


__all__ = ['build_agent_config']
