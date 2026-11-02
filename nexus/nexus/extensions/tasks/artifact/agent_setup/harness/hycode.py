"""HyCode harness: ``@tencent/hycode`` CLI launch config.

HyCode is Claude Code's internal fork, so it reuses the ``CLAUDE_CODE_*`` env switches, but it
is a separate CLI with its own npm package and its own flag surface.
"""

from __future__ import annotations

from typing import Any

from nexus.extensions.tasks.artifact import config
from nexus.extensions.tasks.artifact.agent_setup.claude_config import (
    CLAUDE_CODE_COMMON_ENV,
    STAR_PROXY_ENV,
)
from nexus.extensions.tasks.artifact.agent_setup.harness.base import HarnessSpec


def build_agent_config(spec: HarnessSpec) -> dict[str, Any]:
    """Build the HyCode infer agent's ``agent_config``.

    Every difference from claude_code follows from HyCode being a separate CLI:
    ``--output-format`` is stream-json rather than json, it takes
    ``--no-session-persistence``, and it needs ``npm_registry`` pointed at the internal
    mirror. ``version`` must be a HyCode release -- claude_code's version number makes npm
    fail with ETARGET and the run yields an empty archive.

    HyCode has no MCP support (``HyCodeInput`` has no ``mcp_servers`` field, and pydantic's
    ``extra='ignore'`` drops one passed in without a word), so no playwright config is
    injected: the infer agent has no browser to self-check with. Eval is unaffected -- judges
    are built separately by ``claude_config.default_judge_claude_input``.

    ``working_dir`` comes from *spec* rather than HyCode's own ``/testbed`` default, because
    archive collection and judge deployment both follow artifact's directory convention.

    Args:
        spec: Per-row inputs; see :class:`HarnessSpec`.

    Returns:
        A JSON-serializable ``agent_config`` for ``HyCodeInput``.
    """
    # hycode_cfg stays a config read: HyCodeCfg is this harness's own settings group, and the
    # old signature only threaded it through because the caller happened to hold it.
    cfg = config.get()
    hycode_cfg = cfg.hycode

    extra_cli_args = [
        '--print',
        '--input-format',
        'text',
        '--output-format',
        'stream-json',
        '--verbose',
        '--no-session-persistence',
    ]
    disallowed: list[str] = []
    if not hycode_cfg.allow_native_web:
        disallowed += ['WebSearch', 'WebFetch']
    if not hycode_cfg.allow_plan_mode:
        # Deny the entry only. gemini-3.5-flash entered plan mode on its first turn and
        # returned 1 file instead of 32, so the entry stays blocked while the exit stays open.
        disallowed += ['EnterPlanMode']
    if disallowed:
        extra_cli_args += ['--disallowedTools', *disallowed]
    # ExitPlanMode is pre-authorized unconditionally -- it is the escape hatch, not a
    # capability. Denying it alongside EnterPlanMode traps the agent: anything that still
    # lands it in plan mode then has no way out, and the rollout burns its whole budget
    # waiting for an approval that never comes. Appended after --disallowedTools so the
    # upstream flag order stays a prefix of ours.
    extra_cli_args += ['--allowedTools', 'ExitPlanMode']
    extra_cli_args += list(hycode_cfg.extra_cli_args)

    agent_config: dict[str, Any] = {
        'working_dir': spec.working_dir,
        'timeout': spec.timeout,
        'version': hycode_cfg.version,
        'npm_registry': hycode_cfg.npm_registry,
        # Shared with claude_code: one conversion-log knob for whichever infer harness runs.
        'conversion_log_level': cfg.runtime.infer_conversion_log_level,
        'thinking_preset': hycode_cfg.thinking_preset or spec.thinking_preset,
        'extra_cli_args': extra_cli_args,
        # HyCode reuses claude_code's CLAUDE_CODE_* switches (same CLI, internal fork), so the
        # traffic/auth tuning and star proxy come from the same constants;
        # PLAYWRIGHT_MCP_CONFIG is useless to it and is not injected.
        'extra_env': {**STAR_PROXY_ENV, **CLAUDE_CODE_COMMON_ENV, **spec.env_dict()},
        'pre_start_scripts': list(spec.extra_pre_start_scripts),
    }
    if cfg.models.llm_router_enable:
        agent_config['no_thinking_request_args'] = {
            'model': cfg.models.no_thinking_model or 'bedrock/anthropic.claude-opus-4-8'
        }
        if cfg.models.haiku_model:
            agent_config['haiku_model'] = cfg.models.haiku_model
    return agent_config


__all__ = ['build_agent_config']
