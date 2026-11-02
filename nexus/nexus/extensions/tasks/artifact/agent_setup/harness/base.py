"""Shared types for the harness registry.

A *harness* is a way of launching an infer agent -- which CLI flags it takes, what has to run
before it starts, which env it needs. The registry maps a harness name to a builder that
returns the ``agent_config`` dict the agent's input model consumes.

Two rules keep the builders honest:

* A builder reads its own settings from :func:`config.get` at call time, not from arguments and
  not at import time. Import-time reads would capture the config cache before the run script
  has exported its ``ARTIFACT_*`` env, silently pinning stale values.
* A builder never emits a key in :data:`RUNNER_OWNED_KEYS`. Those are injected by the RL runner
  with ``setdefault``, so a builder-supplied value wins silently and the runner's intent is lost.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Any


# Keys the RL runner writes into agent_config after the builder has run, via setdefault
# (``nexus/runner/hytuner.py``). setdefault does not overwrite, so a builder that emits any of
# these silently takes precedence over the runner -- the failure shows up as a training-time
# behaviour change with nothing in the logs.
#
# Only agent_config keys belong here. ``nexus_agent_loop.py`` also calls setdefault, but on
# task_config / processors / runtime_provider_config, which builders never touch.
RUNNER_OWNED_KEYS = frozenset({
    'wrap_failed_tool_response_as_tool_error',  # runner/hytuner.py:130
    'max_tool_response_tokens',  # runner/hytuner.py:222, 439
})


@dataclass(frozen=True)
class HarnessSpec:
    """Per-row inputs to a harness builder.

    Only values that genuinely differ between rows live here. Everything global (CLI version,
    log level, scaffolding switches, LLM router) is read from :func:`config.get` by the builder
    itself, so a knob is defined once in ``config.py`` instead of being threaded through this
    dataclass by hand. Adding a global knob therefore does not touch this file.

    ``mcp_servers`` is the exception: the caller filters the known servers by config *and*
    injects the figma token, so what arrives here is a derived value rather than a config read.

    Args:
        working_dir: Agent's working directory. A dataset row may override the default.
        timeout: Agent wall-clock budget in seconds. A dataset row may override the default.
        thinking_preset: Anthropic-proxy thinking preset. Derived per row from the row's model
            alias, since the alias's provider decides the preset.
        extra_pre_start_scripts: Scripts the caller builds at runtime (e.g. the cc diagnostic).
            Appended after any scripts the harness itself requires, so relative order is part
            of the contract. Tuple-of-dicts rather than list so the spec stays shallow-immutable.
        mcp_servers: Resolved MCP server specs. claude_code only -- HyCodeInput has no such
            field and pydantic's extra='ignore' would drop it silently.
        env: Extra env for the agent, as pairs so the frozen spec cannot be mutated through it.
    """

    working_dir: str = '/workspace'
    timeout: float = 1800.0
    thinking_preset: str = 'hunyuan'
    extra_pre_start_scripts: tuple[dict[str, Any], ...] = ()
    mcp_servers: tuple[dict[str, Any], ...] = ()
    # env is a tuple of pairs, not a dict: this dataclass is frozen, and a dict field would let
    # callers mutate shared state through it while still looking immutable.
    env: tuple[tuple[str, str], ...] = ()
    codearena_preset: str = ''
    """dynamicagent ``--agent`` value. Empty = use ``dynamicagent.default_preset``."""

    def env_dict(self) -> dict[str, str]:
        """Return :attr:`env` as a dict for merging into an agent's ``extra_env``."""
        return dict(self.env)


HarnessBuilder = Callable[[HarnessSpec], dict[str, Any]]

# Populated by the per-harness modules at import time (see __init__.py). A dict rather than a
# match statement so a caller can introspect the supported set for error messages.
HARNESS_BUILDERS: dict[str, HarnessBuilder] = {}


def build_infer_agent_config(harness_name: str, spec: HarnessSpec) -> dict[str, Any]:
    """Build the ``agent_config`` for *harness_name*.

    Args:
        harness_name: Registered harness, e.g. ``'claude_code'`` or ``'hycode'``.
        spec: Per-row inputs; see :class:`HarnessSpec`.

    Returns:
        A JSON-serializable dict for the agent's input model.

    Raises:
        ValueError: If *harness_name* is not registered. Falling back to a default harness
            would hide a config typo until the sandbox misbehaved, so an unknown name fails
            here instead.
    """
    builder = HARNESS_BUILDERS.get(harness_name)
    if builder is None:
        raise ValueError(f'Unsupported harness {harness_name!r}. Registered: {sorted(HARNESS_BUILDERS) or "(none)"}')
    return builder(spec)


__all__ = [
    'HARNESS_BUILDERS',
    'RUNNER_OWNED_KEYS',
    'HarnessBuilder',
    'HarnessSpec',
    'build_infer_agent_config',
]
