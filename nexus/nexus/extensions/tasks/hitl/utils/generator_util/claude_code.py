"""The ``worker_config`` a Claude Code backend accepts.

Every key here is one the two Claude Code managers read: its CLI flags, the npm
package or binary to install, the model identifier it shows for its own ``/model``,
and its auto-compact window. A backend driving a different harness assembles its own
config and shares only what :mod:`worker` holds.
"""

import os

from typing import Any

from nexus.extensions.tasks.hitl.actor.input_spec import BUILTIN_COMMANDS_KEY
from nexus.extensions.tasks.hitl.model import resolve_profile
from nexus.extensions.tasks.hitl.utils.generator_util.env import _env_bool, _env_int, _env_list
from nexus.extensions.tasks.hitl.utils.generator_util.worker import (
    _read_user_memory,
    _worker_context_window,
    _worker_profile_kwargs,
    _worker_turn_timeout,
    _worker_version,
    worker_mcp_servers,
)


DEFAULT_CC_TURN_TIMEOUT_SECONDS = 7200

DEFAULT_CC_VERSION = '2.1.117'
"""The Claude Code npm release a run installs when it names none.

The same release the run scripts name, so a run driving the generator directly
installs what a scripted batch does.
"""


# How the worker's client encodes thinking blocks on the wire. Overridable via
# HITL_WORKER_THINKING_PRESET. The client backend comes from the resolved protocol's
# client_name.
#
# The two names are shapes rather than vendors (``anthropic_proxy/models.py``):
# ``hunyuan`` is one flat ``reasoning_content`` string; ``bedrock-claude`` is a block
# array whose signature has to survive the round trip, which Claude over either
# channel needs.
_PROFILE_THINKING_PRESET: dict[str, str] = {
    'hy3': 'hunyuan',
    'naci_glm': 'hunyuan',
    'openai_basic': 'hunyuan',
    'default': 'hunyuan',
    'gpt_responses': 'hunyuan',
    'bedrock_claude': 'bedrock-claude',
    'gcp_claude': 'bedrock-claude',
}


def _worker_thinking_preset(profile: str) -> str:
    env_preset = os.environ.get('HITL_WORKER_THINKING_PRESET', '').strip()
    if env_preset:
        return env_preset
    return _PROFILE_THINKING_PRESET.get(profile, '')


def _build_worker_config(profile: str, *, session_id: str = '') -> dict[str, Any]:
    """Build the Claude Code worker config dict for the selected profile."""
    protocol = resolve_profile(profile)(**_worker_profile_kwargs(session_id=session_id))

    disallowed = ['WebSearch', 'WebFetch']
    if _env_bool('HITL_WORKER_DISABLE_AGENT'):
        disallowed.append('Agent')
    extra_cli_args = ['--disallowedTools', ','.join(disallowed)]
    # In -p mode over a self-built ANTHROPIC_BASE_URL (ToolSearch off), Grep/Glob are
    # not surfaced to the worker unless explicitly allowed. --allowedTools grants
    # these and does not restrict the worker to them, so it coexists with
    # --disallowedTools above.
    extra_cli_args += ['--allowedTools', 'Grep,Glob']

    append_system_prompt = os.environ.get('HITL_WORKER_APPEND_SYSTEM_PROMPT', '')
    if append_system_prompt:
        extra_cli_args.extend(['--append-system-prompt', append_system_prompt])

    config: dict[str, Any] = {
        'extra_cli_args': extra_cli_args,
    }

    user_memory = _read_user_memory()
    if user_memory:
        config['user_memory'] = user_memory

    mcp_servers = worker_mcp_servers(config)
    if mcp_servers:
        config['mcp_servers'] = mcp_servers

    binary_url = os.environ.get('HITL_WORKER_BINARY_URL', '')
    if binary_url:
        config['binary_url'] = binary_url

    # The npm release to install. Named rather than left to the manager's ``latest``,
    # which moves under a batch and makes two runs of it different runs.
    config['version'] = _worker_version(DEFAULT_CC_VERSION)

    # Haiku-class and no-think model strings come from the environment. The profile
    # contributes the no-think protocol (request_args shape, incl. routing headers).
    haiku_model = os.environ.get('HITL_WORKER_HAIKU_MODEL', '')
    if haiku_model:
        config['haiku_model'] = haiku_model

    no_thinking_model = os.environ.get('HITL_WORKER_NO_THINKING_MODEL', '')
    if no_thinking_model:
        config['no_thinking_request_args'] = {'model': no_thinking_model, **protocol.no_think_request_args}

    # Env override wins, else the profile's default.
    thinking_preset = _worker_thinking_preset(profile)
    if thinking_preset:
        config['thinking_preset'] = thinking_preset

    # The identifier Claude Code shows for its own /model and context accounting. The
    # host worker client's real model is separate and the proxy substitutes it on every
    # call. Defaults to claude-opus-4-8 in the manager, which also appends the
    # 1M-context [1m] suffix.
    cc_model = os.environ.get('HITL_WORKER_CC_IDENTIFIER', '').strip()
    if cc_model:
        config['cc_model'] = cc_model

    # Extended (1M) context is the default; auto-compact sized to match. The CC specs
    # name the same number as their own default (``spec.DEFAULT_AUTO_COMPACT_WINDOW``).
    auto_compact_window = _worker_context_window(default=950000)
    autocompact_pct = _env_int('HITL_CC_AUTOCOMPACT_PCT')

    if auto_compact_window:
        config['auto_compact_window'] = auto_compact_window
    if autocompact_pct:
        config['autocompact_pct_override'] = autocompact_pct

    config['timeout'] = _worker_turn_timeout(DEFAULT_CC_TURN_TIMEOUT_SECONDS)

    # Progressive tool disclosure, which is the only way Grep and Glob reach the
    # worker: both backends disable experimental betas by default. The spec turns
    # these two into the variables Claude Code reads.
    config['enable_tool_search'] = _env_bool('HITL_WORKER_TOOL_SEARCH', default=True)
    tool_search_mode = os.environ.get('HITL_WORKER_TOOL_SEARCH_MODE', '').strip()
    if tool_search_mode:
        config['tool_search_mode'] = tool_search_mode

    # Which of Claude Code's own commands this run offers the user simulator. None by
    # default. A run offers the skills it installed either way.
    builtins = _env_list('HITL_CC_BUILTIN_COMMANDS')
    if builtins:
        config[BUILTIN_COMMANDS_KEY] = builtins

    return config


__all__ = [
    'DEFAULT_CC_TURN_TIMEOUT_SECONDS',
    'DEFAULT_CC_VERSION',
    '_build_worker_config',
]
