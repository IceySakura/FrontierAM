"""Native harness policy: configured by harness knobs, never by model family.

Client routing/auth/reasoning wire formats belong to the client/bridge. These
builders define the same CLI, prompt, budget and completion loop for every client.
"""

from __future__ import annotations

import json
import os

from pathlib import Path
from typing import Any

from nexus.extensions.tasks.os_world.data import task_source
from nexus.extensions.tasks.os_world.execution.claude_code import _guide_text
from nexus.extensions.tasks.os_world.execution.codex_profile import ASSETS, OSWorldCodexOptions


def _positive(name: str, default: int) -> int:
    value = task_source.env_int(name, default)
    if value is None or value <= 0:
        raise ValueError(f'{name} must be positive')
    return value


def cc_config() -> dict[str, Any]:
    """Standard Claude Code execution, followed by the OSWorld evaluator."""
    model = os.getenv('CC_MODEL') or os.getenv('CC_ANTHROPIC_MODEL') or 'claude-opus-4-8'
    extra_env = {
        'ANTHROPIC_MODEL': model,
        'CLAUDE_CODE_ATTRIBUTION_HEADER': '0',
        'CLAUDE_CODE_DISABLE_NONESSENTIAL_TRAFFIC': '1',
        'CLAUDE_CODE_DISABLE_EXPERIMENTAL_BETAS': '1',
        'API_TIMEOUT_MS': str(_positive('CC_API_TIMEOUT_MS', 600_000)),
    }
    # Optional capacity override is explicit and identical for all clients.
    # Native entries never rewrite/patch a Hunyuan-only CC binary.
    if os.getenv('CC_AUTO_COMPACT_WINDOW', '').strip():
        extra_env['CLAUDE_CODE_AUTO_COMPACT_WINDOW'] = str(_positive('CC_AUTO_COMPACT_WINDOW', 200000))
    if os.getenv('CC_AUTOCOMPACT_PCT_OVERRIDE', '').strip():
        percent = _positive('CC_AUTOCOMPACT_PCT_OVERRIDE', 94)
        if percent > 100:
            raise ValueError('CC_AUTOCOMPACT_PCT_OVERRIDE must be <= 100')
        extra_env['CLAUDE_AUTOCOMPACT_PCT_OVERRIDE'] = str(percent)
    timeout = task_source.env_float('CC_TIMEOUT', 1800)
    if timeout <= 0:
        raise ValueError('CC_TIMEOUT must be positive')
    options = {
        'runtime_name': 'main',
        'guide_text': _guide_text(
            version=os.getenv('CC_GUIDE_VERSION', '').strip(),
            filename=os.getenv('CC_GUIDE_FILE', '').strip(),
        ),
        'cc_model': model,
        'expected_claude_code_version': os.getenv('CC_VERSION', '2.1.205'),
        'thinking_preset': os.getenv('CC_THINKING_PRESET', '').strip() or 'auto',
        'disallowed_tools': os.getenv('CC_DISALLOWED_TOOLS', ''),
        'process_guard': task_source.env_bool('OSWORLD_CC_PROCESS_GUARD', False),
        'extra_env': extra_env,
    }
    return {
        'execution_profile': 'osworld',
        'timeout': timeout,
        'max_turns': _positive('CC_MAX_TURNS', 100),
        'execution_options': options,
    }


def codex_config() -> dict[str, Any]:
    """One Codex driver/catalog policy; capacity is explicit, not inferred from a brand."""
    model = os.getenv('CODEX_MODEL', '').strip() or 'gpt-6-astra'
    version = os.getenv('CODEX_VERSION', '0.154.0')
    default_catalog = 'gpt56_sol_catalog.json' if model == 'gpt-5.6-sol' else 'gpt6_catalog_v8.json'
    catalog = json.loads(Path(os.getenv('CODEX_MODEL_CATALOG_JSON', ASSETS / default_catalog)).read_text())
    entries = [entry for entry in catalog['models'] if entry['slug'] == model]
    if len(entries) != 1 or entries[0].get('tool_mode') not in {None, 'direct'} or entries[0].get('use_responses_lite'):
        raise ValueError('Codex catalog must contain the selected model with classic tools (not code-mode/lite)')
    window = _positive('CODEX_CONTEXT_LIMIT', 262144)
    output = _positive('CODEX_MAX_OUTPUT_TOKENS', 64000)
    margin = _positive('CODEX_CONTEXT_TOKEN_MARGIN', 4096)
    compact_limit = _positive('CODEX_AUTO_COMPACT_TOKEN_LIMIT', window - output - margin)
    if compact_limit > window - output - margin:
        raise ValueError('Codex compact limit must reserve output tokens and context margin')
    entries[0].update(context_window=window, max_context_window=window, auto_compact_token_limit=compact_limit)
    binary_url = os.getenv('CODEX_BINARY_URL')
    checksum = os.getenv('CODEX_BINARY_SHA256')
    if bool(binary_url) != bool(checksum):
        raise ValueError('Provide both CODEX_BINARY_URL and CODEX_BINARY_SHA256')
    if checksum and (len(checksum) != 64 or any(c not in '0123456789abcdefABCDEF' for c in checksum)):
        raise ValueError('CODEX_BINARY_SHA256 must be a SHA-256 hex digest')
    config = OSWorldCodexOptions(
        user_prompt='',
        model=model,
        version=version,
        guide_text=Path(os.getenv('CODEX_GUIDE_FILE', ASSETS / 'guide_codex_linux.md')).read_text(),
        model_catalog_json=json.dumps(catalog, ensure_ascii=False),
        timeout=_positive('TASK_TIMEOUT', 5400),
        reasoning_effort=os.getenv('CODEX_REASONING_EFFORT', 'none'),
        auto_compact_token_limit=compact_limit,
        binary_url=binary_url,
        binary_checksum=checksum,
    ).model_dump(exclude={'user_prompt'})
    summary = os.getenv('CODEX_REASONING_SUMMARY', 'none')
    config['extra_config_toml'] = f'model_reasoning_summary = {json.dumps(summary)}\n' + config['extra_config_toml']
    return {'execution_profile': 'osworld', 'execution_options': config}
