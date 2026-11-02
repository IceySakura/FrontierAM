"""Runtime-side setup and execution for OpenClaw."""

from __future__ import annotations

import asyncio
import json
import logging
import math
import os
import re
import shlex
import shutil
import signal
import tempfile
import time

from contextlib import suppress
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal
from urllib.parse import urlparse

from nexus.extensions.agents.installed_common import (
    DEFAULT_NPM_REGISTRY,
    DEFAULT_PROXY_API_KEY,
    DEFAULT_RUNTIME_BIN_DIR,
    build_npm_env,
    build_runtime_path,
    classify_command_failure,
    find_free_port,
    find_nexus_runtime_binary,
    find_runtime_binary,
    resolve_npm_bin_entry,
    unpack_command_result,
)

from .installer import DEFAULT_INSTALL_DIR, MIN_NODE_VERSION, install_openclaw


logger = logging.getLogger(__name__)


DEFAULT_PROVIDER_ID = 'nexusproxy'
DEFAULT_AGENT_ID = 'main'
DUMMY_PROXY_API_KEY = DEFAULT_PROXY_API_KEY
DEFAULT_CONTEXT_WINDOW = 200_000
DEFAULT_MAX_TOKENS = 65_536
DEFAULT_ONBOARD_TIMEOUT = 900
DEFAULT_LLM_IDLE_TIMEOUT_MS = 600_000
# Releases before this one reject ``models.providers.<id>.timeoutSeconds`` outright, and a rejected key invalidates the whole config file rather than just that field.  Measured by loading the config on each release: 2026.4.25 still rejects it, 2026.4.26-beta.1 is the first to accept, and those two are adjacent in the registry.
_MIN_PROVIDER_TIMEOUT_SECONDS_VERSION = '2026.4.26-beta.1'
# First release whose ``agent.wait`` hangs.  Earlier ones let the CLI outlive their subagents on
# its own, and cancelling their waits truncates a still-running child instead.  The boundary sits
# inside the 2026.6.5 beta series: beta.1 and beta.2 do not hang, beta.3 onward does.
_MIN_SUBAGENT_WAIT_ABORT_VERSION = '2026.6.5-beta.3'
_OPENCLAW_VERSION_PATTERN = re.compile(r'v?(\d+(?:\.\d+)*)(?:-([0-9A-Za-z.-]+))?')
# sessions_spawn talks to a local gateway; without one every spawn fails with WS 1006.
DEFAULT_GATEWAY_AUTH_TOKEN = 'nexus-openclaw-gateway'
DEFAULT_GATEWAY_READY_TIMEOUT = 60
# Grace period between SIGTERM and SIGKILL of the gateway process group.
_GATEWAY_STOP_GRACE_SECONDS = 30
OpenClawExitStatus = Literal['completed', 'timeout', 'truncation', 'error']
_OPENCLAW_EXIT_GRACE_SECONDS = 30
NEXUS_BRAVE_API_KEY = 'nexus-serper-bridge'
BRAVE_PLUGIN_ID = 'brave'
BRAVE_PLUGIN_PACKAGE = '@openclaw/brave-plugin'
WEB_SEARCH_ENDPOINT_PATCH_MARKER = '/* nexus-serper-bridge-endpoint-patch-v1 */'
SERPER_BRIDGE_CHILD_ENV_VARS = (
    'SERPER_API_KEY',
    'SERPER_API_TYPE',
    'SEARCH_SERPER_URL',
    'SEARCH_SERPER_APP_ID',
    'SEARCH_SERPER_APP_KEY',
    'SERPER_KEY_ID',
)


@dataclass
class OpenClawResult:
    """Result of running OpenClaw in the runtime."""

    success: bool
    output: str
    stderr: str
    exit_code: int
    execution_time: float


@dataclass(frozen=True)
class WebSearchCompat:
    """Resolved OpenClaw web_search compatibility mode."""

    enabled: bool
    base_url: str | None
    uses_brave_base_url: bool


def build_openclaw_config(
    *,
    model: str,
    working_dir: str,
    proxy_port: int,
    provider_id: str = DEFAULT_PROVIDER_ID,
    context_window: int = DEFAULT_CONTEXT_WINDOW,
    max_tokens: int = DEFAULT_MAX_TOKENS,
    llm_idle_timeout_ms: int | float | str = DEFAULT_LLM_IDLE_TIMEOUT_MS,
    supports_images: bool = False,
) -> dict[str, Any]:
    """Build a minimal OpenClaw config bound to the local OpenAI proxy."""
    model_ref = f'{provider_id}/{model}'
    input_modalities = ['text', 'image'] if supports_images else ['text']
    resolved_llm_idle_timeout_seconds = _resolve_llm_idle_timeout_seconds(llm_idle_timeout_ms)
    return {
        'agents': {
            'defaults': {
                'workspace': working_dir,
                'sandbox': {'mode': 'off'},
                'model': {'primary': model_ref},
                'models': {
                    model_ref: {'alias': model},
                },
            },
        },
        'tools': {
            'exec': {
                'host': 'gateway',
                'security': 'full',
                'ask': 'off',
            },
        },
        'models': {
            'mode': 'replace',
            'providers': {
                provider_id: {
                    'baseUrl': f'http://127.0.0.1:{proxy_port}/v1',
                    'apiKey': DUMMY_PROXY_API_KEY,
                    'api': 'openai-completions',
                    'timeoutSeconds': resolved_llm_idle_timeout_seconds,
                    'models': [
                        {
                            'id': model,
                            'name': model,
                            'reasoning': True,
                            'input': input_modalities,
                            'contextWindow': context_window,
                            'maxTokens': max_tokens,
                        }
                    ],
                }
            },
        },
    }


def build_onboard_command(
    *,
    node_path: str,
    js_entry: str | Path,
    model: str,
    working_dir: str,
    proxy_port: int,
    provider_id: str = DEFAULT_PROVIDER_ID,
) -> list[str]:
    """Build the non-interactive onboarding command used to initialize local state."""
    return [
        node_path,
        str(js_entry),
        'onboard',
        '--non-interactive',
        '--accept-risk',
        '--mode',
        'local',
        '--auth-choice',
        'custom-api-key',
        '--custom-provider-id',
        provider_id,
        '--custom-compatibility',
        'openai',
        '--custom-base-url',
        f'http://127.0.0.1:{proxy_port}/v1',
        '--custom-model-id',
        model,
        '--custom-api-key',
        DUMMY_PROXY_API_KEY,
        '--skip-channels',
        '--skip-health',
        '--skip-search',
        '--skip-skills',
        '--skip-ui',
        '--skip-daemon',
        '--workspace',
        working_dir,
        '--json',
    ]


def build_agent_command(
    *,
    node_path: str,
    js_entry: str | Path,
    task: str,
    timeout: int,
    agent_id: str = DEFAULT_AGENT_ID,
) -> list[str]:
    """Build the OpenClaw CLI command for a single embedded agent turn."""
    return [
        node_path,
        str(js_entry),
        'agent',
        '--agent',
        agent_id,
        '--local',
        '--json',
        '--timeout',
        str(timeout),
        '--message',
        task,
    ]


def _resolve_onboard_timeout(*, agent_timeout: float, onboard_timeout: float | None) -> int:
    cli_timeout = max(1, math.ceil(agent_timeout))
    configured_timeout = DEFAULT_ONBOARD_TIMEOUT if onboard_timeout is None else onboard_timeout
    return max(1, min(cli_timeout, math.ceil(configured_timeout)))


def _merge_no_proxy(*values: str | None) -> str:
    """Merge comma-separated no_proxy values while preserving order."""
    merged: list[str] = []
    seen: set[str] = set()
    for value in values:
        for entry in (value or '').split(','):
            normalized = entry.strip()
            if normalized and normalized not in seen:
                seen.add(normalized)
                merged.append(normalized)
    return ','.join(merged)


def _local_bypass_entries(url: str | None) -> list[str]:
    if not url:
        return []
    parsed = urlparse(url)
    host = parsed.hostname
    port = parsed.port
    if not host:
        return []
    entries = [host]
    if port is not None:
        entries.append(f'{host}:{port}')
        if host in {'127.0.0.1', 'localhost'}:
            alternate = 'localhost' if host == '127.0.0.1' else '127.0.0.1'
            entries.append(f'{alternate}:{port}')
    return entries


def _build_openclaw_env(
    *,
    home_dir: Path,
    state_dir: Path,
    config_path: Path,
    proxy_port: int,
    web_search: WebSearchCompat,
    curl_binary: str | None,
    base_env: dict[str, str] | None = None,
) -> dict[str, str]:
    """Build the OpenClaw child process env while masking real Serper credentials."""
    source_env = os.environ if base_env is None else base_env
    no_proxy = _merge_no_proxy(
        source_env.get('no_proxy'),
        source_env.get('NO_PROXY'),
        'localhost',
        '127.0.0.1',
        f'localhost:{proxy_port}',
        f'127.0.0.1:{proxy_port}',
        *_local_bypass_entries(web_search.base_url),
    )

    env = {
        'HOME': str(home_dir),
        'OPENCLAW_HOME': str(home_dir),
        'OPENCLAW_STATE_DIR': str(state_dir),
        'OPENCLAW_CONFIG_PATH': str(config_path),
        'PATH': build_runtime_path(),
        'CI': '1',
        'NO_COLOR': '1',
        'no_proxy': no_proxy,
        'NO_PROXY': no_proxy,
        'BRAVE_API_KEY': NEXUS_BRAVE_API_KEY if web_search.enabled else '',
    }
    env.update(dict.fromkeys(SERPER_BRIDGE_CHILD_ENV_VARS, ''))

    if web_search.enabled and web_search.base_url:
        env['NEXUS_SERPER_BRIDGE_BASE_URL'] = web_search.base_url
    else:
        env['NEXUS_SERPER_BRIDGE_BASE_URL'] = ''

    if curl_binary:
        env['NEXUS_CURL_BINARY'] = curl_binary
        env['PATH'] = build_runtime_path(Path(curl_binary).parent)

    return env


def _build_openclaw_plugin_install_env(
    *,
    openclaw_env: dict[str, str],
    npm_registry: str | None,
) -> dict[str, str]:
    """Build env for OpenClaw plugin install commands."""
    env = openclaw_env.copy()
    env.update(build_npm_env(npm_registry=npm_registry))
    # Keep plugin installs usable in runtime images without the host/corporate
    # CA bundle. This mirrors OpenClaw's npm install environment.
    env['NPM_CONFIG_STRICT_SSL'] = 'false'
    env['npm_config_strict_ssl'] = 'false'
    return env


async def _try_install_brave_plugin_for_openclaw(
    *,
    node_path: str,
    js_entry: str | Path,
    working_dir: str,
    openclaw_env: dict[str, str],
    npm_registry: str | None,
    timeout: float,
) -> None:
    """Best-effort install/enable of the Brave plugin for plugin-schema OpenClaw builds."""
    from nexus.runtime.executors.command import CommandRunStatus, RunCommandRequest, run_command

    plugin_env = _build_openclaw_plugin_install_env(openclaw_env=openclaw_env, npm_registry=npm_registry)
    plugin_timeout = max(1, min(math.ceil(timeout), 300))

    commands = (
        ('install', [node_path, str(js_entry), 'plugins', 'install', BRAVE_PLUGIN_PACKAGE]),
        ('enable', [node_path, str(js_entry), 'plugins', 'enable', BRAVE_PLUGIN_ID]),
    )
    for action, args in commands:
        command = shlex.join(args)
        result = await run_command(
            RunCommandRequest(
                command=command,
                cwd=working_dir,
                timeout=plugin_timeout,
                extra_env=plugin_env,
            )
        )
        if result.status == CommandRunStatus.TimeLimitExceeded:
            logger.warning('OpenClaw Brave plugin %s timed out: %s', action, command)
            continue
        if result.status != CommandRunStatus.Finished:
            logger.warning('OpenClaw Brave plugin %s command failed: %s', action, result.stderr or command)
            continue
        if (result.return_code or 0) != 0:
            logger.warning(
                'OpenClaw Brave plugin %s returned %s: %s',
                action,
                result.return_code,
                result.stderr or result.stdout or command,
            )


def _has_ready_brave_web_search_provider(payload: Any) -> bool:
    """Return true when OpenClaw reports the Brave web_search provider is usable."""
    if not isinstance(payload, dict):
        return False
    search_providers = payload.get('search')
    if not isinstance(search_providers, list):
        return False

    for entry in search_providers:
        if not isinstance(entry, dict):
            continue
        if entry.get('id') != 'brave':
            continue
        if entry.get('available') is not True or entry.get('configured') is not True:
            return False
        return entry.get('selected', True) is not False
    return False


async def _verify_brave_web_search_provider(
    *,
    node_path: str,
    js_entry: str | Path,
    working_dir: str,
    openclaw_env: dict[str, str],
    timeout: float,
) -> bool:
    """Ask OpenClaw whether the Brave web_search provider is available and configured."""
    from nexus.runtime.executors.command import CommandRunStatus, RunCommandRequest, run_command

    command = shlex.join([node_path, str(js_entry), 'infer', 'web', 'providers', '--json'])
    result = await run_command(
        RunCommandRequest(
            command=command,
            cwd=working_dir,
            timeout=max(1, min(math.ceil(timeout), 120)),
            extra_env=openclaw_env,
        )
    )
    if result.status == CommandRunStatus.TimeLimitExceeded:
        logger.warning('OpenClaw web_search provider preflight timed out: %s', command)
        return False
    if result.status != CommandRunStatus.Finished:
        logger.warning('OpenClaw web_search provider preflight command failed: %s', result.stderr or command)
        return False
    if (result.return_code or 0) != 0:
        logger.warning(
            'OpenClaw web_search provider preflight returned %s: %s',
            result.return_code,
            result.stderr or result.stdout or command,
        )
        return False

    try:
        payload = json.loads(result.stdout or '{}')
    except Exception as exc:
        logger.warning('OpenClaw web_search provider preflight returned invalid JSON: %s', exc)
        return False
    return _has_ready_brave_web_search_provider(payload)


def parse_openclaw_response(stdout: str) -> dict[str, Any] | None:
    """Parse the JSON payload emitted by ``openclaw agent --json``."""
    payload = stdout.strip()
    if not payload:
        return None

    try:
        parsed = json.loads(payload)
    except json.JSONDecodeError:
        parsed = None
    if isinstance(parsed, dict):
        return parsed

    decoder = json.JSONDecoder()
    candidates: list[dict[str, Any]] = []
    for index, character in enumerate(payload):
        if character != '{':
            continue
        try:
            parsed, _end = decoder.raw_decode(payload[index:])
        except json.JSONDecodeError:
            continue
        if isinstance(parsed, dict):
            candidates.append(parsed)

    envelopes = [item for item in candidates if 'payloads' in item or 'meta' in item]
    return envelopes[-1] if envelopes else (candidates[-1] if candidates else None)


def visible_openclaw_payload_texts(payload: Any) -> list[str]:
    """Return only visible, non-error text from an OpenClaw ReplyPayload."""
    if not isinstance(payload, dict):
        return []
    payloads = payload.get('payloads')
    if not isinstance(payloads, list):
        return []
    return [
        item['text']
        for item in payloads
        if (
            isinstance(item, dict)
            and item.get('isReasoning') is not True
            and item.get('isError') is not True
            and isinstance(item.get('text'), str)
            and item['text'].strip()
        )
    ]


def classify_openclaw_response(
    stdout: str,
    exit_code: int,
) -> tuple[OpenClawExitStatus, str | None]:
    """Classify the embedded ReplyPayload instead of trusting a zero process exit alone."""
    if exit_code != 0:
        return 'error', None

    payload = parse_openclaw_response(stdout)
    if payload is None:
        return 'error', 'OpenClaw returned no valid ReplyPayload'

    meta = payload.get('meta')
    if isinstance(meta, dict):
        if meta.get('aborted') is True:
            return 'error', 'OpenClaw reported an aborted embedded run'

        raw_stop_reason = meta.get('stopReason', meta.get('stop_reason'))
        stop_reason = str(raw_stop_reason or '').strip().lower().replace('-', '_')
        if stop_reason in {'timeout', 'timed_out', 'time_limit_exceeded'}:
            return 'timeout', f'OpenClaw stopped with reason {raw_stop_reason!r}'
        if stop_reason in {'context_overflow', 'context_length_exceeded', 'max_tokens', 'length'}:
            return 'truncation', f'OpenClaw stopped with reason {raw_stop_reason!r}'
        if stop_reason in {'error', 'aborted', 'blocked', 'abandoned', 'cancelled', 'canceled', 'tool_calls'}:
            return 'error', f'OpenClaw stopped with reason {raw_stop_reason!r}'

        raw_error = meta.get('error')
        if raw_error:
            error_text = json.dumps(raw_error, ensure_ascii=False) if isinstance(raw_error, dict) else str(raw_error)
            normalized_error = error_text.lower().replace('-', '_')
            if 'context_overflow' in normalized_error or 'context_length_exceeded' in normalized_error:
                return 'truncation', f'OpenClaw reported an error: {error_text}'
            if 'timeout' in normalized_error or 'timed_out' in normalized_error:
                return 'timeout', f'OpenClaw reported an error: {error_text}'
            return 'error', f'OpenClaw reported an error: {error_text}'

        raw_liveness = meta.get('livenessState', meta.get('liveness_state'))
        liveness = str(raw_liveness or '').strip().lower().replace('-', '_')
        if liveness in {'error', 'failed', 'dead', 'blocked', 'aborted', 'abandoned', 'timeout', 'timed_out'}:
            status: OpenClawExitStatus = 'timeout' if 'timeout' in liveness else 'error'
            return status, f'OpenClaw liveness state is {raw_liveness!r}'

    if not isinstance(payload.get('payloads'), list):
        return 'error', 'OpenClaw ReplyPayload has no payloads list'
    if not visible_openclaw_payload_texts(payload):
        return 'error', 'OpenClaw returned no visible non-error reply'
    return 'completed', None


def is_openclaw_success(stdout: str, exit_code: int) -> bool:
    """Backward-compatible boolean view of :func:`classify_openclaw_response`."""
    status, _failure_message = classify_openclaw_response(stdout, exit_code)
    return status == 'completed'


def _resolve_llm_idle_timeout_ms(value: int | float | str) -> int:
    if isinstance(value, bool):
        raise ValueError(f'llm_idle_timeout_ms must be a positive number of milliseconds, got {value!r}')
    try:
        timeout_ms = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f'llm_idle_timeout_ms must be a positive number of milliseconds, got {value!r}') from exc

    if not math.isfinite(timeout_ms) or timeout_ms <= 0:
        raise ValueError(f'llm_idle_timeout_ms must be a positive number of milliseconds, got {value!r}')
    return max(1, math.ceil(timeout_ms))


def _resolve_llm_idle_timeout_seconds(value: int | float | str) -> int:
    return max(1, math.ceil(_resolve_llm_idle_timeout_ms(value) / 1000))


def _get_openclaw_proxy_class():
    """OpenClaw talks OpenAI-completions; thinking-id re-anchor lives on the host cache.

    OpenClaw strips ``_`` from provider tool ids when it persists history. Guessing the
    original spelling with a per-vendor regex does not scale. ``LLMRequestHandler``
    remembers the host-issued id and matches/rewrites the stripped form instead.
    """
    from nexus.runtime.server.services.openai_proxy import OpenAIProxy

    return OpenAIProxy


def _openclaw_version_key(version: str) -> tuple[Any, ...] | None:
    """Order an OpenClaw version the way npm does, or return None if it does not parse.

    Pre-releases sort below their own release and against each other by identifier (``2026.6.5-beta.1`` < ``2026.6.5-beta.2`` < ``2026.6.5``).  That precision matters because one behaviour gated below changes part-way through a beta series.
    """
    match = _OPENCLAW_VERSION_PATTERN.fullmatch(version.strip())
    if not match:
        return None
    release = tuple(int(part) for part in match.group(1).split('.'))
    suffix = match.group(2)
    if suffix is None:
        return (release, 1, ())
    # Numeric identifiers compare numerically and rank below alphanumeric ones, per semver.
    identifiers = tuple((0, int(part), '') if part.isdigit() else (1, 0, part) for part in suffix.split('.'))
    return (release, 0, identifiers)


def _openclaw_version_at_least(openclaw_version: str | None, minimum: str) -> bool:
    """Compare an installed version against a threshold, defaulting to True when unknown.

    Both a missing version and an unparsable one answer True: every gate here gates a fix,
    and withholding a fix from a current release is worse than applying it to an old one.
    """
    if not openclaw_version:
        return True
    key = _openclaw_version_key(openclaw_version)
    if key is None:
        return True
    threshold = _openclaw_version_key(minimum)
    assert threshold is not None, f'unparsable version threshold {minimum!r}'
    return key >= threshold


def _supports_provider_timeout_seconds(openclaw_version: str | None) -> bool:
    """Whether this release accepts ``models.providers.<id>.timeoutSeconds``.

    Everything up to 2026.4.25 rejects the key with ``Unrecognized key: "timeoutSeconds"`` and refuses to load the config at all, so the gateway never starts.
    """
    return _openclaw_version_at_least(openclaw_version, _MIN_PROVIDER_TIMEOUT_SECONDS_VERSION)


def _patch_model_config(
    *,
    config_path: Path,
    provider_id: str,
    model: str,
    context_window: int,
    max_tokens: int,
    llm_idle_timeout_ms: int | float | str = DEFAULT_LLM_IDLE_TIMEOUT_MS,
    openclaw_version: str | None = None,
) -> None:
    """Patch model limits and provider request timeout in OpenClaw's config file.

    After ``openclaw onboard`` runs, custom-provider models get conservative
    defaults (often 16 K context).  This function reads the generated config,
    updates the matching model entry, and writes it back so the agent runs with
    the correct limits.  OpenClaw 2026.5+ validates model entries strictly and
    derives the internal ``requestTimeoutMs`` from the provider-level
    ``timeoutSeconds`` field; older releases have no equivalent knob and reject
    the field, so ``openclaw_version`` gates whether it is written.
    """
    resolved_llm_idle_timeout_seconds = _resolve_llm_idle_timeout_seconds(llm_idle_timeout_ms)
    if not config_path.exists():
        logger.warning('OpenClaw config not found at %s; skipping model patch', config_path)
        return

    try:
        cfg = json.loads(config_path.read_text(encoding='utf-8'))
    except Exception as exc:
        logger.warning('Failed to read OpenClaw config for patching: %s', exc)
        return

    providers = cfg.get('models', {}).get('providers', {})
    provider = providers.get(provider_id)
    if not isinstance(provider, dict):
        logger.warning('Provider %s not found in config %s; skipping model patch', provider_id, config_path)
        return

    changed = False
    timeout_supported = _supports_provider_timeout_seconds(openclaw_version)
    if timeout_supported:
        if provider.get('timeoutSeconds') != resolved_llm_idle_timeout_seconds:
            provider['timeoutSeconds'] = resolved_llm_idle_timeout_seconds
            changed = True
    else:
        if 'timeoutSeconds' in provider:
            del provider['timeoutSeconds']
            changed = True
        logger.warning(
            'OpenClaw %s rejects provider timeoutSeconds; leaving the LLM idle timeout at its built-in default',
            openclaw_version,
        )

    patched_model = False
    model_entries = provider.get('models', [])
    if not isinstance(model_entries, list):
        model_entries = []
    for entry in model_entries:
        if not isinstance(entry, dict):
            continue
        if 'requestTimeoutMs' in entry:
            entry.pop('requestTimeoutMs', None)
            changed = True
        if entry.get('id') == model or entry.get('name') == model:
            entry['contextWindow'] = context_window
            entry['maxTokens'] = max_tokens
            patched_model = True
            changed = True

    if changed:
        config_path.write_text(json.dumps(cfg, indent=2), encoding='utf-8')
        logger.debug(
            'Patched OpenClaw provider config: %s/%s  contextWindow=%d  maxTokens=%d  timeoutSeconds=%s',
            provider_id,
            model,
            context_window,
            max_tokens,
            resolved_llm_idle_timeout_seconds if timeout_supported else 'unsupported',
        )

    if not patched_model:
        logger.warning(
            'Model %s/%s not found in config %s; patched provider timeout only',
            provider_id,
            model,
            config_path,
        )


def _patch_full_config(
    *,
    config_path: Path,
    provider_id: str,
    model: str,
    supports_images: bool = False,
    web_search_enabled: bool = False,
    web_search_base_url: str | None = None,
    web_search_uses_brave_base_url: bool = False,
) -> None:
    """Apply the full set of config patches that ``openclaw onboard`` does not set.

    This aligns the Nexus runtime behaviour with the original WildClawBench
    ``run_batch.py``, which:

    1. Sets ``tools.exec.security = "full"`` and ``tools.exec.ask = "off"``
       so the agent can run arbitrary commands without confirmation.
    2. Sets ``agents.defaults.sandbox.mode = "off"`` (no sandbox).
    3. Sets ``agents.defaults.imageModel.primary`` to the current model so
       the ``image`` tool is available (when the model supports images).
    4. Updates input modalities (e.g. image support) for the model entry.
    5. Disables SSRF private-network blocking for ``browser`` and
       ``web_fetch`` so that DNS responses pointing to CDN / anycast
       addresses are not incorrectly rejected.
    """
    if not config_path.exists():
        logger.warning('OpenClaw config not found at %s; skipping full patch', config_path)
        return

    try:
        cfg = json.loads(config_path.read_text(encoding='utf-8'))
    except Exception as exc:
        logger.warning('Failed to read OpenClaw config for full patching: %s', exc)
        return

    model_ref = f'{provider_id}/{model}'

    # 1. tools.exec — unrestricted command execution
    tools_cfg = cfg.setdefault('tools', {})
    exec_cfg = tools_cfg.setdefault('exec', {})
    exec_cfg['host'] = 'gateway'
    exec_cfg['security'] = 'full'
    exec_cfg['ask'] = 'off'

    # 2. agents.defaults.sandbox — off
    agents_cfg = cfg.setdefault('agents', {})
    defaults_cfg = agents_cfg.setdefault('defaults', {})
    defaults_cfg.setdefault('sandbox', {})['mode'] = 'off'

    # 3. agents.defaults.imageModel — enable the image tool
    #    Only set imageModel when the model actually supports images;
    #    otherwise OpenClaw logs a misleading "Model does not support
    #    images" warning every time the image tool is invoked.
    if supports_images:
        defaults_cfg.setdefault('imageModel', {})['primary'] = model_ref

    # 4. Mark model as reasoning-capable + update input modalities
    providers = cfg.get('models', {}).get('providers', {})
    provider = providers.get(provider_id, {})
    model_entries = provider.get('models', []) if isinstance(provider, dict) else []
    if not isinstance(model_entries, list):
        model_entries = []
    for entry in model_entries:
        if not isinstance(entry, dict):
            continue
        if (entry.get('id') == model or entry.get('name') == model) and supports_images:
            entry['input'] = ['text', 'image']

    # 5. browser.ssrfPolicy — allow private-network IPs
    #    The Docker runtime DNS may resolve public domains (e.g.
    #    zh.wikipedia.org, www.google.com) to CDN / anycast addresses
    #    that OpenClaw's SSRF checker classifies as "special-use".
    #    Disabling this check is safe because the container already
    #    runs in a controlled Docker bridge network.
    #
    #    NOTE: OpenClaw's config schema does NOT support an
    #    ``allowedDomains`` key under ``ssrfPolicy`` — adding it will
    #    cause ``Config invalid`` and the process won't start.  The
    #    JS-level patches (_patch_web_fetch_ssrf, _patch_web_fetch_ua)
    #    handle the web_fetch code path instead.
    browser_cfg = cfg.setdefault('browser', {})
    ssrf_policy = browser_cfg.setdefault('ssrfPolicy', {})
    ssrf_policy['dangerouslyAllowPrivateNetwork'] = True

    # 6. tools.web.search — enable only when the local Serper bridge is
    #    actually available. OpenClaw still sees a Brave provider, but all
    #    traffic goes to the Nexus bridge on 127.0.0.1.
    web_cfg = tools_cfg.setdefault('web', {})
    search_cfg = web_cfg.setdefault('search', {})
    plugins_cfg = cfg.setdefault('plugins', {})
    plugin_entries = plugins_cfg.setdefault('entries', {})

    if web_search_enabled:
        search_cfg['enabled'] = True
        search_cfg['provider'] = 'brave'

        if web_search_uses_brave_base_url and web_search_base_url:
            search_cfg.pop('apiKey', None)
            search_cfg.pop('brave', None)
            brave_entry = plugin_entries.setdefault('brave', {})
            brave_entry['enabled'] = True
            brave_web_search_cfg = brave_entry.setdefault('config', {}).setdefault('webSearch', {})
            brave_web_search_cfg['apiKey'] = NEXUS_BRAVE_API_KEY
            brave_web_search_cfg['baseUrl'] = web_search_base_url.rstrip('/')
            brave_web_search_cfg['mode'] = 'web'
        else:
            search_cfg['apiKey'] = NEXUS_BRAVE_API_KEY
            search_cfg.pop('brave', None)
            plugin_entries.pop('brave', None)
    else:
        search_cfg['enabled'] = False
        search_cfg.pop('provider', None)
        search_cfg.pop('apiKey', None)
        search_cfg.pop('brave', None)
        plugin_entries.pop('brave', None)

    if not plugin_entries:
        plugins_cfg.pop('entries', None)
    if not plugins_cfg:
        cfg.pop('plugins', None)

    config_path.write_text(json.dumps(cfg, indent=2), encoding='utf-8')
    logger.debug(
        'Applied full config patch: exec=full, sandbox=off, imageModel=%s, ssrfPolicy=allowPrivate, supports_images=%s',
        model_ref if supports_images else '(not set)',
        supports_images,
    )


def _patch_gateway_config(config_path: Path, *, port: int) -> None:
    """Enable a local loopback gateway so sessions_spawn can attach to it."""
    if not config_path.exists():
        logger.warning('OpenClaw config not found at %s; skipping gateway patch', config_path)
        return

    try:
        cfg = json.loads(config_path.read_text(encoding='utf-8'))
    except Exception as exc:
        logger.warning('Failed to read OpenClaw config for gateway patching: %s', exc)
        return

    gateway = cfg.setdefault('gateway', {})
    gateway['mode'] = 'local'
    gateway['port'] = int(port)
    gateway['bind'] = 'loopback'
    auth = gateway.setdefault('auth', {})
    auth['mode'] = 'token'
    auth['token'] = DEFAULT_GATEWAY_AUTH_TOKEN
    config_path.write_text(json.dumps(cfg, indent=2), encoding='utf-8')
    logger.info('Patched OpenClaw gateway config: mode=local bind=loopback port=%s', port)


async def _wait_for_tcp_port(host: str, port: int, *, timeout: float) -> None:
    """Block until a TCP accept succeeds on ``host:port``, or raise TimeoutError."""
    deadline = time.monotonic() + max(0.1, timeout)
    last_error: Exception | None = None
    while time.monotonic() < deadline:
        try:
            reader, writer = await asyncio.open_connection(host, port)
        except OSError as exc:
            last_error = exc
            await asyncio.sleep(0.2)
            continue
        writer.close()
        with suppress(Exception):
            await writer.wait_closed()
        return
    raise TimeoutError(f'Timed out waiting for {host}:{port}') from last_error


async def _start_openclaw_gateway(
    *,
    node_path: str,
    js_entry: str | Path,
    working_dir: str,
    openclaw_env: dict[str, str],
    port: int,
    state_dir: Path,
    ready_timeout: float = DEFAULT_GATEWAY_READY_TIMEOUT,
) -> asyncio.subprocess.Process:
    """Start ``openclaw gateway`` in the background and wait until it accepts connections."""
    from nexus.utils.shell import get_os_env_without_ld_library_path

    log_path = state_dir / 'gateway.log'
    log_file = open(log_path, 'wb')  # noqa: SIM115 — fd is inherited by the gateway process
    env = {
        **get_os_env_without_ld_library_path(),
        **openclaw_env,
        'OPENCLAW_GATEWAY_TOKEN': DEFAULT_GATEWAY_AUTH_TOKEN,
        'OPENCLAW_GATEWAY_PORT': str(port),
    }
    cmd = [
        node_path,
        str(js_entry),
        'gateway',
        '--port',
        str(port),
        '--bind',
        'loopback',
        '--token',
        DEFAULT_GATEWAY_AUTH_TOKEN,
    ]
    logger.info('Starting OpenClaw gateway on 127.0.0.1:%s', port)
    try:
        proc = await asyncio.create_subprocess_exec(
            *cmd,
            stdout=log_file,
            stderr=asyncio.subprocess.STDOUT,
            cwd=working_dir,
            env=env,
            start_new_session=True,
        )
    except Exception:
        log_file.close()
        raise
    else:
        # Close our handle; the child keeps the inherited write end.
        log_file.close()

    try:
        await _wait_for_tcp_port('127.0.0.1', port, timeout=ready_timeout)
    except Exception as exc:
        await _stop_openclaw_gateway(proc)
        raise RuntimeError(
            f'OpenClaw gateway failed to become ready on 127.0.0.1:{port} within {ready_timeout}s '
            f'(see {log_path}): {exc}'
        ) from exc

    logger.info('OpenClaw gateway ready on 127.0.0.1:%s (pid=%s)', port, proc.pid)
    return proc


def _gateway_process_group(proc: asyncio.subprocess.Process) -> int | None:
    """Return the gateway's process group id, or None when it cannot be signalled as a group.

    ``start_new_session=True`` at spawn makes the gateway a session leader, so its group id is
    its own pid and no other process shares the group.  Any other pgid means that isolation did
    not take effect, and signalling the group would reach unrelated processes -- including the
    Nexus runtime itself -- so the caller falls back to the leader alone.
    """
    try:
        pgid = os.getpgid(proc.pid)
    except ProcessLookupError:
        return None
    if pgid != proc.pid:
        logger.warning(
            'OpenClaw gateway pid %s is not its own process group leader (pgid=%s); '
            'signalling the leader only, descendants may survive',
            proc.pid,
            pgid,
        )
        return None
    return pgid


def _signal_openclaw_gateway(proc: asyncio.subprocess.Process, sig: int) -> None:
    """Send *sig* to the whole gateway process group, or to the leader when that is not safe."""
    if os.name != 'posix':  # pragma: no cover - Nexus runtimes are Linux, but keep dev machines safe.
        with suppress(ProcessLookupError):
            if sig == signal.SIGKILL:
                proc.kill()
            else:
                proc.terminate()
        return
    pgid = _gateway_process_group(proc)
    with suppress(ProcessLookupError):
        if pgid is None:
            os.kill(proc.pid, sig)
        else:
            os.killpg(pgid, sig)


async def _stop_openclaw_gateway(proc: asyncio.subprocess.Process | None) -> None:
    """Terminate the gateway and everything it spawned.

    The gateway hosts the ``exec`` tool (``tools.exec.host = "gateway"``), so every command the
    parent or a subagent runs is one of its descendants, and long ones are backgrounded with a
    pid the agent is told to poll later.  Signalling only the leader leaves those behind to
    outlive the task, so the whole process group goes down together.
    """
    if proc is None or proc.returncode is not None:
        return
    _signal_openclaw_gateway(proc, signal.SIGTERM)
    try:
        await asyncio.wait_for(proc.wait(), timeout=_GATEWAY_STOP_GRACE_SECONDS)
        return
    except TimeoutError:
        logger.warning(
            'OpenClaw gateway did not exit within %ss of SIGTERM; killing its process group',
            _GATEWAY_STOP_GRACE_SECONDS,
        )
    _signal_openclaw_gateway(proc, signal.SIGKILL)
    with suppress(Exception):
        await proc.wait()


def _find_openclaw_dist_dir(openclaw_path: Path) -> Path | None:
    """Locate the ``dist/`` directory of the OpenClaw package.

    The ``openclaw_path`` may be:
      a) /tmp/openclaw/<ver>/node_modules/.bin/openclaw  (symlink)
      b) /usr/bin/openclaw  (symlink)
      c) /usr/lib/node_modules/openclaw/bin/openclaw.mjs

    Returns the ``dist/`` Path or None if not found.
    """
    real = openclaw_path.resolve()

    for ancestor in [real.parent, real.parent.parent, real.parent.parent.parent]:
        candidate = ancestor / 'dist'
        if candidate.is_dir():
            return candidate

    for probe in (
        Path('/usr/lib/node_modules/openclaw/dist'),
        Path('/usr/local/lib/node_modules/openclaw/dist'),
    ):
        if probe.is_dir():
            return probe

    return None


def _read_openclaw_package_version(openclaw_path: Path) -> str | None:
    """Read the installed OpenClaw version from its ``package.json``.

    The requested version is often ``latest``, so the installed package is the only
    authoritative source for the feature gates that depend on it.
    """
    dist_dir = _find_openclaw_dist_dir(openclaw_path)
    if dist_dir is None:
        logger.warning(
            'OpenClaw dist directory not found from %s; version-gated patches will assume latest', openclaw_path
        )
        return None
    package_json = dist_dir.parent / 'package.json'
    try:
        version = json.loads(package_json.read_text(encoding='utf-8')).get('version')
    except (OSError, json.JSONDecodeError, AttributeError) as exc:
        logger.warning('Failed to read OpenClaw version from %s: %s', package_json, exc)
        return None
    if not isinstance(version, str) or not version.strip():
        logger.warning('OpenClaw package.json at %s has no usable version field', package_json)
        return None
    return version.strip()


def _get_js_candidates(dist_dir: Path) -> list[str]:
    """Return all JS file paths in the dist directory (and plugin-sdk/)."""
    import glob

    return glob.glob(str(dist_dir / '*.js')) + glob.glob(str(dist_dir / 'plugin-sdk' / '*.js'))


# Prepended to every bundle touched by ``_patch_subagent_wait_abort``.  The state lives on ``globalThis`` because the wait is issued from one module and cancelled from another.  Re-signalling a run also drops its previous wait, so the retry and resume paths cannot leave orphaned polls behind.
_SUBAGENT_WAIT_ABORT_HELPERS = """// nexus: cancellation plumbing for sub-agent gateway waits
globalThis.__nexusAgentWaitAborts ??= new Map();
globalThis.__nexusWaitSignal ??= function (runId) {
\tglobalThis.__nexusAbortWait(runId);
\tconst controller = new AbortController();
\tglobalThis.__nexusAgentWaitAborts.set(runId, controller);
\treturn controller.signal;
};
globalThis.__nexusAbortWait ??= function (runId) {
\tconst controller = globalThis.__nexusAgentWaitAborts.get(runId);
\tif (!controller) return;
\tglobalThis.__nexusAgentWaitAborts.delete(runId);
\ttry {
\t\tcontroller.abort();
\t} catch {}
};
"""

# Bundle filenames are content-hashed, so each module is located by a marker.  The registry's completion path was reshaped twice between 2026.6 and 2026.9 -- an inner `try` block in 2026.7, then a free function in 2026.8 -- moving the enclosing declaration and its indentation each time.  Anchoring on the one statement that survived all three shapes keeps a single marker valid across the range; it is unique across dist/ in every release from 2026.6.5-beta.3 to 2026.9.2, and carries no leading whitespace so re-indentation cannot break it.
_SUBAGENT_WAIT_ABORT_MARKERS = {
    'run_wait': ('async function waitForAgentRun(params) {',),
    'registry': ('params.clearPendingLifecycleError(completeParams.runId);',),
}

_SUBAGENT_WAIT_ABORT_EDITS = (
    # Give every agent.wait a cancellable signal, keyed by the run it belongs to.
    (
        'run_wait',
        '\t\t\tmethod: "agent.wait",\n',
        '\t\t\tmethod: "agent.wait",\n\t\t\tsignal: params.signal ?? globalThis.__nexusWaitSignal(params.runId),\n',
    ),
    # A deliberate abort must not read as a recoverable failure, or the registry immediately schedules a replacement wait.
    (
        'run_wait',
        '\t} catch (err) {\n\t\tconst error = formatErrorMessage(err);',
        '\t} catch (err) {\n\t\tif (err && err.name === "AbortError") return { status: "pending" };\n'
        '\t\tconst error = formatErrorMessage(err);',
    ),
    # Drop the wait the moment the run reaches a terminal state.  ``None`` reuses the module's
    # own marker as the anchor, so the abort is injected at the head of the completion path
    # without repeating the marker string here.
    ('registry', None, '\n\tglobalThis.__nexusAbortWait(completeParams.runId);'),
)


def _needs_subagent_wait_abort(openclaw_version: str | None) -> bool:
    """Whether this release needs the ``agent.wait`` cancellation patch.

    From 2026.6.5-beta.3 on, the CLI sits on the long-poll until the outer budget kills it.
    Everything earlier that was measured -- 2026.3.11 through 2026.6.5-beta.2 -- takes the
    same edit cleanly but is harmed by it: those CLIs already outlive their subagents, and
    cancelling the wait cuts a still-running child off mid-task.
    """
    return _openclaw_version_at_least(openclaw_version, _MIN_SUBAGENT_WAIT_ABORT_VERSION)


def _patch_subagent_wait_abort(openclaw_path: Path, openclaw_version: str | None = None) -> None:
    """Stop an accepted ``sessions_spawn`` from pinning the CLI process open.

    ``registerSubagentRun`` issues an ``agent.wait`` long-poll whose timeout
    defaults to "no timeout" (~24.8 days) and passes no ``AbortSignal``.  The
    gateway answers that poll only from a live lifecycle event or a cached
    terminal snapshot, so once both are gone the request never settles, its
    WebSocket stays open, and the CLI's event loop never drains — the reply
    payload is already on stdout but the process cannot exit, and the outer
    budget eventually kills it.

    The registry separately learns of completion through the lifecycle events
    ``ensureListener`` subscribes to, so the missing piece is simply tearing the
    poll down once a run reaches a terminal state.
    """
    if not _needs_subagent_wait_abort(openclaw_version):
        logger.info('OpenClaw %s does not hang on agent.wait; skipping subagent wait abort patch', openclaw_version)
        return

    dist_dir = _find_openclaw_dist_dir(openclaw_path)
    if dist_dir is None:
        logger.warning('OpenClaw dist/ not found; skipping subagent wait abort patch')
        return

    modules: dict[str, Path] = {}
    sources: dict[str, str] = {}
    matched_marker: dict[str, str] = {}
    for js_path in _get_js_candidates(dist_dir):
        path = Path(js_path)
        try:
            source = path.read_text(encoding='utf-8')
        except Exception as exc:
            # Swallowing this silently would surface only as the generic "modules not found"
            # warning below, with no hint that the bundle was present but unreadable.
            logger.warning('OpenClaw bundle %s unreadable during marker scan: %s', path, exc)
            continue
        for key, markers in _SUBAGENT_WAIT_ABORT_MARKERS.items():
            if key in modules:
                continue
            for marker in markers:
                if marker in source:
                    modules[key] = path
                    sources[key] = source
                    matched_marker[key] = marker
                    break

    missing = sorted(set(_SUBAGENT_WAIT_ABORT_MARKERS) - set(modules))
    if missing:
        logger.warning('OpenClaw modules %s not found; skipping subagent wait abort patch', missing)
        return
    if '__nexusAgentWaitAborts' in sources['run_wait']:
        return

    for key in modules:
        sources[key] = _SUBAGENT_WAIT_ABORT_HELPERS + sources[key]
    for key, old, new in _SUBAGENT_WAIT_ABORT_EDITS:
        # ``old is None`` means "append after whichever marker matched", which is how
        # one edit spans registry layouts that share no other stable anchor.
        anchor = old if old is not None else matched_marker[key]
        replacement = new if old is not None else anchor + new
        if anchor not in sources[key]:
            logger.warning('OpenClaw anchor missing in %s; skipping subagent wait abort patch', modules[key].name)
            return
        sources[key] = sources[key].replace(anchor, replacement, 1)

    for key, path in modules.items():
        path.write_text(sources[key], encoding='utf-8')
    logger.info('Patched subagent agent.wait cancellation into %s', ', '.join(p.name for p in modules.values()))


def _patch_web_fetch_ssrf(openclaw_path: Path) -> None:
    """Patch OpenClaw's ``web_fetch`` tool to allow private-network DNS results.

    ``browser.ssrfPolicy.dangerouslyAllowPrivateNetwork`` only affects the
    ``browser`` tool.  The ``web_fetch`` tool uses a separate code path
    (``fetchWithWebToolsNetworkGuard``) that does **not** read any SSRF
    policy from the config – it always runs in strict mode.

    In Docker containers, public domain names (e.g. ``zh.wikipedia.org``,
    ``www.reddit.com``) may resolve to CDN / anycast / CGNAT addresses
    that OpenClaw's SSRF checker classifies as "special-use" or "private",
    causing legitimate ``web_fetch`` calls to be blocked with
    ``SsrFBlockedError: Blocked: resolves to private/internal/special-use
    IP address``.

    This function uses a multi-strategy approach (from most targeted to
    broadest) to neutralise the SSRF guard:

    Strategy 1 – Inject ``policy`` into the ``fetchWithWebToolsNetworkGuard``
        call inside ``runWebFetch`` (the original approach, matches a
        specific ``Accept-Language`` / closing-brace pattern).
    Strategy 2 – Broader regex that matches **any**
        ``fetchWithWebToolsNetworkGuard({...})`` call and injects
        ``policy`` before the closing ``})``.
    Strategy 3 (nuclear) – Replace the ``fetchWithWebToolsNetworkGuard``
        function itself with a thin wrapper that always sets
        ``policy.dangerouslyAllowPrivateNetwork = true`` before
        delegating to the real implementation.

    The patch is idempotent: re-running it on an already-patched file is
    a safe no-op.
    """
    dist_dir = _find_openclaw_dist_dir(openclaw_path)
    if dist_dir is None:
        logger.warning('OpenClaw dist/ directory not found; skipping web_fetch SSRF patch')
        return

    logger.debug('Found OpenClaw dist/ at %s for SSRF patching', dist_dir)

    import re

    candidates = _get_js_candidates(dist_dir)

    # ── Strategy 1: precise injection after Accept-Language header ──
    strat1_pattern = re.compile(
        r'("Accept-Language":\s*"[^"]*"\s*\}\s*\})'  # end of init: { headers: {...} }
        r'(\s*\}\))',  # closing of outer call })
    )

    # ── Strategy 2: broader – any fetchWithWebToolsNetworkGuard({...}) call ──
    # Match the function call and inject policy before the final `})`.
    # We look for the call pattern and capture up to the last `}` before `)`.
    strat2_pattern = re.compile(
        r'(fetchWithWebToolsNetworkGuard\(\{)'
        r'(.*?)'
        r'(\}\s*\))',
        re.DOTALL,
    )

    # ── Strategy 3 (nuclear): redefine fetchWithWebToolsNetworkGuard itself ──
    # We prepend a wrapper that forces policy.dangerouslyAllowPrivateNetwork.
    strat3_wrapper = (
        '/* nexus-ssrf-patch-v2: override fetchWithWebToolsNetworkGuard to '
        'always allow private-network DNS */\n'
        'var _origFetchWithWebToolsNetworkGuard = typeof fetchWithWebToolsNetworkGuard !== "undefined" '
        '? fetchWithWebToolsNetworkGuard : null;\n'
        'if (_origFetchWithWebToolsNetworkGuard) {\n'
        '  fetchWithWebToolsNetworkGuard = function(opts) {\n'
        '    if (!opts.policy) opts.policy = {};\n'
        '    opts.policy.dangerouslyAllowPrivateNetwork = true;\n'
        '    return _origFetchWithWebToolsNetworkGuard(opts);\n'
        '  };\n'
        '}\n'
    )

    patch_marker = '/* nexus-ssrf-patch'  # common prefix for all patch markers

    patched_count = 0
    for filepath in candidates:
        try:
            content = Path(filepath).read_text(encoding='utf-8')
        except Exception:
            continue

        if 'fetchWithWebToolsNetworkGuard' not in content:
            continue

        # Skip if already patched (any version)
        if patch_marker in content:
            logger.debug('web_fetch SSRF patch already applied in %s', filepath)
            continue

        new_content = content
        strategy_used = None

        # Try Strategy 1 (most precise)
        result = strat1_pattern.sub(
            r'\1, policy: { dangerouslyAllowPrivateNetwork: true } /* nexus-ssrf-patch */\2',
            content,
        )
        if result != content:
            new_content = result
            strategy_used = 1
        else:
            # Try Strategy 2 (broader call-site injection)
            def _inject_policy(m: re.Match) -> str:
                prefix = m.group(1)
                body = m.group(2)
                close = m.group(3)
                return f'{prefix}{body}, policy: {{ dangerouslyAllowPrivateNetwork: true }} /* nexus-ssrf-patch-v2 */{close}'

            result = strat2_pattern.sub(_inject_policy, content, count=0)
            if result != content:
                new_content = result
                strategy_used = 2
            else:
                # Strategy 3 (nuclear): prepend function wrapper
                # Find a safe injection point — right before the first
                # function that references fetchWithWebToolsNetworkGuard.
                idx = content.find('fetchWithWebToolsNetworkGuard')
                if idx > 0:
                    # Walk backwards to find a statement boundary (`;` or `\n`)
                    insert_at = max(content.rfind(';', 0, idx), content.rfind('\n', 0, idx))
                    if insert_at > 0:
                        new_content = content[: insert_at + 1] + '\n' + strat3_wrapper + content[insert_at + 1 :]
                        strategy_used = 3

        if new_content != content:
            Path(filepath).write_text(new_content, encoding='utf-8')
            patched_count += 1
            logger.debug(
                'Patched web_fetch SSRF policy in %s (strategy %d)',
                filepath,
                strategy_used,
            )

    if patched_count > 0:
        logger.debug(
            'web_fetch SSRF patch applied to %d file(s) – private-network DNS results will no longer be blocked',
            patched_count,
        )
    else:
        logger.warning(
            'web_fetch SSRF patch: no matching files found in %s. '
            'The web_fetch tool may still block URLs that resolve to '
            'private/special-use IP addresses.',
            dist_dir,
        )


# A realistic browser User-Agent string used to replace the default
# bot/fetch UA.  Sites like arXiv, Wikipedia, ECVA, and GitHub return
# 403/406 when they detect non-browser User-Agents.
BROWSER_USER_AGENT = (
    'Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/131.0.0.0 Safari/537.36'
)

UA_PATCH_MARKER = '/* nexus-ua-patch'


def _patch_web_fetch_ua(openclaw_path: Path) -> None:
    """Patch ``web_fetch`` User-Agent to a real browser string.

    Many websites (arXiv, Wikipedia, ECVA, GitHub raw) return 403 or 406
    when the request User-Agent looks like a bot or headless fetch tool.
    OpenClaw's default ``web_fetch`` sends ``params.userAgent`` which is
    typically something like ``"OpenClaw/x.y.z"``.

    This function replaces references to ``params.userAgent`` in the
    ``web_fetch`` code path with a realistic Chrome UA string.

    If the simple regex fails (e.g. minified code changed), a broader
    fallback replaces *all* occurrences of ``params.userAgent`` in files
    that contain ``runWebFetch``.

    The patch is idempotent.
    """
    dist_dir = _find_openclaw_dist_dir(openclaw_path)
    if dist_dir is None:
        logger.warning('OpenClaw dist/ directory not found; skipping web_fetch UA patch')
        return

    import re

    candidates = _get_js_candidates(dist_dir)

    # Strategy 1: precise – only inside the headers object of web_fetch
    precise_pattern = re.compile(r'("User-Agent":\s*)params\.userAgent')

    # Strategy 2: broader – any `userAgent:params.userAgent` or
    # `"User-Agent":params.userAgent` pattern
    broad_pattern = re.compile(r'((?:"User-Agent"|userAgent)\s*:\s*)params\.userAgent')

    patched_count = 0
    for filepath in candidates:
        try:
            content = Path(filepath).read_text(encoding='utf-8')
        except Exception:
            continue

        if 'runWebFetch' not in content and 'fetchWithWebToolsNetworkGuard' not in content:
            continue

        # Skip if already patched
        if UA_PATCH_MARKER in content:
            logger.debug('web_fetch UA patch already applied in %s', filepath)
            continue

        browser_ua_js = f'"{BROWSER_USER_AGENT}"'

        # Try precise pattern first
        new_content = precise_pattern.sub(
            rf'\1{browser_ua_js} {UA_PATCH_MARKER} */',
            content,
        )
        if new_content == content:
            # Fallback to broader pattern
            new_content = broad_pattern.sub(
                rf'\1{browser_ua_js} {UA_PATCH_MARKER}-v2 */',
                content,
            )

        if new_content != content:
            Path(filepath).write_text(new_content, encoding='utf-8')
            patched_count += 1
            logger.debug('Patched web_fetch User-Agent in %s', filepath)

    if patched_count > 0:
        logger.debug(
            'web_fetch UA patch applied to %d file(s) – requests will use '
            'a real browser User-Agent to avoid 403 rejections',
            patched_count,
        )
    else:
        logger.warning(
            'web_fetch UA patch: no matching files found in %s. '
            'Some websites may return 403 due to bot User-Agent detection.',
            dist_dir,
        )


CURL_FALLBACK_PATCH_MARKER = '/* nexus-curl-fallback-patch-v2'


def _find_runtime_curl_binary() -> str | None:
    """Locate curl in runtime images where it may not be on PATH."""
    resolved = find_runtime_binary('curl')
    if resolved:
        return resolved

    for candidate in (
        Path('/usr/bin/curl'),
        Path('/usr/local/bin/curl'),
        Path('/bin/curl'),
    ):
        if candidate.is_file():
            return str(candidate)

    nix_store = Path('/nix/store')
    if nix_store.is_dir():
        for pattern in ('*-curl-*-bin/bin/curl', '*-curl-*/bin/curl'):
            for candidate in nix_store.glob(pattern):
                if candidate.is_file():
                    return str(candidate)

    return None


def _patch_web_fetch_curl_fallback(openclaw_path: Path) -> None:
    """Patch ``web_fetch`` to fall back to ``curl`` when Node.js fetch cannot retrieve a page.

    Many websites (Wikipedia, some arXiv pages) use **TLS fingerprint
    (JA3/JA4) detection** to block non-browser HTTP clients.  Node.js's
    built-in ``fetch`` (BoringSSL/OpenSSL) and Python's ``requests``
    both produce TLS handshake signatures that are fingerprinted and
    rejected, while ``curl`` (using the system's libcurl/OpenSSL stack)
    produces a different JA3 hash that is allowed through.

    Simply changing the ``User-Agent`` header is not sufficient — the
    anti-bot systems inspect the TLS layer, not the HTTP headers.

    This patch injects a **curl fallback** into the ``runWebFetch``
    function: when the initial ``fetchWithWebToolsNetworkGuard`` call
    returns a 403/429 status or throws a network error, the code spawns
    ``curl`` as a child process to re-fetch the same URL.  ``curl``'s
    different TLS fingerprint and proxy handling bypass these failures.

    The injected code:

    1. Detects HTTP 403/429 in the ``!res.ok`` branch (after the
       Firecrawl fallback has already been tried and failed), and network
       exceptions from the initial Node.js ``fetch`` call.
    2. Spawns ``curl -sL -H "User-Agent: <Chrome UA>" <url>`` via
       ``child_process.execFileSync``.
    3. If ``curl`` succeeds and returns non-empty content, constructs a
       synthetic ``web_fetch`` result payload (with ``extractor:
       "curl-fallback"``) and returns it — short-circuiting the error.
    4. If ``curl`` fails (not installed, network error, etc.), the
       original ``throw new Error(...)`` is executed as before.

    The patch is idempotent: re-running it on an already-patched file
    is a safe no-op.
    """
    dist_dir = _find_openclaw_dist_dir(openclaw_path)
    if dist_dir is None:
        logger.warning('OpenClaw dist/ directory not found; skipping web_fetch curl fallback patch')
        return

    import re

    candidates = _get_js_candidates(dist_dir)

    # We target the exact line that throws the "Web fetch failed (STATUS)" error
    # inside the `if (!res.ok) { ... }` block of `runWebFetch`.
    #
    # Original code (after Firecrawl fallback failed):
    #   if (payload) return payload;
    #   const rawDetail = (await readResponseText(res, { ... })).text;
    #   const wrappedDetail = wrapWebFetchContent(...);
    #   throw new Error(`Web fetch failed (${res.status}): ${wrappedDetail.text}`);
    #
    # We replace the throw with a curl fallback attempt.

    # Match the throw line precisely — it's unique per file.
    throw_pattern = re.compile(
        r'(\t\t\tthrow new Error\(`Web fetch failed \(\$\{res\.status\}\): \$\{wrappedDetail\.text\}`\);)'
    )

    helper_pattern = re.compile(r'(async function runWebFetch\(params\) \{)')
    fetch_error_pattern = re.compile(
        r'(\t\tif \(payload\) return payload;\n)'
        r'(\t\tthrow error;\n\t\}\n\ttry \{\n\t\tif \(!res\.ok\) \{)'
    )

    curl_helper_js = r"""/* nexus-curl-fallback-patch-v2: curl fallback for HTTP blocks and Node fetch network errors */
async function nexusCurlWebFetchFallback(params, cacheKey, start, reason) {
	try {
		const { execFileSync } = await import("node:child_process");
		const curlBinary = process.env.NEXUS_CURL_BINARY || process.env.CURL_BINARY || "curl";
		const curlUA = "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/131.0.0.0 Safari/537.36";
		const curlArgs = [
			"-sSL",
			"--max-time", "30",
			"-H", `User-Agent: ${curlUA}`,
			"-H", "Accept: text/html,application/xhtml+xml,*/*;q=0.1",
			"-H", "Accept-Language: en-US,en;q=0.9",
			params.url
		];
		logDebug(`[web-fetch] trying curl fallback for ${params.url}: ${reason}`);
		const curlBuf = execFileSync(curlBinary, curlArgs, {
			maxBuffer: 10 * 1024 * 1024,
			timeout: 35000,
			env: process.env
		});
		const curlBody = curlBuf.toString("utf-8");
		if (!curlBody || curlBody.length <= 100) return null;
		logDebug(`[web-fetch] curl fallback succeeded for ${params.url} (${curlBody.length} bytes)`);
		let text = curlBody;
		let title;
		let extractor = "curl-fallback-raw";
		if (curlBody.includes("<html") || curlBody.includes("<HTML") || curlBody.includes("<!DOCTYPE") || curlBody.includes("<!doctype")) {
			if (typeof extractReadableContent === "function" && params.readabilityEnabled) {
				try {
					const readable = await extractReadableContent({ html: curlBody, url: params.url, extractMode: params.extractMode });
					if (readable?.text) {
						text = readable.text;
						title = readable.title;
						extractor = "curl-fallback-readability";
					}
				} catch(e) { /* readability failed, use raw */ }
			}
			if (extractor === "curl-fallback-raw") {
				text = curlBody
					.replace(/<script[\s\S]*?<\/script>/gi, "")
					.replace(/<style[\s\S]*?<\/style>/gi, "")
					.replace(/<[^>]+>/g, " ")
					.replace(/\s+/g, " ")
					.trim();
				extractor = "curl-fallback-strip";
			}
		}
		const wrapped = wrapWebFetchContent(text, params.maxChars);
		const wrappedTitle = title ? wrapWebFetchField(title) : void 0;
		const curlPayload = {
			url: params.url, finalUrl: params.url, status: 200,
			contentType: "text/html",
			title: wrappedTitle, extractMode: params.extractMode, extractor,
			externalContent: { untrusted: true, source: "web_fetch", wrapped: true },
			truncated: wrapped.truncated, length: wrapped.wrappedLength,
			rawLength: wrapped.rawLength, wrappedLength: wrapped.wrappedLength,
			fetchedAt: new Date().toISOString(), tookMs: Date.now() - start,
			text: wrapped.text
		};
		writeCache(FETCH_CACHE, cacheKey, curlPayload, params.cacheTtlMs);
		return curlPayload;
	} catch(curlErr) {
		logDebug(`[web-fetch] curl fallback failed for ${params.url}: ${curlErr.message}`);
		return null;
	}
}

"""

    # The HTTP-status fallback JS code that replaces the throw.
    curl_fallback_js = r"""/* nexus-curl-fallback-patch-v2: try curl when Node.js fetch gets blocked */
			if (res.status === 403 || res.status === 429) {
				const curlPayload = await nexusCurlWebFetchFallback(params, cacheKey, start, `HTTP ${res.status}`);
				if (curlPayload) return curlPayload;
			}
			throw new Error(`Web fetch failed (${res.status}): ${wrappedDetail.text}`);"""

    fetch_error_fallback_js = (
        r'\1'
        '\t\tconst curlPayload = await nexusCurlWebFetchFallback(params, cacheKey, start, error?.message || "fetch error");\n'
        '\t\tif (curlPayload) return curlPayload;\n'
        r'\2'
    )

    patched_count = 0
    for filepath in candidates:
        try:
            content = Path(filepath).read_text(encoding='utf-8')
        except Exception:
            continue

        if 'runWebFetch' not in content:
            continue

        # Skip if already patched
        if CURL_FALLBACK_PATCH_MARKER in content:
            logger.debug('web_fetch curl fallback patch already applied in %s', filepath)
            continue

        new_content = helper_pattern.sub(lambda m: curl_helper_js + m.group(1), content, count=1)

        # Use a lambda to avoid Python's re.sub interpreting backslash
        # sequences (like \s, \S) in the replacement JS code.
        new_content = throw_pattern.sub(lambda m: curl_fallback_js, new_content)
        new_content = fetch_error_pattern.sub(fetch_error_fallback_js, new_content, count=1)

        if new_content != content:
            Path(filepath).write_text(new_content, encoding='utf-8')
            patched_count += 1
            logger.debug('Patched web_fetch with curl fallback in %s', filepath)

    if patched_count > 0:
        logger.debug(
            'web_fetch curl fallback patch applied to %d file(s) – '
            '403/429 responses and Node.js fetch errors will be retried '
            'via curl',
            patched_count,
        )
    else:
        logger.warning(
            'web_fetch curl fallback patch: no matching files found in %s. '
            'Websites using TLS fingerprint anti-bot may still block web_fetch.',
            dist_dir,
        )


def _openclaw_supports_brave_base_url(openclaw_path: Path) -> bool:
    """Return true for OpenClaw builds that expose Brave provider baseUrl config."""
    dist_dir = _find_openclaw_dist_dir(openclaw_path)
    if dist_dir is None:
        return False

    for filepath in _get_js_candidates(dist_dir):
        try:
            content = Path(filepath).read_text(encoding='utf-8')
        except Exception:
            continue
        if 'BRAVE_SEARCH_ENDPOINT_PATH' in content or 'resolveBraveBaseUrl' in content:
            return True
    return False


def _patch_web_search_bridge_endpoint(openclaw_path: Path) -> bool:
    """Patch legacy OpenClaw Brave endpoint to read the local bridge URL from env."""
    dist_dir = _find_openclaw_dist_dir(openclaw_path)
    if dist_dir is None:
        logger.warning('OpenClaw dist/ directory not found; cannot patch legacy web_search endpoint')
        return False

    endpoint_pattern = re.compile(r'const\s+BRAVE_SEARCH_ENDPOINT\s*=\s*["\'][^"\']+["\'];')
    replacement = (
        'const BRAVE_SEARCH_ENDPOINT = '
        '((process.env.NEXUS_SERPER_BRIDGE_BASE_URL || "http://127.0.0.1:0").replace(/\\/+$/u, "") '
        '+ "/res/v1/web/search"); '
        f'{WEB_SEARCH_ENDPOINT_PATCH_MARKER}'
    )

    patched = False
    already_env_based = False
    incompatible_direct_serper_patch = False

    for filepath in _get_js_candidates(dist_dir):
        path = Path(filepath)
        try:
            content = path.read_text(encoding='utf-8')
        except Exception:
            continue

        if '/* nexus-serper-patch-v2' in content:
            incompatible_direct_serper_patch = True
            continue
        if 'BRAVE_SEARCH_ENDPOINT' not in content or 'runWebSearch' not in content:
            continue
        if 'process.env.NEXUS_SERPER_BRIDGE_BASE_URL' in content:
            already_env_based = True
            continue

        new_content = endpoint_pattern.sub(replacement, content, count=1)
        if new_content != content:
            path.write_text(new_content, encoding='utf-8')
            patched = True

    if incompatible_direct_serper_patch:
        logger.warning('Found old direct Serper web_search patch; disabling web_search instead of stacking patches')
        return False
    if patched or already_env_based:
        return True

    logger.warning('Legacy OpenClaw web_search endpoint patch did not find a compatible endpoint constant')
    return False


def _prepare_web_search_compat(
    *,
    openclaw_bin: Path,
    bridge_base_url: str | None,
    serper_configured: bool,
) -> WebSearchCompat:
    """Resolve and apply the OpenClaw web_search compatibility path."""
    if not serper_configured or not bridge_base_url:
        return WebSearchCompat(enabled=False, base_url=None, uses_brave_base_url=False)

    base_url = bridge_base_url.rstrip('/')
    if _openclaw_supports_brave_base_url(openclaw_bin):
        return WebSearchCompat(enabled=True, base_url=base_url, uses_brave_base_url=True)

    if _patch_web_search_bridge_endpoint(openclaw_bin):
        return WebSearchCompat(enabled=True, base_url=base_url, uses_brave_base_url=False)

    return WebSearchCompat(enabled=False, base_url=None, uses_brave_base_url=False)


async def _start_web_search_bridge_for_openclaw(
    *,
    openclaw_bin: Path,
    serper_config: Any | None,
    timeout: float,
    bridge_factory: Any,
) -> tuple[Any | None, WebSearchCompat]:
    """Start the local bridge and resolve OpenClaw web_search compatibility."""
    disabled = WebSearchCompat(enabled=False, base_url=None, uses_brave_base_url=False)
    if serper_config is None:
        logger.warning('OpenClaw web_search disabled because Serper credentials are not configured')
        return None, disabled

    candidate_bridge = bridge_factory(
        config=serper_config,
        listen_port=find_free_port(),
        listen_host='127.0.0.1',
        request_timeout=timeout,
    )
    try:
        await candidate_bridge.start()
        candidate_web_search = _prepare_web_search_compat(
            openclaw_bin=openclaw_bin,
            bridge_base_url=candidate_bridge.base_url,
            serper_configured=True,
        )
        if candidate_web_search.enabled:
            logger.debug('Enabled OpenClaw web_search via local Serper bridge at %s', candidate_bridge.base_url)
            return candidate_bridge, candidate_web_search

        await candidate_bridge.stop()
        logger.warning('OpenClaw web_search disabled because bridge compatibility setup failed')
    except Exception as exc:
        with suppress(Exception):
            await candidate_bridge.stop()
        logger.warning('OpenClaw web_search disabled because Serper bridge setup failed: %s', exc)

    return None, disabled


def _register_injected_skills(
    config_path: Path,
    injected_skills_dir: str = '/root/skills',
) -> None:
    """Register *injected_skills_dir* as an extra skills root in OpenClaw's config.

    OpenClaw discovers user skills from several well-known directories but
    **not** ``~/skills/``.  The actual discovery roots are:

    * ``bundledSkillsDir``  - the ``skills/`` folder next to the OpenClaw
      package (read-only, ships built-in skills).
    * ``managedSkillsDir``  - ``$OPENCLAW_STATE_DIR/skills``.
    * ``personalAgentsSkillsDir`` - ``$HOME/.agents/skills``.
    * ``projectAgentsSkills`` - ``<workspace>/.agents/skills``.
    * ``workspaceSkillsDir`` - ``<workspace>/skills``.
    * ``extraDirs`` - arbitrary directories listed in the config key
      ``skills.load.extraDirs``.

    Additionally, each root applies a **symlink-escape check**
    (``resolveContainedSkillPath``): if a candidate skill path resolves
    (via ``fs.realpathSync``) to a location *outside* its root directory
    the skill is silently skipped.  This means that simply symlinking
    ``$HOME/skills → /root/skills`` does **not** work – the realpath of
    each skill file resolves to ``/root/skills/…`` which is outside the
    ``$HOME/skills`` root.

    The most reliable approach is therefore to add *injected_skills_dir*
    to ``skills.load.extraDirs`` in OpenClaw's JSON config.  When the
    root itself *is* ``/root/skills``, the realpath check passes
    naturally.

    The function is intentionally a no-op when *injected_skills_dir* does
    not exist (i.e. no skills were injected for the current task).
    """
    src = Path(injected_skills_dir)
    if not src.is_dir():
        return

    if not config_path.exists():
        logger.warning(
            'OpenClaw config not found at %s; cannot register injected skills',
            config_path,
        )
        return

    try:
        cfg = json.loads(config_path.read_text(encoding='utf-8'))
    except Exception as exc:
        logger.warning('Failed to read OpenClaw config for skills registration: %s', exc)
        return

    # Ensure skills.load.extraDirs contains the injected directory.
    skills_cfg = cfg.setdefault('skills', {})
    load_cfg = skills_cfg.setdefault('load', {})
    extra_dirs: list[str] = load_cfg.setdefault('extraDirs', [])

    abs_src = str(src.resolve())
    if abs_src not in extra_dirs:
        extra_dirs.append(abs_src)
        config_path.write_text(json.dumps(cfg, indent=2), encoding='utf-8')
        logger.debug(
            'Registered injected skills dir %s in OpenClaw config (%s)',
            abs_src,
            config_path,
        )
    else:
        logger.debug('Injected skills dir %s already registered', abs_src)


def _read_transcript(stdout: str, state_dir: Path) -> list[dict[str, Any]]:
    """Read the OpenClaw session transcript from the state directory."""
    agent_dir = state_dir / 'agents'
    payload = parse_openclaw_response(stdout)
    if payload is None:
        return []

    meta = payload.get('meta')
    if not isinstance(meta, dict):
        return []
    agent_meta = meta.get('agentMeta')
    if not isinstance(agent_meta, dict):
        return []
    session_id = agent_meta.get('sessionId')
    if not session_id:
        return []

    # Search for {session_id}.jsonl recursively in agent_dir
    matching_files = list(agent_dir.glob(f'**/{session_id}.jsonl'))

    if not matching_files:
        return []

    transcript_path = matching_files[0]

    transcript: list[dict[str, Any]] = []
    try:
        for line in transcript_path.read_text(encoding='utf-8').splitlines():
            if not line.strip():
                continue
            try:
                transcript.append(json.loads(line))
            except json.JSONDecodeError as exc:
                transcript.append({'raw': line, 'parse_error': str(exc)})
    except Exception:
        return []

    return transcript


def _append_artifact_path(stderr: str, artifact_dir: Path) -> str:
    note = f'OpenClaw runtime artifacts preserved at {artifact_dir}'
    return f'{stderr}\n{note}'.strip() if stderr else note


async def setup_and_run_openclaw(
    task: str,
    model: str,
    working_dir: str = '/workspace',
    timeout: float = 3600,
    onboard_timeout: float | None = DEFAULT_ONBOARD_TIMEOUT,
    version: str = 'latest',
    npm_registry: str | None = DEFAULT_NPM_REGISTRY,
    context_window: int = DEFAULT_CONTEXT_WINDOW,
    max_tokens: int = DEFAULT_MAX_TOKENS,
    llm_idle_timeout_ms: int | float | str = DEFAULT_LLM_IDLE_TIMEOUT_MS,
    supports_images: bool = False,
) -> dict[str, Any]:
    """Install OpenClaw, configure it against the local proxy, then run one agent turn.

    ``sessions_spawn`` always needs a gateway, but *where the main agent runs* splits the supported range in two, and that choice is what decides whether the wait patch and the gateway lifetime matter at all:

    1) ``--local``: the CLI runs the main agent in-process, the gateway only hosts sub-agents.  The CLI blocks until every child is done -- even a fire-and-forget spawn -- so the gateway may be stopped as soon as it returns.  From 2026.6.5-beta.3 to 2026.7.2-beta.7 this mode needs ``_patch_subagent_wait_abort``: the CLI issues ``agent.wait`` with no ``AbortSignal`` and nothing cancels it when a child ends, so the poll never settles, its WebSocket stays open, and Node's event loop cannot drain.  The reply is already on stdout, yet the process will not exit and the outer budget kills it.  The defect is OpenClaw's, not Node's.
    2) no ``--local``: the gateway runs the main agent too and the CLI only submits and waits, so the patch has no effect.  The CLI still blocks while the parent explicitly waits, but a fire-and-forget spawn lets it return in seconds, so here the gateway must outlive the run or the child is cut off mid-flight.

    Measured in-sandbox.  "node 22.20.0" is whether nexus' default runtime node can run the release at all:

    release                node 22.20.0  mode  wait patch  note
    ---------------------  ------------  ----  ----------  --------------------------------------------------
    2026.3.x - 4.2         ok            1     no target   ``timeoutSeconds`` rejected below 2026.4.26-beta.1
    2026.4.5 - 6.5-beta.2  ok            1     harmful     patching cuts a fire-and-forget child short
    2026.6.5-beta.3 +      ok            1     required    starting from 2026.6.5-beta.3, need wait patch
    2026.7.1-beta.6        ok            1     required    newest release the default node still runs
    2026.7.1 - 7.2-beta.7  no (22.22.3)  1     required    from 2026.7.1 on, nexus' default node is too old
    2026.8.1-beta.1 +      no (22.22.3)  2     no effect   ``--local`` no longer usable from here on
    2026.9.3               no (24.16.0)  2     no effect   ``dist/`` moved to ``.mjs``; see below

    Mind the ordering: 2026.x.y-beta.N precedes stable 2026.x.y. Thus, 2026.6.5 is between 2026.6.5-beta.3 and 2026.7.1-beta.6. Suggested releases to use: 2026.3.13, 2026.5.28, 2026.6.5

    FIXME:
    1) Currently, nexus default runtime node only supports up to 2026.7.1-beta.6.
    2) Mode 1 --local dies at 2026.8.1-beta.1, from this release on, we need a polling mechanism to check whether all sub-agents are done after CLI returns.
    3) 2026.9.3 moved ``dist/`` to ``.mjs``, which the ``.js``-only glob in ``_get_js_candidates`` never reaches, so the patch is a silent no-op there.
    """
    from nexus.runtime.context import require_runtime_control
    from nexus.runtime.executors.command import CommandRunStatus, RunCommandRequest, run_command

    from .web_search_bridge import SerperBridge, resolve_serper_bridge_config

    rt = require_runtime_control()
    resolved_llm_idle_timeout_ms = _resolve_llm_idle_timeout_ms(llm_idle_timeout_ms)

    start_time = time.time()
    run_root = Path(tempfile.mkdtemp(prefix='openclaw-run-'))
    home_dir = run_root / 'home'
    state_dir = run_root / '.openclaw'
    config_path = state_dir / 'openclaw.json'
    preserve_artifacts = False

    home_dir.mkdir(parents=True, exist_ok=True)
    state_dir.mkdir(parents=True, exist_ok=True)
    Path(working_dir).mkdir(parents=True, exist_ok=True)

    try:
        openclaw_bin = await install_openclaw(
            version=version,
            install_dir=DEFAULT_INSTALL_DIR,
            npm_registry=npm_registry,
        )
    except Exception as exc:
        preserve_artifacts = True
        return {
            'success': False,
            'output': '',
            'stderr': _append_artifact_path(f'Failed to install OpenClaw: {exc}', run_root),
            'exit_code': -1,
            'execution_time': time.time() - start_time,
            'transcript': [],
            'failure_kind': 'infrastructure',
            'failure_stage': 'install',
        }

    # openclaw_bin is e.g. /tmp/openclaw/2026.3.13/node_modules/.bin/openclaw
    # The npm prefix (install_dir for resolve_npm_bin_entry) is three levels up.
    npm_prefix = openclaw_bin.parent.parent.parent

    openclaw_version = _read_openclaw_package_version(openclaw_bin)
    logger.info('Resolved OpenClaw version %s from %s', openclaw_version or 'unknown', openclaw_bin)

    # Patch the web_fetch tool's SSRF guard so that DNS responses pointing
    # to CDN / anycast / CGNAT addresses are not incorrectly rejected.
    # This complements the browser.ssrfPolicy config patch applied later.
    _patch_web_fetch_ssrf(openclaw_bin)

    # Let an accepted sessions_spawn stop pinning the CLI process open.
    _patch_subagent_wait_abort(openclaw_bin, openclaw_version)

    # Patch the User-Agent header to a real browser string so that
    # sites like arXiv, Wikipedia, GitHub, etc. don't return 403.
    _patch_web_fetch_ua(openclaw_bin)

    # Patch web_fetch to fall back to curl when Node.js fetch gets 403/429
    # or raises a network error before receiving a response.
    # Sites like Wikipedia use TLS fingerprint (JA3) detection to block
    # Node.js/Python HTTP clients; curl's different TLS stack bypasses this.
    _patch_web_fetch_curl_fallback(openclaw_bin)

    node_path = find_nexus_runtime_binary('node')
    if not node_path:
        required = '.'.join(str(part) for part in MIN_NODE_VERSION)
        preserve_artifacts = True
        return {
            'success': False,
            'output': '',
            'stderr': _append_artifact_path(
                f'Node.js >= {required} is required but not found in runtime (checked {DEFAULT_RUNTIME_BIN_DIR} and PATH)',
                run_root,
            ),
            'exit_code': -1,
            'execution_time': time.time() - start_time,
            'transcript': [],
            'failure_kind': 'infrastructure',
            'failure_stage': 'runtime_dependency',
        }
    logger.debug('Using Node.js binary for OpenClaw CLI: %s', node_path)

    js_entry = resolve_npm_bin_entry(npm_prefix, 'openclaw', 'openclaw')
    if not js_entry or not js_entry.exists():
        preserve_artifacts = True
        return {
            'success': False,
            'output': '',
            'stderr': _append_artifact_path(f'OpenClaw JS entry point not found under {npm_prefix}', run_root),
            'exit_code': -1,
            'execution_time': time.time() - start_time,
            'failure_kind': 'infrastructure',
            'failure_stage': 'entrypoint',
        }

    proxy_port = find_free_port()
    openclaw_proxy_class = _get_openclaw_proxy_class()
    proxy = openclaw_proxy_class(
        sio=rt.sio,
        listen_port=proxy_port,
        listen_host='127.0.0.1',
        request_timeout=timeout,
        models=[model],
    )
    bridge: SerperBridge | None = None
    web_search = WebSearchCompat(enabled=False, base_url=None, uses_brave_base_url=False)

    try:
        await proxy.start()
    except Exception as exc:
        preserve_artifacts = True
        return {
            'success': False,
            'output': '',
            'stderr': _append_artifact_path(f'Failed to start OpenAI proxy: {exc}', run_root),
            'exit_code': -1,
            'execution_time': time.time() - start_time,
            'transcript': [],
            'failure_kind': 'infrastructure',
            'failure_stage': 'proxy',
        }

    try:
        bridge, web_search = await _start_web_search_bridge_for_openclaw(
            openclaw_bin=openclaw_bin,
            serper_config=resolve_serper_bridge_config(os.environ),
            timeout=timeout,
            bridge_factory=SerperBridge,
        )

        curl_binary = _find_runtime_curl_binary()
        openclaw_env = _build_openclaw_env(
            home_dir=home_dir,
            state_dir=state_dir,
            config_path=config_path,
            proxy_port=proxy_port,
            web_search=web_search,
            curl_binary=curl_binary,
        )
        if curl_binary:
            logger.debug('Using curl binary for OpenClaw web_fetch fallback: %s', curl_binary)
        else:
            logger.warning('curl binary not found; OpenClaw web_fetch curl fallback may be unavailable')

        cli_timeout = max(1, math.ceil(timeout))
        resolved_onboard_timeout = _resolve_onboard_timeout(
            agent_timeout=timeout,
            onboard_timeout=onboard_timeout,
        )
        onboard_cmd = build_onboard_command(
            node_path=node_path,
            js_entry=js_entry,
            model=model,
            working_dir=working_dir,
            proxy_port=proxy_port,
        )

        onboard_result = await run_command(
            RunCommandRequest(
                command=shlex.join(onboard_cmd),
                cwd=working_dir,
                timeout=resolved_onboard_timeout,
                extra_env=openclaw_env,
            )
        )
        if onboard_result.status == CommandRunStatus.TimeLimitExceeded:
            raise RuntimeError(f'Command timed out after {resolved_onboard_timeout}s: {shlex.join(onboard_cmd)}')
        if onboard_result.status != CommandRunStatus.Finished:
            raise RuntimeError(
                f'Command execution failed: {shlex.join(onboard_cmd)}\n{onboard_result.stderr or ""}'.strip()
            )
        if (onboard_result.return_code or 0) != 0:
            raise RuntimeError(
                (
                    f'Command returned {onboard_result.return_code}: {shlex.join(onboard_cmd)}\n'
                    f'stdout:\n{onboard_result.stdout or ""}\n'
                    f'stderr:\n{onboard_result.stderr or ""}'
                ).strip()
            )

        # Patch the model entry written by onboard to set the correct
        # contextWindow / maxTokens. Onboard uses conservative defaults
        # for unknown custom providers (often only 16K), which causes
        # compaction warnings and limits agent performance.
        _patch_model_config(
            config_path=config_path,
            provider_id=DEFAULT_PROVIDER_ID,
            model=model,
            context_window=context_window,
            max_tokens=max_tokens,
            llm_idle_timeout_ms=resolved_llm_idle_timeout_ms,
            openclaw_version=openclaw_version,
        )

        if web_search.enabled and web_search.uses_brave_base_url:
            await _try_install_brave_plugin_for_openclaw(
                node_path=node_path,
                js_entry=js_entry,
                working_dir=working_dir,
                openclaw_env=openclaw_env,
                npm_registry=npm_registry,
                timeout=timeout,
            )

        # Apply the full set of config patches (tools.exec, sandbox,
        # imageModel, ssrfPolicy) that align Nexus with the
        # original WildClawBench run_batch.py behaviour.
        _patch_full_config(
            config_path=config_path,
            provider_id=DEFAULT_PROVIDER_ID,
            model=model,
            supports_images=supports_images,
            web_search_enabled=web_search.enabled,
            web_search_base_url=web_search.base_url,
            web_search_uses_brave_base_url=web_search.uses_brave_base_url,
        )
        if web_search.enabled:
            brave_ready = await _verify_brave_web_search_provider(
                node_path=node_path,
                js_entry=js_entry,
                working_dir=working_dir,
                openclaw_env=openclaw_env,
                timeout=timeout,
            )
            if not brave_ready:
                logger.warning('OpenClaw web_search disabled because Brave provider is not available/configured')
                if bridge is not None:
                    await bridge.stop()
                    bridge = None
                web_search = WebSearchCompat(enabled=False, base_url=None, uses_brave_base_url=False)
                _patch_full_config(
                    config_path=config_path,
                    provider_id=DEFAULT_PROVIDER_ID,
                    model=model,
                    supports_images=supports_images,
                    web_search_enabled=False,
                    web_search_base_url=None,
                    web_search_uses_brave_base_url=False,
                )
                openclaw_env = _build_openclaw_env(
                    home_dir=home_dir,
                    state_dir=state_dir,
                    config_path=config_path,
                    proxy_port=proxy_port,
                    web_search=web_search,
                    curl_binary=curl_binary,
                )

        # Register /root/skills as an extra skills root in OpenClaw's
        # config so that user-injected skills (e.g. from WildClawBench)
        # are discovered at agent bootstrap.
        _register_injected_skills(config_path, '/root/skills')

        # sessions_spawn requires a local gateway (``--local`` agent still dials it).
        gateway_port = find_free_port()
        _patch_gateway_config(config_path, port=gateway_port)
        openclaw_env['OPENCLAW_GATEWAY_TOKEN'] = DEFAULT_GATEWAY_AUTH_TOKEN
        openclaw_env['OPENCLAW_GATEWAY_PORT'] = str(gateway_port)
        openclaw_env['no_proxy'] = _merge_no_proxy(
            openclaw_env.get('no_proxy'),
            f'127.0.0.1:{gateway_port}',
            f'localhost:{gateway_port}',
        )
        openclaw_env['NO_PROXY'] = openclaw_env['no_proxy']

        gateway_proc: asyncio.subprocess.Process | None = None
        try:
            gateway_proc = await _start_openclaw_gateway(
                node_path=node_path,
                js_entry=js_entry,
                working_dir=working_dir,
                openclaw_env=openclaw_env,
                port=gateway_port,
                state_dir=state_dir,
            )

            cmd = build_agent_command(
                node_path=node_path,
                js_entry=js_entry,
                task=task,
                timeout=cli_timeout,
            )

            # Outer run_command budget is slightly larger than openclaw's --timeout so the CLI can finish dumping its ReplyPayload after the embedded idle/timeout fires.
            result = await run_command(
                RunCommandRequest(
                    command=shlex.join(cmd),
                    cwd=working_dir,
                    timeout=cli_timeout + _OPENCLAW_EXIT_GRACE_SECONDS,
                    extra_env=openclaw_env,
                )
            )
        finally:
            await _stop_openclaw_gateway(gateway_proc)

        stdout, stderr, exit_code = unpack_command_result(
            result,
            timeout_message='Execution timed out',
            error_message='Command execution error',
        )
        if exit_code == 0 and parse_openclaw_response(stdout) is None:
            stderr_payload = parse_openclaw_response(stderr)
            if stderr_payload is not None:
                stdout = json.dumps(stderr_payload, ensure_ascii=False)

        exit_status, embedded_failure = classify_openclaw_response(stdout, exit_code)
        timed_out = result.status == CommandRunStatus.TimeLimitExceeded or exit_status == 'timeout'
        if timed_out:
            exit_status = 'timeout'
        success = exit_status == 'completed'

        execution_time = time.time() - start_time
        if embedded_failure:
            stderr = f'{stderr}\n{embedded_failure}'.strip() if stderr else embedded_failure
        if not success:
            preserve_artifacts = True
            stderr = _append_artifact_path(stderr, run_root)
        failure_kind, failure_stage = classify_command_failure(
            result,
            process_failed=not success,
        )

        # Read session transcript
        transcript = _read_transcript(stdout, state_dir)

        return {
            'success': success,
            'output': stdout,
            'stderr': stderr,
            'exit_code': exit_code,
            'execution_time': execution_time,
            'timed_out': timed_out,
            'exit_status': exit_status,
            'transcript': transcript,
            **({'failure_kind': failure_kind, 'failure_stage': failure_stage} if failure_kind is not None else {}),
        }
    finally:
        try:
            if bridge is not None:
                await bridge.stop()
        finally:
            try:
                await proxy.stop()
            finally:
                if not preserve_artifacts:
                    shutil.rmtree(run_root, ignore_errors=True)


__all__ = [
    'DEFAULT_AGENT_ID',
    'DEFAULT_CONTEXT_WINDOW',
    'DEFAULT_LLM_IDLE_TIMEOUT_MS',
    'DEFAULT_MAX_TOKENS',
    'DEFAULT_ONBOARD_TIMEOUT',
    'DEFAULT_PROVIDER_ID',
    'DUMMY_PROXY_API_KEY',
    'OpenClawExitStatus',
    'OpenClawResult',
    'build_agent_command',
    'build_openclaw_config',
    'build_onboard_command',
    'classify_openclaw_response',
    'find_free_port',
    'is_openclaw_success',
    'parse_openclaw_response',
    'setup_and_run_openclaw',
    'visible_openclaw_payload_texts',
]
