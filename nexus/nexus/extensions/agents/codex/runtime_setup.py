"""Runtime-side setup and execution for Codex CLI."""

from __future__ import annotations

import json
import logging
import os
import re
import shlex
import subprocess
import sys
import time
import uuid

from collections.abc import Callable
from dataclasses import dataclass
from functools import partial
from pathlib import Path
from typing import Any, Literal

from packaging.version import InvalidVersion, Version

from nexus.extensions.agents.installed_common import (
    DEFAULT_AGENT_INSTALL_ROOT,
    DEFAULT_NPM_REGISTRY,
    DEFAULT_PROXY_API_KEY,
    build_runtime_command,
    build_runtime_path,
    classify_command_failure,
    find_free_port,
    find_nexus_runtime_binary,
    resolve_npm_bin_entry,
    unpack_command_result,
)
from nexus.extensions.agents.installed_common.execution import AgentExecutionTransport, ExecutionTransportError
from nexus.trace import get_current_span_id
from nexus.utils.platform import is_windows

from .installer import DEFAULT_CODEX_VERSION


logger = logging.getLogger(__name__)


DEFAULT_INSTALL_DIR = f'{DEFAULT_AGENT_INSTALL_ROOT}/codex-cli'
_LEGACY_FREEFORM_APPLY_PATCH_CONFIG_REMOVED_VERSION = Version('0.131.0a15')
# Per-run CODEX_HOME must stay outside working_dir (else its session state pollutes the working directory)
# and outside the temp dir (Codex refuses to create apply_patch binaries there), so HOME it is.
CODEX_HOME_DIR_NAME = '.nexus-codex'
_CODEX_HOME_NAME_PATTERN = re.compile(r'[A-Za-z0-9][A-Za-z0-9._-]{0,127}\Z')


@dataclass
class CodexResult:
    """Result of running Codex."""

    success: bool
    output: str
    stderr: str
    exit_code: int
    execution_time: float


def _toml_string(value: str) -> str:
    return json.dumps(value)


# Root-level `model_instructions_file = ...` assignment inside extra_config_toml;
# the task-staged file replaces it instead of colliding into invalid TOML.
# Key forms cover bare, "double-quoted", and 'single-quoted' keys (all legal
# TOML). Occurrences *inside* a multi-line string value (e.g. example lines
# embedded in developer_instructions) must survive: a line-anchored regex
# cannot tell a root key from a continuation line of a string value, so the
# strip runs as a scanner that tracks multi-line delimiters and only removes
# top-level keys -- a key whose value spans a multi-line string is removed
# together with that value.
_ROOT_MODEL_INSTRUCTIONS_KEY_RE = re.compile(
    r'[ \t]*(?:"model_instructions_file"|\'model_instructions_file\'|model_instructions_file)[ \t]*=[ \t]*'
)


def _strip_root_model_instructions_key(extra_config_toml: str) -> str:
    """Remove root-level ``model_instructions_file`` assignments.

    Lines inside an unterminated multi-line string value (``\"\"\"``/``'''``)
    are passed through untouched, so a same-named example line inside a
    multi-line ``developer_instructions`` value survives; a key whose own
    value opens an unterminated multi-line string is dropped together with
    the rest of that value.
    """
    out: list[str] = []
    in_ml: str | None = None  # active multi-line delimiter while inside a value
    dropping = False  # True while the tracked value belongs to the removed key
    for line in extra_config_toml.splitlines(keepends=True):
        if in_ml is not None:
            if not dropping:
                out.append(line)
            if line.count(in_ml) % 2 == 1:
                in_ml = None
                dropping = False
            continue
        if _ROOT_MODEL_INSTRUCTIONS_KEY_RE.match(line):
            for delim in ('"""', "'''"):
                if line.count(delim) % 2 == 1:
                    in_ml = delim
                    dropping = True
                    break
            continue
        out.append(line)
        for delim in ('"""', "'''"):
            if line.count(delim) % 2 == 1:
                in_ml = delim
                break
    return ''.join(out)


def _should_enable_legacy_freeform_apply_patch(version: str, configured: bool | None) -> bool:
    """Resolve the legacy setting while preserving pre-0.131 Codex behavior."""
    if configured is not None:
        return configured
    if version == 'latest':
        return False
    try:
        return Version(version) < _LEGACY_FREEFORM_APPLY_PATCH_CONFIG_REMOVED_VERSION
    except InvalidVersion:
        logger.warning(
            'Cannot infer legacy apply_patch support from Codex version %r; preserving the enabled default. '
            'Set experimental_use_freeform_apply_patch explicitly to silence this warning.',
            version,
        )
        return True


def _write_model_catalog_json(codex_home: Path, model_catalog_json: dict[str, Any]) -> Path:
    """Materialize an inline Codex model catalog in the per-run home.

    Local writer, used by the interactive PTY path. The batch path goes through the
    execution transport instead, so a remote target gets the file on its own disk.
    """
    catalog_path = codex_home / 'model_catalog.json'
    catalog_path.write_text(
        json.dumps(model_catalog_json, ensure_ascii=False, indent=2) + '\n',
        encoding='utf-8',
    )
    return catalog_path


def _override_model_catalog_entry(
    catalog: dict[str, Any],
    model: str,
    overrides: dict[str, Any],
) -> dict[str, Any]:
    """Clone or update one bundled catalog entry for the requested model."""
    models = catalog.get('models')
    if not isinstance(models, list) or not models:
        raise ValueError('Codex bundled model catalog does not contain a non-empty models list')

    template_slug = overrides.get('template_slug')
    entry_overrides = {key: value for key, value in overrides.items() if key != 'template_slug'}
    if 'slug' in entry_overrides:
        raise ValueError('model_catalog_overrides cannot override the catalog slug')

    template: dict[str, Any] | None = None
    updated_models: list[Any] = []
    for entry in models:
        if not isinstance(entry, dict):
            updated_models.append(entry)
            continue
        if entry.get('slug') == model:
            template = entry
            updated_models.append({**entry, **entry_overrides})
            continue
        if template_slug is not None and entry.get('slug') == template_slug:
            template = entry
        updated_models.append(entry)

    if not any(isinstance(entry, dict) and entry.get('slug') == model for entry in models):
        if template is None:
            candidates = [entry for entry in models if isinstance(entry, dict) and entry.get('slug')]
            if not candidates:
                raise ValueError('Codex bundled model catalog has no usable template entry')
            template = max(candidates, key=lambda entry: (entry.get('priority') or -1, str(entry.get('slug'))))
        updated_models.append({**template, 'slug': model, **entry_overrides})

    return {**catalog, 'models': updated_models}


async def _load_bundled_model_catalog(
    *,
    command_argv: list[str],
    codex_home: Path,
    working_dir: str,
    transport: AgentExecutionTransport,
    extra_env: dict[str, str] | None = None,
) -> dict[str, Any]:
    """Export the installed Codex catalog so a selected entry can be patched."""
    env = {
        **(extra_env or {}),
        'CODEX_HOME': str(codex_home),
        'HOME': os.environ.get('HOME', '/root'),
        'PATH': build_runtime_path(),
    }
    result = await transport.exec_as_agent(
        [*command_argv, 'debug', 'models', '--bundled'],
        working_dir=working_dir,
        timeout=60,
        env=env,
    )
    stdout, stderr, exit_code = unpack_command_result(
        result,
        timeout_message='Timed out exporting the Codex bundled model catalog',
        error_message='Failed to export the Codex bundled model catalog',
    )
    if exit_code != 0:
        raise RuntimeError(f'Failed to export the Codex bundled model catalog: {stderr or stdout}')
    try:
        catalog = json.loads(stdout)
    except json.JSONDecodeError as error:
        raise RuntimeError('Codex returned an invalid bundled model catalog') from error
    if not isinstance(catalog, dict):
        raise RuntimeError('Codex bundled model catalog must be a JSON object')
    return catalog


def _resolve_codex_home(codex_home_name: str | None, *, home: Path | None = None) -> Path:
    """Return a random default home or an explicitly named isolated parity home."""
    if home is None:
        home = Path.home() if is_windows() else Path(os.environ.get('HOME', '/root'))
    if codex_home_name is None:
        return home / CODEX_HOME_DIR_NAME / uuid.uuid4().hex
    if not _CODEX_HOME_NAME_PATTERN.fullmatch(codex_home_name) or codex_home_name in {'.', '..'}:
        raise ValueError(
            'codex_home_name must be a single safe directory name containing only letters, digits, dot, dash, '
            'or underscore'
        )
    return home / CODEX_HOME_DIR_NAME / codex_home_name


def _build_codex_command(argv: list[str]) -> str:
    if not is_windows():
        return build_runtime_command(argv)
    # Windows PowerShell's native argument marshalling drops embedded quotes/empty
    # arguments. Start-Process accepts an already quoted Windows command line.
    executable = "'" + argv[0].replace("'", "''") + "'"
    arguments = "'" + subprocess.list2cmdline(argv[1:]).replace("'", "''") + "'"
    return (
        f'$codexProcess = Start-Process -FilePath {executable} -ArgumentList {arguments} '
        '-NoNewWindow -Wait -PassThru -ErrorAction Stop; exit $codexProcess.ExitCode'
    )


def _resolve_mcp_placeholders(value: Any) -> Any:
    """Substitute ``${NEXUS_*}`` placeholders in an MCP server entry, recursively.

    The worker venv python path is random per sandbox (venv ids are fresh
    ``uuid`` hex), so a task-side ``mcp_servers`` entry cannot name it; it uses
    ``${NEXUS_WORKER_PYTHON}`` instead. Resolution happens here because this
    setup runs inside the worker process, where ``sys.executable`` IS that
    python and ``os.environ['LD_LIBRARY_PATH']`` carries the nix runtime libs it
    needs (codex's own subprocess env strips them via ``get_clean_sandbox_env``,
    so a venv-python MCP server must re-declare its own). Idempotent and
    side-effect free for entries without placeholders.
    """
    if isinstance(value, str):
        value = value.replace('${NEXUS_WORKER_PYTHON}', sys.executable)
        return value.replace('${NEXUS_LD_LIBRARY_PATH}', os.environ.get('LD_LIBRARY_PATH', ''))
    if isinstance(value, list):
        return [_resolve_mcp_placeholders(item) for item in value]
    if isinstance(value, dict):
        return {key: _resolve_mcp_placeholders(item) for key, item in value.items()}
    return value


def _mcp_config_lines(servers: list[dict[str, Any]]) -> list[str]:
    """Render the stdio MCP schema shared with codex_native_bridge."""
    lines: list[str] = []
    for raw_server in servers:
        server = dict(raw_server)
        for key in ('command', 'args', 'env'):
            if key in server:
                server[key] = _resolve_mcp_placeholders(server[key])
        name = str(server.get('name') or '')
        command = str(server.get('command') or '')
        if not name or not command:
            raise ValueError('Each Codex MCP server requires non-empty name and command fields')
        args_value = server.get('args') or []
        if not isinstance(args_value, list):
            raise ValueError(f'Codex MCP server {name!r} args must be a list')
        args = [str(arg) for arg in args_value]
        lines.extend([
            '',
            f'[mcp_servers.{json.dumps(name)}]',
            f'command = {json.dumps(command)}',
            f'args = {json.dumps(args)}',
        ])
        env = server.get('env')
        if env:
            if not isinstance(env, dict):
                raise ValueError(f'Codex MCP server {name!r} env must be a mapping')
            lines.append(
                'env = {'
                + ', '.join(f'{json.dumps(str(key))} = {json.dumps(str(value))}' for key, value in env.items())
                + '}'
            )
        for key in ('cwd', 'startup_timeout_sec', 'tool_timeout_sec', 'enabled', 'required'):
            if key in server:
                lines.append(f'{key} = {json.dumps(server[key])}')
        for key in ('enabled_tools', 'disabled_tools', 'omit_tools_from'):
            if key not in server:
                continue
            tools = server[key]
            if not isinstance(tools, list):
                raise ValueError(f'Codex MCP server {name!r} {key} must be a list')
            lines.append(f'{key} = {json.dumps([str(tool) for tool in tools])}')
    return lines


def build_codex_config_lines(
    *,
    codex_home: Path,
    proxy_base_url: str,
    version: str,
    model: str | None = None,
    model_catalog_json: dict[str, Any] | None = None,
    auto_compact_token_limit: int | None = None,
    compact_prompt: str | None = None,
    experimental_use_freeform_apply_patch: bool | None = None,
    default_mode_request_user_input: bool | None = None,
    trusted_project: str | None = None,
    use_azure_provider: bool = True,
    mcp_servers: list[dict[str, Any]] | None = None,
    extra_config_toml: str | None = None,
    model_instructions_file: str | None = None,
) -> list[str]:
    """Render ``config.toml`` for a Codex run.

    Shared by the batch (``codex exec``) and interactive (PTY) paths so provider,
    model, and MCP wiring cannot drift between them.
    """
    config_lines: list[str] = []
    if model:
        config_lines.append(f'model = {_toml_string(model)}')
    if model_catalog_json is not None:
        catalog_path = _write_model_catalog_json(codex_home, model_catalog_json)
        config_lines.append(f'model_catalog_json = {_toml_string(str(catalog_path))}')
    if auto_compact_token_limit is not None:
        config_lines.append(f'model_auto_compact_token_limit = {int(auto_compact_token_limit)}')
    if _should_enable_legacy_freeform_apply_patch(version, experimental_use_freeform_apply_patch):
        config_lines.append('experimental_use_freeform_apply_patch = true')
    if compact_prompt is not None:
        config_lines.append(f'compact_prompt = {_toml_string(compact_prompt)}')
    if default_mode_request_user_input is not None:
        config_lines.append(
            f'features.default_mode_request_user_input = {str(default_mode_request_user_input).lower()}'
        )
    config_lines.append('model_provider = "nexus"')
    extra_config = (extra_config_toml or '').strip()
    if model_instructions_file is not None:
        # Root-level key; must be emitted before any table header so it cannot
        # be scoped into one. A same-named key in extra_config_toml is removed
        # (not left in place): two identical keys -- however equal their
        # values -- make the TOML invalid, so this root line is the only
        # authority for the key.
        extra_config = _strip_root_model_instructions_key(extra_config).strip()
        config_lines.append(f'model_instructions_file = {_toml_string(model_instructions_file)}')
    if extra_config:
        config_lines.extend(['', extra_config])
    provider_name = 'azure' if use_azure_provider else 'Nexus Proxy'
    config_lines.extend([
        '',
        '[model_providers.nexus]',
        f'name = {_toml_string(provider_name)}',
        f'base_url = {_toml_string(proxy_base_url)}',
        f'env_key = {_toml_string("NEXUS_PROXY_API_KEY")}',
        'wire_api = "responses"',
    ])
    config_lines.extend(_mcp_config_lines(mcp_servers or []))
    if trusted_project is not None:
        config_lines.extend([
            '',
            f'[projects.{_toml_string(trusted_project)}]',
            'trust_level = "trusted"',
        ])
    return config_lines


async def install_codex_binary(
    *,
    install_dir: str,
    version: str,
    npm_registry: str | None,
    binary_url: str | None = None,
    binary_checksum: str | None = None,
) -> Path:
    """Install Codex CLI (npm or direct binary) and return its path."""
    from nexus.extensions.agents.codex.installer import install_codex_cli, install_codex_from_url

    if binary_url:
        return await install_codex_from_url(binary_url, install_dir, binary_checksum)
    return await install_codex_cli(install_dir=install_dir, version=version, npm_registry=npm_registry)


def resolve_codex_launch_argv(*, codex_path: Path, install_dir: str, binary_url: str | None) -> list[str]:
    """Return argv that starts Codex, mirroring the batch launch resolution."""
    if binary_url:
        return [str(codex_path)]
    node_path = find_nexus_runtime_binary('node')
    if not node_path:
        raise RuntimeError('Node.js is required but not found in runtime')
    js_entry = resolve_npm_bin_entry(install_dir, '@openai/codex', 'codex')
    if not js_entry or not js_entry.exists():
        raise RuntimeError(f'Codex CLI JS entry point not found under {install_dir}')
    return [node_path, str(js_entry)]


async def setup_and_run_codex(
    task: str,
    working_dir: str = '/workspace',
    timeout: float = 3600,
    version: str | None = DEFAULT_CODEX_VERSION,
    model: str | None = None,
    model_catalog_json: dict[str, Any] | None = None,
    model_catalog_overrides: dict[str, Any] | None = None,
    npm_registry: str | None = DEFAULT_NPM_REGISTRY,
    binary_url: str | None = None,
    binary_checksum: str | None = None,
    extra_env: dict[str, str] | None = None,
    mcp_servers: list[dict[str, Any]] | None = None,
    extra_cli_args: list[str] | None = None,
    developer_instructions: str | None = None,
    extra_config_toml: str | None = None,
    model_instructions_file: str | None = None,
    experimental_use_freeform_apply_patch: bool | None = None,
    responses_to_openai_converter: Callable[[dict[str, Any]], dict[str, Any]] | None = None,
    openai_to_responses_events_converter: Callable[[dict[str, Any]], list[dict[str, Any]]] | None = None,
    wrap_plaintext_reasoning_content: bool = True,
    use_azure_provider: bool = True,
    auto_compact_token_limit: int | None = None,
    mode: Literal['exec', 'exec_json', 'app_server'] = 'exec_json',
    goal_mode: bool = False,
    goal_token_budget: int | None = None,
    compact_prompt: str | None = None,
    codex_home_name: str | None = None,
    normalize_exec_command_metadata: bool = False,
    execution_target: dict[str, Any] | None = None,
    logs_dir: str | None = None,
    supports_tool_references: bool | None = None,
    tool_search_output_format: Literal['tool_reference', 'tool_search_output'] = 'tool_reference',
) -> dict[str, Any]:
    """Install Codex CLI, start the Responses proxy, and execute the task.

    ``use_azure_provider``: when True (default), the config.toml provider name is "azure", which
    makes Codex treat this as an Azure Responses endpoint (store=true + original reasoning item ids
    re-attached on replay) -- required for GPT-5.5 encrypted_content reasoning to survive across
    turns. Set False for models that do NOT use encrypted reasoning (e.g. HY3, which has plaintext
    reasoning_content and no reasoning id to preserve): a non-azure provider name keeps store=false,
    so Codex routes auto-compaction to LOCAL summarization instead of native Responses compaction.
    HyTuner training must also use False: the local summary completion retains token ids/logprobs,
    while a provider-native opaque compaction item is not a trainable summary segment.
    Polarist/trpc GPT evals must also use False (stateless gateway: ``rs_ not found`` on id replay).
    Toggling this does NOT affect tool-call parsing -- function/custom tool calls are paired by
    `call_id`, which `set_id(None)` never clears (only the optional `id` field is dropped on
    non-azure replay).

    ``auto_compact_token_limit``: when set, written as ``model_auto_compact_token_limit`` in
    config.toml -- the token threshold at which Codex triggers auto-compaction. Codex resolves the
    effective trigger as ``min(this, context_window * 90%)`` (see codex openai_models.rs
    auto_compact_token_limit()), so a value >= 90% of the model's context window is a no-op that
    just inherits the default 90% clamp; to actually push compaction later, also raise
    ``model_context_window``. Pass None to omit the line entirely (pure catalog default).

    ``mode`` selects plain ``codex exec``, ``codex exec --json``, or the app-server stdio driver.
    The default preserves the previous JSON trace behavior.

    ``model_catalog_json``: optional inline model-catalog payload. It is materialized under the
    unique per-run CODEX_HOME and selected through Codex's official ``model_catalog_json`` config
    field, so sandbox workers do not need access to a host-local catalog path.

    ``model_catalog_overrides``: optional entry fields applied to the requested model in Codex's
    bundled catalog. The runtime clones a bundled entry when the model slug is not present, so
    callers can select protocol metadata without shipping a model-specific catalog file.

    ``experimental_use_freeform_apply_patch``: explicitly enable or disable Codex's legacy setting.
    When None, enable it for versions before 0.131.0-alpha.15 and omit it for newer versions, where
    apply_patch availability comes from model metadata instead.

    ``compact_prompt``: optional custom local-compaction instruction. It is written into Codex's
    config and passed to the Responses proxy so the matching request can be identified later.

    ``codex_home_name`` selects a stable directory under the existing Codex home root for isolated
    parity runtimes. Unset preserves UUID isolation. Reusing a name overwrites its run files.

    ``logs_dir`` enables native session persistence and writes merged CLI output to codex.txt.
    Only sessions are copied from CODEX_HOME, never config or credentials. The directory is on
    the outer Runtime; execution-target log mounts make it visible to the inner service.

    ``normalize_exec_command_metadata`` normalizes only random chunk IDs and measured wall times
    in exec_command outputs before tokenization; it is disabled by default.

    ``wrap_plaintext_reasoning_content``: origin/main default True. When False, the default
    Responses converter does not mint a reasoning item from ``reasoning_content``. Native
    ``reasoning_items`` are still forwarded. Ignored when a custom events converter is supplied.

    ``supports_tool_references`` enables inline tool references and deferred schemas explicitly.
    None and False use standard Chat schemas and text search results. A custom request converter
    owns this capability itself and cannot be combined with the override.
    ``tool_search_output_format`` selects tool_reference for HY4 text templates or
    tool_search_output with embedded schemas for HY4 serving/VLM templates. Embedded schemas
    require supports_tool_references=True.
    """
    from nexus.extensions.agents.codex.installer import install_codex_cli, install_codex_from_url
    from nexus.runtime.context import require_runtime_control
    from nexus.runtime.executors.command import CommandRunStatus, RunCommandRequest, run_command
    from nexus.runtime.server.services.responses_proxy import ResponsesProxy, responses_to_openai

    if tool_search_output_format not in ('tool_reference', 'tool_search_output'):
        raise ValueError(
            f'unsupported tool_search_output_format={tool_search_output_format!r}; '
            'choose tool_reference or tool_search_output'
        )
    if supports_tool_references is not None and responses_to_openai_converter is not None:
        raise ValueError('supports_tool_references requires the default Responses converter; remove one override')
    if tool_search_output_format != 'tool_reference' and responses_to_openai_converter is not None:
        raise ValueError('tool_search_output_format requires the default Responses converter; remove one override')
    if tool_search_output_format == 'tool_search_output' and supports_tool_references is not True:
        raise ValueError('tool_search_output format requires supports_tool_references=True')

    if version is None:
        version = DEFAULT_CODEX_VERSION

    if logs_dir is not None:
        if mode == 'app_server':
            raise ValueError('Codex app_server mode does not support logs_dir')
        if is_windows():
            raise ValueError('Codex logs_dir currently requires a Linux Runtime')
        if '--ephemeral' in (extra_cli_args or []):
            raise ValueError('Codex logs_dir requires native sessions; remove --ephemeral')
        Path(logs_dir).mkdir(parents=True, exist_ok=True)

    rt = require_runtime_control()

    start_time = time.time()
    transport = AgentExecutionTransport(execution_target)
    transport.validate()
    invocation_id = uuid.uuid4().hex
    try:
        await transport.initialize()
        install_dir = transport.install_dir(agent_name='codex-cli', default=DEFAULT_INSTALL_DIR)
        if is_windows():
            working_dir = os.path.expandvars(os.path.expanduser(working_dir))
        await transport.mkdir(working_dir)
        home = str(Path.home()) if is_windows() else os.environ.get('HOME', '/root')
        home = str(transport.home_dir(invocation_id=invocation_id, default=home))
        codex_home = _resolve_codex_home(codex_home_name, home=Path(home))
        await transport.mkdir(codex_home)
        output_file = codex_home / 'last_message.txt'
        config_path = codex_home / 'config.toml'
        install_kwargs: dict[str, Any] = {'transport': transport} if not transport.is_local else {}
        if binary_url:
            codex_path = await install_codex_from_url(binary_url, install_dir, binary_checksum, **install_kwargs)
        else:
            codex_path = await install_codex_cli(
                install_dir=install_dir,
                version=version,
                npm_registry=npm_registry,
                **install_kwargs,
            )
    except Exception as e:
        logger.error(f'Failed to install Codex CLI: {e}')
        return {
            'success': False,
            'output': '',
            'stderr': f'Failed to install Codex CLI: {e}',
            'exit_code': -1,
            'execution_time': time.time() - start_time,
            'failure_kind': 'infrastructure',
            'failure_stage': 'install',
        }

    if model_catalog_overrides is not None:
        if model_catalog_json is not None:
            raise ValueError('model_catalog_json and model_catalog_overrides are mutually exclusive')
        if not model:
            raise ValueError('model is required when model_catalog_overrides is set')

    npm_binary = 'codex.cmd' if transport.is_local and is_windows() else 'codex'
    managed_codex_path = Path(install_dir) / 'node_modules' / '.bin' / npm_binary
    # Preinstalled CLIs may be native binaries and do not belong to our npm prefix.
    if binary_url or codex_path != managed_codex_path:
        launch_argv = [str(codex_path)]
    elif not transport.is_local:
        node_path = await transport.find_binary('node')
        js_entry = await transport.file('npm_entry', install_dir, package='@openai/codex', name='codex')
        launch_argv = [node_path, js_entry]
    else:
        node_path = find_nexus_runtime_binary('node')
        if not node_path:
            raise RuntimeError('Node.js is required but not found in runtime')
        js_entry = resolve_npm_bin_entry(install_dir, '@openai/codex', 'codex')
        if not js_entry or not js_entry.exists():
            raise RuntimeError(f'Codex CLI JS entry point not found under {install_dir}')
        launch_argv = [node_path, str(js_entry)]

    if model_catalog_overrides is not None:
        if model is None:
            raise ValueError('model is required when model_catalog_overrides is set')
        bundled_catalog = await _load_bundled_model_catalog(
            command_argv=launch_argv,
            codex_home=codex_home,
            working_dir=working_dir,
            transport=transport,
            extra_env=extra_env,
        )
        model_catalog_json = _override_model_catalog_entry(bundled_catalog, model, model_catalog_overrides)
        logger.info(
            'Codex catalog overrides applied for model=%s: %s',
            model,
            sorted(model_catalog_overrides),
        )

    proxy_port = find_free_port()
    proxy_kwargs: dict[str, Any] = {}
    if responses_to_openai_converter is not None:
        proxy_kwargs['responses_to_openai_converter'] = responses_to_openai_converter
    elif supports_tool_references is not None:
        proxy_kwargs['responses_to_openai_converter'] = partial(
            responses_to_openai,
            supports_tool_references=supports_tool_references,
            tool_search_output_format=tool_search_output_format,
        )
    if openai_to_responses_events_converter is not None:
        proxy_kwargs['openai_to_responses_events_converter'] = openai_to_responses_events_converter
    elif not wrap_plaintext_reasoning_content:
        from nexus.runtime.server.services.responses_proxy.converter import openai_to_responses_events

        proxy_kwargs['openai_to_responses_events_converter'] = partial(
            openai_to_responses_events,
            wrap_plaintext_reasoning_content=False,
        )

    proxy = ResponsesProxy(
        sio=rt.sio,
        listen_port=proxy_port,
        listen_host='127.0.0.1',
        # The command owns the active-time budget; a second wall-clock deadline
        # here would still expire while the host parks an aborted rollout.
        request_timeout=None,
        codex_version=version,
        compact_prompt=compact_prompt,
        normalize_exec_command_metadata=normalize_exec_command_metadata,
        **proxy_kwargs,
    )

    try:
        await proxy.start()
        logger.debug('Responses Proxy started')
    except Exception as e:
        logger.error(f'Failed to start Responses Proxy: {e}')
        return {
            'success': False,
            'output': '',
            'stderr': f'Failed to start Responses Proxy: {e}',
            'exit_code': -1,
            'execution_time': time.time() - start_time,
            'failure_kind': 'infrastructure',
            'failure_stage': 'proxy',
        }

    try:
        # The model catalog is written through the transport, so a remote execution target
        # gets the file on its own filesystem. build_codex_config_lines() writes it locally,
        # which is why model_catalog_json is rendered here and not passed down.
        catalog_lines: list[str] = []
        if model_catalog_json is not None:
            catalog_path = codex_home / 'model_catalog.json'
            await transport.write_text(
                catalog_path, json.dumps(model_catalog_json, ensure_ascii=False, indent=2) + '\n'
            )
            catalog_lines.append(f'model_catalog_json = {_toml_string(str(catalog_path))}')
        # Token threshold that triggers auto-compaction. Codex clamps the effective limit to
        # min(this, context_window * 90%), so values >= 90% of the context window just inherit the
        # default clamp (to push compaction later, raise model_context_window too). Omit when None.
        # Provider name "azure" makes Codex treat this as an Azure Responses endpoint, which sets
        # store=true and re-attaches the original reasoning item ids on replay. That is required
        # for GPT-5.5 encrypted_content reasoning to continue across turns (otherwise the gateway
        # rejects it with invalid_encrypted_content). is_azure_responses_provider() keys off either
        # the name == "azure" or azure markers in base_url; our base_url is a local 127.0.0.1 proxy
        # with no such markers, so the name alone decides it.
        # Codex >=0.136 also uses this marker to enable remote compaction. ResponsesProxy routes
        # its compaction trigger to the native compact endpoint and preserves the returned opaque
        # compaction item when Codex replays the replacement history.
        # For models without encrypted reasoning (e.g. HY3 plaintext reasoning_content), pass
        # use_azure_provider=False: store stays false (the dropped item `id` is irrelevant -- there
        # is no encrypted reasoning to continue), and auto-compaction falls back to LOCAL
        # summarization over the ordinary /v1/responses endpoint the proxy already serves. HyTuner
        # training uses this same local path because its summary segment needs token ids/logprobs.
        # Polarist GPT evals also pass False (stateless gateway).
        config_lines = build_codex_config_lines(
            codex_home=codex_home,
            proxy_base_url=f'http://127.0.0.1:{proxy_port}/v1',
            version=version,
            model=model,
            auto_compact_token_limit=auto_compact_token_limit,
            compact_prompt=compact_prompt,
            experimental_use_freeform_apply_patch=experimental_use_freeform_apply_patch,
            use_azure_provider=use_azure_provider,
            mcp_servers=mcp_servers,
            extra_config_toml=extra_config_toml,
            model_instructions_file=model_instructions_file,
        )
        if catalog_lines:
            # Keep the catalog key adjacent to `model`, matching the order the shared
            # builder uses, and ahead of every table header (root keys must precede tables).
            insert_at = 1 if model else 0
            config_lines[insert_at:insert_at] = catalog_lines
        logger.debug(
            'Codex config.toml provider name=%r (azure=%s) auto_compact_token_limit=%s',
            'azure' if use_azure_provider else 'Nexus Proxy',
            use_azure_provider,
            auto_compact_token_limit,
        )
        await transport.write_text(config_path, '\n'.join(config_lines) + '\n')

        await transport.remove(output_file)

        codex_env = {
            'CODEX_HOME': str(codex_home),
            'NEXUS_AGENT_PROXY_URL': f'http://127.0.0.1:{proxy_port}',
            'NEXUS_PROXY_API_KEY': DEFAULT_PROXY_API_KEY,
            'HOME': home,
            'PATH': build_runtime_path() if transport.is_local else transport.agent_env().get('PATH', ''),
        }

        codex_env.update(extra_env or {})
        codex_env = transport.agent_env(codex_env)

        timeout_control_id = get_current_span_id()
        if timeout_control_id is None:
            raise RuntimeError('Codex active-time timeout requires a distributed trace context')

        if mode == 'app_server':
            from .app_server_runtime import run_codex_app_server

            result = await run_codex_app_server(
                launch_argv=launch_argv,
                timeout_control_id=timeout_control_id,
                working_dir=working_dir,
                env=codex_env,
                task=task,
                timeout=timeout,
                model=model,
                extra_cli_args=extra_cli_args,
                developer_instructions=developer_instructions,
                goal_mode=goal_mode,
                goal_token_budget=goal_token_budget,
            )
            result['execution_time'] = time.time() - start_time
            return result
        if mode not in {'exec', 'exec_json'}:
            raise ValueError(f'Unsupported Codex execution mode: {mode!r}')

        effective_task = task
        if developer_instructions:
            effective_task = f'{developer_instructions}\n\n---\nUSER TASK:\n{task}'

        # Deliberately NOT passing --ephemeral. multi_agent_version=v2 spawns a subagent by reconstructing the parent context from that persistent store: an omitted fork_turns defaults to "all", and load_agent_model_context then fails with `collab spawn failed: no thread with id: <root thread id>`.
        cmd = [
            *launch_argv,
            'exec',
            '--dangerously-bypass-approvals-and-sandbox',
            '--skip-git-repo-check',
        ]
        # --json makes codex print its thread-event stream (JSONL) to stdout. It does NOT change
        # the final last_message.txt output or the exit code, so the existing success/output path
        # is unaffected; we only additionally capture stdout for host-side trace rebuild.
        emit_json_trace = mode == 'exec_json'
        if emit_json_trace:
            cmd.append('--json')
        if extra_cli_args:
            cmd.extend(extra_cli_args)
        cmd.extend([
            '-C',
            working_dir,
            '-o',
            str(output_file),
            effective_task,
        ])

        async with transport.prepare(
            cmd,
            working_dir=working_dir,
            invocation_id=invocation_id,
            env=codex_env,
            proxy_files=[config_path],
        ) as prepared:
            shell_cmd = _build_codex_command(prepared.command) if transport.is_local else shlex.join(prepared.command)
            if logs_dir is not None:
                log_path = shlex.quote(str(Path(logs_dir) / 'codex.txt'))
                shell_cmd = f'set -o pipefail; {shell_cmd} 2>&1 | tee {log_path}'
            logger.debug(f'Running Codex: {shell_cmd[:200]}...')
            result = await run_command(
                RunCommandRequest(
                    command=shell_cmd,
                    cwd=prepared.cwd,
                    timeout=timeout,
                    timeout_control_id=timeout_control_id,
                    extra_env=prepared.env,
                )
            )

        _stdout, stderr, exit_code = unpack_command_result(
            result,
            timeout_message='Execution timed out',
            error_message='Command execution error',
        )

        # unpack_command_result folds both TimeLimitExceeded and Error into exit_code=-1, so the
        # host side (run_codex) cannot tell a timeout apart from a generic failure. Surface the
        # timeout explicitly here so run_codex can classify the termination reason as TASK_TIMEOUT.
        timed_out = getattr(result, 'status', None) == CommandRunStatus.TimeLimitExceeded

        # With --json, result.stdout is the JSONL event stream, NOT the final message. Only fall
        # back to stdout-as-output when we are NOT in json mode, so the plain-text output contract
        # is preserved; the JSONL is returned separately under json_events_stdout instead.
        output = ''
        saved_output = await transport.read_text(output_file)
        if saved_output is not None:
            output = saved_output
        elif result.stdout and not emit_json_trace:
            output = result.stdout

        execution_time = time.time() - start_time
        logger.info(f'Codex finished with exit code {exit_code} in {execution_time:.2f}s')
        failure_kind, failure_stage = classify_command_failure(
            result,
            process_failed=exit_code != 0,
        )
        # Log mode merges CLI stderr into stdout, including docker compose exec errors.
        diagnostic_output = (result.stdout or '') if logs_dir is not None else stderr
        transport_failure_kind, transport_failure_stage = transport.classify_failure(
            exit_code, stderr + diagnostic_output
        )
        if transport_failure_kind == 'infrastructure':
            failure_kind, failure_stage = transport_failure_kind, transport_failure_stage

        return {
            'success': exit_code == 0,
            'output': output,
            'stderr': stderr,
            'exit_code': exit_code,
            'execution_time': execution_time,
            'timed_out': timed_out,
            **({'failure_kind': failure_kind, 'failure_stage': failure_stage} if failure_kind is not None else {}),
            # Present only in json mode; empty string otherwise so the key is always well-typed.
            'json_events_stdout': (result.stdout or '') if emit_json_trace else '',
        }
    except ExecutionTransportError as exc:
        logger.error('Codex execution transport failed: %s', exc)
        return {
            'success': False,
            'output': '',
            'stderr': str(exc),
            'exit_code': -1,
            'execution_time': time.time() - start_time,
            'timed_out': False,
            'failure_kind': 'infrastructure',
            'failure_stage': 'execution_transport',
            'json_events_stdout': '',
        }
    finally:
        try:
            if logs_dir is not None:
                await transport.copy_directory_to_local(codex_home / 'sessions', Path(logs_dir) / 'sessions')
        except (OSError, ExecutionTransportError):
            logger.exception('Failed to copy Codex sessions to logs_dir')
        finally:
            logger.debug('Stopping Responses Proxy...')
            await proxy.stop()


__all__ = [
    'CODEX_HOME_DIR_NAME',
    'DEFAULT_INSTALL_DIR',
    'CodexResult',
    'build_codex_config_lines',
    'install_codex_binary',
    'resolve_codex_launch_argv',
    'setup_and_run_codex',
]
