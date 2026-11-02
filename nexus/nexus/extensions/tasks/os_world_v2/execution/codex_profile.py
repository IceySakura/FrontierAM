"""Task-owned Codex options, resources and execution policy."""

from __future__ import annotations

import hashlib
import json
import re
import time

from pathlib import Path
from typing import Any, cast

from packaging.version import InvalidVersion, Version
from pydantic import ConfigDict, Field, PositiveInt, TypeAdapter, model_validator

from nexus.core import AgentRunInput
from nexus.core.agent import WithUserPrompt
from nexus.extensions.agents.codex.json_events import _terminal_error_message, parse_codex_jsonl, parse_codex_stream
from nexus.extensions.agents.codex.native_bridge.adapters.shell import build_codex_config
from nexus.extensions.agents.codex.native_bridge.adapters.types import CodexInstallSpec, CodexRunRequest
from nexus.extensions.tasks.os_world_v2.execution.guest_network import guest_proxy_env
from nexus.extensions.tasks.os_world_v2.execution.guide import render_guide
from nexus.runtime.providers.ags.osworld import require_ags_osworld_runtime


ASSETS = Path(__file__).resolve().parents[1] / 'assets' / 'codex'
GUIDE_FILES = {'v1': 'guide_codex_linux.md'}
CATALOG_FILES = {
    'gpt6_catalog_v8': 'gpt6_catalog_v8.json',
    'gpt56_sol_catalog': 'gpt56_sol_catalog.json',
}


class OSWorldCodexOptions(AgentRunInput, WithUserPrompt):
    model_config = ConfigDict(extra='forbid')
    runtime_name: str = 'main'
    osworld_task_id: str = ''
    osworld_domain: str = ''
    osworld_client_password: str = 'osworld-public-evaluation'
    trajectory_dir: str = ''
    model: str = 'gpt-6-astra'
    version: str = '0.154.0'
    timeout: float = Field(default=5400, gt=0)
    working_dir: str = '/home/user'
    guide_text: str
    model_catalog_json: str | dict[str, Any]
    guide_version: str | None = None
    model_catalog_preset: str | None = None
    context_window: int | None = Field(default=None, gt=0)
    max_output_tokens: int | None = Field(default=None, gt=0)
    context_token_margin: int | None = Field(default=None, gt=0)
    catalog_source_sha256: str = ''
    binary_url: str | None = None
    binary_checksum: str | None = None
    npm_registry: str = 'http://mirrors.tencent.com/npm/'
    auto_compact_token_limit: int = Field(default=194048, gt=0)
    reasoning_effort: str = 'none'
    extra_config_toml: str = '[tools]\n\n[tools.update_plan]\nenabled = true\n\n[features]\nunified_image_budget = true'

    @model_validator(mode='before')
    @classmethod
    def resolve_assets(cls, value):
        if not isinstance(value, dict):
            return value
        value = dict(value)
        # Explicit content wins, including an empty guide. Invalid explicit
        # content must fail validation, never silently fall back to a preset.
        if value.get('guide_text') is None and value.get('guide_version') is not None:
            version = value['guide_version']
            if not isinstance(version, str) or version not in GUIDE_FILES:
                raise ValueError(f'Unknown Codex guide_version: {version!r}; expected v1')
            value['guide_text'] = (ASSETS / GUIDE_FILES[version]).read_text(encoding='utf-8')
        preset = value.get('model_catalog_preset')
        if value.get('model_catalog_json') is None and preset is not None:
            if not isinstance(preset, str) or preset not in CATALOG_FILES:
                raise ValueError(f'Unknown Codex model_catalog_preset: {preset!r}')
            catalog = json.loads((ASSETS / CATALOG_FILES[preset]).read_text(encoding='utf-8'))
            model = value.get('model', 'gpt-6-astra')
            entries = [entry for entry in catalog['models'] if entry['slug'] == model]
            if len(entries) != 1:
                raise ValueError('Codex catalog preset must contain exactly one selected model')
            positive = TypeAdapter(PositiveInt)
            window, output, margin = (
                positive.validate_python(value[key] if value.get(key) is not None else default)
                for key, default in (
                    ('context_window', 262144),
                    ('max_output_tokens', 64000),
                    ('context_token_margin', 4096),
                )
            )
            compact = positive.validate_python(value.get('auto_compact_token_limit', window - output - margin))
            if compact <= 0 or compact > window - output - margin:
                raise ValueError('Codex compact limit must reserve output tokens and context margin')
            entries[0].update(context_window=window, max_context_window=window, auto_compact_token_limit=compact)
            value['auto_compact_token_limit'] = compact
            value['model_catalog_json'] = json.dumps(catalog, ensure_ascii=False)
        return value

    @model_validator(mode='after')
    def validate_catalog(self):
        raw = (
            self.model_catalog_json
            if isinstance(self.model_catalog_json, str)
            else json.dumps(self.model_catalog_json, ensure_ascii=False)
        )
        if not self.catalog_source_sha256:
            self.catalog_source_sha256 = hashlib.sha256(raw.encode()).hexdigest()
        self.model_catalog_json = normalize_catalog(self.model_catalog_json, self.model, self.version)
        # Apply local-compaction policy structurally, never by appending a key
        # under whichever TOML table a caller happened to write last.
        try:
            import tomllib
        except ModuleNotFoundError:
            import tomli as tomllib

        parsed = tomllib.loads(self.extra_config_toml)
        features = parsed.get('features', {})
        if features.get('remote_compaction_v2') is True:
            raise ValueError('OSWorld profile requires local compaction (remote_compaction_v2=false)')
        if 'remote_compaction_v2' not in features:
            if '[features]' in self.extra_config_toml:
                self.extra_config_toml = self.extra_config_toml.replace(
                    '[features]', '[features]\nremote_compaction_v2 = false', 1
                )
            else:
                self.extra_config_toml += '\n[features]\nremote_compaction_v2 = false\n'
        return self


def normalize_catalog(value: str | dict[str, Any], model: str, version: str) -> str:
    """Accept JSON strings and objects while preserving valid raw catalog content."""
    raw = value if isinstance(value, str) else json.dumps(value, ensure_ascii=False)
    catalog = json.loads(raw)
    entries = [entry for entry in catalog.get('models', []) if entry.get('slug') == model]
    if len(entries) != 1 or entries[0].get('use_responses_lite') or entries[0].get('tool_mode') not in {None, 'direct'}:
        raise ValueError('Codex catalog must contain the selected model with classic tools (not code-mode/lite)')
    try:
        modern = Version(version) >= Version('0.154.0')
    except InvalidVersion as exc:
        raise ValueError('OSWorld requires an explicit supported Codex version') from exc
    mode = 'direct' if modern else None
    if entries[0].get('tool_mode') != mode:
        entries[0]['tool_mode'] = mode
        return json.dumps(catalog, ensure_ascii=False)
    return raw


def classify_exit(rc: int, timed_out: bool, terminal: str | None, final: str | None, trajectory: list) -> str:
    # An API/bridge failure is never converted to a scored success by a stale
    # completion marker or by the outer wall clock expiring at the same time.
    if terminal in {'failed', 'error'}:
        return 'tool_error'
    if timed_out:
        return 'budget_exhausted' if any(item.get('kind') != 'error' for item in trajectory) else 'tool_error'
    if rc != 0 or not trajectory:
        return 'tool_error'
    if re.search(r'^\s*TASK_INFEASIBLE\s*$', final or '', flags=re.MULTILINE):
        return 'infeasible'
    return 'finished'


def _instruction(prompt: Any) -> str:
    if isinstance(prompt, str):
        return prompt
    # As in the CC harness, CLI takes its own screenshot through view_image.
    return '\n'.join(p['text'] for p in prompt if isinstance(p, dict) and p.get('type') == 'text')


async def _execute_osworld(*, runtime, launch, timeout, inp: OSWorldCodexOptions):
    from nexus.extensions.tasks.os_world_v2.execution.codex import TaskCodexAdapter

    runtime = require_ags_osworld_runtime(runtime)
    # The task resolved this (environment first, then its own field); this agent reads no
    # environment of its own, so an unset root simply means no dump was asked for.
    root = inp.trajectory_dir
    trajectory_dir = Path(root) if root else None
    if trajectory_dir:
        trajectory_dir.mkdir(parents=True, exist_ok=True)
    adapter = TaskCodexAdapter(runtime, diagnostic_dir=trajectory_dir)
    from nexus.extensions.tasks.os_world_v2.outputs import record_agent_output

    try:
        executable = await adapter.ensure_codex(
            CodexInstallSpec(
                version=inp.version,
                npm_registry=inp.npm_registry,
                binary_url=inp.binary_url,
                binary_checksum=inp.binary_checksum,
            )
        )
        version_result = await runtime.run_command(
            command=f'{executable} --version', timeout=30, extra_env=adapter.execution_env
        )
        if version_result.return_code != 0 or (version_result.stdout or '').strip() != f'codex-cli {inp.version}':
            raise RuntimeError('Installed Codex version differs from the pinned version')
    except Exception as exc:
        record_agent_output(
            {
                'success': False,
                'error': f'{type(exc).__name__}: {exc}',
                'failure_stage': 'install',
            },
            harness='codex',
        )
        raise
    extra_env = {
        **adapter.execution_env,
        'DISPLAY': ':0',
        'NO_PROXY': 'localhost,127.0.0.1,::1',
        'no_proxy': 'localhost,127.0.0.1,::1',
    }
    if getattr(runtime, 'osworld_use_proxy', False) is True:
        extra_env.update(guest_proxy_env(extra_env))
    req = CodexRunRequest(
        codex_executable=executable,
        launch=launch,
        task=render_guide(inp.guide_text, inp.osworld_client_password).rstrip()
        + '\n\n'
        + _instruction(inp.user_prompt),
        working_dir=inp.working_dir,
        timeout=timeout,
        version=inp.version,
        npm_registry=inp.npm_registry,
        model=inp.model,
        model_catalog_json=cast(str, inp.model_catalog_json),  # Normalized by validate_catalog.
        extra_cli_args=['-c', f'model_reasoning_effort={json.dumps(inp.reasoning_effort)}'],
        extra_config_toml=inp.extra_config_toml,
        extra_env=extra_env,
        auto_compact_token_limit=inp.auto_compact_token_limit,
        use_azure_provider=False,
        emit_json_trace=True,
        record_shell_history=True,
        strict_cleanup=True,
    )
    if trajectory_dir:
        (trajectory_dir / 'codex_config.toml').write_text(build_codex_config(req, '/tmp/codex-home-audit'))
        (trajectory_dir / 'codex_audit.json').write_text(
            json.dumps(
                {
                    'version': inp.version,
                    'model': inp.model,
                    'agent_timeout_seconds': timeout,
                    'max_turns': None,
                    'max_continues': 0,
                    'transport': 'responses',
                    'guide_sha256': hashlib.sha256(inp.guide_text.encode()).hexdigest(),
                    'catalog_sha256': hashlib.sha256(cast(str, inp.model_catalog_json).encode()).hexdigest(),
                    'catalog_source_sha256': inp.catalog_source_sha256,
                },
                indent=2,
            )
        )
    result = await adapter.run_codex(req)
    trajectory, final, terminal = parse_codex_stream(result.stdout)
    status = classify_exit(result.exit_code, result.timed_out, terminal, final, trajectory)
    if result.failure_kind == 'infrastructure':
        status = 'tool_error'
    stage_output = record_agent_output(
        {
            'success': status != 'tool_error',
            'exit_status': status,
            'exit_code': result.exit_code,
            'codex_return_code': result.exit_code,
            'main_trajectory': trajectory,
            'final_answer': final,
            'stderr': result.stderr,
            'failure_kind': result.failure_kind,
            'failure_stage': result.failure_stage,
            'budget_stop_confirmed': (
                result.timed_out
                and result.failure_stage == 'timeout'
                and adapter.diagnostics.get('termination_confirmed') is True
                and not adapter.diagnostics.get('cleanup_error')
            ),
            'terminal_api_error': (
                _terminal_error_message(parse_codex_jsonl(result.stdout))
                or result.stderr
                or f'Codex terminal event: {terminal}'
            )
            if terminal in {'failed', 'error'}
            else None,
        },
        harness='codex',
    )
    if trajectory_dir:
        (trajectory_dir / 'codex_stream.jsonl').write_text(result.stdout)
        (trajectory_dir / 'codex_stderr.txt').write_text(result.stderr)
        (trajectory_dir / 'codex_trajectory.json').write_text(
            json.dumps(
                {
                    'trajectory': trajectory,
                    'exit_status': status,
                    'return_code': result.exit_code,
                    'terminal_status': terminal,
                    'final_text': final,
                    'timed_out': result.timed_out,
                    'failure_kind': result.failure_kind,
                    'failure_stage': result.failure_stage,
                },
                ensure_ascii=False,
                indent=2,
            )
        )
    if status == 'tool_error' and not (stage_output is not None and stage_output.exit_code == 0):
        raise RuntimeError(
            f'Codex infrastructure/CLI failure: rc={result.exit_code}, terminal={terminal}, '
            f'timed_out={result.timed_out}, failure_kind={result.failure_kind}, '
            f'failure_stage={result.failure_stage}; stderr_tail={result.stderr[-2000:]!r}'
        )
    return {
        'success': True,
        'output': final or '',
        'stderr': result.stderr,
        'exit_code': result.exit_code,
        'execution_time': 0.0,
        'timed_out': result.timed_out,
        'json_events_stdout': result.stdout,
        'main_trajectory': trajectory,
        'exit_status': status,
        'iterations': len(trajectory),
        'final_answer': final,
        'codex_return_code': result.exit_code,
    }


class OSWorldCodexExecutionProfile:
    options_type = OSWorldCodexOptions

    def __init__(self, options, task_context, user_prompt):
        keys = {'osworld_task_id', 'osworld_domain', 'osworld_client_password', 'trajectory_dir'}
        context = {key: task_context[key] for key in keys if key in task_context}
        self.options = self.options_type.model_validate({**options, **context, 'user_prompt': user_prompt})

    def bridge_options(self, client):
        from nexus.runtime.bridge.adapters.ags_osworld import codex_bridge_options

        return codex_bridge_options(self.options, client)

    async def run(self, client):
        from nexus.runtime.bridge.adapters.ags_osworld import call_native_bridge_runtime_agent

        return await call_native_bridge_runtime_agent(self.execute, **self.bridge_options(client))

    async def execute(self, *, runtime, launch, timeout):
        started = time.monotonic()
        result = await _execute_osworld(runtime=runtime, launch=launch, timeout=timeout, inp=self.options)
        result['execution_time'] = time.monotonic() - started
        return result
