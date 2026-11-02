"""V2 Codex profile: preserve shared execution, add explicit system instructions."""

import json
import time

from pathlib import Path
from typing import Literal, cast

from pydantic import Field

from nexus.core import require_context
from nexus.extensions.agents.codex.json_events import _terminal_error_message, parse_codex_jsonl, parse_codex_stream
from nexus.extensions.agents.codex.native_bridge.adapters.base import execution_profile_registry
from nexus.extensions.agents.codex.native_bridge.adapters.types import CodexInstallSpec, CodexRunRequest
from nexus.extensions.tasks.os_world_v2.adapter.prompts import RESOURCE_POLICY
from nexus.extensions.tasks.os_world_v2.execution.codex_profile import OSWorldCodexExecutionProfile, OSWorldCodexOptions
from nexus.extensions.tasks.os_world_v2.execution.guest_network import guest_proxy_env
from nexus.extensions.tasks.os_world_v2.execution.guide import render_guide
from nexus.extensions.tasks.os_world_v2.outputs import record_agent_output


class V2CodexOptions(OSWorldCodexOptions):
    session_mode: Literal['auto', 'exec', 'app_server'] = 'auto'
    max_turns: int = Field(default=20, ge=1)
    max_user_requests: int = Field(default=100, ge=1)
    hook_input_timeout: float = Field(default=300, gt=0)


@execution_profile_registry.register('osworld_v2')
class OSWorldV2CodexProfile(OSWorldCodexExecutionProfile):
    options_type = V2CodexOptions
    options: V2CodexOptions

    def __init__(self, options, task_context, user_prompt):
        super().__init__(options, task_context, user_prompt)
        self.has_simulator = bool(task_context.get('osworld_user_simulator'))
        self.callback = None
        if self.has_simulator and self.options.session_mode == 'exec':
            raise ValueError('Codex HITL requires auto or app_server session_mode')
        # A root TOML developer_instructions entry enters Codex's system-level
        # instructions, rather than relying on the ignored task system_prompt.
        try:
            import tomllib
        except ModuleNotFoundError:
            import tomli as tomllib

        text = self.options.extra_config_toml
        if 'developer_instructions' in tomllib.loads(text):
            # Preserve caller instructions using the CLI override, which wins over TOML.
            # The shared profile has no extra CLI seam, so reject ambiguous duplicates.
            raise ValueError('V2 Codex reserves developer_instructions; put custom guidance in guide_text')
        self.options.extra_config_toml = 'developer_instructions = ' + json.dumps(RESOURCE_POLICY) + '\n' + text

    def set_user_input(self, callback):
        self.callback = callback

    async def execute(self, *, runtime, launch, timeout):
        if self.options.session_mode != 'app_server' and not self.has_simulator:
            return await super().execute(runtime=runtime, launch=launch, timeout=timeout)
        if self.has_simulator and self.callback is None:
            raise ValueError('V2 Codex user simulator callback is missing')
        from nexus.extensions.tasks.os_world_v2.adapter.codex_hitl import HITLCodexAdapter

        options = self.options
        directory = Path(options.trajectory_dir) if options.trajectory_dir else None
        if directory:
            directory.mkdir(parents=True, exist_ok=True)
        adapter = HITLCodexAdapter(
            runtime,
            diagnostic_dir=directory,
            options=options,
            callback=self.callback,
            has_simulator=self.has_simulator,
        )
        started = time.monotonic()
        try:
            executable = await adapter.ensure_codex(
                CodexInstallSpec(
                    options.version,
                    options.npm_registry,
                    options.binary_url,
                    options.binary_checksum,
                )
            )
            probe = await runtime.run_command(
                command=f'{executable} --version',
                timeout=30,
                extra_env=adapter.execution_env,
            )
            if probe.return_code != 0 or (probe.stdout or '').strip() != f'codex-cli {options.version}':
                raise RuntimeError('Installed Codex does not match pinned version')
            text = options.user_prompt
            if not isinstance(text, str):
                text = '\n'.join(p.get('text', '') for p in text if p.get('type') == 'text')
            config = options.extra_config_toml.replace(
                '[features]',
                '[features]\ndefault_mode_request_user_input = ' + ('true' if self.has_simulator else 'false'),
                1,
            )
            if '[features]' not in config:
                config += (
                    '\n[features]\ndefault_mode_request_user_input = '
                    + ('true' if self.has_simulator else 'false')
                    + '\n'
                )
            extra_env = {
                'DISPLAY': ':0',
                'NO_PROXY': 'localhost,127.0.0.1,::1',
                'no_proxy': 'localhost,127.0.0.1,::1',
            }
            if getattr(runtime, 'osworld_use_proxy', False) is True:
                extra_env.update(guest_proxy_env(extra_env))
            request = CodexRunRequest(
                executable,
                launch,
                render_guide(options.guide_text, options.osworld_client_password).rstrip() + '\n\n' + text,
                options.working_dir,
                timeout,
                options.version,
                options.npm_registry,
                model=options.model,
                model_catalog_json=cast(str, options.model_catalog_json),
                extra_config_toml=config,
                auto_compact_token_limit=options.auto_compact_token_limit,
                extra_env=extra_env,
                strict_cleanup=True,
                record_shell_history=True,
                use_azure_provider=False,
            )
            if directory:
                (directory / 'codex_audit.json').write_text(
                    json.dumps(
                        {
                            'session_mode': 'app_server',
                            'version': options.version,
                            'model': options.model,
                            'max_turns': options.max_turns,
                            'max_user_requests': options.max_user_requests,
                            'timeout': timeout,
                            'user_simulator': self.has_simulator,
                        },
                        indent=2,
                    )
                )
            result = await adapter.run_codex(request)
            report = adapter.session_result or {}
            trajectory, final, terminal = parse_codex_stream(result.stdout)
            if directory:
                (directory / 'codex_stream.jsonl').write_text(result.stdout)
                (directory / 'codex_stderr.txt').write_text(result.stderr)
            success = bool(report.get('success')) and result.exit_code == 0 and result.failure_kind is None
            error = None if success else report.get('error') or result.stderr or 'Missing app-server result'
            # Evidence fields let the task contract classify the failure precisely
            # (timeout, inference/context/output limit, sandbox) instead of unknown.
            timed_out = bool(getattr(result, 'timed_out', False))
            api_error = None
            if not success and result.failure_kind != 'infrastructure' and not timed_out:
                api_error = (
                    _terminal_error_message(parse_codex_jsonl(result.stdout))
                    or report.get('error')
                    or (f'Codex terminal event: {terminal}' if terminal in {'failed', 'error'} else None)
                )
            data = {
                'success': success,
                'exit_status': report.get('exit_status') if success else 'tool_error',
                'exit_code': result.exit_code,
                'codex_return_code': result.exit_code,
                'output': report.get('final_answer') or final or '',
                'stderr': result.stderr,
                'final_answer': report.get('final_answer') or final,
                'error': error,
                'failure_kind': result.failure_kind,
                'failure_stage': result.failure_stage,
                'timed_out': timed_out,
                'budget_stop_confirmed': timed_out and result.failure_stage == 'timeout',
                'terminal_api_error': api_error,
                'main_trajectory': trajectory,
                'json_events_stdout': result.stdout,
                'execution_time': time.monotonic() - started,
                'iterations': len(trajectory),
            }
            stage_output = record_agent_output(data, harness='codex')
            if not success and not (stage_output is not None and stage_output.exit_code == 0):
                raise RuntimeError(data['error'])
            return data
        except Exception as exc:
            if require_context().tool_state.get('osworld_agent_output') is None:
                record_agent_output(
                    {'success': False, 'error': str(exc), 'failure_stage': 'execution'}, harness='codex'
                )
            raise
