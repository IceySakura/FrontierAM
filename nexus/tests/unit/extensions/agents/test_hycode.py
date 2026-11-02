import asyncio
import base64
import contextlib
import inspect
import json
import logging
import os
import pickle
import sys

from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

from nexus.core import require_context
from nexus.core.context import ExecutionContext
from nexus.extensions.agents.hycode import agent as hycode_agent
from nexus.extensions.agents.hycode import generator, installer, runtime_setup, telemetry
from nexus.extensions.agents.hycode.agent import HyCodeInput
from nexus.runtime.client import remote as runtime_client
from nexus.runtime.client.base import RuntimeUnhealthyError
from nexus.runtime.executors import command as command_executor
from nexus.runtime.executors import file as file_executor
from nexus.runtime.executors.models.file import DownloadFileRequest
from nexus.runtime.models import InvokeRequest, InvokeResponse
from nexus.trace import RemoteTraceContext, SpanController, processor_context, span_context
from nexus.trace.processors.base import SpanProcessor


# Marks a key that the fake runtime should omit entirely, as opposed to sending as None.
_ABSENT = object()


class TestHyCodeVersion:
    def test_all_entrypoints_default_to_pinned_release(self):
        assert installer.NPM_PACKAGE == '@tencent/hycode'
        assert installer.DEFAULT_HYCODE_VERSION == '0.1.14'
        assert generator.DEFAULT_HYCODE_VERSION == '0.1.14'
        assert HyCodeInput(user_prompt='test', working_dir='/workspace').version == '0.1.14'
        assert inspect.signature(runtime_setup.setup_and_run_hycode).parameters['version'].default == '0.1.14'

    async def test_installer_uses_experimental_package_and_resolves_its_binary(self, monkeypatch, tmp_path):
        captured: dict[str, object] = {}

        async def fake_run_command(request: command_executor.RunCommandRequest) -> command_executor.RunCommandResponse:
            captured['command'] = request.command
            package_dir = tmp_path / 'node_modules' / '@tencent' / 'hycode'
            package_dir.mkdir(parents=True)
            (package_dir / 'package.json').write_text('{"version": "0.1.14"}')
            (package_dir / installer.BIN_GLIBC).write_text('')
            return command_executor.RunCommandResponse(
                status=command_executor.CommandRunStatus.Finished,
                execution_time=0.1,
                return_code=0,
                stdout='ok',
                stderr='',
            )

        monkeypatch.setattr(command_executor, 'run_command', fake_run_command)
        monkeypatch.setattr(installer, '_is_musl', lambda: False)

        binary = await installer.install_hycode_via_npm(install_dir=str(tmp_path))

        assert '@tencent/hycode@0.1.14' in captured['command']
        assert binary.resolve() == (tmp_path / 'node_modules' / '@tencent' / 'hycode' / 'opencc-bin')
        assert installer.get_installed_version(str(tmp_path)) == '0.1.14'


class TestHyCodeRuntimeConfig:
    @pytest.mark.parametrize(
        ('configured', 'task_default', 'expected'),
        [(False, True, False), (None, None, True), (None, False, False)],
        ids=['explicit-off', 'default-on', 'task-default'],
    )
    async def test_explicit_span_setting_precedes_task_default(
        self, monkeypatch: pytest.MonkeyPatch, configured: bool | None, task_default: bool | None, expected: bool
    ) -> None:
        call = AsyncMock(return_value={'success': True, 'exit_code': 0, 'execution_time': 0.1})
        monkeypatch.setattr(hycode_agent, 'call_runtime_agent', call)
        monkeypatch.setattr(
            hycode_agent.RemoteTraceContext,
            'from_current_span',
            classmethod(lambda cls: RemoteTraceContext(trace_id='trace', parent_span_id='span-config')),
        )
        options = {} if configured is None else {'emit_spans': configured}
        task_config = {} if task_default is None else {'hycode_emit_spans': task_default}
        async with ExecutionContext(task_config=task_config):
            await hycode_agent.run_hycode(HyCodeInput(user_prompt='test', working_dir='/workspace', **options))

        kwargs = call.call_args.kwargs
        assert 'emit_spans' not in kwargs
        assert ('HYCODE_SPANS_FILE' in kwargs['extra_env']) is expected

    @pytest.mark.parametrize(('purpose', 'expected'), [(None, None), ('eval', 'eval'), ('test', 'test')])
    def test_gongfeng_purpose_is_deployment_opt_in(self, monkeypatch, purpose, expected):
        monkeypatch.setenv('GONGFENG_RUNTIME_TOKEN', 'test-token')
        if purpose is None:
            monkeypatch.delenv('GONGFENG_PURPOSE', raising=False)
        else:
            monkeypatch.setenv('GONGFENG_PURPOSE', purpose)

        config = generator._gongfeng_runtime_provider_config()
        assert config.get('purpose') == expected
        if expected is None:
            assert 'purpose' not in config

    @pytest.mark.parametrize('disable_invoke_wait_timeout', [None, False, True])
    async def test_long_invoke_requires_opt_in_and_keeps_http_timeout(self, monkeypatch, disable_invoke_wait_timeout):
        """A CLI run can outlive the invoke budget without relaxing HTTP or process limits."""
        runtime = runtime_client.RemoteRuntime(
            base_url='http://sandbox:8000',
            request_timeout=4000.0,
            auto_connect_ws=False,
            poll_interval=0.0,
        )
        clock = SimpleNamespace(now=0.0)
        clocked_asyncio = SimpleNamespace(**vars(asyncio))
        clocked_asyncio.get_event_loop = lambda: SimpleNamespace(time=lambda: clock.now)
        monkeypatch.setattr(runtime_client, 'asyncio', clocked_asyncio)
        captured = {}
        http_timeouts = []
        polls = 0

        async def fake_connect():
            pass

        async def get_runtime():
            return runtime

        async def fake_post(_url, **kwargs):
            http_timeouts.append(kwargs['timeout'].total)
            captured['wire'] = kwargs['json']
            _, captured['payload'] = InvokeRequest(**kwargs['json']).unpack_payload()
            return SimpleNamespace(status_code=202)

        async def fake_get(_url, **kwargs):
            nonlocal polls
            http_timeouts.append(kwargs['timeout'].total)
            polls += 1
            if polls == 1:
                clock.now = 4001.0
                payload = {'status': 'running'}
            else:
                response = InvokeResponse(
                    request_id=captured['wire']['request_id'],
                    success=True,
                    result=base64.b64encode(
                        pickle.dumps({
                            'success': True,
                            'output': 'completed',
                            'stderr': '',
                            'exit_code': 0,
                            'execution_time': clock.now,
                            'result_count': 1,
                            'result_subtype': 'success',
                            'result_is_error': False,
                        })
                    ).decode(),
                )
                payload = {'status': 'done', 'response': response.model_dump(mode='json')}
            return SimpleNamespace(status_code=200, json=lambda: payload)

        handler = SimpleNamespace(
            request_outcomes={},
            last_request_error_code=None,
            register=lambda: None,
            unregister=lambda: None,
        )
        monkeypatch.setattr(runtime, 'ws_connect', fake_connect)
        monkeypatch.setattr(
            'nexus.core.require_context',
            lambda: SimpleNamespace(
                get_runtime=get_runtime,
                openai_client=object(),
                task_config={},
                runtime_command_timeout_id=None,
            ),
        )
        monkeypatch.setattr(
            hycode_agent.RemoteTraceContext,
            'from_current_span',
            classmethod(lambda cls: RemoteTraceContext(trace_id='test', parent_span_id='hycode-invoke')),
        )
        monkeypatch.setattr('nexus.runtime.server.services.llm_handler.LLMRequestHandler', lambda **_kwargs: handler)
        monkeypatch.setattr(runtime_client.http_client, 'post', fake_post)
        monkeypatch.setattr(runtime_client.http_client, 'get', fake_get)
        monkeypatch.setattr(hycode_agent, 'get_current_span_id', lambda: None)
        options = (
            {} if disable_invoke_wait_timeout is None else {'disable_invoke_wait_timeout': disable_invoke_wait_timeout}
        )
        agent_input = HyCodeInput(
            user_prompt='test', working_dir='/workspace', timeout=86400.0, emit_spans=False, **options
        )

        if disable_invoke_wait_timeout:
            result = await hycode_agent.run_hycode(agent_input)
            assert result.success is True
            assert result.output == 'completed'
            assert polls == 2
        else:
            with pytest.raises(TimeoutError, match='timed out'):
                await hycode_agent.run_hycode(agent_input)
            assert polls == 1
        assert http_timeouts == [4000.0] * (polls + 1)
        assert captured['payload']['timeout'] == 86400.0
        assert 'disable_invoke_wait_timeout' not in captured['payload']
        assert 'disable_invoke_wait_timeout' not in captured['wire']

    async def test_runtime_uses_pause_aware_timeout_controls(self, monkeypatch, tmp_path):
        captured: dict[str, object] = {}
        release_file = tmp_path / 'release'

        async def fake_call_runtime_agent(_runtime_fn, **kwargs):
            captured.update(kwargs)
            captured['context_timeout_id'] = require_context().runtime_command_timeout_id
            return {'success': True, 'exit_code': 0, 'execution_time': 0.1}

        trace_context = RemoteTraceContext(trace_id='trace', parent_span_id='hycode-timeout')
        monkeypatch.setattr(hycode_agent, 'call_runtime_agent', fake_call_runtime_agent)
        monkeypatch.setattr(
            hycode_agent.RemoteTraceContext,
            'from_current_span',
            classmethod(lambda cls: trace_context),
        )

        context = ExecutionContext(task_config={'is_validation': False})
        async with context:
            await hycode_agent.run_hycode(HyCodeInput(user_prompt='test', working_dir='/workspace'))
            assert context.runtime_command_timeout_id is None

        assert captured['runtime_trace_context'] is trace_context
        assert captured['disable_invoke_wait_timeout'] is True
        assert captured['context_timeout_id'] == 'hycode-timeout'

        command_task = asyncio.create_task(
            runtime_setup._run_hycode_streaming(
                [
                    sys.executable,
                    '-c',
                    f'import pathlib, time\nwhile not pathlib.Path({str(release_file)!r}).exists(): time.sleep(0.01)',
                ],
                cwd=str(tmp_path),
                env={},
                timeout=0.3,
                timeout_control_id='hycode-runtime-timeout',
            )
        )
        for _ in range(100):
            if await command_executor.pause_command_timeout('hycode-runtime-timeout'):
                break
            await asyncio.sleep(0.005)
        else:
            raise AssertionError('HyCode subprocess timeout was not registered')

        await asyncio.sleep(0.4)
        assert command_task.done() is False
        assert await command_executor.resume_command_timeout('hycode-runtime-timeout') is True
        release_file.touch()
        exit_code, _stdout, _stderr, timed_out = await asyncio.wait_for(command_task, timeout=1.0)
        assert exit_code == 0
        assert timed_out is False

    @pytest.mark.parametrize('explicit_override', [None, False, True])
    @pytest.mark.parametrize(
        ('task_config', 'disable_invoke_wait_timeout'),
        [
            pytest.param({'is_validation': False}, True, id='train'),
            pytest.param({'is_validation': True}, False, id='validation'),
            pytest.param({}, False, id='unspecified'),
            pytest.param({'is_validation': None}, False, id='unknown'),
            pytest.param(None, False, id='standalone'),
        ],
    )
    async def test_invoke_wait_timeout_is_disabled_only_for_training(
        self, monkeypatch, task_config, disable_invoke_wait_timeout, explicit_override
    ):
        captured: dict[str, object] = {}

        async def fake_call_runtime_agent(_runtime_fn, **kwargs):
            captured.update(kwargs)
            return {'success': True, 'exit_code': 0, 'execution_time': 0.1}

        trace_context = RemoteTraceContext(trace_id='trace', parent_span_id='hycode-timeout')
        monkeypatch.setattr(hycode_agent, 'call_runtime_agent', fake_call_runtime_agent)
        monkeypatch.setattr(
            hycode_agent.RemoteTraceContext,
            'from_current_span',
            classmethod(lambda cls: trace_context),
        )

        agent_input = HyCodeInput(
            user_prompt='test', working_dir='/workspace', disable_invoke_wait_timeout=explicit_override
        )
        if task_config is None:
            await hycode_agent.run_hycode(agent_input)
        else:
            async with ExecutionContext(task_config=task_config):
                await hycode_agent.run_hycode(agent_input)

        expected = disable_invoke_wait_timeout if explicit_override is None else explicit_override
        assert captured['disable_invoke_wait_timeout'] is expected

    async def test_runtime_setup_disables_proxy_deadline(self, monkeypatch, tmp_path):
        captured: dict[str, object] = {}

        class RuntimeControl:
            sio = object()

        class Proxy:
            def __init__(self, **kwargs):
                captured['proxy_timeout'] = kwargs['request_timeout']

            async def start(self):
                pass

            async def stop(self):
                pass

        async def install_hycode_via_npm(**_kwargs):
            return tmp_path / 'opencc-bin'

        async def run_hycode_streaming(**kwargs):
            captured['timeout_control_id'] = kwargs['timeout_control_id']
            return 0, '', '', False

        monkeypatch.setattr('nexus.extensions.agents.hycode.installer.install_hycode_via_npm', install_hycode_via_npm)
        monkeypatch.setattr('nexus.runtime.context.require_runtime_control', lambda: RuntimeControl())
        monkeypatch.setattr('nexus.runtime.server.services.anthropic_proxy.AnthropicProxy', Proxy)
        monkeypatch.setattr(runtime_setup, '_run_hycode_streaming', run_hycode_streaming)
        monkeypatch.setattr(runtime_setup, 'find_free_port', lambda: 12345)
        monkeypatch.setattr('nexus.trace.get_current_span_id', lambda: 'hycode-runtime-timeout')

        await runtime_setup.setup_and_run_hycode(task='test', working_dir=str(tmp_path), timeout=10)

        assert captured == {'proxy_timeout': None, 'timeout_control_id': 'hycode-runtime-timeout'}


class TestHyCodeTermination:
    @pytest.mark.parametrize(
        ('exit_code', 'expected'),
        [
            (hycode_agent.EXIT_CODE_MAX_TURNS, 'MAX_STEPS'),
            (hycode_agent.EXIT_CODE_TOOL_ERROR, 'FORMAT_ERROR'),
            (hycode_agent.EXIT_CODE_CONTEXT_LENGTH, 'TRUNCATION'),
        ],
    )
    async def test_control_exit_without_result_keeps_reason(self, monkeypatch, exit_code, expected):
        async def fake_call_runtime_agent(_runtime_fn, **_kwargs):
            return {
                'success': False,
                'output': '',
                'stderr': '',
                'exit_code': exit_code,
                'execution_time': 0.1,
                'result_count': 0,
            }

        monkeypatch.setattr(hycode_agent, 'call_runtime_agent', fake_call_runtime_agent)
        monkeypatch.setattr(hycode_agent, 'get_current_span_id', lambda: None)

        result = await hycode_agent.run_hycode(HyCodeInput(user_prompt='test', working_dir='/workspace'))

        assert result.exit_status == expected.lower()
        assert result.termination_reason == expected

    @pytest.mark.parametrize(
        ('exit_code', 'exit_name'),
        [(-9, 'SIGKILL'), (1, 'GENERAL_ERROR'), (143, 'SIGTERM')],
    )
    async def test_runtime_exit_without_result_remains_scoreable(
        self,
        monkeypatch,
        exit_code,
        exit_name,
    ):
        async def fake_call_runtime_agent(_runtime_fn, **_kwargs):
            return {
                'success': False,
                'output': '',
                'stderr': '',
                'exit_code': exit_code,
                'execution_time': 0.1,
                'result_count': 0,
            }

        monkeypatch.setattr(hycode_agent, 'call_runtime_agent', fake_call_runtime_agent)
        monkeypatch.setattr(hycode_agent, 'get_current_span_id', lambda: None)

        result = await hycode_agent.run_hycode(HyCodeInput(user_prompt='test', working_dir='/workspace'))

        assert result.success is True
        # exit_status carries the name so harbor_bridge can map it to a distinct reason.
        assert result.exit_status == exit_name.lower()
        assert result.termination_reason == 'ENV_DONE'
        assert result.scoreable_process_exit is True
        assert result.hycode_exit_name == exit_name

    @pytest.mark.parametrize(
        ('exit_code', 'subtype', 'is_error', 'expected'),
        [
            (0, 'success', False, 'ENV_DONE'),
            (64, 'error_max_turns', True, 'MAX_STEPS'),
            (64, 'error_max_turns', None, 'MAX_STEPS'),
            (65, 'error_tool_use', True, 'FORMAT_ERROR'),
            (66, 'error_context_length', True, 'TRUNCATION'),
            # No dedicated exit code -> classify_termination keeps the legacy default. run_hycode
            # then masks exit 1 + is_error=true as ENV_ERROR (test_exit_1_error_result_is_env_error).
            (1, 'error_during_execution', True, 'ENV_DONE'),
            (1, 'error_max_budget_usd', True, 'ENV_DONE'),
            (1, 'error_max_structured_output_retries', True, 'ENV_DONE'),
            # Exit code and subtype must agree; a bare 64 could come from a task script.
            (64, 'unexpected', True, 'ENV_DONE'),
            (1, 'error_max_turns', True, 'ENV_DONE'),
            (66, 'success', True, 'ENV_DONE'),
            # is_error=false is authoritative: the run finished, so no control exit applies.
            (65, 'error_tool_use', False, 'ENV_DONE'),
            (66, 'error_context_length', False, 'ENV_DONE'),
        ],
    )
    def test_only_explicit_control_exits_override_legacy_default(
        self,
        exit_code: int,
        subtype: str,
        is_error: bool | None,
        expected: str,
    ):
        assert hycode_agent.classify_termination(exit_code, subtype, is_error) == expected

    def test_final_stream_result_is_parsed_without_losing_preceding_events(self):
        output = '\n'.join(
            [
                '{"type":"assistant","message":{"content":"working"}}',
                '{"type":"result","subtype":"error_max_turns","is_error":true,"num_turns":4}',
            ]
        )

        assert runtime_setup._parse_last_result(output) == {
            'result_subtype': 'error_max_turns',
            'num_turns': 4,
            'result_is_error': True,
            'result_count': 1,
        }

    async def test_run_passes_typed_controls_and_records_termination(self, monkeypatch):
        captured: dict[str, object] = {}
        span_updates: list[tuple[str, dict[str, object]]] = []

        async def fake_call_runtime_agent(_runtime_fn, **kwargs):
            captured.update(kwargs)
            return {
                'success': False,
                'output': 'result',
                'stderr': '',
                'exit_code': 64,
                'execution_time': 1.5,
                'result_subtype': 'error_max_turns',
                'result_is_error': True,
                'num_turns': 3,
                'result_count': 1,
            }

        monkeypatch.setattr(hycode_agent, 'call_runtime_agent', fake_call_runtime_agent)
        monkeypatch.setattr(hycode_agent, 'get_current_span_id', lambda: 'span-1')
        monkeypatch.setattr(
            hycode_agent,
            'update_span',
            lambda span_id, *, attributes: span_updates.append((span_id, attributes)),
        )

        result = await hycode_agent.run_hycode(
            HyCodeInput(
                user_prompt='test',
                working_dir='/workspace',
                extra_cli_args=['--output-format', 'stream-json'],
                no_thinking_request_args_mode='inherit',
                max_turns=3,
                stop_on_tool_error=True,
            )
        )

        assert captured['extra_cli_args'] == [
            '--output-format',
            'stream-json',
            '--max-turns',
            '3',
            '--stop-on-tool-error',
        ]
        assert captured['no_thinking_request_args_mode'] == 'inherit'
        assert result.exit_status == 'max_steps'
        assert result.termination_reason == 'MAX_STEPS'
        assert result.result_subtype == 'error_max_turns'
        assert result.num_turns == 3
        assert span_updates == [
            (
                'span-1',
                {
                    'hytuner.termination_reason': 'MAX_STEPS',
                    'hycode.termination_source': 'opencc_exit_protocol',
                    'hycode.exit_code': 64,
                    'hycode.result_subtype': 'error_max_turns',
                    'hycode.num_turns': 3,
                },
            )
        ]

    async def test_runtime_setup_failure_without_result_is_unhandled(self, monkeypatch):
        async def fake_call_runtime_agent(_runtime_fn, **_kwargs):
            return {
                'success': False,
                'output': '',
                'stderr': 'Failed to install HyCode',
                'exit_code': -1,
                'execution_time': 0.1,
                'result_count': 0,
            }

        monkeypatch.setattr(hycode_agent, 'call_runtime_agent', fake_call_runtime_agent)
        monkeypatch.setattr(hycode_agent, 'get_current_span_id', lambda: None)

        result = await hycode_agent.run_hycode(HyCodeInput(user_prompt='test', working_dir='/workspace'))

        assert result.success is False
        assert result.exit_status == 'unhandled_error'
        assert result.termination_reason == 'UNHANDLED_ERROR'
        assert result.scoreable_process_exit is False
        assert result.hycode_exit_name is None

    @pytest.mark.parametrize(
        ('exit_code', 'expected'),
        [
            # Codes opencc assigns on purpose survive a missing result event: subtype and is_error
            # both come from that event, so the paired lookup cannot match here by construction.
            (hycode_agent.EXIT_CODE_MAX_TURNS, 'MAX_STEPS'),
            (hycode_agent.EXIT_CODE_TOOL_ERROR, 'FORMAT_ERROR'),
            (hycode_agent.EXIT_CODE_CONTEXT_LENGTH, 'TRUNCATION'),
            # The host never got an exit_code key back, yet the run itself completed -- measured at
            # a median of 144 turns with 35% scoring full reward, so it trains as a clean finish.
            (hycode_agent.EXIT_CODE_MISSING, 'ENV_DONE'),
            (-9, 'ENV_DONE'),
            (143, 'ENV_DONE'),
            # print.ts exits 1 through `is_error ? 1 : 0`, but that path emits a result event and
            # so never lands here; a bare exit 1 means the CLI died before printing one.
            (1, 'ENV_DONE'),
        ],
    )
    async def test_exit_code_is_honoured_when_no_result_event_arrives(self, monkeypatch, exit_code, expected):
        async def fake_call_runtime_agent(_runtime_fn, **_kwargs):
            return {
                'success': False,
                'output': '',
                'stderr': '',
                'exit_code': exit_code,
                'execution_time': 0.1,
                'result_count': 0,
            }

        monkeypatch.setattr(hycode_agent, 'call_runtime_agent', fake_call_runtime_agent)
        monkeypatch.setattr(hycode_agent, 'get_current_span_id', lambda: None)

        result = await hycode_agent.run_hycode(HyCodeInput(user_prompt='test', working_dir='/workspace'))

        assert result.termination_reason == expected

    async def test_missing_exit_code_key_is_not_confused_with_a_sandbox_failure(self, monkeypatch):
        """A result dict with no exit_code key must not be read as the sandbox's own -1.

        Both used to collapse into -1, which made an intact trajectory indistinguishable from an
        infra early-return and dropped 24% of rollouts from training.
        """

        async def fake_call_runtime_agent(_runtime_fn, **_kwargs):
            return {
                'success': False,
                'output': '',
                'stderr': '',
                'execution_time': 0.1,
                'result_count': 0,
            }

        monkeypatch.setattr(hycode_agent, 'call_runtime_agent', fake_call_runtime_agent)
        monkeypatch.setattr(hycode_agent, 'get_current_span_id', lambda: None)

        result = await hycode_agent.run_hycode(HyCodeInput(user_prompt='test', working_dir='/workspace'))

        assert result.exit_code == hycode_agent.EXIT_CODE_MISSING
        assert result.success is True
        assert result.termination_reason == 'ENV_DONE'
        # -5 is now a scoreable exit named EXIT_MISSING, kept distinct from the sandbox's own -1.
        assert result.exit_status == 'exit_missing'
        assert result.scoreable_process_exit is True
        assert result.hycode_exit_name == 'EXIT_MISSING'

    @pytest.mark.parametrize('exit_code', [-9, 143, 1, None])
    async def test_scoreable_exit_without_any_model_call_is_env_error(self, monkeypatch, exit_code):
        """A scoreable exit with no settled model call has no trajectory to train or score."""
        from nexus.extensions.harbor_bridge.agent_result import agent_termination_reason, build_t3_agent_result

        async def fake_call_runtime_agent(_runtime_fn, **_kwargs):
            result = {
                'success': False,
                'output': '',
                'stderr': '',
                'execution_time': 0.1,
                'result_count': 0,
                'llm_request_outcomes': {},
            }
            if exit_code is not None:
                result['exit_code'] = exit_code
            return result

        span_updates: list[tuple[str, dict[str, object]]] = []
        monkeypatch.setattr(hycode_agent, 'call_runtime_agent', fake_call_runtime_agent)
        monkeypatch.setattr(hycode_agent, 'get_current_span_id', lambda: 'span-1')
        monkeypatch.setattr(
            hycode_agent,
            'update_span',
            lambda span_id, *, attributes: span_updates.append((span_id, attributes)),
        )

        result = await hycode_agent.run_hycode(HyCodeInput(user_prompt='test', working_dir='/workspace'))

        assert result.exit_code == (hycode_agent.EXIT_CODE_MISSING if exit_code is None else exit_code)
        assert result.success is False
        assert result.termination_reason == 'ENV_ERROR'
        assert result.exit_status == 'env_error'
        assert result.scoreable_process_exit is False
        assert result.hycode_exit_name is None
        assert result.failure_kind == 'infrastructure'
        assert result.failure_stage == 'no_model_call'
        attributes = span_updates[-1][1]
        assert attributes['hytuner.termination_reason'] == 'ENV_ERROR'
        assert attributes['hycode.termination_source'] == 'no_model_call'
        assert attributes['hycode.exit_code'] == result.exit_code
        assert 'hycode.scoreable_process_exit' not in attributes
        agent_result = build_t3_agent_result(mode='agent', agent_output=result)
        assert agent_termination_reason(agent_result) == 'ENV_ERROR'

    async def test_scoreable_exit_after_a_model_call_stays_scoreable(self, monkeypatch):
        async def fake_call_runtime_agent(_runtime_fn, **_kwargs):
            return {
                'success': False,
                'output': '',
                'stderr': '',
                'exit_code': -9,
                'execution_time': 0.1,
                'result_count': 0,
                'llm_request_outcomes': {'req-1': None},
            }

        monkeypatch.setattr(hycode_agent, 'call_runtime_agent', fake_call_runtime_agent)
        monkeypatch.setattr(hycode_agent, 'get_current_span_id', lambda: None)

        result = await hycode_agent.run_hycode(HyCodeInput(user_prompt='test', working_dir='/workspace'))

        assert result.success is True
        assert result.termination_reason == 'ENV_DONE'
        assert result.exit_status == 'sigkill'
        assert result.scoreable_process_exit is True
        assert result.failure_kind is None

    def test_every_scoreable_exit_name_maps_to_its_own_reason(self):
        """Each exit name must reach the T3 table, or it silently degrades to UNHANDLED_ERROR."""
        from nexus.extensions.agents.hycode.exit_policy import SCOREABLE_EXIT_NAMES
        from nexus.extensions.harbor_bridge.agent_result import _AGENT_EXIT_RULES

        # GENERAL_ERROR (exit 1) is the one exception: harbor_bridge drops it as ENV_ERROR.
        expected_reasons = {'GENERAL_ERROR': 'ENV_ERROR'}
        for code, name in SCOREABLE_EXIT_NAMES.items():
            rule = _AGENT_EXIT_RULES.get(name.lower())
            assert rule is not None, f'{name} ({code}) is missing from _AGENT_EXIT_RULES'
            assert rule.termination_reason == expected_reasons.get(name, name)
            # 'success' would short-circuit in agent_termination_reason before the table is read.
            assert rule.status != 'success', f'{name} would collapse back into ENV_DONE'
        # -1 is the sandbox's infra marker; keeping it out is what -5 exists for.
        assert -1 not in SCOREABLE_EXIT_NAMES

    @pytest.mark.parametrize('stage', ['install', 'proxy', 'pre_start'])
    async def test_sandbox_setup_failure_reaches_initialization_error(self, monkeypatch, stage):
        """The sandbox's own -1 early-returns must classify, not land in UNHANDLED_ERROR.

        Without the failure_kind/failure_stage passthrough these three are indistinguishable
        from a crashed agent, and the task layer scores a sandbox that never ran a binary.
        """
        from nexus.extensions.harbor_bridge.agent_result import (
            _INITIALIZATION_FAILURE_STAGES,
            agent_termination_reason,
            build_t3_agent_result,
        )

        async def fake_call_runtime_agent(_runtime_fn, **_kwargs):
            return {
                'success': False,
                'output': '',
                'stderr': f'{stage} failed',
                'exit_code': -1,
                'execution_time': 0.1,
                'failure_kind': 'infrastructure',
                'failure_stage': stage,
            }

        monkeypatch.setattr(hycode_agent, 'call_runtime_agent', fake_call_runtime_agent)
        monkeypatch.setattr(hycode_agent, 'get_current_span_id', lambda: None)

        result = await hycode_agent.run_hycode(HyCodeInput(user_prompt='test', working_dir='/workspace'))

        assert stage in _INITIALIZATION_FAILURE_STAGES
        assert result.failure_kind == 'infrastructure'
        assert result.failure_stage == stage
        agent_result = build_t3_agent_result(mode='agent', agent_output=result)
        assert agent_termination_reason(agent_result) == 'INITIALIZATION_ERROR'

    async def test_timeout_kill_keeps_task_timeout_without_a_stage(self, monkeypatch):
        """The fourth -1 site is a killed run, not a setup failure, and must stay TASK_TIMEOUT."""
        from nexus.extensions.harbor_bridge.agent_result import agent_termination_reason, build_t3_agent_result

        async def fake_call_runtime_agent(_runtime_fn, **_kwargs):
            return {
                'success': False,
                'output': '',
                'stderr': '[timeout]',
                'exit_code': -1,
                'execution_time': 0.1,
                'timed_out': True,
                'result_count': 0,
            }

        monkeypatch.setattr(hycode_agent, 'call_runtime_agent', fake_call_runtime_agent)
        monkeypatch.setattr(hycode_agent, 'get_current_span_id', lambda: None)

        result = await hycode_agent.run_hycode(HyCodeInput(user_prompt='test', working_dir='/workspace'))

        assert result.termination_reason == 'TASK_TIMEOUT'
        assert result.failure_stage is None
        agent_result = build_t3_agent_result(mode='agent', agent_output=result)
        assert agent_termination_reason(agent_result) == 'TASK_TIMEOUT'

    async def test_context_refusal_without_result_stays_truncation(self, monkeypatch):
        """The guard for a refusal that killed the CLI before it emitted a result event.

        opencc reports a context refusal as exit 1, which the no-result table reads as ENV_DONE,
        so without this ordering a long-context refusal would train as a clean finish again.
        """

        async def fake_call_runtime_agent(_runtime_fn, **_kwargs):
            return {
                'success': False,
                'output': '',
                'stderr': 'LLM request failed',
                'exit_code': 1,
                'execution_time': 0.1,
                'result_count': 0,
                'last_llm_error_code': hycode_agent.CONTEXT_LENGTH_EXCEEDED_CODE,
            }

        monkeypatch.setattr(hycode_agent, 'call_runtime_agent', fake_call_runtime_agent)
        monkeypatch.setattr(hycode_agent, 'get_current_span_id', lambda: None)

        result = await hycode_agent.run_hycode(HyCodeInput(user_prompt='test', working_dir='/workspace'))

        assert result.termination_reason == 'TRUNCATION'

    async def test_any_context_refusal_outcome_stays_truncation(self, monkeypatch):
        async def fake_call_runtime_agent(_runtime_fn, **_kwargs):
            return {
                'success': False,
                'output': '',
                'stderr': 'LLM request failed',
                'exit_code': 1,
                'execution_time': 0.1,
                'result_count': 0,
                'last_llm_error_code': None,
                'llm_request_outcomes': {
                    'main': hycode_agent.CONTEXT_LENGTH_EXCEEDED_CODE,
                    'later_auxiliary': None,
                },
            }

        monkeypatch.setattr(hycode_agent, 'call_runtime_agent', fake_call_runtime_agent)
        monkeypatch.setattr(hycode_agent, 'get_current_span_id', lambda: None)

        result = await hycode_agent.run_hycode(HyCodeInput(user_prompt='test', working_dir='/workspace'))

        assert result.termination_reason == 'TRUNCATION'

    @pytest.mark.parametrize(
        ('exit_code', 'expected'),
        [
            # Codes opencc assigns on purpose survive a missing result event: subtype and is_error
            # both come from that event, so the paired lookup cannot match here by construction.
            (hycode_agent.EXIT_CODE_MAX_TURNS, 'MAX_STEPS'),
            (hycode_agent.EXIT_CODE_TOOL_ERROR, 'FORMAT_ERROR'),
            (hycode_agent.EXIT_CODE_CONTEXT_LENGTH, 'TRUNCATION'),
            # The host never got an exit_code key back, yet the run itself completed -- measured at
            # a median of 144 turns with 35% scoring full reward, so it trains as a clean finish.
            (hycode_agent.EXIT_CODE_MISSING, 'ENV_DONE'),
            (-9, 'ENV_DONE'),
            (143, 'ENV_DONE'),
            # print.ts exits 1 through `is_error ? 1 : 0`, but that path emits a result event and
            # so never lands here; a bare exit 1 means the CLI died before printing one.
            (1, 'ENV_DONE'),
        ],
    )
    async def test_exit_code_is_honoured_when_no_result_event_arrives(self, monkeypatch, exit_code, expected):
        async def fake_call_runtime_agent(_runtime_fn, **_kwargs):
            return {
                'success': False,
                'output': '',
                'stderr': '',
                'exit_code': exit_code,
                'execution_time': 0.1,
                'result_count': 0,
            }

        monkeypatch.setattr(hycode_agent, 'call_runtime_agent', fake_call_runtime_agent)
        monkeypatch.setattr(hycode_agent, 'get_current_span_id', lambda: None)

        result = await hycode_agent.run_hycode(HyCodeInput(user_prompt='test', working_dir='/workspace'))

        assert result.termination_reason == expected

    async def test_missing_exit_code_key_is_not_confused_with_a_sandbox_failure(self, monkeypatch):
        """A result dict with no exit_code key must not be read as the sandbox's own -1.

        Both used to collapse into -1, which made an intact trajectory indistinguishable from an
        infra early-return and dropped 24% of rollouts from training.
        """

        async def fake_call_runtime_agent(_runtime_fn, **_kwargs):
            return {
                'success': False,
                'output': '',
                'stderr': '',
                'execution_time': 0.1,
                'result_count': 0,
            }

        monkeypatch.setattr(hycode_agent, 'call_runtime_agent', fake_call_runtime_agent)
        monkeypatch.setattr(hycode_agent, 'get_current_span_id', lambda: None)

        result = await hycode_agent.run_hycode(HyCodeInput(user_prompt='test', working_dir='/workspace'))

        assert result.exit_code == hycode_agent.EXIT_CODE_MISSING
        assert result.termination_reason == 'ENV_DONE'

    async def test_context_refusal_without_result_stays_truncation(self, monkeypatch):
        """The guard for a refusal that killed the CLI before it emitted a result event.

        opencc reports a context refusal as exit 1, which the no-result table reads as ENV_DONE,
        so without this ordering a long-context refusal would train as a clean finish again.
        """

        async def fake_call_runtime_agent(_runtime_fn, **_kwargs):
            return {
                'success': False,
                'output': '',
                'stderr': 'LLM request failed',
                'exit_code': 1,
                'execution_time': 0.1,
                'result_count': 0,
                'last_llm_error_code': hycode_agent.CONTEXT_LENGTH_EXCEEDED_CODE,
            }

        monkeypatch.setattr(hycode_agent, 'call_runtime_agent', fake_call_runtime_agent)
        monkeypatch.setattr(hycode_agent, 'get_current_span_id', lambda: None)

        result = await hycode_agent.run_hycode(HyCodeInput(user_prompt='test', working_dir='/workspace'))

        assert result.termination_reason == 'TRUNCATION'

    @pytest.mark.parametrize(
        ('result_overrides', 'expected_reason', 'expected_source'),
        [
            # An unpatched opencc cannot name a context refusal -- the error crosses the
            # Socket.IO boundary as an opaque 500 -- so the host's own per-request outcome is
            # the only place the cause survives. Measured on @tencent/hycode@0.1.8.
            (
                {'exit_code': 1, 'result_subtype': 'success', 'result_is_error': True},
                'TRUNCATION',
                'host_llm_error_code',
            ),
            # Same fallback, reached from a subtype that opencc marked as an error but that has
            # no exit code of its own, so it also sits in ENV_DONE.
            (
                {'exit_code': 1, 'result_subtype': 'error_during_execution', 'result_is_error': True},
                'TRUNCATION',
                'host_llm_error_code',
            ),
            # A run that really finished cannot be recast as truncated, even though the host
            # refused some call along the way (an auxiliary completion, say).
            (
                {'exit_code': 0, 'result_subtype': 'success', 'result_is_error': False},
                'ENV_DONE',
                'opencc_exit_protocol',
            ),
            # The CLI named the reason itself; a host guess must not outrank it.
            (
                {'exit_code': 64, 'result_subtype': 'error_max_turns', 'result_is_error': True},
                'MAX_STEPS',
                'opencc_exit_protocol',
            ),
            # A failure with no error code attached (429, auth, a crash) must not be read as a
            # context refusal: only a budget refusal is assigned an error code. It is still an
            # exit 1 with is_error=true, so it is masked rather than trained as a clean finish.
            (
                {
                    'exit_code': 1,
                    'result_subtype': 'success',
                    'result_is_error': True,
                    'last_llm_error_code': None,
                },
                'ENV_ERROR',
                'opencc_exit_protocol',
            ),
            # Old runtime images omit the key entirely rather than sending None.
            (
                {
                    'exit_code': 1,
                    'result_subtype': 'success',
                    'result_is_error': True,
                    'last_llm_error_code': _ABSENT,
                },
                'ENV_ERROR',
                'opencc_exit_protocol',
            ),
            # A different error code is not a context refusal either.
            (
                {
                    'exit_code': 1,
                    'result_subtype': 'success',
                    'result_is_error': True,
                    'last_llm_error_code': 'rate_limit_exceeded',
                },
                'ENV_ERROR',
                'opencc_exit_protocol',
            ),
        ],
    )
    async def test_host_error_code_fills_in_context_refusal(
        self, monkeypatch, result_overrides, expected_reason, expected_source
    ):
        async def fake_call_runtime_agent(_runtime_fn, **_kwargs):
            result = {
                'success': True,
                'output': 'result',
                'stderr': '',
                'execution_time': 1.0,
                'result_count': 1,
                'last_llm_error_code': hycode_agent.CONTEXT_LENGTH_EXCEEDED_CODE,
                **result_overrides,
            }
            # _ABSENT distinguishes "the runtime sent last_llm_error_code: null" from "the
            # runtime never sent the key", which is what an image predating this change does.
            return {k: v for k, v in result.items() if v is not _ABSENT}

        span_updates: list[tuple[str, dict[str, object]]] = []
        monkeypatch.setattr(hycode_agent, 'call_runtime_agent', fake_call_runtime_agent)
        monkeypatch.setattr(hycode_agent, 'get_current_span_id', lambda: 'span-1')
        monkeypatch.setattr(
            hycode_agent,
            'update_span',
            lambda span_id, *, attributes: span_updates.append((span_id, attributes)),
        )

        result = await hycode_agent.run_hycode(HyCodeInput(user_prompt='test', working_dir='/workspace'))

        assert result.termination_reason == expected_reason
        # Length first: run_hycode must write the termination exactly once, and asserting on
        # span_updates[0] alone would silently pass if a second write overwrote it.
        assert len(span_updates) == 1
        assert span_updates[0][0] == 'span-1'
        assert span_updates[0][1]['hycode.termination_source'] == expected_source
        # Any run left in ENV_DONE despite is_error=true is a known gap and must be flagged --
        # this is the only trace of a failure that stays trainable.
        expected_flag = expected_reason == 'ENV_DONE' and result_overrides.get('result_is_error') is True
        assert span_updates[0][1].get('hycode.env_done_despite_error', False) is expected_flag

    async def test_exit_66_reports_truncation_without_the_host_fallback(self, monkeypatch):
        """Once opencc emits exit 66 the fallback must retire on its own.

        TRUNCATION is absent from _TRUNCATION_OVERRIDABLE_REASONS, so the host cannot overwrite
        it and the recorded source flips back to the exit protocol with no code change.
        """

        async def fake_call_runtime_agent(_runtime_fn, **_kwargs):
            return {
                'success': False,
                'output': '',
                'stderr': '',
                'exit_code': 66,
                'execution_time': 1.0,
                'result_subtype': 'error_context_length',
                'result_is_error': True,
                'result_count': 1,
                'last_llm_error_code': hycode_agent.CONTEXT_LENGTH_EXCEEDED_CODE,
            }

        span_updates: list[tuple[str, dict[str, object]]] = []
        monkeypatch.setattr(hycode_agent, 'call_runtime_agent', fake_call_runtime_agent)
        monkeypatch.setattr(hycode_agent, 'get_current_span_id', lambda: 'span-1')
        monkeypatch.setattr(
            hycode_agent,
            'update_span',
            lambda span_id, *, attributes: span_updates.append((span_id, attributes)),
        )

        result = await hycode_agent.run_hycode(HyCodeInput(user_prompt='test', working_dir='/workspace'))

        assert result.termination_reason == 'TRUNCATION'
        assert len(span_updates) == 1
        assert span_updates[0][0] == 'span-1'
        assert span_updates[0][1]['hycode.termination_source'] == 'opencc_exit_protocol'

    @pytest.mark.parametrize(
        ('subtype', 'result_count'),
        [
            # The production failure: the router gave up on a paused cluster, the bridge failed
            # the call, and opencc (no retries) closed with an API-error turn.
            ('success', 1),
            # Runs that emit several result events are judged by the last one.
            ('success', 3),
            # Error subtypes without a dedicated exit code are masked once exit 1 agrees.
            ('error_during_execution', 1),
            ('error_max_budget_usd', 1),
            ('error_max_structured_output_retries', 1),
        ],
    )
    async def test_exit_1_error_result_is_env_error(self, monkeypatch, subtype, result_count):
        """exit 1 + is_error=true means the task stopped unfinished, so it must not train."""
        from nexus.extensions.harbor_bridge.agent_result import agent_termination_reason, build_t3_agent_result

        async def fake_call_runtime_agent(_runtime_fn, **_kwargs):
            return {
                'success': False,
                'output': 'result',
                'stderr': '',
                'exit_code': 1,
                'execution_time': 1.0,
                'result_subtype': subtype,
                'result_is_error': True,
                'num_turns': 7,
                'result_count': result_count,
                'last_llm_error_code': None,
            }

        span_updates: list[tuple[str, dict[str, object]]] = []
        monkeypatch.setattr(hycode_agent, 'call_runtime_agent', fake_call_runtime_agent)
        monkeypatch.setattr(hycode_agent, 'get_current_span_id', lambda: 'span-1')
        monkeypatch.setattr(
            hycode_agent,
            'update_span',
            lambda span_id, *, attributes: span_updates.append((span_id, attributes)),
        )

        result = await hycode_agent.run_hycode(HyCodeInput(user_prompt='test', working_dir='/workspace'))

        assert result.termination_reason == 'ENV_ERROR'
        assert result.exit_status == 'env_error'
        assert result.scoreable_process_exit is False
        assert result.hycode_exit_name is None
        agent_result = build_t3_agent_result(mode='agent', agent_output=result)
        assert agent_result.status == 'error'
        assert agent_termination_reason(agent_result) == 'ENV_ERROR'
        assert len(span_updates) == 1
        assert span_updates[0][0] == 'span-1'
        assert span_updates[0][1]['hytuner.termination_reason'] == 'ENV_ERROR'
        assert span_updates[0][1]['hycode.exit_code'] == 1
        assert span_updates[0][1]['hycode.result_subtype'] == subtype
        assert 'hycode.env_done_despite_error' not in span_updates[0][1]

    async def test_error_result_without_agreeing_exit_code_is_flagged_but_not_reclassified(self, monkeypatch):
        """is_error=true with no exit code to back it stays ENV_DONE, but the span shows it."""

        async def fake_call_runtime_agent(_runtime_fn, **_kwargs):
            return {
                'success': True,
                'output': 'result',
                'stderr': '',
                'execution_time': 1.0,
                'result_subtype': 'error_max_budget_usd',
                'result_is_error': True,
                'result_count': 1,
            }

        span_updates: list[tuple[str, dict[str, object]]] = []
        monkeypatch.setattr(hycode_agent, 'call_runtime_agent', fake_call_runtime_agent)
        monkeypatch.setattr(hycode_agent, 'get_current_span_id', lambda: 'span-1')
        monkeypatch.setattr(
            hycode_agent,
            'update_span',
            lambda span_id, *, attributes: span_updates.append((span_id, attributes)),
        )

        result = await hycode_agent.run_hycode(HyCodeInput(user_prompt='test', working_dir='/workspace'))

        assert result.exit_code == hycode_agent.EXIT_CODE_MISSING
        assert result.termination_reason == 'ENV_DONE'
        assert len(span_updates) == 1
        assert span_updates[0][0] == 'span-1'
        assert span_updates[0][1]['hycode.env_done_despite_error'] is True
        assert span_updates[0][1]['hycode.unclassified_error_subtype'] == 'error_max_budget_usd'

    async def test_exit_1_clean_result_is_not_masked(self, monkeypatch):
        """Only an is_error=true result is masked; exit 1 alone is not evidence of a failed run."""

        async def fake_call_runtime_agent(_runtime_fn, **_kwargs):
            return {
                'success': False,
                'output': 'result',
                'stderr': '',
                'exit_code': 1,
                'execution_time': 1.0,
                'result_subtype': 'success',
                'result_is_error': False,
                'result_count': 1,
            }

        monkeypatch.setattr(hycode_agent, 'call_runtime_agent', fake_call_runtime_agent)
        monkeypatch.setattr(hycode_agent, 'get_current_span_id', lambda: None)

        result = await hycode_agent.run_hycode(HyCodeInput(user_prompt='test', working_dir='/workspace'))

        assert result.termination_reason == 'ENV_DONE'

    async def test_clean_finish_is_not_flagged_as_env_done_despite_error(self, monkeypatch):
        """The flag must mark only the gap, not every successful run."""

        async def fake_call_runtime_agent(_runtime_fn, **_kwargs):
            return {
                'success': True,
                'output': 'result',
                'stderr': '',
                'exit_code': 0,
                'execution_time': 1.0,
                'result_subtype': 'success',
                'result_is_error': False,
                'result_count': 1,
            }

        span_updates: list[tuple[str, dict[str, object]]] = []
        monkeypatch.setattr(hycode_agent, 'call_runtime_agent', fake_call_runtime_agent)
        monkeypatch.setattr(hycode_agent, 'get_current_span_id', lambda: 'span-1')
        monkeypatch.setattr(
            hycode_agent,
            'update_span',
            lambda span_id, *, attributes: span_updates.append((span_id, attributes)),
        )

        result = await hycode_agent.run_hycode(HyCodeInput(user_prompt='test', working_dir='/workspace'))

        assert result.termination_reason == 'ENV_DONE'
        assert len(span_updates) == 1
        assert 'hycode.env_done_despite_error' not in span_updates[0][1]

    async def test_wall_clock_timeout_outranks_a_context_refusal(self, monkeypatch):
        """The timeout cannot be an exit code: the process is killed, so only the host knows."""

        async def fake_call_runtime_agent(_runtime_fn, **_kwargs):
            return {
                'success': False,
                'output': 'partial',
                'stderr': '[timeout]',
                'exit_code': -1,
                'execution_time': 3600.0,
                'result_count': 1,
                'timed_out': True,
                'last_llm_error_code': hycode_agent.CONTEXT_LENGTH_EXCEEDED_CODE,
            }

        monkeypatch.setattr(hycode_agent, 'call_runtime_agent', fake_call_runtime_agent)
        monkeypatch.setattr(hycode_agent, 'get_current_span_id', lambda: None)

        result = await hycode_agent.run_hycode(HyCodeInput(user_prompt='test', working_dir='/workspace'))

        assert result.termination_reason == 'TASK_TIMEOUT'

    async def test_missing_new_keys_preserve_legacy_behaviour(self, monkeypatch):
        """Old runtime images send neither timed_out nor last_llm_error_code."""

        async def fake_call_runtime_agent(_runtime_fn, **_kwargs):
            return {
                'success': True,
                'output': 'result',
                'stderr': '',
                'exit_code': 0,
                'execution_time': 1.0,
                'result_subtype': 'success',
                'result_is_error': False,
                'result_count': 1,
            }

        monkeypatch.setattr(hycode_agent, 'call_runtime_agent', fake_call_runtime_agent)
        monkeypatch.setattr(hycode_agent, 'get_current_span_id', lambda: None)

        result = await hycode_agent.run_hycode(HyCodeInput(user_prompt='test', working_dir='/workspace'))

        assert result.termination_reason == 'ENV_DONE'

    async def test_unhealthy_runtime_is_reported_as_env_error(self, monkeypatch):
        """RuntimeUnhealthyError used to escape run_hycode, losing the sandbox-death attribution."""

        async def fake_call_runtime_agent(_runtime_fn, **_kwargs):
            raise RuntimeUnhealthyError('sandbox OOMKilled')

        span_updates: list[tuple[str, dict[str, object]]] = []
        monkeypatch.setattr(hycode_agent, 'call_runtime_agent', fake_call_runtime_agent)
        monkeypatch.setattr(hycode_agent, 'get_current_span_id', lambda: 'span-1')
        monkeypatch.setattr(
            hycode_agent,
            'update_span',
            lambda span_id, *, attributes: span_updates.append((span_id, attributes)),
        )

        result = await hycode_agent.run_hycode(HyCodeInput(user_prompt='test', working_dir='/workspace'))

        assert result.success is False
        assert result.termination_reason == 'ENV_ERROR'
        assert result.failure_kind == 'infrastructure'
        assert result.failure_stage == 'runtime_unhealthy'
        assert span_updates == [
            (
                'span-1',
                {
                    'hytuner.termination_reason': 'ENV_ERROR',
                    'hycode.termination_source': 'runtime_unhealthy',
                },
            )
        ]


class TestHyCodeSandboxSetup:
    @pytest.mark.parametrize('stage', ['install', 'proxy', 'pre_start', 'timeout'])
    async def test_setup_reports_stage_and_keeps_process_timeout_distinct(self, monkeypatch, tmp_path, stage):
        from nexus.runtime import context as runtime_context
        from nexus.runtime.server.services import anthropic_proxy

        events = []

        async def install(**kwargs):
            events.append('install')
            assert kwargs['version'] == '0.1.11'
            if stage == 'install':
                raise RuntimeError('installation unavailable')
            return tmp_path / 'opencc-bin'

        class Proxy:
            def __init__(self, **_kwargs):
                pass

            async def start(self):
                events.append('proxy_start')
                if stage == 'proxy':
                    raise RuntimeError('proxy unavailable')

            async def stop(self):
                events.append('proxy_stop')

        async def pre_start(*_args, **_kwargs):
            events.append('pre_start')
            if stage == 'pre_start':
                raise RuntimeError('startup script failed')

        async def run_cli(**_kwargs):
            events.append('cli')
            assert stage == 'timeout', 'The CLI must not run after setup fails.'
            return -1, '', '[timeout]', True

        monkeypatch.setattr(runtime_context, 'require_runtime_control', lambda: SimpleNamespace(sio=object()))
        monkeypatch.setattr(installer, 'install_hycode_via_npm', install)
        monkeypatch.setitem(anthropic_proxy.__dict__, 'AnthropicProxy', Proxy)
        monkeypatch.setattr(runtime_setup, 'find_free_port', lambda: 12345)
        monkeypatch.setattr(runtime_setup, 'build_runtime_path', lambda _directory: '/fake/bin')
        monkeypatch.setattr(runtime_setup, '_run_pre_start_script', pre_start)
        monkeypatch.setattr(runtime_setup, '_run_hycode_streaming', run_cli)

        result = await runtime_setup.setup_and_run_hycode(
            task='test', working_dir=str(tmp_path), version='0.1.11', pre_start_scripts=[{}]
        )

        assert result['success'] is False
        assert result['exit_code'] == -1
        if stage == 'timeout':
            assert result['timed_out'] is True
            assert result['failure_kind'] == 'agent'
            assert result['failure_stage'] == 'process'
            assert events == ['install', 'proxy_start', 'pre_start', 'cli', 'proxy_stop']
        else:
            assert result['failure_kind'] == 'infrastructure'
            assert result['failure_stage'] == stage
            assert {
                'install': 'installation unavailable',
                'proxy': 'proxy unavailable',
                'pre_start': 'startup script failed',
            }[stage] in result['stderr']
            assert 'cli' not in events
            assert ('proxy_stop' in events) is (stage == 'pre_start')


class TestHyCodeStreamCapture:
    async def test_preserves_long_lines_and_unterminated_final_result(self):
        long_event = '{"type":"assistant","content":"' + ('x' * 100_000) + '"}'
        final_result = '{"type":"result","subtype":"success","is_error":false,"num_turns":1}'
        reader = asyncio.StreamReader(limit=64 * 1024)
        reader.feed_data(f'{long_event}\n{final_result}'.encode())
        reader.feed_eof()
        captured: list[str] = []

        await runtime_setup._pipe_to_logger(reader, '[hycode] ', logging.INFO, captured)

        assert captured == [long_event, final_result]
        assert runtime_setup._parse_last_result('\n'.join(captured))['result_subtype'] == 'success'


_RECORD_ARGV_AND_STDIN = (
    'import json, pathlib, sys\n'
    'data = sys.stdin.buffer.read()\n'
    'pathlib.Path(sys.argv[1]).write_bytes(data)\n'
    'pathlib.Path(sys.argv[2]).write_text(json.dumps(sys.argv))\n'
    'print(json.dumps({"type": "result", "subtype": "success", "is_error": False, "num_turns": 1}))\n'
)


class TestHyCodePromptViaStdin:
    PROMPT = '请修复 nl2product_worker.py，然后 `pkill -f nl2product_worker`\n\'"$HOME\'\n' + 'x' * 300_000

    async def test_prompt_reaches_child_stdin_byte_for_byte(self, tmp_path):
        stdin_file, argv_file = tmp_path / 'stdin', tmp_path / 'argv'

        exit_code, stdout, _stderr, timed_out = await runtime_setup._run_hycode_streaming(
            [sys.executable, '-c', _RECORD_ARGV_AND_STDIN, str(stdin_file), str(argv_file)],
            cwd=str(tmp_path),
            env={},
            timeout=30,
            stdin_input=self.PROMPT,
        )

        assert (exit_code, timed_out) == (0, False)
        assert stdin_file.read_bytes() == self.PROMPT.encode('utf-8')
        assert 'nl2product_worker' not in argv_file.read_text()
        assert runtime_setup._parse_last_result(stdout)['result_subtype'] == 'success'

    async def test_prompt_bytes_match_command_line_encoding(self, tmp_path):
        stdin_file, argv_file = tmp_path / 'stdin', tmp_path / 'argv'
        prompt = 'raw byte \udcff from surrogateescape'

        await runtime_setup._run_hycode_streaming(
            [sys.executable, '-c', _RECORD_ARGV_AND_STDIN, str(stdin_file), str(argv_file)],
            cwd=str(tmp_path),
            env={},
            timeout=30,
            stdin_input=prompt,
        )

        assert stdin_file.read_bytes() == os.fsencode(prompt)

    async def test_unencodable_prompt_fails_before_spawn(self, monkeypatch, tmp_path):
        async def spawn(*_args, **_kwargs):
            raise AssertionError('HyCode must not start with a prompt it cannot send')

        monkeypatch.setattr(asyncio, 'create_subprocess_exec', spawn)

        with pytest.raises(UnicodeEncodeError):
            await runtime_setup._run_hycode_streaming(
                ['opencc-bin'], cwd=str(tmp_path), env={}, timeout=30, stdin_input='lone \ud800 surrogate'
            )

    async def test_child_exiting_without_reading_reports_its_own_exit_code(self, tmp_path):
        exit_code, _stdout, stderr, timed_out = await runtime_setup._run_hycode_streaming(
            [sys.executable, '-c', 'import sys; sys.stderr.write("bad flag"); sys.exit(2)'],
            cwd=str(tmp_path),
            env={},
            timeout=30,
            stdin_input=self.PROMPT,
        )

        assert (exit_code, timed_out) == (2, False)
        assert 'bad flag' in stderr

    async def test_child_that_never_reads_stdin_is_bounded_by_timeout(self, tmp_path):
        exit_code, _stdout, _stderr, timed_out = await asyncio.wait_for(
            runtime_setup._run_hycode_streaming(
                [sys.executable, '-c', 'import time; time.sleep(60)'],
                cwd=str(tmp_path),
                env={},
                timeout=0.5,
                stdin_input=self.PROMPT,
            ),
            timeout=10,
        )

        assert (exit_code, timed_out) == (-1, True)

    async def test_setup_keeps_prompt_out_of_command_line(self, monkeypatch, tmp_path):
        from nexus.runtime import context as runtime_context
        from nexus.runtime.server.services import anthropic_proxy

        captured: dict[str, object] = {}

        async def install(**_kwargs):
            return tmp_path / 'opencc-bin'

        class Proxy:
            def __init__(self, **_kwargs):
                pass

            async def start(self):
                pass

            async def stop(self):
                pass

        async def run_cli(**kwargs):
            captured.update(kwargs)
            return 0, '', '', False

        monkeypatch.setattr(runtime_context, 'require_runtime_control', lambda: SimpleNamespace(sio=object()))
        monkeypatch.setattr(installer, 'install_hycode_via_npm', install)
        monkeypatch.setitem(anthropic_proxy.__dict__, 'AnthropicProxy', Proxy)
        monkeypatch.setattr(runtime_setup, 'find_free_port', lambda: 12345)
        monkeypatch.setattr(runtime_setup, '_run_hycode_streaming', run_cli)

        await runtime_setup.setup_and_run_hycode(
            task=self.PROMPT,
            working_dir=str(tmp_path),
            extra_cli_args=['--print', '--input-format', 'text', '--disallowedTools', 'WebSearch', 'WebFetch'],
        )

        assert captured['stdin_input'] == self.PROMPT
        assert captured['cmd'] == [
            str(tmp_path / 'opencc-bin'),
            '--dangerously-skip-permissions',
            '--print',
            '--input-format',
            'text',
            '--disallowedTools',
            'WebSearch',
            'WebFetch',
        ]


class TestHyCodeCompose:
    async def test_explicit_mcp_cli_args_are_preserved_without_injected_servers(self):
        input_data = HyCodeInput(user_prompt='test', working_dir='/workspace')
        extra_cli_args = ['--mcp-config', '/workspace/mcp.json']

        await hycode_agent._prepare_strict_mcp_file(input_data, extra_cli_args)

        assert extra_cli_args == ['--mcp-config', '/workspace/mcp.json']

    async def test_explicit_mcp_cli_args_conflict_with_injected_servers(self):
        input_data = HyCodeInput(
            user_prompt='test',
            working_dir='/workspace',
            mcp_config_servers={'injected': {'type': 'http', 'url': 'http://mcp:8000/mcp'}},
        )

        with pytest.raises(ValueError, match='cannot be combined with explicit MCP CLI arguments'):
            await hycode_agent._prepare_strict_mcp_file(
                input_data,
                ['--mcp-config=/workspace/mcp.json'],
            )

    async def test_run_uploads_strict_task_mcp_for_execution_target(self, monkeypatch):
        from types import SimpleNamespace

        from nexus import core as core_module
        from nexus.extensions.agents.installed_common.execution import (
            ComposeExecutionTarget,
            agent_execution_target,
            get_agent_execution_target,
        )

        uploads = []
        commands = []
        captured = {}
        assert hycode_agent.run_hycode.nexus_meta['meta']['supports_execution_target'] is True

        class FakeRuntime:
            async def upload_file(self, files, overwrite=False):
                uploads.append((files, overwrite))
                return SimpleNamespace(
                    results=[SimpleNamespace(path=item.path, success=True, message='') for item in files]
                )

            async def run_command(self, **kwargs):
                commands.append(kwargs)
                return SimpleNamespace(return_code=0, stdout='', stderr='')

        class FakeContext(ExecutionContext):
            async def get_runtime(self):
                return FakeRuntime()

        async def fake_call_runtime_agent(_runtime_fn, **kwargs):
            captured.update(kwargs)
            captured['active_execution_target'] = get_agent_execution_target()
            return {
                'success': True,
                'output': '{"type":"result","subtype":"success","is_error":false,"num_turns":1}',
                'stderr': '',
                'exit_code': 0,
                'execution_time': 1,
                'result_subtype': 'success',
                'result_is_error': False,
                'num_turns': 1,
                'result_count': 1,
                'failure_kind': None,
                'failure_stage': None,
            }

        monkeypatch.setattr(core_module, 'require_context', lambda: FakeContext())
        monkeypatch.setattr(hycode_agent, 'call_runtime_agent', fake_call_runtime_agent)
        monkeypatch.setattr(hycode_agent, 'get_current_span_id', lambda: None)

        execution_target = ComposeExecutionTarget(
            project_name='test-project',
            project_directory='/nexus/compose/task',
            compose_files=['/nexus/compose/task/docker-compose.yaml'],
            service='agent',
            transport_host_dir='/tmp/nexus-agent/abc',
            transport_container_dir='/opt/nexus-agent',
            socket_bridge_host_path='/tmp/nexus-agent/abc/bin/nexus-socket-bridge',
            socket_bridge_container_path='/opt/nexus-agent/bin/nexus-socket-bridge',
        )
        with agent_execution_target(execution_target):
            result = await hycode_agent.run_hycode(
                HyCodeInput(
                    user_prompt='test',
                    working_dir='/workspace/task/agent_workspace',
                    max_turns=2,
                    mcp_servers=[
                        {
                            'name': 'task-tools',
                            'client': {
                                'transport': 'streamable_http',
                                'url': 'http://backend:18182/mcp/00-tools',
                            },
                        }
                    ],
                )
            )

        assert result.success is True
        assert len(uploads) == 1
        ([uploaded], overwrite) = uploads[0]
        assert overwrite is True
        assert uploaded.path == '/tmp/nexus-agent/abc/hycode-mcp.json'
        assert json.loads(uploaded.content) == {
            'mcpServers': {'task-tools': {'type': 'http', 'url': 'http://backend:18182/mcp/00-tools'}}
        }
        assert commands == [{'command': 'chmod 0644 -- /tmp/nexus-agent/abc/hycode-mcp.json', 'timeout': 30}]
        assert captured['active_execution_target'] == execution_target
        assert 'compose_target' not in captured
        assert captured['extra_cli_args'][-5:] == [
            '--mcp-config',
            '/tmp/nexus-agent/abc/hycode-mcp.json',
            '--strict-mcp-config',
            '--max-turns',
            '2',
        ]
        assert captured['pre_start_scripts'] == []


def _hycode_tool_span(start: int, end: int, *, span_type: str = 'tool', success: bool = True) -> dict:
    return {
        'name': 'claude_code.tool',
        'spanId': '1111111111111111',
        'traceId': '22222222222222222222222222222222',
        'parentSpanId': '3333333333333333',
        'startTimeUnixNano': str(start * 10**9),
        'endTimeUnixNano': str(end * 10**9),
        'attributes': [
            {'key': 'span.type', 'value': {'stringValue': span_type}},
            {'key': 'tool.name', 'value': {'stringValue': 'Bash'}},
            {'key': 'input.value', 'value': {'stringValue': '{"command":"pwd"}'}},
            {'key': 'execution.success', 'value': {'boolValue': success}},
            *([] if success else [{'key': 'execution.error', 'value': {'stringValue': 'command failed'}}]),
        ],
    }


class _LocalRuntime:
    async def download_file(self, files):
        return await file_executor.download_file(DownloadFileRequest(files=files))


class TestHyCodeTelemetry:
    @pytest.fixture
    def captured_spans(self, monkeypatch):
        monkeypatch.setattr(SpanController, '_instance', None)
        controller = SpanController.get_instance_sync()
        captured = []

        class Collector(SpanProcessor):
            def on_end(self, span, event):
                captured.append(span)

        try:
            with processor_context([Collector()]):
                yield captured
        finally:
            controller.shutdown()

    async def test_fetch_keeps_tool_spans_and_handles_bad_or_missing_files(self, tmp_path):
        tool = _hycode_tool_span(10, 80)
        kept = {
            'startTimeUnixNano': tool['startTimeUnixNano'],
            'endTimeUnixNano': tool['endTimeUnixNano'],
            'attributes': [item for item in tool['attributes'] if item['key'] != 'span.type'],
        }
        tool['attributes'] = [*tool['attributes'], {'key': 'output.value', 'value': {'stringValue': 'unused'}}]
        batch = {
            'resourceSpans': [{'scopeSpans': [{'spans': [_hycode_tool_span(1, 100, span_type='llm_request'), tool]}]}]
        }
        path = tmp_path / 'spans.jsonl'
        path.write_text(json.dumps(batch) + '\nnot-json\n{}\n{"resourceSpans":')
        size = path.stat().st_size

        runtime = _LocalRuntime()
        assert await telemetry.fetch_hycode_spans(runtime, str(path)) == {
            'spans': [kept],
            'size_bytes': size,
            'invalid_batches': 3,
        }
        assert await telemetry.fetch_hycode_spans(runtime, str(tmp_path / 'missing.jsonl')) == {
            'spans': [],
            'size_bytes': 0,
            'invalid_batches': 0,
            'missing': True,
        }

    async def test_fetch_keeps_unicode_line_separators_inside_json(self, tmp_path):
        tool_input = 'a\u2028b\u2029c\x85d'
        tool = _hycode_tool_span(10, 80)
        tool['attributes'] = [
            {'key': 'input.value', 'value': {'stringValue': tool_input}} if item['key'] == 'input.value' else item
            for item in tool['attributes']
        ]
        path = tmp_path / 'spans.jsonl'
        batch = {'resourceSpans': [{'scopeSpans': [{'spans': [tool]}]}]}
        path.write_text(json.dumps(batch, ensure_ascii=False) + '\n', encoding='utf-8')

        result = await telemetry.fetch_hycode_spans(_LocalRuntime(), str(path))

        assert result['invalid_batches'] == 0
        (span,) = result['spans']
        assert telemetry._decode_attributes(span['attributes'])['input.value'] == tool_input

    def test_replay_creates_observations_and_merges_tool_time(self, captured_spans):
        attribution = {'_run_id': 'run-1', '_run_span_id': 'run-span-1', '_run_kind': 'live'}
        unknown = _hycode_tool_span(20, 85)
        unknown['attributes'] = [item for item in unknown['attributes'] if item['key'] != 'execution.success']
        with span_context('agent.hycode', attributes={'type': 'agent', **attribution}) as parent:
            telemetry.replay_hycode_spans(
                [
                    _hycode_tool_span(10, 70),
                    _hycode_tool_span(20, 90, success=False),
                    unknown,
                    _hycode_tool_span(10, 1),
                ],
                collect_tool_args=True,
                tool_args_min_duration=60,
            )
            assert parent.span.attributes['hytuner.observation_time'] == 80.0
        with span_context('agent.hycode') as empty:
            telemetry.replay_hycode_spans([], collect_tool_args=True, tool_args_min_duration=60)
            assert 'hytuner.observation_time' not in empty.span.attributes
        ok, failed, unknown = [span for span in captured_spans if span.name == 'agent.observation']
        common = {
            **attribution,
            'type': 'agent_observation',
            'tool_name': 'Bash',
            'hytuner.tool_args': '{"command":"pwd"}',
        }
        assert ok.attributes == {**common, 'hytuner.tool_success': True}
        assert failed.attributes == {**common, 'hytuner.tool_success': False, 'hytuner.tool_error': 'command failed'}
        assert ok.parent_span_id == failed.parent_span_id == parent.span_id
        assert (ok.start_time_unix_nano, ok.end_time_unix_nano) == (10 * 10**9, 70 * 10**9)
        assert ok.status.code == 'OK'
        assert (failed.status.code, failed.status.message) == ('ERROR', 'command failed')
        assert unknown.attributes == common
        assert unknown.status.code == 'UNSET'

        with span_context('agent.hycode') as disabled:
            telemetry.replay_hycode_spans(
                [_hycode_tool_span(10, 70)], collect_tool_args=False, tool_args_min_duration=60
            )
        (observation,) = [
            span
            for span in captured_spans
            if span.name == 'agent.observation' and span.parent_span_id == disabled.span_id
        ]
        assert 'hytuner.tool_args' not in observation.attributes

    @pytest.mark.parametrize('outcome', ['ok', 'error', 'cancelled'])
    async def test_import_is_best_effort_except_cancellation(self, monkeypatch, captured_spans, outcome):
        result = {'spans': [_hycode_tool_span(10, 80)], 'size_bytes': 1234, 'invalid_batches': 0}
        side_effects = {'error': RuntimeError('transport failed'), 'cancelled': asyncio.CancelledError}
        monkeypatch.setattr(
            telemetry, 'fetch_hycode_spans', AsyncMock(return_value=result, side_effect=side_effects.get(outcome))
        )
        monkeypatch.setattr(
            'nexus.core.require_context', lambda: SimpleNamespace(get_runtime=AsyncMock(return_value=SimpleNamespace()))
        )
        expectation = pytest.raises(asyncio.CancelledError) if outcome == 'cancelled' else contextlib.nullcontext()
        with span_context('agent.hycode'), expectation:
            await telemetry.import_hycode_spans(
                '/tmp/test-spans.jsonl', collect_tool_args=True, tool_args_min_duration=60
            )

        statuses = [
            span.attributes['hycode.span_transfer']['status']
            for span in captured_spans
            if 'hycode.span_transfer' in span.attributes
        ]
        assert statuses == ([] if outcome == 'cancelled' else [outcome])
        assert sum(span.name == 'agent.observation' for span in captured_spans) == (outcome == 'ok')
