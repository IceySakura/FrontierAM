from __future__ import annotations

import argparse
import asyncio
import json
import stat

from pathlib import Path
from typing import Any

import pytest

from nexus.core.openai_client import OpenAICompatibleClient, OpenAIConfig
from nexus.core.types import ExtendedChatCompletion
from nexus.extensions.projects.hy_live.example import cli, harness
from nexus.extensions.projects.hy_live.example.cases import CASES, get_case
from nexus.extensions.projects.hy_live.example.harness import build_run_input, load_case_data
from nexus.extensions.projects.hy_live.example.prepare_scripted_user_data import build_scripted_user_row
from nexus.extensions.projects.hy_live.task import LiveAgentTaskInput
from nexus.extensions.projects.hy_live.user_simulator import (
    LLMUserSimulatorConfig,
    parse_scripted_turn,
)
from nexus.runner.single import ErrorInfo, RunHooks, RunInput, RunOutput
from nexus.trace import Status, processor_context, span_context


SHANGHAI = {'country': '中国', 'province': '上海', 'city': '上海'}
CODEBUDDY_DEFER = {
    'working_dir': '/workspace',
    'skip_permissions': False,
    'permission_policy': 'defer',
    'npm_registry': 'http://mirrors.tencent.com/npm/',
}
WORKBUDDY_ASK_EACH_TOOL = {'permission_mode': 'default'}
WORKBUDDY_DEFER = {
    'working_dir': '/workspace',
    'skip_permissions': False,
    'permission_policy': 'defer',
    'workbuddy': WORKBUDDY_ASK_EACH_TOOL,
}
SETTING_ENV = (
    'LIVE_AGENT_POLICY_BASE_URL',
    'LIVE_AGENT_POLICY_MODEL',
    'GONGFENG_API_BASE_URL',
    'NEXUS_RUNTIME_IMAGE',
    'LIVE_AGENT_RUNTIME_IMAGE',
)
CODEBUDDY_CASES = [name for name, case in CASES.items() if 'codebuddy' in case.backends]
WORKBUDDY_ONLY_CASES = [name for name, case in CASES.items() if case.backends == frozenset({'workbuddy'})]


async def _never_run_agent(**kwargs: Any) -> Any:
    raise AssertionError(f'unexpected run_agent call: {kwargs}')


# ── Frozen case data ───────────────────────────────────────────────────────────


@pytest.mark.parametrize('name', list(CASES))
def test_cases_share_place_time_backend_and_timeouts(name: str) -> None:
    case = get_case(name)
    assert case.spec_path.is_file(), case.spec_relpath
    task_config = load_case_data(case)['task_config']
    assert task_config['user_place'] == SHANGHAI
    # Every case reviews permission notices, so WorkBuddy asks for each tool instead of its bypass default.
    workbuddy_overlay = {'workbuddy': {'workbuddy': WORKBUDDY_ASK_EACH_TOOL}}
    expected = (
        {'workbuddy': WORKBUDDY_DEFER}
        if name in WORKBUDDY_ONLY_CASES
        else {'codebuddy': CODEBUDDY_DEFER, **workbuddy_overlay}
    )
    if name in {'backend_web_research', 'backend_web_fetch'}:
        expected = {
            'codebuddy': {
                **CODEBUDDY_DEFER,
                'extra_env': {
                    'CODEBUDDY_API_KEY__env': 'CODEBUDDY_API_KEY',
                    'CODEBUDDY_INTERNET_ENVIRONMENT': 'internal',
                },
            },
            **workbuddy_overlay,
        }
    assert task_config['work_agent_configs'] == expected, (
        'A case writes the config of the backend it is written for, plus the WorkBuddy permission overlay'
    )
    assert task_config['task_delivery_timeout'] == 1800, 'Every trajectory needs a timeout'
    assert task_config['default_budget']['timeout_s'] > 0
    validated = LiveAgentTaskInput.model_validate({**task_config, 'run_agent': _never_run_agent})
    assert validated.user_now is not None, 'A fixed clock keeps the system prompt reproducible'
    assert validated.user_now.isoformat() == task_config['user_now']


@pytest.mark.parametrize('name', list(CASES))
def test_case_user_simulator_validates_as_the_task_reads_it(name: str) -> None:
    data = load_case_data(get_case(name))

    validated = LiveAgentTaskInput.model_validate({
        **data['task_config'],
        'user_simulator': data['user_simulator'],
        'run_agent': _never_run_agent,
    })

    if isinstance(validated.user_simulator, LLMUserSimulatorConfig):
        assert validated.user_simulator.initial_prompt == data['prompt'], 'The LLM user opens with the case prompt'
        assert validated.user_simulator.resolved_system_prompt, 'The scenario card renders a user system prompt'


@pytest.mark.parametrize('name', list(CASES))
def test_examples_carry_only_task_config_keys_the_task_reads(name: str) -> None:
    unknown = set(load_case_data(get_case(name))['task_config']) - set(LiveAgentTaskInput.model_fields)
    assert not unknown, f'{name} carries task_config keys the task ignores'


def test_unknown_case_raises() -> None:
    with pytest.raises(KeyError, match='Unknown example case'):
        get_case('not_a_case')


def _scripted_texts(case_name: str) -> list[str]:
    turns = load_case_data(get_case(case_name))['user_simulator']['turns']
    return [str(turn.content) for turn in map(parse_scripted_turn, turns)]


# ── Harness and CLI wiring ─────────────────────────────────────────────────────


def _run_input(case: str = 'single_snake', **overrides: Any) -> RunInput:
    return build_run_input(
        get_case(case),
        run_id='test-run',
        api_base_url='http://example.invalid/v1',
        model='dummy',
        reasoning_effort='high',
        temperature=0.0,
        sandbox_api_url='http://example.invalid',
        runtime_image='runtime',
        workspace_image='workspace',
        timeout=60.0,
        **overrides,
    )


@pytest.mark.parametrize(
    ('case', 'override', 'expected'),
    [
        ('mixed_task_and_cancel', None, 'yuanbao'),
        ('mixed_task_and_cancel', 'none', 'none'),
        ('ask_permission_reject', None, None),
    ],
)
def test_build_run_input_lets_the_cli_override_the_case_search(
    case: str, override: str | None, expected: str | None
) -> None:
    run_input = _run_input(case, web_search_provider=override)
    assert run_input.task_config.get('web_search_provider') == expected


def test_build_run_input_wires_clients_runtime_and_sandbox() -> None:
    run_input = _run_input('multi_task_query')
    assert set(run_input.openai_clients) == {'main', 'midbrain', 'external', 'user_simulator'}
    assert run_input.task_config['midbrain_client_name'] == 'midbrain'
    assert run_input.task_config['user_simulator']['client_name'] == 'user_simulator'
    assert run_input.runtime_provider_config['api_base_url'] == 'http://example.invalid'
    assert run_input.runtime_provider_config['runtime_image'] == 'runtime'
    assert run_input.runtime_provider_config['python_version'] == '3.12'
    main = run_input.runtime_configs['main']
    assert (main['image'], main['worker_venv']) == ('workspace', False)
    proxy_keys = {'HTTP_PROXY', 'HTTPS_PROXY', 'http_proxy', 'https_proxy', 'NO_PROXY', 'no_proxy'}
    assert not proxy_keys & set(main['environment']), 'CodeBuddy web tools reach the internet via the platform proxy'


def test_backend_web_research_delegates_source_reading_and_keeps_search_credentials_as_environment_refs() -> None:
    data = load_case_data(get_case('backend_web_research'))
    assert 'research.md' in data['prompt']
    assert '至少两篇来源页面' in data['prompt']
    assert data['user_simulator']['type'] == 'llm'
    run_input = _run_input('backend_web_research')
    assert run_input.task_config['web_search_provider'] == 'none'
    config = run_input.task_config['work_agent_configs']['workbuddy']
    assert config['extra_env']['CODEBUDDY_API_KEY__env'] == 'CODEBUDDY_API_KEY'
    assert config['extra_env']['CODEBUDDY_INTERNET_ENVIRONMENT'] == 'internal'


@pytest.mark.parametrize(('case', 'client_questions'), [('single_snake', False), ('ask_user_question', True)])
def test_build_run_input_always_writes_client_questions(case: str, client_questions: bool) -> None:
    run_input = _run_input(case, client_questions=client_questions)
    assert run_input.task_config['work_agent_configs']['workbuddy']['client_questions'] is client_questions, (
        'Leaving it out would let the hy-live default reach an image that rejects the key'
    )


def test_build_run_input_refuses_a_question_case_without_client_questions() -> None:
    with pytest.raises(ValueError, match='client_questions'):
        _run_input('ask_user_question', client_questions=False)


def _package(tmp_path: Path) -> Path:
    package = tmp_path / 'wb-5.11.0-linux-x64.tgz'
    package.write_bytes(b'not a real package')
    return package


@pytest.mark.parametrize('name', CODEBUDDY_CASES)
def test_build_run_input_on_codebuddy_is_unchanged_by_the_workbuddy_options(name: str, tmp_path: Path) -> None:
    default = _run_input(name, backend='codebuddy')
    explicit = _run_input(
        name, backend='codebuddy', workbuddy_package=str(_package(tmp_path)), workbuddy_package_sha256='0' * 64
    )
    assert explicit.model_dump(mode='json') == default.model_dump(mode='json'), (
        'WorkBuddy options must not leak into a CodeBuddy run'
    )
    task_config = default.task_config
    assert task_config['backend_adapter'] == 'codebuddy', 'An explicit CodeBuddy run overrides the WorkBuddy default'
    base = json.loads(build_scripted_user_row(backend='codebuddy')['nexus'])['task_config']['work_agent_configs'][
        'codebuddy'
    ]
    case_config = load_case_data(get_case(name))['task_config']['work_agent_configs']['codebuddy']
    assert task_config['work_agent_configs'] == {
        'codebuddy': {**base, **case_config, 'relay_updates': True, 'client_questions': True}
    }
    assert 'workbuddy' not in json.dumps(default.model_dump(mode='json'), ensure_ascii=False)


def test_build_run_input_on_workbuddy_moves_the_case_config_and_adds_the_package(tmp_path: Path) -> None:
    package = _package(tmp_path)
    run_input = _run_input('single_snake', backend='workbuddy', workbuddy_package=str(package))
    task_config = run_input.task_config
    assert task_config['backend_adapter'] == 'workbuddy'
    assert task_config['work_agent_configs'] == {
        'workbuddy': {
            'working_dir': '/workspace',
            'llm_client_name': 'external',
            'skip_permissions': False,
            'permission_policy': 'defer',
            'relay_updates': True,
            'client_questions': True,
            'workbuddy': {'fork_mode': 'quick', **WORKBUDDY_ASK_EACH_TOOL, 'package_path': str(package.resolve())},
        }
    }, 'The CodeBuddy npm version and registry do not configure wb, so they are dropped'
    LiveAgentTaskInput.model_validate({**task_config, 'run_agent': _never_run_agent})


@pytest.mark.parametrize('name', list(CASES))
def test_build_run_input_defaults_to_preinstalled_workbuddy_for_every_case(name: str) -> None:
    default = _run_input(name)
    explicit = _run_input(name, backend='workbuddy')
    task_config = default.task_config

    assert default.model_dump(mode='json') == explicit.model_dump(mode='json')
    assert task_config['backend_adapter'] == 'workbuddy'
    assert set(task_config['work_agent_configs']) == {'workbuddy'}
    config = task_config['work_agent_configs']['workbuddy']
    assert config['workbuddy']['package_path'] is None
    assert not {'version', 'npm_registry'} & set(config)
    assert config['llm_client_name'] == 'external'


def test_example_yaml_selects_workbuddy_and_current_worker_code(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv('NEXUS_RUNTIME_IMAGE', raising=False)
    monkeypatch.setenv('LIVE_AGENT_EXTERNAL_API_KEY', 'test-key')
    config_path = Path(__file__).resolve().parents[5] / 'docs/examples/configs/hy_live.yaml'
    run_input = RunInput.from_file(config_path)

    assert run_input.task_config['backend_adapter'] == 'workbuddy'
    assert set(run_input.task_config['work_agent_configs']) == {'workbuddy'}
    assert run_input.runtime_provider_config['runtime_image'] == harness.DEFAULT_RUNTIME_IMAGE
    assert run_input.runtime_provider_config['python_version'] == '3.12'
    assert run_input.runtime_configs['main']['worker_venv'] is False


def test_backend_web_fetch_case_requires_native_fetch_without_cerebellum_search() -> None:
    case = get_case('backend_web_fetch')
    data = load_case_data(case)
    run_input = _run_input(case.name)

    assert 'https://docs.typesafe.ai/introduction' in data['prompt']
    assert 'WebFetch' in data['prompt']
    assert run_input.task_config['web_search_provider'] == 'none'
    assert case.case_checks == ('native_web_fetch_succeeded', 'tasks_done')


def test_build_run_input_runs_a_workbuddy_case_with_its_own_config(tmp_path: Path) -> None:
    package = _package(tmp_path)
    run_input = _run_input(
        'workbuddy_followup_and_steer',
        backend='workbuddy',
        workbuddy_package=str(package),
        workbuddy_package_sha256='ab' * 32,
        relay_updates=False,
    )
    config = run_input.task_config['work_agent_configs']['workbuddy']
    assert config['workbuddy'] == {
        'fork_mode': 'quick',
        **WORKBUDDY_ASK_EACH_TOOL,
        'package_path': str(package.resolve()),
        'package_sha256': 'ab' * 32,
    }
    assert 'relay_updates' not in config
    assert set(run_input.task_config['work_agent_configs']) == {'workbuddy'}


@pytest.mark.parametrize('package', [None, ''])
def test_build_run_input_uses_preinstalled_workbuddy_without_a_host_package(package: str | None) -> None:
    run_input = _run_input('single_snake', backend='workbuddy', workbuddy_package=package)
    assert run_input.task_config['backend_adapter'] == 'workbuddy'
    configs = run_input.task_config['work_agent_configs']
    assert set(configs) == {'workbuddy'}
    assert configs['workbuddy']['workbuddy']['package_path'] is None


def test_build_run_input_refuses_a_missing_workbuddy_package_file(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match='WorkBuddy package'):
        _run_input('single_snake', backend='workbuddy', workbuddy_package=str(tmp_path / 'missing.tgz'))


@pytest.mark.parametrize('name', WORKBUDDY_ONLY_CASES)
def test_build_run_input_refuses_a_workbuddy_case_on_codebuddy(name: str) -> None:
    with pytest.raises(ValueError, match='--backend workbuddy'):
        _run_input(name, backend='codebuddy')


def test_cli_refuses_a_workbuddy_case_on_codebuddy_before_writing_anything(tmp_path: Path) -> None:
    args = _cli_args(tmp_path, case='workbuddy_followup_and_steer', backend='codebuddy')
    with pytest.raises(SystemExit, match='--backend workbuddy'):
        asyncio.run(cli._run_case(args))
    assert not args.output.parent.exists(), 'Refuse before opening any artifact'


def test_cli_backend_defaults_to_workbuddy_and_reads_the_package_from_the_environment(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv('HY_LIVE_WORKBUDDY_PACKAGE', raising=False)
    monkeypatch.delenv('HY_LIVE_WORKBUDDY_PACKAGE_SHA256', raising=False)
    args = cli.parse_args(['--case', 'single_snake', '--output', 'result.json'])
    assert (args.backend, args.workbuddy_package, args.workbuddy_package_sha256) == ('workbuddy', None, None)
    codebuddy_args = cli.parse_args(['--case', 'single_snake', '--output', 'result.json', '--backend', 'codebuddy'])
    assert codebuddy_args.backend == 'codebuddy'
    monkeypatch.setenv('HY_LIVE_WORKBUDDY_PACKAGE', '/packages/wb.tgz')
    monkeypatch.setenv('HY_LIVE_WORKBUDDY_PACKAGE_SHA256', 'cd' * 32)
    args = cli.parse_args(['--case', 'single_snake', '--output', 'result.json', '--backend', 'workbuddy'])
    assert (args.backend, args.workbuddy_package, args.workbuddy_package_sha256) == (
        'workbuddy',
        '/packages/wb.tgz',
        'cd' * 32,
    )


@pytest.mark.parametrize('backend', ['workbuddy', 'codebuddy'])
def test_build_run_input_passes_host_environment_refs_to_the_work_agent(
    backend: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv('TMAP_WEBSERVICE_KEY', 'tmap-secret')
    run_input = _run_input('backend_web_research', backend=backend, pass_env=('TMAP_WEBSERVICE_KEY',))
    extra_env = run_input.task_config['work_agent_configs'][backend]['extra_env']
    assert extra_env['TMAP_WEBSERVICE_KEY__env'] == 'TMAP_WEBSERVICE_KEY'
    assert extra_env['CODEBUDDY_API_KEY__env'] == 'CODEBUDDY_API_KEY', 'The case environment stays'
    assert 'tmap-secret' not in json.dumps(run_input.model_dump(mode='json')), 'Only the variable name enters the data'


def test_cli_pass_env_is_repeatable_and_defaults_to_none() -> None:
    base = ['--case', 'single_snake', '--output', 'result.json']
    assert cli.parse_args(base).pass_env == []
    args = cli.parse_args([*base, '--pass-env', 'TMAP_WEBSERVICE_KEY', '--pass-env', 'OTHER_KEY'])
    assert args.pass_env == ['TMAP_WEBSERVICE_KEY', 'OTHER_KEY']


@pytest.mark.parametrize(('sandbox_purpose', 'expected'), [(None, None), ('', None), ('prod', 'prod')])
def test_build_run_input_sends_sandbox_purpose_only_when_set(sandbox_purpose: str | None, expected: str | None) -> None:
    run_input = _run_input(sandbox_purpose=sandbox_purpose)
    assert run_input.runtime_provider_config.get('purpose') == expected


def _cli_args(tmp_path: Path, **overrides: Any) -> argparse.Namespace:
    values: dict[str, Any] = {
        'case': 'single_snake',
        'output': tmp_path / 'nested' / 'result.json',
        'run_id': 'test-cli',
        'api_base_url': 'http://example.invalid/v1',
        'model': 'dummy',
        'reasoning_effort': 'high',
        'temperature': 0.0,
        'sandbox_api_url': 'http://example.invalid',
        'runtime_image': 'runtime',
        'workspace_image': 'workspace',
        'timeout': 60.0,
        'worker_venv': False,
        'sandbox_purpose': '',
        'relay_updates': False,
    }
    values.update(overrides)
    return argparse.Namespace(**values)


def test_cli_refuses_a_question_case_without_client_questions(tmp_path: Path) -> None:
    args = _cli_args(tmp_path, case='ask_user_question', client_questions=False)
    with pytest.raises(SystemExit, match='client_questions'):
        asyncio.run(cli._run_case(args))
    assert not args.output.parent.exists(), 'Refuse before opening any artifact'


def test_cli_settings_default_to_the_harness_and_yield_to_the_environment(monkeypatch: pytest.MonkeyPatch) -> None:
    for name in SETTING_ENV:
        monkeypatch.delenv(name, raising=False)
    args = cli.parse_args(['--case', 'single_snake', '--output', 'result.json'])
    assert (args.model, args.runtime_image) == (harness.DEFAULT_MODEL, harness.DEFAULT_RUNTIME_IMAGE)
    for name in SETTING_ENV:
        monkeypatch.setenv(name, f'value-of-{name}')
    args = cli.parse_args(['--case', 'single_snake', '--output', 'result.json'])
    assert (args.model, args.runtime_image) == ('value-of-LIVE_AGENT_POLICY_MODEL', 'value-of-NEXUS_RUNTIME_IMAGE')


# ── Policy request capture ─────────────────────────────────────────────────────


@pytest.mark.parametrize(
    ('span_name', 'client_key'),
    [
        ('openai_completion', 'external'),
        ('openai_completion', 'user_simulator'),
        ('openai_completion', None),
        ('openai_count_tokens', 'main'),
    ],
)
def test_policy_request_capture_ignores_non_policy_spans(
    tmp_path: Path, span_name: str, client_key: str | None
) -> None:
    output_path = tmp_path / 'policy_requests.jsonl'
    capture = cli._PolicyRequestProcessor(output_path, sensitive_source={})
    with processor_context([capture]):
        with span_context(span_name, attributes={'_client_key': client_key}):
            pass
    assert output_path.read_text() == '', 'Only main openai_completion spans may be exported'
    assert capture._stream.closed, 'Processor finalization must close the file'


@pytest.mark.parametrize('status_code', ['OK', 'ERROR'])
def test_policy_request_capture_flushes_redacted_start_and_end(tmp_path: Path, status_code: str) -> None:
    output_path = tmp_path / 'policy_requests.jsonl'
    output_path.write_text('{"previous":true}\n', encoding='utf-8')
    output_path.chmod(0o644)
    sensitive_source = {
        'LIVE_AGENT_POLICY_API_KEY': 'policy-credential-value',
        'LIVE_AGENT_EXTERNAL_API_KEY': 'external-credential-value',
        'GONGFENG_RUNTIME_TOKEN': 'runtime-credential-value',
    }
    messages = [
        {'role': 'system', 'content': 'Use the tools.'},
        {'role': 'user', 'content': ' '.join(sensitive_source.values())},
        {'role': 'tool', 'content': 'https://user:password@example.invalid/path?api_key=other-secret'},
    ]
    attributes = {
        '_client_key': 'main',
        'inputs': {
            'messages': messages,
            'request_meta': {'invocation_kind': 'policy', 'turn_step': 3, 'private': 'not-for-export'},
            'kwargs': {
                'tools': [{'type': 'function', 'function': {'name': 'delegate'}}],
                'tool_choice': 'auto',
                'parallel_tool_calls': False,
                'headers': {'Authorization': 'Bearer excluded-header'},
                'config': {'opaque': 'excluded-config'},
            },
            'model': 'requested-model',
        },
        'provider_call': {'private': 'excluded-provider-call'},
    }
    capture = cli._PolicyRequestProcessor(output_path, sensitive_source=sensitive_source)
    assert capture.dispatch_mode == 'inline', 'Request start must be saved before the provider can block'
    assert stat.S_IMODE(output_path.stat().st_mode) == 0o600, 'Existing request logs must become owner-only'
    with processor_context([capture]):
        with span_context('openai_completion', attributes=attributes) as request_span:
            records = [json.loads(line) for line in output_path.read_text().splitlines()]
            assert records[0] == {'previous': True}, 'Capture must append without deleting earlier records'
            assert len(records) == 2, 'The start record must be visible without end or finalize'
            start = records[1]
            assert set(start) == {
                'trace_id',
                'span_id',
                'model',
                'phase',
                'start_time_unix_nano',
                'status',
                'messages',
                'tools',
                'tool_choice',
                'parallel_tool_calls',
                'request_meta',
            }, 'Export only the request field allowlist'
            assert start['phase'] == 'start', 'Identify request start records'
            assert start['status'] == 'UNSET', 'Start must retain its initial state'
            assert start['model'] == 'requested-model', 'Record an explicitly supplied model argument'
            assert start['messages'][0] == messages[0], 'Non-sensitive policy messages must remain intact'
            assert start['messages'][1]['content'] == '<redacted> <redacted> <redacted>', 'Redact credential echoes'
            assert start['messages'][2]['content'].startswith('https://example.invalid/'), 'Strip URL credentials'
            assert 'other-secret' not in start['messages'][2]['content'], 'Redact URL query credentials'
            assert start['tools'] == attributes['inputs']['kwargs']['tools'], 'Preserve offered tool schemas'
            assert start['tool_choice'] == 'auto', 'Preserve tool selection mode'
            assert start['parallel_tool_calls'] is False, 'Preserve parallel tool call constraints'
            assert start['request_meta'] == {'invocation_kind': 'policy', 'turn_step': 3}, 'Filter request metadata'
            request_span.update(
                status=Status(code=status_code, message='policy-credential-value'),
                attributes={'outputs': {'private': 'excluded-response'}},
            )
        records = [json.loads(line) for line in output_path.read_text().splitlines()]
        end = records[-1]
        assert len(records) == 3, 'End must flush immediately without waiting for finalize'
        assert set(end) == {
            'trace_id',
            'span_id',
            'phase',
            'start_time_unix_nano',
            'end_time_unix_nano',
            'status',
            'status_message',
            'completion',
        }, 'End must export only completion fields, not raw span data'
        assert end['completion'] == {}, 'Unknown output fields must not be exported as completion data'
        assert end['status_message'] == '<redacted>', 'Redact credentials in the terminal status message'
        assert end['trace_id'] == start['trace_id'], 'Start and end must identify the same trace'
        assert end['span_id'] == start['span_id'], 'Start and end must identify the same request'
        assert end['phase'] == 'end', 'Identify request end records'
        assert end['status'] == status_code, 'Preserve the terminal status'
        assert end['end_time_unix_nano'] >= end['start_time_unix_nano'], 'Keep request timestamps'
    contents = output_path.read_text()
    for secret in sensitive_source.values():
        assert secret not in contents, 'Never export any configured credential'
    assert messages[1]['content'] == ' '.join(sensitive_source.values()), 'Redaction must not mutate policy inputs'
    assert capture._stream.closed, 'Finalization must close the JSONL stream'
    capture.finalize()


@pytest.mark.parametrize('client_key', ['main', 'external', 'user_simulator'])
@pytest.mark.parametrize('finish_reason', ['stop', 'tool_calls'])
async def test_policy_capture_observes_unmodified_completion_before_return(
    tmp_path: Path, client_key: str, finish_reason: str
) -> None:
    output_path = tmp_path / 'policy_requests.jsonl'
    messages = [{'role': 'user', 'content': 'Complete the task.'}]
    tools = [{'type': 'function', 'function': {'name': 'delegate'}}]
    tool_calls = (
        [{'id': 'invalid-call', 'type': 'function', 'function': {'name': 'unknown_tool', 'arguments': '{broken'}}]
        if finish_reason == 'tool_calls'
        else None
    )
    response = ExtendedChatCompletion.model_validate({
        'id': 'completion-final',
        'created': 1,
        'model': 'served-model',
        'choices': [
            {
                'index': 0,
                'finish_reason': finish_reason,
                'message': {
                    'role': 'assistant',
                    'content': '  最后的回答。\n ' if finish_reason == 'stop' else None,
                    'reasoning_content': 'Reasoning with policy-credential-value',
                    'tool_calls': tool_calls,
                },
            }
        ],
        'usage': {'prompt_tokens': 11, 'completion_tokens': 7, 'total_tokens': 18},
        'provider_specific_fields': {'private': 'excluded-provider-payload'},
    })
    expected_completion = response.model_dump(include={'id', 'model', 'choices', 'usage'})
    expected_completion['choices'][0]['message']['reasoning_content'] = 'Reasoning with <redacted>'

    class LocalPolicyClient(OpenAICompatibleClient):
        async def _create_completion(
            self, messages: list[dict[str, Any]], model: str | None = None, **kwargs: Any
        ) -> ExtendedChatCompletion:
            in_flight = [json.loads(line) for line in output_path.read_text().splitlines()]
            if client_key == 'main':
                assert len(in_flight) == 1, 'Observe must flush start before the client body runs'
                assert in_flight[0]['phase'] == 'start', 'Only start exists while the request is in flight'
            else:
                assert in_flight == [], 'Non-policy client starts must never be exported'
            messages[0]['content'] = 'Changed inside the client.'
            return response

    client = LocalPolicyClient(OpenAIConfig(client_args={}, request_args={}))
    client._client_key = client_key
    capture = cli._PolicyRequestProcessor(
        output_path, sensitive_source={'LIVE_AGENT_POLICY_API_KEY': 'policy-credential-value'}
    )
    with processor_context([capture]):
        returned = await client.create_completion(
            messages,
            model='requested-model',
            tools=tools,
            tool_choice='auto',
            parallel_tool_calls=False,
            request_meta={'invocation_kind': 'policy', 'turn_step': 4, 'private': 'excluded-request-meta'},
            extra_headers={'Authorization': 'excluded-header'},
        )
        saved = output_path.read_text()
        assert returned is response, 'Capture must not replace the response returned to the caller'
        assert returned.choices[0].message.reasoning_content == 'Reasoning with policy-credential-value', (
            'Redacting the persisted snapshot must not mutate the response seen by the agent'
        )
        returned.choices[0].message.content = 'Rewritten by the agent.'
        returned.choices[0].message.reasoning_content = None
        returned.choices[0].message.tool_calls = None
        returned.usage.completion_tokens = 99
        messages.append({'role': 'assistant', 'content': 'Rewritten history.'})
        tools[0]['function']['name'] = 'changed_tool'
        assert output_path.read_text() == saved, 'Later mutations must not alter the flushed request/response snapshot'
        if client_key != 'main':
            assert saved == '', 'External and user simulator completions must never be exported'
            return
        records = [json.loads(line) for line in saved.splitlines()]
        assert len(records) == 2, 'The last logical completion must have a start/end pair before finalization'
        start, end = records
        assert start['messages'] == [{'role': 'user', 'content': 'Complete the task.'}], 'Keep the original request'
        assert start['tools'][0]['function']['name'] == 'delegate', 'Keep the offered tool schema snapshot'
        assert start['model'] == 'requested-model', 'Capture the explicit model argument, not client configuration'
        assert start['request_meta'] == {'invocation_kind': 'policy', 'turn_step': 4}, 'Keep only policy metadata'
        assert end['phase'] == 'end', 'Identify the completed logical call'
        assert end['status'] == 'OK', 'Record the real successful terminal status'
        assert end['status_message'] is None, 'Success must not fabricate an error message'
        assert end['trace_id'] == start['trace_id'], 'Pair trace identities'
        assert end['span_id'] == start['span_id'], 'Pair each logical request by span identity'
        assert end['completion'] == expected_completion, 'Keep final text, reasoning, invalid tool calls and usage'
        assert end['end_time_unix_nano'] >= start['start_time_unix_nano'], 'Keep the full logical call interval'
        assert 'excluded-' not in saved, 'Do not dump client inputs, headers or provider-specific top-level outputs'
        assert 'policy-credential-value' not in saved, 'Redact credentials echoed in completion reasoning'
        assert stat.S_IMODE(output_path.stat().st_mode) == 0o600, 'New request logs must be owner-only'


async def test_policy_capture_observes_error_without_fabricating_completion(tmp_path: Path) -> None:
    output_path = tmp_path / 'policy_requests.jsonl'

    class FailingPolicyClient(OpenAICompatibleClient):
        async def _create_completion(
            self, messages: list[dict[str, Any]], model: str | None = None, **kwargs: Any
        ) -> ExtendedChatCompletion:
            raise RuntimeError('Provider rejected policy-credential-value')

    client = FailingPolicyClient(OpenAIConfig(client_args={}, request_args={}))
    client._client_key = 'main'
    capture = cli._PolicyRequestProcessor(
        output_path, sensitive_source={'LIVE_AGENT_POLICY_API_KEY': 'policy-credential-value'}
    )
    with processor_context([capture]):
        with pytest.raises(RuntimeError, match='Provider rejected policy-credential-value'):
            await client.create_completion([{'role': 'user', 'content': 'Complete the task.'}])
        start, end = [json.loads(line) for line in output_path.read_text().splitlines()]
        assert 'model' not in start, 'Do not infer an unspecified model from client configuration'
        assert end['phase'] == 'end', 'Observe must finish the failed logical call'
        assert end['status'] == 'ERROR', 'Observe must record the client exception'
        assert end['status_message'] == 'Provider rejected <redacted>', 'Preserve the error without credentials'
        assert 'completion' not in end, 'Failed calls must not fabricate a completion'
        assert 'choices' not in end, 'Failed calls must not fabricate model choices'
        assert end['trace_id'] == start['trace_id'], 'Errors must retain trace pairing'
        assert end['span_id'] == start['span_id'], 'Errors must retain logical request pairing'
        assert end['end_time_unix_nano'] >= start['start_time_unix_nano'], 'Errors must have an end timestamp'


@pytest.mark.parametrize(('run_success', 'validation_passed'), [(True, True), (True, False), (False, False)])
def test_cli_wires_policy_capture_and_preserves_result_validation(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    run_success: bool,
    validation_passed: bool,
) -> None:
    for name, value in (
        ('LIVE_AGENT_POLICY_API_KEY', 'policy-credential-value'),
        ('LIVE_AGENT_EXTERNAL_API_KEY', 'external-credential-value'),
        ('GONGFENG_RUNTIME_TOKEN', 'runtime-credential-value'),
        ('CODEBUDDY_API_KEY', 'codebuddy-credential-value'),
    ):
        monkeypatch.setenv(name, value)
    args = _cli_args(tmp_path)
    result = RunOutput(
        success=run_success,
        task_output={'policy_messages': [{'role': 'assistant', 'content': 'done'}]},
        error=None
        if run_success
        else ErrorInfo(
            type='RuntimeError',
            message=(
                'policy-credential-value external-credential-value runtime-credential-value codebuddy-credential-value'
            ),
        ),
    )
    validation = {'passed': validation_passed, 'checks': {'original': validation_passed}}
    captures = []

    async def run_with_hooks(run_input: RunInput, hooks: RunHooks) -> RunOutput:
        assert run_input.run_id == args.run_id, 'CLI must preserve the supplied run ID'
        assert len(hooks.span_processors) == 1, 'CLI must attach exactly one request capture processor'
        captures.extend(hooks.span_processors)
        with processor_context(hooks.span_processors):
            with span_context(
                'openai_completion',
                attributes={
                    '_client_key': 'main',
                    'inputs': {'messages': [{'role': 'user', 'content': 'policy-credential-value'}], 'kwargs': {}},
                },
            ):
                saved = (args.output.parent / 'policy_requests.jsonl').read_text()
                assert '"phase": "start"' in saved, 'CLI must expose an in-flight request'
                assert 'policy-credential-value' not in saved, 'CLI must pass environment credentials to redaction'
        return result

    def validate_output(task_output: Any, *, output_dir: Path | None = None, case_checks: Any = ()) -> dict[str, Any]:
        assert task_output is result.task_output, 'CLI must validate the original task output'
        assert output_dir == args.output.parent, 'CLI must point the validator at the files it wrote'
        assert tuple(case_checks) == (), 'single_snake asks for no case-specific check'
        return validation

    monkeypatch.setattr(cli, 'run', run_with_hooks)
    monkeypatch.setattr(cli, 'validate_run', validate_output)
    exit_code = asyncio.run(cli._run_case(args))
    assert exit_code == (0 if run_success and validation_passed else 1), 'Keep the existing exit code contract'
    payload = json.loads(args.output.read_text())
    assert payload == {**result.model_dump(mode='json'), 'example_validation': validation}, (
        'Capture must not rewrite final result or validation data'
    )
    summary_text = capsys.readouterr().out
    summary = json.loads(summary_text)
    assert summary['success'] is (run_success and validation_passed), 'Keep the existing success semantics'
    assert 'credential-value' not in summary_text, 'Never print credentials in error summaries'
    assert captures[0]._stream.closed, 'CLI must close the capture stream after runner completion'
    assert stat.S_IMODE((args.output.parent / 'policy_requests.jsonl').stat().st_mode) == 0o600, 'Use owner-only mode'


def test_cli_closes_policy_capture_if_runner_fails_before_processor_context(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    args = _cli_args(tmp_path)
    captures = []

    async def fail_before_context(run_input: RunInput, hooks: RunHooks) -> RunOutput:
        captures.extend(hooks.span_processors)
        raise RuntimeError('Runner setup failed')

    monkeypatch.setattr(cli, 'run', fail_before_context)
    with pytest.raises(RuntimeError, match='Runner setup failed'):
        asyncio.run(cli._run_case(args))
    assert captures[0]._stream.closed, 'CLI must close capture even if runner never enters its processor context'
    assert not args.output.exists(), 'A setup failure must not fabricate a result'
