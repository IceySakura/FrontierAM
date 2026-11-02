import base64
import hashlib
import json
import subprocess
import time

from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest
import yaml

from omegaconf import OmegaConf

from nexus.core import OpenAIConfig
from nexus.dataset import RemoteZipTaskFiles, task_files_from_spec
from nexus.extensions.agents.base.schema import TerminationReason
from nexus.extensions.openai_clients import openai_responses as responses_module
from nexus.extensions.openai_clients.vertex import VertexAnthropicHTTPError
from nexus.extensions.tasks.context_agent.cache_clients import (
    ContextAgentOpenAIResponsesClient,
    ContextAgentVertexClient,
    _ContextAgentVertexAnthropicAdapter,
)
from nexus.extensions.tasks.context_agent.enterprisebench_native.dataset import (
    DEFAULT_AGENT_NAME,
    build_item,
    filter_boot_tool_sets,
    filter_tool_sets,
    prepare_files_tree,
)
from nexus.extensions.tasks.context_agent.enterprisebench_native.dataset import (
    build_item as build_native_item,
)
from nexus.extensions.tasks.context_agent.enterprisebench_native.eval import aggregate
from nexus.extensions.tasks.context_agent.enterprisebench_native.evidence import (
    persist_run_records,
    persist_workspace_evidence,
)
from nexus.extensions.tasks.context_agent.enterprisebench_native.judge_proxy import (
    _OpenCodeResponsesReplay,
)
from nexus.extensions.tasks.context_agent.enterprisebench_native.judge_proxy import (
    opencode_responses_to_openai as native_opencode_responses_to_openai,
)
from nexus.extensions.tasks.context_agent.enterprisebench_native.openai_converter import (
    openai_to_responses_events as native_openai_to_responses_events,
)
from nexus.extensions.tasks.context_agent.enterprisebench_native.task import (
    EnterpriseBenchNativeTaskInput,
    EnterpriseBenchNativeTaskOutput,
    EnterpriseTaskSpec,
    _YamlAuthoritativeJudgeClient,
    _agent_termination_reason,
    _apply_environment_uploads,
    _boot_image_services,
    _build_enterprise_task_spec,
    _judge_launch_settings,
    _list_workdir_context_files,
    _opencode_events_to_messages,
    _read_runtime_verifier_result,
    _reward_for_mode,
    _run_enterprise_verifier,
    _skipped_judge_summary,
    _summarize_judge,
)
from nexus.extensions.tasks.context_agent.enterprisebench_native.task import (
    _agent_termination_reason as _native_agent_termination_reason,
)
from nexus.extensions.tasks.context_agent.enterprisebench_native.task import (
    _reward_for_mode as _native_reward_for_mode,
)
from nexus.extensions.tasks.context_agent.enterprisebench_native.verifier_config import (
    normalize_judge_config,
    render_judge_toml,
    render_opencode_config,
)
from nexus.extensions.tasks.context_agent.output import judge_skip_reason
from nexus.extensions.tasks.context_agent.task_files_auth import bind_task_wedata_key


REPO_ROOT = Path(__file__).resolve().parents[5]


def _write_task(task_dir: Path) -> None:
    environment = task_dir / 'environment'
    tests = task_dir / 'tests'
    environment.mkdir(parents=True)
    tests.mkdir()
    (task_dir / 'task.toml').write_text(
        '\n'.join(
            [
                '[environment]',
                'cpus = 2',
                'memory_mb = 4096',
                '[environment.env]',
                'INPUTDIR = "/data"',
                'OUTPUTDIR = "/data"',
                'WORLDBENCH_TOOL_SETS = "core slack google_drive"',
                '[[environment.mcp_servers]]',
                'name = "mcp-proxy"',
                'transport = "streamable-http"',
                'url = "http://localhost:8000/mcp"',
                '[agent]',
                'timeout_sec = 1800',
                '[verifier]',
                'timeout_sec = 3600',
            ]
        ),
        encoding='utf-8',
    )
    (environment / 'Dockerfile').write_text(
        '\n'.join(
            [
                'FROM syntara-base',
                'ENV WORLDBENCH_TOOL_SETS="core slack google_drive"',
                'ENV ANCHOR_DATETIME="2026-08-24T00:00:00Z"',
            ]
        ),
        encoding='utf-8',
    )
    (environment / 'docker-compose.yaml').write_text(
        'services:\n  mcp-proxy:\n    command: ["/app/scripts/start.sh", "--method", "http", "--port", "8000"]\n',
        encoding='utf-8',
    )
    (tests / 'judge.toml').write_text(
        '[[criterion]]\nname = "one"\ndescription = "Complete the task"\n[judge]\nmodel = "gpt-test"\n',
        encoding='utf-8',
    )
    (tests / 'test.sh').write_text('#!/bin/bash\nexit 0\n', encoding='utf-8')
    (tests / 'prompt_template.md').write_text(
        'Inspect.\n\nYour job is to grade the agent.\n\n{criteria}\n', encoding='utf-8'
    )
    (task_dir / 'instruction.md').write_text('Do the task.', encoding='utf-8')


async def test_boot_image_services_detaches_without_holding_command_pipe(tmp_path: Path) -> None:
    class _Runtime:
        async def run_command(self, *, command: str, timeout: float):
            started = time.monotonic()
            result = subprocess.run(
                command,
                shell=True,
                capture_output=True,
                text=True,
                timeout=timeout,
            )
            assert time.monotonic() - started < 1.0
            return type(
                'Result',
                (),
                {
                    'return_code': result.returncode,
                    'stdout': result.stdout,
                    'stderr': result.stderr,
                    'status': 'Completed',
                },
            )()

    done = tmp_path / 'done'
    task_input = EnterpriseBenchNativeTaskInput(
        run_agent=lambda **_kwargs: None,
        boot_command=f'sleep 2; touch {done}',
    )
    await _boot_image_services(_Runtime(), task_input)
    assert not done.exists()


async def test_list_workdir_context_files_builds_standard_workspace_manifest() -> None:
    commands: list[str] = []

    class _Runtime:
        async def run_command(self, *, command: str, timeout: float):
            commands.append(command)
            assert timeout == 60
            if command.startswith('find '):
                assert "find -L /workdir -path /workdir/workspace -prune -o -type f -printf '%P\\n'" in command
                stdout = 'Policy.pdf\nattachments/photo.jpg\nworkspace/.keep\n../escape.txt\n'
            else:
                stdout = ''
            return type(
                'Result',
                (),
                {
                    'return_code': 0,
                    'stdout': stdout,
                    'stderr': '',
                },
            )()

    assert await _list_workdir_context_files(_Runtime(), '/workdir', max_files=500) == [
        {
            'name': 'Policy.pdf',
            'uri': './workspace/Policy.pdf',
            'runtime_path': '/workdir/workspace/Policy.pdf',
        },
        {
            'name': 'attachments/photo.jpg',
            'uri': './workspace/attachments/photo.jpg',
            'runtime_path': '/workdir/workspace/attachments/photo.jpg',
        },
    ]
    assert commands[1].splitlines() == [
        'set -eu',
        'rm -rf /workdir/workspace',
        'mkdir -p /workdir/workspace',
        'ln -sfn /workdir/Policy.pdf /workdir/workspace/Policy.pdf',
        'mkdir -p /workdir/workspace/attachments',
        'ln -sfn /workdir/attachments/photo.jpg /workdir/workspace/attachments/photo.jpg',
    ]


async def test_environment_upload_creates_empty_target_when_task_files_drop_empty_directory(tmp_path: Path) -> None:
    commands: list[tuple[str, float]] = []

    class _Runtime:
        async def run_command(self, *, command: str, timeout: float):
            commands.append((command, timeout))
            return type('Result', (), {'return_code': 0, 'stdout': '', 'stderr': ''})()

    task_input = EnterpriseBenchNativeTaskInput(
        run_agent=lambda **_kwargs: None,
        environment_uploads=[
            {
                'source': 'environment/initial_workspace',
                'target': '/initial_workspace',
                'ensure_directory': True,
            }
        ],
    )

    await _apply_environment_uploads(
        _Runtime(),
        task_dir_path=tmp_path,
        workdir='/workdir',
        task_input=task_input,
    )

    assert commands == [('mkdir -p /initial_workspace', 60)]


def _native_judge_config() -> dict:
    return {
        'judge': {
            'judge': 'opencode',
            'prompt_template': 'prompt_template.md',
            'mode': 'batched',
        },
        'criteria': [{'name': 'c001', 'description': 'Do the task.', 'type': 'binary'}],
        'scoring': {'aggregation': 'weighted_mean', 'threshold': 0.5},
    }


def test_native_task_uses_nexus_item_config_without_reading_task_toml(tmp_path: Path) -> None:
    _write_task(tmp_path)
    (tmp_path / 'task.toml').write_text('this is deliberately invalid TOML', encoding='utf-8')

    task_input = EnterpriseBenchNativeTaskInput(
        run_agent=lambda **_kwargs: None,
        instance_id='task-id',
        benchmark_task_name='office-tasks/task-id',
        instruction='Configured instruction.',
        judge_config=_native_judge_config(),
    )
    spec = _build_enterprise_task_spec(tmp_path, task_input)

    assert spec.name == 'office-tasks/task-id'
    assert spec.instruction == 'Configured instruction.'
    assert spec.tests_dir == tmp_path / 'tests'


def test_native_judge_config_is_batched_and_rendered_by_nexus() -> None:
    normalized = normalize_judge_config(_native_judge_config())
    rendered = render_judge_toml(normalized, timeout=2400)

    assert normalized['judge']['mode'] == 'batched'
    assert 'mode = "batched"' in rendered
    assert 'model = "nexus-responses/judge"' in rendered
    assert '[[criterion]]' in rendered
    assert 'name = "c001"' in rendered
    assert 'description = "Do the task."' in rendered


def test_native_dataset_converts_task_and_judge_toml_into_item_config(tmp_path: Path) -> None:
    _write_task(tmp_path)
    (tmp_path / 'tests' / 'judge.toml').write_text(
        '[judge]\n'
        'judge = "opencode"\n'
        'model = "openai/old-model"\n'
        'prompt_template = "prompt_template.md"\n'
        'timeout = 2400\n\n'
        '[[criterion]]\n'
        'name = "c001"\n'
        'description = "Do the task."\n'
        'type = "binary"\n',
        encoding='utf-8',
    )

    item = build_native_item(
        'task-id',
        tmp_path,
        image='registry/syntara-base:v2',
        agent_name='context_agent_general',
        task_name='context_agent_enterprisebench_native_task',
        max_tool_rounds=999,
        cpu=2,
        memory_gb=10,
        port=8900,
    )

    task_config = item['body']['task_config']
    assert task_config['instruction'] == 'Do the task.'
    assert task_config['benchmark_task_name'] == 'task-id'
    assert task_config['judge_config']['judge'] == {
        'judge': 'opencode',
        'prompt_template': 'prompt_template.md',
        'mode': 'batched',
    }
    assert task_config['judge_config']['criteria'] == [
        {'name': 'c001', 'description': 'Do the task.', 'type': 'binary'}
    ]
    assert task_config['judge_mcp_url'] == 'http://localhost:8900/mcp'
    assert item['body']['task_name'] == 'context_agent_enterprisebench_native_task'


def test_native_dataset_keeps_case_owned_grading_context_helper(tmp_path: Path) -> None:
    tests_dir = tmp_path / 'tests'
    tests_dir.mkdir()
    (tests_dir / 'prompt_template.md').write_text('{criteria}\n', encoding='utf-8')
    helper = tests_dir / 'fetch_grading_context.py'
    helper.write_text('print("case-owned")\n', encoding='utf-8')
    for filename in ('judge.toml', 'opencode.json', 'rewardkit_opencode_agent.py', 'test.sh'):
        (tests_dir / filename).write_text('obsolete\n', encoding='utf-8')
    (tmp_path / 'task.toml').write_text('[task]\n', encoding='utf-8')
    (tmp_path / 'instruction.md').write_text('Do the task.\n', encoding='utf-8')

    prepare_files_tree(tmp_path)

    assert helper.read_text(encoding='utf-8') == 'print("case-owned")\n'
    assert (tests_dir / 'prompt_template.md').is_file()
    assert not (tmp_path / 'task.toml').exists()
    assert not (tmp_path / 'instruction.md').exists()
    for filename in ('judge.toml', 'opencode.json', 'rewardkit_opencode_agent.py', 'test.sh'):
        assert not (tests_dir / filename).exists()


def test_native_opencode_config_is_owned_by_nexus() -> None:
    config = json.loads(render_opencode_config(mcp_url='http://localhost:9999/mcp', output_limit=65536))

    assert config['mcp']['world']['url'] == 'http://localhost:9999/mcp'
    provider = config['provider']['nexus-responses']
    assert provider['options']['baseURL'] == '{env:NEXUS_JUDGE_PROXY_URL}'
    assert 'judge' in provider['models']
    assert provider['models']['judge']['limit']['output'] == 65536


@pytest.mark.asyncio
async def test_native_judge_yaml_request_args_override_opencode_defaults() -> None:
    class _Client:
        config = OpenAIConfig(
            client_args={},
            request_args={
                'model': 'deepseek-v4.1-flash-expires-on-0910',
                'max_output_tokens': 65536,
                'reasoning': {'effort': 'high'},
                'temperature': 1.0,
            },
        )

        def __init__(self):
            self.call = None

        async def create_completion(self, **kwargs):
            self.call = kwargs
            return 'completion'

    client = _Client()
    wrapped = _YamlAuthoritativeJudgeClient(client)
    result = await wrapped.create_completion(
        messages=[{'role': 'user', 'content': 'grade'}],
        model='judge',
        max_output_tokens=32768,
        reasoning={'effort': 'medium'},
        temperature=0.2,
        tools=[{'type': 'function', 'function': {'name': 'read'}}],
    )

    assert result == 'completion'
    assert client.call == {
        'messages': [{'role': 'user', 'content': 'grade'}],
        'model': None,
        'tools': [{'type': 'function', 'function': {'name': 'read'}}],
    }
    assert _judge_launch_settings(client) == (65536, 'high')


@pytest.mark.asyncio
async def test_native_verifier_preserves_reward_and_judge_artifacts(tmp_path: Path, monkeypatch) -> None:
    import nexus.extensions.tasks.context_agent.enterprisebench_native.task as native_task_module

    tests_dir = tmp_path / 'tests'
    tests_dir.mkdir()
    (tests_dir / 'test.sh').write_text('#!/bin/bash\n', encoding='utf-8')
    (tests_dir / 'prompt_template.md').write_text(
        'Inspect.\n\nYour job is to grade the agent.\n\n{criteria}\n',
        encoding='utf-8',
    )
    commands: list[dict] = []
    invoke_calls: list[tuple] = []

    class _Runtime:
        async def run_command(self, **kwargs):
            commands.append(kwargs)
            return SimpleNamespace(status='Finished', return_code=0, stdout='', stderr='')

        async def upload_file(self, files, overwrite):
            assert overwrite is True
            uploaded = {item.path: item.content for item in files}
            assert 'mode = "batched"' in uploaded['/tests/judge.toml']
            assert 'nexus-responses' in uploaded['/tests/opencode.json']
            assert 'NEXUS AGENT DELIVERABLES:' in uploaded['/tests/prompt_template.md']
            assert '/tests/nexus_judge_runtime.json' not in uploaded
            assert 'register_agent(OpenCodeCLI)' in uploaded['/tests/rewardkit_opencode_agent.py']
            assert "tempfile.mkdtemp(prefix='opencode-judge-')" in uploaded['/tests/rewardkit_opencode_agent.py']
            assert 'XDG_DATA_HOME=' in uploaded['/tests/rewardkit_opencode_agent.py']
            assert '/tests/fetch_grading_context.py' not in uploaded
            assert 'judge_runtime.py stage dependencies' in uploaded['/tests/test.sh']
            assert 'pip install' not in uploaded['/tests/test.sh']
            return SimpleNamespace(
                results=[SimpleNamespace(success=True, path=item.path, message='') for item in files]
            )

        async def download_file(self, *_args, **_kwargs):
            raise AssertionError('verifier results must be parsed in Runtime, not downloaded to Host')

        async def invoke(self, function, *args, **kwargs):
            invoke_calls.append((function, args, kwargs))
            assert function is _read_runtime_verifier_result
            assert args == ('/logs/verifier',)
            assert kwargs == {'stdout_limit': 10_000}
            return {
                'rewards': {'reward': 0.5},
                'reward_metadata': {'metadata': {'source': 'native'}},
                'reward_error': '',
                'test_stdout': 'judge output',
                'judge_details': {'reward': {'score': 0.5, 'criteria': []}},
                'judge_trajectories': [],
                'external_services_diff': '(no external-service state changes)\n',
            }

    async def fake_upload(runtime, source, target):
        assert source == tests_dir
        assert target == '/tests'

    monkeypatch.setattr(native_task_module, '_upload_path_to_runtime', fake_upload)

    rewards, metadata, stdout, details, trajectories, verifier_error, external_diff = await _run_enterprise_verifier(
        task_spec=EnterpriseTaskSpec(
            name='office-tasks/test',
            instruction='Do the task.',
            tests_dir=tests_dir,
            solution_dir=tmp_path / 'solution',
        ),
        judge_config=_native_judge_config(),
        judge_mcp_url='http://localhost:8900/mcp',
        runtime=_Runtime(),
        skip_tests_upload=False,
        timeout_sec=7200,
        judge_timeout_sec=2400,
        rewardkit_max_concurrent_llm=4,
        rewardkit_max_concurrent_agent=2,
    )

    assert rewards == {'reward': 0.5}
    assert metadata == {'metadata': {'source': 'native'}}
    assert stdout == 'judge output'
    assert details['reward']['score'] == 0.5
    assert trajectories == []
    assert verifier_error == ''
    assert external_diff == '(no external-service state changes)\n'
    assert len(invoke_calls) == 1
    assert commands[0]['extra_env']['REWARDKIT_MODEL'] == 'nexus-responses/judge'
    assert commands[0]['extra_env']['REWARDKIT_OPENCODE_TRACE_DIR'] == '/logs/verifier/opencode_judge'
    assert commands[0]['extra_env']['REWARDKIT_MAX_CONCURRENT_LLM'] == '4'
    assert commands[0]['extra_env']['REWARDKIT_MAX_CONCURRENT_AGENT'] == '2'
    assert 'REWARDKIT_OPENCODE_VARIANT' not in commands[0]['extra_env']
    assert 0 < commands[0]['timeout'] <= 2400


@pytest.mark.asyncio
async def test_enterprisebench_task_files_prefers_task_scoped_wedata_key(monkeypatch) -> None:
    import nexus.extensions.tasks.context_agent.task_files_auth as auth_module

    calls = []

    async def fake_sign(spec, *, auth):
        calls.append({'spec': spec, 'auth': auth})
        return {'urls': {spec.path: 'https://cos.example/signed'}}

    monkeypatch.setattr(auth_module, '_sign_remote_task_files', fake_sign)
    files = task_files_from_spec(
        {
            'type': 'zip',
            'path': 'blobs/case-001.zip',
            'size': 123,
            'remote': {'namespace': 'ns', 'dataset': 'ds', 'version': 'v2', 'split': 'test'},
        }
    )
    assert isinstance(files, RemoteZipTaskFiles)

    class _Context:
        task_files = files
        task_config = {'WEDATA_KEY': 'must-not-leak'}

    task_input = EnterpriseBenchNativeTaskInput(
        run_agent=lambda **_kwargs: None,
        WEDATA_KEY={'subject': 'task-user', 'key': 'task-cmk', 'id': 'task-id'},
    )
    scoped = bind_task_wedata_key(files, task_input.WEDATA_KEY)
    assert isinstance(scoped, RemoteZipTaskFiles)
    _Context.task_files = scoped
    signed_url = await scoped._get_signed_url()

    assert signed_url == 'https://cos.example/signed'
    assert calls == [
        {
            'spec': scoped.spec,
            'auth': {
                'kind': 'wedata_key',
                'key': {'subject': 'task-user', 'key': 'task-cmk', 'id': 'task-id'},
            },
        }
    ]
    assert 'WEDATA_KEY' not in task_input.model_dump()


def test_filter_tool_sets_supports_space_and_comma_separators() -> None:
    assert filter_tool_sets('core core_read core_write slack google_drive', {'core'}) == 'slack google_drive'
    assert filter_tool_sets('Core,core_read_multimodal,slack,google_drive', {'core'}) == 'slack,google_drive'
    assert filter_tool_sets('core', {'core'}) == ''


def test_filter_boot_tool_sets_removes_core_family() -> None:
    assert filter_boot_tool_sets(
        [
            '/app/scripts/start.sh',
            '--port',
            '8900',
            '--tool-sets',
            'core_read',
            'core_write',
            'google_mail_read',
            'suitecrm_write',
        ],
        {'core'},
    ) == [
        '/app/scripts/start.sh',
        '--port',
        '8900',
        '--tool-sets',
        'google_mail_read',
        'suitecrm_write',
    ]


def test_judge_summary_uses_one_or_nothing_and_preserves_rubric_accuracy() -> None:
    summary = _summarize_judge(
        {
            'reward': {
                'score': 0.5,
                'criteria': [
                    {'name': 'c001', 'value': 1.0, 'raw': 'yes', 'reasoning': 'present'},
                    {'name': 'c002', 'value': 0.0, 'raw': 'no', 'reasoning': 'missing'},
                ],
            }
        },
        [{'events': [{'type': 'step_start'}, {'type': 'step_finish'}]}],
        verifier_error='',
        fallback_reward=0.5,
    )

    assert summary['score'] == 0.0
    assert summary['sample_accuracy'] == 0.0
    assert summary['raw_reward'] == 0.5
    assert summary['rubric_accuracy'] == 0.5
    assert summary['criteria_passed'] == 1
    assert summary['criteria_total'] == 2
    assert summary['requirement_status'] == ['yes', 'no']
    assert summary['judge_rounds'] == 1
    assert summary['criteria'][0]['id'] == 'c001'
    assert summary['criteria'][0]['verdict'] == 'pass'
    assert summary['criteria'][0]['evidence'] == 'present'


def test_judge_summary_returns_one_only_when_every_criterion_passes() -> None:
    summary = _summarize_judge(
        {
            'reward': {
                'score': 1.0,
                'criteria': [
                    {'name': 'c001', 'value': 1.0},
                    {'name': 'c002', 'raw': 'yes'},
                ],
            }
        },
        [],
        verifier_error='',
        fallback_reward=1.0,
    )

    assert summary['score'] == 1.0
    assert summary['sample_accuracy'] == 1.0
    assert summary['rubric_accuracy'] == 1.0
    assert summary['criteria_passed'] == 2
    assert summary['criteria_total'] == 2


def test_enterprise_reward_mode_selects_sample_or_rubric_accuracy() -> None:
    summary = {'score': 0.0, 'sample_accuracy': 0.0, 'rubric_accuracy': 0.75}

    assert _reward_for_mode(summary, 'sample_acc') == 0.0
    assert _reward_for_mode(summary, 'rubric_acc') == 0.75
    assert _native_reward_for_mode(summary, 'sample_acc') == 0.0
    assert _native_reward_for_mode(summary, 'rubric_acc') == 0.75


def test_enterprise_output_carries_context_agent_grading_fields() -> None:
    fields = EnterpriseBenchNativeTaskOutput.model_fields
    assert {
        'final_answer',
        'grading_style',
        'judge_unavailable',
        'evaluation_skipped',
        'rationale',
        'requirement_status',
        'graded_on',
        'criteria_passed',
        'criteria_total',
        'judge_rounds',
        'criteria',
        'workspace_artifact',
        'termination_reason',
    }.issubset(fields)
    assert {'raw_reward', 'sample_accuracy', 'rubric_accuracy', 'evidence_artifact', 'external_services_diff'}.issubset(
        fields
    )
    assert {'judge_config', 'judge_mcp_url'} <= EnterpriseBenchNativeTaskInput.model_fields.keys()
    assert 'verifier_env' not in EnterpriseBenchNativeTaskInput.model_fields


def test_enterprise_agent_termination_reason_is_normalized() -> None:
    assert _agent_termination_reason(SimpleNamespace(exit_status=TerminationReason.MAX_ITERATIONS)) == 'MAX_STEPS'
    assert _agent_termination_reason({'exit_status': 'TRUNCATION'}) == 'TRUNCATION'
    assert _agent_termination_reason({'exit_status': 'PROVIDER_ERROR'}) == 'UNHANDLED_ERROR'
    assert _agent_termination_reason({'exit_status': 'SUCCESS'}) == 'ENV_DONE'
    assert _agent_termination_reason(None) == 'ENV_DONE'
    assert (
        _native_agent_termination_reason(SimpleNamespace(exit_status=TerminationReason.MAX_ITERATIONS)) == 'MAX_STEPS'
    )
    assert _native_agent_termination_reason({'exit_status': 'PROVIDER_ERROR'}) == 'UNHANDLED_ERROR'
    assert _native_agent_termination_reason({'exit_status': 'OUTPUT_LENGTH_EXCEEDED'}) == 'TRUNCATION'
    assert _native_agent_termination_reason({'exit_status': 'SUCCESS'}) == 'ENV_DONE'
    assert _native_agent_termination_reason(None) == 'ENV_DONE'
    assert EnterpriseBenchNativeTaskOutput(score=0.0).termination_reason == 'ENV_DONE'


@pytest.mark.parametrize('exit_status', ['TRUNCATION', 'OUTPUT_LENGTH_EXCEEDED'])
def test_native_enterprise_skips_judge_for_token_limit_exits(exit_status: str) -> None:
    reason = judge_skip_reason(SimpleNamespace(exit_status=exit_status), 'partial answer')
    assert reason is not None
    assert reason == f'Judge skipped because agent exited with {exit_status}.'
    summary = _skipped_judge_summary(reason)
    assert summary['score'] == 0.0
    assert summary['sample_accuracy'] == 0.0
    assert summary['judge_unavailable'] is False
    assert summary['criteria_total'] is None
    assert exit_status in summary['rationale']


@pytest.mark.parametrize('exit_status', ['SUCCESS', 'MAX_STEPS'])
def test_native_enterprise_keeps_judge_for_complete_or_max_step_exits(exit_status: str) -> None:
    assert judge_skip_reason({'exit_status': exit_status}, 'usable final answer') is None


def test_native_enterprise_skips_judge_for_max_steps_without_final_answer() -> None:
    assert judge_skip_reason({'exit_status': 'MAX_STEPS'}, None) == (
        'Judge skipped because agent exited with MAX_STEPS without a final answer.'
    )


def test_aggregate_reports_binary_score_and_fractional_rubric_accuracy(tmp_path: Path) -> None:
    results = tmp_path / 'results.jsonl'
    results.write_text(
        json.dumps(
            {
                'status': 'success',
                'metadata': {'task_id': 'task-one', 'world': 'mail', 'n_criteria': 26},
                'task_output': {
                    'score': 0.0,
                    'raw_reward': 0.9615,
                    'rubric_accuracy': 25 / 26,
                    'criteria_passed': 25,
                    'criteria_total': 26,
                    'verifier_error': '',
                },
            }
        )
        + '\n',
        encoding='utf-8',
    )

    report = aggregate(results)

    assert report['overall']['mean_score'] == 0.0
    assert report['overall']['mean_rubric_accuracy'] == 0.9615
    assert report['overall']['solved'] == 0


async def test_workspace_evidence_saves_manifest_and_only_changed_files(tmp_path: Path) -> None:
    initial = tmp_path / 'initial'
    final = tmp_path / 'final'
    initial.mkdir()
    final.mkdir()
    (initial / 'same.txt').write_text('same', encoding='utf-8')
    (final / 'same.txt').write_text('same', encoding='utf-8')
    (initial / 'modified.txt').write_text('before', encoding='utf-8')
    (final / 'modified.txt').write_text('after', encoding='utf-8')
    (initial / 'deleted.txt').write_text('deleted', encoding='utf-8')
    (final / 'created.txt').write_text('created', encoding='utf-8')

    class _Runtime:
        async def invoke(self, function, *args):
            return function(*args)

        async def download_file(self, files):
            return SimpleNamespace(
                errors=[],
                files=[
                    SimpleNamespace(
                        content=base64.b64encode(Path(files[0].path).read_bytes()).decode('ascii'),
                    )
                ],
            )

    result = await persist_workspace_evidence(
        _Runtime(),
        artifact_root=tmp_path / 'artifacts',
        run_id='run/one',
        task_id='task-one',
        final_root=str(final),
        initial_root=str(initial),
    )

    assert result['available'] is True
    assert result['counts'] == {'created': 1, 'modified': 1, 'unchanged': 1, 'deleted': 1}
    changed = Path(result['changed_files_path'])
    assert (changed / 'created.txt').read_text(encoding='utf-8') == 'created'
    assert (changed / 'modified.txt').read_text(encoding='utf-8') == 'after'
    assert not (changed / 'same.txt').exists()
    assert not (changed / 'deleted.txt').exists()


def test_persist_run_records_saves_judge_and_service_evidence(tmp_path: Path) -> None:
    result = persist_run_records(
        artifact_root=tmp_path,
        run_id='run-one',
        final_answer='answer',
        agent_trajectory=[{'role': 'assistant', 'content': 'answer'}],
        judge_details={'reward': {'score': 1.0}},
        judge_trajectories=[{'request': {'prompt': 'grade'}, 'messages': []}],
        external_services_diff='<services>[no changes]</services>\n',
        result_summary={'score': 1.0, 'criteria_passed': 2, 'criteria_total': 2},
    )

    assert result['available'] is True
    paths = result['paths']
    assert Path(paths['agent_final_answer']).read_text(encoding='utf-8') == 'answer'
    assert Path(paths['external_services_diff']).read_text(encoding='utf-8').startswith('<services>')
    assert '"prompt": "grade"' in Path(paths['judge_requests']).read_text(encoding='utf-8')
    assert '"criteria_total": 2' in Path(paths['result_summary']).read_text(encoding='utf-8')


def test_native_converter_preserves_history_until_replay_validation() -> None:
    dsv4 = native_opencode_responses_to_openai(
        {
            'model': 'judge',
            'input': [
                {
                    'type': 'reasoning',
                    'id': 'rs-dsv4',
                    'summary': [],
                    'content': [{'type': 'reasoning_text', 'text': 'Inspect first.'}],
                    'encrypted_content': 'dsv4-token',
                },
                {
                    'type': 'function_call',
                    'call_id': 'call-1',
                    'name': 'read',
                    'arguments': '{"filePath":"/tmp/answer.txt"}',
                },
                {'type': 'function_call_output', 'call_id': 'call-1', 'output': 'answer'},
            ],
        }
    )
    assert dsv4['messages'][0]['tool_calls'][0]['id'] == 'call-1'
    assert dsv4['messages'][0]['reasoning_items'][0]['id'] == 'rs-dsv4'
    assert dsv4['messages'][1] == {'role': 'tool', 'tool_call_id': 'call-1', 'content': 'answer'}
    gpt = native_opencode_responses_to_openai(
        {
            'model': 'judge',
            'input': [
                {
                    'type': 'reasoning',
                    'id': 'rs-gpt',
                    'summary': [{'type': 'summary_text', 'text': 'Inspect first.'}],
                    'encrypted_content': 'gpt-token-copy',
                },
                {
                    'type': 'function_call',
                    'call_id': 'call-2',
                    'name': 'read',
                    'arguments': '{}',
                },
            ],
        }
    )
    assert gpt['messages'][0]['tool_calls'][0]['id'] == 'call-2'
    assert gpt['messages'][0]['reasoning_items'][0]['id'] == 'rs-gpt'
    with pytest.raises(ValueError, match='unfinished tool calls'):
        _OpenCodeResponsesReplay()._inject_output_turns(gpt['messages'])


def test_native_proxy_replays_provider_output_before_matching_tool_result() -> None:
    replay = _OpenCodeResponsesReplay()
    reasoning = {
        'type': 'reasoning',
        'id': 'rs-provider',
        'summary': [],
        'content': [{'type': 'reasoning_text', 'text': 'Inspect first.'}],
        'encrypted_content': 'provider-token',
        'status': 'completed',
    }
    function_call = {
        'type': 'function_call',
        'id': 'fc-provider',
        'call_id': 'call-1',
        'name': 'read',
        'arguments': '{"filePath":"/tmp/answer.txt"}',
        'status': 'completed',
    }
    response = {
        'choices': [
            {
                'message': {
                    'reasoning_items': [reasoning],
                    'tool_calls': [
                        {
                            'id': 'call-1',
                            'type': 'function',
                            'function': {'name': 'read', 'arguments': '{"filePath":"/tmp/answer.txt"}'},
                        }
                    ],
                    'provider_specific_fields': {'responses_output_items': [reasoning, function_call]},
                }
            }
        ]
    }

    response['provider_specific_fields'] = {
        'response_status': 'completed',
        'response_state': {'id': 'resp-replay', 'status': 'completed'},
        'response_output': [reasoning, function_call],
    }
    replay.response_to_events(response)
    converted = replay.request_to_openai(
        {
            'model': 'judge',
            'input': [
                {'type': 'reasoning', 'id': 'rs-rebuilt', 'summary': [], 'encrypted_content': 'wrong-token'},
                {
                    'type': 'function_call',
                    'call_id': 'call-1',
                    'name': 'read',
                    'arguments': '{"filePath":"/tmp/answer.txt"}',
                },
                {'type': 'function_call_output', 'call_id': 'call-1', 'output': 'answer'},
            ],
        }
    )

    assert converted['messages'] == [
        {
            'role': 'assistant',
            'content': None,
            'reasoning_items': [reasoning],
            'reasoning_content': 'Inspect first.',
            'tool_calls': [
                {
                    'id': 'call-1',
                    'type': 'function',
                    'function': {'name': 'read', 'arguments': '{"filePath":"/tmp/answer.txt"}'},
                }
            ],
            'provider_specific_fields': {'responses_output_items': [reasoning, function_call]},
        },
        {'role': 'tool', 'tool_call_id': 'call-1', 'content': 'answer'},
    ]
    assert responses_module._responses_input(converted['messages']) == [
        reasoning,
        function_call,
        {'type': 'function_call_output', 'call_id': 'call-1', 'output': 'answer'},
    ]


def test_native_proxy_exposes_replayed_response_text_to_chat_consumers() -> None:
    replay = _OpenCodeResponsesReplay()
    message = {
        'type': 'message',
        'id': 'msg-provider',
        'role': 'assistant',
        'status': 'completed',
        'content': [{'type': 'output_text', 'text': 'I will inspect the answer.', 'annotations': []}],
    }
    function_call = {
        'type': 'function_call',
        'id': 'fc-provider',
        'call_id': 'call-1',
        'name': 'read',
        'arguments': '{}',
        'status': 'completed',
    }
    replay.response_to_events(
        {
            'provider_specific_fields': {
                'response_status': 'completed',
                'response_state': {'id': 'resp-text', 'status': 'completed'},
                'response_output': [message, function_call],
            },
            'choices': [
                {
                    'message': {
                        'content': 'I will inspect the answer.',
                        'tool_calls': [
                            {'id': 'call-1', 'type': 'function', 'function': {'name': 'read', 'arguments': '{}'}}
                        ],
                        'provider_specific_fields': {'responses_output_items': [message, function_call]},
                    }
                }
            ],
        }
    )

    converted = replay.request_to_openai(
        {
            'model': 'judge',
            'input': [{'type': 'function_call_output', 'call_id': 'call-1', 'output': 'answer'}],
        }
    )

    assert converted['messages'][0] == {
        'role': 'assistant',
        'content': 'I will inspect the answer.',
        'tool_calls': [{'id': 'call-1', 'type': 'function', 'function': {'name': 'read', 'arguments': '{}'}}],
        'provider_specific_fields': {'responses_output_items': [message, function_call]},
    }


def test_native_proxy_rejects_unknown_tool_result() -> None:
    replay = _OpenCodeResponsesReplay()

    with pytest.raises(ValueError, match='call-missing.*no cached provider output turn'):
        replay.request_to_openai(
            {
                'model': 'judge',
                'input': [{'type': 'function_call_output', 'call_id': 'call-missing', 'output': 'answer'}],
            }
        )


def test_opencode_events_convert_to_chat_trajectory() -> None:
    messages = _opencode_events_to_messages(
        [
            {'type': 'step_start'},
            {'type': 'reasoning', 'part': {'text': 'Inspect the answer.'}},
            {
                'type': 'tool_use',
                'part': {
                    'callID': 'call-1',
                    'tool': 'read',
                    'state': {'input': {'path': '/tmp/agent_deliverable/answer.txt'}, 'output': 'answer text'},
                },
            },
            {'type': 'step_start'},
            {'type': 'text', 'part': {'text': '{"c001":{"score":"yes","reasoning":"present"}}'}},
            {'type': 'step_finish'},
        ]
    )

    assert messages[0]['reasoning_content'] == 'Inspect the answer.'
    assert messages[0]['tool_calls'][0]['function']['name'] == 'read'
    assert messages[1] == {
        'role': 'tool',
        'tool_call_id': 'call-1',
        'name': 'read',
        'content': 'answer text',
    }
    assert messages[-1]['content'].startswith('{"c001"')


def test_native_converter_exposes_reasoning_text_to_opencode_trace() -> None:
    events = native_openai_to_responses_events(
        {
            'id': 'resp-dsv4',
            'model': 'deepseek-v4.1-flash-expires-on-0910',
            'choices': [
                {
                    'message': {
                        'content': None,
                        'reasoning_items': [
                            {
                                'type': 'reasoning',
                                'id': 'reasoning-1',
                                'status': 'completed',
                                'summary': [],
                                'content': [{'type': 'reasoning_text', 'text': 'Inspect the workspace first.'}],
                                'encrypted_content': 'opaque-replay-token',
                            }
                        ],
                    }
                }
            ],
        }
    )

    completed = next(
        event['item']
        for event in events
        if event.get('type') == 'response.output_item.done' and event.get('item', {}).get('type') == 'reasoning'
    )
    assert completed['summary'] == [{'type': 'summary_text', 'text': 'Inspect the workspace first.'}]
    assert completed['content'] == [{'type': 'reasoning_text', 'text': 'Inspect the workspace first.'}]
    assert completed['encrypted_content'] == 'opaque-replay-token'
    assert any(
        event.get('type') == 'response.reasoning_summary_text.delta'
        and event.get('delta') == 'Inspect the workspace first.'
        for event in events
    )


def test_read_runtime_verifier_result_parses_files_without_host_download(tmp_path: Path) -> None:
    verifier = tmp_path / 'verifier'
    trace_dir = verifier / 'opencode_judge'
    trace_dir.mkdir(parents=True)
    (verifier / 'reward.json').write_text(
        '{"reward":0.75,"metadata":{"source":"runtime"}}',
        encoding='utf-8',
    )
    (verifier / 'reward-details.json').write_text(
        '{"reward":{"score":0.75,"criteria":[]}}',
        encoding='utf-8',
    )
    (verifier / 'test-stdout.txt').write_text('judge output', encoding='utf-8')
    (trace_dir / 'judge-1.request.json').write_text('{"prompt":"grade","schema":{}}', encoding='utf-8')
    (trace_dir / 'judge-1.latest.jsonl').write_text(
        '{"type":"step_start"}\n'
        '{"type":"reasoning","part":{"text":"inspect"}}\n'
        '{"type":"text","part":{"text":"{\\"c001\\":{\\"score\\":\\"yes\\"}}"}}\n'
        '{"type":"step_finish"}\n',
        encoding='utf-8',
    )
    external_diff = tmp_path / 'external-services.md'
    external_diff.write_text('<services>[no changes]</services>\n', encoding='utf-8')

    result = _read_runtime_verifier_result(
        str(verifier),
        stdout_limit=5,
        external_services_diff_path=str(external_diff),
    )

    assert result['rewards'] == {'reward': 0.75}
    assert result['reward_metadata'] == {'metadata': {'source': 'runtime'}}
    assert result['reward_error'] == ''
    assert result['test_stdout'] == 'judge'
    assert result['judge_details']['reward']['score'] == 0.75
    assert result['judge_trajectories'][0]['messages'][1]['reasoning_content'] == 'inspect'
    assert result['external_services_diff'] == '<services>[no changes]</services>\n'


def test_enterprisebench_policy_uses_opus48_vertex_profile(monkeypatch) -> None:
    monkeypatch.setenv('CTXAGENT_POLICY_API_KEY', 'test-token?provider=aws_third')
    monkeypatch.setenv('CTXAGENT_JUDGE_API_KEY', 'judge-app:judge-key')
    config = OmegaConf.to_container(
        OmegaConf.load(REPO_ROOT / 'nexus/extensions/tasks/context_agent/enterprisebench_native/base.yaml'),
        resolve=True,
    )
    assert isinstance(config, dict)
    main = config['body']['openai_clients']['main']
    judge = config['body']['openai_clients']['judge']

    assert config['body']['runtime_provider_config']['network_policy'] == {
        'default_action': 'deny',
        'egress': [],
    }
    assert main['name'] == 'context_agent_vertex'
    assert main['config']['client_args'] == {
        'protocol': 'anthropic',
        'api_key': 'test-token?provider=aws_third',
        'cache_task_id_prefix': 'enterprisebench-opus48',
        'endpoint': 'http://api.aihub.woa.com/model/anthropic.claude-opus-4-8/invoke',
        'timeout': 7200,
        'max_retries': 10,
        'cache_control_enabled': True,
        'cache_control_strategy': 'claude_code',
    }
    assert main['config']['request_args'] == {
        'model': 'anthropic.claude-opus-4-8',
        'anthropic_version': 'bedrock-2023-05-31',
        'thinking': {'type': 'adaptive', 'display': 'summarized'},
        'output_config': {'effort': 'medium'},
        'max_tokens': 128000,
        'drop_forced_tool_choice_with_thinking': True,
    }
    assert judge['name'] == 'context_agent_openai_responses'
    assert judge['config']['client_args']['max_retries'] == 2
    assert judge['config']['client_args']['cache_task_id_prefix'] == 'enterprisebench-gpt56-judge'
    assert judge['config']['request_args']['model'] == 'gpt-5.6-sol'
    assert judge['config']['request_args']['reasoning']['effort'] == 'high'


def test_enterprisebench_cache_clients_generate_unique_per_run_ids(monkeypatch) -> None:
    vertex_keys: list[str] = []
    monkeypatch.setattr(
        'nexus.extensions.openai_clients.vertex.VertexAnthropicAdapter._post_once',
        lambda *_args, **_kwargs: None,
    )
    policy_config = OpenAIConfig(
        client_args={
            'protocol': 'anthropic',
            'api_key': 'policy-secret?provider=aws_third',
            'endpoint': 'https://vertex.test/invoke',
            'cache_task_id_prefix': 'policy-case',
        },
        request_args={'model': 'claude-opus-4-8'},
    )
    first_policy = ContextAgentVertexClient(policy_config)
    second_policy = ContextAgentVertexClient(policy_config)
    vertex_keys.extend([first_policy._adapter.api_key, second_policy._adapter.api_key])
    assert vertex_keys[0] != vertex_keys[1]
    assert all('cache_task_id=policy-case%3A' in key for key in vertex_keys)

    init_calls: list[dict] = []

    class FakeAsyncOpenAI:
        def __init__(self, **kwargs):
            init_calls.append(kwargs)

    monkeypatch.setattr('nexus.extensions.openai_clients.openai_responses.openai.AsyncOpenAI', FakeAsyncOpenAI)
    judge_config = OpenAIConfig(
        client_args={
            'api_key': 'judge-secret?provider=azure',
            'cache_task_id_prefix': 'judge-case',
        },
        request_args={'model': 'gpt-5.6-sol'},
    )
    ContextAgentOpenAIResponsesClient(judge_config)
    ContextAgentOpenAIResponsesClient(judge_config)
    first_hash = init_calls[0]['api_key'].split('cache_task_id=', 1)[1]
    second_hash = init_calls[1]['api_key'].split('cache_task_id=', 1)[1]
    assert first_hash != second_hash
    assert len(first_hash) == len(hashlib.md5(b'x').hexdigest())


def test_context_agent_cache_clients_leave_cache_id_empty_by_default(monkeypatch) -> None:
    policy_config = OpenAIConfig(
        client_args={
            'protocol': 'anthropic',
            'api_key': 'policy-secret?provider=aws_third',
            'endpoint': 'https://vertex.test/invoke',
        },
        request_args={'model': 'claude-opus-4-8'},
    )
    policy = ContextAgentVertexClient(policy_config)
    assert 'cache_task_id=' not in policy._adapter.api_key

    init_calls: list[dict] = []

    class FakeAsyncOpenAI:
        def __init__(self, **kwargs):
            init_calls.append(kwargs)

    monkeypatch.setattr('nexus.extensions.openai_clients.openai_responses.openai.AsyncOpenAI', FakeAsyncOpenAI)
    judge_config = OpenAIConfig(
        client_args={'api_key': 'judge-secret?provider=azure'},
        request_args={'model': 'gpt-5.6-sol'},
    )
    ContextAgentOpenAIResponsesClient(judge_config)
    assert init_calls[0]['api_key'] == 'judge-secret?provider=azure'


@pytest.mark.asyncio
async def test_context_agent_vertex_retries_aihub_user_auth_2003(monkeypatch) -> None:
    client = ContextAgentVertexClient(
        OpenAIConfig(
            client_args={
                'protocol': 'anthropic',
                'api_key': 'policy-secret?provider=aws_third',
                'endpoint': 'https://vertex.test/invoke',
                'max_retries': 3,
            },
            request_args={'model': 'claude-opus-4-8'},
        )
    )
    transient_401 = VertexAnthropicHTTPError(
        401,
        json.dumps(
            {
                'error': {
                    'message': '鉴权失败，请检查 API Key',
                    'type': 'UserAuthError',
                    'code': '2003',
                }
            }
        ),
    )
    post_once = AsyncMock(side_effect=[transient_401, {'ok': True}])
    sleep = AsyncMock()
    assert isinstance(client._adapter, _ContextAgentVertexAnthropicAdapter)
    monkeypatch.setattr(client._adapter, '_post_once', post_once)
    monkeypatch.setattr('nexus.extensions.tasks.context_agent.cache_clients.asyncio.sleep', sleep)
    provider_call = {'attempts': []}

    response = await client._adapter._post_json('https://vertex.test/invoke', {}, provider_call)

    assert response == {'ok': True}
    assert post_once.await_count == 2
    sleep.assert_awaited_once_with(1)
    assert [attempt['status'] for attempt in provider_call['attempts']] == ['error', 'success']
    assert provider_call['attempts'][0]['status_code'] == 401


@pytest.mark.asyncio
@pytest.mark.parametrize(
    'error',
    [
        VertexAnthropicHTTPError(401, '{"error":{"type":"UserAuthError","code":"different"}}'),
        VertexAnthropicHTTPError(401, 'not-json'),
        VertexAnthropicHTTPError(403, '{"error":{"type":"UserAuthError","code":"2003"}}'),
    ],
)
async def test_context_agent_vertex_does_not_retry_other_auth_errors(monkeypatch, error) -> None:
    client = ContextAgentVertexClient(
        OpenAIConfig(
            client_args={
                'protocol': 'anthropic',
                'api_key': 'policy-secret?provider=aws_third',
                'endpoint': 'https://vertex.test/invoke',
                'max_retries': 3,
            },
            request_args={'model': 'claude-opus-4-8'},
        )
    )
    post_once = AsyncMock(side_effect=error)
    sleep = AsyncMock()
    assert isinstance(client._adapter, _ContextAgentVertexAnthropicAdapter)
    monkeypatch.setattr(client._adapter, '_post_once', post_once)
    monkeypatch.setattr('nexus.extensions.tasks.context_agent.cache_clients.asyncio.sleep', sleep)

    with pytest.raises(VertexAnthropicHTTPError):
        await client._adapter._post_json('https://vertex.test/invoke', {}, {'attempts': []})

    assert post_once.await_count == 1
    sleep.assert_not_awaited()


def test_enterprisebench_single_example_matches_other_context_agent_configs(monkeypatch) -> None:
    monkeypatch.setenv('CTXAGENT_POLICY_API_KEY', 'test-token?provider=aws_third')
    monkeypatch.setenv('CTXAGENT_JUDGE_API_KEY', 'judge-app:judge-key')
    native_example = OmegaConf.to_container(
        OmegaConf.load(REPO_ROOT / 'nexus/extensions/tasks/context_agent/enterprisebench_native/single_case.yaml'),
        resolve=True,
    )
    assert isinstance(native_example, dict)
    assert native_example['compose'][0]['data']['uri'] == 'yufeihuang/enterprisebench_native_eval_75@v1'
    assert native_example['compose'][0]['data']['split'] == 'enterprisebench_native_eval_75'
    assert native_example['compose'][1]['use'] == 'base.yaml'

    ds_v4_example = OmegaConf.to_container(
        OmegaConf.load(
            REPO_ROOT / 'nexus/extensions/tasks/context_agent/enterprisebench_native/single_case_ds_v4_judge.yaml'
        ),
        resolve=True,
    )
    assert isinstance(ds_v4_example, dict)
    ds_v4_judge = ds_v4_example['compose'][2]['input']['openai_clients']['judge']
    assert ds_v4_example['compose'][1]['use'] == 'base.yaml'
    assert ds_v4_judge['name'] == 'context_agent_openai_responses'
    assert ds_v4_judge['config']['client_args']['endpoint'] == 'http://api.aihub.woa.com/v1/responses'
    assert ds_v4_judge['config']['client_args']['api_key'].endswith('?provider=deepseek&timeout=7200')
    assert ds_v4_judge['config']['request_args'] == {
        'model': 'deepseek-v4.1-flash-expires-on-0910',
        'reasoning': {'effort': 'high'},
        'store': False,
        'max_output_tokens': 65536,
    }
    assert native_example['compose'][2]['input']['task_config']['artifact_root'] == (
        'workspace/enterprisebench_single_native/artifacts'
    )

    native_base = OmegaConf.to_container(
        OmegaConf.load(REPO_ROOT / 'nexus/extensions/tasks/context_agent/enterprisebench_native/base.yaml'),
        resolve=True,
    )
    assert isinstance(native_base, dict)
    assert native_base['body']['task_name'] == 'context_agent_enterprisebench_native_task'
    assert 'nexus.extensions.tasks.context_agent.enterprisebench_native.task' in native_base['body']['imports']
    assert native_base['body']['processors']['tokens']['args']['client_keys'] == ['main', 'judge']
    assert native_base['body']['runner']['timeout'] == 21600


def test_enterprise_item_uses_context_agent_and_disables_world_core(tmp_path: Path) -> None:
    task_dir = tmp_path / 'task-id'
    _write_task(task_dir)

    item = build_item(
        'task-id',
        task_dir,
        image='registry/syntara-base:v2',
        agent_name=DEFAULT_AGENT_NAME,
        task_name='context_agent_enterprisebench_native_task',
        max_tool_rounds=999,
        cpu=2,
        memory_gb=10,
        port=8900,
    )

    assert item['body']['agent_name'] == 'context_agent_general'
    assert 'working_dir' not in item['body']['agent_config']
    assert item['body']['agent_config']['max_tool_rounds'] == 999
    assert 'max_iterations' not in item['body']['agent_config']
    assert item['body']['agent_config']['enabled_tools'] is None
    assert item['body']['agent_config']['tool_search'] == {'enabled': True, 'defer_all_mcp': True}
    assert item['body']['runtime_configs']['main']['resource_spec'] == {'cpu': 2, 'memory': 10}
    assert item['body']['task_config']['expose_workdir_files'] is True
    assert 'verifier_env' not in item['body']['task_config']
    assert item['body']['task_name'] == 'context_agent_enterprisebench_native_task'
    EnterpriseBenchNativeTaskInput.model_validate(
        {
            **item['body']['task_config'],
            'run_agent': lambda **_kwargs: None,
        }
    )
    assert item['body']['task_config']['boot_env']['WORLDBENCH_TOOL_SETS'] == 'slack google_drive'
    assert item['body']['task_config']['environment_uploads'] == [
        {'source': 'environment/initial_external_services', 'target': '/initial_data'},
        {
            'source': 'environment/initial_workspace',
            'target': '/initial_workspace',
            'ensure_directory': True,
        },
    ]
    assert item['body']['task_config']['boot_command'].startswith('mkdir -p /initial_workspace /data /workdir')
    assert 'core' not in item['body']['task_config']['boot_command'].lower()
    assert item['metadata']['tool_sets'] == 'slack google_drive'
    assert item['body']['agent_config']['mcp_servers'][0]['url'] == 'http://localhost:8900/mcp'
    assert item['body']['runtime_configs']['main']['worker_venv'] == {'modules': []}

    yaml.safe_dump(item)
