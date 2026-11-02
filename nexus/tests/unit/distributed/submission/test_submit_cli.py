from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import zipfile

from copy import deepcopy

import aiohttp
import pytest

from aiohttp import web
from aiohttp.test_utils import TestServer

from nexus.cli.commands import submit as submit_module
from nexus.cli.commands.submit import _read_taihu_token, register_submit_parser
from nexus.distributed.batch.materialize import MaterializedBatchArtifacts, MaterializedDataList, MaterializedTaskList
from nexus.distributed.submission.source_package import PackagedFile
from nexus.distributed.submission.specs import (
    BatchSubmission,
    ConfigOnlyCode,
    ImageCode,
    LiveSubmission,
    SingleSubmission,
    SourcePackageCode,
)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    register_submit_parser(parser.add_subparsers(dest='command'))
    return parser


@pytest.mark.parametrize('mode', ['single', 'live', 'batch'])
def test_auto_cpu_increase_cli_normalizes_dynamic_managers(mode):
    args = _parser().parse_args([
        'submit',
        mode,
        '--environment',
        'dynamic',
        '--manager-cpu-cores',
        '2',
        '--manager-memory-gib',
        '17',
        '--auto-increase-cpu',
        *(['--batch-manager-memory-gib', '8'] if mode == 'batch' else []),
    ])
    submit_module._validate_submission_args(args)
    assert args.manager_cpu_cores == 5
    assert args.manager_memory_gib == 17
    if mode == 'batch':
        assert args.batch_manager_cpu_cores == 2


@pytest.mark.parametrize('mode', ['single', 'live', 'batch'])
async def test_invalid_cpu_memory_ratio_fails_before_packaging_or_upload(mode, monkeypatch):
    captured = _stub_submit_io(monkeypatch)
    args = _parser().parse_args([
        'submit',
        mode,
        '--environment',
        'dynamic',
        '--manager-cpu-cores',
        '2',
        '--manager-memory-gib',
        '16',
        '--no-auto-increase-cpu',
    ])
    with pytest.raises(ValueError, match='enable auto_increase_cpu'):
        await submit_module._submit(args)
    assert captured['uploads'] == []
    assert captured['requests'] == []
    assert 'source_excluded_paths' not in captured


@pytest.mark.parametrize('mode', ['single', 'live', 'batch'])
def test_omitted_auto_cpu_flag_defers_to_server(mode):
    args = _parser().parse_args([
        'submit',
        mode,
        '--environment',
        'dynamic',
        '--manager-cpu-cores',
        '1',
        '--manager-memory-gib',
        '16',
    ])
    submit_module._validate_submission_args(args)
    assert args.auto_increase_cpu is None
    assert args.manager_cpu_cores == 1


def test_auto_cpu_increase_rejects_predefined_pool():
    args = _parser().parse_args([
        'submit',
        'single',
        '--environment',
        'fixed',
        '--pool',
        'execution',
        '--config-only',
        '--auto-increase-cpu',
    ])
    with pytest.raises(ValueError, match='requires dynamic resources'):
        submit_module._validate_submission_args(args)


class _CompiledRun:
    def model_dump(self, *, mode: str):
        assert mode == 'json'
        return {'task_name': 'echo', 'task_config': {'value': 'hello'}}


@pytest.mark.parametrize(
    ('option', 'expected'),
    [(None, None), ('--auto-increase-cpu', True), ('--no-auto-increase-cpu', False)],
)
async def test_submit_preserves_auto_cpu_preference_in_request(option, expected, tmp_path, monkeypatch):
    captured = _stub_submit_io(monkeypatch)
    monkeypatch.setattr(submit_module, 'compile_single', lambda *args, **kwargs: _CompiledRun())
    argv = [
        'submit',
        'single',
        '--use',
        'base-config',
        '--environment',
        'dynamic',
        '--config-only',
        '--project-root',
        str(tmp_path),
        '--taihu-token',
        'test-token',
    ]
    if option:
        argv.append(option)
    await submit_module._submit(_parse(argv))
    manager = captured['requests'][0]['json']['resources']['managers']
    if expected is None:
        assert 'auto_increase_cpu' not in manager
    else:
        assert manager['auto_increase_cpu'] is expected


def _stub_submit_io(monkeypatch):
    captured: dict[str, list | object] = {'uploads': [], 'requests': [], 'waits': []}

    def fake_build_source_package(_root, output_path, *, excluded_paths):
        captured['source_excluded_paths'] = list(excluded_paths)
        output_path.write_bytes(b'source-package')
        return PackagedFile(
            path=output_path,
            sha256='a' * 64,
            size_bytes=output_path.stat().st_size,
            file_count=1,
            unpacked_size_bytes=1,
        )

    def fake_build_wheel(_project_root, output_dir):
        output_dir.mkdir(parents=True)
        path = output_dir / 'nexus-0-py3-none-any.whl'
        path.write_bytes(b'wheel')
        return PackagedFile(
            path=path,
            sha256='b' * 64,
            size_bytes=path.stat().st_size,
            file_count=1,
            unpacked_size_bytes=path.stat().st_size,
        )

    def fake_prepare_uv_lock(_project_root):
        path = _project_root / 'uv.lock'
        path.write_bytes(b'lock')
        return PackagedFile(
            path=path,
            sha256='c' * 64,
            size_bytes=path.stat().st_size,
            file_count=1,
            unpacked_size_bytes=path.stat().st_size,
        )

    async def fake_upload(
        session,
        control_url,
        path,
        *,
        kind,
        sha256,
        size_bytes,
        idempotency_key,
        environment_id,
        auth_headers,
    ):
        assert path.is_file()
        captured['uploads'].append(  # type: ignore[union-attr]
            {
                'authorization': auth_headers.get('Authorization'),
                'control_url': control_url,
                'kind': kind,
                'idempotency_key': idempotency_key,
                'environment_id': environment_id,
            }
        )
        return {
            'artifact_id': f'{kind}-artifact',
            'sha256': sha256,
            'size_bytes': size_bytes,
            'kind': kind,
        }

    async def fake_request_json(session, method, url, **kwargs):
        from nexus.utils.otel.propagation import inject_carrier

        captured['requests'].append(  # type: ignore[union-attr]
            {
                'authorization': (kwargs.get('headers') or {}).get('Authorization'),
                'session_authorization': session.headers.get('Authorization'),
                'method': method,
                'trace_carrier': inject_carrier(),
                'url': url,
                **kwargs,
            }
        )
        return {'execution_id': 'execution-1', 'status': 'queued'}

    async def fake_wait_for_execution(session, control_url, execution_id, *, auth_headers):
        captured['waits'].append(  # type: ignore[union-attr]
            (auth_headers.get('Authorization'), control_url, execution_id)
        )
        return {'execution_id': execution_id, 'status': 'succeeded'}

    monkeypatch.setattr(submit_module, 'build_source_package', fake_build_source_package)
    monkeypatch.setattr(submit_module, '_build_wheel', fake_build_wheel)
    monkeypatch.setattr(submit_module, '_prepare_uv_lock', fake_prepare_uv_lock)
    monkeypatch.setattr(submit_module, '_upload', fake_upload)
    monkeypatch.setattr(submit_module, '_request_json', fake_request_json)
    monkeypatch.setattr(submit_module, '_wait_for_execution', fake_wait_for_execution)
    return captured


def _parse(argv: list[str]) -> argparse.Namespace:
    args = _parser().parse_args(argv)
    args._raw_argv = argv
    return args


def test_submit_has_single_live_and_batch_commands():
    parser = _parser()

    for command in ('single', 'live', 'batch'):
        args = parser.parse_args([
            'submit',
            command,
            '--environment',
            'environment-1',
            '--config-only',
            '--pool',
            'predefined-pool',
            *(['--generator', 'tasks.py', '--max-inflight', '1'] if command == 'batch' else ['config.yaml']),
        ])
        assert args.submit_command == command


@pytest.mark.parametrize('command', ['single', 'live', 'batch'])
def test_submit_accepts_repeated_host_env(command: str):
    args = ['submit', command]
    if command == 'batch':
        args.extend(['--generator', 'tasks.py'])
    else:
        args.append('config.yaml')
    args.extend([
        '--environment',
        'environment-1',
        '--config-only',
        '--host-env',
        'FOO=bar',
        '--host-env',
        'HYTUNER=1',
    ])

    parsed = _parser().parse_args(args)

    assert parsed.host_env == ['FOO=bar', 'HYTUNER=1']


def test_submit_defaults_to_public_cloud_endpoints(monkeypatch):
    monkeypatch.delenv('NEXUS_CONTROL_URL', raising=False)
    monkeypatch.delenv('NEXUS_CONSOLE_URL', raising=False)

    args = _parser().parse_args([
        'submit',
        'single',
        'config.yaml',
        '--environment',
        'environment-1',
        '--config-only',
        '--pool',
        'predefined-pool',
    ])

    assert args.control_url == 'https://control-runner-nexus-agent.mcp.woa.com'
    assert args.console_url == 'https://cloud.nexus-agent.woa.com'


def test_cli_import_does_not_load_sqlalchemy() -> None:
    result = subprocess.run(
        [
            sys.executable,
            '-c',
            "import sys; import nexus.cli.main; raise SystemExit('sqlalchemy' in sys.modules)",
        ],
        check=False,
    )

    assert result.returncode == 0


@pytest.mark.parametrize('argv', [['--help'], ['run', '--help']])
def test_base_cli_help_does_not_require_cloud_dependencies(argv: list[str]) -> None:
    script = f"""
import builtins
import sys

real_import = builtins.__import__


def import_without_cloud_dependencies(name, globals=None, locals=None, fromlist=(), level=0):
    if name.partition('.')[0] in {{'pathspec', 'zstandard'}}:
        raise ModuleNotFoundError(f'blocked optional dependency: {{name}}')
    return real_import(name, globals, locals, fromlist, level)


builtins.__import__ = import_without_cloud_dependencies
sys.argv = ['nexus', *{argv!r}]

from nexus.cli.main import main

main()
"""
    result = subprocess.run([sys.executable, '-c', script], check=False, capture_output=True, text=True)

    assert result.returncode == 0, result.stderr
    assert 'usage: nexus' in result.stdout


@pytest.mark.parametrize(
    ('command', 'task_path'),
    [
        ('single', '/v1/console/executions/execution-1'),
        ('live', '/v1/console/executions/execution-1'),
        ('batch', '/v1/console/batches/execution-1'),
    ],
)
def test_submit_adds_web_ui_links_when_server_has_not_returned_them(command: str, task_path: str) -> None:
    result = submit_module._with_web_ui_urls(
        {
            'execution_id': 'execution-1',
            'status': 'queued',
            'control_url': 'http://internal-control/v1/executions/execution-1',
            **({'batch_url': 'http://internal-control/v1/executions/execution-1/batch'} if command == 'batch' else {}),
        },
        command=command,
        control_url='https://control.example/',
        console_url='https://console.example/',
    )

    assert result['control_url'] == 'https://control.example/v1/executions/execution-1'
    if command == 'batch':
        assert result['batch_url'] == 'https://control.example/v1/executions/execution-1/batch'
    assert result['task_web_ui'] == f'https://console.example{task_path}'
    assert result['environment_web_ui'] == 'https://console.example/v1/console/my-environments'
    assert result['tracedb_web_ui'] == 'https://console.example/v1/console/my-environments'


@pytest.mark.parametrize(
    'forbidden',
    [
        '--placement-label',
        '--slot-cost',
        '--max-attempts',
        '--submit-batch-size',
        '--orchestration-pool',
        '--requirement',
    ],
)
def test_internal_or_arbitrary_dependency_flags_are_not_public(forbidden: str):
    with pytest.raises(SystemExit):
        _parser().parse_args([
            'submit',
            'batch',
            '--environment',
            'environment-1',
            '--config-only',
            '--pool',
            'predefined-pool',
            '--generator',
            'tasks.py',
            '--max-inflight',
            '1',
            forbidden,
            'value',
        ])


def test_v1_batch_cli_accepts_missing_max_inflight():
    args = _parser().parse_args([
        'submit',
        'batch',
        '--environment',
        'environment-1',
        '--config-only',
        '--pool',
        'predefined-pool',
        '--generator',
        'tasks.py',
    ])

    assert args.max_inflight is None


def test_wait_is_available_for_single_and_batch_but_not_live():
    single = _parser().parse_args([
        'submit',
        'single',
        'config.yaml',
        '--environment',
        'env',
        '--pool',
        'pool',
        '--config-only',
        '--wait',
    ])
    batch = _parser().parse_args([
        'submit',
        'batch',
        '--generator',
        'tasks.py',
        '--environment',
        'env',
        '--pool',
        'pool',
        '--config-only',
        '--max-inflight',
        '1',
        '--wait',
    ])

    assert single.wait is True
    assert batch.wait is True
    with pytest.raises(SystemExit):
        _parser().parse_args([
            'submit',
            'live',
            'config.yaml',
            '--environment',
            'env',
            '--pool',
            'pool',
            '--config-only',
            '--wait',
        ])


def test_submit_data_help_only_advertises_dataset_service_uri(capsys):
    with pytest.raises(SystemExit):
        _parser().parse_args(['submit', 'batch', '--help'])

    help_text = capsys.readouterr().out
    assert 'Dataset Service URI' in help_text
    assert 'Dataset URI/path' not in help_text


def test_taihu_token_file_must_be_private(tmp_path):
    token_file = tmp_path / 'taihu-token'
    token_file.write_text('private-token\n')
    args = argparse.Namespace(taihu_token=None, taihu_token_file=token_file)

    os.chmod(token_file, 0o600)
    assert _read_taihu_token(args) == 'private-token'

    os.chmod(token_file, 0o644)
    with pytest.raises(ValueError, match='must not be readable or writable'):
        _read_taihu_token(args)


def test_artifact_upload_idempotency_key_is_bounded_stable_and_kind_scoped():
    submission_key = 'i' * 128
    source_key = submit_module._artifact_upload_idempotency_key(
        submission_key,
        kind='source_package',
        sha256='a' * 64,
        size_bytes=17,
    )

    assert len(source_key) <= 128
    assert source_key == submit_module._artifact_upload_idempotency_key(
        submission_key,
        kind='source_package',
        sha256='a' * 64,
        size_bytes=17,
    )
    assert source_key != submit_module._artifact_upload_idempotency_key(
        submission_key,
        kind='nexus_wheel',
        sha256='a' * 64,
        size_bytes=17,
    )
    assert source_key != submit_module._artifact_upload_idempotency_key(
        submission_key,
        kind='source_package',
        sha256='b' * 64,
        size_bytes=17,
    )


async def test_submit_rejects_oversized_environment_before_artifact_upload(tmp_path, monkeypatch):
    captured = _stub_submit_io(monkeypatch)
    monkeypatch.setattr(submit_module, 'compile_single', lambda *args, **kwargs: _CompiledRun())
    args = _parse([
        'submit',
        'single',
        '--use',
        'base-config',
        '--environment',
        'e' * 65,
        '--manager-cpu-cores',
        '1',
        '--manager-memory-gib',
        '1',
        '--project-root',
        str(tmp_path),
        '--idempotency-key',
        'request-1',
    ])

    with pytest.raises(ValueError, match='invalid_environment_id'):
        await submit_module._submit(args)

    assert captured['uploads'] == []
    assert captured['requests'] == []


async def test_submit_uses_bounded_artifact_keys_for_maximum_submission_key(tmp_path, monkeypatch):
    captured = _stub_submit_io(monkeypatch)
    monkeypatch.setattr(submit_module, 'compile_single', lambda *args, **kwargs: _CompiledRun())
    args = _parse([
        'submit',
        'single',
        '--use',
        'base-config',
        '--environment',
        'environment-dynamic',
        '--manager-cpu-cores',
        '1',
        '--manager-memory-gib',
        '1',
        '--project-root',
        str(tmp_path),
        '--idempotency-key',
        'i' * 128,
    ])

    await submit_module._submit(args)

    upload_keys = [upload['idempotency_key'] for upload in captured['uploads']]  # type: ignore[index]
    assert len(upload_keys) == 3
    assert len(set(upload_keys)) == 3
    assert all(len(key) <= 128 for key in upload_keys)


async def test_artifact_body_upload_forwards_bearer_with_upload_ticket(tmp_path):
    observed = {}
    content = b'artifact-body'

    async def receive_content(request: web.Request) -> web.Response:
        observed['application_authorization'] = request.headers.get('Authorization')
        observed['upload_ticket'] = request.headers.get('X-Nexus-Upload-Ticket')
        observed['content'] = await request.read()
        return web.json_response({
            'ref': {
                'artifact_id': 'artifact-1',
                'sha256': 'a' * 64,
                'size_bytes': len(content),
                'kind': 'source_package',
            }
        })

    application = web.Application()
    application.router.add_put('/upload', receive_content)
    application_server = TestServer(application)
    await application_server.start_server()

    async def create_upload(request: web.Request) -> web.Response:
        observed['control_authorization'] = request.headers.get('Authorization')
        observed['environment_id'] = (await request.json())['environment_id']
        return web.json_response(
            {
                'upload': {'upload_id': 'upload-1'},
                'upload_ticket': 'upload-ticket',
                'upload_url': str(application_server.make_url('/upload')),
            },
            status=201,
        )

    control = web.Application()
    control.router.add_post('/v1/artifact-uploads', create_upload)
    control_server = TestServer(control)
    await control_server.start_server()
    path = tmp_path / 'source.tar.zst'
    path.write_bytes(content)
    try:
        async with aiohttp.ClientSession() as session:
            ref = await submit_module._upload(
                session,
                str(control_server.make_url('/')).rstrip('/'),
                path,
                kind='source_package',
                sha256='a' * 64,
                size_bytes=len(content),
                idempotency_key='request-1:source',
                environment_id='environment-dynamic',
                auth_headers={'Authorization': 'Bearer taihu-personal-token'},
            )
    finally:
        await control_server.close()
        await application_server.close()

    assert ref['artifact_id'] == 'artifact-1'
    assert observed == {
        'control_authorization': 'Bearer taihu-personal-token',
        'environment_id': 'environment-dynamic',
        'application_authorization': 'Bearer taihu-personal-token',
        'upload_ticket': 'upload-ticket',
        'content': content,
    }


async def test_artifact_upload_retry_keeps_environment_scope(tmp_path, monkeypatch):
    observed: dict[str, object] = {'put_attempts': 0}
    content = b'artifact-body'

    async def receive_content(request: web.Request) -> web.Response:
        observed['put_attempts'] = int(observed['put_attempts']) + 1
        await request.read()
        if observed['put_attempts'] == 1:
            raise web.HTTPInternalServerError(text='transient database failure')
        return web.json_response({
            'ref': {
                'artifact_id': 'artifact-1',
                'sha256': 'a' * 64,
                'size_bytes': len(content),
                'kind': 'source_package',
            }
        })

    application = web.Application()
    application.router.add_put('/upload', receive_content)
    application_server = TestServer(application)
    await application_server.start_server()

    async def create_upload(request: web.Request) -> web.Response:
        return web.json_response(
            {
                'upload': {'upload_id': 'upload-1'},
                'upload_ticket': 'upload-ticket-1',
                'upload_url': str(application_server.make_url('/upload')),
            },
            status=201,
        )

    async def get_upload(request: web.Request) -> web.Response:
        observed['status_environment_id'] = request.query.get('environment_id')
        return web.json_response({'upload': {'upload_id': 'upload-1', 'status': 'pending'}})

    async def renew_ticket(request: web.Request) -> web.Response:
        observed['renew_environment_id'] = request.query.get('environment_id')
        return web.json_response({
            'upload_ticket': 'upload-ticket-2',
            'upload_url': str(application_server.make_url('/upload')),
        })

    control = web.Application()
    control.router.add_post('/v1/artifact-uploads', create_upload)
    control.router.add_get('/v1/artifact-uploads/{upload_id}', get_upload)
    control.router.add_post('/v1/artifact-uploads/{upload_id}/ticket', renew_ticket)
    control_server = TestServer(control)
    await control_server.start_server()
    path = tmp_path / 'source.tar.zst'
    path.write_bytes(content)

    async def no_sleep(_seconds: float) -> None:
        return None

    monkeypatch.setattr(submit_module.asyncio, 'sleep', no_sleep)
    try:
        async with aiohttp.ClientSession() as session:
            ref = await submit_module._upload(
                session,
                str(control_server.make_url('/')).rstrip('/'),
                path,
                kind='source_package',
                sha256='a' * 64,
                size_bytes=len(content),
                idempotency_key='request-1:source',
                environment_id='environment-dynamic',
            )
    finally:
        await control_server.close()
        await application_server.close()

    assert ref['artifact_id'] == 'artifact-1'
    assert observed == {
        'put_attempts': 2,
        'status_environment_id': 'environment-dynamic',
        'renew_environment_id': 'environment-dynamic',
    }


@pytest.mark.parametrize('mode', ['single', 'live'])
async def test_dynamic_single_and_live_build_typed_source_package_payload(tmp_path, monkeypatch, mode):
    captured = _stub_submit_io(monkeypatch)

    def fake_compile(parts, *, item_id, split):
        assert [(part.kind, part.value) for part in parts] == [('use', 'base-config')]
        assert item_id is None
        assert split is None
        return _CompiledRun()

    monkeypatch.setattr(submit_module, 'compile_single', fake_compile)
    argv = [
        'submit',
        mode,
        '--use',
        'base-config',
        '--environment',
        'environment-dynamic',
        '--manager-cpu-cores',
        '8',
        '--manager-memory-gib',
        '16',
        '--manager-slots',
        '7',
        '--project-root',
        str(tmp_path),
        '--control-url',
        'https://control.example',
        '--taihu-token',
        'taihu-personal-token',
        '--wheel-extra',
        'extensions',
        '--idempotency-key',
        'request-1',
        '--provisioning-timeout-seconds',
        '60',
        '--queue-ttl-seconds',
        '120',
        '--execution-ttl-seconds',
        '600',
        '--stop-timeout-seconds',
        '30',
        '--host-env',
        'HOST_ENV_MVP_PROBE=ok',
        '--host-env',
        'HYTUNER=1',
    ]
    if mode == 'single':
        argv.append('--wait')
    result = await submit_module._submit(_parse(argv))

    assert [upload['kind'] for upload in captured['uploads']] == [  # type: ignore[index]
        'source_package',
        'nexus_wheel',
        'uv_lock',
    ]
    assert {upload['environment_id'] for upload in captured['uploads']} == {'environment-dynamic'}  # type: ignore[index]
    request = captured['requests'][0]  # type: ignore[index]
    assert request['url'] == f'https://control.example/v1/submissions/{mode}'
    payload = request['json']
    parsed = (SingleSubmission if mode == 'single' else LiveSubmission).model_validate(payload)
    assert isinstance(parsed.code, SourcePackageCode)
    assert parsed.environment_id == 'environment-dynamic'
    assert parsed.resources.managers.model_dump() == {
        'count': 1,
        'cpu_cores': 8.0,
        'memory_gib': 16.0,
        'slots': 7,
    }
    assert parsed.code.extras == ['extensions']
    assert parsed.code.nexus_wheel.kind == 'nexus_wheel'
    assert parsed.code.uv_lock.kind == 'uv_lock'
    assert parsed.execution_ttl_seconds == 600
    assert parsed.model_dump(mode='json')['host_env'] == [
        {'name': 'HOST_ENV_MVP_PROBE', 'value': 'ok'},
        {'name': 'HYTUNER', 'value': '1'},
    ]
    assert not ({'placement', 'slot_cost', 'max_attempts', 'execution_environment'} & payload.keys())
    assert len(captured['waits']) == (1 if mode == 'single' else 0)  # type: ignore[arg-type]
    assert ('terminal' in result) is (mode == 'single')


@pytest.mark.parametrize(
    ('mode', 'submission_type'),
    [
        ('single', SingleSubmission),
        ('live', LiveSubmission),
    ],
)
async def test_dynamic_single_and_live_use_default_resources(
    tmp_path,
    monkeypatch,
    mode,
    submission_type,
):
    captured = _stub_submit_io(monkeypatch)
    monkeypatch.setattr(submit_module, 'compile_single', lambda *args, **kwargs: _CompiledRun())
    argv = [
        'submit',
        mode,
        '--use',
        'base-config',
        '--environment',
        'environment-dynamic',
        '--config-only',
        '--project-root',
        str(tmp_path),
        '--idempotency-key',
        f'{mode}-dynamic-defaults-1',
    ]

    await submit_module._submit(_parse(argv))

    payload = captured['requests'][0]['json']  # type: ignore[index]
    submission = submission_type.model_validate(payload)
    assert submission.resources.managers.model_dump() == {
        'count': 1,
        'cpu_cores': 1.0,
        'memory_gib': 2.0,
        'slots': 20,
    }


async def test_predefined_config_only_skips_source_and_wheel_upload(tmp_path, monkeypatch):
    captured = _stub_submit_io(monkeypatch)
    monkeypatch.setattr(submit_module, 'compile_single', lambda *args, **kwargs: _CompiledRun())
    argv = [
        'submit',
        'single',
        '--use',
        'installed-config',
        '--environment',
        'environment-predefined',
        '--pool',
        'execution-pool',
        '--config-only',
        '--project-root',
        str(tmp_path),
        '--idempotency-key',
        'predefined-1',
    ]

    await submit_module._submit(_parse(argv))

    assert captured['uploads'] == []
    payload = captured['requests'][0]['json']  # type: ignore[index]
    parsed = SingleSubmission.model_validate(payload)
    assert isinstance(parsed.code, ConfigOnlyCode)
    assert parsed.pool == 'execution-pool'
    assert parsed.resources is None


@pytest.mark.parametrize(
    ('mode', 'mode_args', 'submission_type'),
    [
        ('single', ['--use', 'installed-config'], SingleSubmission),
        ('live', ['--use', 'installed-config'], LiveSubmission),
        ('batch', ['--set', 'task_name=echo'], BatchSubmission),
    ],
)
async def test_submit_preserves_ordered_run_trace_sink_selection(
    tmp_path,
    monkeypatch,
    mode,
    mode_args,
    submission_type,
):
    captured = _stub_submit_io(monkeypatch)
    monkeypatch.setattr(submit_module, 'compile_single', lambda *args, **kwargs: _CompiledRun())
    argv = [
        'submit',
        mode,
        *mode_args,
        '--environment',
        'environment-predefined',
        '--pool',
        'execution-pool',
        '--config-only',
        '--project-root',
        str(tmp_path),
        '--idempotency-key',
        f'{mode}-trace-selection',
        '--run-trace-sink',
        'tdsql-team-a',
        '--run-trace-sink',
        'cos-archive',
        *(['--max-inflight', '1'] if mode == 'batch' else []),
    ]

    await submit_module._submit(_parse(argv))

    payload = captured['requests'][0]['json']  # type: ignore[index]
    submission = submission_type.model_validate(payload)
    assert submission.run_trace_backends is not None
    assert submission.run_trace_backends.sinks == ['tdsql-team-a', 'cos-archive']


async def test_submit_wait_terminal_omits_execution_spec_and_runtime_credentials(tmp_path, monkeypatch):
    _stub_submit_io(monkeypatch)
    monkeypatch.setattr(submit_module, 'compile_single', lambda *args, **kwargs: _CompiledRun())

    async def fake_wait_for_execution(*args, **kwargs):
        del args, kwargs
        return {
            'execution_id': 'execution-1',
            'status': 'succeeded',
            'phase': 'releasing',
            'allocation_id': 'allocation-1',
            'current_assignment_id': 'assignment-1',
            'current_attempt_id': 'attempt-1',
            'attempt_count': 1,
            'result_artifact_id': 'artifact-1',
            'cleanup_done_at': None,
            'created_at': '2026-08-10T00:00:00',
            'updated_at': '2026-08-10T00:01:00',
            'spec': {'run_input': {'openai_clients': {'main': {'api_key': 'sensitive-model-key'}}}},
            'current_assignment': {'runtime_info': {'credential': 'sensitive-runtime-credential'}},
        }

    monkeypatch.setattr(submit_module, '_wait_for_execution', fake_wait_for_execution)
    result = await submit_module._submit(
        _parse([
            'submit',
            'single',
            '--use',
            'installed-config',
            '--environment',
            'environment-predefined',
            '--pool',
            'execution-pool',
            '--config-only',
            '--project-root',
            str(tmp_path),
            '--idempotency-key',
            'predefined-wait-1',
            '--wait',
        ])
    )

    assert result['terminal'] == {
        'execution_id': 'execution-1',
        'status': 'succeeded',
        'phase': 'releasing',
        'allocation_id': 'allocation-1',
        'current_assignment_id': 'assignment-1',
        'current_attempt_id': 'attempt-1',
        'attempt_count': 1,
        'result_artifact_id': 'artifact-1',
        'cleanup_done_at': None,
        'created_at': '2026-08-10T00:00:00',
        'updated_at': '2026-08-10T00:01:00',
    }
    assert 'sensitive-model-key' not in json.dumps(result)
    assert 'sensitive-runtime-credential' not in json.dumps(result)


async def test_dynamic_image_single_skips_code_artifact_uploads(tmp_path, monkeypatch):
    captured = _stub_submit_io(monkeypatch)
    monkeypatch.setattr(submit_module, 'compile_single', lambda *args, **kwargs: _CompiledRun())
    argv = [
        'submit',
        'single',
        '--use',
        'installed-config',
        '--environment',
        'environment-dynamic',
        '--workload-image-version',
        'user-build-20260802-001',
        '--manager-cpu-cores',
        '1',
        '--manager-memory-gib',
        '2',
        '--project-root',
        str(tmp_path),
        '--idempotency-key',
        'image-single-1',
    ]

    await submit_module._submit(_parse(argv))

    assert captured['uploads'] == []
    payload = captured['requests'][0]['json']  # type: ignore[index]
    parsed = SingleSubmission.model_validate(payload)
    assert isinstance(parsed.code, ImageCode)
    assert parsed.code.image_version == 'user-build-20260802-001'


async def test_dynamic_image_remote_generator_does_not_require_a_local_generator_file(tmp_path, monkeypatch):
    captured = _stub_submit_io(monkeypatch)
    argv = [
        'submit',
        'batch',
        '--generator',
        'generators/tasks.py',
        '--generation',
        'remote',
        '--environment',
        'environment-dynamic',
        '--workload-image-version',
        'user-build-20260802-001',
        '--max-inflight',
        '20',
        '--project-root',
        str(tmp_path),
        '--idempotency-key',
        'image-batch-1',
    ]

    await submit_module._submit(_parse(argv))

    assert captured['uploads'] == []
    payload = captured['requests'][0]['json']  # type: ignore[index]
    parsed = BatchSubmission.model_validate(payload)
    assert isinstance(parsed.code, ImageCode)
    assert parsed.generation.source.path == 'generators/tasks.py'  # type: ignore[union-attr]


@pytest.mark.parametrize(
    'conflicting',
    [
        ['--config-only'],
        ['--source', '.'],
        ['--wheel-extra', 'extensions'],
    ],
)
async def test_workload_image_version_rejects_other_code_mode_flags(tmp_path, monkeypatch, conflicting):
    _stub_submit_io(monkeypatch)
    argv = [
        'submit',
        'single',
        '--use',
        'installed-config',
        '--environment',
        'environment-dynamic',
        '--workload-image-version',
        'user-build-1',
        '--manager-cpu-cores',
        '1',
        '--manager-memory-gib',
        '1',
        '--project-root',
        str(tmp_path),
        *conflicting,
    ]

    with pytest.raises(ValueError, match='workload-image-version'):
        await submit_module._submit(_parse(argv))


async def test_predefined_pool_rejects_workload_image_version(tmp_path, monkeypatch):
    _stub_submit_io(monkeypatch)
    argv = [
        'submit',
        'single',
        '--use',
        'installed-config',
        '--environment',
        'environment-predefined',
        '--pool',
        'execution-pool',
        '--workload-image-version',
        'user-build-1',
        '--project-root',
        str(tmp_path),
    ]

    with pytest.raises(ValueError, match='predefined.*config-only'):
        await submit_module._submit(_parse(argv))


def test_source_package_requires_a_checked_uv_lock(tmp_path, monkeypatch):
    with pytest.raises(ValueError, match='uv.lock'):
        submit_module._prepare_uv_lock(tmp_path)

    (tmp_path / 'uv.lock').write_text('version = 1\n')
    commands = []

    def fake_run(command, *, cwd, check):
        commands.append((command, cwd, check))

    monkeypatch.setattr(submit_module.subprocess, 'run', fake_run)
    packaged = submit_module._prepare_uv_lock(tmp_path)

    assert packaged.path == tmp_path / 'uv.lock'
    assert commands == [(['uv', 'lock', '--check'], tmp_path, True)]


def test_build_wheel_requires_the_nexus_agent_project(tmp_path, monkeypatch):
    (tmp_path / 'pyproject.toml').write_text('[project]\nname = "other-project"\nversion = "1.0.0"\n')
    monkeypatch.setattr(
        submit_module.subprocess,
        'run',
        lambda *_args, **_kwargs: pytest.fail('uv build must not run for a non-Nexus project'),
    )

    with pytest.raises(ValueError, match='nexus-agent'):
        submit_module._build_wheel(tmp_path, tmp_path / 'dist')


def test_build_wheel_defers_built_wheel_semantics_to_builder(tmp_path, monkeypatch):
    (tmp_path / 'pyproject.toml').write_text('[project]\nname = "nexus-agent"\nversion = "1.0.0"\n')

    def fake_run(_command, *, cwd, check):
        assert cwd == tmp_path
        assert check is True
        output = tmp_path / 'dist'
        output.mkdir(exist_ok=True)
        wheel = output / 'other_project-1.0.0-py3-none-any.whl'
        with zipfile.ZipFile(wheel, 'w') as archive:
            archive.writestr('other_project-1.0.0.dist-info/METADATA', 'Name: other-project\nVersion: 1.0.0\n')
            archive.writestr('other_project-1.0.0.dist-info/WHEEL', 'Wheel-Version: 1.0\nTag: py3-none-any\n')

    monkeypatch.setattr(submit_module.subprocess, 'run', fake_run)

    packaged = submit_module._build_wheel(tmp_path, tmp_path / 'dist')

    assert packaged.path.name == 'other_project-1.0.0-py3-none-any.whl'


@pytest.mark.parametrize(
    ('extra', 'message'),
    [
        ([], 'require --config-only'),
        (['--manager-cpu-cores', '2', '--manager-memory-gib', '4'], 'dynamic resource flags'),
        (['--source', '.'], 'cannot be combined'),
        (['--wheel-extra', 'extensions'], 'cannot be combined'),
    ],
)
async def test_predefined_requires_config_only_and_rejects_source_or_resources(
    tmp_path,
    monkeypatch,
    extra,
    message,
):
    _stub_submit_io(monkeypatch)
    argv = [
        'submit',
        'single',
        '--use',
        'installed-config',
        '--environment',
        'environment-predefined',
        '--pool',
        'execution-pool',
        '--project-root',
        str(tmp_path),
        *extra,
    ]
    if extra:
        argv.append('--config-only')

    with pytest.raises(ValueError, match=message):
        await submit_module._submit(_parse(argv))


async def test_dynamic_rejects_user_pool_even_when_resources_are_supplied(tmp_path, monkeypatch):
    _stub_submit_io(monkeypatch)
    argv = [
        'submit',
        'single',
        '--use',
        'base-config',
        '--environment',
        'environment-dynamic',
        '--pool',
        'user-selected-pool',
        '--config-only',
        '--manager-cpu-cores',
        '2',
        '--manager-memory-gib',
        '4',
        '--project-root',
        str(tmp_path),
    ]

    with pytest.raises(ValueError, match='pool'):
        await submit_module._submit(_parse(argv))


async def test_dynamic_batch_builds_resource_window_and_timeout_payload(tmp_path, monkeypatch):
    captured = _stub_submit_io(monkeypatch)
    generator = tmp_path / 'tasks.py'
    generator.write_text('def generate():\n    return []\n')
    argv = [
        'submit',
        'batch',
        '--generator',
        'tasks.py',
        '--generation',
        'remote',
        '--environment',
        'environment-dynamic',
        '--batch-manager-cpu-cores',
        '4',
        '--batch-manager-memory-gib',
        '8',
        '--manager-count',
        '3',
        '--manager-cpu-cores',
        '2',
        '--manager-memory-gib',
        '4',
        '--manager-slots',
        '5',
        '--max-inflight',
        '15',
        '--generation-timeout-seconds',
        '180',
        '--child-queue-ttl-seconds',
        '300',
        '--timeout-per-run',
        '20',
        '--project-root',
        str(tmp_path),
        '--idempotency-key',
        'batch-dynamic-1',
    ]

    await submit_module._submit(_parse(argv))

    payload = captured['requests'][0]['json']  # type: ignore[index]
    submission = BatchSubmission.model_validate(payload)
    assert submission.resources.managers.count == 3
    assert submission.resources.managers.slots == 5
    assert submission.batch.max_inflight == 15
    assert submission.generation.timeout_seconds == 180
    assert submission.batch.child_queue_ttl_seconds == 300
    assert submission.batch.timeout_per_run == 20
    assert 'submit_batch_size' not in payload['batch']


async def test_dynamic_batch_uses_default_resources(tmp_path, monkeypatch):
    captured = _stub_submit_io(monkeypatch)
    generator = tmp_path / 'tasks.py'
    generator.write_text('def generate():\n    return []\n')
    argv = [
        'submit',
        'batch',
        '--generator',
        'tasks.py',
        '--generation',
        'remote',
        '--environment',
        'environment-dynamic',
        '--max-inflight',
        '45',
        '--project-root',
        str(tmp_path),
        '--idempotency-key',
        'batch-dynamic-defaults-1',
    ]

    await submit_module._submit(_parse(argv))

    payload = captured['requests'][0]['json']  # type: ignore[index]
    submission = BatchSubmission.model_validate(payload)
    assert submission.resources is not None
    assert submission.resources.batch_manager.cpu_cores == 1
    assert submission.resources.batch_manager.memory_gib == 4
    assert submission.resources.managers.count == 3
    assert submission.resources.managers.cpu_cores == 1
    assert submission.resources.managers.memory_gib == 2
    assert submission.resources.managers.slots == 20
    assert submission.batch.max_inflight == 45


async def test_v1_dynamic_batch_cli_defaults_missing_max_inflight_to_capacity(tmp_path, monkeypatch):
    captured = _stub_submit_io(monkeypatch)
    generator = tmp_path / 'tasks.py'
    generator.write_text('def generate():\n    return []\n')
    argv = [
        'submit',
        'batch',
        '--generator',
        'tasks.py',
        '--generation',
        'remote',
        '--environment',
        'environment-dynamic',
        '--project-root',
        str(tmp_path),
        '--idempotency-key',
        'batch-dynamic-v1-defaults',
    ]

    await submit_module._submit(_parse(argv))

    payload = captured['requests'][0]['json']  # type: ignore[index]
    submission = BatchSubmission.model_validate(payload)
    assert submission.resources is not None
    assert submission.resources.managers.count == 1
    assert submission.resources.managers.slots == 20
    assert submission.batch.max_inflight == 20


@pytest.mark.parametrize(
    ('capacity_args', 'expected_count', 'expected_slots'),
    [
        (['--manager-count', '3'], 3, 15),
        (['--manager-slots', '16'], 3, 16),
    ],
)
async def test_dynamic_batch_infers_one_missing_manager_capacity_flag(
    tmp_path,
    monkeypatch,
    capacity_args,
    expected_count,
    expected_slots,
):
    captured = _stub_submit_io(monkeypatch)
    generator = tmp_path / 'tasks.py'
    generator.write_text('def generate():\n    return []\n')
    argv = [
        'submit',
        'batch',
        '--generator',
        'tasks.py',
        '--generation',
        'remote',
        '--environment',
        'environment-dynamic',
        '--max-inflight',
        '45',
        *capacity_args,
        '--project-root',
        str(tmp_path),
    ]

    await submit_module._submit(_parse(argv))

    payload = captured['requests'][0]['json']  # type: ignore[index]
    submission = BatchSubmission.model_validate(payload)
    assert submission.resources is not None
    assert submission.resources.managers.count == expected_count
    assert submission.resources.managers.slots == expected_slots


async def test_remote_batch_reads_wedata_credential_from_private_file(tmp_path, monkeypatch):
    captured = _stub_submit_io(monkeypatch)
    generator = tmp_path / 'tasks.py'
    generator.write_text('def generate():\n    return []\n')
    wedata_key = tmp_path / 'wedata-key.json'
    wedata_key.write_text('{"subject":"alice","key":"secret-key","id":"epoch-1"}', encoding='utf-8')
    wedata_key.chmod(0o600)
    taihu_token = tmp_path / 'taihu-token'
    taihu_token.write_text('taihu-personal-token', encoding='utf-8')
    taihu_token.chmod(0o600)
    argv = [
        'submit',
        'batch',
        '--generator',
        'tasks.py',
        '--generation',
        'remote',
        '--wedata-key-file',
        str(wedata_key),
        '--taihu-token-file',
        str(taihu_token),
        '--environment',
        'environment-dynamic',
        '--batch-manager-cpu-cores',
        '1',
        '--batch-manager-memory-gib',
        '1',
        '--manager-count',
        '1',
        '--manager-cpu-cores',
        '1',
        '--manager-memory-gib',
        '1',
        '--manager-slots',
        '1',
        '--max-inflight',
        '1',
        '--project-root',
        str(tmp_path),
    ]

    await submit_module._submit(_parse(argv))

    payload = captured['requests'][0]['json']  # type: ignore[index]
    assert payload['data_credentials'] == {'wedata': {'subject': 'alice', 'key': 'secret-key', 'id': 'epoch-1'}}
    parsed = BatchSubmission.model_validate(payload)
    assert parsed.data_credentials is not None
    assert parsed.data_credentials.wedata is not None
    assert parsed.data_credentials.wedata.key.get_secret_value() == 'secret-key'
    assert captured['source_excluded_paths'] == [taihu_token, wedata_key]


async def test_wedata_credential_file_must_be_private_and_remote(tmp_path, monkeypatch):
    _stub_submit_io(monkeypatch)
    generator = tmp_path / 'tasks.py'
    generator.write_text('def generate():\n    return []\n')
    wedata_key = tmp_path / 'wedata-key.json'
    wedata_key.write_text('{"subject":"alice","key":"secret-key","id":"epoch-1"}', encoding='utf-8')
    base = [
        'submit',
        'batch',
        '--generator',
        'tasks.py',
        '--wedata-key-file',
        str(wedata_key),
        '--environment',
        'environment-dynamic',
        '--batch-manager-cpu-cores',
        '1',
        '--batch-manager-memory-gib',
        '1',
        '--manager-count',
        '1',
        '--manager-cpu-cores',
        '1',
        '--manager-memory-gib',
        '1',
        '--manager-slots',
        '1',
        '--max-inflight',
        '1',
        '--project-root',
        str(tmp_path),
    ]

    wedata_key.chmod(0o644)
    with pytest.raises(ValueError, match='wedata-key-file.*group/other'):
        await submit_module._submit(_parse([*base, '--generation', 'remote']))

    wedata_key.chmod(0o600)
    with pytest.raises(ValueError, match='wedata-key-file.*remote'):
        await submit_module._submit(_parse([*base, '--generation', 'local']))


async def test_predefined_local_generator_uploads_only_task_list(tmp_path, monkeypatch):
    captured = _stub_submit_io(monkeypatch)
    generator = tmp_path / 'tasks.py'
    generator.write_text('def generate():\n    return []\n')
    monkeypatch.setenv('NEXUS_SUBMIT_MAX_TASK_LIST_BYTES', '123456')

    def fake_materialize(
        generator_path,
        output_path,
        *,
        max_items,
        max_bytes,
        max_line_bytes,
        task_list_version,
        compose_hooks,
    ):
        assert generator_path == generator
        assert max_items is None
        assert max_bytes == 123456
        assert max_line_bytes is None
        assert task_list_version == 2
        assert compose_hooks == []
        output_path.write_bytes(b'{"version":2,"item_id":"item-000000000001","run_input":{},"metadata":{}}\n')
        return MaterializedTaskList(
            path=output_path,
            sha256='c' * 64,
            size_bytes=output_path.stat().st_size,
            total_items=1,
            record_version=2,
            batch_metadata={'group': 'local-generator', 'tags': ['hyway']},
        )

    monkeypatch.setattr(submit_module, 'materialize_generator', fake_materialize)
    argv = [
        'submit',
        'batch',
        '--generator',
        'tasks.py',
        '--generation',
        'local',
        '--environment',
        'environment-predefined',
        '--pool',
        'execution-pool',
        '--config-only',
        '--max-inflight',
        '1',
        '--project-root',
        str(tmp_path),
        '--idempotency-key',
        'batch-local-1',
    ]

    await submit_module._submit(_parse(argv))

    assert [upload['kind'] for upload in captured['uploads']] == ['task_list']  # type: ignore[index]
    payload = captured['requests'][0]['json']  # type: ignore[index]
    submission = BatchSubmission.model_validate(payload)
    assert submission.pool == 'execution-pool'
    assert submission.resources is None
    assert submission.generation.mode == 'local_task_list'
    assert submission.generation.task_list.manifest.record_version == 2
    assert submission.generation.task_list.manifest.batch_metadata == {
        'group': 'local-generator',
        'tags': ['hyway'],
    }
    assert submission.batch.max_inflight == 1


async def test_local_compose_manifest_records_resolved_dataset_version(tmp_path, monkeypatch):
    captured = _stub_submit_io(monkeypatch)
    monkeypatch.setenv('NEXUS_SUBMIT_MAX_TASK_LIST_BYTES', '654321')

    async def fake_materialize_compose(parts, output_path, **kwargs):
        assert [(part.kind, part.value) for part in parts] == [
            ('data', 'research/tasks@latest'),
            ('set', 'task_name=echo'),
        ]
        assert kwargs['split'] == 'test'
        assert kwargs['max_items'] is None
        assert kwargs['max_bytes'] == 654321
        assert kwargs['max_line_bytes'] is None
        output_path.write_bytes(b'{"version":1,"item_id":"case-001","run_input":{}}\n')
        return MaterializedBatchArtifacts(
            task_list=MaterializedTaskList(
                path=output_path,
                sha256='d' * 64,
                size_bytes=output_path.stat().st_size,
                total_items=1,
                record_version=2,
                resolved_datasets=(
                    {
                        'input_uri': 'research/tasks@latest',
                        'resolved_uri': 'research/tasks@v17?split=test',
                    },
                ),
            ),
        )

    monkeypatch.setattr(submit_module, 'materialize_compose', fake_materialize_compose)
    argv = [
        'submit',
        'batch',
        '--data',
        'research/tasks@latest',
        '--set',
        'task_name=echo',
        '--split',
        'test',
        '--generation',
        'local',
        '--environment',
        'environment-predefined',
        '--pool',
        'execution-pool',
        '--config-only',
        '--max-inflight',
        '1',
        '--project-root',
        str(tmp_path),
        '--idempotency-key',
        'batch-local-compose-1',
    ]

    await submit_module._submit(_parse(argv))

    payload = captured['requests'][0]['json']  # type: ignore[index]
    submission = BatchSubmission.model_validate(payload)
    assert submission.generation.task_list.manifest.datasets[0].model_dump() == {
        'input_uri': 'research/tasks@latest',
        'resolved_uri': 'research/tasks@v17?split=test',
    }


async def test_local_compose_accepts_local_dataset_path(tmp_path, monkeypatch):
    captured = _stub_submit_io(monkeypatch)
    dataset_path = tmp_path / 'dataset'

    async def fake_materialize_compose(parts, output_path, **kwargs):
        assert [(part.kind, part.value) for part in parts] == [
            ('data', str(dataset_path)),
            ('set', 'task_name=echo'),
        ]
        output_path.write_bytes(b'{"version":1,"item_id":"case-001","run_input":{}}\n')
        return MaterializedBatchArtifacts(
            task_list=MaterializedTaskList(
                path=output_path,
                sha256='e' * 64,
                size_bytes=output_path.stat().st_size,
                total_items=1,
                record_version=2,
            )
        )

    monkeypatch.setattr(submit_module, 'materialize_compose', fake_materialize_compose)
    argv = [
        'submit',
        'batch',
        '--data',
        str(dataset_path),
        '--set',
        'task_name=echo',
        '--generation',
        'local',
        '--environment',
        'environment-predefined',
        '--pool',
        'execution-pool',
        '--config-only',
        '--max-inflight',
        '1',
        '--project-root',
        str(tmp_path),
        '--idempotency-key',
        'batch-local-dataset-1',
    ]

    await submit_module._submit(_parse(argv))

    payload = captured['requests'][0]['json']  # type: ignore[index]
    submission = BatchSubmission.model_validate(payload)
    assert submission.generation.mode == 'local_task_list'


async def test_local_compose_task_list_v1_disables_data_list_and_rejects_local_task_files(tmp_path, monkeypatch):
    captured = _stub_submit_io(monkeypatch)
    dataset_path = tmp_path / 'dataset'

    async def fake_materialize_compose(parts, output_path, **kwargs):
        assert [(part.kind, part.value) for part in parts] == [('data', str(dataset_path))]
        assert kwargs['task_list_version'] == 1
        assert kwargs['auto_data_list'] is False
        assert kwargs['reject_local_task_files'] is True
        output_path.write_bytes(b'{"version":1,"item_id":"case-001","run_input":{}}\n')
        return MaterializedBatchArtifacts(
            task_list=MaterializedTaskList(
                path=output_path,
                sha256='e' * 64,
                size_bytes=output_path.stat().st_size,
                total_items=1,
                record_version=1,
            )
        )

    monkeypatch.setattr(submit_module, 'materialize_compose', fake_materialize_compose)
    argv = [
        'submit',
        'batch',
        '--data',
        str(dataset_path),
        '--generation',
        'local',
        '--task-list-version',
        '1',
        '--environment',
        'environment-predefined',
        '--pool',
        'execution-pool',
        '--config-only',
        '--max-inflight',
        '1',
        '--project-root',
        str(tmp_path),
        '--idempotency-key',
        'batch-local-dataset-v1',
    ]

    await submit_module._submit(_parse(argv))

    assert [upload['kind'] for upload in captured['uploads']] == ['task_list']  # type: ignore[index]
    payload = captured['requests'][0]['json']  # type: ignore[index]
    submission = BatchSubmission.model_validate(payload)
    assert submission.generation.task_list.manifest.record_version == 1


async def test_local_compose_uploads_data_list_before_v3_task_list(tmp_path, monkeypatch):
    captured = _stub_submit_io(monkeypatch)
    dataset_path = tmp_path / 'dataset'
    monkeypatch.setenv('NEXUS_SUBMIT_MAX_DATA_FRAME_BYTES', '111111')
    monkeypatch.setenv('NEXUS_SUBMIT_MAX_TASK_RECORD_BYTES', '222222')
    monkeypatch.setenv('NEXUS_SUBMIT_MAX_DATA_LIST_BYTES', '333333')

    async def fake_materialize_compose(parts, output_path, **kwargs):
        assert [(part.kind, part.value) for part in parts] == [('data', str(dataset_path))]
        assert kwargs['auto_data_list'] is True
        assert kwargs['max_data_frame_bytes'] == 111111
        assert kwargs['max_task_record_bytes'] == 222222
        assert kwargs['max_data_list_bytes'] == 333333
        assert kwargs['data_list_output_path'] == output_path.with_name('data_list.bin')
        output_path.write_bytes(b'{"version":3,"item_id":"case-001","run_input":{},"metadata":{},"data_ref":null}\n')
        data_path = kwargs['data_list_output_path']
        data_path.write_bytes(b'nxdl')
        return MaterializedBatchArtifacts(
            task_list=MaterializedTaskList(
                path=output_path,
                sha256='e' * 64,
                size_bytes=output_path.stat().st_size,
                total_items=1,
                record_version=3,
            ),
            data_list=MaterializedDataList(
                path=data_path,
                sha256='f' * 64,
                size_bytes=4,
                frame_count=1,
            ),
        )

    monkeypatch.setattr(submit_module, 'materialize_compose', fake_materialize_compose)
    argv = [
        'submit',
        'batch',
        '--data',
        str(dataset_path),
        '--generation',
        'local',
        '--environment',
        'environment-predefined',
        '--pool',
        'execution-pool',
        '--config-only',
        '--max-inflight',
        '1',
        '--project-root',
        str(tmp_path),
        '--idempotency-key',
        'batch-local-data-list-1',
    ]

    await submit_module._submit(_parse(argv))

    assert [upload['kind'] for upload in captured['uploads']] == ['data_list', 'task_list']  # type: ignore[index]
    payload = captured['requests'][0]['json']  # type: ignore[index]
    submission = BatchSubmission.model_validate(payload)
    assert submission.generation.task_list.manifest.record_version == 3
    assert submission.generation.data_list is not None
    assert submission.generation.data_list.ref.kind == 'data_list'
    assert submission.generation.data_list.manifest.frame_count == 1


async def test_dynamic_batch_rejects_max_inflight_above_capacity(tmp_path, monkeypatch):
    _stub_submit_io(monkeypatch)
    generator = tmp_path / 'tasks.py'
    generator.write_text('def generate():\n    return []\n')
    argv = [
        'submit',
        'batch',
        '--generator',
        'tasks.py',
        '--generation',
        'remote',
        '--environment',
        'environment-dynamic',
        '--batch-manager-cpu-cores',
        '1',
        '--batch-manager-memory-gib',
        '1',
        '--manager-count',
        '2',
        '--manager-cpu-cores',
        '1',
        '--manager-memory-gib',
        '1',
        '--manager-slots',
        '3',
        '--max-inflight',
        '7',
        '--project-root',
        str(tmp_path),
    ]

    with pytest.raises(ValueError, match='max-inflight'):
        await submit_module._submit(_parse(argv))


async def test_remote_compose_accepts_dataset_service_uri_and_preserves_order(tmp_path, monkeypatch):
    captured = _stub_submit_io(monkeypatch)
    argv = [
        'submit',
        'batch',
        '--data',
        'research/tasks@v1',
        '--set',
        'runner.timeout=20',
        '--generation',
        'remote',
        '--environment',
        'environment-dynamic',
        '--config-only',
        '--batch-manager-cpu-cores',
        '1',
        '--batch-manager-memory-gib',
        '1',
        '--manager-count',
        '1',
        '--manager-cpu-cores',
        '1',
        '--manager-memory-gib',
        '1',
        '--manager-slots',
        '2',
        '--max-inflight',
        '2',
        '--project-root',
        str(tmp_path),
    ]

    await submit_module._submit(_parse(argv))

    assert captured['uploads'] == []
    payload = captured['requests'][0]['json']  # type: ignore[index]
    assert [(part['kind'], part['value']) for part in payload['generation']['source']['compose']['parts']] == [
        ('data', 'research/tasks@v1'),
        ('set', 'runner.timeout=20'),
    ]
    assert payload['generation']['source']['task_list_version'] == 2


async def test_remote_compose_can_request_task_list_v2(tmp_path, monkeypatch):
    captured = _stub_submit_io(monkeypatch)
    argv = [
        'submit',
        'batch',
        '--data',
        'research/tasks@v1',
        '--generation',
        'remote',
        '--task-list-version',
        '2',
        '--environment',
        'environment-dynamic',
        '--config-only',
        '--batch-manager-cpu-cores',
        '1',
        '--batch-manager-memory-gib',
        '1',
        '--manager-count',
        '1',
        '--manager-cpu-cores',
        '1',
        '--manager-memory-gib',
        '1',
        '--manager-slots',
        '1',
        '--max-inflight',
        '1',
        '--project-root',
        str(tmp_path),
    ]

    await submit_module._submit(_parse(argv))

    payload = captured['requests'][0]['json']  # type: ignore[index]
    assert payload['generation']['source']['task_list_version'] == 2


@pytest.mark.parametrize('path', ['./tasks.jsonl', '/tmp/tasks.jsonl', 'file:///tmp/tasks.jsonl'])
async def test_cloud_batch_rejects_local_dataset_paths_before_upload(tmp_path, monkeypatch, path):
    captured = _stub_submit_io(monkeypatch)
    argv = [
        'submit',
        'batch',
        '--data',
        path,
        '--generation',
        'remote',
        '--environment',
        'environment-dynamic',
        '--config-only',
        '--batch-manager-cpu-cores',
        '1',
        '--batch-manager-memory-gib',
        '1',
        '--manager-count',
        '1',
        '--manager-cpu-cores',
        '1',
        '--manager-memory-gib',
        '1',
        '--manager-slots',
        '1',
        '--max-inflight',
        '1',
        '--project-root',
        str(tmp_path),
    ]

    with pytest.raises(ValueError, match='Dataset Service URI'):
        await submit_module._submit(_parse(argv))
    assert captured['uploads'] == []


async def test_remote_generator_rejects_config_only(tmp_path, monkeypatch):
    _stub_submit_io(monkeypatch)
    generator = tmp_path / 'tasks.py'
    generator.write_text('def generate():\n    return []\n')
    argv = [
        'submit',
        'batch',
        '--generator',
        'tasks.py',
        '--generation',
        'remote',
        '--environment',
        'environment-dynamic',
        '--config-only',
        '--batch-manager-cpu-cores',
        '1',
        '--batch-manager-memory-gib',
        '1',
        '--manager-count',
        '1',
        '--manager-cpu-cores',
        '1',
        '--manager-memory-gib',
        '1',
        '--manager-slots',
        '1',
        '--max-inflight',
        '1',
        '--project-root',
        str(tmp_path),
    ]

    with pytest.raises(ValueError, match='remote generator'):
        await submit_module._submit(_parse(argv))


_PREDEFINED_BATCH = ['--environment', 'environment-predefined', '--pool', 'execution-pool', '--config-only']
_IMAGE = ['--environment', 'environment-dynamic', '--workload-image-version', 'user-build-20260802-001']


async def test_local_generator_receives_compose_hooks(tmp_path, monkeypatch):
    captured = _stub_submit_io(monkeypatch)
    (tmp_path / 'tasks.py').write_text('def generate_configs():\n    return []\n')
    hook = tmp_path / 'check.py'
    hook.write_text('def create_compose_hook(params):\n    return lambda item: item\n')
    received = {}

    def fake_materialize(generator_path, output_path, **kwargs):
        received.update(kwargs)
        output_path.write_bytes(b'{"version":2,"item_id":"item-000000000001","run_input":{},"metadata":{}}\n')
        return MaterializedTaskList(
            path=output_path, sha256='c' * 64, size_bytes=output_path.stat().st_size, total_items=1, record_version=2
        )

    monkeypatch.setattr(submit_module, 'materialize_generator', fake_materialize)
    argv = [
        'submit',
        'batch',
        '--generator',
        'tasks.py',
        '--hook',
        f'{hook}:mode=fill',
        *_PREDEFINED_BATCH,
        '--project-root',
        str(tmp_path),
        '--idempotency-key',
        'batch-hook-1',
    ]

    await submit_module._submit(_parse(argv))

    assert [(hook_spec.name, hook_spec.params) for hook_spec in received['compose_hooks']] == [
        (str(hook), {'mode': 'fill'})
    ]
    assert [upload['kind'] for upload in captured['uploads']] == ['task_list']  # type: ignore[index]


@pytest.mark.parametrize(
    'source',
    [
        ['--data', 'research/tasks@latest', '--generation', 'remote'],
        ['--generator', 'tasks.py', '--generation', 'remote', *_IMAGE],
    ],
)
async def test_hooks_require_local_generation(tmp_path, monkeypatch, source):
    _stub_submit_io(monkeypatch)
    (tmp_path / 'tasks.py').write_text('def generate_configs():\n    return []\n')
    mode = [] if '--workload-image-version' in source else _PREDEFINED_BATCH
    argv = ['submit', 'batch', *source, '--hook', 'check.py', *mode, '--project-root', str(tmp_path)]
    with pytest.raises(ValueError, match='--hook requires --generation local'):
        await submit_module._submit(_parse(argv))


@pytest.mark.parametrize('command', ['single', 'live'])
def test_single_and_live_submissions_do_not_accept_hooks(command):
    with pytest.raises(SystemExit):
        _parser().parse_args(['submit', command, '--set', 'task_name=echo', '--hook', 'check.py'])


@pytest.mark.parametrize('generation', [['--generation', 'local', *_PREDEFINED_BATCH], ['--generation', 'remote', *_IMAGE]])
async def test_generator_still_rejects_static_overlays(tmp_path, monkeypatch, generation):
    _stub_submit_io(monkeypatch)
    (tmp_path / 'tasks.py').write_text('def generate_configs():\n    return []\n')
    argv = [
        'submit',
        'batch',
        '--generator',
        'tasks.py',
        *generation,
        '--set',
        'runner.timeout=30',
        '--project-root',
        str(tmp_path),
    ]
    with pytest.raises(ValueError, match='exactly one of --generator or ordered compose arguments is required'):
        await submit_module._submit(_parse(argv))


async def test_submission_http_retry_reuses_the_same_idempotency_key(monkeypatch):
    attempts: list[dict] = []

    async def fake_request_json(session, method, url, **kwargs):
        attempts.append(deepcopy(kwargs['json']))
        if len(attempts) == 1:
            raise aiohttp.ClientConnectionError('connection reset')
        return {'execution_id': 'execution-1', 'status': 'queued'}

    async def no_sleep(_seconds):
        return None

    monkeypatch.setattr(submit_module, '_request_json', fake_request_json)
    monkeypatch.setattr(submit_module.asyncio, 'sleep', no_sleep)
    payload = {'idempotency_key': 'stable-request-key'}

    result = await submit_module._post_submission_with_retry(
        object(),
        'https://control.example/v1/submissions/single',
        payload,
        auth_headers={},
    )

    assert result['execution_id'] == 'execution-1'
    assert [item['idempotency_key'] for item in attempts] == ['stable-request-key', 'stable-request-key']


async def test_wait_for_execution_retries_network_and_gateway_failures_with_bounded_backoff(monkeypatch):
    outcomes = [
        aiohttp.ClientConnectionError('connection reset'),
        submit_module._HttpResponseError('GET', 'https://control.example/execution-1', 503, {}),
        {'execution_id': 'execution-1', 'status': 'running'},
        {'execution_id': 'execution-1', 'status': 'succeeded'},
    ]
    sleeps: list[float] = []

    async def fake_request_json(*args, **kwargs):
        del args, kwargs
        outcome = outcomes.pop(0)
        if isinstance(outcome, Exception):
            raise outcome
        return outcome

    async def fake_sleep(seconds):
        sleeps.append(seconds)

    monkeypatch.setattr(submit_module, '_request_json', fake_request_json)
    monkeypatch.setattr(submit_module.asyncio, 'sleep', fake_sleep)

    result = await submit_module._wait_for_execution(
        object(),
        'https://control.example',
        'execution-1',
    )

    assert result['status'] == 'succeeded'
    assert sleeps == [1, 2, 1]


def test_submit_wait_failure_preserves_accepted_json_result(tmp_path, monkeypatch, capsys):
    _stub_submit_io(monkeypatch)
    monkeypatch.setattr(submit_module, 'compile_single', lambda *args, **kwargs: _CompiledRun())

    async def fail_wait(*args, **kwargs):
        del args, kwargs
        raise aiohttp.ClientConnectionError('control unavailable')

    monkeypatch.setattr(submit_module, '_wait_for_execution', fail_wait)
    args = _parse([
        'submit',
        'single',
        '--use',
        'installed-config',
        '--environment',
        'environment-predefined',
        '--pool',
        'execution',
        '--config-only',
        '--project-root',
        str(tmp_path),
        '--wait',
        '--json',
    ])

    assert submit_module.submit_command(args) == 1
    captured = capsys.readouterr()
    result = json.loads(captured.out)
    assert result['execution_id'] == 'execution-1'
    assert result['status'] == 'queued'
    assert result['wait_error']['type'] == 'ClientConnectionError'
    accepted = json.loads(captured.err.splitlines()[0])
    assert accepted == {
        'event': 'accepted',
        'execution_id': 'execution-1',
        'status': 'queued',
    }


def test_submit_wait_interrupt_preserves_id_without_stopping_remote(monkeypatch, capsys):
    calls: list[str] = []

    async def interrupted_submit(args, *, on_accepted=None):
        del args
        assert on_accepted is not None
        on_accepted({'execution_id': 'execution-interrupted', 'batch_id': 'batch-interrupted', 'status': 'queued'})
        calls.append('accepted')
        raise KeyboardInterrupt

    monkeypatch.setattr(submit_module, '_submit', interrupted_submit)
    args = argparse.Namespace(submit_command='batch', wait=True, json=True)

    assert submit_module.submit_command(args) == 130
    captured = capsys.readouterr()
    result = json.loads(captured.out)
    assert result == {
        'batch_id': 'batch-interrupted',
        'execution_id': 'execution-interrupted',
        'status': 'queued',
        'wait_interrupted': True,
    }
    assert calls == ['accepted']
    assert 'remote execution was not stopped' in captured.err


@pytest.mark.parametrize('enabled', [True, False])
@pytest.mark.parametrize('extras', [[], ['cloud'], ['otel', 'cloud', 'otel']])
async def test_submit_adds_otel_dependencies_without_changing_trace_context(
    tmp_path, monkeypatch, otel_hyway_instrumentor, enabled, extras
):
    from nexus.utils.otel.propagation import use_remote_context

    captured = _stub_submit_io(monkeypatch)
    monkeypatch.setenv('NEXUS_OTEL_ENABLED', 'true' if enabled else 'false')
    monkeypatch.setattr(submit_module, 'compile_single', lambda *args, **kwargs: _CompiledRun())
    args = _parse([
        'submit',
        'single',
        '--use',
        'base-config',
        '--environment',
        'environment-dynamic',
        '--manager-cpu-cores',
        '1',
        '--manager-memory-gib',
        '1',
        '--project-root',
        str(tmp_path),
        '--idempotency-key',
        'client-request',
    ])
    carrier = {
        'traceparent': '00-11111111111111111111111111111111-2222222222222222-01',
        'tracestate': 'vendor=preserved',
    }
    args.wheel_extra = extras.copy()
    with use_remote_context(carrier, fallback_only=False):
        await submit_module._submit(args)
    request = captured['requests'][0]
    assert request['trace_carrier'] == (carrier if enabled else {})
    assert request['json']['idempotency_key'] == 'client-request'
    assert request['json']['code']['extras'] == sorted(set(extras) | ({'otel'} if enabled else set()))
    assert args.wheel_extra == extras
