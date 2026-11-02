"""Parsing and help do not load any Redis service dependencies."""

import argparse
import importlib
import os
import subprocess
import sys

from importlib.resources import files
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from nexus.cli.commands.redis import handle_redis_command, register_redis_parser


def parse(*argv):
    parser = argparse.ArgumentParser()
    register_redis_parser(parser.add_subparsers(dest='command'))
    return parser.parse_args(['redis', *argv])


@pytest.mark.parametrize('action', ['serve', 'up', 'down', 'status', 'smoke', 'recover'])
def test_cluster_dispatch_preserves_result(monkeypatch, action):
    run = Mock(return_value=7)
    monkeypatch.setitem(
        sys.modules, 'nexus.gateway.trajectory_store_backends.redis_store.cluster', SimpleNamespace(run=run)
    )
    assert handle_redis_command(parse('cluster', action)) == 7
    run.assert_called_once_with(action)


@pytest.mark.parametrize('action', ['up', 'serve', 'down'])
def test_unified_dispatch_preserves_result_and_leaves_defaults_to_backend(monkeypatch, action):
    run = Mock(return_value=7)
    monkeypatch.setitem(
        sys.modules, 'nexus.gateway.trajectory_store_backends.redis_store.launcher', SimpleNamespace(run=run)
    )
    assert handle_redis_command(parse(action)) == 7
    run.assert_called_once_with(action)


@pytest.mark.parametrize(
    'command',
    [
        ['up'],
        ['serve'],
        ['down'],
        *[['cluster', action] for action in ('up', 'serve', 'down', 'status', 'smoke', 'recover')],
    ],
)
def test_redis_arguments_are_shared_and_preserve_zero(monkeypatch, command):
    run = Mock(return_value=0)
    module = 'cluster' if command[0] == 'cluster' else 'launcher'
    monkeypatch.setitem(
        sys.modules, f'nexus.gateway.trajectory_store_backends.redis_store.{module}', SimpleNamespace(run=run)
    )
    assert (
        handle_redis_command(
            parse(
                *command,
                '--primary-count',
                '6',
                '--replicas-per-primary',
                '0',
                '--extra-replicas',
                '0',
                '--redis-node-ips',
                '192.0.2.1,192.0.2.2',
                '--redis-client-ip',
                '192.0.2.2',
                '--redis-dir',
                '/mnt/redis',
            )
        )
        == 0
    )
    expected = {
        'primary_count': 6,
        'replicas_per_primary': 0,
        'extra_replicas': 0,
        'redis_node_ips': '192.0.2.1,192.0.2.2',
        'redis_client_ip': '192.0.2.2',
        'redis_dir': '/mnt/redis',
    }
    run.assert_called_once_with(command[-1], **expected)


@pytest.mark.parametrize('command', [['serve'], ['cluster', 'serve']])
def test_serve_passes_endpoint_file(monkeypatch, command):
    run = Mock(return_value=0)
    module = 'cluster' if command[0] == 'cluster' else 'launcher'
    monkeypatch.setitem(
        sys.modules, f'nexus.gateway.trajectory_store_backends.redis_store.{module}', SimpleNamespace(run=run)
    )
    assert handle_redis_command(parse(*command, '--redis-endpoint-file', '/mnt/output/redis.json')) == 0
    assert run.call_args.kwargs['redis_endpoint_file'] == '/mnt/output/redis.json'


@pytest.mark.parametrize(
    'command',
    [['up'], ['down'], *[['cluster', action] for action in ('prepare', 'up', 'down', 'status', 'smoke', 'recover')]],
)
def test_endpoint_file_only_applies_to_serve(command):
    with pytest.raises(SystemExit) as error:
        parse(*command, '--redis-endpoint-file', '/mnt/output/redis.json')
    assert error.value.code == 2


@pytest.mark.parametrize('action', ['up', 'serve'])
def test_unified_start_passes_data_unit_options(monkeypatch, action):
    run = Mock(return_value=0)
    monkeypatch.setitem(
        sys.modules, 'nexus.gateway.trajectory_store_backends.redis_store.launcher', SimpleNamespace(run=run)
    )
    args = parse(
        action,
        '--num-data-units',
        '4',
        '--controller-port',
        '17100',
        '--startup-timeout',
        '120',
        '--ssd-offload-dir',
        '/mnt/ssd',
        '--memory-waterline-bytes',
        '1G',
        '--memory-safe-line-bytes',
        '0',
    )
    assert handle_redis_command(args) == 0
    run.assert_called_once_with(
        action,
        num_data_units=4,
        controller_port=17100,
        startup_timeout=120,
        ssd_offload_dir='/mnt/ssd',
        memory_waterline_bytes=1024**3,
        memory_safe_line_bytes=0,
    )


def test_unified_down_rejects_data_unit_start_options():
    with pytest.raises(SystemExit) as error:
        parse('down', '--num-data-units', '4')
    assert error.value.code == 2


@pytest.mark.parametrize('exit_code', [0, 7])
def test_cluster_prepare_runs_packaged_script_and_preserves_exit_code(monkeypatch, exit_code):
    run = Mock(return_value=SimpleNamespace(returncode=exit_code))
    monkeypatch.setattr(subprocess, 'run', run)

    assert handle_redis_command(parse('cluster', 'prepare')) == exit_code

    script = files('nexus.gateway.trajectory_store_backends.redis_store').joinpath('redis_server_launcher_prepare.sh')
    assert script.is_file()
    run.assert_called_once_with(['bash', str(script)], check=False)


def test_fs_serve_passes_exact_config(monkeypatch, tmp_path):
    path = tmp_path / 'fs.json'
    path.write_text(
        '{"redis_url":"redis://test/0","num_data_units":2,"startup_timeout":60,'
        '"ssd_offload_dir":".hykvstore","memory_waterline_bytes":1024,"memory_safe_line_bytes":512}'
    )
    run = Mock(return_value=0)
    monkeypatch.setitem(
        sys.modules,
        'nexus.gateway.trajectory_store_backends.redis_store.file_storage.deploy',
        SimpleNamespace(serve_ray_cluster=run),
    )
    assert handle_redis_command(parse('fs', 'serve', '--config', str(path))) == 0
    run.assert_called_once_with(
        {
            'redis_url': 'redis://test/0',
            'num_data_units': 2,
            'startup_timeout': 60,
            'ssd_offload_dir': '.hykvstore',
            'memory_waterline_bytes': 1024,
            'memory_safe_line_bytes': 512,
        }
    )


@pytest.mark.parametrize('action', ['up', 'down'])
def test_fs_managed_commands_need_no_config(monkeypatch, action):
    run = Mock(return_value=0)
    monkeypatch.setitem(
        sys.modules,
        'nexus.gateway.trajectory_store_backends.redis_store.file_storage.launcher',
        SimpleNamespace(run=run),
    )
    assert handle_redis_command(parse('fs', action)) == 0
    run.assert_called_once_with(action)


def test_fs_up_accepts_connection_and_size_overrides(monkeypatch):
    run = Mock(return_value=0)
    monkeypatch.setitem(
        sys.modules,
        'nexus.gateway.trajectory_store_backends.redis_store.file_storage.launcher',
        SimpleNamespace(run=run),
    )
    assert (
        handle_redis_command(
            parse(
                'fs',
                'up',
                '--redis-url',
                'redis://existing:16379/0',
                '--num-data-units',
                '4',
                '--controller-port',
                '17100',
                '--startup-timeout',
                '120',
            )
        )
        == 0
    )
    run.assert_called_once_with(
        'up',
        redis_url='redis://existing:16379/0',
        num_data_units=4,
        controller_port=17100,
        startup_timeout=120,
    )


@pytest.mark.parametrize('command', [['up'], ['serve'], ['fs', 'up']])
def test_start_dispatch_preserves_explicit_zero_port_and_timeout(monkeypatch, command):
    run = Mock(return_value=0)
    module = 'file_storage.launcher' if command[0] == 'fs' else 'launcher'
    monkeypatch.setitem(
        sys.modules, f'nexus.gateway.trajectory_store_backends.redis_store.{module}', SimpleNamespace(run=run)
    )
    assert handle_redis_command(parse(*command, '--controller-port', '0', '--startup-timeout', '0')) == 0
    run.assert_called_once_with(command[-1], controller_port=0, startup_timeout=0.0)


@pytest.mark.parametrize('command', [['down'], ['fs', 'down']])
def test_down_dispatch_preserves_explicit_zero_port(monkeypatch, command):
    run = Mock(return_value=0)
    module = 'file_storage.launcher' if command[0] == 'fs' else 'launcher'
    monkeypatch.setitem(
        sys.modules, f'nexus.gateway.trajectory_store_backends.redis_store.{module}', SimpleNamespace(run=run)
    )
    assert handle_redis_command(parse(*command, '--controller-port', '0')) == 0
    run.assert_called_once_with(command[-1], controller_port=0)


@pytest.mark.parametrize('role', ['up', 'data-unit'])
@pytest.mark.parametrize('worker_option', ['request_workers', 'ssd_workers', 'copy_workers'])
def test_fs_dispatch_passes_workers_without_changing_defaults(monkeypatch, role, worker_option):
    run = Mock(return_value=0)
    module = 'launcher' if role == 'up' else 'deploy'
    entrypoint = 'run' if role == 'up' else 'serve_service'
    monkeypatch.setitem(
        sys.modules,
        f'nexus.gateway.trajectory_store_backends.redis_store.file_storage.{module}',
        SimpleNamespace(**{entrypoint: run}),
    )
    options = [] if role == 'up' else ['--topology', 'topology.json', '--unit-id', 'unit-1']
    flag = '--' + worker_option.replace('_', '-')
    assert handle_redis_command(parse('fs', role, *options, flag, '12')) == 0
    assert run.call_args.kwargs[worker_option] == 12
    run.reset_mock()
    assert handle_redis_command(parse('fs', role, *options)) == 0
    assert worker_option not in run.call_args.kwargs


@pytest.mark.parametrize('role', ['up', 'data-unit'])
@pytest.mark.parametrize(
    ('waterline', 'safe_line', 'expected_waterline', 'expected_safe_line'),
    [
        ('1024', '512', 1024, 512),
        ('200G', '100g', 214748364800, 107374182400),
        ('1g', '0', 1073741824, 0),
        ('1g', None, 1073741824, None),
    ],
    ids=['integer-bytes', 'binary-suffixes', 'zero-safe-line', 'waterline-only'],
)
def test_fs_dispatch_passes_ssd_offload_options(
    monkeypatch, role, waterline, safe_line, expected_waterline, expected_safe_line
):
    run = Mock(return_value=0)
    module = 'launcher' if role == 'up' else 'deploy'
    entrypoint = 'run' if role == 'up' else 'serve_service'
    monkeypatch.setitem(
        sys.modules,
        f'nexus.gateway.trajectory_store_backends.redis_store.file_storage.{module}',
        SimpleNamespace(**{entrypoint: run}),
    )
    options = [] if role == 'up' else ['--topology', 'topology.json', '--unit-id', 'unit-1']
    storage_args = ['--memory-waterline-bytes', waterline]
    expected = {'memory_waterline_bytes': expected_waterline}
    if safe_line is not None:
        storage_args.extend(['--ssd-offload-dir', '/mnt/ssd/nexus', '--memory-safe-line-bytes', safe_line])
        expected.update(ssd_offload_dir='/mnt/ssd/nexus', memory_safe_line_bytes=expected_safe_line)
    args = parse('fs', role, *options, *storage_args)
    assert handle_redis_command(args) == 0
    if role == 'up':
        run.assert_called_once_with(role, **expected)
    else:
        run.assert_called_once_with(role, 'topology.json', unit_id='unit-1', **expected)
    actual = run.call_args.kwargs
    for name in ('memory_waterline_bytes', 'memory_safe_line_bytes'):
        if name in actual:
            assert type(actual[name]) is int


@pytest.mark.parametrize(
    ('value', 'expected'),
    [('2K', 2 * 1024), ('3m', 3 * 1024**2), ('4G', 4 * 1024**3), ('5t', 5 * 1024**4), ('123', 123)],
)
def test_fs_memory_limit_suffixes_use_binary_units(value, expected):
    args = parse('fs', 'up', '--memory-waterline-bytes', value)
    assert args.memory_waterline_bytes == expected
    assert 'memory_safe_line_bytes' not in vars(args)
    assert 'ssd_offload_dir' not in vars(args)


@pytest.mark.parametrize(
    ('option', 'value'),
    [
        ('--memory-waterline-bytes', '-1'),
        ('--memory-safe-line-bytes', '-1G'),
        ('--memory-waterline-bytes', '200P'),
        ('--memory-safe-line-bytes', 'G'),
        ('--memory-waterline-bytes', '1.5G'),
        ('--memory-safe-line-bytes', '1.5'),
        ('--memory-waterline-bytes', ''),
        ('--memory-safe-line-bytes', '1B'),
        ('--memory-waterline-bytes', '1KB'),
        ('--memory-safe-line-bytes', '1KiB'),
    ],
)
def test_fs_memory_limits_reject_invalid_cli_input(option, value, capsys):
    with pytest.raises(SystemExit) as error:
        parse('fs', 'up', f'{option}={value}')
    assert error.value.code == 2
    assert f'argument {option}:' in capsys.readouterr().err


@pytest.mark.parametrize('role', ['down', 'controller'])
def test_fs_ssd_offload_options_only_apply_to_data_services(role):
    options = [] if role == 'down' else ['--topology', 'topology.json']
    with pytest.raises(SystemExit) as error:
        parse('fs', role, *options, '--memory-waterline-bytes', '1024')
    assert error.value.code == 2


@pytest.mark.parametrize('role', ['controller', 'data-unit'])
def test_service_dispatch_preserves_source_options(monkeypatch, role):
    run = Mock(return_value=0)
    monkeypatch.setitem(
        sys.modules,
        'nexus.gateway.trajectory_store_backends.redis_store.file_storage.deploy',
        SimpleNamespace(serve_service=run),
    )
    args = parse(
        'fs',
        role,
        '--topology',
        'topology.json',
        '--redis-url',
        'redis://host/0',
        '--controller-host',
        'host',
        '--controller-port',
        '17100',
        '--unit-id',
        'unit-1',
    )
    assert handle_redis_command(args) == 0
    run.assert_called_once_with(
        role,
        'topology.json',
        redis_url='redis://host/0',
        controller_host='host',
        controller_port=17100,
        unit_id='unit-1',
    )


@pytest.mark.parametrize('role', ['controller', 'data-unit'])
@pytest.mark.parametrize('controller_port', [None, 0])
def test_service_dispatch_passes_only_supplied_options(monkeypatch, role, controller_port):
    run = Mock(return_value=0)
    monkeypatch.setitem(
        sys.modules,
        'nexus.gateway.trajectory_store_backends.redis_store.file_storage.deploy',
        SimpleNamespace(serve_service=run),
    )
    options = ['--topology', 'topology.json']
    expected = {}
    if role == 'data-unit':
        options.extend(['--unit-id', 'unit-1'])
        expected['unit_id'] = 'unit-1'
    if controller_port is not None:
        options.extend(['--controller-port', str(controller_port)])
        expected['controller_port'] = controller_port
    assert handle_redis_command(parse('fs', role, *options)) == 0
    run.assert_called_once_with(role, 'topology.json', **expected)


@pytest.mark.parametrize('argv', [[], ['cluster'], ['fs']])
def test_group_without_action_prints_help(argv, capsys):
    assert handle_redis_command(parse(*argv)) == 0
    assert 'usage:' in capsys.readouterr().out


def test_help_and_prepare_dispatch_without_optional_dependencies():
    root = Path(__file__).resolve().parents[3]
    script = r"""
import importlib.abc
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

class BlockServices(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname.split('.')[0] in {'redis', 'zmq', 'ray', 'torch', 'hy_kv_store'}:
            raise ModuleNotFoundError(fullname)

sys.meta_path.insert(0, BlockServices())
from nexus.cli.main import main
commands = [[], ['redis'], ['redis', 'cluster'], ['redis', 'fs']]
commands += [['redis', name] for name in ('up', 'serve', 'down')]
commands += [['redis', 'cluster', name] for name in ('prepare', 'serve', 'up', 'down', 'status', 'smoke', 'recover')]
commands += [['redis', 'fs', name] for name in ('up', 'down', 'serve', 'controller', 'data-unit')]
for command in commands:
    sys.argv = ['nexus', *command, '--help']
    try:
        main()
    except SystemExit as exc:
        assert exc.code == 0
    else:
        raise AssertionError(command)
calls = []
def run(argv, *, check):
    assert argv[0] == 'bash'
    script = Path(argv[1])
    assert script.name == 'redis_server_launcher_prepare.sh'
    assert script.is_file()
    assert check is False
    calls.append(argv)
    return SimpleNamespace(returncode=7)
subprocess.run = run
sys.argv = ['nexus', 'redis', 'cluster', 'prepare']
assert main() == 7
assert len(calls) == 1
assert 'nexus.gateway.trajectory_store_backends.redis_store.cluster' not in sys.modules
assert 'nexus.gateway.trajectory_store_backends.redis_store.file_storage' not in sys.modules
"""
    result = subprocess.run(
        [sys.executable, '-c', script],
        cwd=root,
        env={**os.environ, 'PYTHONPATH': str(root), 'NODE_IP_LIST': '', 'NEXUS_REDIS_PRIMARY_COUNT': 'invalid'},
        capture_output=True,
        text=True,
        timeout=120,
    )
    assert result.returncode == 0, result.stdout + result.stderr


def test_root_cli_registers_redis(monkeypatch):
    main = importlib.import_module('nexus.cli.main')

    handler = Mock(return_value=3)
    monkeypatch.setattr(main, 'handle_redis_command', handler)
    monkeypatch.setattr(sys, 'argv', ['nexus', 'redis', 'cluster', 'status'])
    assert main.main() == 3
    assert handler.call_args.args[0].redis_action == 'status'
