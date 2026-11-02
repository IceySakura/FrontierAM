from __future__ import annotations

import argparse
import json
import sys
import zipfile

import pytest

from nexus.cli.commands.modules import register_modules_parser, resolve_module_wheel_selectors
from nexus.core.module_dependencies import ModuleSelector
from nexus.core.module_meta import discover_all_modules
from nexus.distributed.dependency_build.local import build_local_module_wheelhouse


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    register_modules_parser(parser.add_subparsers(dest='command'))
    return parser


def test_modules_wheel_parser_requires_exactly_one_configuration_source(tmp_path):
    common = [
        'modules',
        'wheel',
        '--lock',
        str(tmp_path / 'uv.lock'),
        '--nexus-wheel',
        str(tmp_path / 'nexus.whl'),
        '--output',
        str(tmp_path / 'output'),
    ]

    with pytest.raises(SystemExit):
        _parser().parse_args(common)
    with pytest.raises(SystemExit):
        _parser().parse_args(common + ['--run-input', 'run.yaml', '--task-list', 'tasks.jsonl'])

    args = _parser().parse_args(common + ['--run-input', 'run.yaml'])
    assert args.modules_command == 'wheel'
    assert args.run_input == 'run.yaml'
    assert args.task_list is None


def test_module_wheel_run_input_resolves_normalized_module_selectors(tmp_path):
    run_input = tmp_path / 'run.yaml'
    run_input.write_text(
        'task_name: frontend\nagent_name: react\nopenai_clients:\n  main:\n    name: litellm\n    config: {}\n'
    )

    representative_item_id, selectors = resolve_module_wheel_selectors(
        run_input_path=run_input,
        task_list_path=None,
        max_line_bytes=64 * 1024,
    )

    assert representative_item_id is None
    assert [selector.model_dump(mode='json') for selector in selectors] == [
        {'kind': 'object', 'value': 'react', 'object_type': 'agents'},
        {'kind': 'object', 'value': 'litellm', 'object_type': 'openai_clients'},
        {'kind': 'object', 'value': 'frontend', 'object_type': 'tasks'},
    ]


def test_module_wheel_task_list_uses_only_first_validated_record(tmp_path):
    task_list = tmp_path / 'tasks.jsonl'
    first = {
        'version': 1,
        'item_id': 'item-000000000001',
        'run_input': {
            'task_name': 'frontend',
            'openai_clients': {'main': {'name': 'litellm', 'config': {}}},
        },
    }
    task_list.write_bytes(
        json.dumps(first, sort_keys=True, separators=(',', ':')).encode() + b'\n{later-record-is-invalid}\n'
    )

    representative_item_id, selectors = resolve_module_wheel_selectors(
        run_input_path=None,
        task_list_path=task_list,
        max_line_bytes=64 * 1024,
    )

    assert representative_item_id == 'item-000000000001'
    assert [selector.model_dump(mode='json') for selector in selectors] == [
        {'kind': 'object', 'value': 'litellm', 'object_type': 'openai_clients'},
        {'kind': 'object', 'value': 'frontend', 'object_type': 'tasks'},
    ]


def test_module_wheel_rejects_empty_task_list(tmp_path):
    task_list = tmp_path / 'empty.jsonl'
    task_list.write_bytes(b'')

    with pytest.raises(ValueError, match='module wheel Task List is empty'):
        resolve_module_wheel_selectors(
            run_input_path=None,
            task_list_path=task_list,
            max_line_bytes=64 * 1024,
        )


@pytest.mark.parametrize('with_files', [False, True])
def test_module_wheel_selects_harbor_and_scaffold_from_materialized_profiles(tmp_path, monkeypatch, with_files):
    from nexus.cli.main import main
    from nexus.core.module_dependencies import resolve_host_module_dependencies

    output = tmp_path / 'tasks.jsonl'
    profile = tmp_path / 'profile.yaml'
    profile.write_text(
        'body:\n'
        '  task_name: harbor_bridge_task\n'
        '  agent_name: mini_swe_agent\n'
        '  runtime_provider_name: gongfeng\n'
        '  openai_clients:\n'
        '    main: {name: litellm, config: {}}\n'
    )
    options = []
    if with_files:
        files = tmp_path / 'files'
        files.mkdir()
        (files / 'instruction.md').write_text('fixture')
        options = ['--set-json', 'task_files=' + json.dumps({'type': 'directory', 'path': str(files)})]
    monkeypatch.setattr(
        sys,
        'argv',
        [
            'nexus',
            'materialize',
            '--use',
            str(profile),
            *options,
            '--output',
            str(output),
        ],
    )

    assert main() == 0
    assert json.loads(output.read_text())['version'] == (3 if with_files else 2)
    _, selectors = resolve_module_wheel_selectors(
        run_input_path=None,
        task_list_path=output,
        max_line_bytes=16 * 1024 * 1024,
    )
    resolved = resolve_host_module_dependencies(selectors, discover_all_modules())

    assert {module.name for module in resolved.modules} == {'harbor_bridge', 'mini_swe_agent', 'gongfeng', 'litellm'}
    assert 'harbor==0.20.0' in resolved.requirements
    assert 'jinja2>=3.0' in resolved.requirements


def test_local_module_wheelhouse_uses_shared_builder_and_writes_non_installing_report(tmp_path):
    project = tmp_path / 'project'
    project.mkdir()
    (project / 'pyproject.toml').write_text(
        '[project]\n'
        'name = "nexus-agent"\n'
        'version = "1.0.0"\n'
        'dependencies = []\n'
        '[project.optional-dependencies]\n'
        'distributed = []\n'
        'runtime = []\n'
    )
    lock = project / 'uv.lock'
    lock.write_text('version = 1\nrevision = 3\nrequires-python = ">=3.10"\n')
    wheel = tmp_path / 'nexus_agent-1.0.0-py3-none-any.whl'
    with zipfile.ZipFile(wheel, 'w') as archive:
        archive.writestr(
            'nexus_agent-1.0.0.dist-info/METADATA',
            'Metadata-Version: 2.4\n'
            'Name: nexus-agent\n'
            'Version: 1.0.0\n'
            'Provides-Extra: distributed\n'
            'Provides-Extra: runtime\n',
        )
        archive.writestr(
            'nexus_agent-1.0.0.dist-info/WHEEL',
            'Wheel-Version: 1.0\nGenerator: test\nRoot-Is-Purelib: true\nTag: py3-none-any\n',
        )
        archive.writestr(
            'nexus/extensions/tasks/identity/pyproject.toml',
            '[project]\n'
            'name = "identity"\n'
            'version = "0.1.0"\n'
            'dependencies = []\n'
            '[tool.nexus]\n'
            'tasks = { identity = "nexus.extensions.tasks.identity.task" }\n',
        )
    commands: list[list[str]] = []

    def fake_run(command, *, cwd, check, **kwargs):
        del cwd, kwargs
        assert check is True
        commands.append(list(command))
        if command[:2] == ['uv', 'export']:
            output = command[command.index('--output-file') + 1]
            with open(output, 'w', encoding='utf-8') as stream:
                stream.write('')

    output = tmp_path / 'output'
    report = build_local_module_wheelhouse(
        project_root=project,
        uv_lock=lock,
        nexus_wheel=wheel,
        output=output,
        selectors=(ModuleSelector(kind='object', object_type='tasks', value='identity'),),
        representative_item_id='item-000000000001',
        runner=fake_run,
    )

    assert report['representative_item_id'] == 'item-000000000001'
    assert report['wheelhouse']['schema_version'] == 2
    assert report['wheelhouse']['module_names'] == ['identity']
    assert report['wheelhouse']['module_paths'] == ['nexus.extensions.tasks.identity']
    assert (output / 'module-wheel-report.json').is_file()
    assert (output / 'wheelhouse-metadata.json').is_file()
    assert (output / 'wheels' / wheel.name).is_file()
    assert not (output / 'locks' / 'module-host.lock').exists()
    assert not any(command[:3] == ['uv', 'pip', 'compile'] for command in commands)
    assert any(command[:3] == ['uv', 'pip', 'install'] for command in commands)
    assert all(command[0] != sys.executable or command[1:3] == ['-m', 'pip'] for command in commands)


def test_local_module_wheelhouse_rejects_symlink_output_without_touching_target(tmp_path):
    project = tmp_path / 'project'
    project.mkdir()
    lock = tmp_path / 'uv.lock'
    lock.write_text('', encoding='utf-8')
    wheel = tmp_path / 'nexus.whl'
    wheel.write_bytes(b'not-needed')
    target = tmp_path / 'existing-output'
    target.mkdir()
    marker = target / 'keep.txt'
    marker.write_text('untouched', encoding='utf-8')
    output = tmp_path / 'output-link'
    output.symlink_to(target, target_is_directory=True)

    with pytest.raises(ValueError, match='output.*symlink'):
        build_local_module_wheelhouse(
            project_root=project,
            uv_lock=lock,
            nexus_wheel=wheel,
            output=output,
            selectors=(ModuleSelector(kind='object', object_type='tasks', value='identity'),),
            representative_item_id=None,
        )

    assert output.is_symlink()
    assert marker.read_text(encoding='utf-8') == 'untouched'
