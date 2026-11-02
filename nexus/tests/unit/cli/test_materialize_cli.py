from __future__ import annotations

import hashlib
import io
import json
import sys
import zipfile

from pathlib import Path

import pytest

from nexus.cli.main import main
from nexus.distributed.batch.data_list import DataRef, read_indexed_frame, scan_data_list, validate_frame


def test_materialize_command_writes_reviewable_v2_task_list(tmp_path, monkeypatch, capsys):
    generator = tmp_path / 'generator.py'
    generator.write_text(
        'from nexus.runner.batch import BatchItem\n'
        'from nexus.runner.single import RunInput\n'
        'def generate_configs():\n'
        '    return [BatchItem(config=RunInput(task_name="echo"), run_id="case-1")]\n'
    )
    output = tmp_path / 'tasks.jsonl'
    monkeypatch.setattr(
        sys,
        'argv',
        [
            'nexus',
            'materialize',
            '--generator',
            str(generator),
            '--output',
            str(output),
        ],
    )

    assert main() == 0

    summary = json.loads(capsys.readouterr().out)
    assert summary == {
        'path': str(output),
        'record_version': 2,
        'sha256': hashlib.sha256(output.read_bytes()).hexdigest(),
        'size_bytes': output.stat().st_size,
        'total_items': 1,
    }
    record = json.loads(output.read_text())
    assert record['version'] == 2
    assert record['item_id'] == 'case-1'
    assert record['run_input']['task_name'] == 'echo'
    assert record['metadata'] == {}


def _invoke(monkeypatch, *arguments):
    monkeypatch.setattr(sys, 'argv', ['nexus', 'materialize', *map(str, arguments)])
    return main()


def _write_dataset(root):
    root.mkdir()
    splits = {}
    for split in ('train', 'test'):
        paths = []
        for suffix in ('a', 'b', 'c'):
            name = f'{split}-{suffix}'
            path = root / f'{name}.json'
            path.write_text(json.dumps({'id': name, 'body': {'task_name': 'echo'}}))
            paths.append(path.name)
        splits[split] = {'items': paths}
    (root / 'dataset.yaml').write_text(
        json.dumps({
            'schema_version': 1,
            'name': 'materialize-fixture',
            'default_split': 'train',
            'splits': splits,
        })
    )


def test_materialize_harbor_profiles_preserves_scaffold_and_deferred_credentials(tmp_path, monkeypatch, capsys):
    profiles = Path(__file__).parents[3] / 'nexus/extensions/harbor_bridge/configs'
    output = tmp_path / 'profiles.jsonl'
    monkeypatch.setenv('GONGFENG_RUNTIME_TOKEN', 'materialize-test-runtime-secret')
    monkeypatch.setenv('OPENAI_API_KEY', 'materialize-test-llm-secret')

    assert (
        _invoke(
            monkeypatch,
            '--use',
            'harbor_bridge:mode=agent',
            '--use',
            profiles / 'runtime/gongfeng-prod.yaml',
            '--use',
            profiles / 'agents/claude-code.yaml',
            '--use',
            profiles / 'llm/aws-opus48-litellm.yaml',
            '--output',
            output,
        )
        == 0
    )

    summary = json.loads(capsys.readouterr().out)
    assert summary['record_version'] == 2
    assert summary['total_items'] == 1
    assert 'data_list' not in summary
    assert not output.with_suffix('.data-list.bin').exists()
    record = json.loads(output.read_text())
    run_input = record['run_input']
    assert run_input['task_name'] == 'harbor_bridge_task'
    assert run_input['agent_name'] == 'claude_code'
    assert run_input['runtime_provider_name'] == 'gongfeng'
    assert run_input['openai_clients']['main']['name'] == 'litellm'
    assert run_input['runtime_provider_config']['token__env'] == 'GONGFENG_RUNTIME_TOKEN'
    assert run_input['openai_clients']['main']['config']['client_args']['api_key__env'] == 'OPENAI_API_KEY'
    assert 'materialize-test-runtime-secret' not in output.read_text()
    assert 'materialize-test-llm-secret' not in output.read_text()


def test_materialize_keeps_interleaved_compose_order(tmp_path, monkeypatch, capsys):
    profile = tmp_path / 'profile.yaml'
    profile.write_text('body:\n  task_name: echo\n  task_config:\n    value: from-profile\n')
    output = tmp_path / 'tasks.jsonl'

    assert (
        _invoke(
            monkeypatch,
            '--set',
            'task_config.value=before-profile',
            '--use',
            profile,
            '--set-json',
            'task_config.values=[1,2]',
            '--set',
            'task_config.value=after-profile',
            '--output',
            output,
        )
        == 0
    )

    assert json.loads(output.read_text())['run_input']['task_config'] == {
        'value': 'after-profile',
        'values': [1, 2],
    }
    assert json.loads(capsys.readouterr().out)['total_items'] == 1


@pytest.mark.parametrize(
    ('overrides', 'expected_ids'),
    [([], ['train-a', 'train-b']), (['--split', 'test', '--limit', '1'], ['test-a'])],
)
def test_materialize_compose_file_resolves_relative_paths_and_selection(
    tmp_path, monkeypatch, capsys, overrides, expected_ids
):
    _write_dataset(tmp_path / 'dataset')
    (tmp_path / 'profile.yaml').write_text('body:\n  agent_name: mini_swe_agent\n')
    experiment = tmp_path / 'experiment.yaml'
    experiment.write_text(
        'compose:\n'
        '  - data: {uri: ./dataset, split: train, limit: 2}\n'
        '  - use: ./profile.yaml\n'
        '  - input: {task_config: {from_experiment: true}}\n'
    )
    output = tmp_path / 'tasks.jsonl'

    assert _invoke(monkeypatch, experiment, *overrides, '--output', output) == 0

    records = [json.loads(line) for line in output.read_text().splitlines()]
    assert [record['item_id'] for record in records] == expected_ids
    assert all(record['run_input']['agent_name'] == 'mini_swe_agent' for record in records)
    assert all(record['run_input']['task_config']['from_experiment'] for record in records)
    assert json.loads(capsys.readouterr().out)['total_items'] == len(expected_ids)


def test_materialize_compose_file_honors_dataset_item_id(tmp_path, monkeypatch, capsys):
    _write_dataset(tmp_path / 'dataset')
    experiment = tmp_path / 'experiment.yaml'
    experiment.write_text('compose:\n  - data: {uri: ./dataset, split: test, id: test-b}\n')
    output = tmp_path / 'tasks.jsonl'

    assert _invoke(monkeypatch, experiment, '--output', output) == 0

    assert json.loads(capsys.readouterr().out)['total_items'] == 1
    assert json.loads(output.read_text())['item_id'] == 'test-b'


@pytest.mark.parametrize('custom_data_output', [False, True])
def test_materialize_run_input_packages_relative_task_files(tmp_path, monkeypatch, capsys, custom_data_output):
    files = tmp_path / 'files'
    files.mkdir()
    (files / 'instruction.md').write_text('Harbor materialization fixture\n')
    config = tmp_path / 'run.yaml'
    config.write_text('task_name: harbor_bridge_task\ntask_files: {type: directory, path: ./files}\n')
    output = tmp_path / 'tasks.jsonl'
    data_output = tmp_path / 'custom.bin' if custom_data_output else output.with_suffix('.data-list.bin')
    options = ['--data-list-output', data_output] if custom_data_output else []

    assert _invoke(monkeypatch, config, '--output', output, *options) == 0

    summary = json.loads(capsys.readouterr().out)
    assert summary['record_version'] == 3
    assert summary['data_list'] == {
        'path': str(data_output),
        'sha256': hashlib.sha256(data_output.read_bytes()).hexdigest(),
        'size_bytes': data_output.stat().st_size,
        'frame_count': 1,
    }
    record = json.loads(output.read_text())
    assert 'task_files' not in record['run_input']
    assert str(files) not in output.read_text()
    index = scan_data_list(data_output, max_frame_bytes=1024 * 1024, max_bytes=1024 * 1024, max_items=1)
    ref = DataRef.model_validate(record['data_ref'])
    payload = read_indexed_frame(
        data_output,
        ref,
        expected_frame=index.by_offset[ref.offset],
        expected_item_id=record['item_id'],
        max_frame_bytes=1024 * 1024,
    )
    frame = validate_frame(
        payload.encoded,
        expected_ref=ref,
        expected_item_id=record['item_id'],
        max_frame_bytes=1024 * 1024,
    )
    assert frame.zip_bytes is not None
    with zipfile.ZipFile(io.BytesIO(frame.zip_bytes)) as archive:
        assert archive.read('instruction.md') == b'Harbor materialization fixture\n'


@pytest.mark.parametrize('version', [1, 2])
def test_materialize_plain_run_input_keeps_requested_version(tmp_path, monkeypatch, capsys, version):
    config = tmp_path / 'run.json'
    config.write_text(json.dumps({'task_name': 'echo', 'run_id': 'explicit-run'}))
    output = tmp_path / 'tasks.jsonl'

    assert _invoke(monkeypatch, config, '--task-list-version', version, '--output', output) == 0

    assert json.loads(capsys.readouterr().out)['record_version'] == version
    record = json.loads(output.read_text())
    assert record['version'] == version
    assert record['item_id'] == 'explicit-run'


def test_materialize_run_input_resolves_typed_environment_interpolations(tmp_path, monkeypatch, capsys):
    monkeypatch.setenv('MATERIALIZE_TEST_TIMEOUT', '42.5')
    monkeypatch.setenv('MATERIALIZE_TEST_HOLD_SIGNALS', 'false')
    config = tmp_path / 'run.yaml'
    config.write_text(
        'task_name: echo\n'
        'runner:\n'
        '  timeout: ${oc.env:MATERIALIZE_TEST_TIMEOUT,60}\n'
        '  hold_signal_handlers: ${oc.env:MATERIALIZE_TEST_HOLD_SIGNALS,true}\n'
    )
    output = tmp_path / 'tasks.jsonl'

    assert _invoke(monkeypatch, config, '--output', output) == 0

    run_input = json.loads(output.read_text())['run_input']
    assert run_input['runner']['timeout'] == 42.5
    assert run_input['runner']['hold_signal_handlers'] is False
    assert json.loads(capsys.readouterr().out)['total_items'] == 1


@pytest.mark.parametrize(
    'sources',
    [
        [],
        ['--generator', 'missing.py', '--use', 'harbor_bridge'],
        ['missing.yaml', '--generator', 'missing.py'],
        ['missing.yaml', '--set', 'task_name=echo'],
        ['--generator', 'missing.py', '--limit', '1'],
        ['--generator', 'missing.py', '--data-list-output', 'unused.bin'],
        ['--set', 'task_name=echo', '--limit', '0'],
        ['--set', 'task_name=echo', '--limit', '-1'],
        ['--set', 'task_name=echo', '--task-list-version', '1', '--data-list-output', 'unused.bin'],
    ],
)
def test_materialize_rejects_invalid_sources_without_touching_output(tmp_path, monkeypatch, capsys, sources):
    output = tmp_path / 'tasks.jsonl'
    output.write_bytes(b'previous task list')

    assert _invoke(monkeypatch, *sources, '--output', output) == 1

    assert output.read_bytes() == b'previous task list'
    captured = capsys.readouterr()
    assert not captured.out
    assert 'materialize failed:' in captured.err
    assert 'No such file' not in captured.err


def test_materialize_does_not_treat_a_profile_file_as_run_input(tmp_path, monkeypatch, capsys):
    profile = tmp_path / 'profile.yaml'
    profile.write_text('body:\n  task_name: echo\n')

    assert _invoke(monkeypatch, profile, '--output', tmp_path / 'tasks.jsonl') == 1

    assert '--use' in capsys.readouterr().err
    assert not (tmp_path / 'tasks.jsonl').exists()


def test_materialize_failed_file_packaging_preserves_previous_outputs(tmp_path, monkeypatch, capsys):
    config = tmp_path / 'bad.yaml'
    config.write_text('task_name: echo\ntask_files: {type: directory, path: ./missing}\n')
    output = tmp_path / 'tasks.jsonl'
    data_output = output.with_suffix('.data-list.bin')
    output.write_bytes(b'previous tasks')
    data_output.write_bytes(b'previous data')

    assert _invoke(monkeypatch, config, '--output', output) == 1

    assert output.read_bytes() == b'previous tasks'
    assert data_output.read_bytes() == b'previous data'
    assert not list(tmp_path.glob('.nexus-materialize-*'))
    assert not capsys.readouterr().out


def test_materialize_v1_rejects_local_task_files(tmp_path, monkeypatch, capsys):
    files = tmp_path / 'files'
    files.mkdir()
    config = tmp_path / 'run.yaml'
    config.write_text('task_name: echo\ntask_files: {type: directory, path: ./files}\n')
    output = tmp_path / 'tasks.jsonl'

    assert _invoke(monkeypatch, config, '--task-list-version', '1', '--output', output) == 1

    assert 'local_compose_task_files_unsupported' in capsys.readouterr().err
    assert not output.exists()


_AIHUB_IDS = Path(__file__).resolve().parents[3] / 'docs/examples/configs/compose-hooks/aihub_ids.py'

_HOOK_GENERATOR = """
from nexus.runner.batch import BatchItem
from nexus.runner.single import RunInput

def generate_configs():
    for index in range(3):
        client = {
            'name': 'openai_responses',
            'config': {'client_args': {'api_key': 'credential?provider=aws_third'}, 'request_args': {'model': 'm'}},
        }
        yield BatchItem(
            run_id=f'case-{index}',
            metadata={'source': 'gen'},
            config=RunInput(task_name='echo', openai_clients={'main': client}),
        )
"""


def test_materialize_generator_runs_hooks_in_order(tmp_path, monkeypatch, capsys):
    generator = tmp_path / 'generator.py'
    generator.write_text(_HOOK_GENERATOR)
    hook = tmp_path / 'stamp.py'
    hook.write_text(
        'def create_compose_hook(params):\n'
        '    def process(item):\n'
        '        item.metadata["key_seen"] = item.config.openai_clients["main"]["config"]["client_args"]["api_key"]\n'
        '        return item\n'
        '    return process\n'
    )
    output = tmp_path / 'tasks.jsonl'

    assert (
        _invoke(
            monkeypatch,
            '--generator',
            generator,
            '--hook',
            f'{_AIHUB_IDS}:fields=cache_task_id,mode=unique,key=openai_clients.main.config.client_args.api_key',
            '--hook',
            hook,
            '--output',
            output,
        )
        == 0
    )

    records = [json.loads(line) for line in output.read_text().splitlines()]
    assert [record['item_id'] for record in records] == ['case-0', 'case-1', 'case-2']
    ids = [record['metadata']['__extra__']['cache_task_id'] for record in records]
    assert len(set(ids)) == 3
    for record, cache_id in zip(records, ids, strict=True):
        api_key = record['run_input']['openai_clients']['main']['config']['client_args']['api_key']
        assert api_key == f'credential?provider=aws_third&cache_task_id={cache_id}'
        assert record['metadata']['key_seen'] == api_key
        assert record['metadata']['source'] == 'gen'


def test_materialize_generator_still_rejects_static_overlays(tmp_path, monkeypatch, capsys):
    generator = tmp_path / 'generator.py'
    generator.write_text(_HOOK_GENERATOR)
    assert _invoke(monkeypatch, '--generator', generator, '--set', 'runner.timeout=1', '--output', tmp_path / 'o') == 1
    assert 'provide exactly one of --generator' in capsys.readouterr().err


def test_materialize_hook_failure_keeps_the_previous_task_list(tmp_path, monkeypatch, capsys):
    generator = tmp_path / 'generator.py'
    generator.write_text(_HOOK_GENERATOR.replace("f'case-{index}'", "'same'"))
    output = tmp_path / 'tasks.jsonl'
    output.write_bytes(b'previous task list')

    assert (
        _invoke(
            monkeypatch,
            '--generator',
            generator,
            '--hook',
            f'{_AIHUB_IDS}:mode=unique,key=openai_clients.main.config.client_args.api_key',
            '--output',
            output,
        )
        == 1
    )

    assert output.read_bytes() == b'previous task list'
    assert 'cache_task_id is also used by item #0' in capsys.readouterr().err
