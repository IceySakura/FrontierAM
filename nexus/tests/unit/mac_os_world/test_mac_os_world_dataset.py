import hashlib
import json
import shutil

from pathlib import Path

import pytest

from nexus.core.context import ExecutionContext
from nexus.dataset.layout import load_local_dataset
from nexus.dataset.publisher import build_publish_layout
from nexus.dataset.task_files import task_files_from_spec
from nexus.extensions.tasks.mac_os_world.dataset import build_mac_os_world_dataset
from nexus.extensions.tasks.mac_os_world.task import MacOSWorldTaskInput, run_mac_os_world
from nexus.runner.compose import compile_single, parse_ordered_parts


def _source(tmp_path: Path) -> Path:
    source = tmp_path / 'upstream'
    source.mkdir()
    (source / 'LICENSE').write_text('Upstream license text', encoding='utf-8')
    for category, task_id in [('sys_apps', 'normal-task'), ('safety', 'safety-task')]:
        path = source / 'tasks' / category / f'{task_id}.json'
        path.parent.mkdir(parents=True)
        path.write_text(
            json.dumps({
                'id': task_id,
                'task': {'en': 'Open the version window.', 'zh': '打开版本窗口。'},
                'pre_command': 'setup',
                'grading_command': [['echo True', 100]],
                'snapshot': 'upstream-snapshot',
            }),
            encoding='utf-8',
        )
    return source


def test_dataset_preserves_source_and_separates_safety(tmp_path: Path) -> None:
    source = _source(tmp_path)
    output = tmp_path / 'dataset'
    counts = build_mac_os_world_dataset(source, output, source_revision='source-commit')
    assert counts == {'safety': 1, 'test': 1}
    local = load_local_dataset(output)
    assert local.default_split == 'test'
    item = local.load_items()[0]
    assert item.id == 'normal-task'
    assert item.body['task_config'] == {
        'task_file': 'task.json',
        'category': 'sys_apps',
        'task_id': 'normal-task',
    }
    files = item.local_task_files_path()
    assert files is not None
    original = (source / 'tasks/sys_apps/normal-task.json').read_bytes()
    assert (files / 'task.json').read_bytes() == original
    assert (files / 'LICENSE').read_bytes() == (source / 'LICENSE').read_bytes()
    assert item.metadata['source_sha256'] == hashlib.sha256(original).hexdigest()
    assert item.metadata['source_revision'] == 'source-commit'
    assert item.metadata['languages'] == ['en', 'zh']
    assert str(source) not in json.dumps(item.body)
    assert [item.id for item in local.load_items(split='safety')] == ['safety-task']
    with pytest.raises(FileExistsError):
        build_mac_os_world_dataset(source, output)


@pytest.mark.asyncio
@pytest.mark.parametrize('file_type', ['directory', 'zip'])
async def test_dataset_task_runs_without_upstream_checkout(tmp_path: Path, monkeypatch, file_type: str) -> None:
    source = _source(tmp_path)
    output = tmp_path / 'dataset'
    build_mac_os_world_dataset(source, output)
    item = load_local_dataset(output).load_items()[0]
    if file_type == 'zip':
        build = build_publish_layout(output, namespace='test', release_version='v1', output_dir=tmp_path / 'build')
        assert build.item_count == 2
        published = json.loads((build.files_root / 'test/raw/normal-task.json').read_text())
        spec = {**published['files'], 'path': str(build.files_root / 'test/blobs/normal-task.zip')}
        spec.pop('remote', None)
        shutil.rmtree(output)
    else:
        spec = item.resolved_task_files()
    shutil.rmtree(source)
    monkeypatch.chdir(tmp_path)
    assert not Path('task.json').exists()

    commands: list[str] = []

    class Runtime:
        async def run_command(self, command, timeout=60):
            commands.append(command)
            return {'return_code': 0, 'stdout': 'True'}

        async def drain_trace(self):
            return None

    async def unused_agent(**kwargs):
        raise AssertionError('Agent should be disabled in the dataset loading test')

    ctx = ExecutionContext(task_name='mac_os_world', task_files=task_files_from_spec(spec))
    ctx._runtimes['main'] = Runtime()
    async with ctx:
        result = await run_mac_os_world(
            MacOSWorldTaskInput(
                **item.body['task_config'],
                language='zh',
                run_preflight=False,
                run_pre_command=False,
                execute_agent=False,
                run_agent=unused_agent,
            )
        )
    assert commands == ['echo True']
    assert result.instruction == '打开版本窗口。'
    assert result.task_id == 'normal-task'
    assert result.category == 'sys_apps'
    assert result.score == 1.0


def test_duplicate_task_ids_are_rejected_before_writing(tmp_path: Path) -> None:
    source = _source(tmp_path)
    path = source / 'tasks/safety/safety-task.json'
    data = json.loads(path.read_text())
    data['id'] = 'normal-task'
    path.write_text(json.dumps(data))
    output = tmp_path / 'dataset'
    with pytest.raises(ValueError, match='Duplicate'):
        build_mac_os_world_dataset(source, output)
    assert not output.exists()


def test_dataset_composes_with_luna_runtime_config(tmp_path: Path, monkeypatch) -> None:
    source = _source(tmp_path)
    output = tmp_path / 'dataset'
    build_mac_os_world_dataset(source, output)
    monkeypatch.setenv('DEVCLOUD_APP_ID', 'test-app')
    monkeypatch.setenv('DEVCLOUD_APP_TOKEN', 'test-token')
    monkeypatch.setenv('GPT_CUA_API_KEY', 'test-model-key')
    repo = Path(__file__).resolve().parents[3]
    config = repo / 'docs/examples/configs/mac_os_world/mac_os_world_gpt56_luna_devcloud_test.yaml'
    run_input = compile_single(
        parse_ordered_parts(['--use', str(config), '--data', str(output)]), item_id='normal-task'
    )
    assert run_input.task_config['task_id'] == 'normal-task'
    assert run_input.task_config['task_file'] == 'task.json'
    assert 'task_root' not in run_input.task_config
    assert run_input.agent_name == 'macosworld_cua_gpt'
    assert 'nexus.extensions.agents.cua.macosworld_cua.gpt' in run_input.imports
    assert run_input.imports.count('nexus.extensions.tasks.mac_os_world') == 1
    assert run_input.task_files is not None
    assert Path(run_input.task_files.path).is_dir()
