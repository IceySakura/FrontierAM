from __future__ import annotations

import argparse
import json
import threading

from pathlib import Path

import pytest
import yaml

from nexus.cli.commands.dataset import dataset_fetch
from nexus.dataset.reader import DatasetReaderError, list_item_ids, load_items
from nexus.dataset.source import DatasetSourceURI
from nexus.dataset.zip_index import read_zip_index_from_bytes
from nexus.distributed.batch.data_list import DataRef, read_indexed_frame, scan_data_list
from nexus.distributed.batch.materialize import materialize_compose
from nexus.extensions.harbor_bridge import compiler, dataset_source, generator
from nexus.extensions.harbor_bridge.dataset_source import HarborDatasetReader
from nexus.extensions.harbor_bridge.run_parts import harbor_bridge_run_part  # noqa: F401
from nexus.runner.compose import compile_batch, parse_ordered_parts
from nexus.runner.compose.model import RawRunPart


def _write_task(root: Path, name: str) -> Path:
    task_dir = root / name
    (task_dir / 'environment').mkdir(parents=True)
    (task_dir / 'tests').mkdir()
    (task_dir / 'task.toml').write_text(
        """schema_version = "1.2"

[environment]
docker_image = "agent:v1"
workdir = "/app"
cpus = 2
memory_mb = 4096
network_mode = "public"

[agent]
timeout_sec = 7200

[verifier]
environment_mode = "shared"
timeout_sec = 1800
"""
    )
    (task_dir / 'instruction.md').write_text(f'Complete {name}.\n')
    (task_dir / 'tests' / 'test.sh').write_text('#!/bin/sh\necho 1 > /logs/verifier/reward.txt\n')
    return task_dir


def _write_broken_task(root: Path, name: str) -> Path:
    task_dir = root / name
    task_dir.mkdir(parents=True)
    (task_dir / 'task.toml').write_text('[environment\n')
    return task_dir


def _open_reader(root: Path, **query: str) -> HarborDatasetReader:
    raw = f't3-harbor://{root}'
    if query:
        raw += '?' + '&'.join(f'{key}={value}' for key, value in query.items())
    from nexus.dataset.reader import open_dataset

    reader = open_dataset(raw)
    assert isinstance(reader, HarborDatasetReader)
    return reader


async def _load_reader_items(
    root: Path,
    *,
    split: str | None = None,
    item_id: str | None = None,
    limit: int | None = None,
    **query: str,
):
    async with _open_reader(root, **query) as reader:
        return await load_items(reader, split=split, item_id=item_id, limit=limit)


async def _reader_item_ids(
    root: Path,
    *,
    split: str | None = None,
    limit: int | None = None,
    **query: str,
) -> list[str]:
    async with _open_reader(root, **query) as reader:
        return await list_item_ids(reader, split=split, limit=limit)


def test_dataset_source_and_legacy_generator_share_the_compiler_api() -> None:
    assert dataset_source.compile_harbor_task is compiler.compile_harbor_task
    assert dataset_source.load_harbor_task is compiler.load_harbor_task


@pytest.mark.asyncio
async def test_t3_harbor_source_scans_and_compiles_items(tmp_path: Path) -> None:
    _write_task(tmp_path, 'alpha')
    _write_task(tmp_path, 'beta')
    reader = HarborDatasetReader(
        DatasetSourceURI(
            scheme='t3-harbor',
            path=str(tmp_path),
            query={'flat': ('true',)},
            raw=f't3-harbor://{tmp_path}',
        )
    )

    async with reader:
        assert await list_item_ids(reader) == ['alpha', 'beta']
        assert await list_item_ids(reader, limit=1) == ['alpha']
        descriptor = await reader.describe()
        assert descriptor.splits[0].item_count == 2
        assert descriptor.resolved_reference == str(tmp_path.resolve())
        assert descriptor.metadata['compile_workers'] == 1
        item = (await load_items(reader, item_id='alpha'))[0]
        with pytest.raises(DatasetReaderError) as missing_item:
            await load_items(reader, item_id='missing')

    assert item.id == 'alpha'
    assert missing_item.value.code == 'dataset_item_not_found'
    assert item.files is not None
    assert item.files.type == 'directory'
    assert item.files.path == str(tmp_path / 'alpha')
    assert 'task_dir' not in item.body['task_config']
    assert item.body['task_config']['instruction'] == 'Complete alpha.\n'
    assert item.body['runtime_configs']['main']['image'] == 'agent:v1'
    assert 'agent_name' not in item.body
    assert 'openai_clients' not in item.body
    assert item.metadata['format'] == 't3-harbor'
    assert item.metadata['task_identity_version'] == 't3h1'
    assert len(item.metadata['task_content_sha256']) == 64
    assert 'source_path' not in item.metadata


@pytest.mark.asyncio
async def test_compile_workers_run_concurrently_and_preserve_item_order(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    for name in ('delta', 'alpha', 'charlie', 'bravo'):
        _write_task(tmp_path, name)
    reader = _open_reader(tmp_path, flat='true', compile_workers='4')
    original_compile = reader._compile_item
    rendezvous = threading.Barrier(4)

    def compile_item(selected_id: str, task_dir: Path):
        rendezvous.wait(timeout=5)
        return original_compile(selected_id, task_dir)

    monkeypatch.setattr(reader, '_compile_item', compile_item)
    async with reader:
        items = await load_items(reader)

    assert [item.id for item in items] == ['alpha', 'bravo', 'charlie', 'delta']


@pytest.mark.asyncio
async def test_compile_workers_bound_speculative_work_for_limit(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    for name in ('alpha', 'bravo', 'charlie', 'delta'):
        _write_task(tmp_path, name)
    reader = _open_reader(tmp_path, flat='true', compile_workers='2')
    original_compile = reader._compile_item
    compiled: list[str] = []
    lock = threading.Lock()

    def compile_item(selected_id: str, task_dir: Path):
        with lock:
            compiled.append(selected_id)
        return original_compile(selected_id, task_dir)

    monkeypatch.setattr(reader, '_compile_item', compile_item)
    async with reader:
        items = await load_items(reader, limit=1)

    assert [item.id for item in items] == ['alpha']
    assert sorted(compiled) == ['alpha', 'bravo']


def test_dataset_fetch_materializes_harbor_source_as_portable_local_dataset(tmp_path: Path) -> None:
    _write_task(tmp_path, 'alpha')
    output = tmp_path / 'harbor-local'

    result = dataset_fetch(
        argparse.Namespace(
            dataset=f't3-harbor://{tmp_path}?flat=true',
            split=None,
            id=None,
            limit=None,
            output=str(output),
            force=False,
            json=False,
            taihu_token=None,
            taihu_token_file=None,
        )
    )

    assert result == 0
    item_path = output / 'items' / 'test' / 'alpha' / 'item.yaml'
    files_path = output / 'items' / 'test' / 'alpha' / 'files.zip'
    assert item_path.is_file()
    assert files_path.is_file()
    payload = yaml.safe_load(item_path.read_text())
    assert payload['files']['path'] == 'files.zip'
    assert payload['files']['type'] == 'zip'
    assert 'task_dir' not in payload['body']['task_config']
    assert str(tmp_path) not in item_path.read_text()
    assert str(tmp_path) not in (output / 'dataset.yaml').read_text()


@pytest.mark.asyncio
async def test_t3_harbor_compose_materializes_task_files_into_data_list(tmp_path: Path) -> None:
    _write_task(tmp_path, 'alpha')
    uri = f't3-harbor://{tmp_path}?flat=true'
    task_list_path = tmp_path / 'tasks.jsonl'
    data_list_path = tmp_path / 'data_list.bin'

    result = await materialize_compose(
        [
            RawRunPart(kind='data', value=uri, source=f'--data {uri}'),
            RawRunPart(kind='use', value='harbor_bridge:mode=agent', source='--use harbor_bridge:mode=agent'),
        ],
        task_list_path,
        task_list_version=2,
        auto_data_list=True,
        data_list_output_path=data_list_path,
    )

    assert result.task_list.record_version == 3
    record = json.loads(task_list_path.read_text(encoding='utf-8'))
    assert 'task_files' not in record['run_input']
    assert 'task_dir' not in record['run_input']['task_config']
    assert str(tmp_path) not in task_list_path.read_text(encoding='utf-8')

    data_index = scan_data_list(data_list_path)
    frame_ref = DataRef.model_validate(record['data_ref'])
    frame = read_indexed_frame(
        data_list_path,
        frame_ref,
        expected_frame=data_index.frames[0],
        expected_item_id=record['item_id'],
    )
    descriptor = read_zip_index_from_bytes(frame.encoded[data_index.frames[0].zip_offset :])
    assert {'task.toml', 'instruction.md', 'tests/test.sh'} <= set(descriptor.entries)


@pytest.mark.asyncio
async def test_t3_harbor_source_compiles_optional_agent_trajectory_request(tmp_path: Path) -> None:
    task_dir = _write_task(tmp_path, 'alpha')
    with (task_dir / 'task.toml').open('a') as task_file:
        task_file.write('needs_agent_trajectory = true\n')

    item = (await _load_reader_items(tmp_path, item_id='alpha', flat='true'))[0]

    assert item.body['task_config']['needs_agent_trajectory'] is True
    assert item.body['processors'] == {
        'harbor_agent_trajectory': {
            'name': 'harbor_agent_trajectory_tracker',
            'args': {'tools_mode': 'branch_last'},
        }
    }


@pytest.mark.asyncio
async def test_t3_harbor_source_rejects_non_boolean_agent_trajectory_request(tmp_path: Path) -> None:
    task_dir = _write_task(tmp_path, 'alpha')
    with (task_dir / 'task.toml').open('a') as task_file:
        task_file.write('needs_agent_trajectory = "true"\n')

    with pytest.raises(ValueError, match='verifier.needs_agent_trajectory must be a boolean'):
        await _load_reader_items(tmp_path, item_id='alpha', flat='true')


@pytest.mark.asyncio
async def test_t3_harbor_source_compiles_shared_browser_declaration(tmp_path: Path) -> None:
    task_dir = _write_task(tmp_path, 'alpha')
    with (task_dir / 'task.toml').open('a') as task_file:
        task_file.write(
            """

[environment.browser]
start_url = "https://example.com/task"
viewport_width = 1440
"""
        )

    item = (await _load_reader_items(tmp_path, item_id='alpha', flat='true'))[0]

    assert item.body['task_config']['browser'] == {
        'start_url': 'https://example.com/task',
        'headless': True,
        'viewport_width': 1440,
        'viewport_height': 720,
        'nav_timeout_ms': 10000,
        'runtime_name': 'main',
    }


@pytest.mark.asyncio
async def test_t3_harbor_source_copies_instance_id_without_ad_hoc_metadata(tmp_path: Path) -> None:
    task_dir = _write_task(tmp_path, 'alpha')
    with (task_dir / 'task.toml').open('a') as task_file:
        task_file.write(
            """

[metadata]
instance_id = "canonical-alpha"
language = "zh"
programming_language = "python"
difficulty = "hard"
category = "code"
"""
        )

    item = (await _load_reader_items(tmp_path, item_id='alpha', flat='true'))[0]

    assert item.metadata['instance_id'] == 'canonical-alpha'
    assert 'language' not in item.metadata
    assert 'programming_language' not in item.metadata
    assert 'difficulty' not in item.metadata
    assert 'category' not in item.metadata


@pytest.mark.asyncio
async def test_t3_harbor_source_freezes_normalized_mcp_specs(tmp_path: Path) -> None:
    task_dir = _write_task(tmp_path, 'alpha')
    with (task_dir / 'task.toml').open('a') as task_file:
        task_file.write(
            """

[[environment.mcp_servers]]
name = "search"
transport = "streamable-http"
url = "http://127.0.0.1:9900/mcp"
"""
        )

    item = (await _load_reader_items(tmp_path, item_id='alpha', flat='true'))[0]
    frozen = item.body['task_config']['mcp_servers']
    (task_dir / 'task.toml').write_text('[environment]\ndocker_image = "changed:v2"\n')

    assert [server['name'] for server in frozen] == ['search']
    assert frozen[0]['client']['transport'] == 'streamable_http'


@pytest.mark.asyncio
async def test_t3_harbor_source_preserves_t3_mcp_extensions(tmp_path: Path) -> None:
    task_dir = _write_task(tmp_path, 'alpha')
    with (task_dir / 'task.toml').open('a') as task_file:
        task_file.write(
            """

[[environment.mcp_servers]]
name = "search"
transport = "streamable-http"
url = "http://127.0.0.1:9900/mcp"
headers = { Authorization = "${MCP_AUTHORIZATION}" }
timeout_s = 30.0
runtime_name = "main"
"""
        )

    item = (await _load_reader_items(tmp_path, item_id='alpha', flat='true'))[0]
    [server] = item.body['task_config']['mcp_servers']

    assert server['runtime_name'] == 'main'
    assert server['nexus_managed'] is False
    assert server['client']['headers'] == {'Authorization__env': 'MCP_AUTHORIZATION'}
    assert server['client']['timeout_s'] == 30.0


@pytest.mark.asyncio
async def test_t3_harbor_source_requires_explicit_mcp_transport(tmp_path: Path) -> None:
    task_dir = _write_task(tmp_path, 'alpha')
    with (task_dir / 'task.toml').open('a') as task_file:
        task_file.write(
            """

[[environment.mcp_servers]]
name = "search"
url = "http://127.0.0.1:9900/mcp"
"""
        )

    with pytest.raises(ValueError, match='transport must be explicitly set'):
        await _load_reader_items(tmp_path, item_id='alpha', flat='true')


@pytest.mark.asyncio
async def test_t3_harbor_source_resolves_composite_runtime_and_boot_env(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    task_dir = _write_task(tmp_path, 'alpha')
    with (task_dir / 'task.toml').open('a') as task_file:
        task_file.write(
            """

[environment.env]
AUTH = "${APP_ID}:${APP_KEY}"
MODE = "${MODE:-production}"

[environment.boot]
command = "/start-services"
"""
        )
    monkeypatch.setenv('APP_ID', 'app')
    monkeypatch.setenv('APP_KEY', 'key')
    monkeypatch.delenv('MODE', raising=False)

    item = (await _load_reader_items(tmp_path, item_id='alpha', flat='true'))[0]

    assert item.body['runtime_configs']['main']['environment'] == {
        'AUTH': 'app:key',
        'MODE': 'production',
    }
    assert item.body['task_config']['boot_env'] == {
        'AUTH': 'app:key',
        'MODE': 'production',
    }


@pytest.mark.asyncio
async def test_t3_harbor_source_ignores_legacy_mcp_injection_override(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    task_dir = _write_task(tmp_path, 'alpha')
    with (task_dir / 'task.toml').open('a') as task_file:
        task_file.write(
            """

[[environment.mcp_servers]]
name = "files"
transport = "stdio"
command = "node"
"""
        )
    monkeypatch.setenv('HARBOR_MCP_INJECT', 'off')

    item = (await _load_reader_items(tmp_path, item_id='alpha', flat='true'))[0]

    assert [server['name'] for server in item.body['task_config']['mcp_servers']] == ['files']
    assert 'compile_env' not in item.metadata


def test_t3_harbor_source_composes_with_harbor_run_part(tmp_path: Path) -> None:
    task_dir = _write_task(tmp_path, 'alpha')
    with (task_dir / 'task.toml').open('a') as task_file:
        task_file.write('\n[metadata]\ninstance_id = "canonical-alpha"\n')
    uri = f't3-harbor://{tmp_path}?flat=true'
    policy_path = Path(dataset_source.__file__).parent / 'configs' / 'policies' / 'code-anti-hack-v1.yaml'

    items = compile_batch(
        parse_ordered_parts([
            '--data',
            uri,
            '--use',
            'harbor_bridge:mode=agent',
            '--use',
            str(policy_path),
        ]),
        limit=1,
    )

    assert len(items) == 1
    assert items[0].run_id.startswith('t3-harbor__alpha__')
    assert items[0].config.run_id == items[0].run_id
    assert items[0].metadata['instance_id'] == 'canonical-alpha'
    assert items[0].config.task_name == 'harbor_bridge_task'
    assert items[0].config.task_config['instance_id'] == 'alpha'
    assert items[0].config.task_config['mode'] == 'agent'
    policy = items[0].config.task_config['anti_hack_policy']
    assert policy['id'] == 'code-anti-hack-v1'
    assert policy['delivery'] == 'instruction_suffix'
    assert policy['trajectory_strategy'] == 'restricted-source-access-v1'
    assert items[0].config.processors['harbor_agent_trajectory'].name == 'harbor_agent_trajectory_tracker'


def test_legacy_generator_can_disable_task_mcp(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    task_dir = _write_task(tmp_path, 'alpha')
    with (task_dir / 'task.toml').open('a') as task_file:
        task_file.write(
            """

[[environment.mcp_servers]]
name = "files"
transport = "stdio"
command = "node"
"""
        )
    monkeypatch.setenv('HARBOR_DATASET_PATH', str(task_dir))
    monkeypatch.setenv('GONGFENG_RUNTIME_TOKEN', 'test-token')
    monkeypatch.setenv('HARBOR_MCP_INJECT', 'off')

    item = next(generator.generate_configs())

    assert item.config.task_config['mcp_servers'] == []


def test_legacy_generator_installs_dedicated_trajectory_tracker(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    task_dir = _write_task(tmp_path, 'alpha')
    with (task_dir / 'task.toml').open('a') as task_file:
        task_file.write('needs_agent_trajectory = true\n')
    monkeypatch.setenv('HARBOR_DATASET_PATH', str(task_dir))
    monkeypatch.setenv('GONGFENG_RUNTIME_TOKEN', 'test-token')

    item = next(generator.generate_configs())

    processor = item.config.processors['harbor_agent_trajectory']
    assert processor.name == 'harbor_agent_trajectory_tracker'
    assert processor.args == {'tools_mode': 'branch_last'}


@pytest.mark.asyncio
async def test_dataset_item_matches_legacy_generator_task_config(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    task_dir = _write_task(tmp_path, 'alpha')
    with (task_dir / 'task.toml').open('a') as task_file:
        task_file.write('needs_agent_trajectory = true\n')
    monkeypatch.setenv('HARBOR_DATASET_PATH', str(task_dir))
    monkeypatch.setenv('HARBOR_DATASET_NAME', 'shared-benchmark')
    monkeypatch.setenv('GONGFENG_RUNTIME_TOKEN', 'test-token')
    monkeypatch.delenv('TASK_NAMES', raising=False)
    monkeypatch.delenv('HARBOR_TASK_INCLUDE', raising=False)
    legacy = next(generator.generate_configs())
    reader = HarborDatasetReader(
        DatasetSourceURI(
            scheme='t3-harbor',
            path=str(tmp_path),
            query={'flat': ('true',), 'name': ('shared-benchmark',)},
            raw=f't3-harbor://{tmp_path}',
        )
    )
    async with reader:
        dataset_item = (await load_items(reader, item_id='alpha'))[0]
    legacy_task_config = dict(legacy.config.task_config)
    legacy_task_config.pop('mode')
    # The Reader carries the task directory as DatasetItem.files so the
    # serialized task config remains portable across submitter and worker.
    legacy_task_config.pop('task_dir', None)

    assert dataset_item.body['task_config'] == legacy_task_config
    assert dataset_item.body['runtime_configs'] == legacy.config.runtime_configs
    assert dataset_item.body['processors'] == legacy.config.model_dump()['processors']
    assert dataset_item.body['run_id'] == legacy.run_id
    legacy_metadata = dict(legacy.metadata)
    legacy_metadata.pop('cache_task_id', None)
    legacy_metadata.pop('source_path', None)
    assert dataset_item.metadata == legacy_metadata


@pytest.mark.asyncio
async def test_compose_source_ignores_legacy_timeout_multiplier(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    _write_task(tmp_path, 'alpha')
    monkeypatch.setenv('TIMEOUT_MULTIPLIER', '10')

    item = (await _load_reader_items(tmp_path, item_id='alpha', flat='true'))[0]

    assert item.body['task_config']['agent_timeout'] == 7200
    assert item.body['task_config']['test_timeout'] == 1800
    assert 'TIMEOUT_MULTIPLIER' not in item.metadata.get('compile_env', {})


@pytest.mark.asyncio
async def test_t3_harbor_source_filters_and_validates_split(tmp_path: Path) -> None:
    _write_task(tmp_path, 'alpha')
    _write_task(tmp_path, 'beta')
    reader = HarborDatasetReader(
        DatasetSourceURI(
            scheme='t3-harbor',
            path=str(tmp_path),
            query={
                'flat': ('true',),
                'include': ('alpha,beta',),
                'exclude': ('beta',),
            },
            raw=f't3-harbor://{tmp_path}',
        )
    )

    async with reader:
        assert await list_item_ids(reader) == ['alpha']
        with pytest.raises(ValueError, match='only the test split'):
            await load_items(reader, split='train')


@pytest.mark.asyncio
async def test_recursive_discovery_covers_mixed_depth_trees(tmp_path: Path) -> None:
    _write_task(tmp_path, 'top-level')
    _write_task(tmp_path / 'category', 'deep-task')

    assert await _reader_item_ids(tmp_path) == ['deep-task', 'top-level']
    assert await _reader_item_ids(tmp_path, flat='true') == ['top-level']


@pytest.mark.asyncio
async def test_duplicate_nested_harbor_item_ids_use_reader_error_code(tmp_path: Path) -> None:
    _write_task(tmp_path / 'category-a', 'same')
    _write_task(tmp_path / 'category-b', 'same')

    with pytest.raises(DatasetReaderError) as duplicate_ids:
        await _reader_item_ids(tmp_path)

    assert duplicate_ids.value.code == 'dataset_duplicate_item_id'


@pytest.mark.asyncio
async def test_discovery_does_not_descend_into_task_dirs(tmp_path: Path) -> None:
    task_dir = _write_task(tmp_path / 'category', 'deep-task')
    fixtures = task_dir / 'tests' / 'fixtures'
    fixtures.mkdir(parents=True)
    (fixtures / 'task.toml').write_text((task_dir / 'task.toml').read_text())

    assert await _reader_item_ids(tmp_path) == ['deep-task']


@pytest.mark.asyncio
async def test_reopening_reader_refreshes_live_directory_discovery(tmp_path: Path) -> None:
    _write_task(tmp_path, 'alpha')
    reader = _open_reader(tmp_path, flat='true')

    async with reader:
        assert await list_item_ids(reader) == ['alpha']

    _write_task(tmp_path, 'beta')
    async with reader:
        assert await list_item_ids(reader) == ['alpha', 'beta']


@pytest.mark.asyncio
async def test_unknown_query_parameters_are_rejected(tmp_path: Path) -> None:
    _write_task(tmp_path, 'alpha')

    with pytest.raises(ValueError, match=r"does not accept parameters: \['flatt'\]"):
        _open_reader(tmp_path, flatt='true')
    with pytest.raises(ValueError, match='workers must be at least 1'):
        _open_reader(tmp_path, workers='0')
    with pytest.raises(ValueError, match='compile_workers must be at least 1'):
        _open_reader(tmp_path, compile_workers='0')
    async with _open_reader(tmp_path, split='test') as reader:
        assert await list_item_ids(reader) == ['alpha']
    async with _open_reader(tmp_path, split='train') as reader:
        with pytest.raises(ValueError, match='only the test split'):
            await list_item_ids(reader)


@pytest.mark.asyncio
async def test_skip_invalid_warns_for_every_dropped_task(
    tmp_path: Path,
    caplog: pytest.LogCaptureFixture,
) -> None:
    _write_task(tmp_path, 'alpha')
    broken = _write_broken_task(tmp_path, 'broken')

    with caplog.at_level('WARNING'):
        items = await _load_reader_items(tmp_path, flat='true', skip_invalid='true')

    assert [item.id for item in items] == ['alpha']
    assert f'Invalid T3-Harbor task {broken}' in caplog.text
    assert 'Skipped 1 invalid T3-Harbor task(s)' in caplog.text


@pytest.mark.asyncio
async def test_invalid_tasks_are_reported_together(tmp_path: Path, caplog: pytest.LogCaptureFixture) -> None:
    _write_task(tmp_path, 'alpha')
    bad_toml = _write_broken_task(tmp_path, 'bad-toml')
    empty_instruction = _write_task(tmp_path, 'empty-instruction')
    (empty_instruction / 'instruction.md').write_text('')

    with (
        caplog.at_level('WARNING'),
        pytest.raises(DatasetReaderError, match='Found 2 invalid T3-Harbor task') as exc_info,
    ):
        await _load_reader_items(tmp_path, flat='true')

    assert exc_info.value.code == 'dataset_item_invalid'
    message = str(exc_info.value)
    assert str(bad_toml) in message
    assert str(empty_instruction) in message
    assert 'skip_invalid=true' in message
    # Each failure is also logged while compiling, so the first bad task shows up immediately.
    assert f'Invalid T3-Harbor task {bad_toml}' in caplog.text


@pytest.mark.asyncio
async def test_all_invalid_tasks_fail_even_when_skipping(tmp_path: Path) -> None:
    broken = _write_broken_task(tmp_path, 'broken')

    with pytest.raises(DatasetReaderError, match='No valid T3-Harbor tasks were loaded') as exc_info:
        await _load_reader_items(tmp_path, flat='true', skip_invalid='true')

    assert exc_info.value.code == 'dataset_item_invalid'
    assert str(broken) in str(exc_info.value)


@pytest.mark.asyncio
async def test_run_id_collisions_after_sanitization_are_rejected(tmp_path: Path) -> None:
    dotted = _write_task(tmp_path, 'my.task')
    dashed = _write_task(tmp_path, 'my-task')
    (dashed / 'instruction.md').write_text((dotted / 'instruction.md').read_text())

    with pytest.raises(ValueError, match='collapse into the same batch run_id') as exc_info:
        await _reader_item_ids(tmp_path, flat='true')

    message = str(exc_info.value)
    assert str(dotted) in message
    assert str(dashed) in message


@pytest.mark.asyncio
async def test_punctuation_variants_with_distinct_canonical_content_are_kept(tmp_path: Path) -> None:
    original = _write_task(tmp_path, 'my.task')
    variant = _write_task(tmp_path, 'my-task')
    (variant / 'instruction.md').write_text((original / 'instruction.md').read_text())
    (variant / 'task.toml').write_text((variant / 'task.toml').read_text() + '\n# variant\n')

    assert await _reader_item_ids(tmp_path, flat='true') == ['my-task', 'my.task']


@pytest.mark.asyncio
async def test_limit_counts_loadable_tasks(tmp_path: Path) -> None:
    _write_broken_task(tmp_path, 'a-broken')
    _write_task(tmp_path, 'b-valid')
    _write_task(tmp_path, 'c-valid')

    items = await _load_reader_items(
        tmp_path,
        limit=2,
        flat='true',
        skip_invalid='true',
        compile_workers='3',
    )

    assert [item.id for item in items] == ['b-valid', 'c-valid']
