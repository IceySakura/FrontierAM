from __future__ import annotations

import hashlib
import json
import os
import zipfile

from pathlib import Path

import pytest

from nexus.dataset.access import DataAccessContext, bind_data_access, credentials_from_env
from nexus.dataset.build import DatasetBuilder
from nexus.dataset.cos import CosDatasetRef, CosPublisher, PublishOutcomeUnknownError
from nexus.dataset.item import CosFileRef, DatasetItem, RemoteDatasetRef, TaskFilesSpec
from nexus.dataset.preparation import TaskFilesMaterializer
from nexus.dataset.reader import open_dataset
from nexus.dataset.records import ImportSpec, JsonlRecordReader, RecordAdapter, RecordDatasetReader
from nexus.dataset.task_files import RuntimeTaskFiles, task_files_from_spec
from nexus.distributed.batch.materialize import materialize_compose
from nexus.runner.compose.model import RawRunPart
from nexus.utils.cos import CosClientError, CosConfigurationError, CosObject, CosServiceError


class MemoryCos:
    def __init__(self):
        self.objects = {}
        self.versioning = None
        self.lose_manifest_response = False
        self.upload_exception_type = TimeoutError
        self.reads_fail = False
        self.reads = []

    async def bucket_versioning(self):
        return self.versioning

    async def upload_file(self, path, key, *, headers):
        assert headers == {'x-cos-forbid-overwrite': 'true'}
        if key in self.objects:
            raise CosServiceError(method='PUT', key=key, status=409)
        self.objects[key] = Path(path).read_bytes()
        if key.endswith('manifest.json') and self.lose_manifest_response:
            raise self.upload_exception_type('response lost')

    async def stat(self, key):
        if self.reads_fail and key.endswith('manifest.json'):
            raise TimeoutError('read unavailable')
        if key not in self.objects:
            raise CosServiceError(method='HEAD', key=key, status=404)
        raw = self.objects[key]
        return CosObject(key=key, size=len(raw), etag='"' + hashlib.sha256(raw).hexdigest() + '"')

    async def read_range(self, key, start, end, *, etag, size):
        metadata = await self.stat(key)
        assert metadata.etag == etag
        assert metadata.size == size
        self.reads.append((key, start, end))
        return self.objects[key][start:end]

    async def close(self):
        pass


@pytest.fixture
def access():
    context = DataAccessContext(credentials={'cos': {'secret_id': 'test-id', 'secret_key': 'test-secret'}})
    client = MemoryCos()
    context._clients[('test-bucket', 'ap-guangzhou')] = client
    return context, client


def write_input(tmp_path, *, bundle='case-01', kind='directory'):
    source = tmp_path / 'source'
    source.mkdir(exist_ok=True)
    directory = source / 'case-01'
    directory.mkdir(exist_ok=True)
    (directory / 'abcd').write_text('portable data', encoding='utf-8')
    (directory / 'run.sh').write_text('#!/bin/sh\necho portable\n', encoding='utf-8')
    (directory / 'run.sh').chmod(0o755)
    path = source / 'tasks.jsonl'
    path.write_text(json.dumps({'id': 'case-01', 'input_file': 'case-01/abcd', 'bundle': bundle}) + '\n')
    spec = ImportSpec.model_validate({
        'record_kind': 'task_input',
        'task_name': 'identity',
        'files': {'path_field': 'bundle', 'type': kind},
        'compat': {'files_to_cwd': True},
    })
    return path, spec


async def build_input(tmp_path, context, *, output='build', **kwargs):
    path, spec = write_input(tmp_path, **kwargs)
    async with RecordDatasetReader(JsonlRecordReader(path), RecordAdapter(spec)) as reader:
        return await DatasetBuilder().build(reader, tmp_path / output, context)


async def test_jsonl_publish_remote_materialize_and_host_runtime_files_without_originals(tmp_path, access):
    context, client = access
    build = await build_input(tmp_path, context)
    published = await CosPublisher(store='cos://test-bucket/datasets', region='ap-guangzhou').publish(
        build,
        'research/demo@v1',
        context,
    )
    (tmp_path / 'source').rename(tmp_path / 'offline-source')
    async with open_dataset(published.data_uri, context=context) as reader:
        items = [item async for item in reader.iter_items()]
        assert (await reader.describe()).provider == 'cos'
    files = task_files_from_spec(items[0].resolved_task_files(), context=context)
    assert await files.read_text('abcd') == 'portable data'
    assert any(
        end - start < build.root.joinpath('objects', f'{items[0].files.sha256}.zip').stat().st_size
        for key, start, end in client.reads
        if key.endswith('.zip')
    )
    workspace = tmp_path / 'pwd'
    workspace.mkdir()
    lease = await TaskFilesMaterializer().prepare(files, files.spec, workspace)
    assert (workspace / 'case-01' / 'abcd').read_text() == 'portable data'
    assert (workspace / 'case-01' / 'run.sh').stat().st_mode & 0o111
    with pytest.raises(FileExistsError):
        await TaskFilesMaterializer().prepare(files, files.spec, workspace)
    lease.release()
    assert not (workspace / 'case-01').exists()
    with bind_data_access(context):
        result = await materialize_compose(
            [RawRunPart('data', published.data_uri, '--data')],
            tmp_path / 'tasks.jsonl',
            task_list_version=2,
            reject_local_task_files=True,
        )
    assert result.resolved_datasets[0]['resolved_uri'].endswith('#sha256=' + published.manifest_sha256)
    assert result.total_items == 1
    task_list = (tmp_path / 'tasks.jsonl').read_text()
    assert 'test-secret' not in task_list
    assert str(tmp_path / 'source') not in task_list
    # The runtime receives its own credential session and uses the same FileResolver.
    session = await files.create_session()
    runtime = RuntimeTaskFiles()
    runtime.set_session(session)
    from unittest.mock import patch

    with patch.object(DataAccessContext, 'cos_client', return_value=client):
        assert await runtime.read_text('abcd') == 'portable data'
    await files.close()


async def test_publish_is_idempotent_and_rejects_conflicting_release(tmp_path, access):
    context, client = access
    build = await build_input(tmp_path, context)
    publisher = CosPublisher(store='cos://test-bucket/datasets', region='ap-guangzhou')
    first = await publisher.publish(build, 'research/demo@v1', context)
    assert (await publisher.publish(build, 'research/demo@v1', context)) == first
    raw = tmp_path / 'source' / 'tasks.jsonl'
    raw.write_text(raw.read_text().replace('case-01/abcd', 'case-01/run.sh'))
    spec = ImportSpec(record_kind='task_input', task_name='identity', files={'path_field': 'bundle'})
    async with open_dataset(raw, options={'import_spec': spec}) as reader:
        different = await DatasetBuilder().build(reader, tmp_path / 'different', context)
    with pytest.raises(FileExistsError):
        await publisher.publish(different, 'research/demo@v1', context)
    ref = CosDatasetRef.parse(first.data_uri)
    assert hashlib.sha256(client.objects[ref.file_ref().key]).hexdigest() == first.manifest_sha256


@pytest.mark.parametrize('error_type', [TimeoutError, CosClientError])
async def test_manifest_response_loss_readback_and_unknown_retry(tmp_path, access, error_type):
    context, client = access
    build = await build_input(tmp_path, context)
    publisher = CosPublisher(store='cos://test-bucket/datasets', region='ap-guangzhou')
    client.lose_manifest_response = True
    client.upload_exception_type = error_type
    published = await publisher.publish(build, 'research/demo@v1', context)
    assert published.item_count == 1
    client.reads_fail = True
    with pytest.raises(PublishOutcomeUnknownError):
        await publisher.publish(build, 'research/demo@v2', context)
    client.reads_fail = False
    assert (await publisher.publish(build, 'research/demo@v2', context)).item_count == 1


async def test_published_manifest_tampering_and_versioning_rejected(tmp_path, access):
    context, client = access
    build = await build_input(tmp_path, context)
    publisher = CosPublisher(store='cos://test-bucket/datasets', region='ap-guangzhou')
    client.versioning = 'Enabled'
    with pytest.raises(ValueError, match='versioning'):
        await publisher.publish(build, 'research/demo@v1', context)
    assert not client.objects
    client.versioning = None
    published = await publisher.publish(build, 'research/demo@v1', context)
    key = CosDatasetRef.parse(published.data_uri).file_ref().key
    client.objects[key] += b' '
    with pytest.raises(ValueError, match='SHA256'):
        async with open_dataset(published.data_uri, context=context):
            pass


async def test_remote_zip_is_copied_and_origin_does_not_override_explicit_source(tmp_path, access):
    context, client = access
    build = await build_input(tmp_path, context)
    publisher = CosPublisher(store='cos://test-bucket/datasets', region='ap-guangzhou')
    published = await publisher.publish(build, 'research/demo@v1', context)
    async with open_dataset(published.data_uri, context=context) as reader:
        items = [item async for item in reader.iter_items()]
        copied = await DatasetBuilder().build(reader, tmp_path / 'copied', context)
    item = items[0]
    item.remote_ref = RemoteDatasetRef(namespace='old', dataset='origin', version='v1')
    assert item.resolved_task_files().source == item.files.source
    second = await publisher.publish(copied, 'research/copied@v1', context)
    for key in list(client.objects):
        if '/demo/' in key:
            del client.objects[key]
    async with open_dataset(second.data_uri, context=context) as reader:
        item = next(iter([item async for item in reader.iter_items()]))
    files = task_files_from_spec(item.files, context=context)
    assert await files.read_text('abcd') == 'portable data'


async def test_ordinary_zip_normalizes_single_root_and_rejects_traversal(tmp_path, access):
    context, _ = access
    path, spec = write_input(tmp_path, bundle='plain.zip', kind='zip')
    archive = path.parent / 'plain.zip'
    with zipfile.ZipFile(archive, 'w') as output:
        output.writestr('case-01/abcd', 'portable data')
    async with open_dataset(path, options={'import_spec': spec}) as reader:
        build = await DatasetBuilder().build(reader, tmp_path / 'build', context)
    record = next(build.iter_records('test'))
    assert record['files']['root_name'] == 'case-01'
    with zipfile.ZipFile(build.root / record['files']['source']['path']) as output:
        assert output.namelist() == ['abcd']
    with zipfile.ZipFile(archive, 'w') as output:
        output.writestr('../escape', 'bad')
    async with open_dataset(path, options={'import_spec': spec}) as reader:
        with pytest.raises(ValueError, match='Unsafe ZIP'):
            await DatasetBuilder().build(reader, tmp_path / 'unsafe', context)


async def test_directory_hash_ignores_timestamps_and_reports_duplicate_line(tmp_path, access):
    context, _ = access
    first = await build_input(tmp_path, context)
    path, spec = write_input(tmp_path)
    os.utime(path.parent / 'case-01' / 'abcd', (1000000000, 1000000000))
    async with open_dataset(path, options={'import_spec': spec}) as reader:
        second = await DatasetBuilder().build(reader, tmp_path / 'second', context)
    assert first.content_sha256 == second.content_sha256
    path.write_text(path.read_text() * 2)
    async with open_dataset(path, options={'import_spec': spec}) as reader:
        with pytest.raises(ValueError, match='tasks.jsonl:2'):
            _ = [item async for item in reader.iter_items()]


@pytest.mark.parametrize('kind', ['task_input', 'run_input', 'dataset_item'])
async def test_record_modes(tmp_path, kind):
    path = tmp_path / 'items.jsonl'
    record = {'id': 'one', 'message': 'hello'}
    if kind == 'run_input':
        record = {'id': 'one', 'task_name': 'identity', 'task_config': {'message': 'hello'}}
    elif kind == 'dataset_item':
        record = {'id': 'one', 'body': {'task_name': 'identity', 'task_config': {'message': 'hello'}}}
    path.write_text(json.dumps(record) + '\n')
    spec = ImportSpec(record_kind=kind, task_name='identity')
    async with open_dataset(path, options={'import_spec': spec}) as reader:
        item = next(iter([item async for item in reader.iter_items()]))
    assert item.id == 'one'
    expected = {'id': 'one', 'message': 'hello'} if kind == 'task_input' else {'message': 'hello'}
    assert item.body['task_config'] == expected


def test_env_snapshot_is_complete_fresh_and_redacted(monkeypatch):
    for name in ('COS_SECRET_ID', 'COS_SECRET_KEY', 'COS_TOKEN', 'WEDATA_KEY'):
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setenv('UNRELATED_SECRET', 'must-not-transfer')
    assert credentials_from_env() == {}
    monkeypatch.setenv('COS_SECRET_ID', 'test-id')
    with pytest.raises(CosConfigurationError, match='secret_key'):
        credentials_from_env()
    monkeypatch.setenv('COS_SECRET_KEY', 'first')
    assert credentials_from_env()['cos']['secret_key'] == 'first'
    monkeypatch.setenv('COS_SECRET_KEY', 'second')
    context = DataAccessContext.from_env()
    assert context.credentials['cos']['secret_key'] == 'second'
    assert 'second' not in repr(context)


def test_cos_reference_requires_region_and_expected_hash():
    for value in ('cos://test-bucket/manifest.json', 'cos://test-bucket/manifest.json?region=ap-guangzhou'):
        with pytest.raises(ValueError, match='requires region'):
            CosDatasetRef.parse(value)
    item = DatasetItem(
        id='one',
        body={},
        files=TaskFilesSpec(
            type='zip',
            source=CosFileRef(
                bucket='test-bucket',
                region='ap-guangzhou',
                key='files.zip',
            ),
        ),
    )
    assert item.local_task_files_path() is None


def test_task_access_context_does_not_fall_back_to_process_taiji_credentials(monkeypatch):
    monkeypatch.setenv('TAI_PAT', 'unrelated-service-credential')
    with bind_data_access(DataAccessContext()):
        with pytest.raises(ValueError, match='Taiji credentials are missing'):
            open_dataset('namespace/dataset@v1')


async def test_build_rejects_output_inside_attached_directory(tmp_path, access):
    context, _ = access
    path, spec = write_input(tmp_path)
    async with open_dataset(path, options={'import_spec': spec}) as reader:
        with pytest.raises(ValueError, match='must not be inside'):
            await DatasetBuilder().build(reader, path.parent / 'case-01' / 'build', context)


@pytest.mark.parametrize('provider', ['jsonl', 'cos'])
async def test_new_readers_preserve_scoped_reader_contract(tmp_path, access, provider):
    from tests.unit.dataset.test_reader_contract import _assert_reader_contract

    context, _ = access
    path = tmp_path / 'records.jsonl'
    path.write_text(
        ''.join(json.dumps({'id': f'case-{i}', 'body': {'task_name': 'identity'}}) + '\n' for i in range(3))
    )
    if provider == 'cos':
        async with open_dataset(path) as source:
            build = await DatasetBuilder().build(source, tmp_path / 'build', context)
        ref = await CosPublisher(store='cos://test-bucket/data', region='ap-guangzhou').publish(
            build,
            'test/contract@v1',
            context,
        )
        reader = open_dataset(ref.data_uri, context=context)
    else:
        reader = open_dataset(path)
    await _assert_reader_contract(reader, provider=provider, expected_ids=['case-0', 'case-1', 'case-2'])
    assert reader._active_provider_iterators == []


async def test_record_single_lookup_still_rejects_later_duplicate_ids(tmp_path):
    path = tmp_path / 'records.jsonl'
    path.write_text((json.dumps({'id': 'duplicate', 'body': {}}) + '\n') * 2)
    async with open_dataset(path) as reader:
        with pytest.raises(ValueError, match='Duplicate item id'):
            _ = [item async for item in reader.iter_items(item_id='duplicate')]


def test_harbor_cli_json_does_not_mix_registry_logs_into_stdout(tmp_path):
    import subprocess
    import sys

    case = tmp_path / 'case'
    case.mkdir()
    (case / 'task.toml').write_text('[environment]\ndocker_image = "fixture:v1"\n')
    result = subprocess.run(
        [sys.executable, '-m', 'nexus.cli.main', 'dataset', 'show', f't3-harbor://{tmp_path}?flat=true', '--json'],
        capture_output=True,
        text=True,
        check=True,
    )
    payload = json.loads(result.stdout)
    assert payload['provider'] == 't3-harbor'
    assert 'registrations' in result.stderr


@pytest.mark.parametrize(('kind', 'field'), [('task_input', 'instance_id'), ('run_input', 'run_id')])
async def test_import_preserves_selected_business_identity_field(tmp_path, kind, field):
    path = tmp_path / 'records.jsonl'
    record = {field: 'case-01', 'task_name': 'identity'}
    path.write_text(json.dumps(record) + '\n')
    async with open_dataset(
        path,
        options={
            'import_spec': {
                'record_kind': kind,
                'id_field': field,
                'task_name': 'identity',
            }
        },
    ) as reader:
        item = next(iter([item async for item in reader.iter_items()]))
    assert item.id == 'case-01'
    assert (item.body['task_config'] if kind == 'task_input' else item.body)[field] == 'case-01'


async def test_generated_ids_retain_input_digest_and_line_without_local_path(tmp_path):
    path = tmp_path / 'records.jsonl'
    path.write_text(json.dumps({'message': 'hello'}) + '\n')
    async with open_dataset(
        path,
        options={
            'import_spec': {
                'record_kind': 'task_input',
                'task_name': 'identity',
                'generate_ids': True,
            }
        },
    ) as reader:
        item = next(iter([item async for item in reader.iter_items()]))
    assert item.id == 'line-00000001'
    assert item.metadata['import_record'] == {'source_sha256': hashlib.sha256(path.read_bytes()).hexdigest(), 'line': 1}
    assert str(path) not in item.model_dump_json()


async def test_taiji_existing_release_is_idempotent_and_conflicting_content_is_rejected(tmp_path, access, monkeypatch):
    from unittest.mock import AsyncMock

    from nexus.dataset import publishing
    from nexus.dataset.reader import BaseDatasetReader, DatasetDescriptor, DatasetSplitDescriptor

    context, _ = access
    path = tmp_path / 'records.jsonl'
    path.write_text(
        ''.join(json.dumps({'id': value, 'body': {'task_name': 'identity'}}) + '\n' for value in ('b', 'a'))
    )
    async with open_dataset(path) as reader:
        build = await DatasetBuilder().build(reader, tmp_path / 'build', context)
    actual = [DatasetItem.model_validate(record) for record in build.iter_records('test')][::-1]

    class ExistingRelease(BaseDatasetReader):
        async def describe(self):
            return DatasetDescriptor(
                reference='test/data@v1',
                resolved_reference='test/data@v1',
                provider='taiji',
                name='data',
                default_split='test',
                splits=[DatasetSplitDescriptor(name='test')],
            )

        async def iter_items(self, **_kwargs):
            for item in actual:
                yield item

    upload = AsyncMock(side_effect=AssertionError('An existing release must not be uploaded again'))
    monkeypatch.setattr(publishing, 'open_dataset', lambda *_args, **_kwargs: ExistingRelease())
    monkeypatch.setattr(publishing, 'publish_local_dataset', upload)
    publisher = publishing.TaijiPublisher()
    result = await publisher.publish(build, 'test/data@v1', context)
    assert result.item_count == 2
    upload.assert_not_awaited()
    actual[0].body['task_name'] = 'different'
    with pytest.raises(FileExistsError, match='different content'):
        await publisher.publish(build, 'test/data@v1', context)
    upload.assert_not_awaited()


@pytest.mark.parametrize('part', ['records', 'attachment'])
async def test_publish_rejects_mutated_frozen_build_before_upload(tmp_path, access, part):
    context, client = access
    build = await build_input(tmp_path, context)
    if part == 'records':
        path = build.root / 'records/test.jsonl'
        path.write_text(path.read_text().replace('identity', 'changed'))
    else:
        path = next((build.root / 'objects').glob('*.zip'))
        path.write_bytes(path.read_bytes() + b'changed')
    with pytest.raises(ValueError, match='Frozen dataset'):
        await CosPublisher(store='cos://test-bucket/data', region='ap-guangzhou').publish(
            build, 'test/data@v1', context
        )
    assert client.objects == {}


async def test_build_verifies_input_zip_entry_checksums(tmp_path, access):
    from nexus.dataset.zip_index import read_zip_index_from_file, write_nexus_zip_directory_to_file

    context, _ = access
    source = tmp_path / 'files'
    source.mkdir()
    (source / 'data').write_text('original data')
    archive = tmp_path / 'files.zip'
    write_nexus_zip_directory_to_file(source, archive)
    entry = read_zip_index_from_file(archive).entries['data']
    raw = bytearray(archive.read_bytes())
    raw[entry.data_offset] ^= 1
    archive.write_bytes(raw)
    records = tmp_path / 'records.jsonl'
    records.write_text(json.dumps({'id': 'one', 'body': {}, 'files': {'type': 'zip', 'path': 'files.zip'}}) + '\n')
    async with open_dataset(records) as reader:
        with pytest.raises(ValueError, match='CRC mismatch'):
            await DatasetBuilder().build(reader, tmp_path / 'build', context)


async def test_flat_zip_can_map_root_name_from_a_record_field(tmp_path, access):
    context, _ = access
    path, spec = write_input(tmp_path, bundle='flat.zip', kind='zip')
    spec.files.root_name_field = 'id'
    with zipfile.ZipFile(path.parent / 'flat.zip', 'w') as archive:
        archive.writestr('abcd', 'portable data')
    async with open_dataset(path, options={'import_spec': spec}) as reader:
        build = await DatasetBuilder().build(reader, tmp_path / 'build', context)
    item = next(build.iter_records('test'))
    assert item['files']['root_name'] == 'case-01'
    assert item['body']['task_config']['id'] == 'case-01'


async def test_default_raw_records_preserve_business_fields_and_generate_missing_ids(tmp_path):
    records = [
        {'id': 'case-01', 'prompt': 'hello', 'task_name': 'business-value', 'image': 'not-an-attachment.png'},
        {'prompt': 'world', 'metadata': {'user': 'value'}, 'files': ['ordinary-business-list']},
        {'id': 'case-03', 'body': {'text': 'business-body'}, 'extra': True},
    ]
    path = tmp_path / 'raw.jsonl'
    path.write_text(''.join(json.dumps(record) + '\n' for record in records))
    async with open_dataset(path) as reader:
        items = [item async for item in reader.iter_items()]
    assert [item.id for item in items] == ['case-01', 'line-00000002', 'case-03']
    assert [item.body for item in items] == [{'task_config': record} for record in records]
    assert all(item.files is None for item in items)
    assert items[1].metadata['import_record'] == {
        'source_sha256': hashlib.sha256(path.read_bytes()).hexdigest(),
        'line': 2,
    }
    assert 'import_record' not in items[0].metadata


async def test_default_standard_record_keeps_body_metadata_and_files(tmp_path):
    directory = tmp_path / 'case-01'
    directory.mkdir()
    path = tmp_path / 'items.jsonl'
    record = {
        'id': 'case-01',
        'body': {'task_config': {'prompt': 'hello'}},
        'metadata': {'tag': 'standard'},
        'merge': {'task_config': 'merge'},
        'files': {'type': 'directory', 'path': 'case-01'},
    }
    path.write_text(json.dumps(record) + '\n')
    async with open_dataset(path) as reader:
        item = [item async for item in reader.iter_items()][0]
    assert item.body == record['body']
    assert item.metadata == record['metadata']
    assert item.merge == record['merge']
    assert item.local_task_files_path() == directory
    assert item.files.root_name == 'case-01'


@pytest.mark.parametrize('bad_id', [None, '', '  ', 12, False, [], {}])
async def test_default_present_invalid_id_does_not_generate_replacement(tmp_path, bad_id):
    path = tmp_path / 'raw.jsonl'
    path.write_text(json.dumps({'id': bad_id, 'prompt': 'hello'}) + '\n')
    async with open_dataset(path) as reader:
        with pytest.raises(ValueError, match='non-empty string ID'):
            _ = [item async for item in reader.iter_items()]


@pytest.mark.parametrize(
    'records',
    [
        [{'id': 'same'}, {'id': 'same'}],
        [{'id': 'line-00000002'}, {'prompt': 'no id'}],
    ],
)
async def test_default_duplicate_explicit_or_generated_id_is_rejected(tmp_path, records):
    path = tmp_path / 'raw.jsonl'
    path.write_text(''.join(json.dumps(record) + '\n' for record in records))
    async with open_dataset(path) as reader:
        with pytest.raises(ValueError, match='dataset_duplicate_item_id'):
            _ = [item async for item in reader.iter_items()]


async def test_field_overrides_keep_spec_settings_and_require_selected_id(tmp_path):
    from nexus.dataset.records import resolve_import_spec

    directory = tmp_path / 'attachments'
    directory.mkdir()
    spec_path = tmp_path / 'import.yaml'
    spec_path.write_text(
        'record_kind: task_input\nid_field: old_id\ngenerate_ids: true\nsplit: train\n'
        'files:\n  path_field: old_bundle\n  type: directory\n  root_name_field: root\n'
    )
    spec = resolve_import_spec(spec_path, id_field='instance_id', files_field='bundle')
    assert spec.split == 'train' and spec.record_kind == 'task_input'
    assert spec.files.type == 'directory' and spec.files.root_name_field == 'root'
    assert spec.generate_ids is False
    path = tmp_path / 'raw.jsonl'
    record = {'instance_id': 'one', 'bundle': 'attachments', 'root': 'case-01', 'prompt': 'hello'}
    path.write_text(json.dumps(record) + '\n')
    options = {'import_spec': spec_path, 'id_field': 'instance_id', 'files_field': 'bundle'}
    async with open_dataset(path, options=options) as reader:
        item = [item async for item in reader.iter_items(split='train')][0]
    assert item.id == 'one'
    assert item.body == {'task_config': {'instance_id': 'one', 'root': 'case-01', 'prompt': 'hello'}}
    assert item.files.root_name == 'case-01'
    del record['instance_id']
    path.write_text(json.dumps(record) + '\n')
    async with open_dataset(path, options=options) as reader:
        with pytest.raises(ValueError, match='non-empty string ID'):
            _ = [item async for item in reader.iter_items()]


@pytest.mark.parametrize(
    'options',
    [
        {'id_field': ''},
        {'files_field': ''},
        {'id_field': 'bundle', 'files_field': 'bundle'},
    ],
)
def test_invalid_field_options_are_rejected(tmp_path, options):
    with pytest.raises(ValueError):
        open_dataset(tmp_path / 'raw.jsonl', options=options)


@pytest.mark.parametrize('options', [{'id_field': 'id'}, {'files_field': 'bundle'}, {'import_spec': {}}])
def test_import_options_are_not_silently_ignored_for_other_sources(tmp_path, options):
    with pytest.raises(ValueError, match='local .jsonl'):
        open_dataset(tmp_path, options=options)


async def test_default_raw_publish_then_compose_supplies_task_name(tmp_path, access):
    from nexus.runner.compose.compiler import PreparedRunParts

    context, _client = access
    path = tmp_path / 'raw.jsonl'
    path.write_text('{"prompt":"hello"}\n')
    async with open_dataset(path) as reader:
        build = await DatasetBuilder().build(reader, tmp_path / 'build', context)
    published = await CosPublisher(store='cos://test-bucket/datasets', region='ap-guangzhou').publish(
        build, 'research/raw@v1', context
    )
    path.unlink()
    async with open_dataset(published.data_uri, context=context) as reader:
        item = [item async for item in reader.iter_items()][0]
    assert item.body == {'task_config': {'prompt': 'hello'}}
    with pytest.raises(ValueError, match='task_name'):
        PreparedRunParts([]).compile(item, log_item=False)
    run = PreparedRunParts([
        RawRunPart(kind='input', value={'task_name': 'identity'}, source='profile'),
    ]).compile(item, log_item=False)
    assert run.task_name == 'identity' and run.task_config == {'prompt': 'hello'}


@pytest.mark.parametrize('command', ['import', 'publish'])
def test_cli_default_raw_mapping_and_mixed_bundle_types(tmp_path, monkeypatch, capsys, command):
    import argparse

    from nexus.cli.commands.dataset import dataset_import, dataset_publish, register_dataset_parser

    monkeypatch.setattr(DataAccessContext, 'from_env', classmethod(lambda cls: cls()))
    directory = tmp_path / 'case-dir.zip'
    directory.mkdir()
    (directory / 'abcd').write_text('directory data')
    archive = tmp_path / 'bundle.zip'
    with zipfile.ZipFile(archive, 'w') as output:
        output.writestr('case-zip/abcd', 'archive data')
    path = tmp_path / 'tasks.JSONL'
    path.write_text(
        json.dumps({'instance_id': 'directory', 'bundle': directory.name, 'prompt': 'first'})
        + '\n'
        + json.dumps({'instance_id': 'zip', 'bundle': archive.name, 'prompt': 'second'})
        + '\n'
    )
    parser = argparse.ArgumentParser()
    register_dataset_parser(parser.add_subparsers(dest='command'))
    output = tmp_path / 'output'
    arguments = [
        'dataset',
        command,
        str(path),
        '--id-field',
        'instance_id',
        '--files-field',
        'bundle',
        '--files-to-cwd',
        '--output',
        str(output),
        '--json',
    ]
    if command == 'publish':
        arguments += ['research/mixed@v1', '--build-only']
    args = parser.parse_args(arguments)
    assert (dataset_import if command == 'import' else dataset_publish)(args) == 0
    assert json.loads(capsys.readouterr().out)['item_count'] == 2

    async def verify():
        if command == 'import':
            async with open_dataset(output) as reader:
                items = [item async for item in reader.iter_items()]
        else:
            records = [json.loads(line) for line in (output / 'records' / 'test.jsonl').read_text().splitlines()]
            items = [DatasetItem.from_mapping(record, source_path=output / 'dataset.yaml') for record in records]
        assert [item.id for item in items] == ['directory', 'zip']
        assert [item.files.root_name for item in items] == ['case-dir.zip', 'case-zip']
        for item, expected in zip(items, ['directory data', 'archive data']):
            assert item.body['task_config']['instance_id'] == item.id
            assert 'bundle' not in item.body['task_config'] and 'task_name' not in item.body
            assert item.body['runner']['task_files_materialization'] == {'mode': 'cwd', 'target': 'host'}
            files = task_files_from_spec(item.resolved_task_files(), context=DataAccessContext())
            try:
                assert await files.read_text('abcd') == expected
            finally:
                await files.close()

    import asyncio

    asyncio.run(verify())


@pytest.mark.parametrize(('record', 'options'), [
    ({'id': 'one', 'body': 'business text'}, {}),
    ({'id': 'one', 'body': {'prompt': 'business object'}}, {'import_spec': {'record_kind': 'task_input'}}),
])
async def test_business_body_field_can_be_preserved(tmp_path, record, options):
    path = tmp_path / 'raw.jsonl'
    path.write_text(json.dumps(record) + '\n')
    async with open_dataset(path, options=options) as reader:
        item = [item async for item in reader.iter_items()][0]
    assert item.body == {'task_config': record}


async def test_standard_record_accepts_auto_files_field_and_custom_id(tmp_path):
    (tmp_path / 'attachments').mkdir()
    path = tmp_path / 'items.jsonl'
    path.write_text('{"instance_id":"one","body":{"task_config":{"prompt":"hello"}},"bundle":"attachments"}\n')
    async with open_dataset(path, options={'id_field': 'instance_id', 'files_field': 'bundle'}) as reader:
        item = [item async for item in reader.iter_items()][0]
    assert item.id == 'one' and item.body == {'task_config': {'prompt': 'hello'}}
    assert item.files.type == 'directory' and item.files.root_name == 'attachments'
