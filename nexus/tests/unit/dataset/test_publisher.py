import asyncio
import json
import zipfile

from dataclasses import replace

import pytest

from nexus.core import dep_checker
from nexus.dataset import publisher
from nexus.dataset.exceptions import DatasetClientError, UcAssetNotFoundError
from nexus.dataset.layout import create_dataset_template
from nexus.dataset.publisher import PublishBuild, PublishFile, build_publish_layout, resolve_with_parquet
from nexus.dataset.task_files import task_files_from_spec


class _FakePyArrow:
    @staticmethod
    def table(data):
        return data


class _FakeParquet:
    @staticmethod
    def write_table(table, buffer):
        buffer.write(json.dumps(table, sort_keys=True).encode('utf-8'))


class _FakeUcClient:
    def __init__(self, *, get_error: Exception | None = None, register_error: Exception | None = None):
        self.get_error = get_error
        self.register_error = register_error
        self.calls = []

    async def get_asset(self, namespace, dataset):
        self.calls.append(('get_asset', namespace, dataset))
        if self.get_error:
            raise self.get_error
        return {'FullName': f'taiji.nexus.{namespace}/{dataset}'}

    async def register_nexus_asset(self, namespace, dataset, **kwargs):
        self.calls.append(('register_nexus_asset', namespace, dataset, kwargs))
        if self.register_error:
            raise self.register_error
        return {'FullName': f'taiji.nexus.{namespace}/{dataset}'}


class _FakeProgress:
    def __init__(self):
        self.started = 0
        self.finished = 0
        self.closed = False

    def start_file(self):
        self.started += 1

    def finish_file(self, *, success):
        if success:
            self.finished += 1

    def close(self):
        self.closed = True


class _FakeIoClient:
    def __init__(self):
        self.active = 0
        self.max_active = 0
        self.calls = []

    async def upload_signed_local_file(self, signed_url, local_path, **kwargs):
        self.active += 1
        self.max_active = max(self.max_active, self.active)
        try:
            await asyncio.sleep(0.01)
            self.calls.append(('regular', signed_url, local_path, kwargs))
        finally:
            self.active -= 1

    async def upload_local_file(self, namespace, dataset, snapshot, local_path, **kwargs):
        self.active += 1
        self.max_active = max(self.max_active, self.active)
        try:
            await asyncio.sleep(0.01)
            self.calls.append(('multipart', namespace, dataset, snapshot, local_path, kwargs))
        finally:
            self.active -= 1


def _publish_build_for_upload(tmp_path, *, count=5):
    files = []
    upload_urls = {}
    for index in range(count):
        rel_path = f'test/raw/case-{index}.json'
        local_path = tmp_path / f'case-{index}.json'
        data = json.dumps({'id': f'case-{index}'}).encode('utf-8')
        local_path.write_bytes(data)
        files.append(
            PublishFile(
                path=rel_path,
                local_path=local_path,
                sha256='sha',
                size=len(data),
                content_type='application/json',
            )
        )
        upload_urls[rel_path] = f'https://upload/{index}'
    return (
        PublishBuild(
            namespace='research',
            dataset='terminal_cases',
            snapshot='dev-202607040001',
            release_version='v1',
            files_root=tmp_path,
            output_dir=tmp_path,
            files=files,
            default_split='test',
            item_count=count,
            sample_item='case-0',
        ),
        upload_urls,
    )


@pytest.mark.asyncio
async def test_build_publish_layout_packages_directory_task_files(tmp_path):
    dataset_root = create_dataset_template(
        tmp_path / 'my_terminal',
        template='terminal',
        item_id='case-001',
        name='terminal_cases',
    )
    output_dir = tmp_path / 'out'

    build = build_publish_layout(
        dataset_root,
        namespace='research',
        release_version='v1',
        output_dir=output_dir,
        snapshot='dev-202607040001',
    )

    assert build.namespace == 'research'
    assert build.dataset == 'terminal_cases'
    assert build.snapshot == 'dev-202607040001'
    assert build.item_count == 1

    raw = json.loads((build.files_root / 'test/raw/case-001.json').read_text())
    assert raw['files']['type'] == 'zip'
    assert raw['files']['path'] == 'blobs/case-001.zip'
    assert raw['files']['sha256']
    assert raw['files']['remote'] == {
        'namespace': 'research',
        'dataset': 'terminal_cases',
        'version': 'v1',
        'split': 'test',
    }
    assert 'task_dir' not in raw['body']['task_config']

    blob = build.files_root / 'test/blobs/case-001.zip'
    with zipfile.ZipFile(blob) as archive:
        assert 'instruction.md' in archive.namelist()
    files = task_files_from_spec({**raw['files'], 'path': str(blob), 'remote': None})
    assert files is not None
    assert await files.read_text('instruction.md')

    version = json.loads((build.files_root / '_meta/version.json').read_text())
    assert version['namespace'] == 'research'
    assert version['dataset'] == 'terminal_cases'
    assert version['default_split'] == 'test'
    assert version['splits']['test']['ids'] == ['case-001']

    upload_manifest = json.loads((build.files_root / '_meta/upload_manifest.json').read_text())
    manifest_paths = {item['path'] for item in upload_manifest['files']}
    assert 'test/raw/case-001.json' in manifest_paths
    assert 'test/blobs/case-001.zip' in manifest_paths
    assert '_meta/version.json' in manifest_paths
    assert '_meta/upload_manifest.json' in build.file_paths


def test_build_publish_layout_rejects_output_file(tmp_path):
    dataset_root = create_dataset_template(
        tmp_path / 'my_terminal',
        template='terminal',
        item_id='case-001',
        name='terminal_cases',
    )
    output_file = tmp_path / 'out'
    output_file.write_text('not a directory', encoding='utf-8')

    with pytest.raises(FileExistsError, match='not a directory'):
        build_publish_layout(
            dataset_root,
            namespace='research',
            release_version='v1',
            output_dir=output_file,
            snapshot='dev-202607040001',
        )


def test_build_publish_layout_overwrites_nonempty_output_dir_by_default(tmp_path):
    dataset_root = create_dataset_template(
        tmp_path / 'my_terminal',
        template='terminal',
        item_id='case-001',
        name='terminal_cases',
    )
    output_dir = tmp_path / 'out'
    output_dir.mkdir()
    stale = output_dir / 'stale.txt'
    stale.write_text('old build', encoding='utf-8')

    build = build_publish_layout(
        dataset_root,
        namespace='research',
        release_version='v1',
        output_dir=output_dir,
        snapshot='dev-202607040001',
    )

    assert build.output_dir == output_dir
    assert not stale.exists()
    assert (build.files_root / '_meta/version.json').is_file()


def test_build_publish_layout_can_reject_nonempty_output_dir(tmp_path):
    dataset_root = create_dataset_template(
        tmp_path / 'my_terminal',
        template='terminal',
        item_id='case-001',
        name='terminal_cases',
    )
    output_dir = tmp_path / 'out'
    output_dir.mkdir()
    (output_dir / 'stale.txt').write_text('old build', encoding='utf-8')

    with pytest.raises(FileExistsError, match='Output directory is not empty'):
        build_publish_layout(
            dataset_root,
            namespace='research',
            release_version='v1',
            output_dir=output_dir,
            snapshot='dev-202607040001',
            force=False,
        )


def test_build_publish_layout_rejects_duplicate_item_ids(tmp_path):
    dataset_root = create_dataset_template(
        tmp_path / 'my_terminal',
        template='terminal',
        item_id='case-001',
        name='terminal_cases',
    )
    dataset_yaml = dataset_root / 'dataset.yaml'
    dataset_yaml.write_text(
        """
schema_version: 1
name: terminal_cases
default_split: test
splits:
  test:
    items:
      - items/case-001/item.yaml
      - items/case-001/item.yaml
""".lstrip(),
        encoding='utf-8',
    )

    with pytest.raises(ValueError, match='Duplicate item id'):
        build_publish_layout(
            dataset_root,
            namespace='research',
            release_version='v1',
            output_dir=tmp_path / 'out',
            snapshot='dev-202607040001',
        )


def test_publish_parquet_default_is_disabled():
    assert resolve_with_parquet(None, build_only=False) is False
    assert resolve_with_parquet(None, build_only=True) is False
    assert resolve_with_parquet(False, build_only=False) is False
    assert resolve_with_parquet(True, build_only=True) is True


@pytest.mark.asyncio
async def test_ensure_publish_asset_uses_existing_asset():
    uc_client = _FakeUcClient()

    result = await publisher._ensure_publish_asset(
        uc_client,
        'research',
        'terminal_cases',
        creator=None,
        admin_list=None,
        visibility='PRIVATE',
        description=None,
    )

    assert result == {'FullName': 'taiji.nexus.research/terminal_cases'}
    assert uc_client.calls == [('get_asset', 'research', 'terminal_cases')]


@pytest.mark.asyncio
async def test_ensure_publish_asset_registers_missing_asset():
    uc_client = _FakeUcClient(get_error=UcAssetNotFoundError('UC asset not found (HTTP 404)', http_status=404))

    result = await publisher._ensure_publish_asset(
        uc_client,
        'research',
        'terminal_cases',
        creator='alice',
        admin_list=None,
        visibility='PRIVATE',
        description='desc',
    )

    assert result == {'FullName': 'taiji.nexus.research/terminal_cases'}
    assert uc_client.calls == [
        ('get_asset', 'research', 'terminal_cases'),
        (
            'register_nexus_asset',
            'research',
            'terminal_cases',
            {
                'creator': 'alice',
                'admin_list': ['alice'],
                'visibility': 'PRIVATE',
                'description': 'desc',
            },
        ),
    ]


@pytest.mark.asyncio
async def test_ensure_publish_asset_reraises_non_missing_error():
    uc_client = _FakeUcClient(get_error=DatasetClientError('UC HTTP 403: forbidden'))

    with pytest.raises(DatasetClientError, match='forbidden'):
        await publisher._ensure_publish_asset(
            uc_client,
            'research',
            'terminal_cases',
            creator='alice',
            admin_list=['alice'],
            visibility='PRIVATE',
            description=None,
        )


@pytest.mark.asyncio
async def test_ensure_publish_asset_surfaces_namespace_registration_error():
    namespace_error = DatasetClientError(
        'UC error: namespace 未注册，请先创建 namespace 权限记录',
        code='ResourceNotFound.MetaResourceNotFound',
        http_status=200,
    )
    uc_client = _FakeUcClient(
        get_error=UcAssetNotFoundError('asset missing', http_status=200),
        register_error=namespace_error,
    )

    with pytest.raises(DatasetClientError, match='namespace 未注册') as exc:
        await publisher._ensure_publish_asset(
            uc_client,
            'research',
            'terminal_cases',
            creator='alice',
            admin_list=['alice'],
            visibility='PRIVATE',
            description=None,
        )

    assert exc.value is namespace_error
    assert [call[0] for call in uc_client.calls] == ['get_asset', 'register_nexus_asset']


@pytest.mark.asyncio
async def test_upload_publish_files_respects_concurrency(tmp_path, monkeypatch):
    progress = _FakeProgress()
    monkeypatch.setattr(publisher, '_upload_progress_bar', lambda total: progress)
    build, upload_urls = _publish_build_for_upload(tmp_path, count=6)
    io_client = _FakeIoClient()

    uploaded = await publisher._upload_publish_files(
        io_client,
        build,
        upload_urls,
        concurrency=2,
        expires=3600,
    )

    assert uploaded == 6
    assert len(io_client.calls) == 6
    assert io_client.max_active <= 2
    assert all(call[0] == 'regular' for call in io_client.calls)
    assert {call[3]['key'] for call in io_client.calls} == set(build.file_paths)
    assert progress.started == 6
    assert progress.finished == 6
    assert progress.closed is True


@pytest.mark.asyncio
async def test_upload_publish_files_rejects_invalid_concurrency(tmp_path):
    build, upload_urls = _publish_build_for_upload(tmp_path, count=1)

    with pytest.raises(ValueError, match='at least 1'):
        await publisher._upload_publish_files(_FakeIoClient(), build, upload_urls, concurrency=0)


@pytest.mark.asyncio
async def test_upload_publish_files_uses_unified_multipart_for_large_file(tmp_path):
    build, _upload_urls = _publish_build_for_upload(tmp_path, count=1)
    large_file = build.files[0]
    build = replace(build, files=[replace(large_file, size=publisher.DATASET_MULTIPART_THRESHOLD)])
    io_client = _FakeIoClient()

    uploaded = await publisher._upload_publish_files(io_client, build, {}, concurrency=2, expires=3600)

    assert uploaded == 1
    assert io_client.calls == [
        (
            'multipart',
            'research',
            'terminal_cases',
            'dev-202607040001',
            large_file.local_path,
            {
                'key': large_file.path,
                'content_type': 'application/json',
                'expires': 3600,
                'max_concurrency': 2,
            },
        )
    ]


def test_concurrent_upload_progress_renders_three_segments():
    progress = publisher.ConcurrentUploadProgress(4, enabled=False)
    progress.start_file()
    progress.start_file()
    progress.finish_file(success=True)

    line = progress._line()

    assert '#' in line
    assert '~' in line
    assert '.' in line
    assert '1/4' in line
    assert 'uploading=1' in line


def test_parquet_writer_uses_installed_pyarrow_without_install(monkeypatch):
    install_calls = []

    def fail_install(packages):
        install_calls.append(packages)
        return False, 'unexpected install'

    monkeypatch.setattr(publisher, '_import_pyarrow', lambda: (_FakePyArrow, _FakeParquet))
    monkeypatch.setattr(dep_checker, 'install_packages', fail_install)

    data = publisher._parquet_bytes([{'id': 'case-001', 'files': {'path': 'blobs/case-001.zip'}}])

    assert install_calls == []
    assert json.loads(data) == {
        'id': ['case-001'],
        'files': ['{"path": "blobs/case-001.zip"}'],
    }


def test_parquet_writer_auto_installs_missing_pyarrow(monkeypatch):
    import_attempts = []
    install_calls = []

    def import_pyarrow():
        import_attempts.append(None)
        if len(import_attempts) == 1:
            raise ImportError('pyarrow missing')
        return _FakePyArrow, _FakeParquet

    def install_packages(packages):
        install_calls.append(packages)
        return True, 'installed'

    monkeypatch.setattr(publisher, '_import_pyarrow', import_pyarrow)
    monkeypatch.setattr(dep_checker, 'install_packages', install_packages)

    data = publisher._parquet_bytes([{'id': 'case-001'}])

    assert json.loads(data) == {'id': ['case-001']}
    assert install_calls == [[publisher.PYARROW_PACKAGE]]
    assert len(import_attempts) == 2


def test_parquet_writer_reports_auto_install_failure(monkeypatch):
    def import_pyarrow():
        raise ImportError('pyarrow missing')

    monkeypatch.setattr(publisher, '_import_pyarrow', import_pyarrow)
    monkeypatch.setattr(dep_checker, 'install_packages', lambda packages: (False, 'network failed'))

    with pytest.raises(RuntimeError, match='automatic install failed: network failed'):
        publisher._parquet_bytes([{'id': 'case-001'}])
