from unittest.mock import AsyncMock

import pytest

from nexus.dataset import io as dataset_io
from nexus.dataset.exceptions import DatasetClientError
from nexus.dataset.io import DATASET_MULTIPART_THRESHOLD, DatasetIoClient
from nexus.utils.cos import DEFAULT_MULTIPART_PART_SIZE, CosClientError


def _sparse_file(path, size):
    with path.open('wb') as stream:
        stream.truncate(size)
    return path


def _client():
    return DatasetIoClient(server_url='http://io', auth_token='token', session=object())


@pytest.mark.asyncio
@pytest.mark.parametrize('size', [0, DATASET_MULTIPART_THRESHOLD - 1])
async def test_upload_local_file_uses_regular_put_below_threshold(monkeypatch, tmp_path, size):
    path = _sparse_file(tmp_path / 'small file.json', size)
    client = _client()
    response = {'urls': [{'file': path.name, 'url': 'https://cos/upload'}]}
    client.sign_uploads = AsyncMock(return_value=response)
    upload = AsyncMock()
    monkeypatch.setattr(dataset_io, 'upload_presigned_file', upload)

    result = await client.upload_local_file('ns', 'ds', 'dev-123', path, expires=3600)

    assert result is response
    client.sign_uploads.assert_awaited_once_with('ns', 'ds', 'dev-123', [path.name], expires=3600)
    upload.assert_awaited_once()
    assert upload.await_args.kwargs['key'] == path.name
    assert upload.await_args.kwargs['expected_stat'].st_size == size


@pytest.mark.asyncio
@pytest.mark.parametrize(
    'size',
    [DATASET_MULTIPART_THRESHOLD, DATASET_MULTIPART_THRESHOLD + 1, 5_000_000_000, 5 * 1024**3],
)
async def test_upload_local_file_uses_multipart_at_threshold_and_for_5_gib(monkeypatch, tmp_path, size):
    path = _sparse_file(tmp_path / '大文件 with spaces.parquet', size)
    client = _client()
    response = {
        'upload_id': 'upload-id',
        'part_urls': ['https://cos/part'],
        'complete_url': 'https://cos/complete',
        'abort_url': 'https://cos/abort',
        'part_size': DEFAULT_MULTIPART_PART_SIZE,
    }
    client.sign_multipart_uploads = AsyncMock(return_value=response)
    upload = AsyncMock()
    monkeypatch.setattr(dataset_io, 'upload_presigned_multipart_file', upload)

    result = await client.upload_local_file(
        'ns',
        'ds',
        'dev-123',
        path,
        expires=7200,
        max_concurrency=3,
    )

    assert result is response
    client.sign_multipart_uploads.assert_awaited_once_with(
        'ns',
        'ds',
        'dev-123',
        path.name,
        size,
        part_size=DEFAULT_MULTIPART_PART_SIZE,
        expires=7200,
    )
    upload.assert_awaited_once()
    assert upload.await_args.kwargs['key'] == path.name
    assert upload.await_args.kwargs['max_concurrency'] == 3
    assert upload.await_args.kwargs['expected_stat'].st_size == size


@pytest.mark.asyncio
async def test_upload_local_file_preserves_explicit_nested_object_key(monkeypatch, tmp_path):
    path = tmp_path / 'local name.json'
    path.write_text('{}')
    client = _client()
    response = {'urls': [{'file': 'test/raw/远程 name.json', 'url': 'https://cos/upload'}]}
    client.sign_uploads = AsyncMock(return_value=response)
    monkeypatch.setattr(dataset_io, 'upload_presigned_file', AsyncMock())

    await client.upload_local_file('ns', 'ds', 'dev-123', path, key='test/raw/远程 name.json')

    client.sign_uploads.assert_awaited_once_with(
        'ns',
        'ds',
        'dev-123',
        ['test/raw/远程 name.json'],
        expires=None,
    )


@pytest.mark.asyncio
async def test_upload_local_file_validates_path_before_request(tmp_path):
    client = _client()
    client.sign_uploads = AsyncMock()
    client.sign_multipart_uploads = AsyncMock()

    with pytest.raises(FileNotFoundError):
        await client.upload_local_file('ns', 'ds', 'dev-123', tmp_path / 'missing')
    with pytest.raises(ValueError, match='not a regular file'):
        await client.upload_local_file('ns', 'ds', 'dev-123', tmp_path)
    with pytest.raises(ValueError, match='max_concurrency'):
        await client.upload_local_file('ns', 'ds', 'dev-123', _sparse_file(tmp_path / 'file', 1), max_concurrency=0)
    with pytest.raises(ValueError, match='part_size'):
        await client.upload_local_file(
            'ns',
            'ds',
            'dev-123',
            _sparse_file(tmp_path / 'large', DATASET_MULTIPART_THRESHOLD),
            part_size=0,
        )

    client.sign_uploads.assert_not_awaited()
    client.sign_multipart_uploads.assert_not_awaited()


@pytest.mark.asyncio
async def test_upload_local_file_propagates_multipart_initialization_failure(monkeypatch, tmp_path):
    path = _sparse_file(tmp_path / 'large.bin', DATASET_MULTIPART_THRESHOLD)
    client = _client()
    client.sign_multipart_uploads = AsyncMock(side_effect=DatasetClientError('initialization failed'))
    upload = AsyncMock()
    monkeypatch.setattr(dataset_io, 'upload_presigned_multipart_file', upload)

    with pytest.raises(DatasetClientError, match='initialization failed'):
        await client.upload_local_file('ns', 'ds', 'dev-123', path)

    upload.assert_not_awaited()


@pytest.mark.asyncio
async def test_upload_local_file_wraps_cos_transfer_errors(monkeypatch, tmp_path):
    path = _sparse_file(tmp_path / 'large.bin', DATASET_MULTIPART_THRESHOLD)
    client = _client()
    client.sign_multipart_uploads = AsyncMock(return_value={})
    monkeypatch.setattr(
        dataset_io,
        'upload_presigned_multipart_file',
        AsyncMock(side_effect=CosClientError('part upload failed')),
    )

    with pytest.raises(DatasetClientError, match='part upload failed') as exc:
        await client.upload_local_file('ns', 'ds', 'dev-123', path)

    assert isinstance(exc.value.__cause__, CosClientError)


@pytest.mark.asyncio
async def test_upload_local_file_rejects_missing_signed_url(monkeypatch, tmp_path):
    path = tmp_path / 'small.json'
    path.write_text('{}')
    client = _client()
    client.sign_uploads = AsyncMock(return_value={'urls': []})
    upload = AsyncMock()
    monkeypatch.setattr(dataset_io, 'upload_presigned_file', upload)

    with pytest.raises(DatasetClientError, match='did not include a URL'):
        await client.upload_local_file('ns', 'ds', 'dev-123', path)

    upload.assert_not_awaited()
