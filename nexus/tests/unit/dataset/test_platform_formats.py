import os
import stat
import struct
import zipfile

import pytest

from nexus.dataset._format import format_bytes, format_bytes_detail
from nexus.dataset.client import _extract_item_ids, _extract_signed_urls
from nexus.dataset.item import DatasetItem, RemoteDatasetRef
from nexus.dataset.publisher import _extract_upload_urls
from nexus.dataset.task_files import _decode_http_body, _ssl_verify_for_url, task_files_from_spec
from nexus.dataset.zip_index import (
    NexusZipError,
    read_entry_bytes,
    read_zip_index_from_bytes,
    write_nexus_zip_from_directory,
)


def test_extract_item_ids_accepts_items_string_list():
    assert _extract_item_ids({'items': ['case-001', 'case-002']}) == ['case-001', 'case-002']


def test_extract_signed_urls_accepts_urls_list():
    assert _extract_signed_urls({'urls': [{'file': 'blobs/case.zip', 'url': 'https://cos/download'}]}) == {
        'blobs/case.zip': 'https://cos/download'
    }


def test_extract_upload_urls_accepts_urls_list():
    assert _extract_upload_urls({'urls': [{'file': '_meta/version.json', 'url': 'https://cos/upload'}]}) == {
        '_meta/version.json': 'https://cos/upload'
    }


def test_format_bytes_uses_decimal_units():
    assert format_bytes(999) == '999 B'
    assert format_bytes(1000) == '1.0 KB'
    assert format_bytes(11219) == '11.2 KB'
    assert format_bytes(11534336) == '11.5 MB'
    assert format_bytes_detail(999) == '999 B'
    assert format_bytes_detail(11219) == '11.2 KB (11219 bytes)'


def test_task_files_disables_ssl_verify_for_cos_internal_urls():
    assert _ssl_verify_for_url('https://bucket.cos-internal.ap-guangzhou.tencentcos.cn/path') is False
    assert _ssl_verify_for_url('https://bucket.cos.ap-guangzhou.myqcloud.com/path') is True
    assert _ssl_verify_for_url('https://example.com/path') is True


def test_decode_http_body_replaces_invalid_utf8():
    assert _decode_http_body(b'bad:\xaf', None) == 'bad:�'


async def _fake_http_range(data: bytes, _url: str, start: int, end: int) -> bytes:
    return data[start:end]


@pytest.mark.asyncio
async def test_remote_task_files_reads_zip_entry_by_range(tmp_path, monkeypatch):
    import nexus.dataset.task_files as task_files_module

    root = tmp_path / 'files'
    root.mkdir()
    (root / 'instruction.md').write_text('downloaded', encoding='utf-8')
    archive = write_nexus_zip_from_directory(root)
    ref = RemoteDatasetRef(namespace='ns', dataset='ds', version='v1', split='test')
    item = DatasetItem.from_mapping(
        {
            'id': 'case-001',
            'files': {
                'type': 'zip',
                'path': 'blobs/case-001.zip',
                'size': len(archive),
            },
            'body': {'task_name': 'echo', 'task_config': {'message': 'hello'}},
        },
        remote_ref=ref,
    )
    calls = []

    class FakeIoClient:
        def __init__(self, *, session, **_kwargs):
            self.session = session

        async def sign_files(self, namespace, dataset, version, files, *, split=None):
            calls.append(('sign', namespace, dataset, version, files, split))
            return {'urls': {'blobs/case-001.zip': 'https://cos/download'}}

    async def fake_range(url: str, start: int, end: int) -> bytes:
        calls.append(('range', url, start, end))
        return await _fake_http_range(archive, url, start, end)

    monkeypatch.setattr(task_files_module, 'DatasetIoClient', FakeIoClient)
    monkeypatch.setattr(task_files_module, '_http_range_read', fake_range)

    files = task_files_from_spec(item.resolved_task_files())
    assert files is not None

    assert await files.read_text('instruction.md') == 'downloaded'
    assert calls[0] == ('sign', 'ns', 'ds', 'v1', ['blobs/case-001.zip'], 'test')
    assert any(call[0] == 'range' for call in calls)


@pytest.mark.asyncio
async def test_remote_task_files_reads_empty_zip_entry_without_range(tmp_path, monkeypatch):
    import nexus.dataset.task_files as task_files_module

    root = tmp_path / 'files'
    root.mkdir()
    (root / 'tests').mkdir()
    (root / 'tests' / '__init__.py').write_bytes(b'')
    archive = write_nexus_zip_from_directory(root)
    ref = RemoteDatasetRef(namespace='ns', dataset='ds', version='v1', split='test')
    item = DatasetItem.from_mapping(
        {
            'id': 'case-001',
            'files': {
                'type': 'zip',
                'path': 'blobs/case-001.zip',
                'size': len(archive),
            },
            'body': {'task_name': 'echo', 'task_config': {'message': 'hello'}},
        },
        remote_ref=ref,
    )
    calls = []

    class FakeIoClient:
        def __init__(self, *, session, **_kwargs):
            self.session = session

        async def sign_files(self, namespace, dataset, version, files, *, split=None):
            calls.append(('sign', namespace, dataset, version, files, split))
            return {'urls': {'blobs/case-001.zip': 'https://cos/download'}}

    async def fake_range(url: str, start: int, end: int) -> bytes:
        if start == end:
            raise AssertionError('zero-length ZIP entries must not issue HTTP range requests')
        calls.append(('range', url, start, end))
        return await _fake_http_range(archive, url, start, end)

    monkeypatch.setattr(task_files_module, 'DatasetIoClient', FakeIoClient)
    monkeypatch.setattr(task_files_module, '_http_range_read', fake_range)

    files = task_files_from_spec(item.resolved_task_files())
    assert files is not None

    assert await files.read_bytes('tests/__init__.py') == b''
    assert calls[0] == ('sign', 'ns', 'ds', 'v1', ['blobs/case-001.zip'], 'test')


@pytest.mark.asyncio
async def test_remote_task_files_supports_utf8_file_names(tmp_path, monkeypatch):
    import nexus.dataset.task_files as task_files_module

    root = tmp_path / 'files'
    root.mkdir()
    (root / '测试').mkdir()
    (root / '测试' / '说明-你好.txt').write_text('中文内容\n', encoding='utf-8')
    (root / '测试' / 'emoji-雪-🚀.txt').write_text('utf8 symbols\n', encoding='utf-8')
    (root / '测试' / 'accent-é-ß.txt').write_text('accented\n', encoding='utf-8')
    archive = write_nexus_zip_from_directory(root)
    ref = RemoteDatasetRef(namespace='ns', dataset='ds', version='v1', split='test')
    item = DatasetItem.from_mapping(
        {
            'id': 'case-utf8',
            'files': {
                'type': 'zip',
                'path': 'blobs/case-utf8.zip',
                'size': len(archive),
            },
            'body': {'task_name': 'echo', 'task_config': {'message': 'hello'}},
        },
        remote_ref=ref,
    )

    class FakeIoClient:
        def __init__(self, *, session, **_kwargs):
            self.session = session

        async def sign_files(self, namespace, dataset, version, files, *, split=None):
            return {'urls': {'blobs/case-utf8.zip': 'https://cos/download'}}

    async def fake_range(url: str, start: int, end: int) -> bytes:
        return await _fake_http_range(archive, url, start, end)

    monkeypatch.setattr(task_files_module, 'DatasetIoClient', FakeIoClient)
    monkeypatch.setattr(task_files_module, '_http_range_read', fake_range)

    files = task_files_from_spec(item.resolved_task_files())
    assert files is not None

    assert await files.list('测试/**') == [
        '测试/accent-é-ß.txt',
        '测试/emoji-雪-🚀.txt',
        '测试/说明-你好.txt',
    ]
    assert await files.read_text('测试/说明-你好.txt') == '中文内容\n'
    assert await files.read_text('测试/emoji-雪-🚀.txt') == 'utf8 symbols\n'

    target = tmp_path / 'out'
    await files.download('测试/**', target)

    assert (target / 'accent-é-ß.txt').read_text(encoding='utf-8') == 'accented\n'
    assert (target / 'emoji-雪-🚀.txt').read_text(encoding='utf-8') == 'utf8 symbols\n'
    assert (target / '说明-你好.txt').read_text(encoding='utf-8') == '中文内容\n'


@pytest.mark.asyncio
async def test_read_entry_bytes_skips_range_for_empty_entry(tmp_path):
    root = tmp_path / 'files'
    root.mkdir()
    (root / 'empty.txt').write_bytes(b'')
    archive = write_nexus_zip_from_directory(root)
    entry = read_zip_index_from_bytes(archive).entries['empty.txt']

    async def read_range(_start: int, _end: int) -> bytes:
        raise AssertionError('empty ZIP entries should not call read_range')

    assert await read_entry_bytes(entry, read_range) == b''


def test_nexus_zip_uses_stored_for_small_files(tmp_path):
    root = tmp_path / 'files'
    root.mkdir()
    (root / 'small.txt').write_text('small', encoding='utf-8')

    archive = write_nexus_zip_from_directory(root)
    index = read_zip_index_from_bytes(archive)

    assert index.entries['small.txt'].compression_method == zipfile.ZIP_STORED


def test_nexus_zip_uses_deflate_for_large_compressible_files(tmp_path):
    root = tmp_path / 'files'
    root.mkdir()
    (root / 'large.txt').write_bytes(b'a' * (11 * 1024 * 1024))

    archive = write_nexus_zip_from_directory(root)
    index = read_zip_index_from_bytes(archive)

    entry = index.entries['large.txt']
    assert entry.compression_method == zipfile.ZIP_DEFLATED
    assert entry.compressed_size < entry.uncompressed_size


@pytest.mark.parametrize('special_kind', ['symlink-file', 'symlink-directory', 'fifo'])
def test_nexus_zip_writer_rejects_symlinks_and_special_files(tmp_path, special_kind):
    root = tmp_path / 'files'
    root.mkdir()
    outside = tmp_path / 'outside'
    outside.mkdir()
    (outside / 'secret.txt').write_text('secret', encoding='utf-8')
    if special_kind == 'symlink-file':
        (root / 'unsafe').symlink_to(outside / 'secret.txt')
    elif special_kind == 'symlink-directory':
        (root / 'unsafe').symlink_to(outside, target_is_directory=True)
    else:
        os.mkfifo(root / 'unsafe')

    with pytest.raises(NexusZipError, match='unsupported|symlink|regular'):
        write_nexus_zip_from_directory(root)


@pytest.mark.parametrize(
    ('name', 'replacement'),
    [
        ('absolute', b'/a.txt'),
        ('traversal', b'../x.x'),
        ('backslash', b'a\\.txt'),
    ],
)
def test_nexus_zip_index_rejects_unsafe_entry_names(tmp_path, name, replacement):
    del name
    root = tmp_path / 'files'
    root.mkdir()
    (root / 'aa.txt').write_text('safe', encoding='utf-8')
    archive = bytearray(write_nexus_zip_from_directory(root))
    eocd = archive.rfind(b'PK\x05\x06')
    cd_offset = struct.unpack_from('<I', archive, eocd + 16)[0]
    name_len = struct.unpack_from('<H', archive, cd_offset + 28)[0]
    assert name_len == len(replacement)
    archive[cd_offset + 46 : cd_offset + 46 + name_len] = replacement

    with pytest.raises(NexusZipError, match='path|entry'):
        read_zip_index_from_bytes(bytes(archive))


@pytest.mark.parametrize('corruption', ['multi-disk', 'compression', 'symlink', 'directory-signature'])
def test_nexus_zip_index_rejects_unsupported_or_damaged_archives(tmp_path, corruption):
    root = tmp_path / 'files'
    root.mkdir()
    (root / 'aa.txt').write_text('safe', encoding='utf-8')
    archive = bytearray(write_nexus_zip_from_directory(root))
    eocd = archive.rfind(b'PK\x05\x06')
    cd_offset = struct.unpack_from('<I', archive, eocd + 16)[0]
    if corruption == 'multi-disk':
        struct.pack_into('<H', archive, eocd + 4, 1)
    elif corruption == 'compression':
        struct.pack_into('<H', archive, cd_offset + 10, 99)
    elif corruption == 'symlink':
        struct.pack_into('<I', archive, cd_offset + 38, (stat.S_IFLNK | 0o777) << 16)
    else:
        archive[cd_offset : cd_offset + 4] = b'BAD!'

    with pytest.raises(NexusZipError):
        read_zip_index_from_bytes(bytes(archive))


@pytest.mark.asyncio
async def test_nexus_zip_reads_zip64_entry_fields(tmp_path):
    root = tmp_path / 'files'
    root.mkdir()
    (root / 'file.txt').write_text('zip64', encoding='utf-8')
    archive = _patch_single_entry_as_zip64(write_nexus_zip_from_directory(root))
    index = read_zip_index_from_bytes(archive)
    entry = index.entries['file.txt']

    async def read_range(start: int, end: int) -> bytes:
        return archive[start:end]

    files = task_files_from_spec({'type': 'zip', 'path': str(tmp_path / 'zip64.zip')})
    assert files is not None
    (tmp_path / 'zip64.zip').write_bytes(archive)
    assert await files.read_text('file.txt') == 'zip64'
    assert archive[entry.data_offset : entry.data_offset + entry.compressed_size] == await read_range(
        entry.data_offset,
        entry.data_offset + entry.compressed_size,
    )


def _patch_single_entry_as_zip64(data: bytes) -> bytes:
    raw = bytearray(data)
    eocd = raw.rfind(b'PK\x05\x06')
    assert eocd >= 0
    cd_size = struct.unpack_from('<I', raw, eocd + 12)[0]
    cd_offset = struct.unpack_from('<I', raw, eocd + 16)[0]
    assert raw[cd_offset : cd_offset + 4] == b'PK\x01\x02'
    compressed_size = struct.unpack_from('<I', raw, cd_offset + 20)[0]
    uncompressed_size = struct.unpack_from('<I', raw, cd_offset + 24)[0]
    name_len, extra_len, _comment_len = struct.unpack_from('<HHH', raw, cd_offset + 28)
    local_header_offset = struct.unpack_from('<I', raw, cd_offset + 42)[0]
    extra_start = cd_offset + 46 + name_len
    zip64_extra = struct.pack('<HHQQQ', 0x0001, 24, uncompressed_size, compressed_size, local_header_offset)

    raw[extra_start:extra_start] = zip64_extra
    struct.pack_into('<II', raw, cd_offset + 20, 0xFFFFFFFF, 0xFFFFFFFF)
    struct.pack_into('<H', raw, cd_offset + 30, extra_len + len(zip64_extra))
    struct.pack_into('<I', raw, cd_offset + 42, 0xFFFFFFFF)
    eocd += len(zip64_extra)
    struct.pack_into('<I', raw, eocd + 12, cd_size + len(zip64_extra))
    return bytes(raw)
