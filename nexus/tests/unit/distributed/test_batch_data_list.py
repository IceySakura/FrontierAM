from __future__ import annotations

import hashlib
import json
import struct

import pytest

from nexus.dataset.zip_index import write_nexus_zip_directory_to_file
from nexus.distributed.batch import data_list as data_list_module
from nexus.distributed.batch.data_list import (
    DataListError,
    DataListWriter,
    DataRef,
    IndexedFramePayload,
    read_indexed_frame,
    scan_data_list,
    validate_frame,
)


FILE_HEADER = struct.Struct('<4sHHQQQ')
FRAME_HEADER = struct.Struct('<QQQ')


def _zip_file(tmp_path, *, name: str = 'files.zip'):
    root = tmp_path / f'{name}-root'
    root.mkdir()
    (root / 'instruction.txt').write_text('hello data-list\n', encoding='utf-8')
    target = tmp_path / name
    write_nexus_zip_directory_to_file(root, target)
    return target


def _write_one(tmp_path, *, name: str = 'data_list.bin', max_frame_bytes=1024 * 1024, max_bytes=1024 * 1024):
    zip_path = _zip_file(tmp_path, name=f'{name}.zip')
    output = tmp_path / name
    with DataListWriter(output, max_frame_bytes=max_frame_bytes, max_bytes=max_bytes) as writer:
        ref = writer.write_frame(item_id='case-a', zip_path=zip_path)
    return output, zip_path, ref, writer.result()


def _raw_data_list(item_id: str, zip_bytes: bytes, *, task_files_sha256: str | None = None) -> bytes:
    metadata = {
        'item_id': item_id,
        'task_files': {
            'sha256': task_files_sha256 or hashlib.sha256(zip_bytes).hexdigest(),
            'size_bytes': len(zip_bytes),
            'type': 'zip',
        },
        'version': 1,
    }
    encoded = json.dumps(
        metadata,
        ensure_ascii=False,
        allow_nan=False,
        sort_keys=True,
        separators=(',', ':'),
    ).encode()
    frame = FRAME_HEADER.pack(FRAME_HEADER.size + len(encoded) + len(zip_bytes), len(encoded), len(zip_bytes))
    frame += encoded + zip_bytes
    return FILE_HEADER.pack(b'NXDL', 1, 0, FILE_HEADER.size, 1, 0) + frame


def test_data_list_writer_emits_exact_nxdl_v1_bytes_and_manifest(tmp_path):
    output, zip_path, ref, manifest = _write_one(tmp_path)
    zip_bytes = zip_path.read_bytes()
    metadata = {
        'item_id': 'case-a',
        'task_files': {
            'sha256': hashlib.sha256(zip_bytes).hexdigest(),
            'size_bytes': len(zip_bytes),
            'type': 'zip',
        },
        'version': 1,
    }
    encoded_metadata = json.dumps(
        metadata,
        ensure_ascii=False,
        allow_nan=False,
        sort_keys=True,
        separators=(',', ':'),
    ).encode()
    frame = (
        FRAME_HEADER.pack(
            FRAME_HEADER.size + len(encoded_metadata) + len(zip_bytes),
            len(encoded_metadata),
            len(zip_bytes),
        )
        + encoded_metadata
        + zip_bytes
    )
    expected = FILE_HEADER.pack(b'NXDL', 1, 0, FILE_HEADER.size, 1, 0) + frame

    assert output.read_bytes() == expected
    assert ref == DataRef(
        version=1,
        offset=FILE_HEADER.size,
        length=len(frame),
        sha256=hashlib.sha256(frame).hexdigest(),
    )
    assert manifest.path == output
    assert manifest.size_bytes == len(expected)
    assert manifest.sha256 == hashlib.sha256(expected).hexdigest()
    assert manifest.frame_count == 1


def test_data_list_scan_read_and_validate_one_frame(tmp_path):
    output, zip_path, ref, manifest = _write_one(tmp_path)

    index = scan_data_list(
        output,
        max_frame_bytes=ref.length,
        max_bytes=manifest.size_bytes,
        max_items=1,
    )
    payload = read_indexed_frame(
        output,
        ref,
        expected_frame=index.frames[0],
        expected_item_id='case-a',
        max_frame_bytes=ref.length,
    )
    frame = validate_frame(
        payload.encoded,
        expected_ref=ref,
        expected_item_id='case-a',
        max_frame_bytes=ref.length,
    )

    assert index.manifest == manifest
    assert len(index.frames) == 1
    assert index.frames[0].offset == ref.offset
    assert index.by_offset[ref.offset] == index.frames[0]
    assert frame.item_id == 'case-a'
    assert frame.zip_sha256 == hashlib.sha256(zip_path.read_bytes()).hexdigest()
    assert frame.zip_bytes == zip_path.read_bytes()


def test_data_list_index_reuses_offset_lookup(tmp_path):
    output, _zip_path, ref, manifest = _write_one(tmp_path)

    index = scan_data_list(
        output,
        max_frame_bytes=ref.length,
        max_bytes=manifest.size_bytes,
        max_items=1,
    )

    assert index.by_offset is index.by_offset


def test_data_list_scan_reads_container_once(tmp_path, monkeypatch):
    output, _zip_path, ref, manifest = _write_one(tmp_path)
    open_count = 0
    original_open = data_list_module._open_regular_no_follow

    def counting_open(path):
        nonlocal open_count
        open_count += 1
        return original_open(path)

    monkeypatch.setattr(data_list_module, '_open_regular_no_follow', counting_open)

    index = scan_data_list(
        output,
        max_frame_bytes=ref.length,
        max_bytes=manifest.size_bytes,
        max_items=1,
    )

    assert index.manifest.sha256 == manifest.sha256
    assert open_count == 1


def test_read_indexed_frame_reuses_initial_zip_validation(tmp_path, monkeypatch):
    output, _zip_path, ref, manifest = _write_one(tmp_path)
    index = scan_data_list(
        output,
        max_frame_bytes=ref.length,
        max_bytes=manifest.size_bytes,
        max_items=1,
    )
    expected_frame = index.frames[0]

    def unexpected_zip_validation(_zip_bytes):
        raise AssertionError('indexed frame read must not revalidate the ZIP')

    monkeypatch.setattr(data_list_module, 'read_zip_index_from_bytes', unexpected_zip_validation)

    payload = read_indexed_frame(
        output,
        ref,
        expected_frame=expected_frame,
        expected_item_id='case-a',
        max_frame_bytes=ref.length,
    )

    assert isinstance(payload, IndexedFramePayload)
    assert payload.descriptor is expected_frame
    assert hashlib.sha256(payload.encoded).hexdigest() == ref.sha256


def test_data_list_writer_enforces_complete_frame_and_file_limits(tmp_path):
    baseline, zip_path, ref, manifest = _write_one(tmp_path, name='baseline.bin')
    assert baseline.exists()

    exact = tmp_path / 'exact.bin'
    with DataListWriter(exact, max_frame_bytes=ref.length, max_bytes=manifest.size_bytes) as writer:
        exact_ref = writer.write_frame(item_id='case-a', zip_path=zip_path)
    assert exact_ref.length == ref.length
    assert writer.result().size_bytes == manifest.size_bytes

    frame_too_small = tmp_path / 'frame-too-small.bin'
    with pytest.raises(DataListError) as frame_error:
        with DataListWriter(
            frame_too_small,
            max_frame_bytes=ref.length - 1,
            max_bytes=manifest.size_bytes,
        ) as writer:
            writer.write_frame(item_id='case-a', zip_path=zip_path)
    assert frame_error.value.code == 'batch_data_frame_exceeds_byte_limit'

    list_too_small = tmp_path / 'list-too-small.bin'
    with pytest.raises(DataListError) as list_error:
        with DataListWriter(
            list_too_small,
            max_frame_bytes=ref.length,
            max_bytes=manifest.size_bytes - 1,
        ) as writer:
            writer.write_frame(item_id='case-a', zip_path=zip_path)
    assert list_error.value.code == 'batch_data_list_exceeds_byte_limit'

    assert not frame_too_small.exists()
    assert not list_too_small.exists()
    assert not list(tmp_path.glob('.*too-small.bin.*.tmp'))


def test_data_list_writer_requires_at_least_one_frame_and_preserves_existing_output(tmp_path):
    output = tmp_path / 'data_list.bin'
    output.write_bytes(b'previous')

    with pytest.raises(DataListError) as exc_info:
        with DataListWriter(output, max_frame_bytes=1024, max_bytes=1024):
            pass

    assert exc_info.value.code == 'invalid_batch_data_list'
    assert output.read_bytes() == b'previous'
    assert not list(tmp_path.glob(f'.{output.name}.*.tmp'))


@pytest.mark.parametrize(
    ('field_offset', 'replacement'),
    [
        (0, b'BAD!'),
        (4, struct.pack('<H', 2)),
        (6, struct.pack('<H', 1)),
        (8, struct.pack('<Q', 31)),
        (16, struct.pack('<Q', 0)),
        (24, struct.pack('<Q', 1)),
    ],
)
def test_data_list_scan_rejects_invalid_file_headers(tmp_path, field_offset, replacement):
    output, _zip_path, ref, manifest = _write_one(tmp_path)
    raw = bytearray(output.read_bytes())
    raw[field_offset : field_offset + len(replacement)] = replacement
    output.write_bytes(raw)

    with pytest.raises(DataListError) as exc_info:
        scan_data_list(output, max_frame_bytes=ref.length, max_bytes=manifest.size_bytes, max_items=1)
    assert exc_info.value.code == 'invalid_batch_data_list'


@pytest.mark.parametrize('mutation', ['truncated', 'trailing', 'bad_frame_size'])
def test_data_list_scan_rejects_bad_file_boundaries(tmp_path, mutation):
    output, _zip_path, ref, manifest = _write_one(tmp_path)
    raw = bytearray(output.read_bytes())
    if mutation == 'truncated':
        raw = raw[:-1]
    elif mutation == 'trailing':
        raw.extend(b'x')
    else:
        struct.pack_into('<Q', raw, FILE_HEADER.size, ref.length + 1)
    output.write_bytes(raw)

    with pytest.raises(DataListError) as exc_info:
        scan_data_list(output, max_frame_bytes=ref.length + 1, max_bytes=manifest.size_bytes + 1, max_items=1)
    assert exc_info.value.code == 'invalid_batch_data_list'


def test_data_list_rejects_noncanonical_metadata_and_invalid_zip(tmp_path):
    zip_path = _zip_file(tmp_path)
    zip_bytes = zip_path.read_bytes()
    canonical = _raw_data_list('case-a', zip_bytes)
    metadata_size = struct.unpack_from('<Q', canonical, FILE_HEADER.size + 8)[0]
    metadata = json.loads(canonical[FILE_HEADER.size + FRAME_HEADER.size :][:metadata_size])
    noncanonical_metadata = json.dumps(metadata, ensure_ascii=False, sort_keys=False).encode()
    frame = (
        FRAME_HEADER.pack(
            FRAME_HEADER.size + len(noncanonical_metadata) + len(zip_bytes),
            len(noncanonical_metadata),
            len(zip_bytes),
        )
        + noncanonical_metadata
        + zip_bytes
    )
    noncanonical = FILE_HEADER.pack(b'NXDL', 1, 0, FILE_HEADER.size, 1, 0) + frame
    noncanonical_path = tmp_path / 'noncanonical.bin'
    noncanonical_path.write_bytes(noncanonical)

    invalid_zip_path = tmp_path / 'invalid-zip.bin'
    invalid_zip_path.write_bytes(_raw_data_list('case-a', b'not-a-zip'))

    bad_identity_path = tmp_path / 'bad-identity.bin'
    bad_identity_path.write_bytes(_raw_data_list('case-a', zip_bytes, task_files_sha256='0' * 64))

    for path in (noncanonical_path, invalid_zip_path, bad_identity_path):
        with pytest.raises(DataListError) as exc_info:
            scan_data_list(path, max_frame_bytes=1024 * 1024, max_bytes=1024 * 1024, max_items=1)
        assert exc_info.value.code == 'invalid_batch_data_list'


def test_read_indexed_frame_rejects_ref_digest_and_bounds(tmp_path):
    output, _zip_path, ref, manifest = _write_one(tmp_path)
    index = scan_data_list(output, max_frame_bytes=ref.length, max_bytes=manifest.size_bytes, max_items=1)
    expected_frame = index.frames[0]

    with pytest.raises(DataListError) as digest_error:
        read_indexed_frame(
            output,
            ref.model_copy(update={'sha256': '0' * 64}),
            expected_frame=expected_frame,
            expected_item_id='case-a',
            max_frame_bytes=ref.length,
        )
    assert digest_error.value.code == 'batch_task_list_data_ref_invalid'

    with pytest.raises(DataListError) as bounds_error:
        read_indexed_frame(
            output,
            ref.model_copy(update={'length': ref.length + 1}),
            expected_frame=expected_frame,
            expected_item_id='case-a',
            max_frame_bytes=ref.length + 1,
        )
    assert bounds_error.value.code == 'batch_task_list_data_ref_invalid'


def test_data_list_scan_enforces_limits_and_item_count(tmp_path):
    output, _zip_path, ref, manifest = _write_one(tmp_path)

    with pytest.raises(DataListError) as frame_error:
        scan_data_list(output, max_frame_bytes=ref.length - 1, max_bytes=manifest.size_bytes, max_items=1)
    assert frame_error.value.code == 'batch_data_frame_exceeds_byte_limit'

    with pytest.raises(DataListError) as list_error:
        scan_data_list(output, max_frame_bytes=ref.length, max_bytes=manifest.size_bytes - 1, max_items=1)
    assert list_error.value.code == 'batch_data_list_exceeds_byte_limit'

    with pytest.raises(DataListError) as item_error:
        scan_data_list(output, max_frame_bytes=ref.length, max_bytes=manifest.size_bytes, max_items=0)
    assert item_error.value.code == 'invalid_batch_data_list'
