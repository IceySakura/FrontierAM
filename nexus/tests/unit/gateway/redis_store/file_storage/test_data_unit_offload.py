"""Waterlines, immutable SSD replicas, and failures at storage commit boundaries."""

from pathlib import Path
from uuid import uuid4

import pytest

from nexus.gateway.trajectory_store_backends.redis_store.file_storage.data_unit import DataUnitService
from nexus.gateway.trajectory_store_backends.redis_store.file_storage.types import DataUnitRegistration, Topology
from nexus.gateway.trajectory_store_backends.redis_store.types import RedisClientError


@pytest.fixture
def make_unit(tmp_path):
    units = []

    def make(**options):
        options.setdefault('memory_waterline_bytes', 12)
        options.setdefault('ssd_offload_dir', tmp_path / 'ssd')
        unit = DataUnitService('unit', **options).start()
        unit._topology = Topology.create([DataUnitRegistration(unit.data_unit_id, unit.endpoint)])
        units.append(unit)
        return unit

    yield make
    for unit in units:
        unit.stop()


def put(unit, value):
    url = unit._topology.new_url()
    unit._handle_request_sync('put', {'url': url}, value)
    return url


def get(unit, url):
    return unit._handle_request_sync('get', {'url': url}, b'')[1]


def stats(unit):
    result = unit._handle_request_sync('stats', {}, b'')[0]
    entries = list(unit._store.values())
    resident = [entry for entry in entries if entry.memory is not None]
    ssd = [entry for entry in entries if entry.disk_path is not None]
    cached = [entry for entry in ssd if entry.memory is not None]
    assert result == {
        'objects': len(entries),
        'used_bytes': sum(entry.size for entry in entries),
        'resident_bytes': sum(entry.size for entry in resident),
        'resident_objects': len(resident),
        'ssd_bytes': sum(entry.size for entry in ssd),
        'ssd_objects': len(ssd),
        'cached_bytes': sum(entry.size for entry in cached),
        'cached_objects': len(cached),
    }
    assert result['used_bytes'] == result['resident_bytes'] + result['ssd_bytes'] - result['cached_bytes']
    assert result['objects'] == result['resident_objects'] + result['ssd_objects'] - result['cached_objects']
    return result


def test_default_waterlines_enable_ssd_directory_at_startup(tmp_path):
    unit = DataUnitService('unit', ssd_offload_dir=tmp_path / 'ssd').start()
    unit._topology = Topology.create([DataUnitRegistration(unit.data_unit_id, unit.endpoint)])
    try:
        assert unit._memory_waterline_bytes == 214748364800
        assert unit._memory_safe_line_bytes == 107374182400
        assert unit._ssd_dir.is_dir()
        assert unit._ssd_dir.parent == tmp_path / 'ssd'
        url = put(unit, b'x' * 100)
        assert get(unit, url) == b'x' * 100
        assert stats(unit)['resident_bytes'] == 100
        assert stats(unit)['ssd_objects'] == 0
    finally:
        unit.stop()
    assert list((tmp_path / 'ssd').iterdir()) == []


@pytest.mark.parametrize(
    ('options', 'error_type'),
    [
        ({'memory_waterline_bytes': 0}, ValueError),
        ({'memory_waterline_bytes': -1}, ValueError),
        ({'memory_waterline_bytes': True}, ValueError),
        ({'memory_waterline_bytes': 1.5}, ValueError),
        ({'memory_waterline_bytes': None}, ValueError),
        ({'memory_waterline_bytes': None, 'memory_safe_line_bytes': 0}, ValueError),
        ({'memory_safe_line_bytes': -1}, AssertionError),
        ({'memory_safe_line_bytes': 13}, AssertionError),
        ({'memory_safe_line_bytes': True}, AssertionError),
        ({'memory_safe_line_bytes': 1.5}, AssertionError),
    ],
)
def test_invalid_waterlines_rejected_before_files_created(make_unit, tmp_path, options, error_type):
    with pytest.raises(error_type, match='memory_' if error_type is ValueError else None):
        make_unit(**options)
    assert not (tmp_path / 'ssd').exists()


def test_default_root_is_resolved_at_startup_and_units_are_isolated(tmp_path, monkeypatch):
    unit = DataUnitService('../unsafe/id', memory_waterline_bytes=12)
    monkeypatch.chdir(tmp_path)
    sibling = DataUnitService('../unsafe/id', memory_waterline_bytes=12)
    try:
        unit.start()
        sibling.start()
        assert unit._ssd_dir.parent == tmp_path / '.hykvstore'
        assert sibling._ssd_dir.parent == tmp_path / '.hykvstore'
        assert unit._ssd_dir != sibling._ssd_dir
        unit.stop()
        assert sibling._ssd_dir.is_dir()
    finally:
        unit.stop()
        sibling.stop()
    assert list((tmp_path / '.hykvstore').iterdir()) == []


def test_duplicate_start_does_not_replace_directory_or_leak_blobs(make_unit):
    unit = make_unit()
    url = put(unit, b'x' * 7)
    directory = unit._ssd_dir
    with pytest.raises(RuntimeError, match='start'):
        unit.start()
    assert unit._ssd_dir == directory
    assert get(unit, url) == b'x' * 7
    unit.stop()
    assert not directory.exists()
    assert list(directory.parent.iterdir()) == []


def test_duplicate_start_does_not_stop_a_running_unit(make_unit):
    unit = make_unit(memory_waterline_bytes=214748364800)
    url = put(unit, b'x' * 100)
    with pytest.raises(RuntimeError, match='start'):
        unit.start()
    assert unit._server._thread.is_alive()
    assert unit._ssd_dir.is_dir()
    assert get(unit, url) == b'x' * 100


def test_failed_server_start_cleans_allocated_directory(tmp_path, monkeypatch):
    unit = DataUnitService('unit', memory_waterline_bytes=12, ssd_offload_dir=tmp_path)

    def fail():
        raise OSError('bind failed')

    monkeypatch.setattr(unit._server, 'start', fail)
    with pytest.raises(OSError, match='bind failed'):
        unit.start()
    assert unit._ssd_dir is None
    assert list(tmp_path.iterdir()) == []
    unit.stop()


def test_above_safe_line_goes_directly_to_ssd_without_evicting(make_unit):
    unit = make_unit()
    small = put(unit, b'aaaaaa')
    large = put(unit, b'b' * 7)
    assert list(unit._resident_lru) == [small]
    assert unit._store[large].memory is None
    assert unit._store[large].disk_path.read_bytes() == b'b' * 7
    assert get(unit, large) == b'b' * 7
    result = stats(unit)
    assert (result['resident_bytes'], result['ssd_bytes'], result['cached_bytes']) == (6, 7, 0)


def test_stats_reports_mixed_tiers_without_traversing_the_store(make_unit):
    unit = make_unit(memory_waterline_bytes=8)
    cached = put(unit, b'cccc')
    put(unit, b'dddd')
    put(unit, b'large')
    assert get(unit, cached) == b'cccc'
    put(unit, b'm')

    class NoTraversal(dict):
        def values(self):
            pytest.fail('stats must not traverse stored blobs')

        items = __iter__ = values

    unit._store = NoTraversal(unit._store)
    assert unit._handle_request_sync('stats', {}, b'')[0] == {
        'objects': 4,
        'used_bytes': 14,
        'resident_bytes': 5,
        'resident_objects': 2,
        'ssd_bytes': 13,
        'ssd_objects': 3,
        'cached_bytes': 4,
        'cached_objects': 1,
    }


def test_odd_waterline_defaults_to_half_rounded_down(make_unit):
    unit = make_unit(memory_waterline_bytes=13)
    small = put(unit, b'a' * 6)
    large = put(unit, b'b' * 7)
    assert unit._store[small].memory == b'a' * 6
    assert unit._store[large].memory is None


def test_custom_safe_line_above_half_still_bounds_admission(make_unit):
    unit = make_unit(memory_safe_line_bytes=8)
    first = put(unit, b'a' * 8)
    put(unit, b'b' * 4)
    assert stats(unit)['resident_bytes'] == 4
    assert get(unit, first) == b'a' * 8
    assert stats(unit)['resident_bytes'] == 8
    assert stats(unit)['cached_bytes'] == 8


def test_safe_line_equal_to_waterline_allows_full_memory_then_evicts_lru(make_unit):
    unit = make_unit(memory_safe_line_bytes=12)
    first = put(unit, b'a' * 6)
    second = put(unit, b'b' * 6)
    result = stats(unit)
    assert (result['resident_bytes'], result['resident_objects'], result['ssd_objects']) == (12, 2, 0)
    third = put(unit, b'ccc')
    assert list(unit._resident_lru) == [second, third]
    assert unit._store[first].memory is None
    assert unit._store[first].disk_path.read_bytes() == b'a' * 6
    result = stats(unit)
    assert (result['objects'], result['used_bytes'], result['resident_bytes'], result['ssd_bytes']) == (3, 15, 9, 6)
    assert (result['resident_objects'], result['ssd_objects'], result['cached_objects']) == (2, 1, 0)


def test_ssd_refill_at_waterline_keeps_replica_and_larger_values_stay_on_ssd(make_unit):
    unit = make_unit(memory_safe_line_bytes=12)
    full = put(unit, b'f' * 12)
    small = put(unit, b's')
    source = unit._store[full].disk_path
    assert source.read_bytes() == b'f' * 12
    assert get(unit, full) == b'f' * 12
    assert unit._store[full].disk_path == source
    assert unit._store[small].memory is None
    result = stats(unit)
    assert (result['resident_bytes'], result['ssd_bytes'], result['cached_bytes']) == (12, 13, 12)
    assert (result['resident_objects'], result['ssd_objects'], result['cached_objects']) == (1, 2, 1)
    large = put(unit, b'l' * 13)
    assert unit._store[large].memory is None
    assert unit._store[large].disk_path.read_bytes() == b'l' * 13
    before = stats(unit)
    assert (before['objects'], before['used_bytes'], before['resident_bytes'], before['ssd_bytes']) == (3, 26, 12, 26)
    assert get(unit, large) == b'l' * 13
    assert stats(unit) == before


def test_high_waterline_evicts_lru_until_new_value_fits_safe_line(make_unit):
    unit = make_unit()
    first = put(unit, b'aaa')
    second = put(unit, b'bbb')
    third = put(unit, b'ccc')
    assert stats(unit)['resident_bytes'] == 9
    assert get(unit, first) == b'aaa'
    fourth = put(unit, b'ddd')
    assert list(unit._resident_lru) == [first, fourth]
    assert unit._store[second].memory is None
    assert unit._store[third].memory is None
    assert stats(unit)['resident_bytes'] == 6
    assert stats(unit)['ssd_bytes'] == 6


def test_get_caches_ssd_replica_and_re_eviction_does_not_rewrite(make_unit, monkeypatch):
    unit = make_unit()
    first = put(unit, b'aaaaaa')
    second = put(unit, b'bbbbbb')
    source = unit._store[first].disk_path
    assert source.is_file()
    assert get(unit, first) == b'aaaaaa'
    assert unit._store[first].disk_path == source
    assert stats(unit)['cached_bytes'] == 6
    assert stats(unit)['ssd_bytes'] == 12

    def forbidden_read(path):
        pytest.fail('memory hit must not read SSD')

    def forbidden_write(url, value):
        pytest.fail('backed cached entry must not be written again')

    with monkeypatch.context() as patch:
        patch.setattr(Path, 'read_bytes', forbidden_read)
        assert get(unit, first) == b'aaaaaa'
    with monkeypatch.context() as patch:
        patch.setattr(unit, '_write_blob', forbidden_write)
        third = put(unit, b'cccccc')
    assert unit._store[first].memory is None
    assert source.read_bytes() == b'aaaaaa'
    assert get(unit, third) == b'cccccc'
    assert unit._store[second].disk_path.is_file()
    result = stats(unit)
    assert (result['ssd_bytes'], result['ssd_objects'], result['cached_bytes'], result['cached_objects']) == (
        12,
        2,
        0,
        0,
    )


def test_empty_blob_and_zero_safe_line_are_valid(make_unit):
    unit = make_unit(memory_waterline_bytes=1, memory_safe_line_bytes=0)
    empty = put(unit, b'')
    large = put(unit, b'x')
    assert get(unit, empty) == b''
    assert get(unit, large) == b'x'
    assert unit._handle_request_sync('delete', {'url': empty}, b'')[0] == {'deleted': True}
    assert unit._handle_request_sync('delete', {'url': empty}, b'')[0] == {'deleted': False}
    assert stats(unit)['objects'] == 1
    assert stats(unit)['resident_bytes'] == 0


def test_zero_byte_eviction_cache_and_delete_update_object_counts(make_unit):
    unit = make_unit(memory_waterline_bytes=8)
    empty = put(unit, b'')
    put(unit, b'aaaa')
    put(unit, b'bbbb')
    empty_file = unit._store[empty].disk_path
    assert empty_file.read_bytes() == b''
    before = stats(unit)
    assert (before['ssd_objects'], before['cached_objects'], before['resident_objects']) == (2, 0, 1)

    assert get(unit, empty) == b''
    after_cache = stats(unit)
    assert after_cache['cached_bytes'] == 0
    assert (after_cache['ssd_objects'], after_cache['cached_objects'], after_cache['resident_objects']) == (2, 1, 2)
    assert unit._handle_request_sync('delete', {'url': empty}, b'')[0] == {'deleted': True}
    assert not empty_file.exists()
    assert stats(unit) == {**before, 'objects': 2, 'ssd_objects': 1}
    assert unit._handle_request_sync('delete', {'url': empty}, b'')[0] == {'deleted': False}
    assert stats(unit) == {**before, 'objects': 2, 'ssd_objects': 1}


def test_put_url_remains_immutable(make_unit):
    unit = make_unit()
    url = put(unit, b'a' * 7)
    with pytest.raises(RedisClientError) as error:
        unit._handle_request_sync('put', {'url': url}, b'new')
    assert error.value.code == 'already_exists'
    assert get(unit, url) == b'a' * 7
    assert stats(unit)['used_bytes'] == 7


@pytest.mark.parametrize('stage', ['open', 'write', 'close', 'replace'])
def test_failed_direct_put_is_not_indexed_and_temp_files_are_cleaned(make_unit, monkeypatch, stage):
    unit = make_unit()
    url = unit._topology.new_url()
    original_open = Path.open

    def fail(*args, **kwargs):
        raise OSError('disk failure')

    class FailedWriter:
        def __enter__(self):
            return self

        def __exit__(self, *args):
            if stage == 'close':
                fail()
            return False

        def write(self, value):
            if stage == 'close':
                return len(value)
            return fail()

    def patched_open(path, *args, **kwargs):
        if path.suffix == '.tmp':
            if stage == 'open':
                fail()
            original_open(path, 'xb').close()
            return FailedWriter()
        return original_open(path, *args, **kwargs)

    if stage == 'replace':
        monkeypatch.setattr(Path, 'replace', fail)
    else:
        monkeypatch.setattr(Path, 'open', patched_open)
    with pytest.raises(OSError, match='disk failure'):
        unit._handle_request_sync('put', {'url': url}, b'x' * 7)
    assert stats(unit)['objects'] == 0
    assert list(unit._ssd_dir.iterdir()) == []


def test_failed_eviction_preserves_candidate_and_commits_earlier_victims(make_unit, monkeypatch):
    unit = make_unit()
    first = put(unit, b'aaa')
    second = put(unit, b'bbb')
    third = put(unit, b'ccc')
    original_write = unit._write_blob

    def write(url, value):
        if value == b'bbb':
            raise OSError('disk full')
        return original_write(url, value)

    monkeypatch.setattr(unit, '_write_blob', write)
    with pytest.raises(OSError, match='disk full'):
        put(unit, b'ddd')
    assert unit._store[first].memory is None
    assert unit._store[first].disk_path.read_bytes() == b'aaa'
    assert unit._store[second].memory == b'bbb'
    assert unit._store[second].disk_path is None
    assert list(unit._resident_lru) == [second, third]
    result = stats(unit)
    assert (result['objects'], result['resident_bytes'], result['ssd_bytes']) == (3, 6, 3)
    assert (result['ssd_objects'], result['cached_objects']) == (1, 0)


def test_cache_admission_failure_reports_error_and_keeps_source_ssd(make_unit, monkeypatch):
    unit = make_unit()
    source = put(unit, b'aaa')
    put(unit, b'bbb')
    put(unit, b'ccc')
    put(unit, b'ddd')
    survivor = put(unit, b'eee')
    migrated_victim = list(unit._resident_lru)[0]
    failed_victim = list(unit._resident_lru)[1]
    original_write = unit._write_blob

    def write(url, value):
        if url == failed_victim:
            raise OSError('cache eviction failed')
        return original_write(url, value)

    monkeypatch.setattr(unit, '_write_blob', write)
    with pytest.raises(OSError, match='cache eviction failed'):
        get(unit, source)
    assert unit._store[source].memory is None
    assert unit._store[source].disk_path.read_bytes() == b'aaa'
    assert unit._store[migrated_victim].memory is None
    assert unit._store[survivor].memory == b'eee'
    result = stats(unit)
    assert result['resident_bytes'] == 6
    assert (result['ssd_bytes'], result['ssd_objects'], result['cached_bytes'], result['cached_objects']) == (
        9,
        3,
        0,
        0,
    )


@pytest.mark.parametrize('failure', ['missing', 'truncated', 'read_error'])
def test_ssd_read_failure_preserves_index_and_is_not_not_found(make_unit, monkeypatch, failure):
    unit = make_unit()
    url = put(unit, b'x' * 7)
    path = unit._store[url].disk_path
    if failure == 'missing':
        path.unlink()
        expected = FileNotFoundError
    elif failure == 'truncated':
        path.write_bytes(b'x')
        expected = RedisClientError
    else:

        def fail(path):
            raise OSError('read failed')

        monkeypatch.setattr(Path, 'read_bytes', fail)
        expected = OSError
    with pytest.raises(expected) as error:
        get(unit, url)
    if failure == 'truncated':
        assert error.value.code == 'server_error'
    assert stats(unit)['used_bytes'] == 7
    assert unit._store[url].memory is None


def test_delete_unlink_failure_preserves_both_replicas_and_stats(make_unit, monkeypatch):
    unit = make_unit()
    url = put(unit, b'aaaaaa')
    put(unit, b'bbbbbb')
    assert get(unit, url) == b'aaaaaa'
    before = stats(unit)

    def fail(path, **kwargs):
        raise PermissionError('unlink denied')

    with monkeypatch.context() as patch:
        patch.setattr(Path, 'unlink', fail)
        with pytest.raises(PermissionError, match='unlink denied'):
            unit._handle_request_sync('delete', {'url': url}, b'')
    assert stats(unit) == before
    assert get(unit, url) == b'aaaaaa'
    assert unit._store[url].disk_path.is_file()
    assert unit._handle_request_sync('delete', {'url': url}, b'')[0] == {'deleted': True}
    result = stats(unit)
    assert result['used_bytes'] == 6
    assert (result['ssd_objects'], result['cached_objects'], result['resident_objects']) == (1, 0, 0)


def test_explicit_delete_can_remove_entry_whose_ssd_file_is_missing(make_unit):
    unit = make_unit()
    url = put(unit, b'x' * 7)
    unit._store[url].disk_path.unlink()
    assert unit._handle_request_sync('delete', {'url': url}, b'')[0] == {'deleted': True}
    assert not any(stats(unit).values())


def test_stop_cleans_only_own_directory_and_preserves_files_if_server_stop_fails(make_unit, monkeypatch):
    unit = make_unit()
    sibling = make_unit()
    cached = put(unit, b'aaaaaa')
    put(unit, b'bbbbbb')
    assert get(unit, cached) == b'aaaaaa'
    put(unit, b'm')
    put(unit, b'x' * 7)
    sibling_url = put(sibling, b'y' * 7)
    directory = unit._ssd_dir
    before = stats(unit)
    assert (before['resident_objects'], before['ssd_objects'], before['cached_objects']) == (2, 3, 1)

    def fail():
        raise TimeoutError('server still running')

    with monkeypatch.context() as patch:
        patch.setattr(unit._server, 'stop', fail)
        with pytest.raises(TimeoutError, match='server still running'):
            unit.stop()
    assert directory.is_dir()
    assert stats(unit) == before
    unit.stop()
    assert not directory.exists()
    assert stats(unit) == dict.fromkeys(before, 0)
    assert get(sibling, sibling_url) == b'y' * 7
    unit.stop()
    assert stats(unit) == dict.fromkeys(before, 0)


def test_wrong_owner_is_rejected_before_disk_write(make_unit):
    unit = make_unit()
    other = DataUnitRegistration('other', type(unit.endpoint)('127.0.0.1', unit.endpoint.port + 1))
    unit._topology = Topology.create([DataUnitRegistration(unit.data_unit_id, unit.endpoint), other])
    url = unit._topology.new_url()
    while unit._topology.owner(url).data_unit_id == unit.data_unit_id:
        url = f'hykvfs://{unit._topology.cluster_id}/{uuid4().hex}'
    with pytest.raises(RedisClientError) as error:
        unit._handle_request_sync('put', {'url': url}, b'x' * 7)
    assert error.value.code == 'wrong_owner'
    assert list(unit._ssd_dir.iterdir()) == []
