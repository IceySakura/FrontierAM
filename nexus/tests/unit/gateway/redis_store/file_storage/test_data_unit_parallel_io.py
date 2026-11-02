from collections import OrderedDict
from concurrent.futures import Future, ThreadPoolExecutor
from contextlib import contextmanager
from pathlib import Path
from threading import Event, Lock

import pytest

from nexus.gateway.trajectory_store_backends.redis_store.file_storage import data_unit
from nexus.gateway.trajectory_store_backends.redis_store.file_storage.data_unit import DataUnitService
from nexus.gateway.trajectory_store_backends.redis_store.file_storage.types import DataUnitRegistration, Topology
from nexus.gateway.trajectory_store_backends.redis_store.types import RedisClientError


@pytest.fixture
def make_unit(tmp_path):
    units = []

    def make(**options):
        unit = DataUnitService('unit', memory_waterline_bytes=32, ssd_offload_dir=tmp_path / 'ssd', **options).start()
        unit._topology = Topology.create([DataUnitRegistration(unit.data_unit_id, unit.endpoint)])
        units.append(unit)
        return unit

    yield make
    for unit in units:
        unit.stop()


@pytest.fixture
def source_blob(tmp_path):
    path = tmp_path / 'source.blob'
    path.write_bytes(b'payload')
    return path


def prepare(unit, job, urls):
    result = unit.prepare_dump_sources(job, urls)
    assert isinstance(result, Future)
    return result.result(timeout=5)


def test_prepare_ssd_files_overlap_and_duplicate_request_reuses_pins(make_unit, monkeypatch):
    unit = make_unit(ssd_workers=2)
    urls = [unit._topology.new_url() for _ in range(2)]
    for url in urls:
        unit._handle_request_sync('put', {'url': url}, b'aaaa')
    entered, release, lock = Event(), Event(), Lock()
    writes = 0
    original = unit._write_blob

    def write(url, value):
        nonlocal writes
        with lock:
            writes += 1
            if writes == 2:
                entered.set()
        assert release.wait(5)
        return original(url, value)

    monkeypatch.setattr(unit, '_write_blob', write)
    preparing = unit.prepare_dump_sources('job', urls)
    try:
        assert isinstance(preparing, Future)
        assert entered.wait(2)
        assert unit.prepare_dump_sources('job', urls) is preparing
        assert unit._snapshot_stats()['ssd_objects'] == 0
    finally:
        release.set()
    preparing.result(timeout=5)
    assert writes == 2
    assert all(unit._store[url].dump_pin_count == 1 for url in urls)


def test_export_parallel_files_are_not_finished_by_last_submitted_file(make_unit, tmp_path, monkeypatch):
    unit = make_unit(copy_workers=2)
    urls = [unit._topology.new_url() for _ in range(2)]
    for url in urls:
        unit._handle_request_sync('put', {'url': url}, b'aaaa')
    prepare(unit, 'job', urls)
    first, second, release = Event(), Event(), Event()
    original = unit._copy_file_atomically

    def copy(source, target):
        if source == unit._store[urls[0]].disk_path:
            first.set()
            assert release.wait(5)
        else:
            second.set()
        original(source, target)

    monkeypatch.setattr(unit, '_copy_file_atomically', copy)
    unit.start_dump_copy('job', str(tmp_path / 'dump'))
    try:
        assert first.wait(2)
        assert second.wait(2)
        with pytest.raises(RedisClientError) as error:
            unit.finalize_dump('job')
        assert error.value.code == 'dump_busy'
        assert unit.get_dump_status('job')['state'] == 'running'
        assert all(unit._store[url].dump_pin_count == 1 for url in urls)
    finally:
        release.set()
    unit._dump_jobs['job'].copy_future.result(timeout=5)
    assert unit.get_dump_status('job')['state'] == 'complete'
    unit.finalize_dump('job')
    assert all(unit._store[url].dump_pin_count == 0 for url in urls)


def test_path_import_is_ssd_only_and_abort_cleans_kept_owned_entry(make_unit, source_blob):
    unit = make_unit(copy_workers=2)
    url = unit._topology.new_url()
    result = unit.import_file_to_ssd(url, str(source_blob))
    assert isinstance(result, Future)
    result.result(timeout=5)
    entry = unit._store[url]
    assert entry.memory is None
    assert entry.disk_path.read_bytes() == b'payload'
    assert unit._snapshot_stats()['resident_bytes'] == 0
    assert unit._snapshot_stats()['ssd_bytes'] == 7
    unit.finalize_ssd_imports([url], [url]).result(timeout=5)
    assert unit.get_ssd_import_status([url])['items'][url]['state'] == 'kept'
    unit.finalize_ssd_imports([url], []).result(timeout=5)
    assert url not in unit._store
    assert not any(unit._snapshot_stats().values())
    assert unit.get_ssd_import_status([url])['items'][url]['state'] == 'finished'


def test_path_import_reuses_same_source_and_rejects_conflicts(make_unit, source_blob, monkeypatch):
    unit = make_unit()
    url = unit._topology.new_url()
    entered, release = Event(), Event()
    copies = []
    original = unit._copy_file_atomically

    def copy(source, target):
        copies.append(source)
        entered.set()
        assert release.wait(5)
        original(source, target)

    monkeypatch.setattr(unit, '_copy_file_atomically', copy)
    result = unit.import_file_to_ssd(url, str(source_blob))
    try:
        assert entered.wait(2)
        assert unit.import_file_to_ssd(url, str(source_blob)) is result
        with pytest.raises(RedisClientError) as error:
            unit.import_file_to_ssd(url, str(source_blob.with_name('different.blob')))
        assert error.value.code == 'import_conflict'
    finally:
        release.set()
    result.result(timeout=5)
    unit.finalize_ssd_imports([url], [url]).result(timeout=5)
    assert unit.import_file_to_ssd(url, str(source_blob)) is result
    unit.finalize_ssd_imports([url], []).result(timeout=5)
    with pytest.raises(RedisClientError) as error:
        unit.import_file_to_ssd(url, str(source_blob))
    assert error.value.code == 'import_conflict'
    assert copies == [source_blob]


def test_import_submission_failure_completes_put_before_abort(make_unit, source_blob, monkeypatch):
    unit = make_unit()
    url = unit._topology.new_url()
    failed = Future()
    failed.set_exception(RuntimeError('copy submission failed'))
    with monkeypatch.context() as patch:
        patch.setattr(unit._copy_executor, 'submit', lambda *args: failed)
        result = unit.import_file_to_ssd(url, str(source_blob))
        with pytest.raises(RuntimeError, match='copy submission failed'):
            result.result(timeout=5)
    item = unit.get_ssd_import_status([url])['items'][url]
    assert item == {'state': 'failed', 'error': 'copy submission failed'}
    assert url not in unit._store
    item = unit.finalize_ssd_imports([url], []).result(timeout=5)[0]['items'][url]
    assert item['state'] == 'finished'
    assert source_blob.exists()


def test_import_cleanup_submission_failure_retains_entry_for_retry(make_unit, source_blob, monkeypatch):
    unit = make_unit()
    url = unit._topology.new_url()
    unit.import_file_to_ssd(url, str(source_blob)).result(timeout=5)
    entry = unit._store[url]
    path = entry.disk_path
    failed = Future()
    failed.set_exception(RuntimeError('cleanup submission failed'))
    with monkeypatch.context() as patch:
        patch.setattr(unit._ssd_executor, 'submit', lambda *args: failed)
        item = unit.finalize_ssd_imports([url], []).result(timeout=5)[0]['items'][url]
    assert item == {'state': 'failed', 'error': 'cleanup submission failed'}
    assert unit._store[url] is entry
    assert path.exists()
    item = unit.finalize_ssd_imports([url], []).result(timeout=5)[0]['items'][url]
    assert item == {'state': 'finished', 'error': None}
    assert url not in unit._store
    assert not path.exists()
    assert source_blob.exists()


def test_abort_pending_import_completes_waiters_and_prevents_late_start(make_unit, source_blob, monkeypatch):
    unit = make_unit(copy_workers=1)
    urls = [unit._topology.new_url() for _ in range(3)]
    entered, release = Event(), Event()
    original = unit._copy_file_atomically

    def copy(source, target):
        entered.set()
        assert release.wait(5)
        original(source, target)

    monkeypatch.setattr(unit, '_copy_file_atomically', copy)
    first = unit.import_file_to_ssd(urls[0], str(source_blob))
    assert entered.wait(2)
    pending = unit.import_file_to_ssd(urls[1], str(source_blob))
    abort = unit.finalize_ssd_imports(urls, [])
    try:
        assert not abort.done()
        with pytest.raises(RedisClientError):
            unit.import_file_to_ssd(urls[2], str(source_blob))
    finally:
        release.set()
    abort.result(timeout=5)
    for result in (first, pending):
        with pytest.raises(RedisClientError):
            result.result(timeout=5)
    assert not unit._store
    assert list(unit._ssd_dir.iterdir()) == []


@pytest.mark.parametrize('workers', [1, 2])
def test_global_copy_limit_shared_by_export_and_import_jobs(make_unit, tmp_path, source_blob, monkeypatch, workers):
    unit = make_unit(copy_workers=workers)
    exports = [unit._topology.new_url() for _ in range(2)]
    for url in exports:
        unit._handle_request_sync('put', {'url': url}, b'aaaa')
    prepare(unit, 'export', exports)
    entered, release, lock = Event(), Event(), Lock()
    count = 0
    original = unit._copy_file_atomically

    def copy(source, target):
        nonlocal count
        with lock:
            count += 1
            if count == workers:
                entered.set()
        assert release.wait(5)
        original(source, target)

    monkeypatch.setattr(unit, '_copy_file_atomically', copy)
    unit.start_dump_copy('export', str(tmp_path / 'export'))
    imports = [unit.import_file_to_ssd(unit._topology.new_url(), str(source_blob)) for _ in range(6)]
    try:
        assert entered.wait(2)
        assert count == workers
        assert unit._snapshot_stats()['objects'] == 2
    finally:
        release.set()
    for result in imports:
        result.result(timeout=5)
    unit._dump_jobs['export'].copy_future.result(timeout=5)
    assert count == 8


def test_keep_completion_cannot_complete_a_new_abort_before_cleanup(make_unit, source_blob, monkeypatch):
    unit = make_unit()
    url = unit._topology.new_url()
    unit.import_file_to_ssd(url, str(source_blob)).result(timeout=5)
    job = unit._ssd_imports[url]
    job.finish_future = Future()
    claimed, release_keep, cleaning, release_cleanup = Event(), Event(), Event(), Event()
    original_complete, original_unlink = job.finish_future.set_result, Path.unlink

    def complete(value):
        claimed.set()
        assert release_keep.wait(5)
        original_complete(value)

    def unlink(path, *args, **kwargs):
        if path == job.published_entry.disk_path:
            cleaning.set()
            assert release_cleanup.wait(5)
        return original_unlink(path, *args, **kwargs)

    monkeypatch.setattr(job.finish_future, 'set_result', complete)
    monkeypatch.setattr(Path, 'unlink', unlink)
    with ThreadPoolExecutor(max_workers=1) as executor:
        keeping = executor.submit(unit.finalize_ssd_imports, [url], [url])
        try:
            assert claimed.wait(2)
            aborting = unit.finalize_ssd_imports([url], [])
            assert cleaning.wait(2)
            release_keep.set()
            keeping.result(timeout=5).result(timeout=5)
            assert not aborting.done()
        finally:
            release_keep.set()
            release_cleanup.set()
        aborting.result(timeout=5)
    assert url not in unit._store


@pytest.mark.parametrize('fail_cleanup', [False, True])
def test_last_dump_pin_cleanup_is_idempotent(make_unit, monkeypatch, fail_cleanup):
    unit = make_unit()
    url = unit._topology.new_url()
    unit._handle_request_sync('put', {'url': url}, b'aaaa')
    prepare(unit, 'one', [url])
    prepare(unit, 'two', [url])
    entry = unit._store[url]
    source = entry.disk_path
    unit._delete_blob(url)
    assert not any(unit._snapshot_stats().values())
    unit.finalize_dump('one')
    assert source.exists()
    assert entry.dump_pin_count == 1
    assert not any(unit._snapshot_stats().values())
    if fail_cleanup:
        original = Path.unlink

        def unlink(path, *args, **kwargs):
            if path == source:
                raise OSError('cleanup denied')
            return original(path, *args, **kwargs)

        with monkeypatch.context() as patch:
            patch.setattr(Path, 'unlink', unlink)
            with pytest.raises(OSError, match='cleanup denied'):
                unit.finalize_dump('two')
        assert entry.dump_pin_count == 0
        assert source.exists()
    unit.finalize_dump('two')
    assert not source.exists()
    assert entry.dump_pin_count == 0
    assert entry.disk_path is None
    assert unit.finalize_dump('two')['state'] == 'finished'


def test_delete_before_selected_eviction_skips_removed_entry(make_unit, monkeypatch):
    unit = make_unit(ssd_workers=2)
    urls = [unit._topology.new_url() for _ in range(4)]
    for url in urls[:3]:
        unit._handle_request_sync('put', {'url': url}, b'a' * 8)
    selected, release = Event(), Event()
    original = unit._ssd_executor.submit

    def submit(function, *args):
        if args[0] != urls[0]:
            return original(function, *args)

        def execute():
            selected.set()
            assert release.wait(5)
            return function(*args)

        return original(execute)

    monkeypatch.setattr(unit._ssd_executor, 'submit', submit)
    with ThreadPoolExecutor(max_workers=1) as executor:
        putting = executor.submit(unit._handle_request_sync, 'put', {'url': urls[3]}, b'a' * 8)
        try:
            assert selected.wait(2)
            assert unit._delete_blob(urls[0])
        finally:
            release.set()
        putting.result(timeout=5)
    assert unit._snapshot_stats()['objects'] == 3
    assert unit._snapshot_stats()['resident_bytes'] == 16
    assert not (unit._ssd_dir / f'{urls[0].rsplit("/", 1)[1]}.blob').exists()
    assert not list(unit._ssd_dir.glob('*.tmp'))


@pytest.mark.parametrize('mutation', ['get', 'prepare'])
def test_selected_eviction_ignores_lru_change_and_reuses_dump_replica(make_unit, monkeypatch, mutation):
    unit = make_unit(ssd_workers=2)
    urls = [unit._topology.new_url() for _ in range(4)]
    for url in urls[:3]:
        unit._handle_request_sync('put', {'url': url}, b'a' * 8)
    selected, release = Event(), Event()
    original = unit._ssd_executor.submit

    def submit(function, *args):
        if args[0] != urls[0] or selected.is_set():
            return original(function, *args)

        def execute():
            selected.set()
            assert release.wait(5)
            return function(*args)

        return original(execute)

    monkeypatch.setattr(unit._ssd_executor, 'submit', submit)
    with ThreadPoolExecutor(max_workers=1) as executor:
        putting = executor.submit(unit._handle_request_sync, 'put', {'url': urls[3]}, b'a' * 8)
        try:
            assert selected.wait(2)
            if mutation == 'get':
                value = unit._handle_request_sync('get', {'url': urls[0]}, b'')[1]
                assert value == b'a' * 8
            else:
                prepare(unit, 'pin', [urls[0]])
                path = unit._store[urls[0]].disk_path
                inode = path.stat().st_ino
        finally:
            release.set()
        putting.result(timeout=5)
    assert all(unit._store[url].memory is None for url in urls[:2])
    assert unit._snapshot_stats()['cached_objects'] == 0
    assert unit._snapshot_stats()['ssd_objects'] == 2
    if mutation == 'prepare':
        assert unit._store[urls[0]].disk_path == path
        assert path.stat().st_ino == inode
        assert unit._store[urls[0]].dump_pin_count == 1
    else:
        assert value == b'a' * 8
    assert not list(unit._ssd_dir.glob('*.tmp'))


def test_get_waits_for_same_url_eviction_but_unrelated_get_remains_available(make_unit, monkeypatch):
    unit = make_unit(ssd_workers=2)
    urls = [unit._topology.new_url() for _ in range(4)]
    for url in urls[:3]:
        unit._handle_request_sync('put', {'url': url}, b'a' * 8)
    entered, release, getting_started = Event(), Event(), Event()
    original, original_get = unit._write_blob, unit._get_blob

    def write(url, value):
        if url == urls[0]:
            entered.set()
            assert release.wait(5)
        return original(url, value)

    def get(url):
        getting_started.set()
        return original_get(url)

    monkeypatch.setattr(unit, '_write_blob', write)
    monkeypatch.setattr(unit, '_get_blob', get)
    with ThreadPoolExecutor(max_workers=2) as executor:
        putting = executor.submit(unit._handle_request_sync, 'put', {'url': urls[3]}, b'a' * 8)
        try:
            assert entered.wait(2)
            getting = executor.submit(unit._handle_request_sync, 'get', {'url': urls[0]}, b'')
            assert getting_started.wait(2)
            assert not getting.done()
            assert unit._handle_request_sync('get', {'url': urls[2]}, b'')[1] == b'a' * 8
            assert unit._snapshot_stats()['objects'] == 3
        finally:
            release.set()
        putting.result(timeout=5)
        assert getting.result(timeout=5)[1] == b'a' * 8


def test_eviction_writes_overlap_and_reclaims_first_before_second_finishes(make_unit, monkeypatch):
    unit = make_unit(ssd_workers=2)
    urls = [unit._topology.new_url() for _ in range(4)]
    for url in urls[:3]:
        unit._handle_request_sync('put', {'url': url}, b'a' * 8)
    entered = {url: Event() for url in urls[:2]}
    release = {url: Event() for url in urls[:2]}
    reclaimed = Event()
    original_write, original_lock = unit._write_blob, unit._url_lock

    def write(url, value):
        entered[url].set()
        assert release[url].wait(5)
        return original_write(url, value)

    @contextmanager
    def url_lock(url):
        with original_lock(url):
            yield
            if url == urls[0] and unit._store[url].memory is None:
                reclaimed.set()

    monkeypatch.setattr(unit, '_write_blob', write)
    monkeypatch.setattr(unit, '_url_lock', url_lock)
    with ThreadPoolExecutor(max_workers=1) as executor:
        putting = executor.submit(unit._handle_request_sync, 'put', {'url': urls[3]}, b'a' * 8)
        try:
            assert all(event.wait(2) for event in entered.values())
            release[urls[0]].set()
            assert reclaimed.wait(2)
            assert not putting.done()
            assert unit._store[urls[0]].disk_path.read_bytes() == b'a' * 8
            assert unit._snapshot_stats()['resident_bytes'] == 16
            assert unit._snapshot_stats()['ssd_objects'] == 1
            assert unit._snapshot_stats()['cached_objects'] == 0
        finally:
            for event in release.values():
                event.set()
        putting.result(timeout=5)
    assert unit._snapshot_stats()['ssd_objects'] == 2
    assert not list(unit._ssd_dir.glob('*.tmp'))


@pytest.mark.parametrize('workers', [2, 8])
def test_eviction_reuses_workers_within_batch_and_waits_before_next_batch(make_unit, monkeypatch, workers):
    batch_size = workers * 2
    monkeypatch.setattr(data_unit, '_EVICTION_BATCH_SIZE', batch_size)
    unit = make_unit(ssd_workers=workers)
    urls = [unit._topology.new_url() for _ in range(32)]
    for url in urls[:-1]:
        unit._handle_request_sync('put', {'url': url}, b'a')
    first, batch_finished, next_batch, release, lock = Event(), Event(), Event(), Event(), Lock()
    active = maximum = finished = 0
    started = []
    original = unit._write_blob

    def write(url, value):
        nonlocal active, maximum, finished
        with lock:
            started.append(url)
            active += 1
            maximum = max(maximum, active)
            if url not in urls[:batch_size]:
                next_batch.set()
        try:
            if url == urls[0]:
                first.set()
                assert release.wait(5)
            return original(url, value)
        finally:
            with lock:
                active -= 1
                if url in urls[1:batch_size]:
                    finished += 1
                    if finished == batch_size - 1:
                        batch_finished.set()

    monkeypatch.setattr(unit, '_write_blob', write)
    putting = unit._dispatch_request('put', {'url': urls[-1]}, b'b' * 16)
    try:
        assert first.wait(2)
        assert batch_finished.wait(2)
        assert urls[workers] in started
        assert not next_batch.wait(2)
        assert not putting.done()
    finally:
        release.set()
    putting.result(timeout=5)
    assert next_batch.is_set()
    assert maximum <= workers
    assert set(started) == set(urls[:-1])
    assert len(started) == 31
    assert unit._snapshot_stats() == {
        'objects': 32,
        'used_bytes': 47,
        'resident_bytes': 16,
        'resident_objects': 1,
        'ssd_bytes': 31,
        'ssd_objects': 31,
        'cached_bytes': 0,
        'cached_objects': 0,
    }


def test_eviction_bounds_metadata_scan_and_rechecks_lru_between_batches(make_unit, monkeypatch):
    default_batch = getattr(data_unit, '_EVICTION_BATCH_SIZE', None)
    monkeypatch.setattr(data_unit, '_EVICTION_BATCH_SIZE', 4, raising=False)
    unit = make_unit(ssd_workers=2)
    urls = [unit._topology.new_url() for _ in range(32)]
    for url in urls[:-1]:
        unit._handle_request_sync('put', {'url': url}, b'a')

    class BoundedLRU(OrderedDict):
        def __iter__(self):
            for count, url in enumerate(super().__iter__(), 1):
                assert count <= 4, 'metadata scan exceeded its candidate batch'
                yield url

    unit._resident_lru = BoundedLRU(unit._resident_lru)
    submitted = []
    original_submit, original_write = unit._ssd_executor.submit, unit._write_blob
    with ThreadPoolExecutor(max_workers=1) as executor:

        def submit(function, *args):
            assert not unit._metadata_lock._is_owned()
            if len(submitted) == 4:
                assert executor.submit(unit._snapshot_stats).result(timeout=2)['objects'] == 31
                assert unit._get_blob(urls[10])[1] == b'a'
                assert unit._delete_blob(urls[20])
            submitted.append(args[0])
            return original_submit(function, *args)

        def write(url, value):
            assert not unit._metadata_lock._is_owned()
            return original_write(url, value)

        monkeypatch.setattr(unit._ssd_executor, 'submit', submit)
        monkeypatch.setattr(unit, '_write_blob', write)
        unit._dispatch_request('put', {'url': urls[-1]}, b'b' * 16).result(timeout=5)
    assert default_batch == 1000
    assert len(submitted) > 4
    assert len(submitted) == len(set(submitted))
    assert urls[20] not in unit._store
    stats = unit._snapshot_stats()
    assert stats['objects'] == 31
    assert stats['used_bytes'] == 46
    assert stats['resident_bytes'] == sum(entry.size for entry in unit._store.values() if entry.memory is not None)
    assert stats['ssd_bytes'] == sum(entry.size for entry in unit._store.values() if entry.disk_path is not None)
    assert not list(unit._ssd_dir.glob('*.tmp'))


def test_eviction_submission_failure_drains_already_submitted_files(make_unit, monkeypatch):
    monkeypatch.setattr(data_unit, '_EVICTION_BATCH_SIZE', 4)
    unit = make_unit(ssd_workers=2)
    urls = [unit._topology.new_url() for _ in range(32)]
    for url in urls[:-1]:
        unit._handle_request_sync('put', {'url': url}, b'a')
    entered, failed, release = Event(), Event(), Event()
    submitted = []
    original_submit, original_write = unit._ssd_executor.submit, unit._write_blob

    def submit(function, *args):
        if len(submitted) == 2:
            result = Future()
            result.set_exception(RuntimeError('submit failed'))
            failed.set()
        else:
            result = original_submit(function, *args)
        submitted.append(result)
        assert entered.wait(2)
        return result

    def write(url, value):
        if url == urls[0]:
            entered.set()
            assert release.wait(5)
        return original_write(url, value)

    monkeypatch.setattr(unit._ssd_executor, 'submit', submit)
    monkeypatch.setattr(unit, '_write_blob', write)
    putting = unit._dispatch_request('put', {'url': urls[-1]}, b'b' * 16)
    try:
        assert failed.wait(2)
        assert not putting.done()
    finally:
        release.set()
    with pytest.raises(RuntimeError, match='submit failed'):
        putting.result(timeout=5)
    assert all(result.done() for result in submitted)
    assert unit._store[urls[0]].memory is None
    assert urls[-1] not in unit._store
    assert not list(unit._ssd_dir.glob('*.tmp'))


@pytest.mark.parametrize('workers', [1, 2])
def test_eviction_failure_waits_for_all_submitted_files(make_unit, monkeypatch, workers):
    unit = make_unit(ssd_workers=workers)
    urls = [unit._topology.new_url() for _ in range(4)]
    for url in urls[:-1]:
        unit._handle_request_sync('put', {'url': url}, b'a' * 8)
    entered, release = Event(), Event()
    started = []
    original = unit._write_blob

    def write(url, value):
        started.append(url)
        if url == urls[0]:
            raise OSError('eviction write failed')
        if url == urls[1]:
            entered.set()
            assert release.wait(5)
        return original(url, value)

    monkeypatch.setattr(unit, '_write_blob', write)
    putting = unit._dispatch_request('put', {'url': urls[-1]}, b'b' * 16)
    try:
        assert entered.wait(2)
        assert not putting.done()
    finally:
        release.set()
    with pytest.raises(OSError, match='eviction write failed'):
        putting.result(timeout=5)
    assert set(started) == set(urls[:-1])
    assert unit._store[urls[0]].memory == b'a' * 8
    assert unit._store[urls[0]].disk_path is None
    assert all(unit._store[url].memory is None for url in urls[1:-1])
    assert urls[-1] not in unit._store
    assert unit._snapshot_stats()['resident_bytes'] == 8
    assert unit._snapshot_stats()['ssd_bytes'] == 16
    assert not list(unit._ssd_dir.glob('*.tmp'))


def test_import_abort_never_removes_a_replacement_entry(make_unit, tmp_path):
    unit = make_unit()
    source = tmp_path / 'source.blob'
    source.write_bytes(b'original')
    url = unit._topology.new_url()
    unit.import_file_to_ssd(url, str(source)).result(timeout=5)
    unit.finalize_ssd_imports([url], [url]).result(timeout=5)
    unit._delete_blob(url)
    unit._handle_request_sync('put', {'url': url}, b'new')
    unit.finalize_ssd_imports([url], []).result(timeout=5)
    assert unit._handle_request_sync('get', {'url': url}, b'')[1] == b'new'


def test_import_finish_reports_per_url_cleanup_failure(make_unit, source_blob, monkeypatch):
    unit = make_unit()
    urls = [unit._topology.new_url() for _ in range(2)]
    for url in urls:
        unit.import_file_to_ssd(url, str(source_blob)).result(timeout=5)
    denied = unit._store[urls[0]].disk_path
    original = Path.unlink

    def unlink(path, *args, **kwargs):
        if path == denied:
            raise OSError('cleanup denied')
        return original(path, *args, **kwargs)

    with monkeypatch.context() as patch:
        patch.setattr(Path, 'unlink', unlink)
        items = unit.finalize_ssd_imports(urls, []).result(timeout=5)[0]['items']
    assert items[urls[0]]['state'] == 'failed'
    assert 'cleanup denied' in items[urls[0]]['error']
    assert items[urls[1]]['state'] == 'finished'
    items = unit.finalize_ssd_imports(urls, []).result(timeout=5)[0]['items']
    assert all(item['state'] == 'finished' for item in items.values())


def test_unknown_keep_allows_delayed_import_but_unknown_abort_rejects_it(make_unit, source_blob):
    unit = make_unit()
    kept, aborted = unit._topology.new_url(), unit._topology.new_url()
    result = unit.finalize_ssd_imports([kept, aborted], [kept]).result(timeout=5)[0]['items']
    assert result[kept]['state'] == 'unknown'
    assert result[aborted]['state'] == 'finished'
    assert unit._ssd_imports[aborted].import_future.done()
    unit.import_file_to_ssd(kept, str(source_blob)).result(timeout=5)
    with pytest.raises(RedisClientError) as error:
        unit.import_file_to_ssd(aborted, str(source_blob))
    assert error.value.code == 'import_conflict'


def test_failed_import_is_not_reported_kept(make_unit, tmp_path):
    unit = make_unit()
    url = unit._topology.new_url()
    with pytest.raises(RedisClientError) as error:
        unit.import_file_to_ssd(url, str(tmp_path / 'missing.blob')).result(timeout=5)
    assert error.value.code == 'io_error'
    item = unit.finalize_ssd_imports([url], [url]).result(timeout=5)[0]['items'][url]
    assert item['state'] == 'failed'
    assert item['error']


def test_stop_preserves_failed_import_cleanup_for_retry(tmp_path, source_blob, monkeypatch):
    unit = DataUnitService('unit', ssd_offload_dir=tmp_path / 'ssd').start()
    unit._topology = Topology.create([DataUnitRegistration(unit.data_unit_id, unit.endpoint)])
    url = unit._topology.new_url()
    original = Path.unlink
    try:
        unit.import_file_to_ssd(url, str(source_blob)).result(timeout=5)
        entry = unit._store[url]
        path, directory = entry.disk_path, unit._ssd_dir
        assert path is not None
        assert directory is not None

        def unlink(target, *args, **kwargs):
            if target == path:
                raise OSError('import cleanup denied')
            return original(target, *args, **kwargs)

        with monkeypatch.context() as patch:
            patch.setattr(Path, 'unlink', unlink)
            with pytest.raises(RedisClientError, match='SSD import cleanup') as error:
                unit.stop()
            assert error.value.code == 'cleanup_failed'
            assert error.value.body['items'][url]['state'] == 'failed'
            assert unit._ssd_dir == directory
            assert directory.exists()
            assert path.exists()
            assert unit._store[url] is entry
        unit.stop()
        assert unit._ssd_dir is None
        assert not directory.exists()
        assert not unit._store
        assert source_blob.exists()
    finally:
        if unit._ssd_dir is not None:
            unit.stop()


def test_conflicting_finish_validates_all_urls_before_changing_jobs(make_unit, source_blob):
    unit = make_unit()
    live, aborted = unit._topology.new_url(), unit._topology.new_url()
    unit.import_file_to_ssd(live, str(source_blob)).result(timeout=5)
    unit.finalize_ssd_imports([aborted], []).result(timeout=5)
    with pytest.raises(RedisClientError) as error:
        unit.finalize_ssd_imports([live, aborted], [aborted])
    assert error.value.code == 'import_conflict'
    assert unit._ssd_imports[live].finish_action is None
    assert unit._store[live].disk_path.exists()


@pytest.mark.parametrize('failure_at', [1, 2])
def test_bounded_executor_submission_failure_finishes_future_and_wait_for_idle(make_unit, monkeypatch, failure_at):
    unit = make_unit()
    pool = unit._ssd_executor
    original = pool._executor.submit
    calls = 0

    def submit(*args, **kwargs):
        nonlocal calls
        calls += 1
        if calls == failure_at:
            raise RuntimeError('executor unavailable')
        return original(*args, **kwargs)

    monkeypatch.setattr(pool._executor, 'submit', submit)
    work = pool.submit(lambda: 42)
    assert work.cancel() is False
    with pytest.raises(RuntimeError, match='executor unavailable'):
        work.result(timeout=5)
    with ThreadPoolExecutor(max_workers=1) as executor:
        executor.submit(pool.wait_for_idle).result(timeout=2)


@pytest.mark.parametrize('name', ['request_workers', 'ssd_workers', 'copy_workers'])
@pytest.mark.parametrize('value', [0, -1, True, 1.5])
def test_invalid_worker_count_is_rejected(name, value):
    with pytest.raises(ValueError, match=name):
        DataUnitService('unit', **{name: value})
