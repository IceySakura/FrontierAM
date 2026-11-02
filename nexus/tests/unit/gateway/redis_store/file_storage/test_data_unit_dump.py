import time

from concurrent.futures import Future, ThreadPoolExecutor
from pathlib import Path
from threading import Event

import pytest

from nexus.gateway.trajectory_store_backends.redis_store.file_storage.client import FSClient
from nexus.gateway.trajectory_store_backends.redis_store.file_storage.data_unit import DataUnitService
from nexus.gateway.trajectory_store_backends.redis_store.file_storage.network import DealerClient, RouterServer
from nexus.gateway.trajectory_store_backends.redis_store.file_storage.types import (
    DataUnitRegistration,
    Endpoint,
    Topology,
)
from nexus.gateway.trajectory_store_backends.redis_store.types import RedisClientError


@pytest.fixture
def unit(tmp_path):
    service = DataUnitService('unit', memory_waterline_bytes=16, ssd_offload_dir=tmp_path / 'ssd').start()
    service._topology = Topology.create([DataUnitRegistration(service.data_unit_id, service.endpoint)])
    yield service
    service.stop()


@pytest.mark.parametrize('control_operations', [(), ('stats',)])
def test_router_future_does_not_occupy_request_worker(control_operations):
    deferred = Future()

    def handle(operation, body, blob):
        return deferred if operation == 'wait' else ({}, b'')

    server = RouterServer(Endpoint('127.0.0.1', 0), handle, control_operations=control_operations).start()
    client = DealerClient(server.endpoint, timeout=2)
    try:
        with ThreadPoolExecutor(max_workers=1) as executor:
            waiting = executor.submit(client.request, 'wait')
            try:
                assert client.request('other') == ({}, b'')
            finally:
                deferred.set_result(({'done': True}, b''))
            assert waiting.result()[0] == {'done': True}
    finally:
        client.close()
        server.stop()


def test_ensure_ssd_copy_retains_memory_and_accounts_dual_copy(unit):
    url = unit._topology.new_url()
    unit._handle_request_sync('put', {'url': url}, b'hello')
    before = unit._snapshot_stats()
    unit.ensure_ssd_copy(url)
    unit.ensure_ssd_copy(url)
    assert unit._store[url].memory == b'hello'
    assert unit._store[url].disk_path.read_bytes() == b'hello'
    assert unit._snapshot_stats() == {
        **before,
        'ssd_bytes': 5,
        'ssd_objects': 1,
        'cached_bytes': 5,
        'cached_objects': 1,
    }


@pytest.mark.parametrize('operation', ['prepare', 'export', 'import'])
def test_file_workers_start_ninth_file_while_first_is_blocked(unit, tmp_path, monkeypatch, operation):
    urls = [unit._topology.new_url() for _ in range(9)]
    if operation != 'import':
        for url in urls:
            unit._handle_request_sync('put', {'url': url}, b'a')
    if operation == 'export':
        unit.prepare_dump_sources('job', urls).result(timeout=5)
    first, ninth, release = Event(), Event(), Event()
    method = '_write_blob' if operation == 'prepare' else '_copy_file_atomically'
    original = getattr(unit, method)
    names = urls if operation == 'prepare' else [url.rsplit('/', 1)[1] for url in urls]

    def transfer(source, target):
        name = source if operation == 'prepare' else target.stem
        if name == names[0]:
            first.set()
            assert release.wait(5)
        elif name == names[-1]:
            ninth.set()
        return original(source, target)

    monkeypatch.setattr(unit, method, transfer)
    if operation == 'prepare':
        pending = [unit.prepare_dump_sources('job', urls)]
    elif operation == 'export':
        unit.start_dump_copy('job', str(tmp_path / 'dump'))
        pending = [unit._dump_jobs['job'].copy_future]
    else:
        source = tmp_path / 'source.blob'
        source.write_bytes(b'a')
        pending = [unit.import_file_to_ssd(url, str(source)) for url in urls]
    try:
        assert first.wait(2)
        assert ninth.wait(2)
        assert not pending[0].done()
    finally:
        release.set()
    for future in pending:
        future.result(timeout=5)
    if operation != 'import':
        unit.finalize_dump('job')


def test_delete_during_copy_preserves_ssd_until_finish(unit, tmp_path, monkeypatch):
    url = unit._topology.new_url()
    unit._handle_request_sync('put', {'url': url}, b'hello')
    unit.prepare_dump_sources('job', [url]).result(timeout=5)
    source = unit._store[url].disk_path
    entered, release = Event(), Event()
    original = unit._copy_file_atomically

    def copy(source, target):
        entered.set()
        assert release.wait(5)
        original(source, target)

    monkeypatch.setattr(unit, '_copy_file_atomically', copy)
    unit.start_dump_copy('job', str(tmp_path / 'dump'))
    try:
        assert entered.wait(2)
        assert unit._handle_request_sync('delete', {'url': url}, b'')[0]['deleted']
        assert source.exists()
    finally:
        release.set()
    unit._dump_jobs['job'].copy_future.result(timeout=5)
    assert source.exists()
    unit.finalize_dump('job')
    assert not source.exists()
    assert (tmp_path / 'dump' / 'data' / f'{url.rsplit("/", 1)[1]}.blob').read_bytes() == b'hello'


def wait_for(predicate):
    deadline = time.monotonic() + 5
    while not predicate():
        assert time.monotonic() < deadline
        time.sleep(0.005)


def test_waiting_memory_puts_do_not_exhaust_request_workers(unit, monkeypatch):
    urls = [unit._topology.new_url() for _ in range(3)]
    for url, value in zip(urls, (b'aaaa', b'bbbb', b'cccc'), strict=True):
        unit._handle_request_sync('put', {'url': url}, value)
    entered, release = Event(), Event()
    original = unit._write_blob

    def write(url, value):
        if value == b'aaaa':
            entered.set()
            assert release.wait(5)
        return original(url, value)

    monkeypatch.setattr(unit, '_write_blob', write)
    client = DealerClient(unit.endpoint, timeout=5)
    try:
        with ThreadPoolExecutor(max_workers=16) as executor:
            puts = [executor.submit(client.request, 'put', {'url': unit._topology.new_url()}, b'dddd')]
            try:
                assert entered.wait(2)
                puts.extend(
                    executor.submit(client.request, 'put', {'url': unit._topology.new_url()}, b'dddd')
                    for _ in range(12)
                )
                assert executor.submit(client.request, 'get', {'url': urls[2]}).result(timeout=2)[1] == b'cccc'
                assert executor.submit(client.request, 'stats').result(timeout=2)[0]['objects'] == 3
            finally:
                release.set()
            for pending in puts:
                pending.result(timeout=5)
    finally:
        release.set()
        client.close()


@pytest.mark.parametrize('request_workers', [1, 2])
def test_cold_gets_occupy_request_workers_until_refill_completes(tmp_path, monkeypatch, request_workers):
    unit = DataUnitService(
        'unit',
        memory_waterline_bytes=16,
        request_workers=request_workers,
        ssd_workers=1,
        ssd_offload_dir=tmp_path / 'ssd',
    ).start()
    unit._topology = Topology.create([DataUnitRegistration(unit.data_unit_id, unit.endpoint)])
    urls = [unit._topology.new_url() for _ in range(6)]
    values = (b'aaaa', b'bbbb', b'cccc', b'dddd', b'eeee')
    for url, value in zip(urls[:5], values, strict=True):
        unit._handle_request_sync('put', {'url': url}, value)
    unit.prepare_dump_sources('empty', []).result(timeout=5)
    entered, release = Event(), Event()
    original_write, original_read = unit._write_blob, Path.read_bytes
    original_submit = ThreadPoolExecutor.submit
    reads, requests = [], {}

    def write(url, value):
        if value == b'dddd':
            entered.set()
            assert release.wait(5)
        return original_write(url, value)

    def read(path):
        value = original_read(path)
        reads.append(path)
        return value

    def submit(executor, function, *args, **kwargs):
        pending = original_submit(executor, function, *args, **kwargs)
        if function == unit._server._execute_request and args[0].msg_type == 'get':
            requests[args[0].body['url']] = pending
        return pending

    monkeypatch.setattr(unit, '_write_blob', write)
    monkeypatch.setattr(Path, 'read_bytes', read)
    monkeypatch.setattr(ThreadPoolExecutor, 'submit', submit)
    client = DealerClient(unit.endpoint, timeout=5)
    try:
        with ThreadPoolExecutor(max_workers=6) as callers:
            putting = callers.submit(client.request, 'put', {'url': urls[5]}, b'ffffffff')
            try:
                assert entered.wait(2)
                getting = [callers.submit(client.request, 'get', {'url': url}) for url in urls[:request_workers]]
                wait_for(lambda: len(reads) == request_workers)
                extra = callers.submit(client.request, 'get', {'url': urls[2]})
                hot = callers.submit(client.request, 'get', {'url': urls[4]})
                # These are actual Router submissions, not merely queued client calls.
                wait_for(lambda: {urls[2], urls[4]}.issubset(requests))
                assert client.request('stats')[0]['objects'] == 5
                assert client.request('get_dump_status', {'dump_id': 'empty'})[0]['state'] == 'prepared'
                assert all(requests[url].running() for url in urls[:request_workers])
                assert not requests[urls[2]].running()
                assert not extra.done()
                assert not requests[urls[4]].running()
                assert not hot.done()
                assert len(reads) == request_workers
            finally:
                release.set()
            putting.result(timeout=5)
            for index, pending in enumerate(getting):
                assert pending.result(timeout=5)[1] == values[index]
            assert extra.result(timeout=5)[1] == values[2]
            assert hot.result(timeout=5)[1] == values[4]
        assert not unit._url_locks
    finally:
        release.set()
        client.close()
        unit.stop()


def test_failed_copy_retains_pins_until_retry_or_explicit_abort(unit, tmp_path, monkeypatch):
    url = unit._topology.new_url()
    unit._handle_request_sync('put', {'url': url}, b'hello')
    unit.prepare_dump_sources('job', [url, url]).result(timeout=5)
    unit.prepare_dump_sources('job', [url]).result(timeout=5)
    entry = unit._store[url]
    assert entry.dump_pin_count == 1
    original = unit._copy_file_atomically

    def fail(source, target):
        raise OSError('copy denied')

    monkeypatch.setattr(unit, '_copy_file_atomically', fail)
    unit.start_dump_copy('job', str(tmp_path / 'dump'))
    wait_for(lambda: unit.get_dump_status('job')['state'] == 'failed')
    unit._handle_request_sync('delete', {'url': url}, b'')
    assert entry.disk_path.exists()
    assert entry.dump_pin_count == 1
    monkeypatch.setattr(unit, '_copy_file_atomically', original)
    unit.start_dump_copy('job', str(tmp_path / 'dump'))
    wait_for(lambda: unit.get_dump_status('job')['state'] == 'complete')
    unit.finalize_dump('job')
    assert entry.disk_path is None
    assert entry.dump_pin_count == 0
    assert unit.finalize_dump('job')['state'] == 'finished'


def test_partial_prepare_is_abortable(unit):
    url = unit._topology.new_url()
    unit._handle_request_sync('put', {'url': url}, b'hello')
    with pytest.raises(RedisClientError, match='not found'):
        unit.prepare_dump_sources('partial', [url, unit._topology.new_url()]).result(timeout=5)
    assert unit._store[url].dump_pin_count == 1
    unit.finalize_dump('partial')
    assert unit._store[url].dump_pin_count == 0


def test_direct_ssd_ingestion_requires_a_shared_file_path(unit):
    with pytest.raises(ValueError, match='absolute shared file path'):
        unit.import_file_to_ssd(unit._topology.new_url(), b'value')


def test_abort_before_late_prepare_prevents_new_pins(unit, monkeypatch):
    url = unit._topology.new_url()
    unit._handle_request_sync('put', {'url': url}, b'hello')
    entered, release = Event(), Event()
    original = unit.prepare_dump_sources

    def delayed(dump_id, urls):
        entered.set()
        assert release.wait(5)
        return original(dump_id, urls)

    monkeypatch.setattr(unit, 'prepare_dump_sources', delayed)
    client = DealerClient(unit.endpoint, timeout=5)
    try:
        with ThreadPoolExecutor(max_workers=1) as executor:
            preparing = executor.submit(client.request, 'prepare_dump_sources', {'dump_id': 'late', 'urls': [url]})
            try:
                assert entered.wait(2)
                assert unit.finalize_dump('late')['state'] == 'finished'
            finally:
                release.set()
            with pytest.raises(RedisClientError) as error:
                preparing.result(timeout=5)
            assert error.value.code == 'dump_conflict'
        assert unit._store[url].dump_pin_count == 0
        assert unit._store[url].disk_path is None
    finally:
        release.set()
        client.close()


def test_memory_put_rechecks_duplicate_url_after_releasing_it_for_eviction(unit, monkeypatch):
    old_urls = [unit._topology.new_url() for _ in range(3)]
    for url, value in zip(old_urls, (b'aaaa', b'bbbb', b'cccc'), strict=True):
        unit._handle_request_sync('put', {'url': url}, value)
    target = unit._topology.new_url()
    entered, release = Event(), Event()
    original = unit._write_blob

    def write(url, value):
        if value == b'aaaa':
            entered.set()
            assert release.wait(5)
        return original(url, value)

    monkeypatch.setattr(unit, '_write_blob', write)
    with ThreadPoolExecutor(max_workers=1) as executor:
        memory_put = executor.submit(unit._handle_request_sync, 'put', {'url': target}, b'mmmm')
        try:
            assert entered.wait(2)
            unit._handle_request_sync('put', {'url': target}, b'larger-than-safe-line')
        finally:
            release.set()
        with pytest.raises(RedisClientError) as error:
            memory_put.result(timeout=5)
        assert error.value.code == 'already_exists'
    assert unit._handle_request_sync('get', {'url': target}, b'')[1] == b'larger-than-safe-line'


def test_fs_client_accumulates_dump_participants_and_exports_on_each_unit(tmp_path):
    units = [DataUnitService(str(index), ssd_offload_dir=tmp_path / f'ssd-{index}').start() for index in range(2)]
    topology = Topology.create([DataUnitRegistration(unit.data_unit_id, unit.endpoint) for unit in units])
    for unit in units:
        unit._topology = topology

    def handle(operation, body, blob):
        if operation == 'handshake':
            return {'protocol_version': 1, 'cluster_id': topology.cluster_id, 'controller_id': 'controller'}, b''
        assert operation == 'discover'
        return topology.dump(), b''

    controller = RouterServer(Endpoint('127.0.0.1', 0), handle).start()
    client = FSClient(controller.endpoint)
    urls = {}
    while len(urls) != 2:
        url = topology.new_url()
        urls[topology.owner(url).data_unit_id] = url
    try:
        for unit_id, url in urls.items():
            client.put(url, unit_id.encode())
            client.prepare_dump_sources('job', [url])
        assert set(client.get_dump_status('job')) == {'0', '1'}
        client.prepare_dump_sources('job', list(urls.values()))
        assert all(unit._store[urls[unit.data_unit_id]].dump_pin_count == 1 for unit in units)
        client.start_dump_copy('job', str(tmp_path / 'dump'))
        wait_for(lambda: all(status['state'] == 'complete' for status in client.get_dump_status('job').values()))
        assert all(status['state'] == 'finished' for status in client.finalize_dump('job').values())
        for unit_id, url in urls.items():
            assert (tmp_path / 'dump' / 'data' / f'{url.rsplit("/", 1)[1]}.blob').read_bytes() == unit_id.encode()
    finally:
        client.close()
        for unit in units:
            unit.stop()
        controller.stop()
