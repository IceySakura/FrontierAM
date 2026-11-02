"""Concurrent clients exercise DataUnit object and metadata coordination over TCP."""

import json

from concurrent.futures import ThreadPoolExecutor
from concurrent.futures import TimeoutError as FutureTimeoutError
from contextlib import contextmanager
from pathlib import Path
from threading import Barrier, Event

import pytest
import zmq

from nexus.gateway.trajectory_store_backends.redis_store.file_storage.data_unit import DataUnitService
from nexus.gateway.trajectory_store_backends.redis_store.file_storage.network import DealerClient
from nexus.gateway.trajectory_store_backends.redis_store.file_storage.types import (
    DataUnitRegistration,
    Topology,
)
from nexus.gateway.trajectory_store_backends.redis_store.types import RedisClientError


CLIENT_COUNT = 16


@pytest.fixture
def unit(tmp_path):
    service = DataUnitService(
        'unit-0',
        memory_waterline_bytes=128,
        memory_safe_line_bytes=64,
        ssd_offload_dir=tmp_path,
    ).start()
    service._topology = Topology.create([DataUnitRegistration(service.data_unit_id, service.endpoint)])
    try:
        yield service
    finally:
        service.stop()


@contextmanager
def connection(unit):
    client = DealerClient(unit.endpoint, timeout=10)
    try:
        yield client
    finally:
        client.close()


def check_stats(result):
    assert result['resident_bytes'] < 128
    assert result['used_bytes'] == result['resident_bytes'] + result['ssd_bytes'] - result['cached_bytes']
    assert result['objects'] == result['resident_objects'] + result['ssd_objects'] - result['cached_objects']
    assert 0 <= result['cached_bytes'] <= min(result['resident_bytes'], result['ssd_bytes'])
    assert 0 <= result['cached_objects'] <= min(result['resident_objects'], result['ssd_objects'])
    return result


def stats(client, *, objects, used_bytes):
    result, _ = client.request('stats')
    assert result['objects'] == objects
    assert result['used_bytes'] == used_bytes
    return check_stats(result)


def stored_files(root):
    return [path for path in root.rglob('*') if path.is_file()]


def track_sent_requests(monkeypatch):
    sent = {'delete': Event(), 'stats': Event()}
    original = zmq.Socket.send_multipart

    def send(socket, frames, *args, **kwargs):
        result = original(socket, frames, *args, **kwargs)
        if socket.getsockopt(zmq.TYPE) == zmq.DEALER:
            operation = json.loads(frames[0])['msg_type']
            if operation in sent:
                sent[operation].set()
        return result

    monkeypatch.setattr(zmq.Socket, 'send_multipart', send)
    return sent


def assert_delete_waits_while_stats_respond(sent, deleting, observing):
    assert sent['delete'].wait(timeout=5)
    assert sent['stats'].wait(timeout=5)
    assert not deleting.done()
    return check_stats(observing.result(timeout=2)[0])


def block_write(monkeypatch, payload, *, fail=False):
    entered, release = Event(), Event()
    original = Path.open

    class Writer:
        def __init__(self, file):
            self.file = file

        def __enter__(self):
            self.file.__enter__()
            return self

        def __exit__(self, *args):
            return self.file.__exit__(*args)

        def write(self, value):
            if value == payload:
                entered.set()
                assert release.wait(timeout=10), 'test did not release the SSD writer'
                if fail:
                    raise OSError('injected SSD write failure')
            return self.file.write(value)

    def open_file(path, mode='r', *args, **kwargs):
        file = original(path, mode, *args, **kwargs)
        return Writer(file) if path.suffix == '.tmp' and 'x' in mode else file

    monkeypatch.setattr(Path, 'open', open_file)
    return entered, release


def test_concurrent_put_same_url_has_one_immutable_winner(unit, tmp_path):
    url = unit._topology.new_url()
    barrier = Barrier(CLIENT_COUNT)
    payloads = [bytes([index]) * 80 for index in range(CLIENT_COUNT)]

    def put(index):
        with connection(unit) as client:
            barrier.wait(timeout=10)
            try:
                client.request('put', {'url': url}, payloads[index])
            except RedisClientError as error:
                return index, error.code
            return index, 'ok'

    with ThreadPoolExecutor(max_workers=CLIENT_COUNT) as pool:
        outcomes = list(pool.map(put, range(CLIENT_COUNT)))
    winners = [index for index, code in outcomes if code == 'ok']
    assert len(winners) == 1
    assert sum(code == 'already_exists' for _, code in outcomes) == CLIENT_COUNT - 1
    with connection(unit) as client:
        assert client.request('get', {'url': url})[1] == payloads[winners[0]]
        result = stats(client, objects=1, used_bytes=80)
    assert result['resident_bytes'] == 0
    assert result['ssd_bytes'] == 80
    assert [path.read_bytes() for path in stored_files(tmp_path)] == [payloads[winners[0]]]


def test_many_clients_refill_once_and_preserve_the_ssd_copy_during_eviction(unit, tmp_path, monkeypatch):
    initial_urls = [unit._topology.new_url() for _ in range(6)]
    initial_blobs = [bytes([index]) * 32 for index in range(6)]
    with connection(unit) as client:
        for url, blob in zip(initial_urls, initial_blobs, strict=True):
            client.request('put', {'url': url}, blob)
    source_path = next(path for path in stored_files(tmp_path) if path.read_bytes() == initial_blobs[0])
    rewriting, allow_write = block_write(monkeypatch, initial_blobs[0], fail=True)
    allow_write.set()
    barrier = Barrier(CLIENT_COUNT)

    def get(index):
        with connection(unit) as client:
            barrier.wait(timeout=10)
            return client.request('get', {'url': initial_urls[0]})[1]

    with ThreadPoolExecutor(max_workers=CLIENT_COUNT) as pool:
        assert list(pool.map(get, range(CLIENT_COUNT))) == [initial_blobs[0]] * CLIENT_COUNT
    with connection(unit) as client:
        result = stats(client, objects=6, used_bytes=192)
    assert result['resident_bytes'] == 96
    assert result['cached_bytes'] == 32
    assert result['cached_objects'] == 1

    new_urls = [unit._topology.new_url() for _ in range(CLIENT_COUNT)]
    new_blobs = [bytes([index + 16]) * 32 for index in range(CLIENT_COUNT)]
    barrier = Barrier(CLIENT_COUNT)

    def put_then_get(index):
        with connection(unit) as client:
            barrier.wait(timeout=10)
            client.request('put', {'url': new_urls[index]}, new_blobs[index])
            return client.request('get', {'url': initial_urls[0]})[1]

    with ThreadPoolExecutor(max_workers=CLIENT_COUNT) as pool:
        assert list(pool.map(put_then_get, range(CLIENT_COUNT))) == [initial_blobs[0]] * CLIENT_COUNT
    assert source_path.read_bytes() == initial_blobs[0]
    with connection(unit) as client:
        for url, expected in zip(initial_urls + new_urls, initial_blobs + new_blobs, strict=True):
            assert client.request('get', {'url': url})[1] == expected
        result = stats(client, objects=22, used_bytes=704)
    assert len(stored_files(tmp_path)) == result['ssd_objects']
    assert source_path.read_bytes() == initial_blobs[0]
    assert not rewriting.is_set()


def test_concurrent_get_and_delete_do_not_revive_a_cached_blob(unit, tmp_path):
    urls = [unit._topology.new_url() for _ in range(4)]
    blobs = [bytes([index + 1]) * 32 for index in range(4)]
    with connection(unit) as client:
        for url, blob in zip(urls, blobs, strict=True):
            client.request('put', {'url': url}, blob)
    source_path = next(path for path in stored_files(tmp_path) if path.read_bytes() == blobs[0])
    barrier = Barrier(CLIENT_COUNT)

    def operate(index):
        with connection(unit) as client:
            barrier.wait(timeout=10)
            if index % 2:
                return 'delete', client.request('delete', {'url': urls[0]})[0]['deleted']
            try:
                return 'get', client.request('get', {'url': urls[0]})[1]
            except RedisClientError as error:
                return 'get', error.code

    with ThreadPoolExecutor(max_workers=CLIENT_COUNT) as pool:
        outcomes = list(pool.map(operate, range(CLIENT_COUNT)))
    assert sum(value is True for operation, value in outcomes if operation == 'delete') == 1
    assert all(value in (blobs[0], 'not_found') for operation, value in outcomes if operation == 'get')
    assert not source_path.exists()
    with connection(unit) as client:
        with pytest.raises(RedisClientError) as error:
            client.request('get', {'url': urls[0]})
        assert error.value.code == 'not_found'
        for url, expected in zip(urls[1:], blobs[1:], strict=True):
            assert client.request('get', {'url': url})[1] == expected
        result = stats(client, objects=3, used_bytes=96)
    assert len(stored_files(tmp_path)) == result['ssd_objects']


@pytest.mark.parametrize('fail_write', [False, True])
def test_slow_ssd_eviction_delays_delete_but_not_stats(unit, tmp_path, monkeypatch, fail_write):
    urls = [unit._topology.new_url() for _ in range(4)]
    blobs = [bytes([index + 1]) * 32 for index in range(4)]
    with connection(unit) as client:
        for url, blob in zip(urls[:3], blobs[:3], strict=True):
            client.request('put', {'url': url}, blob)
    before_stats = unit._snapshot_stats()
    entered, release = block_write(monkeypatch, blobs[0], fail=fail_write)
    saved = Event()
    original_lock = unit._url_lock

    @contextmanager
    def url_lock(url):
        with original_lock(url):
            yield
            if url == urls[1] and unit._store[url].disk_path is not None:
                saved.set()

    monkeypatch.setattr(unit, '_url_lock', url_lock)
    sent = track_sent_requests(monkeypatch)

    with connection(unit) as writer, connection(unit) as deleter, connection(unit) as observer:
        with ThreadPoolExecutor(max_workers=3) as pool:
            putting = pool.submit(writer.request, 'put', {'url': urls[3]}, blobs[3])
            try:
                assert entered.wait(timeout=5)
                assert saved.wait(timeout=5)
                deleting = pool.submit(deleter.request, 'delete', {'url': urls[0]})
                observing = pool.submit(observer.request, 'stats')
                expected_during = {
                    **before_stats,
                    'resident_bytes': 64,
                    'resident_objects': 2,
                    'ssd_bytes': 32,
                    'ssd_objects': 1,
                }
                assert assert_delete_waits_while_stats_respond(sent, deleting, observing) == expected_during
            finally:
                release.set()
            if fail_write:
                with pytest.raises(RedisClientError, match='injected SSD write failure') as error:
                    putting.result(timeout=5)
                assert error.value.code == 'server_error'
            else:
                assert putting.result(timeout=5) == ({}, b'')
            assert deleting.result(timeout=5)[0]['deleted'] is True
            observed = check_stats(observing.result(timeout=5)[0])
        assert observed == expected_during
        with pytest.raises(RedisClientError) as error:
            observer.request('get', {'url': urls[0]})
        assert error.value.code == 'not_found'
        remaining_urls = urls[1:3] if fail_write else urls[1:]
        remaining_blobs = blobs[1:3] if fail_write else blobs[1:]
        for url, expected in zip(remaining_urls, remaining_blobs, strict=True):
            assert observer.request('get', {'url': url})[1] == expected
        if fail_write:
            with pytest.raises(RedisClientError) as error:
                observer.request('get', {'url': urls[3]})
            assert error.value.code == 'not_found'
        result = stats(observer, objects=len(remaining_urls), used_bytes=32 * len(remaining_urls))
    assert all(path.suffix == '.blob' for path in stored_files(tmp_path))
    assert len(stored_files(tmp_path)) == result['ssd_objects']


def test_slow_ssd_read_finishes_before_delete_and_cannot_refill_after_delete(unit, tmp_path, monkeypatch):
    urls = [unit._topology.new_url() for _ in range(4)]
    blobs = [bytes([index + 1]) * 32 for index in range(4)]
    with connection(unit) as client:
        for url, blob in zip(urls, blobs, strict=True):
            client.request('put', {'url': url}, blob)
    before_stats = unit._snapshot_stats()
    source_path = next(path for path in stored_files(tmp_path) if path.read_bytes() == blobs[0])
    entered, release = Event(), Event()
    original = Path.read_bytes

    def read(path):
        if path == source_path:
            entered.set()
            assert release.wait(timeout=10), 'test did not release the SSD reader'
        return original(path)

    monkeypatch.setattr(Path, 'read_bytes', read)
    sent = track_sent_requests(monkeypatch)
    with connection(unit) as reader, connection(unit) as deleter, connection(unit) as observer:
        with ThreadPoolExecutor(max_workers=3) as pool:
            getting = pool.submit(reader.request, 'get', {'url': urls[0]})
            try:
                assert entered.wait(timeout=5)
                deleting = pool.submit(deleter.request, 'delete', {'url': urls[0]})
                observing = pool.submit(observer.request, 'stats')
                assert assert_delete_waits_while_stats_respond(sent, deleting, observing) == before_stats
            finally:
                release.set()
            assert getting.result(timeout=5)[1] == blobs[0]
            assert deleting.result(timeout=5)[0]['deleted'] is True
            observed = check_stats(observing.result(timeout=5)[0])
        assert observed == before_stats
        with pytest.raises(RedisClientError) as error:
            observer.request('get', {'url': urls[0]})
        assert error.value.code == 'not_found'
        result = stats(observer, objects=3, used_bytes=96)
    assert not source_path.exists()
    assert len(stored_files(tmp_path)) == result['ssd_objects']


def test_failed_cache_eviction_does_not_block_deleting_the_read_object(unit, tmp_path, monkeypatch):
    urls = [unit._topology.new_url() for _ in range(5)]
    blobs = [bytes([index + 1]) * 32 for index in range(5)]
    with connection(unit) as client:
        for url, blob in zip(urls, blobs, strict=True):
            client.request('put', {'url': url}, blob)
    before_stats = unit._snapshot_stats()
    source_path = next(path for path in stored_files(tmp_path) if path.read_bytes() == blobs[0])
    entered, release = block_write(monkeypatch, blobs[2], fail=True)
    saved = Event()
    original_lock = unit._url_lock

    @contextmanager
    def url_lock(url):
        with original_lock(url):
            yield
            if url == urls[3] and unit._store[url].disk_path is not None:
                saved.set()

    monkeypatch.setattr(unit, '_url_lock', url_lock)
    with connection(unit) as reader, connection(unit) as deleter, connection(unit) as observer:
        with ThreadPoolExecutor(max_workers=3) as pool:
            getting = pool.submit(reader.request, 'get', {'url': urls[0]})
            try:
                assert entered.wait(timeout=5)
                assert saved.wait(timeout=5)
                deleting = pool.submit(deleter.request, 'delete', {'url': urls[0]})
                assert deleting.result(timeout=2)[0]['deleted'] is True
                observing = pool.submit(observer.request, 'stats')
                expected_during = {
                    **before_stats,
                    'objects': 4,
                    'used_bytes': 128,
                    'resident_bytes': 64,
                    'resident_objects': 2,
                    'ssd_bytes': 64,
                    'ssd_objects': 2,
                }
                assert check_stats(observing.result(timeout=2)[0]) == expected_during
            finally:
                release.set()
            with pytest.raises(RedisClientError, match='injected SSD write failure') as error:
                getting.result(timeout=5)
            assert error.value.code == 'server_error'
            assert deleting.result(timeout=5)[0]['deleted'] is True
            observed = check_stats(observing.result(timeout=5)[0])
        assert observed == expected_during
        result = stats(observer, objects=4, used_bytes=128)
        assert result['resident_bytes'] == 64
        assert result['ssd_bytes'] == 64
        assert result['cached_bytes'] == 0
        for url, expected in zip(urls[2:], blobs[2:], strict=True):
            assert observer.request('get', {'url': url})[1] == expected
        with pytest.raises(RedisClientError) as error:
            observer.request('get', {'url': urls[0]})
        assert error.value.code == 'not_found'
    assert not source_path.exists()
    assert {path.read_bytes() for path in stored_files(tmp_path)} == {blobs[1], blobs[3]}


def test_stop_waits_for_inflight_ssd_write_and_cleans_only_its_directory(unit, tmp_path, monkeypatch):
    url = unit._topology.new_url()
    payload = b's' * 80
    unrelated = tmp_path / 'unrelated.keep'
    unrelated.write_bytes(b'leave this file')
    entered, release = block_write(monkeypatch, payload)
    stopping_started = Event()

    def stop():
        stopping_started.set()
        unit.stop()

    with connection(unit) as client:
        client.timeout = 2

        def put():
            try:
                client.request('put', {'url': url}, payload)
            except RedisClientError as error:
                return error.code
            return 'ok'

        with ThreadPoolExecutor(max_workers=2) as pool:
            putting = pool.submit(put)
            try:
                assert entered.wait(timeout=5)
                temporary = [path for path in stored_files(tmp_path) if path.suffix == '.tmp']
                assert len(temporary) == 1
                owned_directory = temporary[0].parent
                stopping = pool.submit(stop)
                assert stopping_started.wait(timeout=5)
                with pytest.raises(FutureTimeoutError):
                    stopping.result(timeout=0.05)
                assert owned_directory.exists()
            finally:
                release.set()
            stopping.result(timeout=5)
            assert putting.result(timeout=5) in {'ok', 'timeout'}
    assert not owned_directory.exists()
    assert stored_files(tmp_path) == [unrelated]
    assert unrelated.read_bytes() == b'leave this file'
