"""Control RPCs must keep progressing while storage requests are blocked."""

from concurrent.futures import ThreadPoolExecutor
from contextlib import ExitStack
from threading import Event

import pytest
import zmq

from nexus.gateway.trajectory_store_backends.redis_store.file_storage.network import DealerClient, RouterServer
from nexus.gateway.trajectory_store_backends.redis_store.file_storage.types import Endpoint
from nexus.gateway.trajectory_store_backends.redis_store.types import RedisClientError


@pytest.fixture
def rpc():
    with ExitStack() as cleanup:

        def start(*, timeouts=(5,), **handler):
            server = RouterServer(Endpoint('127.0.0.1', 0), **handler, control_operations={'stats'}).start()
            cleanup.callback(server.stop)
            clients = []
            for timeout in timeouts:
                client = DealerClient(server.endpoint, timeout=timeout)
                cleanup.callback(client.close)
                clients.append(client)
            return server, *clients

        yield start


@pytest.mark.parametrize('wire_handler', [False, True])
@pytest.mark.parametrize('blocked_operation', ['put', 'stats'])
def test_control_and_data_requests_use_independent_handlers(rpc, wire_handler, blocked_operation):
    entered, release = Event(), Event()

    def handle(operation, body, blob):
        if operation == blocked_operation:
            entered.set()
            assert release.wait(5), 'test did not release the blocked handler'
        return {'operation': operation}, blob

    def handle_wire(message):
        body, blob = handle(message.msg_type, message.body, message.payload)
        return message.reply(body, payload=blob)

    handler = {'wire_handler': handle_wire} if wire_handler else {'handler': handle}
    _, blocked, independent = rpc(**handler, timeouts=(5, 2))
    other_operation = 'stats' if blocked_operation == 'put' else 'put'
    with ThreadPoolExecutor(max_workers=1) as pool:
        request = pool.submit(blocked.request, blocked_operation)
        try:
            assert entered.wait(2)
            body, _ = independent.request(other_operation)
            assert body == {'operation': other_operation}
            assert not request.done()
        finally:
            release.set()
        assert request.result()[0] == {'operation': blocked_operation}


def test_data_requests_remain_serial_with_a_separate_control_handler(rpc):
    first_entered, second_entered, release = Event(), Event(), Event()

    def handle(operation, body, blob):
        if operation == 'put':
            if body['index'] == 1:
                first_entered.set()
                assert release.wait(5), 'test did not release the first request'
            else:
                second_entered.set()
        return {}, b''

    _, client = rpc(handler=handle)
    with ThreadPoolExecutor(max_workers=2) as pool:
        first = pool.submit(client.request, 'put', {'index': 1})
        try:
            assert first_entered.wait(2)
            second = pool.submit(client.request, 'put', {'index': 2})
            assert client.request('stats') == ({}, b'')
            assert not second_entered.is_set()
            assert not second.done()
        finally:
            release.set()
        assert first.result() == second.result() == ({}, b'')


@pytest.mark.parametrize('operation', ['put', 'stats'])
def test_handler_errors_preserve_request_correlation(rpc, operation):
    def handle(operation, body, blob):
        if body.get('fail'):
            raise RedisClientError('rejected', code='invalid_request')
        return body, blob

    _, client = rpc(handler=handle, timeouts=(2,))
    with pytest.raises(RedisClientError, match='rejected') as error:
        client.request(operation, {'fail': True})
    assert error.value.code == 'invalid_request'
    assert client.request(operation, {'next': True}, b'next') == ({'next': True}, b'next')


def test_stop_drains_blocked_handler_before_returning(rpc):
    entered, release, stopping = Event(), Event(), Event()

    def handle(operation, body, blob):
        entered.set()
        assert release.wait(5), 'test did not release the handler'
        return {}, b''

    server, client = rpc(handler=handle)

    def stop():
        stopping.set()
        server.stop()

    with ThreadPoolExecutor(max_workers=2) as pool:
        request = pool.submit(client.request, 'put')
        try:
            assert entered.wait(2)
            stopped = pool.submit(stop)
            assert stopping.wait(2)
            assert not stopped.done()
        finally:
            release.set()
        assert request.result() == ({}, b'')
        stopped.result()
    assert not server._thread.is_alive()


def test_completed_handler_wakes_network_poller(rpc, monkeypatch):
    entered, polling, release = Event(), Event(), Event()
    original_poll = zmq.Poller.poll

    def slow_poll(poller, timeout=None):
        if entered.is_set():
            polling.set()
        return original_poll(poller, 1000)

    def handle(operation, body, blob):
        entered.set()
        assert release.wait(5), 'test did not release the handler'
        return {}, b''

    monkeypatch.setattr(zmq.Poller, 'poll', slow_poll)
    _, client = rpc(handler=handle)
    with ThreadPoolExecutor(max_workers=1) as pool:
        request = pool.submit(client.request, 'put')
        try:
            assert entered.wait(2)
            assert polling.wait(2), 'router did not wait for input while the handler was blocked'
            release.set()
            assert request.result(timeout=0.5) == ({}, b'')
        finally:
            release.set()
