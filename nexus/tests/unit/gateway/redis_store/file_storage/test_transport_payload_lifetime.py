"""Idle TCP routers must not keep the previous request's temporary blobs alive."""

import gc
import sys
import threading
import weakref

from concurrent.futures import ThreadPoolExecutor
from contextlib import contextmanager

import pytest
import zmq

from nexus.gateway.trajectory_store_backends.redis_store.file_storage import network
from nexus.gateway.trajectory_store_backends.redis_store.file_storage.network import (
    Codec,
    DealerClient,
    Message,
    RouterServer,
)
from nexus.gateway.trajectory_store_backends.redis_store.file_storage.types import Endpoint


class _ObservedSocket:
    def __init__(self, socket):
        self.socket = socket
        self.received = False
        self.settled = threading.Event()
        self.fail_send = False

    def __getattr__(self, name):
        return getattr(self.socket, name)

    def poll(self, timeout):
        if self.received:
            self.settled.set()
        return self.socket.poll(timeout)

    def recv_multipart(self):
        frames = self.socket.recv_multipart()
        self.received = True
        return frames

    def send_multipart(self, frames, **kwargs):
        if self.fail_send:
            raise zmq.ZMQError(zmq.EHOSTUNREACH)
        return self.socket.send_multipart(frames, **kwargs)


@pytest.fixture
def observed_router(monkeypatch):
    original = network._socket
    probes = []

    def socket(kind, context=None):
        transport = original(kind, context)
        if kind == zmq.ROUTER:
            transport = _ObservedSocket(transport)
            probes.append(transport)
        return transport

    monkeypatch.setattr(network, '_socket', socket)
    return probes


@pytest.fixture
def observed_dealer(monkeypatch):
    original = network._socket
    probes = []

    def socket(kind, context=None):
        transport = original(kind, context)
        if kind == zmq.DEALER:
            transport = _ObservedSocket(transport)
            probes.append(transport)
        return transport

    monkeypatch.setattr(network, '_socket', socket)
    return probes


def _retains_blob(value, blob):
    if isinstance(value, Message):
        return _retains_blob(value.payload, blob)
    if isinstance(value, (bytes, bytearray, memoryview)):
        return value == blob
    if isinstance(value, (list, tuple)):
        return any(_retains_blob(item, blob) for item in value)
    return False


@contextmanager
def router_connection(server):
    server.start()
    client = zmq.Context.instance().socket(zmq.DEALER)
    client.setsockopt(zmq.LINGER, 0)
    try:
        client.connect(server.endpoint.tcp_address)
        yield client
    finally:
        client.close()
        server.stop()


@pytest.mark.parametrize('outcome', ['success', 'handler_error', 'malformed', 'send_error'])
def test_router_releases_input_blob_after_request(observed_router, outcome):
    def handler(operation, body, blob):
        if outcome == 'handler_error':
            raise ValueError('rejected')
        return {}, b''

    blob = b'request payload' * 4096
    server = RouterServer(Endpoint('127.0.0.1', 0), handler)
    with router_connection(server) as client:
        probe = observed_router[0]
        probe.fail_send = outcome == 'send_error'
        frames = [b'not JSON', blob] if outcome == 'malformed' else Codec.encode(Message.create('put', payload=blob))
        client.send_multipart(frames)
        assert probe.settled.wait(5), 'router did not return to polling'
        if outcome in ('success', 'handler_error'):
            assert client.poll(5000)
            reply, _ = Codec.decode(client.recv_multipart())
            assert reply.msg_type == ('error' if outcome == 'handler_error' else 'ok')
        frame = sys._current_frames()[server._thread.ident]
        while frame is not None and frame.f_code.co_name != '_serve':
            frame = frame.f_back
        assert frame is not None, 'router thread left its serve loop'
        retained = [name for name, value in frame.f_locals.items() if _retains_blob(value, blob)]
        assert not retained, f'idle router retains request blob in {retained}'


def test_router_releases_temporary_reply_blob_after_send(observed_router):
    class ReplyBlob(bytearray):
        pass

    references = []

    def handler(operation, body, blob):
        value = ReplyBlob(b'reply payload' * 4096)
        references.append(weakref.ref(value))
        return {}, value

    server = RouterServer(Endpoint('127.0.0.1', 0), handler)
    with router_connection(server) as client:
        client.send_multipart(Codec.encode(Message.create('get')))
        assert client.poll(5000)
        _, value = Codec.decode(client.recv_multipart())
        assert value == b'reply payload' * 4096
        assert observed_router[0].settled.wait(5)
        gc.collect()
        assert references[0]() is None, 'idle router retains its temporary reply blob'


def test_router_preserves_payloads_of_messages_retained_by_handler(observed_router):
    retained = []

    def handler(request):
        reply = request.reply(payload=bytearray(b'reply payload'))
        retained.extend((request, reply))
        return reply

    server = RouterServer(Endpoint('127.0.0.1', 0), wire_handler=handler)
    with router_connection(server) as client:
        client.send_multipart(Codec.encode(Message.create('put', payload=b'request payload')))
        assert client.poll(5000)
        _, value = Codec.decode(client.recv_multipart())
        assert value == b'reply payload'
        assert observed_router[0].settled.wait(5)
        assert retained[0].payload == b'request payload'
        assert retained[1].payload == b'reply payload'


def test_idle_dealer_releases_decoded_reply_without_invalidating_caller_bytes(observed_dealer, monkeypatch):
    references = []
    original_decode = Codec.decode

    def decode(frames):
        message, blob = original_decode(frames)
        if message.msg_type == 'ok':
            references.append(weakref.ref(message))
        return message, blob

    monkeypatch.setattr(Codec, 'decode', staticmethod(decode))
    expected = b'reply payload' * 65536
    server = RouterServer(Endpoint('127.0.0.1', 0), lambda operation, body, blob: ({}, expected)).start()
    client = DealerClient(server.endpoint)
    try:
        _, value = client.request('get')
        assert observed_dealer[0].settled.wait(5)
        gc.collect()
        assert type(value) is bytes
        assert value == expected
        assert references[0]() is None, 'idle dealer retains its decoded reply or completed Future'
        frame = sys._current_frames()[client._thread.ident]
        while frame is not None and frame.f_code.co_name != '_serve':
            frame = frame.f_back
        assert frame is not None, 'dealer thread left its serve loop'
        retained = [name for name, local in frame.f_locals.items() if _retains_blob(local, expected)]
        assert not retained, f'idle dealer retains reply bytes in {retained}'
    finally:
        client.close()
        server.stop()


@pytest.mark.parametrize('outcome', ['success', 'handler_error'])
def test_idle_dealer_releases_last_put_payload_after_reply(observed_dealer, outcome):
    class RequestBlob(bytearray):
        pass

    expected = b'request payload' * 65536

    def handler(operation, body, blob):
        assert blob == expected
        if outcome == 'handler_error':
            raise ValueError('rejected')
        return {'size': len(blob)}, b''

    server = RouterServer(Endpoint('127.0.0.1', 0), handler).start()
    client = DealerClient(server.endpoint)
    value = RequestBlob(expected)
    reference = weakref.ref(value)
    request = Message.create('put', payload=value)
    try:
        reply = client.request(request)
        assert reply.msg_type == ('error' if outcome == 'handler_error' else 'ok')
        assert reply.body == (
            {'code': 'server_error', 'message': 'rejected'} if outcome == 'handler_error' else {'size': len(expected)}
        )
        del request, value
        assert observed_dealer[0].settled.wait(5)
        gc.collect()
        assert reference() is None, 'idle dealer retains the last PUT payload after its caller releases it'
    finally:
        client.close()
        server.stop()


def test_shared_dealer_keeps_pending_replies_matched_to_concurrent_requests():
    server = RouterServer(Endpoint('127.0.0.1', 0), lambda operation, body, blob: (body, blob)).start()
    client = DealerClient(server.endpoint)
    barrier = threading.Barrier(8)

    def echo(index):
        request = Message.create('echo', {'index': index}, payload=bytes([index]) * 16384)
        barrier.wait(timeout=5)
        reply = client.request(request)
        assert reply.request_id == request.request_id
        assert reply.body == request.body
        assert reply.payload == request.payload
        return reply.request_id

    try:
        with ThreadPoolExecutor(max_workers=8) as pool:
            request_ids = list(pool.map(echo, range(8)))
        assert len(set(request_ids)) == 8
    finally:
        client.close()
        server.stop()
