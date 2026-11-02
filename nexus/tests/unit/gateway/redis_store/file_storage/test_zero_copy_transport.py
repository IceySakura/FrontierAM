import gc
import threading
import weakref

import pytest
import zmq

from nexus.gateway.trajectory_store_backends.redis_store.file_storage.blob_codec import decode_blob, encode_blob
from nexus.gateway.trajectory_store_backends.redis_store.file_storage.network import DealerClient, Message, RouterServer
from nexus.gateway.trajectory_store_backends.redis_store.file_storage.types import Endpoint


torch = pytest.importorskip('torch')


def test_zero_copy_reply_outlives_client_and_context():
    expected = torch.arange(1 << 18, dtype=torch.float32)
    blob = bytes(encode_blob({'t': expected}))
    server = RouterServer(Endpoint('127.0.0.1', 0), lambda operation, body, payload: ({}, blob)).start()
    context = zmq.Context(io_threads=2)
    client = DealerClient(server.endpoint, timeout=5, context=context, zero_copy=True)
    try:
        _, payload = client.request('get', {})
        assert isinstance(payload, memoryview)
        tensor = decode_blob(payload)['t']
        del payload
    finally:
        client.close()
        context.term()
        server.stop()
    gc.collect()
    assert torch.equal(tensor, expected)


@pytest.mark.parametrize('message_mode', [False, True], ids=['string', 'message'])
def test_idle_dealer_releases_zero_copy_reply_after_tensor_is_released(monkeypatch, message_mode):
    references = []
    idle = threading.Event()
    original_recv, original_poll = zmq.Socket.recv_multipart, zmq.Socket.poll

    def recv_multipart(socket, *args, **kwargs):
        frames = original_recv(socket, *args, **kwargs)
        if socket.type == zmq.DEALER and kwargs.get('copy') is False:
            references.append(weakref.ref(frames[1]))
        return frames

    def poll(socket, *args, **kwargs):
        if socket.type == zmq.DEALER and references:
            idle.set()
        return original_poll(socket, *args, **kwargs)

    monkeypatch.setattr(zmq.Socket, 'recv_multipart', recv_multipart)
    monkeypatch.setattr(zmq.Socket, 'poll', poll)
    expected = torch.arange(1 << 18, dtype=torch.float32)
    blob = bytes(encode_blob({'t': expected}))
    server = RouterServer(Endpoint('127.0.0.1', 0), lambda operation, body, payload: ({}, blob)).start()
    context = zmq.Context(io_threads=2)
    client = DealerClient(server.endpoint, timeout=5, context=context, zero_copy=True)
    try:
        if message_mode:
            reply = client.request(Message.create('get'))
            payload = reply.payload
        else:
            _, payload = client.request('get', {})
        assert isinstance(payload, memoryview)
        tensor = decode_blob(payload)['t']
        assert idle.wait(5), 'dealer did not return to polling after receiving the reply'
        if message_mode:
            assert reply.payload is payload
            del reply
        assert len(references) == 1
        del payload
        gc.collect()
        assert references[0]() is not None, 'decoded tensor did not retain its zero-copy frame'
        assert torch.equal(tensor, expected)
        del tensor
        gc.collect()
        assert not client.closed
        assert references[0]() is None, 'idle dealer retains the released zero-copy reply frame'
    finally:
        client.close()
        context.term()
        server.stop()
