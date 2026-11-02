import pytest

from nexus.core.tool.mcp.transport.aiohttp_streamable_http import AiohttpStreamableHTTPTransport


class _ChunkedContent:
    def __init__(self, chunks: list[bytes]):
        self._chunks = chunks

    async def iter_any(self):
        for chunk in self._chunks:
            yield chunk


class _Response:
    def __init__(self, chunks: list[bytes]):
        self.content = _ChunkedContent(chunks)


@pytest.fixture
def anyio_backend():
    return 'asyncio'


@pytest.mark.anyio
async def test_iter_sse_events_handles_long_data_line_split_across_chunks():
    payload = '{"jsonrpc":"2.0","id":1,"result":"' + ('x' * 200_000) + '"}'
    chunks = [
        b'event: message\n',
        f'data: {payload[:100_000]}'.encode(),
        f'{payload[100_000:]}\n\n'.encode(),
    ]
    transport = AiohttpStreamableHTTPTransport('http://example.test/mcp', session=object())  # type: ignore[arg-type]

    events = [event async for event in transport._iter_sse_events(_Response(chunks))]  # type: ignore[arg-type]

    assert len(events) == 1
    assert events[0].event == 'message'
    assert events[0].data == payload


@pytest.mark.anyio
async def test_iter_sse_events_flushes_final_line_without_trailing_newline():
    transport = AiohttpStreamableHTTPTransport('http://example.test/mcp', session=object())  # type: ignore[arg-type]

    events = [event async for event in transport._iter_sse_events(_Response([b'data: done']))]  # type: ignore[arg-type]

    assert len(events) == 1
    assert events[0].data == 'done'


@pytest.mark.anyio
async def test_iter_sse_events_preserves_utf8_split_across_chunks():
    payload = '{"msg":"中文🙂"}'
    encoded = f'data: {payload}\n\n'.encode()
    split_at = encoded.index('中'.encode()[:1])
    chunks = [
        encoded[:split_at + 1],
        encoded[split_at + 1 :],
    ]
    transport = AiohttpStreamableHTTPTransport('http://example.test/mcp', session=object())  # type: ignore[arg-type]

    events = [event async for event in transport._iter_sse_events(_Response(chunks))]  # type: ignore[arg-type]

    assert len(events) == 1
    assert events[0].data == payload
