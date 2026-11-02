import base64
import copy

from types import SimpleNamespace

import pytest

from nexus.gateway import media
from nexus.gateway.backend import HyTunerServerBackend, OpenAIAPIBackend, VllmCompletionsBackend
from nexus.gateway.media import (
    HOST_CONSUMER_SOURCES,
    REMOTE_CONSUMER_SOURCES,
    MediaPart,
    MediaResolver,
    MediaSourceKind,
    choose_target,
    classify_source,
    register_converter,
)
from nexus.gateway.processor import Processor
from nexus.gateway.types import BackendResult


PNG = b'\x89PNG-bytes'


def _data_url(data: bytes, mime: str = 'image/png') -> str:
    return f'data:{mime};base64,{base64.b64encode(data).decode()}'


def _image(url: str, **extra) -> dict:
    return {'type': 'image_url', 'image_url': {'url': url, **extra}}


@pytest.mark.parametrize(
    ('url', 'kind'),
    [
        ('https://x/a.png', MediaSourceKind.REMOTE_URL),
        ('http://x/a.png', MediaSourceKind.REMOTE_URL),
        ('data:image/png;base64,AAAA', MediaSourceKind.DATA_URL),
        ('file:///ceph/a.png', MediaSourceKind.LOCAL_FILE),
        ('/ceph/a.png', MediaSourceKind.LOCAL_FILE),
        ('images/a.png', None),
    ],
)
def test_classify_source(url, kind):
    assert classify_source(url) is kind


def test_choose_target_converts_only_what_the_consumer_cannot_read():
    assert choose_target(MediaSourceKind.LOCAL_FILE, HOST_CONSUMER_SOURCES) is None
    assert choose_target(MediaSourceKind.REMOTE_URL, REMOTE_CONSUMER_SOURCES) is None
    assert choose_target(MediaSourceKind.LOCAL_FILE, REMOTE_CONSUMER_SOURCES) is MediaSourceKind.DATA_URL
    with pytest.raises(ValueError, match='No media conversion from local_file'):
        choose_target(MediaSourceKind.LOCAL_FILE, {MediaSourceKind.REMOTE_URL})


def test_backends_declare_their_consumers():
    assert HyTunerServerBackend.accepted_media_sources == HOST_CONSUMER_SOURCES
    assert VllmCompletionsBackend.accepted_media_sources == HOST_CONSUMER_SOURCES
    assert OpenAIAPIBackend.accepted_media_sources == REMOTE_CONSUMER_SOURCES


async def test_resolver_returns_input_when_nothing_needs_converting(tmp_path):
    image = tmp_path / 'a.png'
    image.write_bytes(PNG)
    messages = [{'role': 'user', 'content': [_image(str(image)), _image('https://x/b.png')]}]
    assert await MediaResolver().resolve(messages, HOST_CONSUMER_SOURCES) is messages
    text_only = [{'role': 'user', 'content': 'text'}]
    assert await MediaResolver().resolve(text_only, REMOTE_CONSUMER_SOURCES) is text_only


async def test_resolver_rewrites_host_files_for_remote_consumers(tmp_path):
    image = tmp_path / 'a.jpg'
    image.write_bytes(b'jpeg')
    messages = [
        {'role': 'system', 'content': ''},
        {
            'role': 'user',
            'content': [
                _image(str(image), detail='low'),
                _image(f'file://{image}'),
                _image('https://x/b.png'),
                {'type': 'text', 'text': 'q'},
            ],
        },
    ]
    original = copy.deepcopy(messages)
    resolved = await MediaResolver().resolve(messages, REMOTE_CONSUMER_SOURCES)
    expected = _data_url(b'jpeg', 'image/jpeg')
    assert resolved[0] is messages[0]
    assert resolved[1]['content'] == [
        _image(expected, detail='low'),
        _image(expected),
        _image('https://x/b.png'),
        {'type': 'text', 'text': 'q'},
    ]
    assert messages == original


async def test_resolver_rejects_non_image_host_file(tmp_path):
    path = tmp_path / 'notes.txt'
    path.write_text('x')
    with pytest.raises(ValueError, match="has type 'text/plain'"):
        await MediaResolver().resolve([{'role': 'user', 'content': [_image(str(path))]}], REMOTE_CONSUMER_SOURCES)


async def test_resolver_rejects_host_file_without_extension(tmp_path):
    path = tmp_path / '756ab247'
    path.write_bytes(PNG)
    with pytest.raises(ValueError, match='has no recognised file extension'):
        await MediaResolver().resolve([{'role': 'user', 'content': [_image(str(path))]}], REMOTE_CONSUMER_SOURCES)


async def test_resolver_reports_missing_host_file(tmp_path):
    with pytest.raises(FileNotFoundError):
        await MediaResolver().resolve(
            [{'role': 'user', 'content': [_image(str(tmp_path / 'missing.png'))]}], REMOTE_CONSUMER_SOURCES
        )


async def test_resolver_converts_each_reference_once_per_instance(tmp_path, monkeypatch):
    image = tmp_path / 'a.png'
    image.write_bytes(PNG)
    calls = []

    def counting(part: MediaPart) -> str:
        calls.append(part.url)
        return 'data:image/png;base64,cached'

    monkeypatch.setitem(media._CONVERTERS, (MediaSourceKind.LOCAL_FILE, MediaSourceKind.DATA_URL), counting)
    messages = [{'role': 'user', 'content': [_image(str(image)), _image(str(image))]}]
    first, second = MediaResolver(), MediaResolver()
    await first.resolve(messages, REMOTE_CONSUMER_SOURCES)
    await first.resolve(messages, REMOTE_CONSUMER_SOURCES)
    await second.resolve(messages, REMOTE_CONSUMER_SOURCES)
    assert calls == [str(image), str(image)]


def test_register_converter_rejects_duplicates():
    with pytest.raises(ValueError, match='Duplicate media converter'):
        register_converter(MediaSourceKind.LOCAL_FILE, MediaSourceKind.DATA_URL, lambda part: part.url)


class _RecordingBackend:
    """A message-level backend that declares nothing, like a newly added backend."""

    def __init__(self):
        self.calls = []

    async def create_completion(self, request_id, *, messages=None, **kwargs):
        self.calls.append(messages)
        return BackendResult(message={'role': 'assistant', 'content': 'ok'})


async def test_processor_resolves_for_undeclared_backends_once_per_trajectory(tmp_path, monkeypatch):
    image = tmp_path / 'a.png'
    image.write_bytes(PNG)
    reads = []
    real = media.local_file_to_data_url

    def counting(part):
        reads.append(part.url)
        return real(part)

    monkeypatch.setitem(media._CONVERTERS, (MediaSourceKind.LOCAL_FILE, MediaSourceKind.DATA_URL), counting)
    backend = _RecordingBackend()
    processor = Processor(backend=backend)
    first = [{'role': 'user', 'content': [_image(str(image)), {'type': 'text', 'text': 'q1'}]}]
    second = [*first, {'role': 'assistant', 'content': 'a1'}, {'role': 'user', 'content': 'q2'}]

    await processor.create_completion('r', first)
    await processor.create_completion('r', second)

    assert reads == [str(image)]
    assert backend.calls[0][0]['content'][0] == _image(_data_url(PNG))
    assert backend.calls[1][0]['content'][0] == _image(_data_url(PNG))
    assert backend.calls[1][1:] == second[1:]
    assert second[0]['content'][0]['image_url']['url'] == str(image)


async def test_processor_keeps_host_files_for_host_consumers(tmp_path):
    image = tmp_path / 'a.png'
    image.write_bytes(PNG)
    backend = _RecordingBackend()
    backend.accepted_media_sources = HOST_CONSUMER_SOURCES
    messages = [{'role': 'user', 'content': [_image(str(image))]}]

    await Processor(backend=backend).create_completion('r', messages)

    assert backend.calls[0] is messages


async def test_openai_backend_sends_resolved_images_through_processor(tmp_path):
    image = tmp_path / 'a.png'
    image.write_bytes(PNG)
    sent = []

    class Client:
        async def create_completion(self, **kwargs):
            sent.append(kwargs['messages'])
            message = SimpleNamespace(content='ok', tool_calls=None)
            choice = SimpleNamespace(message=message, finish_reason='stop', logprobs=None)
            return SimpleNamespace(choices=[choice], usage=None)

    processor = Processor(backend=OpenAIAPIBackend(Client()))
    await processor.create_completion('r', [{'role': 'user', 'content': [_image(str(image))]}])

    assert sent[0][0]['content'][0] == _image(_data_url(PNG))
