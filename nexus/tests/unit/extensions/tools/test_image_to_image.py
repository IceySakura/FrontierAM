from unittest.mock import mock_open

import pytest

from nexus.extensions.tools.web_search import image_to_image


@pytest.mark.asyncio
async def test_format_search_results_embeds_image_url_fallback(monkeypatch: pytest.MonkeyPatch) -> None:
    image_url = 'https://images.example/original.png'

    async def fake_download_images(urls, **kwargs):
        assert urls == [image_url]
        return [(image_url, b'image-data', 'image/png')]

    monkeypatch.setattr(image_to_image, '_download_images', fake_download_images)

    result = await image_to_image.format_search_results(
        {
            'organic': [
                {
                    'title': 'Result',
                    'source': 'Example',
                    'link': 'https://example/result',
                    'imageUrl': image_url,
                }
            ]
        },
        query='query',
        num=1,
        return_images=True,
    )

    assert isinstance(result, list)
    assert result[-1]['type'] == 'image_url'
    assert result[-1]['image_url']['url'].startswith('data:image/png;base64,')


@pytest.mark.asyncio
async def test_upload_local_image_does_not_preload_file_when_cos_succeeds(monkeypatch: pytest.MonkeyPatch) -> None:
    image_path = '/tmp/image.png'
    cos_url = 'https://cos.example/image.png'
    open_mock = mock_open(read_data=b'image-data')

    class _FakeLoop:
        async def run_in_executor(self, executor, func, *args):
            return func(*args)

    monkeypatch.setattr(image_to_image.os.path, 'isfile', lambda path: path == image_path)
    monkeypatch.setattr('builtins.open', open_mock)
    monkeypatch.setattr(image_to_image, '_cos_upload_image', lambda path, config: cos_url)
    monkeypatch.setattr(image_to_image.asyncio, 'get_running_loop', lambda: _FakeLoop())

    result = await image_to_image.upload_local_image(
        image_path,
        cos_config=image_to_image.CosUploadConfig(
            secret_id='secret-id',
            secret_key='secret-key',
            bucket='bucket',
            prefix='prefix',
            region='region',
        ),
    )

    assert result == cos_url
    open_mock.assert_not_called()
