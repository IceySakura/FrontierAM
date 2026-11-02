import base64

from types import SimpleNamespace

import pytest

from nexus.extensions.tasks.vlm_stem_agent import task as task_module


PNG_PAYLOAD = base64.b64encode(b'png-image-bytes').decode('ascii')
JPEG_PAYLOAD = base64.b64encode(b'jpeg-image-bytes').decode('ascii')


class _Context:
    def __init__(self, runtime):
        self.runtime = runtime

    async def get_runtime(self, name):
        assert name == 'main'
        return self.runtime


@pytest.mark.asyncio
async def test_uploads_exact_model_facing_payloads(monkeypatch):
    captured = {}

    class Runtime:
        async def upload_file(self, files, *, overwrite=True):
            captured['files'] = files
            captured['overwrite'] = overwrite
            return SimpleNamespace(
                results=[
                    SimpleNamespace(
                        path=item.path,
                        success=True,
                        message=None,
                    )
                    for item in files
                ]
            )

    monkeypatch.setattr(
        task_module,
        'require_context',
        lambda: _Context(Runtime()),
    )

    paths = await task_module.upload_images_to_runtime(
        [
            f'data:image/png;base64,{PNG_PAYLOAD}',
            f'data:image/jpeg;base64,{JPEG_PAYLOAD}',
        ]
    )

    assert paths == [
        '/tmp/jupyter_work/input_0.png',
        '/tmp/jupyter_work/input_1.jpg',
    ]
    assert captured['overwrite'] is True
    assert [
        (item.path, item.content, item.encoding)
        for item in captured['files']
    ] == [
        (
            '/tmp/jupyter_work/input_0.png',
            PNG_PAYLOAD,
            'base64',
        ),
        (
            '/tmp/jupyter_work/input_1.jpg',
            JPEG_PAYLOAD,
            'base64',
        ),
    ]


@pytest.mark.asyncio
async def test_upload_failure_is_reported_before_agent_run(
    monkeypatch,
):
    class Runtime:
        async def upload_file(self, files, *, overwrite=True):
            return SimpleNamespace(
                results=[
                    SimpleNamespace(
                        path=files[0].path,
                        success=False,
                        message='disk full',
                    )
                ]
            )

    monkeypatch.setattr(
        task_module,
        'require_context',
        lambda: _Context(Runtime()),
    )

    with pytest.raises(
        RuntimeError,
        match='input_0.png: disk full',
    ):
        await task_module.upload_images_to_runtime(
            [f'data:image/png;base64,{PNG_PAYLOAD}']
        )


def test_rejects_image_path_count_mismatch():
    with pytest.raises(ValueError, match='equal length'):
        task_module.build_multimodal_user_prompt(
            'question',
            [f'data:image/png;base64,{PNG_PAYLOAD}'],
            [],
        )
