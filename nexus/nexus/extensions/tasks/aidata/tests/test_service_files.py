from __future__ import annotations

import base64

from pathlib import Path
from types import SimpleNamespace

import pytest

from nexus.extensions.tasks.aidata.engine.case_spec import _parse_service_spec
from nexus.extensions.tasks.aidata.engine.models import AidataService, AidataServiceFile
from nexus.extensions.tasks.aidata.engine.services import _upload_service_files
from nexus.runtime.executors.command import CommandRunStatus


class _FakeRuntime:
    def __init__(self) -> None:
        self.commands: list[tuple[str, dict]] = []
        self.uploads: list = []

    async def run_command(self, command: str, **kwargs):
        self.commands.append((command, kwargs))
        return SimpleNamespace(status=CommandRunStatus.Finished, return_code=0, stdout='', stderr='')

    async def upload_file(self, *, files, overwrite: bool):
        self.uploads.extend(files)
        return SimpleNamespace(results=[SimpleNamespace(success=True, message='') for _ in files])


def test_parse_service_files() -> None:
    service = _parse_service_spec(
        {
            'name': 'backend',
            'runtime_name': 'service_backend',
            'files': [
                {
                    'source': 'assets/seed.json',
                    'target': '/app/data/seed.json',
                },
            ],
        },
        Path('/cases/example/task.yaml'),
    )

    assert service.files == [AidataServiceFile(source='assets/seed.json', target='/app/data/seed.json')]


@pytest.mark.parametrize(
    ('file_spec', 'error'),
    [
        ({'source': '/etc/passwd', 'target': '/app/passwd'}, 'source must be relative'),
        ({'source': '../secret.txt', 'target': '/app/secret.txt'}, 'source must stay inside case dir'),
        ({'source': 'assets/seed.json', 'target': 'app/data/seed.json'}, 'target must be an absolute path'),
    ],
)
def test_parse_service_files_rejects_bad_paths(file_spec: dict[str, str], error: str) -> None:
    with pytest.raises(ValueError, match=error):
        _parse_service_spec(
            {
                'name': 'backend',
                'runtime_name': 'service_backend',
                'files': [file_spec],
            },
            Path('/cases/example/task.yaml'),
        )


@pytest.mark.asyncio
async def test_upload_service_files_creates_parent_and_uploads_base64(tmp_path: Path) -> None:
    source = tmp_path / 'assets' / 'seed.json'
    source.parent.mkdir()
    payload = b'{"ok": true}\n'
    source.write_bytes(payload)
    service = AidataService(
        name='backend',
        runtime_name='service_backend',
        files=[AidataServiceFile(source='assets/seed.json', target='/app/data/seed.json')],
    )
    runtime = _FakeRuntime()

    await _upload_service_files(runtime, service, case_dir=tmp_path)

    assert runtime.commands == [('mkdir -p /app/data', {'timeout': 10})]
    assert len(runtime.uploads) == 1
    uploaded = runtime.uploads[0]
    assert uploaded.path == '/app/data/seed.json'
    assert uploaded.encoding == 'base64'
    assert base64.b64decode(uploaded.content.encode('ascii')) == payload
