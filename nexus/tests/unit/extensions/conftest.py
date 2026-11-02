"""Shared fixtures for visual agent and task tests."""

from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest


class LocalRuntime:
    def __init__(self):
        self.uploads = []
        self.batches = []
        self.staging_dirs = []
        self.upload_response = None
        self.write_files = True

    async def upload_file(self, files, *, overwrite=True):
        from nexus.runtime.executors.file import upload_file
        from nexus.runtime.executors.models.file import FileOperationResult, UploadFileRequest, UploadFileResponse

        self.uploads.extend(files)
        self.batches.append(files)
        if self.upload_response is not None:
            return self.upload_response
        if self.write_files:
            return await upload_file(UploadFileRequest(files=files, overwrite=overwrite))
        return UploadFileResponse(results=[FileOperationResult(path=f.path, success=True) for f in files])

    async def exec(self, code):
        from nexus.utils.exec import exec_code

        result = await exec_code(code)
        if isinstance(result, str):
            self.staging_dirs.append(Path(result))
        return result

    async def invoke(self, target, *args):
        from nexus.extensions.agents.visual_agent.resources.sandbox_workspace import extract_workspace_archive

        assert target == 'nexus.extensions.agents.visual_agent.resources.sandbox_workspace::extract_workspace_archive'
        extract_workspace_archive(*args)


@pytest.fixture
def visual_runtime():
    return LocalRuntime()


@pytest.fixture
def fake_runtime(monkeypatch, visual_runtime):
    from nexus.extensions.agents.visual_agent.resources import images

    visual_runtime.write_files = False
    context = SimpleNamespace(runtime_provider=object(), get_runtime=AsyncMock(return_value=visual_runtime))
    monkeypatch.setattr(images, 'require_context', lambda: context)
    return visual_runtime.batches


@pytest.fixture
def no_runtime(monkeypatch):
    from nexus.extensions.agents.visual_agent.resources import images

    monkeypatch.setattr(images, 'require_context', lambda: SimpleNamespace(runtime_provider=None))


@pytest.fixture
def main_delegate(monkeypatch):
    from nexus.extensions.agents.main.agent import MainAgentOutput
    from nexus.extensions.agents.visual_agent import agent

    delegate = AsyncMock(
        return_value=MainAgentOutput(
            main_trajectory=[{'role': 'assistant', 'content': 'ok'}],
            exit_status='finished',
            iterations=1,
            final_answer='ok',
        )
    )
    monkeypatch.setattr(agent, 'run_main_agent', delegate)
    return delegate


@pytest.fixture
def captured_main_input(main_delegate):
    captured = []

    async def capture(value):
        captured.append(value)
        return main_delegate.return_value

    main_delegate.side_effect = capture
    return captured
