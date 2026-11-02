"""Unit tests for :mod:`nexus.utils.artifact.uploader` (functional API)."""

from __future__ import annotations

from datetime import datetime, timezone
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from nexus.utils.artifact.models import ArtifactInfo, Folder
from nexus.utils.artifact.uploader import (
    RunFolder,
    _ensure_date_folder,
    _sandbox_hostname,
    artifact_uri_for,
    folder_uri,
    open_run_folder_from_runtime,
    prepare_run_folder,
    upload_bytes,
    upload_file,
)


# ---------------------------------------------------------------------------
# Helpers / fixtures
# ---------------------------------------------------------------------------


def _fake_folder(folder_id: str, name: str) -> Folder:
    return Folder(folder_id=folder_id, name=name, parent_folder_id=None, created_at=None)


def _fake_artifact(artifact_id: str, name: str) -> ArtifactInfo:
    return ArtifactInfo(
        artifact_id=artifact_id,
        name=name,
        content_type='text/plain',
        size_bytes=42,
        status='ready',
        ready_at=None,
        parent_folder_id='folder-run-001',
    )


@pytest.fixture
def mock_client():
    """A minimal ArtifactClient mock with async create_folder / upload_file / upload_bytes."""
    client = MagicMock()
    client.session = MagicMock()  # pretend session is open

    folder_counter = {'n': 0}

    async def _create_folder(name, *, parent_folder_id=None, idempotency_key=None):
        folder_counter['n'] += 1
        return _fake_folder(f'folder-{folder_counter["n"]:03d}', name)

    client.create_folder = AsyncMock(side_effect=_create_folder)
    client.upload_file = AsyncMock(return_value=_fake_artifact('art-abc', 'result.txt'))
    client.upload_bytes = AsyncMock(return_value=_fake_artifact('art-def', 'data.bin'))
    # ``prepare_run_folder`` resolves the owner up front so it can build
    # canonical URIs later; the tests use a stable stand-in login.
    client.whoami = AsyncMock(return_value='junlinyin')
    client.__aenter__ = AsyncMock(return_value=client)
    client.__aexit__ = AsyncMock(return_value=None)
    return client


# ---------------------------------------------------------------------------
# _sandbox_hostname
# ---------------------------------------------------------------------------


def test_sandbox_hostname_prefers_hostname_env_var(monkeypatch):
    monkeypatch.setenv('HOSTNAME', 'nexus-090fb6f888014a1bacfd9fc56fd1d7cf')
    assert _sandbox_hostname() == 'nexus-090fb6f888014a1bacfd9fc56fd1d7cf'


def test_sandbox_hostname_falls_back_to_gethostname(monkeypatch):
    monkeypatch.delenv('HOSTNAME', raising=False)
    with patch('nexus.utils.artifact.uploader.socket') as mock_socket:
        mock_socket.gethostname.return_value = 'my-dev-host'
        assert _sandbox_hostname() == 'my-dev-host'


def test_sandbox_hostname_strips_tenant_prefix(monkeypatch):
    """Gongfeng may prepend a tenant/account prefix to the pod hostname."""
    monkeypatch.setenv('HOSTNAME', 'admin123-nexus-0961a9697d8048088a2b298a1058c235')
    assert _sandbox_hostname() == 'nexus-0961a9697d8048088a2b298a1058c235'


def test_sandbox_hostname_strips_prefix_and_ignores_suffix(monkeypatch):
    """Only the canonical ``nexus-<32hex>`` segment is kept — extra suffixes are dropped."""
    monkeypatch.setenv('HOSTNAME', 'admin123-nexus-0961a9697d8048088a2b298a1058c235-worker-3')
    assert _sandbox_hostname() == 'nexus-0961a9697d8048088a2b298a1058c235'


def test_sandbox_hostname_returns_raw_when_no_canonical_match(monkeypatch):
    """Bare-host / test envs without a nexus-<hex> segment fall through unchanged."""
    monkeypatch.setenv('HOSTNAME', 'devbox-42')
    assert _sandbox_hostname() == 'devbox-42'


# ---------------------------------------------------------------------------
# _ensure_date_folder
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_ensure_date_folder_creates_four_segments(mock_client):
    dt = datetime(2026, 8, 5, 12, 0, 0, tzinfo=timezone.utc)
    folder_id = await _ensure_date_folder(mock_client, dt)

    # Should call create_folder 4 times: tasks / 2026 / 08 / 05
    assert mock_client.create_folder.call_count == 4
    calls = mock_client.create_folder.call_args_list
    assert calls[0].args[0] == 'tasks'
    assert calls[1].args[0] == '2026'
    assert calls[2].args[0] == '08'
    assert calls[3].args[0] == '05'

    # Idempotency keys contain the full path
    assert calls[3].kwargs['idempotency_key'] == 'tasks-dir:/tasks/2026/08/05'

    # Returns the leaf folder id
    assert folder_id == 'folder-004'


@pytest.mark.asyncio
async def test_ensure_date_folder_idempotency_keys_are_path_scoped(mock_client):
    dt = datetime(2026, 1, 1, tzinfo=timezone.utc)
    await _ensure_date_folder(mock_client, dt)
    keys = [c.kwargs['idempotency_key'] for c in mock_client.create_folder.call_args_list]
    assert keys == [
        'tasks-dir:/tasks',
        'tasks-dir:/tasks/2026',
        'tasks-dir:/tasks/2026/01',
        'tasks-dir:/tasks/2026/01/01',
    ]


# ---------------------------------------------------------------------------
# prepare_run_folder
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_prepare_run_folder_name_is_hostname_timestamp(mock_client, monkeypatch):
    """folder name must be <hostname>-<YYYYMMDD-HHMMSS>."""
    monkeypatch.setenv('HOSTNAME', 'nexus-abc123')
    dt = datetime(2026, 8, 5, 9, 0, 0, tzinfo=timezone.utc)
    folder = await prepare_run_folder(mock_client, dt=dt)

    assert folder.folder_path.startswith('/tasks/2026/08/05/')
    folder_leaf = folder.folder_path.split('2026/08/05/')[-1]
    assert folder_leaf == 'nexus-abc123-20260805-090000'


@pytest.mark.asyncio
async def test_prepare_run_folder_total_folder_calls(mock_client, monkeypatch):
    """4 date-folder + 1 run-folder = 5 create_folder calls."""
    monkeypatch.setenv('HOSTNAME', 'nexus-test')
    dt = datetime(2026, 8, 5, tzinfo=timezone.utc)
    await prepare_run_folder(mock_client, dt=dt)
    assert mock_client.create_folder.call_count == 5


@pytest.mark.asyncio
async def test_prepare_run_folder_returns_run_folder(mock_client, monkeypatch):
    """prepare_run_folder must return a plain RunFolder value object."""
    from dataclasses import FrozenInstanceError

    monkeypatch.setenv('HOSTNAME', 'nexus-test')
    dt = datetime(2026, 8, 5, tzinfo=timezone.utc)
    folder = await prepare_run_folder(mock_client, dt=dt)
    assert isinstance(folder, RunFolder)
    # Immutable value object — mutation attempts raise FrozenInstanceError
    with pytest.raises(FrozenInstanceError):
        folder.folder_id = 'other'  # type: ignore[misc]


@pytest.mark.asyncio
async def test_prepare_run_folder_caches_owner_login(mock_client, monkeypatch):
    """prepare_run_folder should ask whoami() once so URIs can be built offline."""
    monkeypatch.setenv('HOSTNAME', 'nexus-test')
    dt = datetime(2026, 8, 5, tzinfo=timezone.utc)
    folder = await prepare_run_folder(mock_client, dt=dt)
    mock_client.whoami.assert_awaited_once()
    assert folder.owner == 'junlinyin'


# ---------------------------------------------------------------------------
# upload_file / upload_bytes
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_upload_file_delegates_to_client(mock_client, monkeypatch):
    monkeypatch.setenv('HOSTNAME', 'nexus-test')
    dt = datetime(2026, 8, 5, tzinfo=timezone.utc)
    folder = await prepare_run_folder(mock_client, dt=dt)
    artifact = await upload_file(mock_client, folder, '/tmp/result.txt')

    mock_client.upload_file.assert_called_once_with(
        '/tmp/result.txt',
        parent_folder_id=folder.folder_id,
        name=None,
        content_type=None,
    )
    assert artifact.artifact_id == 'art-abc'


@pytest.mark.asyncio
async def test_upload_file_passes_name_and_content_type(mock_client, monkeypatch):
    monkeypatch.setenv('HOSTNAME', 'nexus-test')
    dt = datetime(2026, 8, 5, tzinfo=timezone.utc)
    folder = await prepare_run_folder(mock_client, dt=dt)
    await upload_file(
        mock_client, folder, '/tmp/a.pptx', name='slides.pptx', content_type='application/vnd.ms-powerpoint'
    )

    _, kwargs = mock_client.upload_file.call_args
    assert kwargs['name'] == 'slides.pptx'
    assert kwargs['content_type'] == 'application/vnd.ms-powerpoint'


@pytest.mark.asyncio
async def test_upload_bytes_delegates_to_client(mock_client, monkeypatch):
    monkeypatch.setenv('HOSTNAME', 'nexus-test')
    dt = datetime(2026, 8, 5, tzinfo=timezone.utc)
    folder = await prepare_run_folder(mock_client, dt=dt)
    artifact = await upload_bytes(mock_client, folder, b'hello', name='data.bin')

    mock_client.upload_bytes.assert_called_once_with(
        b'hello',
        name='data.bin',
        parent_folder_id=folder.folder_id,
        content_type=None,
    )
    assert artifact.artifact_id == 'art-def'


# ---------------------------------------------------------------------------
# artifact_uri_for / folder_uri
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_artifact_uri_for_composes_owner_qualified_uri(mock_client, monkeypatch):
    monkeypatch.setenv('HOSTNAME', 'nexus-abc123')
    dt = datetime(2026, 8, 5, 9, 0, 0, tzinfo=timezone.utc)
    folder = await prepare_run_folder(mock_client, dt=dt)
    uri = artifact_uri_for(folder, 'report.pdf')

    assert uri == 'artifact:///junlinyin/tasks/2026/08/05/nexus-abc123-20260805-090000/report.pdf'
    from nexus.utils.artifact.uri import parse_artifact_uri

    parsed = parse_artifact_uri(uri)
    assert parsed.owner == 'junlinyin'
    assert parsed.segments[-1] == 'report.pdf'


@pytest.mark.asyncio
async def test_folder_uri_ends_with_trailing_slash(mock_client, monkeypatch):
    monkeypatch.setenv('HOSTNAME', 'nexus-abc123')
    dt = datetime(2026, 8, 5, 9, 0, 0, tzinfo=timezone.utc)
    folder = await prepare_run_folder(mock_client, dt=dt)

    assert folder_uri(folder) == 'artifact:///junlinyin/tasks/2026/08/05/nexus-abc123-20260805-090000/'


def test_uri_helpers_accept_manually_constructed_run_folder():
    """URI helpers are pure functions of RunFolder state, no I/O required."""
    folder = RunFolder(
        folder_id='folder-xyz',
        folder_path='/tasks/2026/08/05/nexus-manual-20260805-000000',
        owner='alice',
    )
    assert folder_uri(folder) == 'artifact:///alice/tasks/2026/08/05/nexus-manual-20260805-000000/'
    assert artifact_uri_for(folder, 'x.bin') == 'artifact:///alice/tasks/2026/08/05/nexus-manual-20260805-000000/x.bin'


# ---------------------------------------------------------------------------
# TaskRunOutput artifact fields
# ---------------------------------------------------------------------------
#
# Convention: the base ``TaskRunOutput`` carries **no artifact fields**.
# The common case for artifact-producing tasks is a single ``artifact_uri``
# field on the task's output subclass. Rarely, a platform layer may want to
# additionally persist ``artifact_id`` for long-term recall — and hide it
# from the terminal log via ``__pformat_exclude__``. These tests enforce
# both parts of that contract.


def test_task_run_output_base_has_no_artifact_fields():
    """The base class stays contract-free — no artifact fields leak in."""
    from nexus.core.task import TaskRunOutput

    fields = set(TaskRunOutput.model_fields.keys())
    assert 'artifact_id' not in fields
    assert 'artifact_uri' not in fields


def test_business_output_declares_artifact_uri_only():
    """The common case: a task output declares a single ``artifact_uri`` field."""
    from nexus.core.task import TaskRunOutput

    class MyOutput(TaskRunOutput):
        artifact_uri: str | None = None
        score: float = 0.0

    out = MyOutput(
        artifact_uri='artifact:///junlinyin/tasks/2026/08/05/run/result.txt',
        score=1.0,
    )
    dumped = out.model_dump()
    assert dumped['artifact_uri'] == 'artifact:///junlinyin/tasks/2026/08/05/run/result.txt'
    assert dumped['score'] == 1.0
    assert 'artifact_path' not in dumped
    assert 'artifact_id' not in dumped


def test_advanced_output_can_also_persist_artifact_id_and_hide_it():
    """Advanced case: platform layer persists ``artifact_id`` and hides it from log."""
    from typing import ClassVar

    from nexus.core.task import TaskRunOutput
    from nexus.utils.common import pformat_for_display

    class MyOutput(TaskRunOutput):
        artifact_uri: str | None = None
        artifact_id: str | None = None
        __pformat_exclude__: ClassVar[frozenset[str]] = frozenset({'artifact_id'})

    out = MyOutput(
        artifact_id='art-xyz',
        artifact_uri='artifact:///junlinyin/tasks/2026/08/05/run/result.txt',
    )
    dumped = out.model_dump()
    assert dumped['artifact_id'] == 'art-xyz'
    assert dumped['artifact_uri'] == 'artifact:///junlinyin/tasks/2026/08/05/run/result.txt'

    rendered = pformat_for_display(out)
    assert 'art-xyz' not in rendered
    assert 'artifact_uri' in rendered


# ---------------------------------------------------------------------------
# open_run_folder_from_runtime (mocked)
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_open_run_folder_from_runtime_calls_artifact_client_from_runtime(mock_client, monkeypatch):
    """open_run_folder_from_runtime() should call ArtifactClient.from_runtime and open the session."""
    monkeypatch.setenv('HOSTNAME', 'nexus-test')
    dt = datetime(2026, 8, 5, tzinfo=timezone.utc)

    with patch(
        'nexus.utils.artifact.client.ArtifactClient.from_runtime',
        new_callable=AsyncMock,
    ) as mock_from_runtime:
        mock_from_runtime.return_value = mock_client
        async with open_run_folder_from_runtime(dt=dt) as (client, folder):
            assert client is mock_client
            assert isinstance(folder, RunFolder)
            # Session was entered via ``async with client``
            mock_client.__aenter__.assert_awaited()

        mock_from_runtime.assert_called_once_with(base_url=None, dev_employee=None)
        # Session was closed on context exit
        mock_client.__aexit__.assert_awaited()


@pytest.mark.asyncio
async def test_open_run_folder_from_runtime_uses_hostname_as_folder_prefix(mock_client, monkeypatch):
    """folder name starts with the sandbox hostname."""
    monkeypatch.setenv('HOSTNAME', 'nexus-090fb6f888014a1bacfd9fc56fd1d7cf')
    dt = datetime(2026, 8, 6, 8, 55, 32, tzinfo=timezone.utc)

    with patch(
        'nexus.utils.artifact.client.ArtifactClient.from_runtime',
        new_callable=AsyncMock,
    ) as mock_from_runtime:
        mock_from_runtime.return_value = mock_client
        async with open_run_folder_from_runtime(dt=dt) as (_client, folder):
            captured_folder = folder

    assert 'nexus-090fb6f888014a1bacfd9fc56fd1d7cf' in captured_folder.folder_path
    assert '20260806-085532' in captured_folder.folder_path


@pytest.mark.asyncio
async def test_open_run_folder_from_runtime_forwards_base_url_and_dev_employee(mock_client, monkeypatch):
    """base_url and dev_employee kwargs must propagate to ArtifactClient.from_runtime."""
    monkeypatch.setenv('HOSTNAME', 'nexus-test')
    dt = datetime(2026, 8, 5, tzinfo=timezone.utc)

    with patch(
        'nexus.utils.artifact.client.ArtifactClient.from_runtime',
        new_callable=AsyncMock,
    ) as mock_from_runtime:
        mock_from_runtime.return_value = mock_client
        async with open_run_folder_from_runtime(
            dt=dt,
            base_url='https://example.woa.com',
            dev_employee='junlinyin',
        ):
            pass

    mock_from_runtime.assert_called_once_with(
        base_url='https://example.woa.com',
        dev_employee='junlinyin',
    )
