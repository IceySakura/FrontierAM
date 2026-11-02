"""Exercise workspace delivery with real file executors and archive extraction."""

from __future__ import annotations

import stat
import zipfile

from pathlib import Path
from unittest.mock import AsyncMock

import pytest

from pydantic import ValidationError

from nexus.core import ExecutionContext
from nexus.extensions.agents.visual_agent.agent import VisualAgentInput, run_visual_agent
from nexus.extensions.agents.visual_agent.resources.sandbox_workspace import (
    extract_workspace_archive,
)
from nexus.extensions.agents.visual_agent.resources.workspace import WithWorkspace, upload_workspace
from nexus.runtime.executors.models.file import FileOperationResult, UploadFileResponse


@pytest.mark.parametrize('field', ['workspace', 'upload_workspace'])
def test_old_mapping_input_is_not_accepted(field: str) -> None:
    with pytest.raises(ValidationError, match='valid string'):
        VisualAgentInput.model_validate({'user_prompt': 'hi', field: [{'source': '/src', 'target': '/dst'}]})


@pytest.mark.parametrize('source', ['', '   ', '\x00'])
def test_empty_or_invalid_upload_source_is_rejected(source: str) -> None:
    with pytest.raises(ValidationError, match='non-empty local path'):
        WithWorkspace(upload_workspace=source)


@pytest.mark.parametrize('source_kind', ['text', 'binary', 'zip'])
async def test_individual_files_are_rejected_before_transfer(visual_runtime, tmp_path: Path, source_kind: str) -> None:
    source = tmp_path / f'input.{source_kind}'
    source.write_bytes(b'\x00\xff' if source_kind == 'binary' else b'input')
    target = tmp_path / 'sandbox'
    with pytest.raises(ValueError, match='directory or a valid ZIP'):
        await upload_workspace(visual_runtime, str(source), str(target))
    assert not target.exists()
    assert visual_runtime.uploads == []
    assert visual_runtime.staging_dirs == []


async def test_directory_maps_contents_to_root_and_merges(visual_runtime, tmp_path: Path) -> None:
    source = tmp_path / 'host tree'
    (source / 'nested' / 'empty').mkdir(parents=True)
    (source / '.hidden').write_text('hidden')
    (source / 'nested' / 'data.bin').write_bytes(b'\x00\xff')
    script = source / 'run.sh'
    script.write_text('#!/bin/sh\nexit 0\n')
    script.chmod(0o755)
    target = tmp_path / "sandbox's directory $(literal)"
    target.mkdir()
    (target / '.hidden').write_text('stale')
    (target / 'keep.txt').write_text('unrelated')

    await upload_workspace(visual_runtime, str(source), str(target))

    assert (target / '.hidden').read_text() == 'hidden'
    assert (target / 'nested' / 'data.bin').read_bytes() == b'\x00\xff'
    assert (target / 'nested' / 'empty').is_dir()
    assert (target / 'keep.txt').read_text() == 'unrelated'
    assert (target / 'run.sh').stat().st_mode & stat.S_IXUSR
    assert not (target / source.name).exists()
    assert visual_runtime.staging_dirs
    assert all(not path.exists() for path in visual_runtime.staging_dirs)


async def test_empty_directory_is_created(visual_runtime, tmp_path: Path) -> None:
    source = tmp_path / 'empty'
    source.mkdir()
    target = tmp_path / 'sandbox' / 'empty'
    await upload_workspace(visual_runtime, str(source), str(target))
    assert target.is_dir()


async def test_zip_extracts_to_directory_and_keeps_internal_structure(visual_runtime, tmp_path: Path) -> None:
    source = tmp_path / 'workspace.zip'
    with zipfile.ZipFile(source, 'w') as archive:
        archive.writestr('project/nested/data.txt', 'hello')
        archive.writestr('project/empty/', '')
    extracted = tmp_path / 'sandbox' / 'directory.zip'

    await upload_workspace(visual_runtime, str(source), str(extracted))

    assert (extracted / 'project' / 'nested' / 'data.txt').read_text() == 'hello'
    assert (extracted / 'project' / 'empty').is_dir()
    assert not (extracted / 'data.txt').exists()
    assert all(not path.exists() for path in visual_runtime.staging_dirs)


async def test_zip_inside_directory_stays_a_zip(visual_runtime, tmp_path: Path) -> None:
    source = tmp_path / 'source'
    source.mkdir()
    with zipfile.ZipFile(source / 'input.zip', 'w') as archive:
        archive.writestr('data.txt', 'packed')
    target = tmp_path / 'sandbox'
    await upload_workspace(visual_runtime, str(source), str(target))
    assert (target / 'input.zip').read_bytes() == (source / 'input.zip').read_bytes()
    assert not (target / 'data.txt').exists()


@pytest.mark.parametrize('target', ['relative/file', '/workspace/../escape', '//host/path', '/workspace\\file'])
def test_ambiguous_targets_are_rejected(target: str) -> None:
    with pytest.raises(ValidationError, match='absolute POSIX'):
        WithWorkspace(workspace=target)


async def test_invalid_source_fails_before_upload(visual_runtime, tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError, match='does not exist'):
        await upload_workspace(visual_runtime, str(tmp_path / 'missing'), str(tmp_path / 'sandbox'))


def test_existing_file_cannot_be_directory_target(tmp_path: Path) -> None:
    source = tmp_path / 'empty.zip'
    with zipfile.ZipFile(source, 'w'):
        pass
    target = tmp_path / 'existing.txt'
    target.write_text('original')
    with pytest.raises(ValueError, match='wrong file type'):
        extract_workspace_archive(str(source), str(target))
    assert target.read_text() == 'original'


async def test_local_directory_symlinks_are_rejected(visual_runtime, tmp_path: Path) -> None:
    source = tmp_path / 'source'
    source.mkdir()
    (source / 'loop').symlink_to(source, target_is_directory=True)
    with pytest.raises(ValueError, match='only supports regular files and directories'):
        await upload_workspace(visual_runtime, str(source), str(tmp_path / 'sandbox'))
    assert not visual_runtime.uploads


@pytest.mark.parametrize('member', ['../escape', '/absolute', 'C:/drive', 'dir\\escape'])
async def test_unsafe_zip_fails_before_writing_and_cleans_staging(visual_runtime, tmp_path: Path, member: str) -> None:
    source = tmp_path / 'unsafe.zip'
    with zipfile.ZipFile(source, 'w') as archive:
        archive.writestr('valid.txt', 'must not be written')
        archive.writestr(member, 'outside')
    target = tmp_path / 'sandbox'
    with pytest.raises(ValueError, match='Unsafe workspace ZIP member'):
        await upload_workspace(visual_runtime, str(source), str(target))
    assert not target.exists()
    assert visual_runtime.staging_dirs
    assert all(not path.exists() for path in visual_runtime.staging_dirs)


def test_zip_symlink_members_are_rejected(tmp_path: Path) -> None:
    source = tmp_path / 'unsafe.zip'
    member = zipfile.ZipInfo('link')
    member.create_system = 3
    member.external_attr = (stat.S_IFLNK | 0o777) << 16
    with zipfile.ZipFile(source, 'w') as archive:
        archive.writestr(member, 'outside')
    target = tmp_path / 'sandbox'
    with pytest.raises(ValueError, match='not a regular file'):
        extract_workspace_archive(str(source), str(target))


@pytest.mark.parametrize('members', [['file', 'file/nested'], ['file/nested', 'file'], ['file', 'file']])
@pytest.mark.filterwarnings('ignore:Duplicate name:UserWarning')
def test_conflicting_zip_members_are_rejected(tmp_path: Path, members: list[str]) -> None:
    source = tmp_path / 'conflict.zip'
    with zipfile.ZipFile(source, 'w') as archive:
        for name in members:
            archive.writestr(name, 'data')
    target = tmp_path / 'sandbox'
    with pytest.raises(ValueError, match='Conflicting workspace ZIP'):
        extract_workspace_archive(str(source), str(target))
    assert not target.exists()


@pytest.mark.parametrize('empty_response', [True, False])
async def test_unsuccessful_upload_aborts_and_cleans_staging(
    visual_runtime, tmp_path: Path, empty_response: bool
) -> None:
    source = tmp_path / 'source'
    source.mkdir()
    visual_runtime.upload_response = UploadFileResponse(
        results=[] if empty_response else [FileOperationResult(path='archive', success=False, message='disk full')]
    )
    with pytest.raises(RuntimeError, match='Workspace upload failed'):
        await upload_workspace(visual_runtime, str(source), str(tmp_path / 'sandbox'))
    assert visual_runtime.staging_dirs
    assert all(not path.exists() for path in visual_runtime.staging_dirs)


def test_zip_dot_root_keeps_contents_at_target(tmp_path: Path) -> None:
    source = tmp_path / 'workspace.zip'
    with zipfile.ZipFile(source, 'w') as archive:
        archive.writestr('./', '')
        archive.writestr('./nested/file.txt', 'data')
    target = tmp_path / 'sandbox'
    extract_workspace_archive(str(source), str(target))
    assert (target / 'nested' / 'file.txt').read_text() == 'data'


@pytest.mark.parametrize('nested', [False, True])
def test_destination_symlinks_do_not_write_outside_workspace(tmp_path, nested):
    source = tmp_path / 'workspace.zip'
    with zipfile.ZipFile(source, 'w') as archive:
        archive.writestr('link/data.txt' if nested else 'data.txt', 'new')
    outside = tmp_path / 'outside'
    outside.mkdir()
    (outside / 'data.txt').write_text('original')
    target = tmp_path / 'sandbox'
    if nested:
        target.mkdir()
    (target / 'link' if nested else target).symlink_to(outside)
    with pytest.raises(ValueError, match='contains a symlink'):
        extract_workspace_archive(str(source), str(target))
    assert (outside / 'data.txt').read_text() == 'original'


@pytest.mark.parametrize('failure', ['missing_source', 'upload_failed', 'no_runtime'])
async def test_workspace_preparation_failure_stops_agent_before_inference(
    visual_runtime, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, failure: str, main_delegate
) -> None:
    source = tmp_path / 'inputs'
    source.mkdir()
    get_runtime = AsyncMock(return_value=visual_runtime)
    expected_error = RuntimeError
    if failure == 'missing_source':
        source.rmdir()
        expected_error = FileNotFoundError
    elif failure == 'upload_failed':
        visual_runtime.upload_response = UploadFileResponse(
            results=[
                FileOperationResult(path=str(tmp_path / 'sandbox'), success=False, message='disk full'),
            ]
        )
    elif failure == 'no_runtime':
        get_runtime.side_effect = RuntimeError('No runtime provider configured')
    else:
        pytest.fail(f'Unknown failure scenario: {failure}')
    monkeypatch.setattr(ExecutionContext, 'get_runtime', get_runtime)

    async with ExecutionContext():
        with pytest.raises(expected_error):
            await run_visual_agent(
                VisualAgentInput(
                    user_prompt='Analyze', upload_workspace=str(source), workspace=str(tmp_path / 'sandbox')
                )
            )
    main_delegate.assert_not_awaited()


async def test_agent_without_workspace_does_not_acquire_runtime(monkeypatch: pytest.MonkeyPatch, main_delegate) -> None:
    async def no_runtime(self, *args, **kwargs):
        raise AssertionError('Empty workspace must not acquire a runtime')

    monkeypatch.setattr(ExecutionContext, 'get_runtime', no_runtime)
    async with ExecutionContext(task_config={'user_prompt': 'Analyze'}):
        output = await run_visual_agent(VisualAgentInput(user_prompt='Analyze'))
    assert output.workspace == '/tmp'
    assert main_delegate.call_args.args[0].user_prompt == 'Analyze'


async def test_oversized_workspace_archive_is_not_read_or_uploaded(visual_runtime, tmp_path, monkeypatch):
    from nexus.extensions.agents.visual_agent.resources import media

    source = tmp_path / 'workspace.zip'
    with zipfile.ZipFile(source, 'w') as archive:
        archive.writestr('file.txt', 'content')
    monkeypatch.setattr(media, 'MAX_MEDIA_BYTES', 4)
    with pytest.raises(ValueError, match='64 MiB'):
        await upload_workspace(visual_runtime, str(source), str(tmp_path / 'sandbox'))
    assert visual_runtime.uploads == []
    assert all(not path.exists() for path in visual_runtime.staging_dirs)
