"""Staging a source video into the runtime, and tearing it down again."""

import base64

from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

from nexus.extensions.tasks.mm_agent import task as task_module
from nexus.extensions.tasks.mm_agent.utils import video_utils
from nexus.extensions.tasks.mm_agent.utils.video_utils import (
    MAX_UPLOAD_BYTES,
    copy_source_video,
    prepare_video_source,
    remove_staged_video,
    staged_video_path,
    stat_file_here,
)
from nexus.runtime.executors.models.file import FileOperationResult, UploadFileResponse


class TestStagedVideoPath:
    def test_is_scoped_by_run_id(self):
        # GRPO runs n rollouts of one prompt concurrently. A scope keyed on the video's own
        # identity gives them all the same file, and the first to finish deletes it out from
        # under the rest.
        assert staged_video_path('/data/a.mp4', 'run-1') == '/tmp/run-1/video_0.mp4'
        assert staged_video_path('/data/a.mp4', 'run-2') == '/tmp/run-2/video_0.mp4'

    def test_the_basename_is_the_one_the_model_was_trained_on(self):
        # HyVision-Toolbox stages media as video_0<ext>, and the SFT data uses that name.
        assert staged_video_path('/some/very/long/shared/path.mkv', 'run-1').endswith('/video_0.mkv')

    def test_an_unknown_suffix_falls_back_to_mp4(self):
        assert staged_video_path('/data/clip', 'run-1') == '/tmp/run-1/video_0.mp4'

    def test_a_missing_run_id_never_shares(self):
        first, second = staged_video_path('/data/a.mp4', None), staged_video_path('/data/a.mp4', None)
        assert first != second
        assert first.startswith('/tmp/anon_')


class TestCopySourceVideo:
    def test_writing_the_staged_copy_leaves_the_source_alone(self, tmp_path):
        # A symlink would be followed: a cell writing to the staged path would overwrite the
        # dataset's own video in place, and the sandbox runs as root.
        source = tmp_path / 'source.mp4'
        source.write_bytes(b'ORIGINAL')
        destination = tmp_path / 'run-1' / 'video_0.mp4'

        copy_source_video(str(source), str(destination))
        destination.write_bytes(b'CLOBBERED')

        assert not destination.is_symlink()
        assert source.read_bytes() == b'ORIGINAL'

    def test_a_missing_source_is_named_in_the_error(self, tmp_path):
        with pytest.raises(ValueError, match='not a file'):
            copy_source_video(str(tmp_path / 'absent.mp4'), str(tmp_path / 'out.mp4'))

    def test_no_partial_copy_is_left_behind(self, tmp_path, monkeypatch):
        # The staged path must never hold a half-written file: the model would cut clips out
        # of it and train on truncated observations.
        source = tmp_path / 'source.mp4'
        source.write_bytes(b'ORIGINAL')
        destination = tmp_path / 'run-1' / 'video_0.mp4'

        def explode(*args, **kwargs):
            raise OSError('disk full')

        monkeypatch.setattr(video_utils.shutil, 'copyfile', explode)
        with pytest.raises(OSError, match='disk full'):
            copy_source_video(str(source), str(destination))

        assert not destination.exists()
        assert list(destination.parent.iterdir()) == []


class TestRemoveStagedVideo:
    def test_removes_the_file_and_its_run_directory(self, tmp_path):
        # Under the local runtime this process is the runtime, so without it /tmp keeps one
        # full copy of every source the run ever touched.
        staged = tmp_path / 'run-1' / 'video_0.mp4'
        staged.parent.mkdir()
        staged.write_bytes(b'x')

        remove_staged_video(str(staged))

        assert not staged.exists()
        assert not staged.parent.exists()

    def test_is_idempotent(self, tmp_path):
        # It runs in the task's finally, which must not fail a sample that answered correctly.
        remove_staged_video(str(tmp_path / 'run-1' / 'video_0.mp4'))

    def test_keeps_a_run_directory_that_still_holds_something(self, tmp_path):
        staged = tmp_path / 'run-1' / 'video_0.mp4'
        staged.parent.mkdir()
        staged.write_bytes(b'x')
        (staged.parent / 'clip.mp4').write_bytes(b'y')

        remove_staged_video(str(staged))

        assert not staged.exists()
        assert staged.parent.is_dir()


class TestStatFileHere:
    def test_reports_identity_not_mere_existence(self, tmp_path):
        # Sharing is decided by (device, inode, size): a sandbox may have an unrelated file at
        # the same path, and cutting clips out of it would train on the wrong observations.
        source = tmp_path / 'source.mp4'
        source.write_bytes(b'12345')

        result = stat_file_here(str(source))

        assert result['size'] == 5
        assert (result['dev'], result['ino']) == (source.stat().st_dev, source.stat().st_ino)

    def test_a_missing_path_is_none(self, tmp_path):
        assert stat_file_here(str(tmp_path / 'absent.mp4')) is None

    def test_a_directory_is_none(self, tmp_path):
        assert stat_file_here(str(tmp_path)) is None


@pytest.fixture
def staging_runtime(monkeypatch, tmp_path):
    """A runtime that executes invoked helpers in-process and records the route taken."""
    calls = []

    async def invoke(target, *args, **kwargs):
        calls.append((getattr(target, '__name__', target), args))
        name = getattr(target, '__name__', '')
        if name == 'probe_source_video_here':
            return {'duration': 12.5, 'width': 64, 'height': 64}
        if name in ('stat_file_here', 'copy_source_video', 'remove_staged_video'):
            return target(*args)
        if name == 'download_source_video':
            return {'size_bytes': 10, 'download_seconds': 0.1}
        return None

    async def upload_file(files, overwrite):
        return UploadFileResponse(results=[FileOperationResult(path=item.path, success=True) for item in files])

    runtime = SimpleNamespace(invoke=AsyncMock(side_effect=invoke), upload_file=AsyncMock(side_effect=upload_file))
    context = SimpleNamespace(get_runtime=AsyncMock(return_value=runtime), run_id='run-1')
    monkeypatch.setattr(video_utils, 'require_context', lambda: context)
    # Keep staged paths inside tmp_path rather than the real /tmp.
    monkeypatch.setattr(
        video_utils, 'staged_video_path', lambda path, run_id: str(tmp_path / str(run_id) / 'video_0.mp4')
    )
    return runtime, calls


class TestPrepareVideoSource:
    async def test_a_shared_source_is_copied_rather_than_uploaded(self, tmp_path, staging_runtime):
        runtime, calls = staging_runtime
        source = tmp_path / 'source.mp4'
        source.write_bytes(b'ORIGINAL')

        state = await prepare_video_source([{'video': str(source)}], None, 60)

        assert [name for name, _ in calls] == ['stat_file_here', 'copy_source_video', 'probe_source_video_here']
        assert state == {'source_path': str(tmp_path / 'run-1' / 'video_0.mp4'), 'duration': 12.5}
        runtime.upload_file.assert_not_awaited()

    async def test_a_presigned_url_wins_over_uploading_bytes(self, tmp_path, staging_runtime):
        runtime, calls = staging_runtime

        state = await prepare_video_source([{'video': '/not/visible.mp4'}], ['https://cos/a.mp4'], 60)

        assert [name for name, _ in calls] == ['download_source_video', 'probe_source_video_here']
        assert state['duration'] == 12.5
        runtime.upload_file.assert_not_awaited()

    async def test_a_host_only_source_is_uploaded(self, tmp_path, staging_runtime, monkeypatch):
        runtime, _ = staging_runtime
        source = tmp_path / 'source.mp4'
        source.write_bytes(b'ORIGINAL')
        # Visible to the Host, invisible to the runtime: the upload route's reason to exist.
        monkeypatch.setattr(video_utils, 'stat_file_here', lambda path: None)

        await prepare_video_source([{'video': str(source)}], None, 60)

        item = runtime.upload_file.call_args.args[0][0]
        assert base64.b64decode(item.content) == b'ORIGINAL'

    async def test_an_upload_over_the_limit_says_what_to_do_instead(self, tmp_path, staging_runtime, monkeypatch):
        # The upload route holds the raw bytes, their Base64 and the pickle in Host memory at
        # once, so it has to refuse before the allocator does.
        runtime, _ = staging_runtime
        source = tmp_path / 'source.mp4'
        source.write_bytes(b'x')
        monkeypatch.setattr(video_utils, 'stat_file_here', lambda path: None)
        monkeypatch.setattr(video_utils, 'MAX_UPLOAD_BYTES', 0)

        with pytest.raises(ValueError, match='video_cos_urls'):
            await prepare_video_source([{'video': str(source)}], None, 60)

        runtime.upload_file.assert_not_awaited()

    async def test_a_failed_probe_takes_its_own_staged_file_with_it(self, tmp_path, staging_runtime):
        # The bytes are on disk by the time probing runs, and the caller only starts tracking
        # the staged copy once this returns.
        runtime, _ = staging_runtime
        source = tmp_path / 'source.mp4'
        source.write_bytes(b'ORIGINAL')
        staged = tmp_path / 'run-1' / 'video_0.mp4'

        async def invoke(target, *args, **kwargs):
            name = getattr(target, '__name__', '')
            if name == 'probe_source_video_here':
                raise RuntimeError('source video failed ffprobe validation')
            return target(*args)

        runtime.invoke.side_effect = invoke
        with pytest.raises(RuntimeError, match='ffprobe'):
            await prepare_video_source([{'video': str(source)}], None, 60)

        assert not staged.exists()
        assert source.read_bytes() == b'ORIGINAL'

    async def test_a_probe_failure_on_the_runtime_falls_back_to_transferring(self, tmp_path, staging_runtime, caplog):
        # An invoke failure is a real fault, not a "not shared" answer — but the transfer
        # routes may still work, so it is logged loudly rather than failing the sample.
        runtime, _ = staging_runtime
        source = tmp_path / 'source.mp4'
        source.write_bytes(b'ORIGINAL')

        async def invoke(target, *args, **kwargs):
            name = getattr(target, '__name__', '')
            if name == 'stat_file_here':
                raise ConnectionError('socket.io dropped')
            if name == 'probe_source_video_here':
                return {'duration': 12.5}
            return target(*args)

        runtime.invoke.side_effect = invoke
        state = await prepare_video_source([{'video': str(source)}], None, 60)

        assert state['duration'] == 12.5
        runtime.upload_file.assert_awaited()
        assert 'visibility probe failed' in caplog.text

    async def test_a_file_at_the_same_path_but_a_different_inode_is_not_shared(self, tmp_path, staging_runtime):
        # Existence alone would let a sandbox's unrelated file pass for the dataset's video.
        runtime, _ = staging_runtime
        source = tmp_path / 'source.mp4'
        source.write_bytes(b'ORIGINAL')

        async def invoke(target, *args, **kwargs):
            name = getattr(target, '__name__', '')
            if name == 'stat_file_here':
                return {'dev': 1, 'ino': 2, 'size': 8}
            if name == 'probe_source_video_here':
                return {'duration': 12.5}
            return target(*args)

        runtime.invoke.side_effect = invoke
        await prepare_video_source([{'video': str(source)}], None, 60)

        runtime.upload_file.assert_awaited()


class TestStagingIsOptIn:
    def test_staging_is_off_by_default(self):
        # The transfer plus installing ffmpeg in the sandbox is not a cost a plain video QA
        # sample should pay for having a ``videos`` column.
        assert task_module.SearchTaskInput.model_fields['enable_video_staging'].default is False

    @pytest.mark.parametrize(
        ('videos', 'runtime_provider', 'match'),
        [
            (None, object(), 'carries no videos'),
            ([{'video': '/a.mp4'}], None, 'needs a runtime provider'),
        ],
    )
    async def test_enabling_it_without_what_it_needs_fails_loudly(self, monkeypatch, videos, runtime_provider, match):
        # Silently skipping would leave the prompt pointing at a path that is not there.
        context = SimpleNamespace(
            runtime_provider=runtime_provider, tool_state={}, run_id='run-1', agent_name='', get_runtime=AsyncMock()
        )
        monkeypatch.setattr(task_module, 'require_context', lambda: context)
        task = task_module.SearchTaskInput(
            question='q', answer='a', videos=videos, enable_video_staging=True, run_agent=AsyncMock()
        )

        with pytest.raises(ValueError, match=match):
            await task_module.mm_agent_task(task)

        task.run_agent.assert_not_awaited()


class TestInstalledHarnessRefusesVideo:
    """``claude_code`` / ``codex`` dispatch before the staging checks, so they need their own."""

    @pytest.mark.parametrize('agent_name', ['codex', 'claude_code', 'Codex'])
    @pytest.mark.parametrize(
        ('videos', 'enable_video_staging'),
        [
            ([{'video': '/data/a.mp4', 'fps': 2, 'max_frames': 384}], True),
            # Either field alone: a harness cannot reach the clip tool or the staged source.
            ([{'video': '/data/a.mp4'}], False),
            (None, True),
        ],
    )
    async def test_it_refuses_instead_of_answering_without_the_video(
        self, monkeypatch, agent_name, videos, enable_video_staging
    ):
        # Silently dropping is the bad outcome: the model answers a video question having never
        # seen the video, and the sample is still scored and trained on.
        context = SimpleNamespace(
            runtime_provider=object(), tool_state={}, run_id='run-1', agent_name=agent_name, get_runtime=AsyncMock()
        )
        monkeypatch.setattr(task_module, 'require_context', lambda: context)
        task = task_module.SearchTaskInput(
            question='what happens at 0:12?',
            answer='a',
            videos=videos,
            enable_video_staging=enable_video_staging,
            run_agent=AsyncMock(),
        )

        with pytest.raises(ValueError, match='cannot take videos or enable_video_staging'):
            await task_module.mm_agent_task(task)

        task.run_agent.assert_not_awaited()

    async def test_a_harness_sample_without_video_is_unaffected(self, monkeypatch):
        # The guard must not break the image/artifact harness path it was added next to.
        context = SimpleNamespace(
            runtime_provider=object(), tool_state={}, run_id='run-1', agent_name='codex', get_runtime=AsyncMock()
        )
        monkeypatch.setattr(task_module, 'require_context', lambda: context)
        task = task_module.SearchTaskInput(question='q', answer='a', images=['/data/a.png'], run_agent=AsyncMock())

        with pytest.raises(ValueError, match='not images'):
            await task_module.mm_agent_task(task)


class TestUploadCeiling:
    def test_the_default_is_a_host_memory_budget(self):
        # Only the upload route is capped, and only because of Host memory: the content is
        # Base64, which InvokeRequest.build pickles and Base64-encodes again, for ~7x the file
        # at peak. The transport does not constrain it -- a large payload goes by HTTP blob,
        # not one Socket.IO message -- so this is the only thing standing between an
        # unreachable dataset and an OOM on the rollout host.
        assert MAX_UPLOAD_BYTES == 256 * 1024**2
