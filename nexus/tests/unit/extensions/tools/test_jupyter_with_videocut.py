"""The sandbox half of the videocut pipeline: clip discovery, validation, transport contract."""

import shutil
import subprocess

import pytest

from nexus.core import ToolContext
from nexus.extensions.tools.code_interpreter import jupyter_with_videocut as videocut
from nexus.extensions.tools.code_interpreter.jupyter_with_videocut import (
    MIN_VIDEO_FRAMES,
    VideoCutInterpreterInput,
    jupyter_with_videocut_tool,
    new_video_paths,
    probe_clip_video,
    single_stream,
    standardize_clip,
    validate_code,
    video_snapshot,
)


needs_ffmpeg = pytest.mark.skipif(
    shutil.which('ffmpeg') is None or shutil.which('ffprobe') is None,
    reason='needs ffmpeg and ffprobe on PATH',
)


@pytest.fixture
def ffmpeg_paths():
    return shutil.which('ffmpeg'), shutil.which('ffprobe')


def write_clip(path, ffmpeg, *, seconds=1.0, fps=8, size='64x64', codec='libx264', audio=False, extra=()):
    """Synthesize a test clip. Defaults already satisfy the transport contract."""
    command = [
        ffmpeg, '-nostdin', '-hide_banner', '-loglevel', 'error',
        '-f', 'lavfi', '-i', f'testsrc=duration={seconds}:size={size}:rate={fps}',
    ]  # fmt: skip
    if audio:
        command += ['-f', 'lavfi', '-i', f'sine=duration={seconds}', '-c:a', 'aac']
    command += ['-c:v', codec, '-pix_fmt', 'yuv420p', *extra, '-y', str(path)]
    subprocess.run(command, check=True, capture_output=True)
    return path


class TestValidateCode:
    def test_rejects_shell_true(self):
        # The one static check: word-splitting a path with a space fails with no hint of why.
        with pytest.raises(ValueError, match='shell=True'):
            validate_code("subprocess.run('ffmpeg -i a.mp4 b.mp4', shell=True)")

    def test_allows_an_argument_list(self):
        validate_code("subprocess.run([ffmpeg_path, '-i', 'a.mp4', 'b.mp4'], check=True)")

    def test_leaves_syntax_errors_to_the_caller(self):
        # The tool entrypoint turns this into an error response rather than a 500.
        with pytest.raises(SyntaxError):
            validate_code('def (:')

    @pytest.mark.parametrize('code', ['run(cmd, shell=False)', 'run(cmd, shell=use_shell)'])
    def test_only_a_literal_true_trips_it(self, code):
        validate_code(code)


class TestClipDiscovery:
    def test_ignores_non_video_and_transport_files(self, tmp_path):
        (tmp_path / 'notes.txt').write_bytes(b'x')
        (tmp_path / f'{videocut._TRANSPORT_PREFIX}abc.mp4').write_bytes(b'x')
        (tmp_path / 'clip.mp4').write_bytes(b'x')
        assert set(video_snapshot(tmp_path)) == {'clip.mp4'}

    def test_missing_directory_is_empty_not_an_error(self, tmp_path):
        assert video_snapshot(tmp_path / 'absent') == {}

    def test_a_rewritten_clip_counts_as_new(self, tmp_path):
        # Name-only comparison would silently drop the second turn's clip when the model
        # reuses its own filename.
        clip = tmp_path / 'clip.mp4'
        clip.write_bytes(b'first')
        before = video_snapshot(tmp_path)
        clip.write_bytes(b'second take')
        assert new_video_paths(tmp_path, before) == [clip]

    def test_untouched_clips_are_not_new(self, tmp_path):
        (tmp_path / 'clip.mp4').write_bytes(b'x')
        assert new_video_paths(tmp_path, video_snapshot(tmp_path)) == []

    def test_newest_comes_last(self, tmp_path):
        # The tool takes candidates[-1]; one action yields one observation.
        before = video_snapshot(tmp_path)
        for index, name in enumerate(('a.mp4', 'b.mov')):
            path = tmp_path / name
            path.write_bytes(b'x')
            import os

            os.utime(path, ns=(0, 1_000_000_000 * (index + 1)))
        assert [path.name for path in new_video_paths(tmp_path, before)] == ['a.mp4', 'b.mov']


class TestSingleStream:
    def test_takes_the_only_stream(self):
        assert single_stream({'streams': [{'codec_type': 'video'}]}) == ({'codec_type': 'video'}, None)

    @pytest.mark.parametrize(
        'payload',
        [{'streams': []}, {'streams': [{}, {}]}, {'streams': 'nope'}, {}],
    )
    def test_anything_else_is_an_error(self, payload):
        stream, error = single_stream(payload)
        assert stream is None
        assert 'exactly one video stream' in error


@needs_ffmpeg
class TestProbeClipVideo:
    def test_accepts_a_conformant_clip(self, tmp_path, ffmpeg_paths):
        ffmpeg, ffprobe = ffmpeg_paths
        clip = write_clip(tmp_path / 'clip.mp4', ffmpeg, seconds=1.0, fps=8, size='64x64')

        metadata, error = probe_clip_video(str(clip), ffprobe, 60)

        assert error is None
        assert (metadata['width'], metadata['height']) == (64, 64)
        assert metadata['frame_count'] == 8
        assert metadata['fps'] == 8

    def test_rejects_audio(self, tmp_path, ffmpeg_paths):
        # The contract is silent video; an audio stream would reach a decoder that cannot use it.
        ffmpeg, ffprobe = ffmpeg_paths
        clip = write_clip(tmp_path / 'clip.mp4', ffmpeg, audio=True)

        _, error = probe_clip_video(str(clip), ffprobe, 60)

        assert error == 'clip must not contain audio'

    def test_rejects_a_single_frame_clip(self, tmp_path, ffmpeg_paths):
        # HF's sample_frames rounds down to a multiple of 2, so one frame becomes zero and
        # dies inside the rollout's decoder subprocess with nothing pointing back here.
        ffmpeg, ffprobe = ffmpeg_paths
        clip = write_clip(tmp_path / 'clip.mp4', ffmpeg, seconds=1.0, fps=1)

        _, error = probe_clip_video(str(clip), ffprobe, 60)

        assert f'below the {MIN_VIDEO_FRAMES}-frame minimum' in error

    def test_rejects_a_clip_over_max_bytes(self, tmp_path, ffmpeg_paths):
        ffmpeg, ffprobe = ffmpeg_paths
        clip = write_clip(tmp_path / 'clip.mp4', ffmpeg)

        _, error = probe_clip_video(str(clip), ffprobe, 60, max_bytes=1)

        assert 'above the 0 MiB limit' in error

    def test_max_bytes_is_overridable_upward(self, tmp_path, ffmpeg_paths):
        ffmpeg, ffprobe = ffmpeg_paths
        clip = write_clip(tmp_path / 'clip.mp4', ffmpeg)

        metadata, error = probe_clip_video(str(clip), ffprobe, 60, max_bytes=videocut.DEFAULT_MAX_VIDEO_BYTES)

        assert error is None
        assert metadata['size'] > 0

    def test_a_non_video_file_fails_probing(self, tmp_path, ffmpeg_paths):
        _, ffprobe = ffmpeg_paths
        broken = tmp_path / 'clip.mp4'
        broken.write_bytes(b'not a video')

        metadata, error = probe_clip_video(str(broken), ffprobe, 60)

        assert metadata is None
        assert error


class TestProbeClipVideoWithoutFfmpeg:
    """Contract checks libx264 will not let a real file reach."""

    @pytest.fixture
    def probe(self, monkeypatch):
        """Probe a stubbed ffprobe payload, starting from one that passes."""

        def _probe(**stream_overrides):
            payload = {
                'streams': [{
                    'codec_type': 'video', 'codec_name': 'h264', 'pix_fmt': 'yuv420p',
                    'width': 64, 'height': 64, 'avg_frame_rate': '8/1', 'r_frame_rate': '8/1',
                    'duration': '1.0', 'nb_read_frames': '8', **stream_overrides,
                }],
                'format': {'format_name': 'mov,mp4,m4a', 'duration': '1.0'},
            }  # fmt: skip
            monkeypatch.setattr(videocut, 'ffprobe_json', lambda *args, **kwargs: (payload, None))
            monkeypatch.setattr(videocut.os.path, 'getsize', lambda path: 1024)
            return probe_clip_video('clip.mp4', 'ffprobe', 60)

        return _probe

    def test_the_stub_itself_passes(self, probe):
        metadata, error = probe()
        assert error is None
        assert metadata['frame_count'] == 8

    @pytest.mark.parametrize(
        ('override', 'expected'),
        [
            # An odd axis will not encode as yuv420p at all, so this is a guard against a
            # hand-muxed file rather than against anything libx264 would produce.
            ({'width': 65}, 'clip dimensions must be positive and even'),
            ({'height': 0}, 'clip dimensions must be positive and even'),
            ({'codec_name': 'vp9'}, 'clip codec must be H264'),
            ({'pix_fmt': 'yuv444p'}, 'clip pixel format must be yuv420p'),
            ({'avg_frame_rate': '0/0', 'r_frame_rate': '0/0'}, 'clip frame rate could not be determined'),
            ({'nb_read_frames': 'N/A'}, 'invalid clip metadata'),
        ],
    )
    def test_each_contract_violation_names_itself(self, probe, override, expected):
        metadata, error = probe(**override)
        assert metadata is None
        assert expected in error


@needs_ffmpeg
class TestStandardizeClip:
    def test_remuxes_a_clip_that_already_conforms(self, tmp_path, ffmpeg_paths):
        ffmpeg, ffprobe = ffmpeg_paths
        source = write_clip(tmp_path / 'raw.mp4', ffmpeg)
        transport = tmp_path / 'transport.mp4'

        metadata, error = standardize_clip(source, transport, ffmpeg, ffprobe, 60)

        assert error is None
        assert transport.is_file()
        assert metadata['frame_count'] == 8

    def test_transcodes_a_non_conformant_container(self, tmp_path, ffmpeg_paths):
        # The model names its own output; MKV is in the accepted extensions.
        ffmpeg, ffprobe = ffmpeg_paths
        source = write_clip(tmp_path / 'raw.mkv', ffmpeg)
        transport = tmp_path / 'transport.mp4'

        metadata, error = standardize_clip(source, transport, ffmpeg, ffprobe, 60)

        assert error is None
        assert metadata['width'] == 64

    def test_strips_audio(self, tmp_path, ffmpeg_paths):
        ffmpeg, ffprobe = ffmpeg_paths
        source = write_clip(tmp_path / 'raw.mp4', ffmpeg, audio=True)
        transport = tmp_path / 'transport.mp4'

        _, error = standardize_clip(source, transport, ffmpeg, ffprobe, 60)

        assert error is None

    def test_passes_max_bytes_through_to_the_probe(self, tmp_path, ffmpeg_paths):
        ffmpeg, ffprobe = ffmpeg_paths
        source = write_clip(tmp_path / 'raw.mp4', ffmpeg)

        _, error = standardize_clip(source, tmp_path / 'transport.mp4', ffmpeg, ffprobe, 60, max_bytes=1)

        assert 'above the 0 MiB limit' in error

    def test_a_broken_source_reports_ffmpeg_rather_than_raising(self, tmp_path, ffmpeg_paths):
        ffmpeg, ffprobe = ffmpeg_paths
        broken = tmp_path / 'raw.mp4'
        broken.write_bytes(b'not a video')

        metadata, error = standardize_clip(broken, tmp_path / 'transport.mp4', ffmpeg, ffprobe, 60)

        assert metadata is None
        assert error


class TestToolEntrypoint:
    async def test_requires_a_run_id(self):
        # Without one every concurrent sample shares a kernel and a working directory, and the
        # first close_videocut_session rmtrees the others' clips.
        response = await jupyter_with_videocut_tool(
            VideoCutInterpreterInput(code='print(1)'), ToolContext(tool_configs={})
        )

        assert response.meta['success'] is False
        assert 'requires a run_id' in response.output[0]['text']

    async def test_rejects_shell_true_before_touching_a_kernel(self):
        response = await jupyter_with_videocut_tool(
            VideoCutInterpreterInput(code="run('x', shell=True)"), ToolContext(tool_configs={}, run_id='run-1')
        )

        assert response.meta['success'] is False
        assert 'shell=True' in response.output[0]['text']

    async def test_rejects_empty_code(self):
        response = await jupyter_with_videocut_tool(
            VideoCutInterpreterInput(code='   '), ToolContext(tool_configs={}, run_id='run-1')
        )

        assert response.meta['success'] is False
        assert 'empty code' in response.output[0]['text']

    def test_an_oversized_timeout_is_clamped_not_rejected(self):
        # A throwaway guess from the model, not a mistake worth spending a turn on; the value
        # above the ceiling would only delay cleanup past the agent loop's own cancellation.
        params = VideoCutInterpreterInput(code='print(1)', timeout=videocut._MAX_TIMEOUT_SECONDS * 10)

        assert params.timeout == videocut._MAX_TIMEOUT_SECONDS

    @pytest.mark.parametrize('timeout', [0, -1, float('inf'), 'sixty', True, None])
    def test_a_timeout_that_is_not_a_positive_number_is_rejected(self, timeout):
        with pytest.raises(ValueError, match='timeout must be'):
            VideoCutInterpreterInput(code='print(1)', timeout=timeout)

    def test_unknown_fields_are_rejected(self):
        with pytest.raises(ValueError, match='dump_dir'):
            VideoCutInterpreterInput(code='print(1)', dump_dir='/tmp/x')


class TestCloseSession:
    async def test_closing_an_unknown_run_is_a_no_op(self):
        # Guards the cleanup path: a run that never started a kernel must not report success
        # and must not reach rmtree.
        assert await videocut.close_videocut_session('never-started') == {'closed': False}

    async def test_a_missing_run_id_closes_nothing(self):
        assert await videocut.close_videocut_session(None) == {'closed': False}
