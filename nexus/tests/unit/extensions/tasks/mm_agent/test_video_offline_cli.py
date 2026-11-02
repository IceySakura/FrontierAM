"""The offline frame-extraction CLI: its sampling policy, and that it still starts."""

import json
import math
import shutil
import subprocess
import sys

from pathlib import Path

import pytest

from nexus.extensions.tasks.mm_agent.utils.video_utils import extract_frames, frame_budget, resize_for_budget


MODULE = 'nexus.extensions.tasks.mm_agent.utils.video_utils'

needs_ffmpeg = pytest.mark.skipif(
    shutil.which('ffmpeg') is None or shutil.which('ffprobe') is None,
    reason='needs ffmpeg and ffprobe on PATH',
)


def write_video(path, *, seconds=4.0, fps=10, size='64x48'):
    command = [
        shutil.which('ffmpeg'), '-nostdin', '-hide_banner', '-loglevel', 'error',
        '-f', 'lavfi', '-i', f'testsrc=duration={seconds}:size={size}:rate={fps}',
        '-c:v', 'libx264', '-pix_fmt', 'yuv420p', '-y', str(path),
    ]  # fmt: skip
    subprocess.run(command, check=True, capture_output=True)
    return path


class TestFrameBudget:
    """What the dataset's ``fps`` / ``max_frames`` / ``min_frames`` ask the decoder for."""

    @pytest.mark.parametrize(
        ('duration', 'fps', 'max_frames', 'min_frames', 'expected'),
        [
            (4.0, 2.0, 384, 1, 8),
            # Rounds up: a 4.5s tail at 2 fps is still a frame the policy would see.
            (4.2, 2.0, 384, 1, 9),
            (600.0, 2.0, 384, 1, 384),  # long source clamps to max_frames
            (0.1, 2.0, 384, 16, 16),  # short source lifts to min_frames
        ],
    )
    def test_the_budget_follows_the_item_policy(self, duration, fps, max_frames, min_frames, expected):
        assert frame_budget(duration, fps, max_frames, min_frames) == expected

    def test_min_frames_wins_when_a_row_contradicts_itself(self):
        # Both bounds come from the row. ``min_frames`` is applied last, so it is the one
        # that survives -- pinned here because the clamp order is what an offline sweep and
        # the rollout decoder have to agree on.
        assert frame_budget(600.0, 2.0, 8, 16) == 16


class TestResizeForBudget:
    def test_both_axes_stay_even(self):
        # libx264/yuv420p rejects an odd axis outright.
        width, height = resize_for_budget(1921, 1081)
        assert (width % 2, height % 2) == (0, 0)

    def test_a_small_source_is_never_upscaled(self):
        assert resize_for_budget(64, 48, target_pixels=100 * 32 * 32) == (64, 48)

    def test_a_large_source_lands_under_the_pixel_budget(self):
        budget = 100 * 32 * 32
        width, height = resize_for_budget(1920, 1080, target_pixels=budget)
        assert width * height <= budget


@needs_ffmpeg
class TestExtractFrames:
    def test_it_writes_exactly_the_budgeted_frames(self, tmp_path):
        source = write_video(tmp_path / 'source.mp4', seconds=4.0, fps=10)
        destination = tmp_path / 'frames'

        result = extract_frames(
            source, destination, shutil.which('ffmpeg'), shutil.which('ffprobe'), 60, fps=2.0, max_frames=384
        )

        assert result['frame_budget'] == 8
        assert result['frame_count'] == result['frame_budget']
        assert len(sorted(destination.glob('frame_*.png'))) == 8

    def test_max_frames_caps_what_lands_on_disk(self, tmp_path):
        # The cap is what keeps an offline sweep from decoding a whole long video.
        source = write_video(tmp_path / 'source.mp4', seconds=4.0, fps=10)

        result = extract_frames(
            source, tmp_path / 'frames', shutil.which('ffmpeg'), shutil.which('ffprobe'), 60, fps=10.0, max_frames=3
        )

        assert (result['frame_budget'], result['frame_count']) == (3, 3)

    def test_the_reported_geometry_is_what_was_written(self, tmp_path):
        source = write_video(tmp_path / 'source.mp4', seconds=1.0, fps=8, size='1280x720')

        result = extract_frames(
            source, tmp_path / 'frames', shutil.which('ffmpeg'), shutil.which('ffprobe'), 60, fps=2.0, max_frames=384
        )

        assert result['width'] * result['height'] <= 100 * 32 * 32
        assert math.isclose(result['source_duration'], 1.0, abs_tol=0.2)

    def test_a_broken_source_names_the_file(self, tmp_path):
        broken = tmp_path / 'source.mp4'
        broken.write_bytes(b'not a video')

        with pytest.raises(RuntimeError, match='source.mp4'):
            extract_frames(
                broken, tmp_path / 'frames', shutil.which('ffmpeg'), shutil.which('ffprobe'), 60, fps=2.0, max_frames=4
            )


class TestCommandLine:
    def test_help_runs(self):
        # The CLI lives in a module RL also imports, so an import-time break shows up here.
        completed = subprocess.run(
            [sys.executable, '-m', MODULE, '--help'], capture_output=True, text=True, timeout=120, check=False
        )

        assert completed.returncode == 0, completed.stderr
        assert '--dataset' in completed.stdout
        assert '--max-frames' in completed.stdout

    def test_it_exits_nonzero_rather_than_raising_on_a_bad_row(self, tmp_path):
        # A sweep over thousands of rows must not die on one unreadable video.
        dataset = tmp_path / 'rows.jsonl'
        dataset.write_text(json.dumps({'videos': [{'video': str(tmp_path / 'absent.mp4')}]}) + '\n')
        out_dir = tmp_path / 'out'

        completed = subprocess.run(
            [sys.executable, '-m', MODULE, '--dataset', str(dataset), '--out-dir', str(out_dir)],
            capture_output=True, text=True, timeout=300, check=False,
        )  # fmt: skip

        if 'ffmpeg/ffprobe unavailable' in completed.stderr:
            pytest.skip('needs ffmpeg and ffprobe on PATH')
        assert completed.returncode == 1
        manifest = json.loads((out_dir / 'manifest.json').read_text())
        assert 'error' in manifest[0]


@needs_ffmpeg
def test_a_whole_dataset_sweep_writes_a_manifest(tmp_path):
    source = write_video(tmp_path / 'source.mp4', seconds=2.0, fps=10)
    dataset = tmp_path / 'rows.jsonl'
    dataset.write_text(json.dumps({'videos': [{'video': str(source), 'fps': 2.0, 'max_frames': 384}]}) + '\n')
    out_dir = tmp_path / 'out'

    completed = subprocess.run(
        [sys.executable, '-m', MODULE, '--dataset', str(dataset), '--out-dir', str(out_dir)],
        capture_output=True, text=True, timeout=300, check=False,
    )  # fmt: skip

    assert completed.returncode == 0, completed.stderr
    entry = json.loads((out_dir / 'manifest.json').read_text())[0]
    assert entry['frame_count'] == entry['frame_budget'] == 4
    assert Path(entry['frame_dir']).is_dir()
