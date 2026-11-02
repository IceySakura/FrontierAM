"""Recording protocol, native dispatch, and detached worker lifecycle."""

import ast
import json
import os
import sys
import time

from pathlib import Path
from unittest.mock import AsyncMock

import pytest

from pydantic import ValidationError

from nexus.runtime.executors.models.gui import GuiStartRecordingRequest, GuiStopRecordingResponse
from nexus.runtime.providers.ags.handle import AGSInstanceHandle
from nexus.runtime.providers.ags.osworld import AGSOSWorldRuntime
from nexus.runtime.providers.ags.waa import AGSWindowsAgentArenaRuntime
from nexus.runtime.providers.devcloud.client import DevCloudRuntime
from nexus.runtime.utils import _recording_worker as recorder


@pytest.mark.parametrize('fps', [0, 61, 2.5])
def test_recording_rejects_invalid_frame_rate(fps):
    with pytest.raises(ValidationError):
        GuiStartRecordingRequest(fps=fps)


@pytest.mark.parametrize('kind', ['osworld', 'waa', 'macos'])
@pytest.mark.asyncio
async def test_native_runtime_recording_dispatch_and_typed_methods(kind):
    handle = AGSInstanceHandle(instance_id='test', access_token='token', domain='example.com')
    if kind == 'macos':
        runtime = DevCloudRuntime(base_url='http://example.com')
        method = '_post_json'
    elif kind == 'waa':
        runtime = AGSWindowsAgentArenaRuntime(handle)
        method = '_post_waa_json'
    else:
        runtime = AGSOSWorldRuntime(handle)
        method = '_post_osworld_json'
    post = AsyncMock(
        return_value=(
            {'output': 'NEXUS_RECORDING_RESULT:{"success":true,"output_path":"/tmp/test.mp4"}\n', 'returncode': 0},
            '',
            200,
        )
    )
    setattr(runtime, method, post)
    response = await runtime.gui_start_recording(fps=12, max_duration_seconds=15)
    assert response.success
    path, payload = post.call_args.args
    assert path == '/execute'
    assert payload['shell'] is False
    assert 'max_duration_seconds' in str(payload['command'])
    assert payload['command'][0] == ('python' if kind == 'waa' else 'bash')
    if kind == 'waa':
        assert sum(len(x) for x in payload['command']) < 30000

    post.return_value = ({'output': 'NEXUS_RECORDING_RESULT:{"success":false,"error":"No recording"}'}, '', 200)
    stopped = await runtime.gui_stop_recording()
    assert isinstance(stopped, GuiStopRecordingResponse)
    assert not stopped.success
    assert stopped.error == 'No recording'


def test_standalone_worker_supports_sandbox_python_37():
    ast.parse(Path(recorder.__file__).read_text(), feature_version=(3, 7))


@pytest.fixture
def recording(tmp_path, monkeypatch):
    """Real detached subprocesses with small FFmpeg/ffprobe stand-ins."""
    bin_dir = tmp_path / 'bin'
    bin_dir.mkdir()
    fake = bin_dir / 'ffmpeg'
    fake.write_text(
        f'#!{sys.executable}\n'
        'import os,sys,time,threading\nfrom pathlib import Path\n'
        'args=sys.argv\n'
        'if "-h" in args: print("-fps_mode"); sys.exit(0)\n'
        'progress=Path(args[args.index("-progress")+1])\n'
        'if os.getenv("RECORDING_FAIL_START"): sys.exit(7)\n'
        'Path(args[-1]).write_bytes(b"test-video")\n'
        'def crash():\n'
        '    while not (progress.parent/"crash").exists(): time.sleep(.01)\n'
        '    os._exit(int(os.getenv("RECORDING_CRASH_CODE", "7")))\n'
        'threading.Thread(target=crash,daemon=True).start()\n'
        'progress.write_text("frame=1\\n")\n'
        'assert sys.stdin.readline().strip()=="q"\n'
    )
    fake.chmod(0o755)
    probe = bin_dir / 'ffprobe'
    probe.write_text(
        f'#!{sys.executable}\nimport json\n'
        'print(json.dumps({"streams":[{"width":1920,"height":1080,"avg_frame_rate":"10/1"}],'
        '"format":{"duration":"1.2"}}))\n'
    )
    probe.chmod(0o755)
    monkeypatch.setenv('PATH', str(bin_dir) + os.pathsep + os.environ['PATH'])
    root = tmp_path / 'state'
    source = Path(recorder.__file__).read_text()

    def dispatch(action, **kwargs):
        return recorder._dispatch(root, {'action': action, 'fps': 10, **kwargs}, source)

    yield dispatch, root
    if (root / 'active.json').exists():
        dispatch('stop')


def test_recording_lifecycle_and_duplicate_calls(recording):
    dispatch, root = recording
    with pytest.raises(RuntimeError, match='No recording'):
        dispatch('stop')
    started = dispatch('start')
    assert started['success']
    assert Path(started['output_path']).is_absolute()
    with pytest.raises(RuntimeError, match='already in progress'):
        dispatch('start')
    with recorder._lock(root), pytest.raises((BlockingIOError, PermissionError)):
        dispatch('stop')
    stopped = dispatch('stop')
    assert stopped == {
        'success': True,
        'output_path': started['output_path'],
        'duration_seconds': 1.2,
        'size_bytes': 10,
        'width': 1920,
        'height': 1080,
        'fps': 10.0,
    }
    assert Path(started['output_path']).exists()
    assert not (root / 'active.json').exists()
    with pytest.raises(RuntimeError, match='No recording'):
        dispatch('stop')
    assert dispatch('start')['success']
    assert dispatch('stop')['success']


def test_recording_does_not_overwrite_and_validates_remote_path(recording, tmp_path):
    dispatch, _ = recording
    for path in ['relative.mp4', str(tmp_path / 'wrong.webm')]:
        with pytest.raises(ValueError, match='absolute .mp4'):
            dispatch('start', output_path=path)
    existing = tmp_path / 'existing.mp4'
    existing.write_bytes(b'keep')
    with pytest.raises(ValueError, match='already exists'):
        dispatch('start', output_path=str(existing))
    assert existing.read_bytes() == b'keep'
    nested = tmp_path / 'new folder' / 'video.mp4'
    assert dispatch('start', output_path=str(nested))['success']
    assert dispatch('stop')['output_path'] == str(nested)


def test_recording_start_failure_releases_singleton(recording, monkeypatch):
    dispatch, root = recording
    monkeypatch.setenv('RECORDING_FAIL_START', '1')
    assert not dispatch('start')['success']
    assert not (root / 'active.json').exists()
    monkeypatch.delenv('RECORDING_FAIL_START')
    assert dispatch('start')['success']


@pytest.mark.parametrize('exit_code', [0, 7])
def test_recording_abnormal_exit_is_reported_by_stop(recording, monkeypatch, exit_code):
    monkeypatch.setenv('RECORDING_CRASH_CODE', str(exit_code))
    dispatch, root = recording
    assert dispatch('start')['success']
    work = Path(json.loads((root / 'active.json').read_text())['work'])
    (work / 'crash').touch()
    deadline = time.monotonic() + 5
    while not (work / 'result.json').exists() and time.monotonic() < deadline:
        time.sleep(0.02)
    assert (work / 'result.json').exists()
    result = dispatch('stop')
    assert not result['success']
    assert 'ffmpeg exited unexpectedly' in result['error']
    assert not (root / 'active.json').exists()


@pytest.mark.parametrize('duration', [0, -1, 1.5, float('inf'), float('nan')])
def test_recording_rejects_invalid_max_duration(duration):
    with pytest.raises(ValidationError):
        GuiStartRecordingRequest(max_duration_seconds=duration)


def test_recording_defaults_to_one_hour(recording):
    assert GuiStartRecordingRequest().max_duration_seconds == 3600
    dispatch, root = recording
    assert dispatch('start')['success']
    work = Path(json.loads((root / 'active.json').read_text())['work'])
    assert json.loads((work / 'config.json').read_text())['max_duration_seconds'] == 3600
    assert dispatch('stop')['success']


def test_recording_duration_limit_finalizes_before_stop_and_preserves_result(recording):
    dispatch, root = recording
    started_at = time.monotonic()
    started = dispatch('start', max_duration_seconds=1)
    assert started['success']
    work = Path(json.loads((root / 'active.json').read_text())['work'])
    deadline = time.monotonic() + 5
    while not (work / 'result.json').exists() and time.monotonic() < deadline:
        time.sleep(0.02)
    assert (work / 'result.json').exists()
    assert time.monotonic() - started_at >= 1
    assert not (work / 'stop').exists()
    assert json.loads((work / 'result.json').read_text())['success']
    with pytest.raises(RuntimeError, match='already in progress'):
        dispatch('start')
    stopped = dispatch('stop')
    assert stopped['success']
    assert stopped['output_path'] == started['output_path']
    assert Path(stopped['output_path']).exists()
    assert not (root / 'active.json').exists()
    assert dispatch('start')['success']
    assert dispatch('stop')['success']


@pytest.mark.parametrize('modern', [True, False])
def test_recording_frame_rate_option_matches_ffmpeg_capabilities(tmp_path, monkeypatch, modern):
    from types import SimpleNamespace

    def run(args, **kwargs):
        if '-list_devices' in args:
            return SimpleNamespace(stderr='[AVFoundation indev] [1] Capture screen 0')
        assert args == ['ffmpeg', '-hide_banner', '-h', 'full']
        return SimpleNamespace(stdout='-fps_mode' if modern else '-vsync')

    monkeypatch.setattr(recorder.sys, 'platform', 'darwin')
    monkeypatch.setattr(recorder.subprocess, 'run', run)
    command = recorder._command('ffmpeg', tmp_path / 'out.mp4', 10, tmp_path)
    expected, absent = ('-fps_mode', '-vsync') if modern else ('-vsync', '-fps_mode')
    assert command[command.index(expected) + 1] == 'cfr'
    assert absent not in command
    assert command[command.index('-r') + 1] == '10'


@pytest.mark.parametrize(
    ('exit_status', 'expected'), [('last exit code = 0', 0), ('last terminating signal = Terminated: 15', -15)]
)
def test_macos_capture_launches_ffmpeg_in_gui_session_and_cleans_up(tmp_path, monkeypatch, exit_status, expected):
    import plistlib

    from types import SimpleNamespace

    calls = []
    statuses = iter(['state = running', exit_status])

    def run(args, **kwargs):
        calls.append(args)
        if args[1] == 'bootstrap':
            data = plistlib.loads(Path(args[-1]).read_bytes())
            assert data['RunAtLoad'] is True
            assert data['Label'].startswith('com.nexus.gui-recording.')
            assert data['ProgramArguments'] == ['/usr/local/bin/ffmpeg', '-i', '0:none']
            assert data['StandardInPath'] == str(tmp_path / 'stdin')
        if args[1] == 'print':
            return SimpleNamespace(stdout=next(statuses))
        return SimpleNamespace(returncode=0)

    monkeypatch.setattr(recorder.subprocess, 'run', run)
    process = recorder._MacOSCaptureProcess(['/usr/local/bin/ffmpeg', '-i', '0:none'], tmp_path)
    assert process.poll() is None
    process.stdin.write(b'q\n')
    assert process.wait(timeout=1) == expected
    assert process.poll() == expected
    process.stdin.close()
    process.close()
    assert [args[1] for args in calls] == ['bootstrap', 'print', 'print', 'bootout']


def test_macos_capture_closes_fifo_when_bootstrap_fails(tmp_path, monkeypatch):
    import subprocess

    from unittest.mock import Mock

    stream = Mock()
    monkeypatch.setattr(recorder.os, 'fdopen', lambda *args, **kwargs: stream)
    monkeypatch.setattr(recorder.os, 'open', lambda *args: 42)
    monkeypatch.setattr(recorder.subprocess, 'run', Mock(side_effect=subprocess.CalledProcessError(5, 'launchctl')))
    with pytest.raises(subprocess.CalledProcessError):
        recorder._MacOSCaptureProcess(['ffmpeg'], tmp_path)
    stream.close.assert_called_once()
