"""Standalone recorder copied into native sandboxes; standard library, Python 3.7+."""

import contextlib
import json
import os
import plistlib
import re
import shutil
import subprocess
import sys
import tempfile
import time

from fractions import Fraction
from pathlib import Path


START_TIMEOUT = 30
STOP_TIMEOUT = 30


def _write(path, value):
    temporary = path.with_suffix('.tmp')
    temporary.write_text(json.dumps(value), encoding='utf-8')
    os.replace(str(temporary), str(path))


@contextlib.contextmanager
def _lock(root):
    # OS locks are released even when a request process exits unexpectedly.
    with (root / 'control.lock').open('a+b') as file:
        file.seek(0)
        if sys.platform == 'win32':
            import msvcrt

            msvcrt.locking(file.fileno(), msvcrt.LK_NBLCK, 1)
        else:
            import fcntl

            fcntl.flock(file, fcntl.LOCK_EX | fcntl.LOCK_NB)
        yield


def _command(ffmpeg, output_path, fps, work):
    args = [ffmpeg, '-n', '-hide_banner', '-loglevel', 'warning']
    if sys.platform == 'win32':
        args += ['-f', 'gdigrab', '-framerate', str(fps), '-draw_mouse', '1', '-i', 'desktop']
    elif sys.platform == 'darwin':
        devices = subprocess.run(
            [ffmpeg, '-hide_banner', '-f', 'avfoundation', '-list_devices', 'true', '-i', ''],
            capture_output=True,
            text=True,
            timeout=10,
        )
        screen = re.search(r'\[(\d+)\] Capture screen 0', devices.stderr)
        if screen is None:
            raise RuntimeError('No AVFoundation desktop capture device found: ' + devices.stderr[-1000:])
        args += [
            '-f',
            'avfoundation',
            '-pixel_format',
            'uyvy422',
            '-framerate',
            str(fps),
            '-capture_cursor',
            '1',
            '-i',
            screen.group(1) + ':none',
        ]
    else:
        args += ['-f', 'x11grab', '-framerate', str(fps), '-draw_mouse', '1', '-i', os.getenv('DISPLAY') or ':0']
    # FFmpeg 9 removed -vsync; older OSWorld images do not have -fps_mode.
    options = subprocess.run([ffmpeg, '-hide_banner', '-h', 'full'], capture_output=True, text=True, timeout=10)
    fps_option = '-fps_mode' if '-fps_mode' in options.stdout else '-vsync'
    # Output frame rate must be explicit for AVFoundation.
    return args + [
        '-r',
        str(fps),
        fps_option,
        'cfr',
        '-c:v',
        'libx264',
        '-preset',
        'ultrafast',
        '-crf',
        '28',
        '-pix_fmt',
        'yuv420p',
        '-movflags',
        '+faststart',
        '-progress',
        str(work / 'progress'),
        str(output_path),
    ]


def _probe(ffprobe, output_path):
    data = json.loads(
        subprocess.check_output(
            [
                ffprobe,
                '-v',
                'error',
                '-select_streams',
                'v:0',
                '-show_streams',
                '-show_format',
                '-of',
                'json',
                str(output_path),
            ],
            universal_newlines=True,
            timeout=15,
        )
    )
    video = data['streams'][0]
    return {
        'success': True,
        'output_path': str(output_path),
        'duration_seconds': float(data['format']['duration']),
        'size_bytes': output_path.stat().st_size,
        'width': video['width'],
        'height': video['height'],
        'fps': float(Fraction(video['avg_frame_rate'])),
    }


class _MacOSCaptureProcess:
    """Run FFmpeg itself in the desktop audit session, preserving its TCC grant."""

    def __init__(self, command, work):
        fifo = work / 'stdin'
        os.mkfifo(str(fifo))
        self.stdin = os.fdopen(os.open(str(fifo), os.O_RDWR), 'wb', buffering=0)
        self.returncode = None
        label = 'com.nexus.gui-recording.' + work.name
        self.target = 'gui/' + str(os.getuid()) + '/' + label
        agent = work / 'agent.plist'
        agent.write_bytes(
            plistlib.dumps({
                'Label': label,
                'ProgramArguments': command,
                'RunAtLoad': True,
                'StandardInPath': str(fifo),
                'StandardOutPath': str(work / 'ffmpeg.log'),
                'StandardErrorPath': str(work / 'ffmpeg.log'),
            })
        )
        try:
            subprocess.run(
                ['launchctl', 'bootstrap', 'gui/' + str(os.getuid()), str(agent)],
                capture_output=True,
                check=True,
                timeout=10,
            )
        except Exception:
            self.stdin.close()
            raise

    def poll(self):
        if self.returncode is None:
            status = subprocess.run(
                ['launchctl', 'print', self.target], capture_output=True, text=True, check=True, timeout=10
            ).stdout
            exit_code = re.search(r'last exit code = (-?\d+)', status)
            signal = re.search(r'last terminating signal = .*?: (\d+)', status)
            if exit_code:
                self.returncode = int(exit_code.group(1))
            elif signal:
                self.returncode = -int(signal.group(1))
        return self.returncode

    def wait(self, timeout=None):
        deadline = time.monotonic() + timeout if timeout is not None else float('inf')
        while self.poll() is None:
            if timeout is not None and time.monotonic() >= deadline:
                raise subprocess.TimeoutExpired('ffmpeg', timeout)
            time.sleep(0.1)
        return self.returncode

    def terminate(self):
        subprocess.run(['launchctl', 'kill', 'SIGTERM', self.target], capture_output=True, check=True, timeout=10)

    def kill(self):
        subprocess.run(['launchctl', 'kill', 'SIGKILL', self.target], capture_output=True, check=True, timeout=10)

    def close(self):
        subprocess.run(['launchctl', 'bootout', self.target], capture_output=True, check=True, timeout=10)


def _worker(work):
    config = json.loads((work / 'config.json').read_text(encoding='utf-8'))
    process = None
    stop_requested = False
    try:
        with (work / 'ffmpeg.log').open('wb') as log:
            if config.get('gui_session') and sys.platform == 'darwin':
                process = _MacOSCaptureProcess(config['command'], work)
            else:
                process = subprocess.Popen(
                    config['command'],
                    stdin=subprocess.PIPE,
                    stdout=log,
                    stderr=log,
                    text=False,
                    creationflags=subprocess.CREATE_NO_WINDOW if sys.platform == 'win32' else 0,
                )
            deadline = time.monotonic() + config['max_duration_seconds']
            while process.poll() is None:
                progress = work / 'progress'
                if not (work / 'ready').exists() and progress.exists():
                    frames = re.findall(r'^frame=(\d+)$', progress.read_text(encoding='utf-8'), re.MULTILINE)
                    if frames and int(frames[-1]) > 0:
                        (work / 'ready').touch()
                if (work / 'stop').exists() or time.monotonic() >= deadline:
                    stop_requested = True
                    # A pipe works on all three platforms, including detached
                    # Windows processes, without console signal delivery.
                    assert process.stdin is not None
                    process.stdin.write(b'q\n')
                    process.stdin.flush()
                    process.wait(timeout=15)
                    break
                time.sleep(0.1)
            if not stop_requested or process.returncode != 0 or not (work / 'ready').exists():
                raise RuntimeError(
                    'ffmpeg exited unexpectedly: ' + (work / 'ffmpeg.log').read_text(errors='replace')[-2000:]
                )
        result = _probe(config['ffprobe'], Path(config['output_path']))
    except Exception as exc:
        result = {'success': False, 'output_path': config['output_path'], 'error': str(exc)}
    finally:
        if process is not None:
            if process.poll() is None:
                process.terminate()
                try:
                    process.wait(timeout=3)
                except subprocess.TimeoutExpired:
                    process.kill()
                    process.wait()
            if process.stdin is not None:
                process.stdin.close()
            if isinstance(process, _MacOSCaptureProcess):
                process.close()
    _write(work / 'result.json', result)


def _collect(root, work):
    result_path = work / 'result.json'
    deadline = time.monotonic() + STOP_TIMEOUT
    while not result_path.exists():
        if time.monotonic() >= deadline:
            raise RuntimeError('Recording has not stopped yet; call gui_stop_recording again')
        time.sleep(0.1)
    result = json.loads(result_path.read_text(encoding='utf-8'))
    (root / 'active.json').unlink()
    shutil.rmtree(str(work), ignore_errors=True)
    return result


def _start(root, request, source):
    if (root / 'active.json').exists():
        raise RuntimeError('A recording is already in progress; call gui_stop_recording first')
    fps = request['fps']
    ffmpeg, ffprobe = shutil.which('ffmpeg'), shutil.which('ffprobe')
    if not ffmpeg or not ffprobe:
        raise RuntimeError('ffmpeg and ffprobe must be available on PATH')
    output = request.get('output_path')
    if output is not None:
        output = Path(output)
        if not output.is_absolute() or output.suffix.lower() != '.mp4':
            raise ValueError('output_path must be an absolute .mp4 path inside the runtime')
        if output.exists():
            raise ValueError('Output file already exists: ' + str(output))
        output.parent.mkdir(parents=True, exist_ok=True)
    work = Path(tempfile.mkdtemp(prefix='recording-', dir=str(root)))
    if output is None:
        output = root / (work.name + '.mp4')
    try:
        command = _command(ffmpeg, output, fps, work)
        _write(
            work / 'config.json',
            {
                'command': command,
                'ffprobe': ffprobe,
                'output_path': str(output),
                'max_duration_seconds': request.get('max_duration_seconds', 3600),
                'gui_session': request.get('gui_session', False),
            },
        )
        script = work / 'worker.py'
        script.write_text(source, encoding='utf-8')
        with (work / 'worker.log').open('wb') as log:
            subprocess.Popen(
                [sys.executable, str(script), str(work)],
                stdin=subprocess.DEVNULL,
                stdout=log,
                stderr=log,
                close_fds=True,
                creationflags=(
                    subprocess.DETACHED_PROCESS | subprocess.CREATE_NEW_PROCESS_GROUP if sys.platform == 'win32' else 0
                ),
                start_new_session=sys.platform != 'win32',
            )
        _write(root / 'active.json', {'work': str(work)})
    except Exception:
        shutil.rmtree(str(work), ignore_errors=True)
        raise
    deadline = time.monotonic() + START_TIMEOUT
    while not (work / 'ready').exists():
        if (work / 'result.json').exists():
            return _collect(root, work)
        if time.monotonic() >= deadline:
            (work / 'stop').touch()
            _collect(root, work)
            raise RuntimeError('ffmpeg did not capture any frames within the startup timeout')
        time.sleep(0.1)
    return {'success': True, 'output_path': str(output)}


def _dispatch(root, request, source):
    root.mkdir(parents=True, exist_ok=True)
    with _lock(root):
        if request['action'] == 'start':
            return _start(root, request, source)
        active = root / 'active.json'
        if not active.exists():
            raise RuntimeError('No recording is in progress')
        work = Path(json.loads(active.read_text(encoding='utf-8'))['work'])
        (work / 'stop').touch()
        return _collect(root, work)


def main(payload, source):
    root = Path(tempfile.gettempdir()) / 'nexus' / 'gui-recording'
    try:
        result = _dispatch(root, json.loads(payload), source)
    except Exception as exc:
        result = {'success': False, 'error': str(exc)}
    print('NEXUS_RECORDING_RESULT:' + json.dumps(result))


if __name__ == '__main__':
    _worker(Path(sys.argv[1]))
