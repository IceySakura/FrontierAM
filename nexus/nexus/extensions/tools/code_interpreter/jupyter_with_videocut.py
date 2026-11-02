"""Stateful Jupyter kernel for video inspection, returning each clip as a video Data URL.

The model cuts intervals out of the source video — whose path it reads from the first-turn
message — with the ``ffmpeg_path`` / ``ffprobe_path`` bound into the kernel. It names its own
output, which the tool finds by diffing the working directory, then normalizes and hands back
as ``data:video/mp4;base64,...`` next to a text part naming the file. The file left at that
path is the normalized clip, so the path and the pixels the model saw agree.

Configuration (``tool_configs``)::

    code_interpreter:
      python_path: /path/to/python      # optional: Python inside the container
      work_dir: /tmp/jupyter_video_work # optional: clip output dir inside the container
    videocut:
      clip_max_frames: 256              # optional: frame cap advertised with each clip
      clip_max_bytes: 50331648          # optional: reject clips above this (default 48 MiB)
"""

# NOTE(jiachengdu): no ``from __future__ import annotations`` — ``coerce_args`` reads
# ``Parameter.annotation`` to decide whether to rebuild the pickled dict into
# VideoCutInterpreterInput, and a stringized annotation never matches, so the tool would
# receive a raw dict and die on ``'dict' object has no attribute 'code'`` inside the sandbox.

import ast
import asyncio
import base64
import json
import logging
import math
import os
import re
import shutil
import subprocess
import threading
import uuid
import weakref

from contextlib import suppress
from pathlib import Path
from typing import Any

from openai.types.chat import ChatCompletionContentPartParam
from pydantic import ConfigDict, Field, field_validator

from nexus.core import ToolContext, ToolResponse, ToolType, tool_registry
from nexus.core.types import RobustBaseModel
from nexus.runtime.executors.jupyter import AsyncDirectPythonKernel


logger = logging.getLogger(__name__)

# Both the default and the ceiling for the model-supplied ``timeout``. It bounds the kernel
# execution and the normalizing ffmpeg pass, and a larger value would not buy anything: the
# agent loop's own ``single_tool_timeout`` cancels the call at the same order of magnitude,
# and ``_run_blocking`` shields the ffmpeg thread, so an oversized value only delays session
# cleanup past that cancellation.
_MAX_TIMEOUT_SECONDS = 600.0
_DEFAULT_RUNTIME_WORK_DIR = '/tmp/jupyter_video_work'
_DEFAULT_CLIP_MAX_FRAMES = 256
_VIDEO_EXTENSIONS = ('.mp4', '.mov', '.webm', '.mkv')
_TRANSPORT_PREFIX = '.video_transport_'
_CODE_BLOCK_RE = re.compile(r'```[^\n]*\n(.+?)```', re.DOTALL)

# Encoded size ceiling for a clip handed back to the model: Base64 inflates it by 4/3, and that
# string crosses the Socket.IO channel out of the sandbox and then stays in the conversation for
# the rest of the run. What the clip costs in tokens and decoder memory is a different budget —
# frames times resolution — held by ``clip_max_frames`` and the HF processor.
# Overridable per run via ``tool_configs.videocut.clip_max_bytes``.
DEFAULT_MAX_VIDEO_BYTES = 48 * 1024**2

# Floor imposed by the frame sampler downstream: HF's Qwen2VLVideoProcessor.sample_frames
# rounds the frame budget down to a multiple of temporal_patch_size (2), so a single-frame
# clip yields num_frames=0 and dies on `torch.arange(0, 1, 1/0)` in the rollout's decoder
# subprocess, where nothing points back at the clip being too short.
MIN_VIDEO_FRAMES = 2

# A rollout fires many tool calls into one sandbox at once, and concurrent `apt-get install`
# invocations fail on the dpkg lock.
_FFMPEG_INSTALL_LOCK = threading.Lock()
_RESOLVED_EXECUTABLES: tuple[str | None, str | None] | None = None


# ============================== ffmpeg / ffprobe ==============================
#
# Inlined rather than shared with ``mm_agent.utils.video_utils``: these run in the sandbox,
# where the only importable thing is what this module itself pulls in, and a shared helper
# module would drag the Host's dependencies in behind it.


def _run_command(command: list[str], timeout: float) -> subprocess.CompletedProcess[str]:
    return subprocess.run(command, capture_output=True, text=True, timeout=timeout, check=False)


def _validate_executable(name: str, path: str | None = None) -> tuple[str | None, str | None]:
    resolved = path or shutil.which(name)
    if resolved is None:
        return None, f'{name} was not found on PATH'
    resolved = os.path.realpath(os.path.abspath(resolved))
    try:
        completed = _run_command([resolved, '-version'], 10)
    except (OSError, subprocess.TimeoutExpired) as error:
        return None, f'{name} could not be executed: {error}'
    if completed.returncode != 0:
        detail = completed.stderr.strip() or completed.stdout.strip() or 'unknown error'
        return None, f'{name} is not usable: {detail}'
    return resolved, None


def _validate_libx264(ffmpeg_path: str) -> str | None:
    """Confirm this ffmpeg can encode H264, the one codec the transport contract names."""
    try:
        completed = _run_command([ffmpeg_path, '-hide_banner', '-encoders'], 10)
    except (OSError, subprocess.TimeoutExpired) as error:
        return f'ffmpeg encoder inspection failed: {error}'
    if completed.returncode != 0:
        detail = completed.stderr.strip() or completed.stdout.strip() or 'unknown error'
        return f'ffmpeg encoder inspection failed: {detail}'
    if re.search(r'^\s*V\S*\s+libx264\s', completed.stdout, re.MULTILINE) is None:
        return 'ffmpeg does not provide required encoder libx264'
    return None


def _validate_pair(ffmpeg_path: str | None, ffprobe_path: str | None) -> tuple[str | None, str | None, list[str]]:
    """Validate one ffmpeg/ffprobe candidate pair, including the libx264 encoder."""
    ffmpeg, ffmpeg_error = _validate_executable('ffmpeg', ffmpeg_path)
    ffprobe, ffprobe_error = _validate_executable('ffprobe', ffprobe_path)
    errors = [error for error in (ffmpeg_error, ffprobe_error) if error is not None]
    if not errors:
        assert ffmpeg is not None
        encoder_error = _validate_libx264(ffmpeg)
        if encoder_error is not None:
            errors.append(encoder_error)
    return (None, None, errors) if errors else (ffmpeg, ffprobe, [])


def _apt_run(apt_get: str, *args: str) -> tuple[bool, str]:
    """Run one apt-get invocation. Returns ``(ok, error_tail)``."""
    try:
        # 600s: this runs at most once per container, and every concurrent tool call in
        # that sandbox waits behind it, so a premature timeout is worse than a slow one.
        completed = subprocess.run(  # noqa: S603
            [apt_get, *args],
            capture_output=True,
            text=True,
            timeout=600.0,
            env={**os.environ, 'DEBIAN_FRONTEND': 'noninteractive'},
        )
    except (OSError, subprocess.TimeoutExpired) as error:
        return False, f'apt-get {" ".join(args)} failed: {error}'
    if completed.returncode == 0:
        return True, ''
    return False, (completed.stderr or completed.stdout or 'unknown error').strip()[-400:]


def _apt_install_ffmpeg() -> tuple[str | None, str | None, str | None]:
    """Install ffmpeg with apt-get. Returns ``(ffmpeg, ffprobe, error)``.

    This is the route that works in the Gongfeng RL sandbox. That image is Nix-based:
    it carries ffmpeg's *libraries* (``/nix/store/*-ffmpeg-headless-7.1.1-lib``) but no
    ffmpeg/ffprobe executables, and the Nix store is read-only, so the ``static_ffmpeg``
    pip package cannot be installed there — its ``twine -> keyring -> cryptography``
    dependency chain pulls a new ``packaging`` and uv dies trying to replace the
    read-only copy already seeded in the worker venv. ``apt-get`` is present and
    ``/tmp`` / ``$HOME`` are writable, so a plain system install is the simplest thing
    that actually works.
    """
    apt_get = shutil.which('apt-get')
    if apt_get is None:
        return None, None, 'apt-get is not available'

    # Install first: the sandbox image ships populated apt lists, so `apt-get update`
    # is redundant and costs ~30s per cold container. Fall back to update + retry only
    # when the package cannot be located, which is the one failure update can fix.
    install_args = ('install', '-y', '--no-install-recommends', 'ffmpeg')
    ok, error = _apt_run(apt_get, *install_args)
    if not ok and 'Unable to locate package' in error:
        ok, update_error = _apt_run(apt_get, 'update')
        if not ok:
            return None, None, f'apt-get update failed: {update_error}'
        ok, error = _apt_run(apt_get, *install_args)
    if not ok:
        return None, None, f'apt-get install failed: {error}'
    return shutil.which('ffmpeg'), shutil.which('ffprobe'), None


def resolve_video_executables() -> tuple[str | None, str | None, list[str]]:
    """Return ``(ffmpeg, ffprobe, errors)``, installing ffmpeg if the host lacks it.

    apt-get is the only install route; see this module's ``pyproject.toml`` for why
    ``static_ffmpeg`` cannot be used where it would be needed.
    """
    # Cached after the first success: every tool call would otherwise spend three subprocesses
    # (two -version, one -encoders) re-deriving an answer that cannot change under us.
    global _RESOLVED_EXECUTABLES
    if _RESOLVED_EXECUTABLES is not None:
        return (*_RESOLVED_EXECUTABLES, [])

    ffmpeg, ffprobe, errors = _validate_pair(None, None)
    if not errors:
        _RESOLVED_EXECUTABLES = (ffmpeg, ffprobe)
        return ffmpeg, ffprobe, []

    with _FFMPEG_INSTALL_LOCK:
        if _RESOLVED_EXECUTABLES is not None:  # Installed while we waited for the lock.
            return (*_RESOLVED_EXECUTABLES, [])
        ffmpeg, ffprobe, retry_errors = _validate_pair(None, None)
        if not retry_errors:
            _RESOLVED_EXECUTABLES = (ffmpeg, ffprobe)
            return ffmpeg, ffprobe, []
        apt_ffmpeg, apt_ffprobe, apt_error = _apt_install_ffmpeg()
    if apt_error is not None:
        return None, None, [*errors, f'apt ffmpeg install failed: {apt_error}']
    if apt_ffmpeg is None or apt_ffprobe is None:
        return None, None, [*errors, 'apt-get reported success but ffmpeg is still not on PATH']
    apt_ffmpeg, apt_ffprobe, apt_errors = _validate_pair(apt_ffmpeg, apt_ffprobe)
    if apt_errors:
        return None, None, [*errors, *apt_errors]
    _RESOLVED_EXECUTABLES = (apt_ffmpeg, apt_ffprobe)
    return apt_ffmpeg, apt_ffprobe, []


def ffprobe_json(
    path: str, ffprobe_path: str, timeout: float, *entry_args: str
) -> tuple[dict[str, Any] | None, str | None]:
    """Run ffprobe with ``entry_args`` and decode its JSON output."""
    command = [
        os.path.realpath(ffprobe_path), '-hide_banner', '-loglevel', 'error',
        *entry_args, '-of', 'json', path,
    ]  # fmt: skip
    try:
        completed = _run_command(command, timeout)
    except subprocess.TimeoutExpired:
        return None, f'ffprobe timed out after {timeout:g} seconds'
    except OSError as error:
        return None, f'failed to run ffprobe: {error}'
    if completed.returncode != 0:
        detail = completed.stderr.strip() or 'unknown ffprobe error'
        return None, f'ffprobe exited with code {completed.returncode}: {detail}'
    try:
        payload = json.loads(completed.stdout)
    except json.JSONDecodeError as error:
        return None, f'ffprobe returned invalid JSON: {error}'
    if not isinstance(payload, dict):
        return None, 'ffprobe returned a non-object result'
    return payload, None


def parse_frame_rate(value: Any) -> float | None:
    """Parse an ffprobe rate, which may be a ``"num/den"`` rational."""
    if not isinstance(value, str) or not value:
        return None
    try:
        if '/' in value:
            numerator, denominator = value.split('/', 1)
            rate = float(numerator) / float(denominator)
        else:
            rate = float(value)
    except (TypeError, ValueError, ZeroDivisionError):
        return None
    return rate if math.isfinite(rate) and rate > 0 else None


def single_stream(payload: dict[str, Any]) -> tuple[dict[str, Any] | None, str | None]:
    """Take the one stream a ``-select_streams v:0`` probe should have returned."""
    streams = payload.get('streams')
    if not isinstance(streams, list) or len(streams) != 1 or not isinstance(streams[0], dict):
        return None, 'ffprobe must return exactly one video stream'
    return streams[0], None


def probe_clip_video(
    path: str, ffprobe_path: str, timeout: float, max_bytes: int = DEFAULT_MAX_VIDEO_BYTES
) -> tuple[dict[str, Any] | None, str | None]:
    """Verify one generated clip against the transport contract.

    ``-count_frames`` decodes the whole file, affordable only because this runs on a clip
    rather than on the source. No ``-select_streams``: the audio check needs every stream.
    """
    payload, error = ffprobe_json(
        path, ffprobe_path, timeout,
        '-count_frames',
        '-show_entries',
        'stream=codec_type,codec_name,pix_fmt,width,height,avg_frame_rate,r_frame_rate,duration,nb_read_frames:'
        'format=format_name,duration',
    )  # fmt: skip
    if error is not None:
        return None, error
    assert payload is not None

    streams = payload.get('streams')
    if not isinstance(streams, list) or not all(isinstance(stream, dict) for stream in streams):
        return None, 'ffprobe returned invalid streams'
    video_streams = [stream for stream in streams if stream.get('codec_type') == 'video']
    if len(video_streams) != 1:
        return None, 'clip must contain exactly one video stream'
    if any(stream.get('codec_type') == 'audio' for stream in streams):
        return None, 'clip must not contain audio'
    stream = video_streams[0]
    format_info = payload.get('format')
    if not isinstance(format_info, dict):
        return None, 'ffprobe returned invalid format metadata'
    format_name = format_info.get('format_name')
    if not isinstance(format_name, str) or 'mp4' not in format_name.split(','):
        return None, 'clip format must be MP4'
    if stream.get('codec_name') != 'h264':
        return None, 'clip codec must be H264'
    if stream.get('pix_fmt') != 'yuv420p':
        return None, 'clip pixel format must be yuv420p'
    # Reported back with the clip, not gated on: the frame rate only has to be readable so
    # the decoder downstream knows how to sample it.
    fps = parse_frame_rate(stream.get('avg_frame_rate')) or parse_frame_rate(stream.get('r_frame_rate'))
    if fps is None:
        return None, 'clip frame rate could not be determined'
    try:
        duration = float(stream.get('duration') or format_info['duration'])
        frame_count = int(stream['nb_read_frames'])
        width = int(stream['width'])
        height = int(stream['height'])
        size = os.path.getsize(path)
    except (KeyError, TypeError, ValueError, OSError) as error:
        return None, f'invalid clip metadata: {error}'
    if not math.isfinite(duration) or duration <= 0 or frame_count <= 0 or size <= 0:
        return None, 'clip duration, frame count and size must be positive'
    if frame_count < MIN_VIDEO_FRAMES:
        return None, (
            f'clip holds {frame_count} frame(s), below the {MIN_VIDEO_FRAMES}-frame minimum; '
            'select a longer interval or raise fps'
        )
    # Even dimensions are a yuv420p requirement, not a budget — an odd axis will not encode
    # at all. Size itself is left to the HF processor, which resizes whatever it receives.
    if width <= 0 or height <= 0 or width % 2 or height % 2:
        return None, 'clip dimensions must be positive and even'
    if size > max_bytes:
        return None, (
            f'clip is {size / 1024**2:.1f} MiB, above the {max_bytes / 1024**2:.0f} MiB limit; '
            'select a shorter interval or lower the fps / resolution'
        )
    return {
        'size': size,
        'duration': duration,
        'frame_count': frame_count,
        'fps': fps,
        'width': width,
        'height': height,
    }, None


def standardize_clip(
    source_path: Path,
    transport_path: Path,
    ffmpeg_path: str,
    ffprobe_path: str,
    timeout: float,
    max_bytes: int = DEFAULT_MAX_VIDEO_BYTES,
) -> tuple[dict[str, Any] | None, str | None]:
    """Normalize one model-produced video into a silent H264/yuv420p MP4.

    Remuxing suffices when the candidate already matches the contract; unknown codec metadata
    falls back to encoding, and ``probe_clip_video`` validates the result either way, so a
    wrong guess cannot ship a bad clip.
    """
    payload, error = ffprobe_json(
        str(source_path), ffprobe_path, timeout,
        '-select_streams', 'v:0',
        '-show_entries', 'stream=codec_name,pix_fmt:format=format_name',
    )  # fmt: skip
    if error is not None:
        return None, error
    assert payload is not None
    stream, error = single_stream(payload)
    if error is not None:
        return None, error
    assert stream is not None
    format_info = payload.get('format')
    format_name = format_info.get('format_name') if isinstance(format_info, dict) else None

    conformant = (
        stream.get('codec_name') == 'h264'
        and stream.get('pix_fmt') == 'yuv420p'
        and 'mp4' in (format_name or '').split(',')
    )
    encode_args = (
        ['-c:v', 'copy', '-avoid_negative_ts', 'make_zero']
        if conformant
        else ['-vf', 'setpts=PTS-STARTPTS,format=yuv420p', '-c:v', 'libx264', '-pix_fmt', 'yuv420p']
    )
    command = [
        ffmpeg_path, '-nostdin', '-hide_banner', '-loglevel', 'error',
        '-i', str(source_path), '-map', '0:v:0', '-an', '-sn', '-dn',
        *encode_args, '-movflags', '+faststart', '-f', 'mp4', '-y', str(transport_path),
    ]  # fmt: skip
    try:
        completed = _run_command(command, timeout)
    except subprocess.TimeoutExpired:
        return None, f'ffmpeg timed out after {timeout:g} seconds'
    except OSError as error:
        return None, f'failed to run ffmpeg: {error}'
    if completed.returncode != 0:
        detail = completed.stderr.strip() or 'unknown ffmpeg error'
        return None, f'ffmpeg exited with code {completed.returncode}: {detail}'
    return probe_clip_video(str(transport_path), ffprobe_path, timeout, max_bytes)


# ============================== input ==============================


class VideoCutInterpreterInput(RobustBaseModel):
    """Python code executed by the persistent video interpreter."""

    model_config = ConfigDict(extra='forbid')

    code: str = Field(description='Python code to execute in the stateful Jupyter kernel.')
    timeout: float = Field(
        default=_MAX_TIMEOUT_SECONDS,
        description=f'Execution timeout in seconds, capped at {_MAX_TIMEOUT_SECONDS:g}.',
    )

    @field_validator('timeout', mode='before')
    @classmethod
    def reject_invalid_timeout(cls, value: Any) -> Any:
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise ValueError('timeout must be a JSON number')
        value = float(value)
        if not math.isfinite(value) or value <= 0:
            raise ValueError('timeout must be finite and positive')
        # Clamp rather than reject: an oversized timeout is a throwaway guess, not a
        # mistake worth spending a turn on.
        return min(value, _MAX_TIMEOUT_SECONDS)


# ============================== kernel management ==============================


# TODO(jiachengdu): this runner, ``_KERNELS`` and ``_get_kernel_runner`` duplicate
# jupyter_with_image.py:49-136. Extract a shared runner in a follow-up MR -- it touches
# jupyter_with_image's own callers, so it does not belong in this one.
class _VideoKernelRunner:
    """One persistent Jupyter kernel plus an isolated output directory per run."""

    def __init__(self, run_id: str, python_path: str | None = None, work_dir: str | None = None) -> None:
        self._kernel = AsyncDirectPythonKernel(python_path)
        self._started = False
        self._start_lock = asyncio.Lock()
        self._run_lock = asyncio.Lock()
        root = Path(work_dir or _DEFAULT_RUNTIME_WORK_DIR).resolve()
        root.mkdir(parents=True, exist_ok=True)
        self.run_id = run_id
        # ``run_id`` alone: it is already unique per run, and the clip paths go into the
        # conversation, so anything else in the name is one more thing a downstream path
        # scrubber has to fold back out before it can key a cache on them.
        self.work_dir = root / run_id
        self.work_dir.mkdir(parents=True, exist_ok=True)
        self.clip_count = 0

    async def _ensure_started(self) -> None:
        if self._started:
            return
        async with self._start_lock:
            if self._started:
                return
            await self._kernel.start()
            self._started = True
            try:
                await self._kernel.execute(f'import os; os.chdir({str(self.work_dir)!r})')
            except BaseException:
                await self._kernel.close()
                self._started = False
                raise

    async def run(self, code: str, bindings: dict[str, Any], timeout: float):
        async with self._run_lock:
            await self._ensure_started()
            preamble = '\n'.join(f'{name} = {value!r}' for name, value in bindings.items())
            await self._kernel.execute(preamble, timeout=timeout)
            return await self._kernel.execute(code, timeout=timeout)

    async def close(self) -> None:
        async with self._run_lock:
            async with self._start_lock:
                if self._started:
                    await self._kernel.close()
                    self._started = False


# Key: event loop -> {run_id -> runner}. run_id comes from ToolContext, so each agent
# instance gets its own kernel and its own clip directory.
_KERNELS: weakref.WeakKeyDictionary[asyncio.AbstractEventLoop, dict[str, _VideoKernelRunner]] = (
    weakref.WeakKeyDictionary()
)


def _get_kernel_runner(run_id: str, python_path: str | None = None, work_dir: str | None = None) -> _VideoKernelRunner:
    runners = _KERNELS.setdefault(asyncio.get_running_loop(), {})
    runner = runners.get(run_id)
    if runner is None:
        runner = _VideoKernelRunner(run_id, python_path=python_path, work_dir=work_dir)
        runners[run_id] = runner
    return runner


async def close_videocut_session(run_id: str | None = None) -> dict[str, Any]:
    """Close this run's kernel and remove its clip directory. Invoked by the task."""
    runners = _KERNELS.get(asyncio.get_running_loop(), {})
    runner = runners.pop(run_id, None) if run_id else None
    if runner is None:
        return {'closed': False}
    try:
        await runner.close()
    finally:
        shutil.rmtree(runner.work_dir, ignore_errors=True)
    return {'closed': True, 'work_dir': str(runner.work_dir)}


# ============================== clip discovery ==============================
#
# Module-level rather than ``_VideoKernelRunner`` methods so an evaluator running cells with
# plain ``subprocess`` can import the same selection logic instead of reimplementing it.


def video_snapshot(work_dir: Path) -> dict[str, tuple[int, int]]:
    """Fingerprint the videos in ``work_dir`` so new ones can be spotted.

    ``(mtime_ns, size)`` rather than a name-only set: the model picks its own filename and
    may well reuse one across turns, which would silently drop the second clip.
    """
    if not Path(work_dir).is_dir():
        return {}
    return {
        entry.name: (entry.stat().st_mtime_ns, entry.stat().st_size)
        for entry in os.scandir(work_dir)
        if entry.is_file()
        and entry.name.lower().endswith(_VIDEO_EXTENSIONS)
        and not entry.name.startswith(_TRANSPORT_PREFIX)
    }


def new_video_paths(work_dir: Path, before: dict[str, tuple[int, int]]) -> list[Path]:
    """Videos written or rewritten since ``before`` was taken, newest last."""
    work_dir = Path(work_dir)
    changed = [name for name, mark in video_snapshot(work_dir).items() if before.get(name) != mark]
    return sorted((work_dir / name for name in changed), key=lambda p: p.stat().st_mtime_ns)


# ============================== code validation ==============================


def _normalize_code(code: str) -> str:
    matches = _CODE_BLOCK_RE.findall(code)
    return (matches[0] if matches else code).strip()


def validate_code(code: str) -> None:
    """Parse the cell and reject ``shell=True``.

    Deliberately the only static check: everything about the clip is read off the file the
    cell actually produced (:func:`probe_clip_video`). ``shell=True`` is the exception because
    its damage is invisible in the result — a path holding a space word-splits into different
    ffmpeg arguments, and the failure surfaces with no hint of the real cause.
    """
    for node in ast.walk(ast.parse(code)):
        if isinstance(node, ast.Call):
            for keyword in node.keywords:
                if keyword.arg == 'shell' and isinstance(keyword.value, ast.Constant) and keyword.value.value is True:
                    raise ValueError('shell=True is not allowed; pass ffmpeg arguments as a list')


# ============================== output helpers ==============================


def _format_kernel_output(result: Any) -> str:
    parts: list[str] = []
    if result.stdout:
        parts.append(f'--- stdout ---\n{result.stdout}')
    if result.stderr:
        parts.append(f'--- stderr ---\n{result.stderr}')
    for item in result.display or []:
        if isinstance(item, dict) and 'text/plain' in item:
            parts.append(f'--- display (text/plain) ---\n{item["text/plain"]}')
    for error in result.error or []:
        parts.append('--- error ---\n' + '\n'.join(error.get('traceback', [])))
    return '\n'.join(parts)


async def _run_blocking(function, *args, **kwargs):
    """Run blocking ffmpeg work in a thread, joining it even if we are cancelled.

    Cancelling a thread await does not stop ffmpeg, so the worker has to finish before
    session cleanup can remove files it is still writing.
    """
    worker = asyncio.create_task(asyncio.to_thread(function, *args, **kwargs))
    try:
        return await asyncio.shield(worker)
    except asyncio.CancelledError:
        with suppress(Exception):
            await asyncio.shield(worker)
        raise


def _error_response(message: str) -> ToolResponse:
    return ToolResponse(
        output=[{'type': 'text', 'text': f'Video interpreter failed: {message}'}],
        meta={'success': False, 'exit_code': 1},
    )


# ============================== tool ==============================

# Exported so an evaluator declaring this tool to a plain OpenAI endpoint describes it the
# same way it was described during training.
TOOL_DESCRIPTION = (
    'Executes Python code in a stateful Jupyter kernel to inspect a video more closely.\n\n'
    'Prebound variables: ffmpeg_path, ffprobe_path. The source video path is given in the '
    'conversation.\n\n'
    'Requirements for each call:\n'
    '- Cut with ffmpeg_path via a subprocess argument list; never use shell=True.\n'
    '- Save at most one video (MP4, MOV, WEBM or MKV) in the current working directory; '
    'if a cell writes several, only the newest comes back.\n\n'
    'The saved video is returned as a silent H264/yuv420p MP4 for you to watch, together with the '
    'path it was written to, which you may reopen in a later call.'
)


@tool_registry.register(
    'jupyter_with_videocut',
    meta={
        'name': 'videocut_interpreter',
        'description': TOOL_DESCRIPTION,
        'tool_type': ToolType.RUNTIME,
    },
)
async def jupyter_with_videocut_tool(params: VideoCutInterpreterInput, context: ToolContext) -> ToolResponse:
    """Cut one interval out of the source video and return it as an inline MP4."""
    try:
        code = _normalize_code(params.code)
        if not code:
            raise ValueError('empty code provided')
        validate_code(code)
    except (SyntaxError, ValueError) as error:
        return _error_response(str(error))

    # Without a run_id every concurrent sample would share one kernel, one working directory
    # and one cleanup — the first ``close_videocut_session`` would rmtree the others' clips.
    if not context.run_id:
        return _error_response('videocut_interpreter requires a run_id on the tool context')

    ffmpeg_path, ffprobe_path, executable_errors = await _run_blocking(resolve_video_executables)
    if executable_errors or ffmpeg_path is None or ffprobe_path is None:
        return _error_response('; '.join(executable_errors))

    interpreter_config = context.tool_configs.get('code_interpreter', {})
    videocut_config = context.tool_configs.get('videocut', {})
    runner = _get_kernel_runner(
        run_id=context.run_id,
        python_path=interpreter_config.get('python_path', os.environ.get('PYTHON_PATH')),
        work_dir=interpreter_config.get(
            'work_dir', os.environ.get('CODE_INTERPRETER_WORK_DIR', _DEFAULT_RUNTIME_WORK_DIR)
        ),
    )
    before = video_snapshot(runner.work_dir)

    try:
        result = await runner.run(
            code,
            {'ffmpeg_path': ffmpeg_path, 'ffprobe_path': ffprobe_path},
            params.timeout,
        )
    except Exception as error:
        logger.exception('Video interpreter execution failed')
        return _error_response(f'error executing code: {error}')

    # Matches ``ToolResponse.output``, which widens the OpenAI part union with a bare dict for
    # the ``{'type': 'video', ...}`` part this tool returns.
    output: list[ChatCompletionContentPartParam | dict] = []
    formatted_output = _format_kernel_output(result)
    if formatted_output:
        output.append({'type': 'text', 'text': formatted_output})

    # One model action yields at most one visual observation; extra files are ignored.
    candidates = new_video_paths(runner.work_dir, before)
    if not candidates:
        # Not necessarily a mistake: a cell may legitimately just probe the source. Say
        # what happened rather than scolding, and let ``success`` follow the cell itself so
        # a clean probe is not reported as a failed tool call.
        ok = result.status == 'ok'
        output.append({
            'type': 'text',
            'text': (
                'No video was written by this cell; save one in the working directory when you want to watch a clip.'
                if ok
                else 'The cell failed and no video was written.'
            ),
        })
        return ToolResponse(
            output=output,
            meta={'success': ok, 'exit_code': 0 if ok else 1, 'clip': None},
        )

    candidate = candidates[-1]
    runner.clip_count += 1

    transport_path = runner.work_dir / f'{_TRANSPORT_PREFIX}{uuid.uuid4().hex}.mp4'
    try:
        metadata, error = await _run_blocking(
            standardize_clip,
            candidate,
            transport_path,
            ffmpeg_path,
            ffprobe_path,
            params.timeout,
            int(videocut_config.get('clip_max_bytes', DEFAULT_MAX_VIDEO_BYTES)),
        )
        if error is not None or metadata is None:
            output.append({'type': 'text', 'text': f'--- video processing error ---\n{candidate.name}: {error}'})
            return ToolResponse(output=output, meta={'success': False, 'exit_code': 1, 'clip': None})

        clip_bytes = await _run_blocking(transport_path.read_bytes)
        # Publish the normalized clip at the path the model was told to write, so the path
        # in the conversation names a file whose bytes are the ones it just watched.
        # Re-encoding leaves the model's own output behind otherwise.
        os.replace(transport_path, candidate)
    finally:
        transport_path.unlink(missing_ok=True)

    max_frames = min(int(videocut_config.get('clip_max_frames', _DEFAULT_CLIP_MAX_FRAMES)), metadata['frame_count'])
    data_url = f'data:video/mp4;base64,{base64.b64encode(clip_bytes).decode("ascii")}'
    if len(candidates) > 1:
        output.append({
            'type': 'text',
            'text': f'Ignored {len(candidates) - 1} additional video file(s); one call returns one clip.',
        })
    output.extend([
        {'type': 'text', 'text': f'\n\n![{candidate.name}]({candidate})'},
        {
            'type': 'text',
            'text': (
                f'Clip {runner.clip_count}: silent H264/yuv420p MP4 at {metadata["fps"]:g} fps, '
                f'{metadata["width"]}x{metadata["height"]}, {metadata["duration"]:.3f}s.'
            ),
        },
        {'type': 'video', 'video': data_url, 'fps': metadata['fps'], 'max_frames': max_frames},
    ])
    return ToolResponse(
        output=output,
        meta={
            'success': result.status == 'ok',
            'exit_code': 0 if result.status == 'ok' else 1,
            'stdout': result.stdout,
            'stderr': result.stderr,
            'clip': {'clip_number': runner.clip_count, 'clip_path': str(candidate), **metadata},
        },
    )
