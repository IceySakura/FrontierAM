"""Staging a source video into the runtime, plus the offline frame-extraction CLI.

The sandbox half of the pipeline — probing and normalizing the clips a model produces —
lives in ``tools/code_interpreter/jupyter_with_videocut.py``, so a tool call running in the
sandbox never imports this task package.

The transfer helpers below are the exception: the Host ships them into the runtime by import
path via ``runtime.invoke``, so they execute there and must keep their imports to the stdlib
plus what the tool module already pulls in.

Offline usage::

    python -m nexus.extensions.tasks.mm_agent.utils.video_utils \\
        --dataset data.jsonl --out-dir frames/
"""

from __future__ import annotations

import argparse
import asyncio
import base64
import hashlib
import json
import logging
import math
import os
import shutil
import stat
import subprocess
import tempfile
import time
import uuid

from contextlib import suppress
from pathlib import Path
from typing import Any
from urllib.parse import urlsplit

from nexus.core import require_context
from nexus.runtime.executors import FileItem


logger = logging.getLogger(__name__)


# ============================== staged path ==============================


def staged_video_path(video_path: str, run_id: str | None) -> str:
    """Where ``video_path`` is staged inside the runtime: ``/tmp/<run_id>/video_0<ext>``.

    ``video_0<ext>`` is the filename HyVision-Toolbox stages media under and the one the SFT
    data already uses, so the model sees a name it was trained on rather than a long
    shared-storage path.

    Scoped by ``run_id`` because the staged copy is per-sample state: GRPO runs n rollouts of
    one prompt concurrently, and a scope keyed on the video's own identity would give them all
    the same file — the first to finish deletes it out from under the rest.
    """
    scope = run_id or f'anon_{uuid.uuid4().hex[:12]}'
    return f'/tmp/{scope}/video_0{Path(video_path).suffix.lower() or ".mp4"}'


# ============================== runtime-side transfer helpers ==============================
#
# Invoked through ``runtime.invoke``, which ships an import path rather than a closure, so
# these execute inside the sandbox and must not rely on anything the worker venv lacks.


class SourceVideoDownloadError(RuntimeError):
    """Source download failure, carrying the HTTP status when the server supplied one."""

    def __init__(self, message: str, *, http_status: int | None = None) -> None:
        super().__init__(message)
        self.http_status = http_status

    def __reduce__(self) -> Any:
        return type(self), (str(self),), {'http_status': self.http_status}


async def download_source_video(
    url: str,
    destination_path: str,
    timeout: float,
    expected_size: int | None = None,
    expected_sha256: str | None = None,
) -> dict[str, Any]:
    """Download one source video inside the sandbox and land it atomically.

    Invoked over ``runtime.invoke`` so the Host never has to read the bytes itself.
    ``expected_size`` / ``expected_sha256`` are optional: datasets that ship a presigned
    URL usually carry neither, in which case the caller validates the result by probing
    it rather than by comparing bytes.
    """
    import aiohttp

    started = time.perf_counter()
    parsed = urlsplit(url)
    if parsed.scheme not in ('http', 'https') or not parsed.hostname:
        raise ValueError('source URL must be an absolute HTTP(S) URL')
    destination = Path(destination_path)
    if not destination.is_absolute():
        raise ValueError('source destination must be an absolute path')
    destination.parent.mkdir(parents=True, exist_ok=True)

    temporary = destination.with_name(f'.{destination.name}.{uuid.uuid4().hex}.part')
    digest = hashlib.sha256()
    actual_size = 0
    try:
        client_timeout = aiohttp.ClientTimeout(total=timeout, connect=min(30.0, timeout), sock_read=timeout)
        try:
            async with aiohttp.ClientSession(timeout=client_timeout) as session:
                async with session.get(url) as response:
                    if response.status != 200:
                        raise SourceVideoDownloadError(
                            f'source download returned HTTP {response.status}', http_status=response.status
                        )
                    with temporary.open('wb') as file:
                        async for chunk in response.content.iter_chunked(1024 * 1024):
                            actual_size += len(chunk)
                            if expected_size is not None and actual_size > expected_size:
                                raise SourceVideoDownloadError('source download exceeded expected size')
                            digest.update(chunk)
                            file.write(chunk)
        except aiohttp.ClientError as error:
            raise SourceVideoDownloadError(f'source download failed: {type(error).__name__}') from None

        actual_sha256 = digest.hexdigest()
        if expected_size is not None and actual_size != expected_size:
            raise SourceVideoDownloadError('source download size does not match expected size')
        if expected_sha256 is not None and actual_sha256 != expected_sha256:
            raise SourceVideoDownloadError('source download SHA256 does not match expected value')
        os.replace(temporary, destination)
        return {
            'path': str(destination),
            'size_bytes': actual_size,
            'sha256': actual_sha256,
            'download_seconds': time.perf_counter() - started,
        }
    finally:
        temporary.unlink(missing_ok=True)


async def probe_source_video_here(path: str, timeout: float) -> dict[str, Any]:
    """Resolve ffprobe inside the runtime and probe ``path``, raising on any failure.

    A Host-resolved ffprobe path is useless in the sandbox, so the lookup has to happen here.
    """
    # Both halves run in threads: this coroutine is on the worker loop that answers the
    # Socket.IO invoke channel, and a cold-container `apt-get install ffmpeg` would block it
    # long enough for the Host to time the whole sample out with num_turns=0.
    from nexus.extensions.tools.code_interpreter.jupyter_with_videocut import resolve_video_executables

    _, ffprobe_path, errors = await asyncio.to_thread(resolve_video_executables)
    if errors or ffprobe_path is None:
        raise RuntimeError(f'ffprobe unavailable: {"; ".join(errors)}')
    metadata, error = await asyncio.to_thread(probe_source_video, path, ffprobe_path, timeout)
    if error is not None or metadata is None:
        raise RuntimeError(f'source video failed ffprobe validation: {error}')
    return metadata


def copy_source_video(source_path: str, destination_path: str) -> None:
    """Copy an already-visible source to ``destination_path`` inside the runtime.

    A real copy rather than a symlink: a cell writing to a symlinked staged path would follow
    the link and overwrite the dataset's own video in place, and the sandbox runs as root, so
    the source's permission bits do not stop it.
    """
    source = Path(source_path)
    destination = Path(destination_path)
    if not source.is_file():
        raise ValueError(f'source video is not a file: {source_path}')
    destination.parent.mkdir(parents=True, exist_ok=True)
    # Copy to a scratch name and rename: a partial copy must never be visible at the staged
    # path, and two samples sharing a source race to create the same one.
    temporary = destination.with_name(f'.{destination.name}.{uuid.uuid4().hex}.part')
    try:
        shutil.copyfile(source, temporary)
        os.replace(temporary, destination)
    finally:
        temporary.unlink(missing_ok=True)  # No-op once the rename succeeded.


def remove_staged_video(destination_path: str) -> None:
    """Delete a staged source video and its run directory.

    Under ``local`` the runtime is this process, so without this ``/tmp`` keeps one full copy
    of every source video the run ever touched.

    Best-effort: this runs in the task's ``finally``, and a sample that answered correctly must
    not fail over a leftover file.
    """
    destination = Path(destination_path)
    with suppress(OSError):
        destination.unlink(missing_ok=True)
    with suppress(OSError):  # Only when empty; never recursive, in case the dir holds more.
        destination.parent.rmdir()


def stat_file_here(path: str) -> dict[str, int] | None:
    """Identity of ``path`` as seen from this process, or ``None`` if it is not a file.

    Lets the Host tell whether the runtime already shares the filesystem holding a
    source video, so a GB-scale copy can be skipped.
    """
    try:
        result = Path(path).stat()
    except OSError:
        return None
    if not stat.S_ISREG(result.st_mode):
        return None
    return {'dev': result.st_dev, 'ino': result.st_ino, 'size': result.st_size}


# ============================== source probing ==============================


SUPPORTED_CONTAINERS = frozenset({'mp4', 'mov', 'matroska', 'webm'})

#: Host-memory ceiling for the upload route only, measured at ~7x the file: the Base64 content
#: is pickled and Base64-encoded again by ``InvokeRequest.build``, then decoded once more
#: before the blob offload, and the copies overlap. 256 MiB peaks around 1.8 GiB, times the
#: concurrent staging samples. Raise with ``NEXUS_MAX_UPLOAD_MIB``, or pass ``video_cos_urls``
#: and let the sandbox download -- that route streams, as does the shared-mount copy.
MAX_UPLOAD_BYTES = int(os.environ.get('NEXUS_MAX_UPLOAD_MIB', '256')) * 1024**2


def probe_source_video(path: str, ffprobe_path: str, timeout: float) -> tuple[dict[str, Any] | None, str | None]:
    """Probe a source video's duration and dimensions in one pass.

    Deliberately permissive about codec, container and frame rate: a source is whatever
    the caller supplied. ``-count_frames`` style probing is unaffordable here — on a
    GB-scale source it measured 82 s versus 59 ms.
    """
    from nexus.extensions.tools.code_interpreter.jupyter_with_videocut import ffprobe_json, single_stream

    payload, error = ffprobe_json(
        path, ffprobe_path, timeout,
        '-select_streams', 'v:0',
        '-show_entries', 'stream=codec_type,width,height,duration:format=format_name,duration',
    )  # fmt: skip
    if error is not None:
        return None, error
    assert payload is not None
    stream, error = single_stream(payload)
    if error is not None:
        return None, error
    assert stream is not None
    if stream.get('codec_type') != 'video':
        return None, 'ffprobe found no video stream'
    format_info = payload.get('format')
    if not isinstance(format_info, dict):
        return None, 'ffprobe returned invalid source format metadata'
    format_name = format_info.get('format_name')
    if not isinstance(format_name, str) or not format_name:
        return None, 'ffprobe returned invalid source format name'
    try:
        duration = float(stream.get('duration') or format_info['duration'])
        width = int(stream['width'])
        height = int(stream['height'])
        size = os.path.getsize(path)
    except (KeyError, TypeError, ValueError, OSError) as error:
        return None, f'invalid source video metadata: {error}'
    if not math.isfinite(duration) or duration <= 0:
        return None, 'source duration must be finite and positive'
    if width <= 0 or height <= 0 or size <= 0:
        return None, 'source dimensions and size must be positive'
    if not {name.strip().lower() for name in format_name.split(',')} & SUPPORTED_CONTAINERS:
        return None, f'unsupported source container: {format_name}'
    return {
        'path': os.path.abspath(path),
        'duration': duration,
        'width': width,
        'height': height,
        'size': size,
        'format_name': format_name,
    }, None


# ============================== host-side staging ==============================


async def prepare_video_source(
    videos: list[dict[str, Any]],
    cos_urls: list[str] | None,
    timeout: float,
) -> dict[str, Any]:
    """Make the first video readable inside the runtime and build the tool's state entry.

    Three routes, cheapest first — local copy if the runtime already shares the filesystem,
    presigned URL if the dataset carries one, byte upload otherwise — all landing at the same
    staged path so the model sees one path shape. On success the staged file becomes the
    caller's to delete (:func:`remove_staged_video`); a failure in here cleans up its own.
    """
    # Sharing is decided by (device, inode, size), not mere existence: a sandbox may have an
    # unrelated file at the same path, and cutting clips out of it would silently train on the
    # wrong observations.
    ctx = require_context()
    runtime = await ctx.get_runtime('main')
    video_path = videos[0]['video']
    cos_url = (cos_urls or [None])[0]
    runtime_path = staged_video_path(video_path, ctx.run_id)

    try:
        local: os.stat_result | None = Path(video_path).stat()
    except OSError:
        # Normal for a dataset the Host cannot see but the sandbox can fetch by URL.
        local = None
    shared = False
    if local is not None:
        # An invoke failure here is a real fault (sandbox import error, dropped connection),
        # not a "not shared" answer — but the transfer routes below may still work, so log it
        # loudly and let them try rather than failing the sample on the probe alone.
        try:
            remote = await runtime.invoke(stat_file_here, video_path)  # type: ignore[arg-type]
        except Exception:
            logger.warning(
                'Runtime visibility probe failed for %s; falling back to transfer', video_path, exc_info=True
            )
        else:
            shared = bool(remote) and (remote['dev'], remote['ino'], remote['size']) == (
                local.st_dev,
                local.st_ino,
                local.st_size,
            )

    try:
        if shared:
            await runtime.invoke(copy_source_video, video_path, runtime_path)
            logger.info('Source video is already visible to the runtime; copied locally: %s', runtime_path)
        elif cos_url:
            result = await runtime.invoke(download_source_video, cos_url, runtime_path, timeout)
            logger.info(
                'Sandbox fetched source video (%s bytes in %.1fs) -> %s',
                result.get('size_bytes'),
                result.get('download_seconds', 0.0),
                runtime_path,
            )
        else:
            # Last resort: the Host holds the file, its Base64 and its pickle at once, so cap
            # it and say what to do instead rather than dying in an allocator.
            size = local.st_size if local is not None else Path(video_path).stat().st_size
            if size > MAX_UPLOAD_BYTES:
                raise ValueError(
                    f'source video is {size / 1024**2:.0f} MiB, over the '
                    f'{MAX_UPLOAD_BYTES / 1024**2:.0f} MiB upload limit; supply a presigned URL '
                    f'(video_cos_urls) or mount the dataset into the runtime'
                )
            payload = await asyncio.to_thread(Path(video_path).read_bytes)
            encoded = await asyncio.to_thread(lambda: base64.b64encode(payload).decode('utf-8'))
            await runtime.upload_file([FileItem(path=runtime_path, content=encoded, encoding='base64')], overwrite=True)
            logger.info('Uploaded source video (%d bytes) -> %s', len(payload), runtime_path)

        metadata = await runtime.invoke(probe_source_video_here, runtime_path, timeout)
    except BaseException:
        # The bytes are already on disk by the time probing runs, and the caller only starts
        # tracking the staged copy once this returns.
        with suppress(Exception):
            await runtime.invoke(remove_staged_video, runtime_path)
        raise
    return {'source_path': runtime_path, 'duration': metadata['duration']}


# ============================== offline data preparation ==============================
#
# Reached only through ``python -m``; RL never imports any of this. During rollout the
# frame sampling is driven by the dataset's ``fps`` / ``max_frames`` and the resize happens
# inside the HF processor that vLLM calls.


def frame_budget(duration: float, fps: float, max_frames: int, min_frames: int = 1) -> int:
    """Frames a ``fps`` sampling of ``duration`` yields, clamped to the item's bounds."""
    return max(int(min_frames), min(int(max_frames), math.ceil(duration * fps)))


def resize_for_budget(width: int, height: int, target_pixels: int = 100 * 32 * 32) -> tuple[int, int]:
    """Scale ``width``x``height`` down to the pixel budget without ever upscaling.

    Each axis is floored to an even integer so libx264/yuv420p can encode it.
    """
    scale = min(1.0, math.sqrt(target_pixels / (width * height)))
    return max(2, int(width * scale) // 2 * 2), max(2, int(height * scale) // 2 * 2)


def extract_frames(
    source_path: Path,
    destination: Path,
    ffmpeg_path: str,
    ffprobe_path: str,
    timeout: float,
    *,
    fps: float,
    max_frames: int,
    min_frames: int = 1,
    target_pixels: int = 100 * 32 * 32,
) -> dict[str, Any]:
    """Sample ``source_path`` to PNG frames under ``destination``.

    Mirrors what the dataset's ``fps`` / ``max_frames`` / ``min_frames`` ask the rollout
    decoder for, so an offline sweep sees the same frames the policy would.
    """
    metadata, error = probe_source_video(str(source_path), ffprobe_path, timeout)
    if error is not None or metadata is None:
        raise RuntimeError(f'{source_path.name} failed ffprobe validation: {error}')

    budget = frame_budget(metadata['duration'], fps, max_frames, min_frames)
    target_fps = max(min(fps, budget / metadata['duration']), 0.001)
    width, height = resize_for_budget(metadata['width'], metadata['height'], target_pixels)
    destination.mkdir(parents=True, exist_ok=True)
    command = [
        ffmpeg_path, '-nostdin', '-hide_banner', '-loglevel', 'error',
        '-i', str(source_path), '-map', '0:v:0',
        '-vf', f'fps={target_fps:.3f},scale=w={width}:h={height},setsar=1',
        '-frames:v', str(budget), '-y', str(destination / 'frame_%05d.png'),
    ]  # fmt: skip
    completed = subprocess.run(command, capture_output=True, text=True, timeout=timeout, check=False)
    if completed.returncode != 0:
        detail = completed.stderr.strip() or 'unknown ffmpeg error'
        raise RuntimeError(f'frame extraction failed with code {completed.returncode}: {detail}')
    frames = sorted(destination.glob('frame_*.png'))
    return {
        'source': str(source_path),
        'frame_dir': str(destination),
        'frame_count': len(frames),
        'frame_budget': budget,
        'sampled_fps': target_fps,
        'width': width,
        'height': height,
        'source_duration': metadata['duration'],
    }


def _iter_video_items(dataset: Path):
    """Yield ``(row_index, video_item)`` for every video in a JSONL dataset."""
    with dataset.open(encoding='utf-8') as file:
        for index, line in enumerate(file):
            line = line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as error:
                logger.warning('Skipping malformed row at %s:%d: %s', dataset, index + 1, error)
                continue
            for item in row.get('videos') or []:
                if isinstance(item, dict) and isinstance(item.get('video'), str):
                    yield index, item


def _atomic_write_bytes(destination: Path, payload: bytes) -> None:
    """Write ``payload`` to ``destination`` via a same-directory temporary file."""
    destination.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f'.{destination.name}.', suffix='.part', dir=destination.parent
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, 'wb') as file:
            file.write(payload)
            file.flush()
            os.fsync(file.fileno())
        os.replace(temporary, destination)
    finally:
        temporary.unlink(missing_ok=True)


def main() -> int:
    """Pre-extract frames for a JSONL dataset's videos, honoring each item's own policy.

    Offline tool, reached only through ``python -m``; RL training never imports it. The
    sampling here (``frame_budget``, then ``resize_for_budget``) has to stay in step with the
    rollout decoder, or a sweep run on these frames measures a view the policy never saw.
    """
    from nexus.extensions.tools.code_interpreter.jupyter_with_videocut import resolve_video_executables

    parser = argparse.ArgumentParser(description=(main.__doc__ or '').split('\n\n')[0])
    parser.add_argument('--dataset', type=Path, required=True, help='JSONL dataset with a "videos" column')
    parser.add_argument('--out-dir', type=Path, required=True, help='Directory receiving one subdir per video')
    parser.add_argument('--fps', type=float, default=None, help='Override every item fps')
    parser.add_argument('--max-frames', type=int, default=None, help='Override every item max_frames')
    parser.add_argument(
        '--target-pixels', type=int, default=100 * 32 * 32, help='Per-frame pixel budget; never upscales'
    )
    parser.add_argument('--timeout', type=float, default=1800.0)
    parser.add_argument('--limit', type=int, default=None, help='Stop after this many videos')
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format='%(levelname)s %(message)s')
    ffmpeg_path, ffprobe_path, errors = resolve_video_executables()
    if errors or ffmpeg_path is None or ffprobe_path is None:
        raise SystemExit(f'ffmpeg/ffprobe unavailable: {"; ".join(errors)}')

    manifest: list[dict[str, Any]] = []
    for count, (row_index, item) in enumerate(_iter_video_items(args.dataset)):
        if args.limit is not None and count >= args.limit:
            break
        source = Path(item['video'])
        destination = args.out_dir / f'{row_index:06d}_{source.stem}'
        try:
            result = extract_frames(
                source,
                destination,
                ffmpeg_path,
                ffprobe_path,
                args.timeout,
                fps=args.fps if args.fps is not None else float(item.get('fps', 2.0)),
                max_frames=args.max_frames if args.max_frames is not None else int(item.get('max_frames', 384)),
                min_frames=int(item.get('min_frames', 1)),
                target_pixels=args.target_pixels,
            )
        except Exception as error:
            logger.warning('Row %d: %s failed: %s', row_index, source.name, error)
            manifest.append({'row': row_index, 'source': str(source), 'error': str(error)})
            continue
        logger.info('Row %d: %s -> %d frames', row_index, source.name, result['frame_count'])
        manifest.append({'row': row_index, **result})

    _atomic_write_bytes(
        args.out_dir / 'manifest.json', json.dumps(manifest, ensure_ascii=False, indent=2).encode('utf-8')
    )
    failures = sum(1 for entry in manifest if 'error' in entry)
    logger.info('Processed %d video(s), %d failed', len(manifest), failures)
    return 1 if failures else 0


if __name__ == '__main__':
    raise SystemExit(main())
