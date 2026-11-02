import io
import logging
import math
import os
import subprocess
import tempfile

from typing import Literal, cast

from pydantic import BaseModel, Field

from nexus.core import ToolContext, ToolResponse, ToolType, tool_registry


logger = logging.getLogger(__name__)


class VideoCutInput(BaseModel):
    """Input for video_cut_tool."""

    temporal_segment: list[float] = Field(
        description=(
            'The temporal segment to zoom in, as [t_start, t_end] in seconds. '
            't_start must be at least 1 second less than t_end.'
        ),
        min_length=2,
        max_length=2,
    )
    sampling_strategy: Literal['coarse', 'medium', 'fine'] = Field(
        default='medium',
        description=(
            'Sampling strategy controlling the resolution of the cropped clip. '
            "'coarse' uses fewer tokens (max 2048), 'medium' uses moderate tokens (max 4096), "
            "'fine' uses more tokens (max 6144)."
        ),
    )


def _crop_video_segment_ffmpeg(
    video_path: str,
    t_start: float,
    t_end: float,
    sampling_strategy: str,
) -> str:
    """Crop a temporal segment from a video file using ffmpeg.

    Produces a temporary video file containing the cropped segment.
    The caller is responsible for cleaning up the temp file if needed.

    Args:
        video_path: Path to the source video file.
        t_start: Start time in seconds.
        t_end: End time in seconds.
        sampling_strategy: One of 'coarse', 'medium', 'fine' (currently unused
            for ffmpeg mode, but reserved for future resolution control).

    Returns:
        output_path (str): Path to the cropped temporary video file.
    """
    duration = t_end - t_start
    if duration <= 0:
        raise ValueError(f'Invalid temporal segment [{t_start}, {t_end}]: duration must be > 0')

    # Create a temp file for the cropped segment
    suffix = os.path.splitext(video_path)[1] or '.mp4'
    fd, output_path = tempfile.mkstemp(suffix=suffix, prefix='videocut_crop_')
    os.close(fd)

    cmd = [
        'ffmpeg',
        '-y',
        '-ss',
        f'{t_start:.3f}',
        '-i',
        video_path,
        '-t',
        f'{duration:.3f}',
        '-c',
        'copy',
        '-avoid_negative_ts',
        'make_zero',
        output_path,
    ]

    logger.debug(f'ffmpeg crop command: {" ".join(cmd)}')

    try:
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=60,
        )
        if result.returncode != 0:
            # If stream copy fails (e.g. keyframe issues), retry with re-encoding
            logger.warning(
                f'ffmpeg stream-copy failed (rc={result.returncode}), '
                f'retrying with re-encode: {result.stderr[-500:] if result.stderr else ""}'
            )
            cmd_reencode = [
                'ffmpeg',
                '-y',
                '-ss',
                f'{t_start:.3f}',
                '-i',
                video_path,
                '-t',
                f'{duration:.3f}',
                '-c:v',
                'libx264',
                '-preset',
                'ultrafast',
                '-crf',
                '18',
                '-c:a',
                'copy',
                '-avoid_negative_ts',
                'make_zero',
                output_path,
            ]
            result = subprocess.run(
                cmd_reencode,
                capture_output=True,
                text=True,
                timeout=120,
            )
            if result.returncode != 0:
                raise RuntimeError(
                    f'ffmpeg re-encode also failed (rc={result.returncode}): '
                    f'{result.stderr[-500:] if result.stderr else ""}'
                )
    except Exception as e:
        if os.path.exists(output_path):
            os.unlink(output_path)
        raise RuntimeError('ffmpeg crop failed') from e

    if not os.path.exists(output_path) or os.path.getsize(output_path) == 0:
        if os.path.exists(output_path):
            os.unlink(output_path)
        raise RuntimeError('ffmpeg produced an empty output file')

    return output_path


# ── Tool registration ──────────────────────────────────────────────────────────


@tool_registry.register(
    'videocut/video_cut_tool',
    meta={
        'name': 'video_cut_tool',
        'description': (
            'Zoom in on a specific temporal segment of a video by cropping it based '
            'on a time range [t_start, t_end] (in seconds) and a sampling strategy. '
            'Use this tool when you need to examine a particular time window of the '
            'video more closely.'
        ),
        'tool_type': ToolType.LOCAL,
    },
)
async def video_cut_tool(params: VideoCutInput, context: ToolContext) -> ToolResponse:
    """Crop a temporal segment from the source video and return the frames.

    Supports two modes based on what is available in tool_state:
      - video_path mode (Mode 1): Uses ffmpeg to crop the video file and returns
        the cropped clip as a base64 video_url.
      - frame_paths mode (Mode 2): Selects frames within the temporal segment from
        the stored frame_paths list and returns them as base64 image_url parts.

    Reads from tool_state:
      - 'video_path' (str): path to the original video file (Mode 1)
      - 'frame_paths' (list[str]): ordered frame image paths or base64 strings (Mode 2)
      - 'video_fps' (float): fps of the video / frames
      - 'video_duration' (float): total duration in seconds
    """
    # ── Retrieve video source from tool state ─────────────────────────────────
    video_path_raw = context.tool_state.get('video_path')
    frame_paths_raw = context.tool_state.get('frame_paths')
    video_fps_raw = context.tool_state.get('video_fps')
    video_duration_state = context.tool_state.get('video_duration')

    video_path = video_path_raw if isinstance(video_path_raw, str) else None
    frame_paths = frame_paths_raw if isinstance(frame_paths_raw, list) else None
    video_fps = float(video_fps_raw) if isinstance(video_fps_raw, (int, float)) else None

    # Determine which mode we are in
    use_ffmpeg = video_path is not None
    use_frames = frame_paths is not None

    if not use_ffmpeg and not use_frames:
        return ToolResponse(
            output=[{'type': 'text', 'text': 'Error: no video_path or frame_paths available in tool state.'}],
            meta={'error': 'no_video'},
        )

    if video_fps is None or video_fps <= 0:
        return ToolResponse(
            output=[{'type': 'text', 'text': 'Error: video_fps in tool state is missing or invalid.'}],
            meta={'error': 'invalid_fps'},
        )

    t_start, t_end = params.temporal_segment

    # Compute max_duration from tool_state or from frame_paths/fps
    if video_duration_state is not None:
        max_duration = cast(float, video_duration_state)
    elif use_frames:
        frame_paths = cast(list[str], frame_paths)
        total_frames = len(frame_paths)
        max_duration = total_frames / video_fps
    else:
        # For ffmpeg mode without explicit duration
        max_duration = float('inf')

    # ── Clamp and validate temporal segment ───────────────────────────────────
    if t_start < 0.0:
        logger.warning(f'video_cut_tool: t_start={t_start} < 0, clamping to 0')
        t_start = 0.0
    if t_end > max_duration:
        logger.warning(f'video_cut_tool: t_end={t_end} > video_duration={max_duration:.2f}s, clamping')
        t_end = max_duration

    if t_start > t_end - 1.0:
        msg = (
            f'Error: temporal_segment [{t_start:.2f}, {t_end:.2f}] is invalid. '
            f't_start must be at least 1 second less than t_end. '
            f'Video duration is {max_duration:.2f}s.'
        )
        logger.warning(f'video_cut_tool: {msg}')
        return ToolResponse(
            output=[{'type': 'text', 'text': msg}],
            meta={'error': 'invalid_temporal_segment', 'temporal_segment': params.temporal_segment},
        )

    # ── Crop and return ───────────────────────────────────────────────────────
    if use_ffmpeg:
        # Mode 1: ffmpeg-based cropping — convert cropped clip to base64 video_url
        import os

        from nexus.extensions.projects.videocut.task import video_url_to_base64

        video_path = cast(str, video_path)

        try:
            cropped_path = _crop_video_segment_ffmpeg(
                video_path=video_path,
                t_start=t_start,
                t_end=t_end,
                sampling_strategy=params.sampling_strategy,
            )
        except Exception as exc:
            msg = f'Error: failed to crop video segment [{t_start:.2f}, {t_end:.2f}] with ffmpeg: {exc}'
            logger.warning(f'video_cut_tool: {msg}')
            return ToolResponse(
                output=[{'type': 'text', 'text': msg}],
                meta={'error': 'crop_failed', 'temporal_segment': params.temporal_segment},
            )

        try:
            video_data_url = video_url_to_base64(video_source=cropped_path)
        finally:
            if os.path.exists(cropped_path):
                os.unlink(cropped_path)

        desc = (
            f'Zoomed in on the video segment [{t_start:.2f}s, {t_end:.2f}s] '
            f'using strategy "{params.sampling_strategy}".'
        )

        logger.debug(
            f'video_cut_tool: ffmpeg segment=[{t_start:.2f}, {t_end:.2f}], '
            f'strategy={params.sampling_strategy}, '
            f'data_url_length={len(video_data_url)}'
        )

        output = cast(
            list,
            [
                {'type': 'text', 'text': desc},
                {'type': 'video_url', 'video_url': {'url': video_data_url}},
            ],
        )

        return ToolResponse(
            output=output,
            meta={
                'temporal_segment': [t_start, t_end],
                'sampling_strategy': params.sampling_strategy,
            },
        )

    # Mode 2: frame_paths-based cropping — select frames in segment, encode as image_url
    from PIL import Image

    from nexus.extensions.projects.videocut.task import encode_images_dynamic_resize

    frame_paths = cast(list[str], frame_paths)
    total_frames = len(frame_paths)
    start_frame = max(0, math.floor(t_start * video_fps))
    end_frame = min(total_frames - 1, math.ceil(t_end * video_fps))

    if start_frame >= end_frame:
        msg = (
            f'Error: temporal_segment [{t_start:.2f}, {t_end:.2f}] maps to '
            f'start_frame={start_frame} >= end_frame={end_frame}.'
        )
        logger.warning(f'video_cut_tool: {msg}')
        return ToolResponse(
            output=[{'type': 'text', 'text': msg}],
            meta={'error': 'invalid_temporal_segment', 'temporal_segment': params.temporal_segment},
        )

    # Determine token budget based on sampling strategy
    strategy_max_frames = {'coarse': 16, 'medium': 32, 'fine': 64}
    max_clip_frames = strategy_max_frames.get(params.sampling_strategy, 32)

    clip_frame_paths = frame_paths[start_frame : end_frame + 1]
    if len(clip_frame_paths) > max_clip_frames:
        indices = [round(i * (len(clip_frame_paths) - 1) / (max_clip_frames - 1)) for i in range(max_clip_frames)]
        clip_frame_paths = [clip_frame_paths[i] for i in indices]

    # Load PIL images
    pil_images = []
    for entry in clip_frame_paths:
        if entry.startswith('data:'):
            import base64 as _b64

            _, _, b64_data = entry.partition(',')
            img = Image.open(io.BytesIO(_b64.b64decode(b64_data))).convert('RGB')
        else:
            img = Image.open(entry).convert('RGB')
        pil_images.append(img)

    frames_data_urls, _ = encode_images_dynamic_resize(pil_images)

    clip_duration = (end_frame - start_frame) / max(video_fps, 1e-6)
    actual_fps = len(pil_images) / max(clip_duration, 1e-6)

    desc = (
        f'Zoomed in on the video segment [{t_start:.2f}s, {t_end:.2f}s] '
        f'using strategy "{params.sampling_strategy}". '
        f'Extracted {len(pil_images)} frames at {actual_fps:.2f} fps, '
        f'frame size: {pil_images[0].size[0]}x{pil_images[0].size[1]}.'
    )

    logger.debug(
        f'video_cut_tool: frame_paths segment=[{t_start:.2f}, {t_end:.2f}], '
        f'strategy={params.sampling_strategy}, '
        f'nframes={len(pil_images)}, fps={actual_fps:.2f}'
    )

    output = cast(list, [{'type': 'text', 'text': desc}])
    for url in frames_data_urls:
        output.append({'type': 'image_url', 'image_url': {'url': url}})

    return ToolResponse(
        output=output,
        meta={
            'temporal_segment': [t_start, t_end],
            'sampling_strategy': params.sampling_strategy,
            'nframes': len(pil_images),
            'actual_fps': actual_fps,
            'frame_size': list(pil_images[0].size),
        },
    )
