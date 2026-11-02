"""VideoCut video tool-calling task.

Implements the VideoCut training task for video temporal reasoning with tool use.

Two input modes are supported:
  Mode 1 (video_path): Provide a video file path. The video is converted to a
      base64 data URL via video_url_to_base64 and passed as a video_url content
      part. The video_cut_tool uses ffmpeg to crop temporal segments.
  Mode 2 (frame_paths): Provide pre-extracted frame image paths or base64 strings.
      Frames are encoded via encode_images_dynamic_resize and passed as image_url
      content parts, consistent with Mode 1's approach.

Data format (from parquet):
{
    "data_source": "videomme",
    "prompt": [
        {"role": "system", "content": "..."},
        {"role": "user", "content": [{"type": "video_url", ...}, {"type": "text", ...}]}
    ],
    "video_path": "/path/to/video.mp4",
    "ground_truth": "...",
    "question": "...",
    "extra_info": {"answer": "...", "index": "0", "split": "train"}
}

The task:
1. For video_path mode: converts video to base64 data URL and passes as video_url;
   for frame_paths mode: encodes frames as base64 and passes as image_url list.
2. Runs the VideoCut agent with the prompt
3. Computes the reward using LLM-as-a-judge (via the 'judge' OpenAI client)
"""

import base64
import io
import logging
import math
import mimetypes
import os
import subprocess
import tempfile

from collections.abc import Sequence
from urllib.parse import urlparse

import requests

from PIL import Image

from nexus.core import (
    Registry,
    TaskRunInput,
    TaskRunOutput,
    WithAgentRunner,
    WithMainTrajectory,
    WithScore,
    require_context,
    task_registry,
)
from nexus.extensions.projects.videocut.prompts import SYSTEM_PROMPT_MULTI_ROUND_MC
from nexus.extensions.projects.videocut.reward import compute_rule_score, compute_score


logger = logging.getLogger(__name__)
Registry.import_module('nexus.extensions.projects.videocut.agent')


def _round_by_factor(number: int, factor: int) -> int:
    return round(number / factor) * factor


def _ceil_by_factor(number: int, factor: int) -> int:
    return math.ceil(number / factor) * factor


def _floor_by_factor(number: int, factor: int) -> int:
    return math.floor(number / factor) * factor


def _resize_frame(width, height, max_pixels, min_pixels, factor, max_ratio=50):
    if max(height, width) / min(height, width) > max_ratio:
        pass  # skip warning in task context
    h_bar = max(factor, _round_by_factor(height, factor))
    w_bar = max(factor, _round_by_factor(width, factor))
    if h_bar * w_bar > max_pixels:
        beta = math.sqrt((height * width) / max_pixels)
        h_bar = _floor_by_factor(height / beta, factor)
        w_bar = _floor_by_factor(width / beta, factor)
    elif h_bar * w_bar < min_pixels:
        beta = math.sqrt(min_pixels / (height * width))
        h_bar_temp = _ceil_by_factor(height * beta, factor)
        w_bar_temp = _ceil_by_factor(width * beta, factor)
        if h_bar_temp * w_bar_temp <= max_pixels:
            h_bar = h_bar_temp
            w_bar = w_bar_temp
    return (w_bar, h_bar), ''


def _adaptive_resize(num_frames, size_info):
    patch_size = 32
    max_image_token_occupy = 48000 - 4096
    default_frame_max_pixels = 576 * 576
    default_frame_min_pixels = 256 * 256

    if not size_info:
        return None, None
    width, height = size_info
    max_video_pixels = max_image_token_occupy * patch_size * patch_size
    max_frame_pixels = max(
        min(max_video_pixels // num_frames, default_frame_max_pixels),
        default_frame_min_pixels,
    )
    (new_width, new_height), _ = _resize_frame(width, height, max_frame_pixels, default_frame_min_pixels, patch_size)
    return new_width, new_height


def _resize_image_to_target(img, target_size_mb):
    scale = 1.0
    while True:
        new_width = int(img.width * scale)
        new_height = int(img.height * scale)
        resized = img.resize((new_width, new_height), Image.Resampling.LANCZOS)
        buf = io.BytesIO()
        resized.save(buf, format=img.format if img.format else 'JPEG', quality=90)
        if buf.tell() <= target_size_mb * 1024 * 1024 or scale <= 0.1:
            buf.seek(0)
            return base64.b64encode(buf.read()).decode('utf-8')
        scale *= 0.8


def encode_images_dynamic_resize(
    images: Sequence[Image.Image | str],
    max_total_size_mb: float | None = None,
    max_single_size_mb: float | None = None,
) -> tuple[list[str], float | None]:
    """Encode a list of PIL Images to base64 data URLs with adaptive resize."""
    num_img = len(images)
    if num_img == 0:
        return [], None
    use_adaptive = max_total_size_mb is None or max_single_size_mb is None
    if use_adaptive:
        target_size_per_img = None
    else:
        assert max_total_size_mb is not None
        assert max_single_size_mb is not None
        target_size_per_img = min(max_total_size_mb / num_img, max_single_size_mb)

    base64_list = []
    for img in images:
        if isinstance(img, str):
            img = Image.open(img)
        # Ensure RGB
        if img.mode in ('RGBA', 'LA', 'P'):
            background = Image.new('RGB', img.size, (255, 255, 255))
            if img.mode == 'P':
                img = img.convert('RGBA')
            if img.mode in ('RGBA', 'LA'):
                background.paste(img, mask=img.split()[-1])
            else:
                background.paste(img)
            img = background
        elif img.mode != 'RGB':
            img = img.convert('RGB')

        if use_adaptive:
            width, height = img.size
            new_width, new_height = _adaptive_resize(num_img, (width, height))
            if new_width and new_height and (new_width != width or new_height != height):
                img = img.resize((new_width, new_height), Image.Resampling.LANCZOS)

        buf = io.BytesIO()
        img.save(buf, format='JPEG', quality=90)
        buf.seek(0)
        current_size = len(buf.getvalue())

        if use_adaptive:
            base64_list.append(base64.b64encode(buf.getvalue()).decode('utf-8'))
        else:
            assert target_size_per_img is not None
            if current_size > target_size_per_img * 1024 * 1024:
                base64_list.append(_resize_image_to_target(img, target_size_per_img))
            else:
                base64_list.append(base64.b64encode(buf.getvalue()).decode('utf-8'))

    base64_list = [f'data:image/jpeg;base64,{b64}' for b64 in base64_list]
    return base64_list, target_size_per_img


def video_url_to_base64(
    video_source: str,
    timeout: float = 20.0,
    fps: float | None = None,
    max_base64_size_mb: float | None = None,
    encode_profiles: list[dict[str, int | float]] | None = None,
    start_time: float = -1,
    end_time: float = -1,
    keep_temp_video: bool = False,
) -> str:
    """Convert a local video file path or remote URL to a base64 data URL."""

    def _default_mime():
        return 'video/mp4'

    def _encode_bytes(content, mime):
        return f'data:{mime};base64,{base64.b64encode(content).decode("utf-8")}'

    def _is_within_size_limit(content, mime, limit_mb):
        if limit_mb is None:
            return True
        prefix = f'data:{mime};base64,'
        base64_len = 4 * ((len(content) + 2) // 3)
        return len(prefix.encode('utf-8')) + base64_len <= int(limit_mb * 1024 * 1024)

    def _build_default_profiles(target_fps):
        base_fps = target_fps if (target_fps is not None and target_fps > 0) else 2
        return [
            {'fps': base_fps, 'max_width': 2048, 'max_height': 2048, 'crf': 28},
            {'fps': base_fps, 'max_width': 720, 'max_height': 720, 'crf': 32},
            {'fps': base_fps, 'max_width': 480, 'max_height': 480, 'crf': 34},
            {'fps': 1, 'max_width': 480, 'max_height': 480, 'crf': 34},
        ]

    def _encode_video_with_profile(input_path, profile, start_time=-1, end_time=-1):
        tmp_path = None
        try:
            with tempfile.NamedTemporaryFile(delete=False, suffix='.mp4') as tmp:
                tmp_path = tmp.name
            vf_filters = []
            target_fps = profile.get('fps')
            if target_fps is not None and target_fps > 0:
                vf_filters.append(f'fps={target_fps}')
            max_w, max_h = profile.get('max_width'), profile.get('max_height')
            if max_w and max_h:
                vf_filters.append(f'scale=w={int(max_w)}:h={int(max_h)}:force_original_aspect_ratio=decrease')
                vf_filters.append('scale=trunc(iw/2)*2:trunc(ih/2)*2')
            cmd = [
                'ffmpeg',
                '-hide_banner',
                '-loglevel',
                'error',
                '-fflags',
                '+discardcorrupt',
                '-err_detect',
                'ignore_err',
            ]
            if start_time != -1:
                cmd.extend(['-ss', str(start_time)])
            if end_time != -1:
                duration = float(end_time) if start_time == -1 else float(end_time) - float(start_time)
                cmd.extend(['-t', str(duration)])
            cmd.extend(['-i', input_path])
            filter_idx = len(cmd)
            cmd.extend([
                '-c:v',
                'libx264',
                '-crf',
                str(profile.get('crf', 23)),
                '-preset',
                profile.get('preset', 'veryfast'),
                '-pix_fmt',
                'yuv420p',
                '-an',
                '-movflags',
                '+faststart',
                '-f',
                'mp4',
                tmp_path,
                '-y',
            ])
            if vf_filters:
                cmd[filter_idx:filter_idx] = ['-vf', ','.join(vf_filters)]
            result = subprocess.run(cmd, capture_output=True, timeout=300)
            if result.returncode != 0:
                raise ValueError(f'FFmpeg error: {result.stderr.decode()}')
            with open(tmp_path, 'rb') as f:
                return f.read()
        finally:
            if tmp_path and os.path.exists(tmp_path):
                if keep_temp_video:
                    logger.debug(f'video_url_to_base64: temp file kept at {tmp_path}')
                else:
                    os.remove(tmp_path)

    parsed = urlparse(video_source)
    is_http = parsed.scheme in {'http', 'https'}
    local_path = video_source
    source_bytes = None
    source_mime = _default_mime()
    tmp_input_path = None

    try:
        if is_http:
            resp = requests.get(video_source, timeout=timeout)
            if resp.status_code != 200:
                raise ValueError(f'HTTP request failed with status {resp.status_code}')
            source_bytes = resp.content
            source_mime = resp.headers.get('Content-Type') or _default_mime()
            if ';' in source_mime:
                source_mime = source_mime.split(';', 1)[0].strip()
            with tempfile.NamedTemporaryFile(delete=False, suffix='.mp4') as tmp:
                tmp.write(source_bytes)
                tmp_input_path = tmp.name
            local_path = tmp_input_path
        else:
            if not os.path.isfile(video_source):
                raise FileNotFoundError(f'Local file not found: {video_source}')
            mime, _ = mimetypes.guess_type(video_source)
            source_mime = mime or _default_mime()
            with open(video_source, 'rb') as f:
                source_bytes = f.read()

        if max_base64_size_mb is None or _is_within_size_limit(source_bytes, source_mime, max_base64_size_mb):
            profile = {'crf': 23} if fps is None else {'fps': fps, 'crf': 23}
            content = _encode_video_with_profile(local_path, profile, start_time=start_time, end_time=end_time)
            return _encode_bytes(content, 'video/mp4')

        profiles = encode_profiles if encode_profiles else _build_default_profiles(fps)
        for profile in profiles:
            content = _encode_video_with_profile(local_path, profile, start_time=start_time, end_time=end_time)
            if _is_within_size_limit(content, 'video/mp4', max_base64_size_mb):
                return _encode_bytes(content, 'video/mp4')
        raise ValueError(f'Cannot compress video to within {max_base64_size_mb}MB with any profile')

    finally:
        if tmp_input_path and os.path.exists(tmp_input_path):
            os.remove(tmp_input_path)


# ── Constants ──────────────────────────────────────────────────────────────────
DEFAULT_FPS = 2.0
DEFAULT_MAX_FRAMES = 512


# ── Video info helper ──────────────────────────────────────────────────────────


def _get_video_info(video_path: str) -> tuple[float, float]:
    """Get video duration and native fps using decord (no full decode).

    Returns:
        video_duration (float): total duration in seconds.
        native_fps (float): the native fps of the video.
    """
    from decord import VideoReader, cpu

    vr = VideoReader(video_path, ctx=cpu(0))
    total_frames = len(vr)
    native_fps = vr.get_avg_fps()
    video_duration = total_frames / max(native_fps, 1e-6)
    return video_duration, native_fps


# ── Prompt builders ────────────────────────────────────────────────────────────


def _build_first_turn_prompt_video(
    question_text: str,
    video_data_url: str,
) -> list[dict]:
    """Build the first-turn prompt for video_path mode (Mode 1).

    Embeds the video as a base64 data URL in a video_url content part.

    Args:
        question_text:  The raw question + options text from the dataset prompt.
        video_data_url: Base64 data URL of the video (data:video/mp4;base64,...).
    """
    return [
        {'role': 'system', 'content': SYSTEM_PROMPT_MULTI_ROUND_MC},
        {
            'role': 'user',
            'content': [
                {'type': 'video_url', 'video_url': {'url': video_data_url}},
                {'type': 'text', 'text': question_text},
            ],
        },
    ]


def _build_first_turn_prompt_frames(
    question_text: str,
    frames_data_urls: list[str],
) -> list[dict]:
    """Build the first-turn prompt for frame_paths mode (Mode 2).

    Embeds pre-extracted frames as base64 image_url content parts,
    consistent with Mode 1's approach of passing base64 data URLs.

    Args:
        question_text:    The raw question + options text from the dataset prompt.
        frames_data_urls: List of base64 data URLs for each frame
                          (data:image/jpeg;base64,...).
    """
    frame_parts = [{'type': 'image_url', 'image_url': {'url': url}} for url in frames_data_urls]

    return [
        {'role': 'system', 'content': SYSTEM_PROMPT_MULTI_ROUND_MC},
        {
            'role': 'user',
            'content': frame_parts + [{'type': 'text', 'text': question_text}],
        },
    ]


# ── Task I/O models ────────────────────────────────────────────────────────────


class VideoCutTaskInput(TaskRunInput, WithAgentRunner):
    """VideoCut task input.

    Fields map directly to the parquet data columns.

    prompt:           raw messages from the dataset (system + user)
    video_path:       path to the video file on disk
    ground_truth:     reference answer
    question:         original question text
    max_iterations:   max agent turns
    reward_style:     'model' for LLM-as-a-judge, 'rule' for exact match
    sample_fps:       fps used to sample the overview video frames for the prompt
    max_frames:       maximum number of overview frames to include in the prompt
    frame_paths:      list of image file paths or base64-encoded image strings
    """

    prompt: list[dict]
    video_path: str | None = None
    frame_paths: list[str] | None = None
    ground_truth: str
    question: str
    max_iterations: int = 10
    reward_style: str = 'model'
    sample_fps: float = DEFAULT_FPS
    max_frames: int = DEFAULT_MAX_FRAMES
    video_fps: float = DEFAULT_FPS
    max_base64_size_mb: float | None = None


class VideoCutTaskOutput(TaskRunOutput, WithScore):
    """VideoCut task output."""

    response: str  # Last assistant message content
    correct: bool  # Whether acc_reward > 0.5


# ── Helpers ────────────────────────────────────────────────────────────────────


def _extract_last_response(trajectory: list[dict]) -> str:
    """Return the last assistant message content from a trajectory."""
    logger.debug(f'VideoCut task: extracting last response from trajectory of {len(trajectory)} messages')
    for msg in reversed(trajectory):
        if msg.get('role') == 'assistant':
            content = msg.get('content', '')
            if isinstance(content, str):
                return content
            if isinstance(content, list):
                texts = [p.get('text', '') for p in content if p.get('type') == 'text']
                return ' '.join(texts)
            refusal = msg.get('refusal')
            if isinstance(refusal, str) and refusal:
                return refusal
            reasoning_content = msg.get('reasoning_content')
            if isinstance(reasoning_content, str) and reasoning_content:
                return reasoning_content
    return ''


# ── Task registration ──────────────────────────────────────────────────────────


@task_registry.register('videocut')
async def videocut_task(task_input: VideoCutTaskInput) -> VideoCutTaskOutput:
    """VideoCut video temporal reasoning task.

    Steps:
    1. Load the video from task_input.video_path using decord
    2. Sample overview frames and inject them into the prompt messages
    3. Store the full video tensor in tool_state so video_cut_tool can access it
    4. Run the VideoCut agent
    5. Compute reward via LLM-as-a-judge using the 'judge' OpenAI client
    """
    logger.debug(
        f'VideoCut task: question="{task_input.question}", '
        f'ground_truth="{task_input.ground_truth}", '
        f'video_path="{task_input.video_path}", '
        f'frame_paths={"[" + str(len(task_input.frame_paths)) + " frames]" if task_input.frame_paths else None}'
    )

    # ── Extract the question text from the raw parquet prompt ───────────────
    question_text = ''
    for msg in task_input.prompt:
        if msg.get('role') == 'user':
            content = msg.get('content', '')
            if isinstance(content, str):
                question_text = content
            elif isinstance(content, list):
                texts = [p.get('text', '') for p in content if p.get('type') == 'text']
                question_text = '\n'.join(texts)
            break

    ctx = require_context()

    # ── Load video / frames and build prompt ──────────────────────────────────
    if task_input.frame_paths:
        # Mode 2: frame_paths — encode frames as base64 image_url list

        from PIL import Image

        frame_paths = task_input.frame_paths
        total_frames = len(frame_paths)
        if total_frames == 0:
            raise ValueError('frame_paths is empty.')

        # Sub-sample to max_frames
        max_frames = task_input.max_frames
        if total_frames > max_frames:
            indices = [round(i * (total_frames - 1) / (max_frames - 1)) for i in range(max_frames)]
            frame_paths = [frame_paths[i] for i in indices]

        video_duration = total_frames / max(task_input.sample_fps, 1e-6)
        actual_fps = len(frame_paths) / max(video_duration, 1e-6)

        # Load PIL images (support file path or base64 data URL)
        pil_images = []
        for entry in frame_paths:
            if entry.startswith('data:'):
                import base64 as _b64
                import io as _io

                _, _, b64_data = entry.partition(',')
                img = Image.open(_io.BytesIO(_b64.b64decode(b64_data))).convert('RGB')
            else:
                img = Image.open(entry).convert('RGB')
            pil_images.append(img)

        # Encode with adaptive resize
        frames_data_urls, _ = encode_images_dynamic_resize(pil_images)

        logger.debug(
            f'VideoCut task: encoded {len(frames_data_urls)} frames from frame_paths, '
            f'fps={actual_fps:.2f}, duration={video_duration:.2f}s'
        )

        prompt = _build_first_turn_prompt_frames(
            question_text=question_text,
            frames_data_urls=frames_data_urls,
        )

        # Store frame_paths in tool_state for video_cut_tool
        ctx.tool_state['frame_paths'] = task_input.frame_paths
        ctx.tool_state['video_fps'] = actual_fps
        ctx.tool_state['video_duration'] = video_duration

    elif task_input.video_path:
        # Mode 1: video_path — convert video to base64 data URL
        video_duration, native_fps = _get_video_info(task_input.video_path)
        logger.debug(
            f'VideoCut task: video info — '
            f'duration={video_duration:.2f}s, native_fps={native_fps:.2f}, '
            f'path="{task_input.video_path}"'
        )

        video_data_url = video_url_to_base64(
            video_source=task_input.video_path,
            fps=task_input.video_fps,
            max_base64_size_mb=task_input.max_base64_size_mb,
        )
        logger.debug(f'VideoCut task: video converted to base64 data URL (length={len(video_data_url)} chars)')

        prompt = _build_first_turn_prompt_video(
            question_text=question_text,
            video_data_url=video_data_url,
        )

        # Store video_path in tool_state for video_cut_tool (ffmpeg-based cropping)
        ctx.tool_state['video_path'] = task_input.video_path
        ctx.tool_state['video_fps'] = native_fps
        ctx.tool_state['video_duration'] = video_duration

    else:
        raise ValueError('VideoCut task requires either video_path or frame_paths to be provided.')

    # ── Run the agent ─────────────────────────────────────────────────────────
    agent_output = await task_input.run_agent(
        prompt=prompt,
        max_iterations=task_input.max_iterations,
    )

    # ── Extract response from trajectory ──────────────────────────────────────
    trajectory = []
    if isinstance(agent_output, WithMainTrajectory):
        trajectory = agent_output.main_trajectory
    response = _extract_last_response(trajectory)

    logger.debug(f'VideoCut task: received response of {len(response)} chars')

    # ── Compute reward ────────────────────────────────────────────────────────
    if task_input.reward_style == 'rule':
        result = compute_rule_score(
            solution_str=response,
            ground_truth=task_input.ground_truth,
            trajectory=trajectory,
        )
    else:
        # LLM-as-a-judge (falls back to 'main' if 'judge' client not configured)
        try:
            judge_client = ctx.get_openai_client('judge')
        except KeyError:
            logger.warning('VideoCut task: no "judge" client configured, using "main" client for scoring')
            judge_client = ctx.openai_client

        result = await compute_score(
            solution_str=response,
            ground_truth=task_input.ground_truth,
            question=task_input.question,
            judge_client=judge_client,
            trajectory=trajectory,
        )

    logger.info(
        f'VideoCut task: score={result.score:.3f}, correct={result.correct}, ground_truth="{task_input.ground_truth}"'
    )

    return VideoCutTaskOutput(
        score=result.score,
        response=response,
        correct=result.correct,
    )
