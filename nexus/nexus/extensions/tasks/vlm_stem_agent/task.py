"""Nexus task for multimodal STEM Agent inference.

Nexus owns multimodal prompt construction, sandbox image staging, the Agent
loop, and tool execution. Scoring is opt-in: with the default
``evaluator_type='none'`` the task returns the raw prediction and leaves
correctness to an external judge, which is what the HLE-VL and MathVision
pipelines rely on. Setting any other ``evaluator_type`` runs the shared
evaluators in-process, mirroring ``search_stateful``.
"""

import asyncio
import base64
import binascii
import json
import logging
import math
import time

from collections import Counter
from io import BytesIO
from pathlib import Path
from typing import Any, Literal

import aiohttp

from PIL import Image, UnidentifiedImageError
from pydantic import Field, model_validator

from nexus.core import (
    TaskRunInput,
    TaskRunOutput,
    WithAgentRunner,
    WithScore,
    require_context,
    task_registry,
)
from nexus.extensions.agents.base.evaluation import evaluate_output
from nexus.extensions.agents.base.utils import extract_text_from_content
from nexus.extensions.agents.main.agent import MainAgentOutput
from nexus.extensions.tasks.search_stateful.evaluator import (
    GeneralEvaluator,
    HLEEvaluator,
    RewardSystemEvaluator,
    SearchStatefulEvaluator,
)
from nexus.runtime.executors import FileItem
from nexus.trace import get_current_span_id, update_span
from nexus.utils.multimodal import place_images_in_question


logger = logging.getLogger(__name__)

# Named explicitly rather than as the Evaluator protocol so the judge-details
# accessor, which the protocol does not declare, stays statically visible.
_JudgeEvaluator = GeneralEvaluator | HLEEvaluator | RewardSystemEvaluator | SearchStatefulEvaluator


DEFAULT_IMAGE_DOWNLOAD_TIMEOUT = 60.0
DEFAULT_MAX_IMAGE_BYTES = 20 * 1024 * 1024
MAX_CONFIGURABLE_IMAGE_BYTES = 100 * 1024 * 1024
_IMAGE_MIME_BY_FORMAT = {
    'BMP': 'image/bmp',
    'GIF': 'image/gif',
    'JPEG': 'image/jpeg',
    'PNG': 'image/png',
    'TIFF': 'image/tiff',
    'WEBP': 'image/webp',
}
_IMAGE_EXTENSION_BY_MIME = {
    'image/bmp': '.bmp',
    'image/gif': '.gif',
    'image/jpeg': '.jpg',
    'image/png': '.png',
    'image/tiff': '.tiff',
    'image/webp': '.webp',
}
_SANDBOX_IMAGE_DIR = '/tmp/jupyter_work'
_INFRA_FAILURE_STATUSES = {
    'inference_error',
    'inference_timeout',
    'observation_timeout',
    'runtime_error',
    'tool_execution_error',
}
_INFRA_FAILURE_SCORE = -100.0


class VLMStemAgentTaskInput(TaskRunInput, WithAgentRunner):
    """One VLM-STEM question and its multimodal context."""

    # Either side may be empty on its own: part of the VLM corpus asks the
    # question inside the picture and carries no text, and the same corpus mixes
    # in plain text rows that ship no picture. ``_check_prompt_is_not_empty``
    # rejects only the combination that leaves nothing to answer.
    question: str = ''
    # Each entry is a Base64 data URI, an HTTPS URL, or a local file path that
    # must exist on the machine running the task.
    images: list[str] = []
    answer: str
    ground_truth: str | None = None
    history: list[dict[str, Any]] | None = None
    image_download_timeout: float = Field(
        default=DEFAULT_IMAGE_DOWNLOAD_TIMEOUT,
        gt=0,
        le=600,
    )
    max_image_bytes: int = Field(
        default=DEFAULT_MAX_IMAGE_BYTES,
        gt=0,
        le=MAX_CONFIGURABLE_IMAGE_BYTES,
    )
    # Total-pixel bounds for the input image, matching the platform's
    # client-side resize (runtime/common/image_resize): scale proportionally so
    # width*height lands in [min_pixels, max_pixels]. Below the lower bound the
    # image is enlarged; max_pixels alone only caps. Both unset means no resize.
    min_pixels: int | None = Field(default=None, gt=0)
    max_pixels: int | None = Field(default=None, gt=0)

    # Scoring is opt-in so the HLE-VL and MathVision pipelines keep handing the
    # raw prediction to their own judge. Anything other than 'none' scores in
    # process with the same evaluators search_stateful uses.
    evaluator_type: Literal[
        'hle',
        'general',
        'none',
        'olympiad',
        'research',
        'oss',
        'reward_system',
    ] = 'none'
    evaluator_none_score: float = 0.0

    # reward_system evaluator config
    reward_system_user_token: str | None = None
    reward_system_exp_id: str | None = None
    reward_system_router_id: str | None = None
    reward_system_extra: dict[str, Any] | None = None
    reward_system_public_test: dict[str, Any] | None = None
    reward_system_ci_test: bool = False
    # The staged images go to the reward system by default so a multimodal router
    # judges against the same picture the policy saw. Set True, or list the
    # text-only router_ids, to keep those routers on a text-only payload.
    reward_system_strip_multimodal_content: bool | list[str] | None = None

    @model_validator(mode='after')
    def _check_prompt_is_not_empty(self) -> 'VLMStemAgentTaskInput':
        # With neither text nor picture the user turn is just the sandbox
        # preamble, and the policy is asked to answer nothing at all.
        if not self.question and not self.images:
            raise ValueError('question and images must not both be empty')
        return self

    @model_validator(mode='after')
    def _check_pixel_bounds(self) -> 'VLMStemAgentTaskInput':
        if self.min_pixels is not None and self.max_pixels is not None and self.min_pixels > self.max_pixels:
            raise ValueError(f'min_pixels must not exceed max_pixels ({self.min_pixels} > {self.max_pixels})')
        return self


class VLMStemAgentTaskOutput(TaskRunOutput, WithScore):
    """Agent result, scored in process or left to an external evaluator."""

    prediction: str
    question: str
    answer: str
    exit_status: str
    tool_call_counts: dict[str, int]
    judge_details: dict[str, Any] | None = None


def _target_size_for_pixel_range(
    width: int,
    height: int,
    min_pixels: int | None,
    max_pixels: int | None,
) -> tuple[int, int] | None:
    """Return the size that puts width*height inside the bounds, or None.

    Rounds each side with the builtin ``round`` to match the platform's
    client-side resize byte for byte, so a with-tools run and a no-tools run see
    the same preprocessed image. Rounding can leave the pixel count a fraction
    of a percent outside the requested bound (one pixel per side), which does
    not change the visual token count for patch-based vision encoders.
    """
    pixels = width * height
    if max_pixels is not None and pixels > max_pixels:
        scale = math.sqrt(max_pixels / pixels)
    elif min_pixels is not None and pixels < min_pixels:
        scale = math.sqrt(min_pixels / pixels)
    else:
        return None
    return max(1, round(width * scale)), max(1, round(height * scale))


def _resize_to_pixel_range(
    data: bytes,
    min_pixels: int | None,
    max_pixels: int | None,
) -> bytes:
    """Scale image bytes so the pixel count lands inside the configured range.

    Re-encodes with the source format so the caller's MIME detection and byte
    limit still apply. JPEG cannot store alpha or palette modes, so those are
    flattened to RGB before saving.
    """
    if min_pixels is None and max_pixels is None:
        return data

    try:
        with Image.open(BytesIO(data)) as image:
            image_format = image.format
            target = _target_size_for_pixel_range(
                image.width,
                image.height,
                min_pixels,
                max_pixels,
            )
            if target is None:
                return data
            resized = image.resize(target, Image.Resampling.LANCZOS)
            if image_format == 'JPEG' and resized.mode not in ('RGB', 'L'):
                resized = resized.convert('RGB')
            buffer = BytesIO()
            resized.save(buffer, format=image_format)
    except (
        Image.DecompressionBombError,
        OSError,
        SyntaxError,
        UnidentifiedImageError,
        ValueError,
    ) as exc:
        raise ValueError(f'Unable to resize VLM-STEM image: {exc}') from exc
    return buffer.getvalue()


def _image_bytes_to_data_uri(data: bytes, max_image_bytes: int) -> str:
    if not data:
        raise ValueError('VLM-STEM image is empty')
    if len(data) > max_image_bytes:
        raise ValueError(f'VLM-STEM image exceeds configured byte limit ({len(data)} > {max_image_bytes})')

    try:
        with Image.open(BytesIO(data)) as image:
            image_format = image.format
            image.verify()
    except (
        Image.DecompressionBombError,
        OSError,
        SyntaxError,
        UnidentifiedImageError,
    ) as exc:
        raise ValueError('VLM-STEM image payload is not a valid image') from exc

    mime_type = _IMAGE_MIME_BY_FORMAT.get(str(image_format).upper())
    if mime_type is None:
        raise ValueError(f'Unsupported VLM-STEM image format: {image_format or "unknown"}')
    encoded = base64.b64encode(data).decode('ascii')
    return f'data:{mime_type};base64,{encoded}'


def _decode_data_uri(source: str, max_image_bytes: int) -> bytes:
    try:
        header, payload = source.split(',', 1)
    except ValueError as exc:
        raise ValueError('Malformed VLM-STEM image data URI') from exc
    if not header.startswith('data:image/') or ';base64' not in header.lower():
        raise ValueError('VLM-STEM image data URI must use base64 encoding')
    try:
        data = base64.b64decode(payload, validate=True)
    except (binascii.Error, ValueError) as exc:
        raise ValueError('Malformed base64 in VLM-STEM image data URI') from exc
    if len(data) > max_image_bytes:
        raise ValueError(f'VLM-STEM image exceeds configured byte limit ({len(data)} > {max_image_bytes})')
    return data


def _read_local_image(source: str, max_image_bytes: int) -> bytes:
    """Read image bytes from a local path, failing loudly when it is missing.

    Datasets that carry filesystem paths instead of URLs are staged from the
    caller's filesystem, so a missing path is a dataset defect rather than a
    recoverable condition: never substitute a placeholder image.
    """
    path = Path(source)
    if not path.exists():
        raise ValueError(f'VLM-STEM local image file does not exist: {source}')
    if not path.is_file():
        raise ValueError(f'VLM-STEM local image path is not a file: {source}')
    size = path.stat().st_size
    if size > max_image_bytes:
        raise ValueError(f'VLM-STEM image exceeds configured byte limit ({size} > {max_image_bytes})')
    try:
        return path.read_bytes()
    except OSError as exc:
        raise ValueError(f'Unable to read VLM-STEM local image: {source}') from exc


async def _download_http_image(
    source: str,
    session: aiohttp.ClientSession,
    max_image_bytes: int,
) -> bytes:
    if not source.startswith('https://'):
        raise ValueError('VLM-STEM remote image source must use HTTPS')
    try:
        async with session.get(source, allow_redirects=True) as response:
            if not str(response.url).startswith('https://'):
                raise ValueError('VLM-STEM image redirect must remain on HTTPS')
            if response.status < 200 or response.status >= 300:
                raise ValueError(f'VLM-STEM image download returned HTTP {response.status}')
            content_length = response.content_length
            if content_length is not None and content_length > max_image_bytes:
                raise ValueError(f'VLM-STEM image exceeds configured byte limit ({content_length} > {max_image_bytes})')

            chunks = []
            total_bytes = 0
            async for chunk in response.content.iter_chunked(64 * 1024):
                total_bytes += len(chunk)
                if total_bytes > max_image_bytes:
                    raise ValueError(
                        f'VLM-STEM image exceeds configured byte limit ({total_bytes} > {max_image_bytes})'
                    )
                chunks.append(chunk)
            return b''.join(chunks)
    except ValueError:
        raise
    except (aiohttp.ClientError, asyncio.TimeoutError) as exc:
        raise ValueError('Unable to download VLM-STEM image') from exc


async def image_source_to_data_uri(
    source: str,
    session: aiohttp.ClientSession,
    max_image_bytes: int,
    min_pixels: int | None = None,
    max_pixels: int | None = None,
) -> str:
    """Normalize a data URI, HTTPS URL, or local file path to validated Base64."""
    source = source.strip()
    if not source:
        raise ValueError('VLM-STEM image source must not be empty')
    if source.startswith('data:'):
        data = _decode_data_uri(source, max_image_bytes)
    elif source.startswith('https://'):
        data = await _download_http_image(
            source,
            session,
            max_image_bytes,
        )
    elif '://' in source:
        # Any other scheme, notably plain HTTP, must not silently fall through
        # to the filesystem branch and report a confusing missing-file error.
        raise ValueError('VLM-STEM image source must be a Base64 data URI, HTTPS URL, or local file path')
    else:
        # Reads run off the event loop so concurrent staging of a multi-image
        # sample is not serialized behind network-filesystem latency.
        data = await asyncio.to_thread(_read_local_image, source, max_image_bytes)
    data = _resize_to_pixel_range(data, min_pixels, max_pixels)
    return _image_bytes_to_data_uri(data, max_image_bytes)


async def normalize_images_to_data_uris(
    images: list[str],
    *,
    image_download_timeout: float,
    max_image_bytes: int,
    min_pixels: int | None = None,
    max_pixels: int | None = None,
) -> list[str]:
    timeout = aiohttp.ClientTimeout(total=image_download_timeout)
    async with aiohttp.ClientSession(timeout=timeout) as session:
        return list(
            await asyncio.gather(
                *(
                    image_source_to_data_uri(
                        image,
                        session,
                        max_image_bytes,
                        min_pixels,
                        max_pixels,
                    )
                    for image in images
                )
            )
        )


def _image_data_uri_payload_and_extension(
    image: str,
) -> tuple[str, str]:
    """Return the existing Base64 payload and a safe file extension."""
    try:
        header, payload = image.split(',', 1)
    except ValueError as exc:
        raise ValueError('Malformed VLM-STEM image data URI') from exc
    if not header.startswith('data:image/') or ';base64' not in header.lower():
        raise ValueError('VLM-STEM image data URI must use base64 encoding')
    mime_type = header[5:].split(';', 1)[0].lower()
    extension = _IMAGE_EXTENSION_BY_MIME.get(mime_type)
    if extension is None:
        raise ValueError(f'Unsupported VLM-STEM image MIME type: {mime_type}')
    return payload, extension


async def upload_images_to_runtime(
    images: list[str],
) -> list[str]:
    """Upload the exact model-facing image bytes into the main sandbox.

    The runtime is acquired even for a text-only row. Staged files are not what
    the sandbox is for here -- the agent runs ``code_interpreter`` to do the
    arithmetic either way -- so provisioning stays on the task's critical path
    instead of landing mid-rollout, matching ``hle`` and ``search_stateful``.
    """
    context = require_context()
    runtime = await context.get_runtime('main')
    if not images:
        return []
    sandbox_paths: list[str] = []
    file_items: list[FileItem] = []
    for index, image in enumerate(images):
        payload, extension = _image_data_uri_payload_and_extension(image)
        sandbox_path = f'{_SANDBOX_IMAGE_DIR}/input_{index}{extension}'
        sandbox_paths.append(sandbox_path)
        file_items.append(
            FileItem(
                path=sandbox_path,
                content=payload,
                encoding='base64',
            )
        )

    upload_result = await runtime.upload_file(
        file_items,
        overwrite=True,
    )
    failures = [result for result in upload_result.results if not result.success]
    if failures:
        raise RuntimeError(
            'Failed to upload VLM-STEM images to sandbox: '
            + '; '.join(f'{result.path}: {result.message}' for result in failures)
        )
    return sandbox_paths


def build_multimodal_user_prompt(
    question: str,
    images: list[str],
    sandbox_paths: list[str],
) -> list[dict[str, Any]]:
    """Build an OpenAI-compatible multimodal message for one VLM-STEM row.

    The sandbox path listing always leads, so the agent can locate the files for
    ``code_interpreter`` regardless of how the question itself is laid out.
    """
    if len(images) != len(sandbox_paths):
        raise ValueError('VLM-STEM images and sandbox paths must have equal length')
    for image in images:
        if not image.startswith('data:image/') or ';base64,' not in image:
            raise ValueError('VLM-STEM images must be normalized Base64 data URIs')

    image_parts: list[dict[str, Any]] = [{'type': 'image_url', 'image_url': {'url': image}} for image in images]
    laid_out = place_images_in_question(question, image_parts)
    if not sandbox_paths:
        # A text-only row stages no files, and the bare heading would promise
        # the agent a listing that never follows.
        return laid_out

    path_message = 'Image file paths available inside the code_interpreter sandbox:\n' + '\n'.join(
        f'- Image {index}: {path}' for index, path in enumerate(sandbox_paths, start=1)
    )
    return [
        {'type': 'text', 'text': path_message},
        *laid_out,
    ]


def _extract_prediction(agent_output: MainAgentOutput) -> str:
    final_answer = agent_output.final_answer
    if final_answer is not None and str(final_answer).strip():
        return str(final_answer).strip()

    for message in reversed(agent_output.main_trajectory):
        if message.get('role') != 'assistant':
            continue
        content = message.get('content') or message.get('reasoning_content')
        if content is None:
            continue
        prediction = extract_text_from_content(content).strip()
        if prediction:
            return prediction
    return ''


def _normalize_exit_status(exit_status: Any) -> str:
    if hasattr(exit_status, 'value'):
        exit_status = exit_status.value
    return str(exit_status or '')


def _response_prediction(
    agent_output: MainAgentOutput,
    exit_status: str,
) -> str:
    """Return the model answer, or a structured error as the sample response.

    Infrastructure failures carry a negative task score, so hy_unify_eval still
    marks the trial as infer_failed and never sends this text to the judge. The
    serialized error nevertheless survives the existing Nexus response adapter
    and appears in the per-sample ``responses`` field, matching the no-tools
    failure presentation.
    """
    prediction = _extract_prediction(agent_output)
    if exit_status not in _INFRA_FAILURE_STATUSES:
        return prediction

    payload: dict[str, Any] = {
        'exit_status': exit_status,
        'error': {
            'stage': 'agent',
            'type': exit_status,
            'code': exit_status,
            'message': f'Agent stopped with infrastructure status {exit_status}.',
        },
    }
    if prediction:
        payload['partial_prediction'] = prediction
    return json.dumps(payload, ensure_ascii=False)


def _tool_call_counts(
    trajectory: list[dict[str, Any]],
) -> dict[str, int]:
    counts: Counter[str] = Counter()
    for message in trajectory:
        if not isinstance(message, dict):
            continue
        for tool_call in message.get('tool_calls') or []:
            if not isinstance(tool_call, dict):
                continue
            function = tool_call.get('function') or {}
            name = function.get('name') if isinstance(function, dict) else None
            if name:
                counts[str(name)] += 1
    return dict(counts)


def _build_evaluator(
    task_input: VLMStemAgentTaskInput,
    images: list[str],
) -> _JudgeEvaluator:
    """Pick the evaluator named by ``evaluator_type``.

    The evaluators are shared with ``search_stateful``. Only the reward system
    takes the images; the prompt-based judges are text-only, so the staged
    images never reach them.
    """
    if task_input.evaluator_type == 'reward_system':
        if not task_input.reward_system_router_id:
            raise ValueError('reward_system_router_id is required for reward_system evaluator')
        return RewardSystemEvaluator(
            user_token=task_input.reward_system_user_token,
            exp_id=task_input.reward_system_exp_id or '',
            router_id=task_input.reward_system_router_id,
            extra=task_input.reward_system_extra,
            public_test=task_input.reward_system_public_test,
            ci_test=task_input.reward_system_ci_test,
            images=images,
            strip_multimodal_content=task_input.reward_system_strip_multimodal_content,
        )
    if task_input.evaluator_type == 'general':
        return GeneralEvaluator()
    if task_input.evaluator_type == 'hle':
        return HLEEvaluator()
    return SearchStatefulEvaluator(evaluator_type=task_input.evaluator_type)


async def _score_prediction(
    task_input: VLMStemAgentTaskInput,
    prediction: str,
    images: list[str],
) -> tuple[float, dict[str, Any] | None]:
    """Score a completed run, or return the configured no-op score.

    ``images`` are the normalized data URIs handed to the policy, not the raw
    task input, so the judge sees the same resized picture the model saw.
    """
    if task_input.evaluator_type == 'none':
        return task_input.evaluator_none_score, None

    evaluator = _build_evaluator(task_input, images)
    # Generators put the judge-facing reference in ground_truth and keep answer
    # as the dataset's display answer; fall back when only answer is populated.
    reference = task_input.ground_truth if task_input.ground_truth is not None else task_input.answer
    start_time = time.time()
    score, _, is_valid_reward = await evaluate_output(
        prediction=prediction,
        reference=reference,
        question=task_input.question,
        evaluator=evaluator,
    )
    span_id = get_current_span_id()
    if span_id:
        update_span(span_id, attributes={'hytuner.reward_time': time.time() - start_time})
        if not is_valid_reward:
            update_span(span_id, attributes={'hytuner.termination_reason': 'REWARD_ERROR'})
    if not is_valid_reward:
        logger.warning(
            'vlm_stem_agent judge did not return a valid reward (evaluator_type=%s)',
            task_input.evaluator_type,
        )
    return score, evaluator.get_last_judge_result()


@task_registry.register('vlm_stem_agent')
async def vlm_stem_agent_task(
    task_input: VLMStemAgentTaskInput,
) -> VLMStemAgentTaskOutput:
    """Run a multimodal STEM Agent and score it only when asked to."""
    images = await normalize_images_to_data_uris(
        task_input.images,
        image_download_timeout=task_input.image_download_timeout,
        max_image_bytes=task_input.max_image_bytes,
        min_pixels=task_input.min_pixels,
        max_pixels=task_input.max_pixels,
    )
    sandbox_paths = await upload_images_to_runtime(images)
    user_prompt = build_multimodal_user_prompt(
        task_input.question,
        images,
        sandbox_paths,
    )
    agent_output = await task_input.run_agent(
        user_prompt=user_prompt,
        history=task_input.history,
    )
    if not isinstance(agent_output, MainAgentOutput):
        raise RuntimeError(f'vlm_stem_agent task requires main_agent output, got {type(agent_output)}')

    exit_status = _normalize_exit_status(agent_output.exit_status)
    prediction = _response_prediction(agent_output, exit_status)
    if exit_status in _INFRA_FAILURE_STATUSES:
        # The negative sentinel is what marks the trial infer_failed downstream,
        # so it outranks any judge and the judge is not called at all: the
        # prediction here is a serialized error, not a model answer.
        score: float = _INFRA_FAILURE_SCORE
        judge_details: dict[str, Any] | None = None
    else:
        score, judge_details = await _score_prediction(task_input, prediction, images)

    return VLMStemAgentTaskOutput(
        score=score,
        prediction=prediction,
        question=task_input.question,
        answer=task_input.answer,
        exit_status=exit_status,
        tool_call_counts=_tool_call_counts(agent_output.main_trajectory),
        judge_details=judge_details,
    )
