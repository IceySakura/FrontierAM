"""Audio Re-Listen tool.

The model inspects a long audio clip embedded in the user message, then
optionally invokes ``audio_relisten_tool`` to "re-listen" to a specific
``[start_time, end_time]`` region. Unlike Echo (which uses a string-protocol
hack: stop-token + audio splice in the rollout loop), this tool participates
in the standard OpenAI function-calling protocol: arguments come through as
``tool_calls``, and the response is a regular ``tool`` message whose content
contains an ``input_audio`` part with the freshly sliced audio.

State contract (read from ``ToolContext.tool_state``):
    - ``audio_samples``  : mono ``np.ndarray[float32]`` in ``[-1, 1]``
    - ``audio_sample_rate``: int
    - ``audio_format``   : str — for the returned ``input_audio.format`` field (default ``'wav'``)

The tool also appends a structured record under ``__audio_relisten_records__``
in ``context_updates`` so the reward function can credit tool usage.
"""

from __future__ import annotations

import base64
import logging

from typing import TYPE_CHECKING, Any

from pydantic import BaseModel, Field

from nexus.core import ToolContext, ToolResponse, ToolType, tool_registry


if TYPE_CHECKING:
    import numpy as np


logger = logging.getLogger(__name__)


AUDIO_RELISTEN_RECORDS_KEY = '__audio_relisten_records__'
AUDIO_STATE_KEY = 'audio_samples'
SAMPLE_RATE_STATE_KEY = 'audio_sample_rate'
AUDIO_FORMAT_STATE_KEY = 'audio_format'

# Guard rails — keep the tool well-behaved against pathological calls.
_MIN_SLICE_SECONDS = 0.2
_MAX_SLICE_SECONDS = 200.0  # ≈ 3.2 M samples @ 16 kHz, matches Echo's safety cap.


class AudioRelistenInput(BaseModel):
    """Input for ``audio_relisten_tool``.

    The model expresses *which* segment of the original audio it wants to
    re-listen to, in seconds, plus an optional natural-language label
    describing what it expects to hear (helpful for training / interpretability,
    purely advisory at runtime).
    """

    time_range: list[float] = Field(
        description=(
            'The temporal range of the original audio to re-listen to, '
            'as [start_time, end_time] in seconds. Must satisfy '
            '0 <= start_time < end_time, with at least 0.2s of duration.'
        ),
        min_length=2,
        max_length=2,
    )
    label: str | None = Field(
        default=None,
        description='Optional short description of what the model expects to hear in this segment.',
    )


def _build_tool_context_updates(context: ToolContext, record: dict[str, Any]) -> dict[str, Any]:
    records = list(context.tool_state.get(AUDIO_RELISTEN_RECORDS_KEY, []))
    records.append(record)
    return {AUDIO_RELISTEN_RECORDS_KEY: records}


def _error(
    context: ToolContext,
    message: str,
    *,
    error: str,
    time_range: list[float],
    label: str | None,
    extra_meta: dict[str, Any] | None = None,
) -> ToolResponse:
    """Helper to build a structured error ``ToolResponse``."""
    meta: dict[str, Any] = {'error': error, 'time_range': time_range}
    if extra_meta:
        meta.update(extra_meta)
    return ToolResponse(
        output=[{'type': 'text', 'text': message}],
        meta=meta,
        context_updates=_build_tool_context_updates(
            context,
            {
                'tool_name': 'audio_relisten_tool',
                'success': False,
                'error': error,
                'time_range': time_range,
                'label': label,
                **(extra_meta or {}),
            },
        ),
    )


@tool_registry.register(
    'audio_relisten/audio_relisten_tool',
    meta={
        'name': 'audio_relisten_tool',
        'description': (
            'Re-listen to a specific time range of the original audio clip. '
            'Use this when the initial listen is insufficient to answer the question — '
            'for example to focus on a brief sound event, verify timing, or hear a '
            'segment at a different position. Returns the sliced audio so the '
            'assistant can incorporate it into the next turn of reasoning.'
        ),
        'tool_type': ToolType.LOCAL,
    },
)
async def audio_relisten_tool(params: AudioRelistenInput, context: ToolContext) -> ToolResponse:
    """Slice and return the requested audio segment.

    The slice is encoded as a base64 WAV and surfaced as an OpenAI
    ``input_audio`` content part, so any audio-capable LLM (Qwen-Omni,
    GPT-4o-audio, etc.) can ingest it directly on the next turn.
    """
    import numpy as np

    samples_obj: Any = context.tool_state.get(AUDIO_STATE_KEY)
    sr_obj: Any = context.tool_state.get(SAMPLE_RATE_STATE_KEY)
    audio_format = context.tool_state.get(AUDIO_FORMAT_STATE_KEY) or 'wav'

    if samples_obj is None or sr_obj is None:
        return _error(
            context,
            'Error: no audio is available in the tool state. The task must populate '
            f"tool_state['{AUDIO_STATE_KEY}'] and tool_state['{SAMPLE_RATE_STATE_KEY}'] "
            'before invoking the agent.',
            error='no_audio',
            time_range=params.time_range,
            label=params.label,
        )

    if not isinstance(samples_obj, np.ndarray):
        return _error(
            context,
            f"Error: tool_state['{AUDIO_STATE_KEY}'] must be a numpy ndarray; got {type(samples_obj).__name__}.",
            error='invalid_audio_type',
            time_range=params.time_range,
            label=params.label,
        )

    samples: np.ndarray = samples_obj
    try:
        sample_rate = int(sr_obj)
    except (TypeError, ValueError):
        return _error(
            context,
            f"Error: tool_state['{SAMPLE_RATE_STATE_KEY}'] must be an int; got {sr_obj!r}.",
            error='invalid_sample_rate',
            time_range=params.time_range,
            label=params.label,
        )

    if sample_rate <= 0 or samples.size == 0:
        return _error(
            context,
            'Error: stored audio is empty or has an invalid sample rate.',
            error='invalid_audio',
            time_range=params.time_range,
            label=params.label,
            extra_meta={'sample_rate': sample_rate, 'n_samples': int(samples.size)},
        )

    duration = samples.size / float(sample_rate)
    start, end = params.time_range
    if not (isinstance(start, (int, float)) and isinstance(end, (int, float))):
        return _error(
            context,
            f'Error: time_range entries must be numeric; got {params.time_range!r}.',
            error='invalid_time_range_type',
            time_range=params.time_range,
            label=params.label,
        )

    start_c = max(0.0, float(start))
    end_c = min(float(duration), float(end))

    if end_c - start_c < _MIN_SLICE_SECONDS:
        return _error(
            context,
            (
                f'Error: requested segment [{start:.2f}s, {end:.2f}s] is shorter than '
                f'{_MIN_SLICE_SECONDS:.1f}s after clamping to the audio duration '
                f'({duration:.2f}s). Please request a wider window.'
            ),
            error='slice_too_short',
            time_range=params.time_range,
            label=params.label,
            extra_meta={
                'duration': duration,
                'effective_time_range': [start_c, end_c],
            },
        )

    if end_c - start_c > _MAX_SLICE_SECONDS:
        # Cap the slice instead of failing — the model can always re-issue.
        end_c = start_c + _MAX_SLICE_SECONDS
        logger.warning(
            'audio_relisten_tool: slice capped at %.1fs (request was %.2fs–%.2fs)',
            _MAX_SLICE_SECONDS,
            start,
            end,
        )

    i0 = int(round(start_c * sample_rate))
    i1 = int(round(end_c * sample_rate))
    segment = samples[i0:i1]

    # Encode to WAV bytes (lazy import keeps the tool importable without numpy installed).
    from nexus.extensions.projects.audio_relisten.audio_io import encode_wav_bytes

    wav_bytes = encode_wav_bytes(segment, sample_rate)
    b64 = base64.b64encode(wav_bytes).decode('ascii')

    label_info = f' (expected: "{params.label}")' if params.label else ''
    desc = (
        f'Re-listened to audio segment [{start_c:.2f}s, {end_c:.2f}s]{label_info}. '
        f'Returning the sliced audio ({segment.size / sample_rate:.2f}s, {sample_rate} Hz).'
    )

    logger.debug(
        'audio_relisten_tool: time_range=[%.2f, %.2f] effective=[%.2f, %.2f] sr=%d nsamples=%d',
        start,
        end,
        start_c,
        end_c,
        sample_rate,
        int(segment.size),
    )

    return ToolResponse(
        output=[
            {'type': 'text', 'text': desc},
            {
                'type': 'input_audio',
                'input_audio': {
                    'data': b64,
                    'format': audio_format,
                },
            },
        ],
        meta={
            'time_range': [start_c, end_c],
            'requested_time_range': [start, end],
            'label': params.label,
            'sample_rate': sample_rate,
            'duration_seconds': segment.size / float(sample_rate),
        },
        context_updates=_build_tool_context_updates(
            context,
            {
                'tool_name': 'audio_relisten_tool',
                'success': True,
                'error': None,
                'time_range': [start_c, end_c],
                'requested_time_range': [start, end],
                'label': params.label,
                'duration_seconds': segment.size / float(sample_rate),
            },
        ),
    )
