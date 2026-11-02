"""Conversation building for the S2T yuanbao-glue search agent.

Differences vs. the text-only ``search_s2t_rl`` conversation helper:

* Preserves multi-modal *content lists* instead of forcing every message
  ``content`` to a plain string, so speech user turns can carry audio.
* Expands ``<audio>`` placeholders into ``{"type": "audio", ...}`` content
  parts backed by the dataset's audio *patch* references, in order of
  appearance across the whole conversation. The patch dict is JSON-safe and is
  loaded lazily by ``process_multi_modal_info`` (``load_audio_from_patch``) with
  the processor's sampling rate — exactly the path ``RLHFDatasetAudio`` uses —
  so both the vLLM rollout and the training-side token expansion see the audio.

Text rows (``audios`` is ``None``/empty and no ``<audio>`` placeholder) build a
plain-text conversation, enabling mixed speech/text training with one code path.
"""

from __future__ import annotations

import re

from typing import Any


POLICY_MESSAGE_ROLES = {'system', 'user', 'assistant'}
ROLE_ALIASES = {
    'human': 'user',
    'gpt': 'assistant',
    'bot': 'assistant',
    'ai': 'assistant',
}

AUDIO_PLACEHOLDER = '<audio>'
_AUDIO_SEGMENT_REGEX = re.compile(r'(<audio>)')


def _normalise_role(role: Any) -> str | None:
    if role is None:
        return None
    value = str(role).strip().lower()
    return ROLE_ALIASES.get(value, value)


def _coerce_patch(audio_item: Any) -> dict[str, Any]:
    """Coerce a dataset audio entry into an ``{"type": "audio", ...}`` content part.

    ``audio_item`` is expected to be a patch dict such as::

        {"patch": "/abs/path.wav", "audio_length": 140160, "sample_rate": 24000,
         "start_num": 0, "size": 0, ...}

    Only ``patch`` is strictly required — ``extract_audio_info_from_messages``
    reads it via ``load_audio_from_patch`` at rollout time. We forward all keys
    unchanged (harmless for ``apply_chat_template``, which only keys off
    ``type``) so packed-binary patches (``start_num``/``size``) keep working.
    """
    if isinstance(audio_item, dict) and ('patch' in audio_item or 'audio' in audio_item or 'audio_url' in audio_item):
        return {'type': 'audio', **audio_item}
    # A dict without any audio key falls through to the error below.
    if isinstance(audio_item, str):
        return {'type': 'audio', 'patch': audio_item}
    raise TypeError(
        f'Unsupported audio item for <audio> placeholder: {type(audio_item)!r}. '
        'Expected a patch dict with a "patch" key or a wav path string.'
    )


def _split_content_with_audio(content: str, audio_iter: list[dict[str, Any]], cursor: list[int]) -> Any:
    """Split a string ``content`` on ``<audio>`` and embed audio parts.

    Returns the original string when there is no ``<audio>`` placeholder, or a
    list of ``{"type": "text"|"audio", ...}`` content parts otherwise. ``cursor``
    is a single-element list used as a mutable index into ``audio_iter``.
    """
    if AUDIO_PLACEHOLDER not in content:
        return content

    parts: list[dict[str, Any]] = []
    for segment in _AUDIO_SEGMENT_REGEX.split(content):
        if segment == AUDIO_PLACEHOLDER:
            if cursor[0] >= len(audio_iter):
                raise ValueError(f'<audio> placeholders exceed available audio payloads ({len(audio_iter)} available).')
            parts.append(_coerce_patch(audio_iter[cursor[0]]))
            cursor[0] += 1
        elif segment:
            parts.append({'type': 'text', 'text': segment})
    return parts


def build_agent_messages(
    raw_messages: list[dict[str, Any]] | None,
    audios: list[dict[str, Any]] | None = None,
    *,
    default_system_prompt: str | None = None,
) -> list[dict[str, Any]]:
    """Build the agent's starting conversation from dataset messages.

    Args:
        raw_messages: dataset ``prompt``/``messages`` (role + string content).
        audios: ordered list of audio patch dicts, one per ``<audio>`` placeholder
            (``None`` for text rows).
        default_system_prompt: fallback system prompt used only when the dataset
            provides no system message.

    Returns:
        OpenAI-style messages ready for ``create_completion``. Audio user turns
        carry ``{"type": "audio", ...}`` content parts.
    """
    audio_items = list(audios or [])
    cursor = [0]

    normalised: list[dict[str, Any]] = []
    has_system = False
    for raw in raw_messages or []:
        if not isinstance(raw, dict):
            continue
        role = _normalise_role(raw.get('role') or raw.get('from'))
        if role not in POLICY_MESSAGE_ROLES:
            continue
        content = raw.get('content')
        if content is None:
            content = raw.get('text')
        if content is None:
            continue

        if isinstance(content, str):
            content = _split_content_with_audio(content, audio_items, cursor)
        # Non-string content (already a content-part list) is passed through as-is.

        if role == 'system':
            has_system = True
        normalised.append({'role': role, 'content': content})

    if cursor[0] != len(audio_items):
        raise ValueError(f'<audio> placeholders ({cursor[0]}) do not match audio payloads ({len(audio_items)}).')

    if not has_system and default_system_prompt:
        normalised.insert(0, {'role': 'system', 'content': default_system_prompt})

    return normalised
