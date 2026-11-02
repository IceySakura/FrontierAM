"""Canonical prompt contract for the unified reward-system task.

``task_config.prompt`` is a strict subset of the OpenAI Chat Completions
``messages`` parameter, so the same row runs unchanged through the HyTuner
token-level Gateway and a message-level Gateway backed by an OpenAI client:

- roles: ``system`` and ``user``; ``assistant`` / ``tool`` only as replayed history;
- plain text is a string; multimodal ``user`` content is a list of ``text`` and
  ``image_url`` parts with no two adjacent ``text`` parts;
- ``image_url.url`` is ``http(s)://``, ``data:image/...;base64,``, ``file://`` or an
  absolute path. Local references are read on the Gateway host.

The OpenAI SDK TypedDicts cannot enforce this: they accept unknown keys and
validate ``Iterable`` content lazily.
"""

from __future__ import annotations

import mimetypes
import os

from collections.abc import Mapping
from typing import Any


_SYSTEM_KEYS = frozenset({'role', 'content'})
_USER_KEYS = frozenset({'role', 'content'})
_ASSISTANT_KEYS = frozenset({'role', 'content', 'tool_calls'})
_TOOL_KEYS = frozenset({'role', 'content', 'tool_call_id'})
_TEXT_PART_KEYS = frozenset({'type', 'text'})
_IMAGE_PART_KEYS = frozenset({'type', 'image_url'})
_IMAGE_URL_KEYS = frozenset({'url', 'detail'})

# Fields the unified task reads from a stored ``task_config``. The runner drops
# unknown keys before the task sees them (hosts inject runtime keys there), so
# stray fields such as a top-level ``images`` list must be caught on the data side.
TASK_CONFIG_KEYS = frozenset({'prompt', 'tools', 'reward'})


def _check_keys(where: str, value: Mapping[str, Any], allowed: frozenset[str]) -> None:
    unknown = sorted(set(value) - allowed)
    if unknown:
        raise ValueError(f'{where}: unsupported keys {unknown}; allowed {sorted(allowed)}')


def _check_local_image_extension(where: str, url: str) -> None:
    path = url[len('file://') :] if url.startswith('file://') else url
    mime_type = mimetypes.guess_type(path)[0]
    if mime_type is None or not mime_type.startswith('image/'):
        raise ValueError(
            f'{where}: local image {path!r} has no recognised image file extension '
            f'(got {os.path.splitext(path)[1] or "none"!r}); name it with one such as .png or .jpg'
        )


def _check_image_url(where: str, url: Any) -> None:
    if not isinstance(url, str) or not url:
        raise ValueError(f'{where}: image_url.url must be a non-empty string')
    if url.startswith(('http://', 'https://')):
        return
    if url.startswith('file://'):
        _check_local_image_extension(where, url)
        return
    if url.startswith('data:'):
        if url.startswith('data:image/') and ';base64,' in url:
            return
        raise ValueError(f'{where}: data URLs must be base64 images')
    if os.path.isabs(url):
        _check_local_image_extension(where, url)
        return
    raise ValueError(f'{where}: image_url.url must be http(s)://, data:image/...;base64, file:// or an absolute path')


def _check_user_parts(where: str, parts: list[Any]) -> None:
    if not parts:
        raise ValueError(f'{where}: content list must not be empty')
    has_image = False
    previous_text = False
    for index, part in enumerate(parts):
        part_where = f'{where}.content[{index}]'
        if not isinstance(part, Mapping):
            raise ValueError(f'{part_where}: content parts must be objects')
        part_type = part.get('type')
        if part_type == 'text':
            _check_keys(part_where, part, _TEXT_PART_KEYS)
            if not isinstance(part.get('text'), str):
                raise ValueError(f'{part_where}: text must be a string')
            if previous_text:
                raise ValueError(f'{part_where}: adjacent text parts must be merged')
            previous_text = True
        elif part_type == 'image_url':
            _check_keys(part_where, part, _IMAGE_PART_KEYS)
            image_url = part.get('image_url')
            if not isinstance(image_url, Mapping):
                raise ValueError(f'{part_where}: image_url must be an object')
            _check_keys(f'{part_where}.image_url', image_url, _IMAGE_URL_KEYS)
            _check_image_url(part_where, image_url.get('url'))
            has_image = True
            previous_text = False
        else:
            raise ValueError(f"{part_where}: part type must be 'text' or 'image_url', got {part_type!r}")
    if not has_image:
        raise ValueError(f'{where}: text-only content must be a string, not a list of text parts')


def validate_prompt(messages: Any) -> None:
    """Raise ``ValueError`` unless ``messages`` follows the canonical prompt contract."""
    if not isinstance(messages, list) or not messages:
        raise ValueError('prompt must be a non-empty list of messages')
    for index, message in enumerate(messages):
        where = f'prompt[{index}]'
        if not isinstance(message, Mapping):
            raise ValueError(f'{where}: messages must be objects')
        role = message.get('role')
        content = message.get('content')
        if role == 'system':
            _check_keys(where, message, _SYSTEM_KEYS)
            if not isinstance(content, str):
                raise ValueError(f'{where}: system content must be a string')
        elif role == 'user':
            _check_keys(where, message, _USER_KEYS)
            if isinstance(content, list):
                _check_user_parts(where, content)
            elif not isinstance(content, str):
                raise ValueError(f'{where}: user content must be a string or a list of parts')
        elif role == 'assistant':
            _check_keys(where, message, _ASSISTANT_KEYS)
            if content is not None and not isinstance(content, str):
                raise ValueError(f'{where}: assistant content must be a string or null')
        elif role == 'tool':
            _check_keys(where, message, _TOOL_KEYS)
            if not isinstance(content, str) or not isinstance(message.get('tool_call_id'), str):
                raise ValueError(f'{where}: tool messages need string content and tool_call_id')
        else:
            raise ValueError(f'{where}: role must be system, user, assistant or tool, got {role!r}')
    if messages[-1].get('role') not in ('user', 'tool'):
        raise ValueError('prompt must end with a user or tool message')


def validate_task_config(task_config: Any) -> None:
    """Validate a stored ``task_config``; also rejects keys the task would silently drop."""
    if not isinstance(task_config, Mapping):
        raise ValueError('task_config must be an object')
    _check_keys('task_config', task_config, TASK_CONFIG_KEYS)
    validate_prompt(task_config.get('prompt'))
    from nexus.extensions.tasks.reward_system.task import RewardConfig

    RewardConfig.model_validate(task_config.get('reward'))


__all__ = ['TASK_CONFIG_KEYS', 'validate_prompt', 'validate_task_config']
