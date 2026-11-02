"""Annotate tool-returned images through main_agent's existing iteration hook."""

import logging
import re

from collections.abc import Iterator
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import replace
from typing import Any

from nexus.core import import_from_path
from nexus.extensions.agents.main.agent import PostIterationHookContext, PostIterationHookResult
from nexus.extensions.agents.visual_agent.resources.media import image_header, pixel_size


logger = logging.getLogger(__name__)
HOOK_PATH = 'nexus.extensions.agents.visual_agent.resources.tool_images::annotate_tool_images'
_user_hook: ContextVar[Any] = ContextVar('visual_image_user_hook', default=None)


@contextmanager
def image_hook_context(user_hook: str | None) -> Iterator[None]:
    """Keep the caller's hook scoped to this agent run, including concurrent runs."""
    token = _user_hook.set(import_from_path(user_hook) if user_hook else None)
    try:
        yield
    finally:
        _user_hook.reset(token)


def _annotate_content(content: list[dict[str, Any]]) -> list[dict[str, Any]]:
    parts: list[dict[str, Any]] = []
    for part in content:
        image = part.get('image_url')
        url = image.get('url') if isinstance(image, dict) else image
        if part.get('type') == 'image_url' and isinstance(url, str) and url.startswith('data:image/'):
            try:
                size = pixel_size(image_header(url))
            except (ValueError, IndexError):
                size = None
            if size:
                suffix = f' ({size[0]}x{size[1]} px)'
                previous = parts[-1].get('text', '') if parts else ''
                if not previous.endswith(suffix):
                    if parts and parts[-1].get('type') == 'text' and re.search(r'!\[[^\]]*\]\([^\n]*\)$', previous):
                        parts[-1] = {**parts[-1], 'text': previous + suffix}
                    else:
                        parts.append({'type': 'text', 'text': f'Image dimensions:{suffix}'})
        parts.append(part)
    return parts


async def annotate_tool_images(ctx, hook_context: PostIterationHookContext) -> PostIterationHookResult:
    """Add size hints before the next model call, then preserve the caller's hook."""
    messages = [
        {**message, 'content': _annotate_content(message['content'])}
        if isinstance(message.get('content'), list)
        else message
        for message in hook_context.tool_messages
    ]
    if user_hook := _user_hook.get():
        try:
            result = await user_hook(ctx, replace(hook_context, tool_messages=tuple(messages)))
        except Exception:
            logger.warning('VisualAgent user post-iteration hook failed', exc_info=True)
        else:
            if result is not None:
                return replace(result, tool_messages=messages if result.tool_messages is None else result.tool_messages)
    return PostIterationHookResult(tool_messages=messages)
