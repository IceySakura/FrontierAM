"""Chat backend compatibility selected by the macOSWorld Codex adapter."""

from typing import Any

from nexus.runtime.server.services.responses_proxy.converter import responses_to_openai


def responses_to_chat(*args: Any, **kwargs: Any) -> dict[str, Any]:
    """Convert for Chat-only backends without changing the original image bytes."""
    params = responses_to_openai(*args, **kwargs)
    messages = []
    pending_images = []
    for message in params['messages']:
        if message['role'] != 'tool' and pending_images:
            messages.append({'role': 'user', 'content': pending_images})
            pending_images = []
        message.pop('reasoning_items', None)
        content = message.get('content')
        for part in content if isinstance(content, list) else []:
            image = part.get('image_url') if isinstance(part, dict) else None
            if isinstance(image, dict) and image.get('detail') == 'original':
                image.pop('detail')
        # Chat tool results only permit text. Keep every tool_call_id paired,
        # then deliver images as user input after the consecutive tool results.
        if message['role'] == 'tool' and isinstance(content, list):
            pending_images.extend(part for part in content if part.get('type') == 'image_url')
            text_parts = [part.get('text', '') for part in content if part.get('type') == 'text']
            message['content'] = '\n'.join(text_parts) or 'Image output is provided in the following user message.'
        messages.append(message)
    if pending_images:
        messages.append({'role': 'user', 'content': pending_images})
    params['messages'] = messages
    return params
