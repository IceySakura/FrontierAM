"""Translate only Nexus budget errors to the generic Hyness model-error contract."""

from __future__ import annotations

import json

from typing import Any


def translate_budget(body: bytes) -> bytes:
    try:
        value = json.loads(body)
    except (ValueError, UnicodeDecodeError):
        return body
    error = value.get('error') if isinstance(value, dict) else None
    if not isinstance(error, dict) or error.get('type') != 'invalid_request_error':
        return body
    if error.get('code') != 'context_length_exceeded' and not (
        value.get('type') == 'error'
        and isinstance(error.get('message'), str)
        and error['message'].startswith('[context_length_exceeded] ')
    ):
        return body
    error['host'] = {'protocol': 'hyness.model-error', 'schemaVersion': 1, 'code': 'host_budget_exhausted'}
    return json.dumps(value, ensure_ascii=False, separators=(',', ':')).encode()


def translate_frame(frame: bytes) -> bytes:
    if len(frame) > 65536 or not any(line.strip() == b'event: error' for line in frame.splitlines()):
        return frame
    lines = frame.splitlines(keepends=True)
    for i, line in enumerate(lines):
        if line.startswith(b'data: '):
            body = line[6:].rstrip(b'\r\n')
            lines[i] = b'data: ' + translate_budget(body) + line[len(line.rstrip(b'\r\n')) :]
    return b''.join(lines)


def budget_translation(app: Any) -> Any:
    """ASGI wrapper with bounded inspection; normal model frames and request bodies pass unchanged."""

    async def wrapped(scope, receive, send):
        if scope['type'] != 'http' or scope.get('path') != '/v1/messages':
            return await app(scope, receive, send)
        mode = None
        pending = b''
        passthrough = False

        async def translated(event):
            nonlocal mode, pending, passthrough
            if event['type'] == 'http.response.start':
                headers = event.get('headers', [])
                content_type = dict(headers).get(b'content-type', b'')
                if event['status'] >= 400 and b'json' in content_type:
                    mode = 'json'
                    event = {**event, 'headers': [(k, v) for k, v in headers if k.lower() != b'content-length']}
                elif b'text/event-stream' in content_type:
                    mode = 'sse'
            elif event['type'] == 'http.response.body' and mode:
                body = event.get('body', b'')
                more = event.get('more_body', False)
                if mode == 'json':
                    if not passthrough:
                        pending += body
                        if len(pending) <= 65536 and more:
                            return
                        body, pending = pending, b''
                        if len(body) <= 65536:
                            body = translate_budget(body)
                        passthrough = True
                else:
                    pending += body
                    output = []
                    while b'\n\n' in pending:
                        frame, pending = pending.split(b'\n\n', 1)
                        output.append(frame + b'\n\n' if passthrough else translate_frame(frame + b'\n\n'))
                        passthrough = False
                    if len(pending) > 65536 or not more:
                        tail = 1 if more and pending.endswith(b'\n') else 0
                        output.append(pending[:-tail] if tail else pending)
                        pending = pending[-tail:] if tail else b''
                        passthrough = True
                    body = b''.join(output)
                event = {**event, 'body': body}
            await send(event)

        await app(scope, receive, translated)
        return None

    return wrapped
