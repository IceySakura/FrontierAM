"""Capture WorkBuddy request metadata before the host sanitizes completion fields."""

from __future__ import annotations

import asyncio
import json
import time

from pathlib import Path
from typing import Any
from uuid import uuid4

from starlette.types import ASGIApp, Message, Receive, Scope, Send

from .gateway_proxy import WorkBuddyOpenAIProxy


class WorkBuddyMetadataProxy(WorkBuddyOpenAIProxy):
    """Save raw request envelopes for opt-in metadata diagnosis, then forward like WorkBuddyOpenAIProxy."""

    def __init__(self, *args: Any, capture_dir: str, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self.capture_dir = Path(capture_dir)
        self.capture_dir.mkdir(parents=True, exist_ok=True, mode=0o700)

    def _create_app(self) -> Any:
        app = super()._create_app()
        app.add_middleware(_WorkBuddyRequestCaptureMiddleware, capture_dir=self.capture_dir)  # ty: ignore[invalid-argument-type] -- ty does not match an ASGI class to Starlette's middleware factory protocol
        return app


class _WorkBuddyRequestCaptureMiddleware:
    def __init__(self, app: ASGIApp, capture_dir: Path) -> None:
        self.app = app
        self.capture_dir = capture_dir

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        if scope['type'] != 'http' or scope['path'] not in {'/v1/chat/completions', '/chat/completions'}:
            await self.app(scope, receive, send)
            return
        chunks: list[bytes] = []

        async def capture_receive() -> Message:
            message = await receive()
            if message['type'] == 'http.request':
                chunks.append(message.get('body', b''))
                if message.get('more_body', False):
                    return message
                sensitive_headers = {'authorization', 'proxy-authorization', 'x-api-key', 'cookie'}
                headers = {
                    name.decode('latin-1'): (
                        '[REDACTED]' if name.decode('latin-1').lower() in sensitive_headers else value.decode('latin-1')
                    )
                    for name, value in scope.get('headers', [])
                }
                payload = {
                    'time_unix_nano': time.time_ns(),
                    'method': scope['method'],
                    'path': scope['path'],
                    'headers': headers,
                    'body': json.loads(b''.join(chunks)),
                }
                path = self.capture_dir / f'{uuid4().hex}.request.json'
                await asyncio.to_thread(_write_capture, path, payload)
            return message

        await self.app(scope, capture_receive, send)


def _write_capture(path: Path, payload: dict[str, Any]) -> None:
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
    path.chmod(0o600)
