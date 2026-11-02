"""A scripted OpenAI-compatible chat completions server for driving a real ``wb --acp`` locally.

It answers ``/v1/chat/completions`` (streamed or not) from a script that sees the request body and
headers, and records every request so a test can tell which conversation asked what.
"""

from __future__ import annotations

import json
import threading
import time
import uuid

from collections.abc import Callable
from dataclasses import dataclass, field
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any


@dataclass
class StubReply:
    """One assistant turn: text streamed in ``chunks`` pieces, optionally followed by one tool call."""

    text: str = ''
    tool_name: str | None = None
    tool_args: dict[str, Any] | None = None
    chunks: int = 1
    chunk_delay: float = 0.0


StubScript = Callable[[dict[str, Any], dict[str, str]], StubReply]
"""Receives ``(request_body, request_headers)`` and returns the reply."""


@dataclass
class StubModelServer:
    """Serve ``script`` on a random loopback port until ``stop``."""

    script: StubScript
    requests: list[dict[str, Any]] = field(default_factory=list)
    _server: ThreadingHTTPServer | None = None

    @property
    def base_url(self) -> str:
        if self._server is None:
            raise RuntimeError('StubModelServer is not started')
        return f'http://127.0.0.1:{self._server.server_address[1]}/v1'

    def start(self) -> str:
        stub = self

        class Handler(BaseHTTPRequestHandler):
            protocol_version = 'HTTP/1.1'

            def log_message(self, format: str, *args: Any) -> None:  # noqa: A002
                return

            def do_GET(self) -> None:
                self._json({'object': 'list', 'data': []})

            def do_POST(self) -> None:
                length = int(self.headers.get('Content-Length') or 0)
                raw = self.rfile.read(length) if length else b''
                if not self.path.rstrip('/').endswith('chat/completions'):
                    self.send_response(404)
                    self.send_header('Content-Length', '0')
                    self.end_headers()
                    return
                body = json.loads(raw or b'{}')
                headers = dict(self.headers.items())
                reply = stub.script(body, headers)
                stub.requests.append({'time': time.time(), 'headers': headers, 'body': body, 'reply': reply})
                if body.get('stream'):
                    self._stream(body, reply)
                else:
                    self._json(_completion(body, reply))

            def _json(self, payload: dict[str, Any]) -> None:
                data = json.dumps(payload).encode()
                self.send_response(200)
                self.send_header('Content-Type', 'application/json')
                self.send_header('Content-Length', str(len(data)))
                self.end_headers()
                self.wfile.write(data)

            def _stream(self, body: dict[str, Any], reply: StubReply) -> None:
                self.send_response(200)
                self.send_header('Content-Type', 'text/event-stream')
                self.send_header('Cache-Control', 'no-cache')
                self.send_header('Connection', 'close')
                self.end_headers()
                completion_id = f'chatcmpl-{uuid.uuid4().hex[:12]}'

                def send(payload: dict[str, Any]) -> None:
                    self.wfile.write(f'data: {json.dumps(payload)}\n\n'.encode())
                    self.wfile.flush()

                def chunk(delta: dict[str, Any], finish_reason: str | None = None) -> dict[str, Any]:
                    return {
                        'id': completion_id,
                        'object': 'chat.completion.chunk',
                        'created': int(time.time()),
                        'model': body.get('model', 'stub-model'),
                        'choices': [{'index': 0, 'delta': delta, 'finish_reason': finish_reason}],
                    }

                try:
                    send(chunk({'role': 'assistant', 'content': ''}))
                    if reply.text:
                        size = max(1, len(reply.text) // max(1, reply.chunks) + 1)
                        for start in range(0, len(reply.text), size):
                            if reply.chunk_delay:
                                time.sleep(reply.chunk_delay)
                            send(chunk({'content': reply.text[start : start + size]}))
                    if reply.tool_name:
                        send(chunk({'tool_calls': [{'index': 0, **_tool_call(reply)}]}))
                    send(chunk({}, 'tool_calls' if reply.tool_name else 'stop'))
                    usage = chunk({})
                    usage['choices'] = []
                    usage['usage'] = {'prompt_tokens': 10, 'completion_tokens': 5, 'total_tokens': 15}
                    send(usage)
                    self.wfile.write(b'data: [DONE]\n\n')
                    self.wfile.flush()
                except (BrokenPipeError, ConnectionResetError):
                    return
                self.close_connection = True

        self._server = ThreadingHTTPServer(('127.0.0.1', 0), Handler)
        self._server.daemon_threads = True
        threading.Thread(target=self._server.serve_forever, name='wb-stub-model', daemon=True).start()
        return self.base_url

    def stop(self) -> None:
        if self._server is not None:
            self._server.shutdown()
            self._server.server_close()
            self._server = None


def _tool_call(reply: StubReply) -> dict[str, Any]:
    return {
        'id': f'call_{uuid.uuid4().hex[:12]}',
        'type': 'function',
        'function': {'name': reply.tool_name, 'arguments': json.dumps(reply.tool_args or {})},
    }


def _completion(body: dict[str, Any], reply: StubReply) -> dict[str, Any]:
    message: dict[str, Any] = {'role': 'assistant', 'content': reply.text or None}
    if reply.tool_name:
        message['tool_calls'] = [_tool_call(reply)]
    return {
        'id': f'chatcmpl-{uuid.uuid4().hex[:12]}',
        'object': 'chat.completion',
        'created': int(time.time()),
        'model': body.get('model', 'stub-model'),
        'choices': [{'index': 0, 'message': message, 'finish_reason': 'tool_calls' if reply.tool_name else 'stop'}],
        'usage': {'prompt_tokens': 10, 'completion_tokens': 5, 'total_tokens': 15},
    }


def message_text(content: Any) -> str:
    """Flatten an OpenAI message ``content`` (a string or a list of parts) into text."""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        return ' '.join(str(part.get('text', '')) for part in content if isinstance(part, dict))
    return ''


__all__ = ['StubModelServer', 'StubReply', 'StubScript', 'message_text']
