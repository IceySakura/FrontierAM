"""Controlled capacity and OpenAI-compatible inference fixture for K8s shared request budget E2E."""

from __future__ import annotations

import json
import os
import threading
import time

from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any


lock = threading.Lock()
test_key = os.environ.get('POLICY_TEST_KEY', 'budget-e2e-admin-key')
platform_key = os.environ.get('TIDE_PLATFORM_API_KEY', 'budget-e2e-quota-group-key')
client_key = os.environ.get('TIDE_CLIENT_API_KEY', 'budget-e2e-client-key')

state: dict[str, Any] = {
    'factor': 1.0,
    'hard_brake': False,
    'remaining': 600,
    'mode': 'ok',
    'delay': 0.0,
    'platform_calls': 0,
    'platform_keyed_calls': 0,
    'client_calls': 0,
    'completions': 0,
}
completion_times: list[float] = []


class Handler(BaseHTTPRequestHandler):
    def do_GET(self) -> None:
        if self.path == '/health':
            self._respond(200, {'ok': True})
            return
        if self.path != '/state' or self.headers.get('X-Policy-Test-Key') != test_key:
            self.send_error(403)
            return
        with lock:
            now = time.monotonic()
            recent = [t for t in completion_times if now - t <= 60.0]
            payload = {**state, 'recent_completions': recent, 'server_monotonic': now}
        self._respond(200, payload)

    def do_POST(self) -> None:
        length = int(self.headers.get('Content-Length', '0'))
        if not 0 <= length <= 65536:
            self.send_error(400)
            return
        raw = self.rfile.read(length) if length else b'{}'
        try:
            payload = json.loads(raw)
        except ValueError:
            self.send_error(400)
            return

        if self.path == '/state':
            if self.headers.get('X-Policy-Test-Key') != test_key:
                self.send_error(403)
                return
            with lock:
                for field in ('factor', 'hard_brake', 'remaining', 'mode', 'delay'):
                    if field in payload:
                        state[field] = payload[field]
                if payload.get('reset_counts'):
                    state['platform_calls'] = 0
                    state['platform_keyed_calls'] = 0
                    state['client_calls'] = 0
                    state['completions'] = 0
                    completion_times.clear()
                now = time.monotonic()
                snapshot = {**state, 'server_monotonic': now}
            self._respond(200, snapshot)
            return

        if self.path == '/capacity':
            api_key = payload.get('api_key')
            with lock:
                if api_key == client_key:
                    state['client_calls'] += 1
                else:
                    state['platform_calls'] += 1
                    if api_key == platform_key:
                        state['platform_keyed_calls'] += 1
                current = dict(state)
            if current['delay'] > 0:
                time.sleep(min(30.0, float(current['delay'])))
            if current['mode'] == 'error':
                self._respond(503, {'error': 'capacity_unavailable'})
                return
            if current['mode'] == 'invalid':
                self._respond(200, {'success': False, 'error': 'invalid_upstream_sample'})
                return
            body: dict[str, Any] = {
                'success': True,
                'model': payload.get('model', 'hy3-offline'),
                'scaling_advice': {
                    'suggest': 'hold',
                    'factor': current['factor'],
                    'hard_brake': bool(current['hard_brake']),
                    'load': 0.5,
                },
            }
            if api_key:
                body['key_remaining_raw_qpm'] = current['remaining']
            self._respond(200, body)
            return

        if self.path == '/v1/chat/completions':
            with lock:
                state['completions'] += 1
                index = state['completions']
                completion_times.append(time.monotonic())
            self._respond(
                200,
                {
                    'id': f'chatcmpl-budget-k8s-{index}',
                    'object': 'chat.completion',
                    'created': int(time.time()),
                    'model': payload.get('model', 'hy3-offline'),
                    'choices': [
                        {
                            'index': 0,
                            'message': {'role': 'assistant', 'content': 'ok'},
                            'finish_reason': 'stop',
                        }
                    ],
                    'usage': {'prompt_tokens': 4, 'completion_tokens': 1, 'total_tokens': 5},
                },
            )
            return

        self.send_error(404)

    def _respond(self, status: int, value: dict[str, Any]) -> None:
        body = json.dumps(value).encode('utf-8')
        try:
            self.send_response(status)
            self.send_header('Content-Type', 'application/json')
            self.send_header('Content-Length', str(len(body)))
            self.end_headers()
            self.wfile.write(body)
        except (BrokenPipeError, ConnectionResetError):
            pass

    def log_message(self, *_args: Any) -> None:
        pass


if __name__ == '__main__':
    ThreadingHTTPServer(('0.0.0.0', 8080), Handler).serve_forever()
