"""Controlled capacity API for runner-dev policy acceptance. Deploy only as a temporary test fixture."""

from __future__ import annotations

import json
import os
import threading
import time

from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer


state = {'remaining': 0, 'factor': 1.0, 'delay': 0, 'invalid': False, 'calls': 0}
lock = threading.Lock()
key = os.environ['POLICY_TEST_KEY']


class Handler(BaseHTTPRequestHandler):
    def do_GET(self):
        if self.path == '/health':
            return self.respond({'ok': True})
        if self.path != '/state' or self.headers.get('X-Policy-Test-Key') != key:
            return self.send_error(403)
        with lock:
            result = dict(state)
        self.respond(result)
        return None

    def do_POST(self):
        length = int(self.headers.get('Content-Length', '0'))
        if not 0 < length <= 4096:
            return self.send_error(400)
        payload = json.loads(self.rfile.read(length))
        if self.path == '/state':
            if self.headers.get('X-Policy-Test-Key') != key:
                return self.send_error(403)
            with lock:
                for name in ('remaining', 'factor', 'delay', 'invalid'):
                    if name in payload:
                        state[name] = payload[name]
                result = dict(state)
            return self.respond(result)
        if self.path != '/capacity' or payload.get('api_key') != key:
            return self.send_error(403)
        with lock:
            state['calls'] += 1
            result = dict(state)
        time.sleep(min(30, max(0, result['delay'])))
        self.respond({
            'success': True,
            'key_remaining_raw_qpm': True if result['invalid'] else result['remaining'],
            'scaling_advice': {'factor': False if result['invalid'] else result['factor']},
        })
        return None

    def respond(self, value):
        body = json.dumps(value).encode()
        try:
            self.send_response(200)
            self.send_header('Content-Type', 'application/json')
            self.send_header('Content-Length', str(len(body)))
            self.end_headers()
            self.wfile.write(body)
        except (BrokenPipeError, ConnectionResetError):
            pass

    def log_message(self, *_args):
        pass


if __name__ == '__main__':
    ThreadingHTTPServer(('0.0.0.0', 8080), Handler).serve_forever()
