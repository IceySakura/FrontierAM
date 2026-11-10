import http.server
import threading

from arena import atr_relay

RATE = (b'{"error":{"message":"Rate limit reached for gpt-6.1-sol on tokens per min (TPM). Please try again in 19ms.",'
        b'"type":"tokens","code":"rate_limit_exceeded"}}')


def upstream(answers):
    calls = []

    class Handler(http.server.BaseHTTPRequestHandler):
        def do_POST(self):
            calls.append(self.rfile.read(int(self.headers['Content-Length'])))
            status, body = answers[min(len(calls), len(answers)) - 1]
            self.send_response(status)
            self.send_header('Content-Type', 'application/json')
            self.send_header('Content-Length', str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def log_message(self, *args):
            pass

    server = http.server.ThreadingHTTPServer(('127.0.0.1', 0), Handler)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    return server, calls


def test_transient_rate_limit_is_retried_until_success():
    server, calls = upstream([(429, RATE), (429, RATE), (200, b'{"ok": true}')])
    events, waits = [], []
    status, _, _, body = atr_relay.forward(server.server_address[1], 'POST', '/v1/responses', [('Content-Length', '2')],
                                           b'{}', events.append, sleep=waits.append)
    assert (status, body) == (200, b'{"ok": true}')
    assert len(calls) == 3 and [e['event'] for e in events] == ['rate_limit_retry'] * 2 and waits == [1.0, 2.0]
    server.shutdown()


def test_quota_and_other_errors_are_returned_unchanged():
    quota = b'{"error":{"code":"insufficient_quota","message":"rate limit"}}'
    for answer in ((429, quota), (500, b'{"error":"boom"}')):
        server, calls = upstream([answer])
        status, _, _, body = atr_relay.forward(server.server_address[1], 'POST', '/v1/responses', [], b'{}', lambda e: None,
                                               sleep=lambda s: None)
        assert (status, body) == answer and len(calls) == 1
        server.shutdown()


def test_suggested_wait_parses_ms_and_seconds():
    assert atr_relay.suggested_wait(b'Please try again in 19ms.') == 0.019
    assert atr_relay.suggested_wait(b'Please try again in 2.5s.') == 2.5
    assert atr_relay.suggested_wait(b'no hint') == 1.0
