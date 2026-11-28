"""Host-side relay from a development container's socket to `atr serve`, retrying upstream rate limits.

The container's ATR client transport sends one non-streaming request per connection (Content-Length body) and waits
for the whole response. This relay forwards each request to atr serve; when the upstream answers 429 for a transient
rate limit (organisation-wide tokens or requests per minute) or AI Hub's 1011 (no available upstream account, HTTP
500), it waits and sends the same request again, with backoff,
for at most RETRY_BUDGET seconds, instead of handing the 429 to the agent's client (whose own retries do not reach the
upstream through this chain). Any other answer, including quota exhaustion, is returned unchanged. Retries are logged.

Usage: python -m arena.atr_relay SOCKET PORT LOG
"""

from __future__ import annotations

import http.client
import json
import os
import re
import socket
import sys
import threading
import time

RETRY_BUDGET = 900.0
# "credit balance is too low": the model gateway's bound upstream account ran dry; ATR rotates the affinity id and
# answers 429, and a later request lands on another account (an opus attempt exited on it, 2026-11-17).
TRANSIENT = (b'rate_limit_exceeded', b'Rate limit reached', b'rate limit', b'credit balance is too low')
# AI Hub 1011 (HTTP 500, PlatformNoAvailableAccount): no upstream account was free for the model at that moment; the
# gateway's retry policy marks it retryable (list_retry_policies; glm-5.4 failed most requests this way, 2026-11-28).
NO_ACCOUNT = b'PlatformNoAvailableAccount'
HOP = {'connection', 'keep-alive', 'transfer-encoding', 'content-length', 'proxy-connection', 'te', 'trailer', 'upgrade'}


def transient(status: int, body: bytes) -> bool:
    if status == 500 and NO_ACCOUNT in body:
        return True
    return status == 429 and any(m in body for m in TRANSIENT) and b'insufficient_quota' not in body


def suggested_wait(body: bytes) -> float:
    match = re.search(rb'try again in ([0-9.]+)\s*(ms|s)', body)
    if not match:
        return 1.0
    value = float(match.group(1))
    return value / 1000 if match.group(2) == b'ms' else value


def read_request(stream) -> tuple[str, str, list[tuple[str, str]], bytes] | None:
    line = stream.readline(65537)
    if not line:
        return None
    method, path, _ = line.decode('latin-1').rstrip('\r\n').split(' ', 2)
    headers = []
    while True:
        raw = stream.readline(65537)
        if raw in (b'\r\n', b'\n', b''):
            break
        name, _, value = raw.decode('latin-1').partition(':')
        headers.append((name.strip(), value.strip()))
    length = int(next((v for n, v in headers if n.lower() == 'content-length'), '0'))
    return method, path, headers, stream.read(length) if length else b''


def forward(port: int, method: str, path: str, headers, body: bytes, log, sleep=time.sleep) -> tuple[int, str, list, bytes]:
    started, attempt = time.monotonic(), 0
    while True:
        connection = http.client.HTTPConnection('127.0.0.1', port, timeout=None)
        connection.request(method, path, body=body, headers={n: v for n, v in headers if n.lower() not in HOP | {'host'}})
        response = connection.getresponse()
        data = response.read()
        result = (response.status, response.reason, response.getheaders(), data)
        connection.close()
        if not transient(response.status, data):
            return result
        attempt += 1
        wait = min(30.0, max(1.0, suggested_wait(data)) * 2 ** min(attempt - 1, 5))
        if time.monotonic() - started + wait > RETRY_BUDGET:
            log({'event': 'rate_limit_gave_up', 'attempts': attempt, 'waited_s': round(time.monotonic() - started, 1)})
            return result
        log({'event': 'rate_limit_retry', 'attempt': attempt, 'wait_s': wait, 'status': response.status})
        sleep(wait)


def serve(sock: socket.socket, port: int, log) -> None:
    try:
        stream = sock.makefile('rb')
        request = read_request(stream)
        if request is None:
            return
        status, reason, headers, data = forward(port, *request, log)
        head = [f'HTTP/1.1 {status} {reason}'] + [f'{n}: {v}' for n, v in headers if n.lower() not in HOP]
        head += [f'Content-Length: {len(data)}', 'Connection: close']
        sock.sendall(('\r\n'.join(head) + '\r\n\r\n').encode('latin-1') + data)
    except (OSError, ValueError, http.client.HTTPException) as error:
        log({'event': 'relay_error', 'error': f'{type(error).__name__}: {error}'})
    finally:
        sock.close()


def main() -> None:
    path, port, log_path = sys.argv[1], int(sys.argv[2]), sys.argv[3]
    lock = threading.Lock()

    def log(event: dict) -> None:
        with lock, open(log_path, 'a') as stream:
            stream.write(json.dumps({'time': time.time(), **event}) + '\n')

    server = socket.socket(socket.AF_UNIX)
    server.bind(path)
    os.chmod(path, 0o666)
    server.listen(64)
    while True:
        client, _ = server.accept()
        threading.Thread(target=serve, args=(client, port, log), daemon=True).start()


if __name__ == '__main__':
    main()
