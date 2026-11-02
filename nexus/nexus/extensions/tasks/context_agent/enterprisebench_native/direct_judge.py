"""Optional Runtime-local, native Responses relay. No Chat conversion or host LLM RPC.

Only an ephemeral public key and encrypted configuration cross traced RPCs.
Credentials live in the relay's memory, never in task files or captured HTTP logs.
"""

from __future__ import annotations

import asyncio
import base64
import contextlib
import json
import os
import time
import uuid

from pathlib import Path
from typing import Any

import aiohttp

from aiohttp import web

from nexus.extensions.tasks.context_agent.enterprisebench_native.errors import judge_length_reason


_RELAYS: dict[str, Any] = {}
ROOT = Path('/logs/verifier/direct_responses')


def _write(path: Path, value: Any) -> None:
    temporary = path.with_suffix('.tmp')
    temporary.write_text(json.dumps(value, ensure_ascii=False), encoding='utf-8')
    temporary.chmod(0o600)
    temporary.replace(path)


class DirectRelay:
    def __init__(self) -> None:
        from cryptography.hazmat.primitives.asymmetric import rsa

        self.private_key = rsa.generate_private_key(public_exponent=65537, key_size=2048)
        self.config: dict[str, Any] = {}
        self.runner: web.AppRunner | None = None
        self.session: aiohttp.ClientSession | None = None
        self.pending: set[asyncio.Task] = set()
        self.started = time.monotonic()
        self.requests = 0
        self.successes = 0
        self.failures = 0
        self.rate_limits = 0
        self.abort_reason: str | None = None
        self.first_request_seconds: float | None = None
        self.records: list[dict[str, Any]] = []
        self.lock = asyncio.Lock()

    def clean(self, value: Any) -> Any:
        if isinstance(value, str):
            for secret in (
                self.config.get('api_key'),
                self.config.get('api_key', '').split('?', 1)[0],
                self.config.get('local_key'),
            ):
                if secret:
                    value = value.replace(secret, '[REDACTED]')
            return value
        if isinstance(value, list):
            return [self.clean(v) for v in value]
        if isinstance(value, dict):
            return {
                k: '[REDACTED]' if k.lower() in {'authorization', 'api_key', 'apikey', 'token'} else self.clean(v)
                for k, v in value.items()
            }
        return value

    def snapshot(self) -> None:
        _write(
            ROOT / 'status.json',
            {
                'requests': self.requests,
                'successful_responses': self.successes,
                'consecutive_failures': self.failures,
                'consecutive_rate_limits': self.rate_limits,
                'first_request_seconds': self.first_request_seconds,
                'abort_reason': self.abort_reason,
                'elapsed_seconds': time.monotonic() - self.started,
                'records': self.records,
            },
        )

    def failed(self, status: int | None) -> None:
        self.failures += 1
        self.rate_limits = self.rate_limits + 1 if status == 429 else 0
        for count, limit, reason in (
            (self.rate_limits, self.config['max_rate_limits'], 'judge.rate_limit'),
            (self.failures, self.config['max_failures'], 'judge.model_failures'),
        ):
            if limit and count >= limit:
                self.abort_reason = reason
                break
        self.snapshot()

    async def handle(self, request: web.Request) -> web.StreamResponse:
        if request.headers.get('Authorization') != 'Bearer ' + self.config['local_key']:
            return web.json_response({'error': {'message': 'Unauthorized'}}, status=401)
        if self.abort_reason:
            return web.json_response({'error': {'message': self.abort_reason}}, status=503)
        if request.path != '/v1/responses':
            return web.json_response({'error': {'message': 'Only POST /v1/responses is supported'}}, status=404)
        # Serialize relay requests so counters and stream files cannot race.
        async with self.lock:
            if self.abort_reason:
                return web.json_response({'error': {'message': self.abort_reason}}, status=503)
            task = asyncio.current_task()
            if task:
                self.pending.add(task)
            try:
                return await self.forward(request)
            finally:
                if task:
                    self.pending.discard(task)

    async def forward(self, request: web.Request) -> web.StreamResponse:
        body = await request.json()
        if not isinstance(body, dict):
            raise web.HTTPBadRequest(text='Expected a JSON object')
        # Preserve OpenCode input/tools/stream verbatim. Only explicit YAML controls win.
        body.update(self.config['request_args'])
        body['parallel_tool_calls'] = False  # Same policy as existing Judge bridge.
        self.requests += 1
        number = self.requests
        started = time.monotonic()
        if self.first_request_seconds is None:
            self.first_request_seconds = started - self.started
        record: dict[str, Any] = {'request': number, 'start_seconds': started - self.started, 'attempts': []}
        self.records.append(record)
        _write(ROOT / f'{number:05d}.request.json', self.clean(body))
        self.snapshot()
        assert self.session is not None
        for attempt in range(1, self.config['max_attempts'] + 1):
            attempt_started = time.monotonic()
            prepared = False
            try:
                async with self.session.post(
                    self.config['endpoint'],
                    json=body,
                    headers={'Authorization': 'Bearer ' + self.config['api_key']},
                    allow_redirects=False,
                ) as upstream:
                    status = upstream.status
                    entry = {'attempt': attempt, 'status': status}
                    record['attempts'].append(entry)
                    if not 200 <= status < 300:
                        error = await upstream.read()
                        _write(
                            ROOT / f'{number:05d}.attempt-{attempt}.error.json',
                            {
                                'status': status,
                                'body': self.clean(error.decode(errors='replace')),
                            },
                        )
                        entry['duration_seconds'] = time.monotonic() - attempt_started
                        try:
                            length_reason = judge_length_reason(json.loads(error))
                        except ValueError:
                            length_reason = None
                        if length_reason:
                            self.abort_reason = 'judge.length'
                            record['length_reason'] = length_reason
                            self.snapshot()
                            return web.json_response(
                                {'error': {'message': 'Judge context length exceeded'}}, status=400
                            )
                        self.snapshot()
                        if (status == 429 or status >= 500) and attempt < self.config['max_attempts']:
                            await asyncio.sleep(min(2 ** (attempt - 1), 10))
                            continue
                        self.failed(status)
                        return web.Response(
                            text=self.clean(error.decode(errors='replace')),
                            status=status,
                            content_type='application/json',
                        )
                    response = web.StreamResponse(
                        status=status,
                        headers={'Content-Type': upstream.headers.get('Content-Type', 'application/json')},
                    )
                    await response.prepare(request)
                    prepared = True
                    is_sse = 'text/event-stream' in upstream.headers.get('Content-Type', '')
                    succeeded = False
                    if is_sse:
                        # aiohttp iterates by line: retain native event ordering and contents.
                        with (ROOT / f'{number:05d}.events.jsonl').open('a', encoding='utf-8') as log:
                            async for line in upstream.content:
                                if 'first_byte_seconds' not in record:
                                    record['first_byte_seconds'] = time.monotonic() - started
                                    self.snapshot()
                                if not line.startswith(b'data:'):
                                    await response.write(line)
                                    continue
                                payload = line[5:].strip()
                                if payload == b'[DONE]':
                                    await response.write(line)
                                    continue
                                try:
                                    event = json.loads(payload)
                                except (ValueError, UnicodeDecodeError):
                                    await response.write(line)
                                    continue
                                log.write(json.dumps(self.clean(event), ensure_ascii=False) + '\n')
                                log.flush()
                                length_reason = judge_length_reason(event.get('response', event))
                                if length_reason:
                                    self.abort_reason = 'judge.length'
                                    record['length_reason'] = length_reason
                                    self.snapshot()
                                    # Close rather than let OpenCode interpret overflow as compaction.
                                    if request.transport:
                                        request.transport.close()
                                    return response
                                await response.write(line)
                                if event.get('type') == 'response.completed':
                                    succeeded = True
                    else:
                        payload = await upstream.read()
                        value = json.loads(payload)
                        _write(ROOT / f'{number:05d}.response.json', self.clean(value))
                        length_reason = judge_length_reason(value)
                        if length_reason:
                            self.abort_reason = 'judge.length'
                            record['length_reason'] = length_reason
                            self.snapshot()
                            if request.transport:
                                request.transport.close()
                            return response
                        await response.write(payload)
                        succeeded = value.get('status') == 'completed'
                    entry['duration_seconds'] = time.monotonic() - attempt_started
                    record['duration_seconds'] = time.monotonic() - started
                    if succeeded:
                        self.successes += 1
                        self.failures = self.rate_limits = 0
                        self.snapshot()
                    else:
                        self.failed(None)
                    await response.write_eof()
                    return response
            except asyncio.CancelledError:
                record['cancelled'] = True
                self.snapshot()
                raise
            except Exception as exc:
                record['attempts'].append({'attempt': attempt, 'error_type': type(exc).__name__})
                self.snapshot()
                # Never replay a request after sending any upstream stream to OpenCode.
                if prepared:
                    self.failed(None)
                    raise ConnectionResetError('Upstream stream interrupted') from None
                if attempt == self.config['max_attempts']:
                    self.failed(None)
                    return web.json_response({'error': {'message': type(exc).__name__}}, status=502)
                await asyncio.sleep(min(2 ** (attempt - 1), 10))
        raise RuntimeError('Unreachable')


async def prepare() -> dict[str, str]:
    from cryptography.hazmat.primitives import serialization

    relay = DirectRelay()
    identifier = uuid.uuid4().hex
    _RELAYS[identifier] = relay
    public = (
        relay.private_key
        .public_key()
        .public_bytes(
            serialization.Encoding.PEM,
            serialization.PublicFormat.SubjectPublicKeyInfo,
        )
        .decode()
    )
    return {'id': identifier, 'public_key': public}


def seal(public_key: str, config: dict[str, Any]) -> str:
    """Host helper; return encrypted bytes only, safe for RPC tracing."""
    from cryptography.hazmat.primitives import hashes, serialization
    from cryptography.hazmat.primitives.asymmetric import padding, rsa
    from cryptography.hazmat.primitives.ciphers.aead import AESGCM

    key = AESGCM.generate_key(bit_length=256)
    nonce = os.urandom(12)
    public = serialization.load_pem_public_key(public_key.encode())
    if not isinstance(public, rsa.RSAPublicKey):
        raise ValueError('Expected RSA public key')
    wrapped = public.encrypt(
        key, padding.OAEP(mgf=padding.MGF1(hashes.SHA256()), algorithm=hashes.SHA256(), label=None)
    )
    ciphertext = AESGCM(key).encrypt(nonce, json.dumps(config).encode(), None)
    return base64.b64encode(wrapped + nonce + ciphertext).decode()


async def configure(identifier: str, encrypted_config: str) -> dict[str, str]:
    from cryptography.hazmat.primitives import hashes
    from cryptography.hazmat.primitives.asymmetric import padding
    from cryptography.hazmat.primitives.ciphers.aead import AESGCM

    relay = _RELAYS[identifier]
    raw = base64.b64decode(encrypted_config)
    key = relay.private_key.decrypt(
        raw[:256],
        padding.OAEP(mgf=padding.MGF1(hashes.SHA256()), algorithm=hashes.SHA256(), label=None),
    )
    relay.config = json.loads(AESGCM(key).decrypt(raw[256:268], raw[268:], None))
    from urllib.parse import urlsplit

    url = urlsplit(relay.config['endpoint'])
    if url.scheme not in {'http', 'https'} or url.username or url.password or url.fragment:
        raise ValueError('Expected a plain HTTP(S) Responses endpoint without embedded credentials')
    relay.private_key = None
    if ROOT.is_symlink():
        raise ValueError('Direct Judge capture directory cannot be a symlink')
    ROOT.mkdir(parents=True, exist_ok=True)
    ROOT.chmod(0o700)
    relay.snapshot()
    relay.session = aiohttp.ClientSession(
        timeout=aiohttp.ClientTimeout(total=relay.config['timeout']),
        read_bufsize=16 * 1024 * 1024,
    )
    app = web.Application(client_max_size=256 * 1024 * 1024)
    app.router.add_post('/{path:.*}', relay.handle)
    relay.runner = web.AppRunner(app, access_log=None)
    await relay.runner.setup()
    site = web.TCPSite(relay.runner, '127.0.0.1', 0)
    await site.start()
    port = relay.runner.addresses[0][1]
    return {'base_url': f'http://127.0.0.1:{port}/v1'}


async def stop(identifier: str) -> bool:
    relay = _RELAYS.pop(identifier, None)
    if relay is None:
        return False
    for task in list(relay.pending):
        task.cancel()
    await asyncio.gather(*relay.pending, return_exceptions=True)
    if relay.runner:
        await relay.runner.cleanup()
    if relay.session:
        await relay.session.close()
    relay.config.clear()
    return True


def read_capture(verifier_dir: str) -> dict[str, Any]:
    root = Path(verifier_dir) / 'direct_responses'
    result: dict[str, Any] = {'mode': 'runtime_direct', 'requests': []}
    for path in sorted(root.glob('*.request.json')):
        prefix = path.name.split('.')[0]
        entry: dict[str, Any] = {'request': json.loads(path.read_text()), 'events': []}
        events = root / f'{prefix}.events.jsonl'
        if events.exists():
            for line in events.read_text().splitlines():
                with contextlib.suppress(ValueError):
                    entry['events'].append(json.loads(line))
        response = root / f'{prefix}.response.json'
        if response.exists():
            entry['response'] = json.loads(response.read_text())
        entry['errors'] = [json.loads(p.read_text()) for p in sorted(root.glob(f'{prefix}.attempt-*.error.json'))]
        result['requests'].append(entry)
    status = root / 'status.json'
    if status.exists():
        result['status'] = json.loads(status.read_text())
    usage = {'input_tokens': 0, 'output_tokens': 0, 'total_tokens': 0}
    for entry in result['requests']:
        response = entry.get('response')
        for event in entry['events']:
            if event.get('type') in {'response.completed', 'response.incomplete', 'response.failed'}:
                response = event.get('response')
        if isinstance(response, dict):
            for key in usage:
                usage[key] += (response.get('usage') or {}).get(key, 0) or 0
    result['usage'] = usage
    return result
