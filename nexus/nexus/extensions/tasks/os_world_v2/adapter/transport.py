"""V2-only host transport into one AGS instance; no evaluator policy."""

import asyncio
import base64
import json
import logging
import secrets
import shlex

from pathlib import Path
from typing import Any
from urllib.parse import urlsplit

import aiohttp

from aiohttp import web

from nexus.runtime.providers.ags.cdp_relay import CDPRelay


HOP = {'host', 'connection', 'transfer-encoding', 'keep-alive', 'proxy-authenticate', 'proxy-authorization', 'upgrade'}


def local_gitlab_origin(environment, name='GITLAB_URL'):
    """Only opt in to guest-local GitLab; remote deployments keep direct access."""
    value = environment.get(name, '').strip()
    if not value:
        return None
    url = urlsplit(value)
    if url.scheme not in {'http', 'https'} or not url.hostname or url.username or url.password:
        raise ValueError(f'{name} must be an absolute HTTP(S) URL without embedded credentials')
    if url.query or url.fragment:
        raise ValueError(f'{name} must not contain a query or fragment')
    if url.hostname not in {'localhost', '127.0.0.1', '::1'} and not (
        name == 'GITLAB_PAGES_URL' and url.hostname.endswith('.localhost')
    ):
        return None
    return (url.scheme, url.hostname, url.port or (443 if url.scheme == 'https' else 80))


def origin(url):
    return (url.scheme, url.hostname, url.port or (443 if url.scheme == 'https' else 80))


WEBSITE_REQUEST = r"""
import base64,json,requests,sys,time
s=json.load(sys.stdin)
session=requests.Session()
session.trust_env=False
for attempt in range(3):
 try:
  r=session.request(s["method"],s["url"],headers=s["headers"],
   data=base64.b64decode(s["body"]),allow_redirects=False,verify=False,timeout=s["timeout"])
 except requests.RequestException as exc:
  if s["method"] not in {"GET","HEAD"} or attempt==2:
   print("NEXUS_WEBSITE_ERROR="+type(exc).__name__)
   sys.exit(1)
 else:
  if s["method"] not in {"GET","HEAD"} or r.status_code not in {408,429,500,502,503,504} or attempt==2:
   break
 time.sleep(attempt+1)
print("NEXUS_WEBSITE_RESPONSE="+json.dumps({"status":r.status_code,
 "headers":dict(r.headers),"body":base64.b64encode(r.content).decode()}))
"""


class Transport:
    def __init__(
        self,
        runtime,
        sites: list[str],
        suffix: str,
        *,
        gitlab_origin=None,
        gitlab_pages_origin=None,
        diagnostic_directory=None,
    ):
        self.runtime = runtime
        self.sites = {f'{site}.{suffix}' for site in sites}
        self.gitlab_origins = {value for value in (gitlab_origin, gitlab_pages_origin) if value is not None}
        self.host = None
        self.runner = None
        self.session = None
        self.cdp_relay = None
        self.cdp_endpoint = None
        self.cdp_lock = asyncio.Lock()
        self.diagnostic_directory = diagnostic_directory

    def _cdp_diagnostic(self, record):
        if self.diagnostic_directory is not None:
            path = Path(self.diagnostic_directory) / 'cdp-relay.jsonl'
            with path.open('a') as stream:
                stream.write(json.dumps(record) + '\n')

    async def __aenter__(self):
        # Honor approved host egress; stream bytes without transparently changing
        # Content-Encoding/Content-Length on uploads or downloads.
        self.session = aiohttp.ClientSession(
            trust_env=True, auto_decompress=False, timeout=aiohttp.ClientTimeout(total=600, sock_connect=30)
        )
        for _ in range(20):
            host = f'127.{secrets.randbelow(200) + 20}.{secrets.randbelow(254)}.{secrets.randbelow(253) + 1}'
            app = web.Application(client_max_size=1024**3)
            app.router.add_post('/_nexus/website', self.website)
            app.router.add_route('*', '/{path:.*}', self.proxy)
            runner = web.AppRunner(app)
            await runner.setup()
            try:
                for port in (5000, 9222, 8080, 8000, 5910, 3000):
                    await web.TCPSite(runner, host, port).start()
            except OSError:
                await runner.cleanup()
                continue
            self.runner, self.host = runner, host
            return self
        await self.session.close()
        raise RuntimeError('Cannot allocate V2.1 loopback transport')

    async def __aexit__(self, *args):
        if self.cdp_relay:
            await self.cdp_relay.close()
        if self.runner:
            await self.runner.cleanup()
        if self.session:
            await self.session.close()

    async def website(self, request):
        spec = await request.json()
        url = urlsplit(spec['url'])
        allowed = url.hostname in self.sites or origin(url) in self.gitlab_origins
        if not allowed or url.scheme not in {'http', 'https'} or url.username or url.password:
            raise web.HTTPBadRequest(text='Undeclared website origin')
        spec['timeout'] = min(float(spec.get('timeout') or 30), 120)
        # AGS implements stdin using a shell heredoc, also subject to MAX_ARG_STRLEN.
        # Large states must travel via the upload endpoint, not /execute argv.
        command = 'python3 -c ' + shlex.quote(WEBSITE_REQUEST)
        attempts = 3 if spec['method'] in {'GET', 'HEAD'} else 1
        payload = json.dumps(spec)
        path = None
        try:
            if len(payload.encode()) > 32 * 1024:
                path = '/tmp/nexus-v2-website-' + secrets.token_hex(16) + '.json'
                created = await self.runtime.run_command(command=f'umask 077; : > {path}', timeout=10)
                if created.return_code != 0:
                    raise web.HTTPBadGateway(text='Cannot stage private guest website request')
                upload = await self.runtime.setup_upload_bytes(path, payload.encode(), filename='request.json')
                if upload.status_code >= 400:
                    raise web.HTTPBadGateway(text=f'Guest website request upload failed: HTTP {upload.status_code}')
                response = await self.runtime.run_command(
                    command=f'chmod 600 {path} && {command} < {path}', timeout=attempts * spec['timeout'] + 20
                )
            else:
                response = await self.runtime.run_command(
                    command=command, stdin=payload, timeout=attempts * spec['timeout'] + 20
                )
        finally:
            if path is not None:
                await self.runtime.run_command(command=f'rm -f {path}', timeout=10)
        if response.return_code != 0:
            logging.getLogger(__name__).warning(
                'Guest website request failed host=%s rc=%s stderr=%s',
                url.hostname,
                response.return_code,
                'ConnectionError' if 'ConnectionError' in (response.stderr or '') else 'guest command failed',
            )
            errors = [line for line in (response.stdout or '').splitlines() if line.startswith('NEXUS_WEBSITE_ERROR=')]
            detail = errors[-1].split('=', 1)[-1] if errors else 'GuestCommandError'
            raise web.HTTPBadGateway(text=f'Guest website transport failed: host={url.hostname} error={detail[:80]}')
        prefix = 'NEXUS_WEBSITE_RESPONSE='
        rows = [line[len(prefix) :] for line in (response.stdout or '').splitlines() if line.startswith(prefix)]
        if len(rows) != 1:
            raise web.HTTPBadGateway(text='Invalid guest website response')
        return web.json_response(json.loads(rows[0]))

    async def proxy(self, request):
        assert self.session is not None
        port = request.transport.get_extra_info('sockname')[1]
        if (
            port == 5000
            and request.path == '/setup/upload'
            and request.method == 'POST'
            and (request.content_length is None or request.content_length > 16 * 1024**2)
        ):
            from nexus.extensions.tasks.os_world_v2.adapter.uploads import upload_large_multipart

            return await upload_large_multipart(self.runtime, request)
        url = self.runtime.handle.url(port, request.rel_url.path_qs)
        headers = {key: value for key, value in request.headers.items() if key.lower() not in HOP}
        headers.update(self.runtime.handle.headers)
        if port == 9222 and request.path.startswith('/json/version'):
            # Same relay as V1: fresh browser endpoint per connection, proxy-aware
            # WebSockets, no 4 MiB frame ceiling, deterministic cancellation.
            async with self.cdp_lock:
                if self.cdp_relay is None:
                    # The image TCP proxy retains a 10s upstream socket timeout.
                    # Ping/pong keeps that transport alive without restarting Chrome.
                    self.cdp_relay = CDPRelay(
                        self.runtime.handle.url(port),
                        self.runtime.handle.headers,
                        heartbeat=3,
                        diagnostic_callback=self._cdp_diagnostic,
                    )
                    self.cdp_endpoint = await self.cdp_relay.start()
            return web.json_response({'webSocketDebuggerUrl': self.cdp_endpoint})
        async with self.session.request(
            request.method,
            url,
            data=request.content.iter_chunked(256 * 1024),
            headers=headers,
            allow_redirects=False,
        ) as response:
            outgoing = {key: value for key, value in response.headers.items() if key.lower() not in HOP}
            if response.status >= 400:
                logging.getLogger(__name__).warning(
                    'V2 guest HTTP failed method=%s port=%s path=%s status=%s',
                    request.method,
                    port,
                    request.path,
                    response.status,
                )
            downstream = web.StreamResponse(status=response.status, headers=outgoing)
            await downstream.prepare(request)
            async for chunk in response.content.iter_chunked(256 * 1024):
                await downstream.write(chunk)
            await downstream.write_eof()
            return downstream


def install_requests_transport(
    proxy_host: str,
    sites: list[str],
    suffix: str,
    compatibility,
    *,
    gitlab_origin=None,
    gitlab_pages_origin=None,
    task_id=None,
):
    """Worker-only requests adapter. Keep original website URLs/cookies in official code."""
    from http.cookies import SimpleCookie

    import requests

    original = requests.Session.send
    declared = {f'{site}.{suffix}' for site in sites}
    gitlab_origins = {value for value in (gitlab_origin, gitlab_pages_origin) if value is not None}
    gateway = requests.Session()
    gateway.trust_env = False
    failures = {}

    def send(session, request, **kwargs):
        parts = urlsplit(request.url)
        if task_id == '048' and parts.hostname == proxy_host and parts.port == 5000 and parts.path == '/setup/execute':
            from nexus.extensions.tasks.os_world_v2.adapter.task048 import rewrite_setup

            body = request.body.encode() if isinstance(request.body, str) else request.body
            request.body = rewrite_setup(body)
            request.headers['Content-Length'] = str(len(request.body))
        if parts.hostname in declared or origin(parts) in gitlab_origins:
            body = request.body or b''
            if isinstance(body, str):
                body = body.encode()
            spec: dict[str, Any] = {
                'url': request.url,
                'method': request.method,
                'headers': dict(request.headers),
                'body': base64.b64encode(body).decode(),
                'timeout': kwargs.get('timeout') if isinstance(kwargs.get('timeout'), (float, int)) else 30,
            }
            timeout = float(spec['timeout'])
            prepared = requests.Request('POST', f'http://{proxy_host}:5000/_nexus/website', json=spec).prepare()
            key = (request.method, parts.hostname, parts.path)
            try:
                attempts = 3 if request.method in {'GET', 'HEAD'} else 1
                wire = original(gateway, prepared, timeout=attempts * timeout + 30, allow_redirects=False)
                if wire.status_code >= 400:
                    # Include a bounded gateway diagnostic, not cookies, payload or query.
                    raise requests.HTTPError(
                        f'Guest website {request.method} host={parts.hostname} path={parts.path} '
                        f'gateway HTTP {wire.status_code}: {wire.text[:250]}',
                        response=wire,
                    )
                data = wire.json()
                response = requests.Response()
                response.status_code = data['status']
                response._content = base64.b64decode(data['body'])
                response._content_consumed = True
                # requests in the guest already decoded the website content.
                response.headers.update({
                    k: v
                    for k, v in data['headers'].items()
                    if k.lower() not in HOP | {'content-length', 'content-encoding'}
                })
                response.url, response.request = request.url, request
                cookies = SimpleCookie()
                cookies.load(response.headers.get('Set-Cookie', ''))
                for name, morsel in cookies.items():
                    response.cookies.set(name, morsel.value, domain=parts.hostname, path=morsel['path'] or '/')
                session.cookies.update(response.cookies)
                if response.status_code >= 500 or response.status_code in {401, 403, 408, 429}:
                    failures[key] = response.status_code
                else:
                    failures.pop(key, None)
                return response
            except Exception:
                failures[key] = 'transport'
                raise
        if (
            parts.hostname == proxy_host
            and parts.port == 5000
            and parts.path in {'/setup/execute', '/setup/launch'}
            and isinstance(request.body, (str, bytes))
            and compatibility[0] is not None
        ):
            body = request.body.encode() if isinstance(request.body, str) else request.body
            result = compatibility[0].rewrite(path=parts.path, body=body)
            if result.synthetic_status is not None:
                response = requests.Response()
                response.status_code = result.synthetic_status
                response._content = result.synthetic_text.encode()
                response._content_consumed = True
                response.url, response.request = request.url, request
                return response
            if result.body is not None:
                request.body = result.body
                request.headers['Content-Length'] = str(len(result.body))
        return original(session, request, **kwargs)

    for name, method in {'send': send}.items():
        setattr(requests.Session, name, method)
    return failures
