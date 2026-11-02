"""Detached OSWorld execution with short, read-only result polling.

Like RemoteRuntime's async invoke protocol, results survive a lost submission
response. This adapter uses the existing OSWorld /execute and /file endpoints
instead of requiring a Nexus server, Socket.IO dependencies or another open port.
"""

from __future__ import annotations

import asyncio
import base64
import json
import logging
import shlex
import uuid

from importlib.resources import files
from typing import Any
from urllib.parse import urlsplit

import aiohttp

from nexus.utils.async_http import http_client
from nexus.utils.timeout import async_timeout


logger = logging.getLogger(__name__)
_REQUEST_TIMEOUT = 10.0
_POLL_INTERVAL = 2.0


def build_launch_payload(job_id: str, spec: dict[str, Any]) -> dict[str, Any]:
    worker = files('nexus.runtime.providers.devcloud').joinpath('_execute_worker.py').read_text(encoding='utf-8')
    # mkdir is the claim: even duplicate delivery of this bootstrap cannot execute
    # the same job twice. A crashed bootstrap is not automatically relaunched.
    bootstrap = f"""
import base64, json, os, pathlib, subprocess, sys
os.umask(0o077)
directory = pathlib.Path('/tmp/nexus-devcloud-execute-{job_id}')
try:
    directory.mkdir(mode=0o700)
except FileExistsError:
    sys.exit(0)
(directory / 'spec.json').write_bytes(base64.b64decode({base64.b64encode(json.dumps(spec).encode()).decode()!r}))
(directory / 'worker.py').write_bytes(base64.b64decode({base64.b64encode(worker.encode()).decode()!r}))
(directory / 'state.tmp').write_text(json.dumps({{'job_id': {job_id!r}, 'state': 'pending'}}))
os.replace(directory / 'state.tmp', directory / 'state.json')
subprocess.Popen([sys.executable, str(directory / 'worker.py'), str(directory)],
                 stdin=subprocess.DEVNULL, stdout=subprocess.DEVNULL,
                 stderr=subprocess.DEVNULL, start_new_session=True, close_fds=True)
"""
    # No login shell: the launcher only needs the VM's Python stdlib.
    return {'command': ['bash', '-c', f'python3 -c {shlex.quote(bootstrap)}'], 'shell': False}


async def execute(
    base_url: str, payload: dict[str, Any], *, proxy_url: str | None, timeout: float
) -> tuple[Any, str, int]:
    job_id = uuid.uuid4().hex
    parsed = urlsplit(base_url)
    spec = {
        'job_id': job_id,
        'url': f'http://127.0.0.1:{parsed.port or 5000}/execute',
        'payload': payload,
        'timeout': timeout,
    }
    state_path = f'/tmp/nexus-devcloud-execute-{job_id}/state.json'
    deadline = asyncio.get_running_loop().time() + timeout

    def request_timeout() -> aiohttp.ClientTimeout:
        return aiohttp.ClientTimeout(
            total=max(0.001, min(_REQUEST_TIMEOUT, deadline - asyncio.get_running_loop().time()))
        )

    try:
        async with async_timeout(timeout):
            logger.info('DevCloud async execute submitting job %s', job_id)
            try:
                response = await http_client.post(
                    f'{base_url}/execute',
                    json=build_launch_payload(job_id, spec),
                    proxy=proxy_url,
                    timeout=request_timeout(),
                )
            except (asyncio.TimeoutError, aiohttp.ClientConnectionError):
                logger.warning('DevCloud async execute submission response lost for job %s; querying only', job_id)
            else:
                if 400 <= response.status_code < 500:
                    raise RuntimeError(f'DevCloud async execute launch rejected: HTTP {response.status_code}')
                if response.status_code == 200:
                    body = response.json()
                    if not isinstance(body, dict) or body.get('returncode') != 0:
                        raise RuntimeError(
                            f'DevCloud async execute bootstrap failed for job {job_id}: {response.text[:512]}'
                        )
                # A gateway 5xx may arrive after acceptance. Never resubmit the job.

            while True:
                try:
                    response = await http_client.post(
                        f'{base_url}/file',
                        data={'file_path': state_path},
                        proxy=proxy_url,
                        timeout=request_timeout(),
                    )
                except (asyncio.TimeoutError, aiohttp.ClientConnectionError):
                    pass
                else:
                    if response.status_code == 200:
                        try:
                            state = response.json()
                        except (ValueError, UnicodeDecodeError):
                            # A truncated /file response says nothing about command completion.
                            # Re-read this job only; replaying /execute could duplicate UI actions.
                            logger.warning('DevCloud async execute unreadable state for job %s; querying only', job_id)
                            await asyncio.sleep(_POLL_INTERVAL)
                            continue
                        if not isinstance(state, dict) or state.get('job_id') != job_id:
                            raise RuntimeError(f'DevCloud async execute invalid result identity for job {job_id}')
                        if state.get('state') == 'completed':
                            text = state['text']
                            try:
                                body = json.loads(text)
                            except ValueError:
                                body = None
                            return body, text, int(state['http_status'])
                        if state.get('state') == 'failed':
                            raise RuntimeError(f'DevCloud async execute job {job_id} failed: {state.get("error")}')
                        if state.get('state') not in {'pending', 'running'}:
                            raise RuntimeError(f'DevCloud async execute invalid state for job {job_id}')
                    elif response.status_code != 404 and response.status_code < 500:
                        raise RuntimeError(f'DevCloud async execute result query failed: HTTP {response.status_code}')
                await asyncio.sleep(_POLL_INTERVAL)
    except asyncio.TimeoutError as exc:
        raise TimeoutError(
            f'DevCloud async execute job {job_id} exceeded {timeout}s; command state unknown, not resubmitted'
        ) from exc
