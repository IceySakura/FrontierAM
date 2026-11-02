"""Per-run staging and artifact-service transfers for OSWorld."""

from __future__ import annotations

import asyncio
import json
import logging
import tarfile
import tempfile
import uuid

from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from nexus.utils.paths import NEXUS_HOME
from nexus.utils.timeout import async_timeout


logger = logging.getLogger(__name__)
PUBLISH_RETRIES = 3


class RunArtifacts:
    """Unique local staging per attempt; failed uploads remain recoverable."""

    def __init__(self, *, taihu_pat: str | None = None, service_url: str | None = None):
        self._taihu_pat = taihu_pat
        self._service_url = service_url
        self.remote: list[dict[str, Any]] = []
        self.version = f'run-{uuid.uuid4().hex}'
        parent = NEXUS_HOME / 'osworld-artifacts'
        parent.mkdir(parents=True, exist_ok=True)
        self.root = Path(tempfile.mkdtemp(prefix=self.version + '-', dir=parent))
        self.files = self.root / 'files'
        self.files.mkdir(parents=True, mode=0o700)
        self.reference: dict[str, Any] = {
            'status': 'pending',
            'local_directory': str(self.root),
            'uri': None,
            'upload_failed': False,
            'attempts': 0,
            'uris': [],
        }

    def finish(self, *, task_id: str, run_id: str | None, result: Any = None, error: BaseException | None = None):
        # Do not serialize task_config, model credentials or ambient environment.
        payload = result.model_dump(mode='json', fallback=str) if hasattr(result, 'model_dump') else result
        summary = {
            'task_id': task_id,
            'run_id': run_id,
            'success': error is None,
            'result': payload,
            'error': {'type': type(error).__name__, 'message': str(error)} if error is not None else None,
        }
        (self.files / 'run-result.json').write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding='utf-8')
        (self.files / 'sandbox-artifacts.json').write_text(
            json.dumps(
                [{key: value for key, value in entry.items() if key != 'runtime'} for entry in self.remote],
                indent=2,
            ),
            encoding='utf-8',
        )
        self._save_reference()

    async def publish(self, *, timeout: float = 300) -> dict[str, Any]:
        from nexus.utils.artifact import ArtifactClient
        from nexus.utils.artifact.uploader import RunFolder, _ensure_date_folder, artifact_uri_for, folder_uri

        archive = self.root / 'host.tar.gz'
        # A timeout cannot stop a worker thread. Re-await the same packaging
        # task on retry, rather than racing two writers against the same .part.
        packaging = asyncio.create_task(asyncio.to_thread(_pack_host_files, self.files, archive))
        folder = None
        now = datetime.now(timezone.utc)
        self.reference['sandbox'] = [
            {key: value for key, value in entry.items() if key != 'runtime'} for entry in self.remote
        ]
        for attempt in range(1, PUBLISH_RETRIES + 2):
            self.reference['attempts'] = attempt
            try:
                async with async_timeout(timeout):
                    if not archive.exists():
                        await asyncio.shield(packaging)
                    # PAT/OAuth resolution and control-plane calls stay on the host.
                    async with ArtifactClient(taihu_pat=self._taihu_pat, base_url=self._service_url) as client:
                        if folder is None:
                            name = 'osworld-' + self.version
                            owner = await client.whoami()
                            parent_id = await _ensure_date_folder(client, now)
                            created = await client.create_folder(name, parent_folder_id=parent_id, idempotency_key=name)
                            folder = RunFolder(created.folder_id, f'/tasks/{now:%Y/%m/%d}/{name}', owner)
                            self.reference['uri'] = folder_uri(folder)
                        uri = artifact_uri_for(folder, archive.name)
                        if uri not in self.reference['uris']:
                            info = await client.upload_file(
                                archive,
                                parent_folder_id=folder.folder_id,
                                idempotency_key=self.version + ':host',
                                follow_symlinks=False,
                            )
                            if info.status != 'ready':
                                raise RuntimeError('Host artifact is not ready')
                            self.reference['uris'].append(uri)
                        for entry in self.remote:
                            if entry['status'] == 'snapshot_failed':
                                continue
                            uri = artifact_uri_for(folder, entry['name'])
                            if uri not in self.reference['uris']:
                                await _publish_remote(client, folder.folder_id, entry, self.version)
                                self.reference['uris'].append(uri)
                    self.reference['sandbox'] = [
                        {key: value for key, value in entry.items() if key not in {'runtime', 'local_archive'}}
                        for entry in self.remote
                    ]
                    self.reference.update(
                        status='published',
                        upload_failed=any(entry.get('status') == 'snapshot_failed' for entry in self.remote),
                    )
                    if self.reference['upload_failed']:
                        self.reference['status'] = 'partial'
                    self._save_reference_safely()
                    return self.reference
            except asyncio.CancelledError:
                raise
            except Exception as exc:
                # Never log exceptions containing pre-signed URLs or auth headers.
                logger.warning('OSWorld artifact attempt %d failed (%s)', attempt, type(exc).__name__)
            if attempt <= PUBLISH_RETRIES:
                await asyncio.sleep(min(2**attempt, 8))
        self.reference.update(status='upload_failed', upload_failed=True)
        self._save_reference_safely()
        return self.reference

    def _save_reference_safely(self):
        try:
            self._save_reference()
        except OSError:
            logger.warning('Unable to save artifact publication status at %s', self.root, exc_info=True)

    async def publish_run_output_details(self, output: Any) -> dict[str, Any]:
        """Archive late processor/runner data without changing the published task archive."""
        late = RunArtifacts(taihu_pat=self._taihu_pat, service_url=self._service_url)
        try:
            late.finish(task_id='run-output-details', run_id=None, result=output)
            return await late.publish()
        except asyncio.CancelledError:
            raise
        except Exception:
            late.reference.update(status='upload_failed', upload_failed=True)
            late._save_reference_safely()
            logger.warning('Unable to publish runner details; retained at %s', late.root)
            return late.reference

    def _save_reference(self):
        (self.root / 'publication.json').write_text(json.dumps(self.reference, indent=2), encoding='utf-8')


def _pack_host_files(source: Path, destination: Path) -> None:
    """Do not follow links or include sockets/devices in an output archive."""
    import os
    import stat

    partial = destination.with_suffix('.part')
    try:
        with tarfile.open(partial, 'w:gz', dereference=False) as archive:
            for directory, _, names in os.walk(source, followlinks=False):
                for name in sorted(names):
                    path = Path(directory) / name
                    if not path.is_file() or path.is_symlink():
                        continue
                    fd = os.open(path, os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK)
                    with os.fdopen(fd, 'rb') as stream:
                        metadata = os.fstat(stream.fileno())
                        if not stat.S_ISREG(metadata.st_mode):
                            continue
                        info = tarfile.TarInfo(path.relative_to(source).as_posix())
                        info.size = metadata.st_size
                        info.mode = 0o600
                        archive.addfile(info, stream)
        partial.replace(destination)
    finally:
        partial.unlink(missing_ok=True)


# Runs with the system interpreter: no nexus installation, PAT or COS key in
# the guest. The host sends only a single-object upload session after evaluation.
_GUEST_TRANSFER = r"""
import base64, hashlib, http.client, json, os, pathlib, signal, stat, sys, tarfile, time
from urllib.parse import urlsplit
spec_path = pathlib.Path(sys.argv[1])
spec = json.loads(spec_path.read_text())
deadline = time.monotonic() + 40
phase, http_status = "prepare", None
def expired(*args):
    raise TimeoutError("transfer deadline")
signal.signal(signal.SIGALRM, expired)
signal.alarm(45)
def check():
    if time.monotonic() >= deadline:
        raise TimeoutError("transfer deadline")
def digest(path):
    md5, sha = hashlib.md5(), hashlib.sha256()
    with open(path, "rb") as f:
        while chunk := f.read(1024 * 1024):
            check()
            md5.update(chunk)
            sha.update(chunk)
    return dict(size=path.stat().st_size, md5=base64.b64encode(md5.digest()).decode(), sha256=sha.hexdigest())
def put(url, headers, source, offset, length):
    global phase, http_status
    parsed = urlsplit(url)
    if parsed.scheme not in ("http", "https") or parsed.username or parsed.password:
        raise ValueError("invalid upload URL")
    conn_type = http.client.HTTPSConnection if parsed.scheme == "https" else http.client.HTTPConnection
    conn = conn_type(parsed.hostname, parsed.port, timeout=10)
    try:
        phase = "connect"
        target = parsed.path or "/"
        if parsed.query:
            target += "?" + parsed.query
        conn.putrequest("PUT", target, skip_host=any(k.lower() == "host" for k in headers))
        for key, value in headers.items():
            if key.lower() != "content-length":
                conn.putheader(key, value)
        conn.putheader("Content-Length", str(length))
        conn.endheaders()
        phase = "send"
        source.seek(offset)
        while length:
            check()
            chunk = source.read(min(length, 1024 * 1024))
            if not chunk:
                raise OSError("short upload source")
            conn.send(chunk)
            length -= len(chunk)
        response = conn.getresponse()
        phase, http_status = "response", response.status
        if not 200 <= response.status < 300:
            raise OSError("COS PUT rejected")
        return response.getheader("ETag")
    finally:
        conn.close()
try:
    destination = pathlib.Path(spec["archive"])
    if spec["action"] == "snapshot":
        # Exact diagnostic allowlist, never a recursive archive of $HOME or the
        # session directory (which contains configs and bridge credentials).
        temporary = destination.with_suffix(".part")
        missing = []
        with tarfile.open(temporary, "w:gz", dereference=False) as archive:
            for name, filename in spec["files"].items():
                check()
                try:
                    fd = os.open(filename, os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK)
                except FileNotFoundError:
                    missing.append(name)
                    continue
                with os.fdopen(fd, "rb") as source:
                    metadata = os.fstat(source.fileno())
                    if not stat.S_ISREG(metadata.st_mode):
                        raise ValueError("non-regular diagnostic")
                    info = tarfile.TarInfo(name)
                    info.size, info.mode = metadata.st_size, 0o600
                    archive.addfile(info, source)
        temporary.replace(destination)
        print(json.dumps(dict(**digest(destination), missing=missing)))
    else:
        if digest(destination) != spec["digest"]:
            raise ValueError("snapshot changed")
        session = spec["upload"]
        urls = session.get("part_urls") or [session["url"]]
        part_size = session.get("part_size") or spec["digest"]["size"]
        if len(urls) != (spec["digest"]["size"] + part_size - 1) // part_size:
            raise ValueError("invalid multipart layout")
        parts = []
        with open(destination, "rb") as source:
            for index, url in enumerate(urls):
                offset = index * part_size
                # Match ArtifactClient: multipart part URLs sign individual
                # PUTs, not the whole-object Content-MD5/content-type headers.
                headers = {} if session.get("part_urls") else session["headers"]
                etag = put(url, headers, source, offset,
                           min(part_size, spec["digest"]["size"] - offset))
                if session.get("part_urls"):
                    if not etag:
                        raise ValueError("missing part ETag")
                    parts.append(dict(part_number=index + 1, etag=etag))
        print(json.dumps(dict(parts=parts)))
except Exception as exc:
    # No signed URLs, HTTP bodies or credentials in error output.
    print(json.dumps(dict(error_type=type(exc).__name__, phase=phase, http_status=http_status,
                          errno=getattr(exc, "errno", None), verify_code=getattr(exc, "verify_code", None))))
    sys.exit(1)
finally:
    spec_path.unlink(missing_ok=True)
"""


async def _guest_command(runtime: Any, command: str, *, timeout: float = 50) -> str:
    response = await asyncio.wait_for(runtime.run_command(command=command, timeout=timeout), timeout=timeout + 1)
    if response.return_code != 0:
        try:
            evidence = json.loads(response.stdout or '')
            if isinstance(evidence, dict):
                # Only copy bounded identifiers/numbers, never arbitrary guest
                # stderr or HTTP response bodies (which may contain signed URLs).
                error_type = str(evidence.get('error_type', ''))
                phase = str(evidence.get('phase', ''))
                if error_type.isidentifier() and phase.isidentifier():
                    status = evidence.get('http_status')
                    number = evidence.get('errno')
                    logger.warning(
                        'OSWorld artifact helper: error_type=%s phase=%s http_status=%s errno=%s verify_code=%s',
                        error_type[:40],
                        phase[:40],
                        status if type(status) is int else None,
                        number if type(number) is int else None,
                        evidence.get('verify_code') if type(evidence.get('verify_code')) is int else None,
                    )
        except (ValueError, TypeError):
            pass
        raise RuntimeError('OSWorld artifact helper failed')
    return response.stdout or ''


async def _guest_transfer(runtime: Any, directory: str, spec: dict[str, Any]) -> dict[str, Any]:
    from nexus.utils.shell import quote_posix

    spec_path = directory + '/' + uuid.uuid4().hex + '.json'
    response = await asyncio.wait_for(
        runtime.setup_upload_bytes(spec_path, json.dumps(spec).encode(), filename='transfer.json'), timeout=10
    )
    if response.status_code != 200:
        raise RuntimeError('Unable to stage artifact transfer')
    output = await _guest_command(
        runtime,
        f'/usr/bin/python3 -I {quote_posix(directory + "/transfer.py")} {quote_posix(spec_path)}',
    )
    result = json.loads(output)
    if not isinstance(result, dict) or result.get('error_type'):
        raise RuntimeError('Artifact transfer did not complete')
    return result


async def snapshot_sandbox_artifacts(runtime: Any, files: dict[str, str], *, label: str) -> None:
    """Snapshot before harness cleanup; upload later without changing cleanup policy."""
    from nexus.core import require_context
    from nexus.utils.shell import quote_posix

    try:
        ctx = require_context()
    except RuntimeError:
        return
    stage = ctx.tool_state.get('osworld_artifact_publisher')
    if not isinstance(stage, RunArtifacts):
        return
    directory = '/tmp/nexus-output-' + uuid.uuid4().hex
    entry: dict[str, Any] = {
        'runtime': runtime,
        'directory': directory,
        'archive': directory + '/diagnostics.tar.gz',
        'name': label + '-' + uuid.uuid4().hex + '.tar.gz',
        'status': 'snapshot_failed',
        'local_archive': str(stage.root / (label + '-' + uuid.uuid4().hex + '.tar.gz')),
    }
    stage.remote.append(entry)
    try:
        async with async_timeout(60):
            await _guest_command(runtime, f'umask 077; mkdir -m 700 {quote_posix(directory)}', timeout=5)
            response = await asyncio.wait_for(
                runtime.setup_upload_bytes(
                    directory + '/transfer.py', _GUEST_TRANSFER.encode(), filename='transfer.py'
                ),
                timeout=10,
            )
            if response.status_code != 200:
                raise RuntimeError('Unable to stage artifact helper')
            metadata = await _guest_transfer(
                runtime, directory, {'action': 'snapshot', 'archive': entry['archive'], 'files': files}
            )
            import base64

            size = metadata.get('size')
            md5 = metadata.get('md5', '')
            sha256 = metadata.get('sha256', '')
            if (
                type(size) is not int
                or size <= 0
                or len(base64.b64decode(md5, validate=True)) != 16
                or len(bytes.fromhex(sha256)) != 32
                or not isinstance(metadata.get('missing'), list)
            ):
                raise ValueError('Invalid sandbox artifact metadata')
            entry.update(status='staged', size=size, md5=md5, sha256=sha256, missing=metadata['missing'])
    except asyncio.CancelledError:
        raise
    except Exception as exc:
        entry['error_type'] = type(exc).__name__
        logger.warning('OSWorld sandbox artifact snapshot failed (%s)', type(exc).__name__)


async def _publish_remote(client: Any, folder_id: str, entry: dict[str, Any], version: str) -> None:
    if entry['status'] == 'snapshot_failed':
        return
    session = await client.create_upload(
        filename=entry['name'],
        size_bytes=entry['size'],
        content_md5=entry['md5'],
        content_type='application/gzip',
        parent_folder_id=folder_id,
        idempotency_key=version + ':' + entry['name'],
    )
    if not session.is_ready:
        if session.upload_id is None or session.method not in (None, 'PUT'):
            raise RuntimeError('Unsupported artifact upload session')
        try:
            result = await _guest_transfer(
                entry['runtime'],
                entry['directory'],
                {
                    'action': 'upload',
                    'archive': entry['archive'],
                    'digest': {key: entry[key] for key in ('size', 'md5', 'sha256')},
                    'upload': {
                        'url': session.url,
                        'headers': session.headers,
                        'part_urls': session.part_urls,
                        'part_size': session.part_size,
                    },
                },
            )
        except asyncio.CancelledError:
            raise
        except Exception as exc:
            logger.warning(
                'OSWorld sandbox COS direct upload failed (%s); relaying the verified snapshot through host',
                type(exc).__name__,
            )
            local = await _relay_snapshot(entry)
            info = await client.upload_file(
                local,
                name=entry['name'],
                parent_folder_id=folder_id,
                idempotency_key=version + ':' + entry['name'],
                content_type='application/gzip',
                follow_symlinks=False,
            )
            entry['transfer'] = 'host_relay'
            if info.status != 'ready':
                raise RuntimeError('Relayed sandbox artifact is not ready') from exc
            return
        else:
            entry['transfer'] = 'sandbox_direct'
            await client.complete_upload(session.upload_id, parts=result['parts'] if session.is_multipart else None)
    info = await client.get_artifact(session.artifact_id)
    if info.status != 'ready':
        raise RuntimeError('Sandbox artifact is not ready')


async def _relay_snapshot(entry: dict[str, Any]) -> Path:
    """Bounded-memory fallback for COS endpoints not reachable from this AGS region."""
    import hashlib

    import aiohttp

    runtime = entry['runtime']
    destination = Path(entry['local_archive'])
    partial = destination.with_suffix('.part')
    digest = hashlib.sha256()
    size = 0
    try:
        async with aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=120)) as session:
            async with session.post(
                runtime.handle.url(runtime.server_port, '/file'),
                data={'file_path': entry['archive']},
                headers=runtime.handle.headers,
            ) as response:
                if response.status != 200:
                    raise RuntimeError(f'Sandbox snapshot download failed: HTTP {response.status}')
                with partial.open('wb') as stream:
                    async for chunk in response.content.iter_chunked(1024 * 1024):
                        size += len(chunk)
                        if size > entry['size']:
                            raise RuntimeError('Sandbox snapshot exceeds declared size')
                        digest.update(chunk)
                        stream.write(chunk)
        if size != entry['size'] or digest.hexdigest() != entry['sha256']:
            raise RuntimeError('Sandbox snapshot checksum mismatch')
        partial.replace(destination)
        return destination
    finally:
        partial.unlink(missing_ok=True)
