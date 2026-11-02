from __future__ import annotations

import base64
import hashlib
import json
import uuid

from pathlib import Path
from types import SimpleNamespace
from typing import ClassVar
from unittest.mock import AsyncMock, Mock

import pytest

from nexus.core import TaskRunInput, TaskRunOutput, task_registry
from nexus.runner.hooks import run_hook_registry
from nexus.runner.hooks.base import RunHook
from nexus.runner.hooks.composed import ComposedHook
from nexus.runner.hooks.output_fetch import OutputFetchError, OutputFetchHook, fetch_runtime_outputs
from nexus.runner.single import RunHooks, RunInput, RunOutput, run
from nexus.runtime.executors.models.command import CommandRunStatus, RunCommandResponse
from nexus.runtime.providers.base import RuntimeProvider
from nexus.utils.artifact import ArtifactInfo, ArtifactUploadError


def _runtime_with_listing(paths: list[tuple[str, bytes]]):
    """A fake runtime whose listing reports *paths* with their real sha256/size."""
    rows = []
    for path, payload in paths:
        rows.append(f'{path}\t{len(payload)}\t{hashlib.sha256(payload).hexdigest()}')

    async def run_command(command, timeout):
        return RunCommandResponse(
            status=CommandRunStatus.Finished,
            stdout='\n'.join(rows),
        )

    async def download_file(files, **kwargs):
        target = files[0].path
        payload = next((p for path, p in paths if path == target), b'')
        return SimpleNamespace(
            errors=[],
            files=[SimpleNamespace(content=base64.b64encode(payload).decode('ascii'))],
        )

    return SimpleNamespace(run_command=run_command, download_file=download_file, base_url='http://sandbox')


def _provider(runtime):
    return SimpleNamespace(get_active_runtimes=lambda: {'main': runtime})


def _provider_runtimes(runtimes: dict):
    return SimpleNamespace(get_active_runtimes=lambda: runtimes)


@pytest.mark.asyncio
async def test_registry_exposes_the_hook():
    hook_cls = run_hook_registry.get('output_fetch')
    assert issubclass(hook_cls, RunHook)
    hook = hook_cls(dest_dir='/tmp/x', globs=['/tmp/work/*/out.pptx'])
    assert isinstance(hook, OutputFetchHook)


@pytest.mark.asyncio
async def test_fetches_files_and_manifest_records_sha256_match(tmp_path: Path):
    payload = b'PK\x03\x04 pptx bytes'
    runtime = _runtime_with_listing([('/tmp/work/a/out.pptx', payload)])
    manifest = await fetch_runtime_outputs(
        _provider(runtime),
        dest_dir=str(tmp_path),
        globs=['/tmp/work/*/out.pptx'],
        run_id='run-1',
    )
    local = tmp_path / 'out.pptx'
    assert local.read_bytes() == payload
    on_disk = json.loads((tmp_path / 'fetch_manifest.json').read_text(encoding='utf-8'))
    assert on_disk['run_id'] == 'run-1'
    assert [f['local_path'] for f in on_disk['files']] == [str(local)]
    assert on_disk['files'][0]['sha256_match'] is True
    assert manifest['files'][0]['sha256_match'] is True


@pytest.mark.asyncio
async def test_listing_timeout_is_an_error_not_a_zero_match(tmp_path: Path):
    """A non-Finished listing (e.g. TimeLimitExceeded) must surface as an
    error instead of being recorded as "no file matched" with a diagnostic
    pointing at the wrong cause. The error raises (the manifest is persisted
    first and travels on the exception), so the composed hook's
    required/optional policy can act on it."""

    async def run_command(command, timeout):
        return RunCommandResponse(status=CommandRunStatus.TimeLimitExceeded, stdout='')

    runtime = SimpleNamespace(run_command=run_command)
    with pytest.raises(OutputFetchError, match='TimeLimitExceeded') as exc_info:
        await fetch_runtime_outputs(
            _provider(runtime),
            dest_dir=str(tmp_path),
            globs=['/tmp/work/*/out.pptx'],
            run_id='run-2',
        )
    manifest = exc_info.value.manifest
    entry = manifest['runtimes'][0]
    assert 'list failed' in entry['error']
    assert 'TimeLimitExceeded' in entry['error']
    assert 'diagnostic' not in entry  # the wrong-cause listing was not taken
    assert manifest['files'] == []
    assert any('TimeLimitExceeded' in e for e in manifest['errors'])
    # the partial manifest is still persisted before the raise
    on_disk = json.loads((tmp_path / 'fetch_manifest.json').read_text(encoding='utf-8'))
    assert any('TimeLimitExceeded' in e for e in on_disk['errors'])


@pytest.mark.asyncio
async def test_rejected_glob_raises_after_manifest_saved(tmp_path: Path):
    """A config-supplied glob that cannot survive shell expansion is rejected
    with a raised error (not a silent zero-match return), and the manifest
    with the recorded cause is still on disk."""

    async def run_command(command, timeout):
        raise AssertionError('the listing command must not run for a rejected glob')

    runtime = SimpleNamespace(run_command=run_command)
    with pytest.raises(OutputFetchError, match='unsafe glob') as exc_info:
        await fetch_runtime_outputs(
            _provider(runtime),
            dest_dir=str(tmp_path),
            globs=['/tmp/work/*/out.pptx; rm -rf /'],
            run_id='run-5',
        )
    assert any('unsafe glob' in e for e in exc_info.value.manifest['errors'])
    on_disk = json.loads((tmp_path / 'fetch_manifest.json').read_text(encoding='utf-8'))
    assert any('unsafe glob' in e for e in on_disk['errors'])


@pytest.mark.asyncio
async def test_same_basename_files_are_disambiguated_not_overwritten(tmp_path: Path):
    first, second = b'first deck', b'second deck'
    runtime = _runtime_with_listing([
        ('/tmp/work/run-a/out.pptx', first),
        ('/tmp/work/run-b/out.pptx', second),
    ])
    manifest = await fetch_runtime_outputs(
        _provider(runtime),
        dest_dir=str(tmp_path),
        globs=['/tmp/work/*/out.pptx'],
    )
    # Both copies survive: the second is suffixed instead of overwriting the
    # first, and both verdicts describe the bytes actually on disk.
    assert (tmp_path / 'out.pptx').read_bytes() == first
    assert (tmp_path / 'out_2.pptx').read_bytes() == second
    disambiguated = [f for f in manifest['files'] if 'disambiguated' in f]
    assert len(disambiguated) == 1
    assert disambiguated[0]['disambiguated'] == 'out_2.pptx'
    assert all(f['sha256_match'] for f in manifest['files'])


@pytest.mark.asyncio
async def test_cross_runtime_same_basename_not_overwritten(tmp_path: Path):
    """With per_runtime_subdir=False every runtime writes the same base_dir,
    so the disambiguation claim set must be shared across runtimes -- a
    per-runtime counter resets and lets runtime 2's out.pptx overwrite
    runtime 1's while both manifest lines still claim sha256_match."""
    first = b'from runtime one'
    second = b'from runtime two'
    rt1 = _runtime_with_listing([('/tmp/work/a/out.pptx', first)])
    rt2 = _runtime_with_listing([('/tmp/work/b/out.pptx', second)])
    manifest = await fetch_runtime_outputs(
        _provider_runtimes({'rt-1': rt1, 'rt-2': rt2}),
        dest_dir=str(tmp_path),
        globs=['/tmp/work/*/out.pptx'],
    )
    assert (tmp_path / 'out.pptx').read_bytes() == first
    assert (tmp_path / 'out_2.pptx').read_bytes() == second
    assert len({f['local_path'] for f in manifest['files']}) == 2
    assert all(f['sha256_match'] for f in manifest['files'])


@pytest.mark.asyncio
async def test_disambiguation_suffix_collides_with_a_real_basename(tmp_path: Path):
    """A generated suffix can collide with a real basename in the same fetch:
    b's out.pptx is saved as out_2.pptx, and c's literal out_2.pptx must not
    land on that path -- the suffix has to keep bumping until the final path
    is actually free."""
    payloads = {
        '/tmp/work/a/out.pptx': b'deck a',
        '/tmp/work/b/out.pptx': b'deck b',
        '/tmp/work/c/out_2.pptx': b'deck c',
    }
    runtime = _runtime_with_listing(list(payloads.items()))
    manifest = await fetch_runtime_outputs(
        _provider(runtime),
        dest_dir=str(tmp_path),
        globs=['/tmp/work/*/*.pptx'],
    )
    assert (tmp_path / 'out.pptx').read_bytes() == b'deck a'
    assert (tmp_path / 'out_2.pptx').read_bytes() == b'deck b'
    assert (tmp_path / 'out_2_2.pptx').read_bytes() == b'deck c'
    by_remote = {f['remote_path']: f for f in manifest['files']}
    assert by_remote['/tmp/work/c/out_2.pptx']['disambiguated'] == 'out_2_2.pptx'
    assert all(f['sha256_match'] for f in manifest['files'])


@pytest.mark.asyncio
async def test_existing_file_on_disk_is_not_overwritten(tmp_path: Path):
    """The claim check also looks at the real filesystem, not just this
    fetch: whatever already sits in dest_dir (an earlier run without
    run_id_subdir) must survive a new fetch of the same basename."""
    (tmp_path / 'out.pptx').write_bytes(b'previous run')
    payload = b'new run'
    runtime = _runtime_with_listing([('/tmp/work/a/out.pptx', payload)])
    manifest = await fetch_runtime_outputs(
        _provider(runtime),
        dest_dir=str(tmp_path),
        globs=['/tmp/work/*/out.pptx'],
    )
    assert (tmp_path / 'out.pptx').read_bytes() == b'previous run'
    assert (tmp_path / 'out_2.pptx').read_bytes() == payload
    assert manifest['files'][0]['disambiguated'] == 'out_2.pptx'
    assert manifest['files'][0]['sha256_match'] is True


@pytest.mark.asyncio
async def test_run_id_subdir_namespaces_shared_dest_dir(tmp_path: Path):
    runtime = _runtime_with_listing([('/tmp/work/a/out.pptx', b'deck')])
    await fetch_runtime_outputs(
        _provider(runtime),
        dest_dir=str(tmp_path),
        globs=['/tmp/work/*/out.pptx'],
        run_id_subdir=True,
        run_id='run-42',
    )
    assert (tmp_path / 'run-42' / 'out.pptx').read_bytes() == b'deck'
    assert (tmp_path / 'run-42' / 'fetch_manifest.json').exists()


@pytest.mark.asyncio
async def test_hook_failure_propagates_and_keeps_partial_summary(tmp_path: Path):
    """A provider failure raises out of on_run_output (no swallow), and the
    partial fetch record is kept so get_result() still reports the errors."""

    def explode():
        raise RuntimeError('provider gone')

    hook = OutputFetchHook(dest_dir=str(tmp_path), globs=['/tmp/work/*/out.pptx'])
    ctx = SimpleNamespace(runtime_provider=SimpleNamespace(get_active_runtimes=explode), run_id='run-3')
    with pytest.raises(OutputFetchError, match='provider gone'):
        await hook.on_run_output(ctx, None)
    result = hook.get_result()
    assert result['errors']
    assert 'get_active_runtimes failed' in result['errors'][0]


@pytest.mark.asyncio
async def test_required_fetch_failure_reaches_composed_hook_channels(tmp_path: Path):
    """A required output_fetch failure must land in the standard failure
    channel: the composed hook records hook_errors, re-raises, and still
    collects the partial summary into hook_results at closing."""

    def explode():
        raise RuntimeError('provider gone')

    hook = OutputFetchHook(dest_dir=str(tmp_path), globs=['/tmp/work/*/out.pptx'])
    hook.required = True
    composed = ComposedHook({'output_fetch': hook})
    ctx = SimpleNamespace(runtime_provider=SimpleNamespace(get_active_runtimes=explode), run_id='run-6')
    output = RunOutput(success=True)

    await composed.on_context_ready(ctx)
    with pytest.raises(OutputFetchError):
        await composed.on_run_output(ctx, output)
    assert 'provider gone' in output.hook_errors['output_fetch.on_run_output']
    await composed.on_context_closing(ctx)

    summary = output.hook_results['output_fetch']
    assert 'get_active_runtimes failed' in summary['errors'][0]
    assert summary['n_files'] == 0


@pytest.mark.asyncio
async def test_optional_fetch_failure_records_hook_errors_without_raising(tmp_path: Path):
    """The same failure under required=False is recorded, not raised: the
    optional policy keeps the run going while hook_errors stays visible."""

    def explode():
        raise RuntimeError('provider gone')

    hook = OutputFetchHook(dest_dir=str(tmp_path), globs=['/tmp/work/*/out.pptx'])
    hook.required = False
    composed = ComposedHook({'output_fetch': hook})
    ctx = SimpleNamespace(runtime_provider=SimpleNamespace(get_active_runtimes=explode), run_id='run-7')
    output = RunOutput(success=True)

    await composed.on_context_ready(ctx)
    await composed.on_run_output(ctx, output)  # does not raise
    assert 'provider gone' in output.hook_errors['output_fetch.on_run_output']
    await composed.on_context_closing(ctx)
    assert output.hook_results['output_fetch']['errors']


@pytest.mark.asyncio
async def test_task_state_survives_required_output_fetch_failure(tmp_path: Path, monkeypatch):
    """End-to-end: a required output_fetch whose provider fails records the
    failure in hook_errors / hook_failures without turning the finished task
    into a failed run."""
    from nexus.runner import single

    events: list[str] = []
    task_name = f'fetch_task_{uuid.uuid4().hex}'

    @task_registry.register(task_name)
    async def _task(_input: TaskRunInput) -> TaskRunOutput:
        events.append('task')
        return TaskRunOutput()

    provider = Mock(spec=RuntimeProvider)
    provider.get_active_runtimes = Mock(side_effect=RuntimeError('provider exploded'))
    provider.release_all = AsyncMock()
    monkeypatch.setattr(single, '_create_runtime_provider', lambda _: provider)

    hooks = RunHooks()
    result = await run(
        RunInput(
            task_name=task_name,
            runner={
                'hooks': {
                    'output_fetch': {
                        'name': 'output_fetch',
                        'required': True,
                        'args': {'dest_dir': str(tmp_path), 'globs': ['/tmp/work/*/out.pptx']},
                    }
                }
            },
        ),
        hooks=hooks,
    )

    assert events == ['task']
    assert result.success is True  # the task itself finished and stays finished
    assert result.error is None
    assert 'provider exploded' in result.hook_errors['output_fetch.on_run_output']
    assert result.hook_results['output_fetch']['errors']
    assert len(hooks.hook_failures) == 1
    assert hooks.hook_failures[0].stage == 'run_output'
    assert isinstance(hooks.hook_failures[0].error, OutputFetchError)


@pytest.mark.asyncio
async def test_hook_collects_result_summary(tmp_path: Path):
    payload = b'deck bytes'
    runtime = _runtime_with_listing([('/tmp/work/a/out.pptx', payload)])
    hook = OutputFetchHook(dest_dir=str(tmp_path), globs=['/tmp/work/*/out.pptx'])
    ctx = SimpleNamespace(runtime_provider=_provider(runtime), run_id='run-4')
    await hook.on_run_output(ctx, None)
    assert hook.get_result() == {
        'dest_dir': str(tmp_path),
        'n_files': 1,
        'n_sha256_match': 1,
        'artifact_mode': 'disk_only',
        'artifacts': [],
        'errors': [],
    }


class _FakeArtifactClient:
    """Duck-typed stand-in for ``nexus.utils.artifact.ArtifactClient``.

    Patched over the hook module's ``ArtifactClient`` name by the
    ``fake_artifact_client`` fixture; ``upload_file`` records every call, and
    the class attributes let a test inject failures (``failure``) or a wrong
    returned size (``size_offset``) without any network.
    """

    instances: ClassVar[list[_FakeArtifactClient]] = []
    failure: Exception | None = None
    size_offset: int = 0
    folder_id: str = 'folder-123'

    def __init__(self) -> None:
        self.uploads: list[dict] = []
        self.resolved: tuple[str, str | None] | None = None
        self.closed = False
        type(self).instances.append(self)

    async def __aenter__(self) -> _FakeArtifactClient:
        return self

    async def __aexit__(self, *_exc: object) -> None:
        self.closed = True

    async def resolve_resource_id(self, uri_or_path: str, *, expect: str | None = None) -> str:
        self.resolved = (uri_or_path, expect)
        return type(self).folder_id

    async def upload_file(
        self,
        local_path,
        *,
        name=None,
        parent_folder_id=None,
        content_type=None,
        storage_pool=None,
        multipart=None,
        max_concurrency=5,
        idempotency_key=None,
        follow_symlinks=True,
        _identity=None,
    ):
        self.uploads.append({
            'local_path': str(local_path),
            'name': name,
            'parent_folder_id': parent_folder_id,
            'storage_pool': storage_pool,
            'idempotency_key': idempotency_key,
        })
        failure = type(self).failure
        if failure is not None:
            raise failure
        return ArtifactInfo(
            artifact_id=f'art_{len(self.uploads):03d}',
            name=name,
            size_bytes=Path(local_path).stat().st_size + type(self).size_offset,
            status='ready',
        )


@pytest.fixture
def fake_artifact_client(monkeypatch):
    """Point the hook's module at the fake client and provide credentials.

    The PAT probe is patched to a non-empty value, so ``artifact_upload=True``
    resolves to the active path; tests that want the disk-only path patch the
    probe themselves (see the without-credentials test).
    """
    from nexus.runner.hooks import output_fetch as output_fetch_module

    _FakeArtifactClient.instances = []
    _FakeArtifactClient.failure = None
    _FakeArtifactClient.size_offset = 0
    monkeypatch.setattr(output_fetch_module, 'ArtifactClient', _FakeArtifactClient)
    monkeypatch.setattr(output_fetch_module, 'load_taihu_pat', lambda: 'tai_pat_test')
    monkeypatch.delenv('ARTIFACT_DEV_EMPLOYEE', raising=False)
    yield _FakeArtifactClient
    _FakeArtifactClient.failure = None
    _FakeArtifactClient.size_offset = 0


def test_credentials_probe_follows_the_artifactclient_sources(monkeypatch):
    """The availability probe mirrors ArtifactClient's own credential sources
    (PAT env/key file, else the dev-only identity header) and nothing else:
    the Cloud execution ticket is not a source."""
    from nexus.runner.hooks import output_fetch as output_fetch_module

    probe = output_fetch_module._artifact_credentials_available
    monkeypatch.setattr(output_fetch_module, 'load_taihu_pat', lambda: None)
    monkeypatch.delenv('ARTIFACT_DEV_EMPLOYEE', raising=False)
    assert probe() is False
    monkeypatch.setenv('ARTIFACT_DEV_EMPLOYEE', 'dev@example.com')
    assert probe() is True
    monkeypatch.setattr(output_fetch_module, 'load_taihu_pat', lambda: 'tai_pat_x')
    assert probe() is True


def test_idempotency_key_stays_within_the_server_length_limit():
    """A long-but-legal file name must not overflow the artifact service's
    1-128 character Idempotency-Key validation: the (run, runtime, remote
    path, content) identity is hashed into a fixed-length key.  Regression for
    the review note where a 36-char UUID run id plus an 85-char name produced
    a 151-character key that the server rejected outright."""
    from nexus.runner.hooks.output_fetch import _artifact_idempotency_key

    run_id = '9f8b2c1a-1111-4def-8abc-0123456789ab'
    runtime = 'main'
    remote_path = '/tmp/work/' + 'a' * 80 + '.pptx'
    sandbox_sha = 'a' * 64
    key = _artifact_idempotency_key(run_id=run_id, runtime=runtime, remote_path=remote_path, sandbox_sha256=sandbox_sha)
    expected = 'task-output:' + hashlib.sha256(f'{run_id}:{runtime}:{remote_path}:{sandbox_sha}'.encode()).hexdigest()
    assert key == expected
    assert len(key) == len('task-output:') + 64
    assert len(key) <= 128


def test_idempotency_key_ignores_the_local_disambiguation_suffix():
    """A retried fetch re-disambiguates same-basename downloads locally, so the
    same (run, runtime, remote path, content) identity must keep one key no
    matter which local suffix the file landed under.  Keying on the local name
    would break a retry's reuse of the same server-side session exactly when
    a retry happens.  Regression for the review note that the key and the
    upload name disagreed on identity."""
    from nexus.runner.hooks.output_fetch import _artifact_idempotency_key

    kwargs = {
        'run_id': 'item-000000000001',
        'runtime': 'main',
        'remote_path': '/tmp/work/a/out.pptx',
        'sandbox_sha256': 'b' * 64,
    }
    key_one = _artifact_idempotency_key(**kwargs)
    key_two = _artifact_idempotency_key(**kwargs)
    assert key_one == key_two

    moved = dict(kwargs, remote_path='/tmp/work/other-dir/out.pptx')
    assert _artifact_idempotency_key(**moved) != key_one

    rewritten = dict(kwargs, runtime='verifier')
    assert _artifact_idempotency_key(**rewritten) != key_one

    different_content = dict(kwargs, sandbox_sha256='c' * 64)
    assert _artifact_idempotency_key(**different_content) != key_one


@pytest.mark.asyncio
async def test_two_runs_same_name_uploads_do_not_collide(tmp_path: Path, fake_artifact_client):
    """Two runs that each fetch and upload their own ``out.pptx`` must land
    at distinct stored names and idempotency keys.  The artifact service keys
    stored files by owner + parent folder + name, so identical basenames
    would make the second upload flip the first run's artifact to
    ``deleting``.  Regression for the review note where run ids sharing their
    first characters (``item-000000000001`` / ``item-000000000002``) collided
    under an id-prefix upload name: the digest now covers the complete run
    id, so both artifacts survive."""
    payload_a = b'PK\x03\x04 run-a deck'
    payload_b = b'PK\x03\x04 run-b deck'
    runtime_a = _runtime_with_listing([('/tmp/work/a/out.pptx', payload_a)])
    runtime_b = _runtime_with_listing([('/tmp/work/b/out.pptx', payload_b)])
    for run_id, dest, runtime in (
        ('item-000000000001', tmp_path / 'a', runtime_a),
        ('item-000000000002', tmp_path / 'b', runtime_b),
    ):
        await fetch_runtime_outputs(
            _provider(runtime),
            dest_dir=str(dest),
            globs=['/tmp/work/*/out.pptx'],
            run_id=run_id,
            artifact_upload=True,
        )
    uploads = [u for client in _FakeArtifactClient.instances for u in client.uploads]
    assert len(uploads) == 2
    names = [u['name'] for u in uploads]
    expected_a = hashlib.sha256(b'item-000000000001:main:/tmp/work/a/out.pptx').hexdigest()[:16] + '-out.pptx'
    expected_b = hashlib.sha256(b'item-000000000002:main:/tmp/work/b/out.pptx').hexdigest()[:16] + '-out.pptx'
    assert names == [expected_a, expected_b]
    assert names[0] != names[1]
    keys = [u['idempotency_key'] for u in uploads]
    assert keys[0] != keys[1]
    assert all(k.startswith('task-output:') for k in keys)


@pytest.mark.asyncio
async def test_same_run_two_runtimes_upload_names_do_not_collide(tmp_path: Path, fake_artifact_client):
    """One run whose runtimes each ship their own ``out.pptx`` (verifier
    alongside main, per_runtime_subdir=True) must keep both artifacts: the
    upload-name digest includes the runtime identity, so the two uploads land
    at distinct stored names instead of the second flipping the first to
    ``deleting``.  Regression for the review note where the run-id-only
    prefix made main's and verifier's uploads share one name."""
    payload_main = b'PK\x03\x04 main deck'
    payload_verifier = b'PK\x03\x04 verifier deck'
    manifest = await fetch_runtime_outputs(
        _provider_runtimes({
            'main': _runtime_with_listing([('/tmp/work/main/out.pptx', payload_main)]),
            'verifier': _runtime_with_listing([('/tmp/work/verifier/out.pptx', payload_verifier)]),
        }),
        dest_dir=str(tmp_path),
        globs=['/tmp/work/*/out.pptx'],
        run_id='run-x',
        per_runtime_subdir=True,
        artifact_upload=True,
    )
    uploads = [u for client in _FakeArtifactClient.instances for u in client.uploads]
    assert len(uploads) == 2
    names = [u['name'] for u in uploads]
    expected_main = hashlib.sha256(b'run-x:main:/tmp/work/main/out.pptx').hexdigest()[:16] + '-out.pptx'
    expected_verifier = hashlib.sha256(b'run-x:verifier:/tmp/work/verifier/out.pptx').hexdigest()[:16] + '-out.pptx'
    assert sorted(names) == sorted([expected_main, expected_verifier])
    assert names[0] != names[1]
    assert manifest['artifact_upload'] == {'mode': 'artifact_service', 'uploaded': 2, 'failed': 0}
    # both references are still downloadable objects, not deleting tombstones
    assert all(r['artifact']['status'] == 'ready' for r in manifest['files'])


def test_upload_name_respects_service_length_limit():
    """A legal-but-long file name must not overflow the artifact service's
    1-255 character name validation once the digest prefix is added.  A
    244-character basename collides with the old ``run-<8>-`` prefix at 257;
    the truncated stem plus the whole extension now stays under the limit
    while the digest keeps the identity unique."""
    from nexus.runner.hooks.output_fetch import (
        ARTIFACT_UPLOAD_NAME_LIMIT,
        _artifact_upload_name,
    )

    long_name = 'a' * 240 + '.pptx'
    assert len(long_name) > ARTIFACT_UPLOAD_NAME_LIMIT - 20
    remote_path = f'/tmp/work/a/{long_name}'
    upload_name = _artifact_upload_name(
        run_id='9f8b2c1a-1111-4def-8abc-0123456789ab',
        runtime='main',
        remote_path=remote_path,
    )
    digest = hashlib.sha256(f'9f8b2c1a-1111-4def-8abc-0123456789ab:main:{remote_path}'.encode()).hexdigest()[:16]
    assert len(upload_name) <= ARTIFACT_UPLOAD_NAME_LIMIT
    assert upload_name.startswith(f'{digest}-')
    assert upload_name.endswith('.pptx')
    # the truncated stem keeps as much of the readable name as the budget allows
    expected_stem = long_name[: ARTIFACT_UPLOAD_NAME_LIMIT - len(digest) - len('.pptx') - 1]
    assert upload_name == f'{digest}-{expected_stem}.pptx'
    # identity is still unique: a different run id cannot share the prefix
    other = _artifact_upload_name(
        run_id='9f8b2c1a-2222-4def-8abc-0123456789ab',
        runtime='main',
        remote_path=remote_path,
    )
    assert other != upload_name


@pytest.mark.asyncio
async def test_credentials_probe_failure_keeps_local_fetch_and_manifest(tmp_path: Path, monkeypatch):
    """A raising credential source (e.g. the PAT file misconfigured as a
    directory) must not abort the fetch: the local files and
    ``fetch_manifest.json`` still land, the probe failure is recorded as an
    error, and the standard required/optional failure semantics decide what
    the lost upload backend means for the run.  Regression for the review
    note where the probe ran before any manifest existed, so a broken PAT
    file cost the outputs for good once the sandbox was released."""
    from nexus.runner.hooks import output_fetch as output_fetch_module

    def broken_load():
        raise ValueError('PAT file is a directory')

    monkeypatch.setattr(output_fetch_module, 'load_taihu_pat', broken_load)
    monkeypatch.delenv('ARTIFACT_DEV_EMPLOYEE', raising=False)
    payload = b'PK\x03\x04 deck'
    runtime = _runtime_with_listing([('/tmp/work/a/out.pptx', payload)])
    with pytest.raises(OutputFetchError) as exc_info:
        await fetch_runtime_outputs(
            _provider(runtime),
            dest_dir=str(tmp_path),
            globs=['/tmp/work/*/out.pptx'],
            run_id='run-cred',
            artifact_upload=True,
        )
    manifest = exc_info.value.manifest
    assert manifest['artifact_upload']['mode'] == 'disk_only'
    assert 'credentials probe failed' in manifest['artifact_upload']['skipped']
    assert any('credentials probe failed' in e for e in manifest['errors'])
    # the outputs themselves survived: disk copy and manifest on disk
    assert (tmp_path / 'out.pptx').read_bytes() == payload
    on_disk = json.loads((tmp_path / 'fetch_manifest.json').read_text(encoding='utf-8'))
    assert any('credentials probe failed' in e for e in on_disk['errors'])


def test_upload_and_size_defaults_match_the_required_semantics():
    """artifact_upload stays opt-in (disk-only default) and the per-file fetch
    cap stays 64 MiB: one wide fetch must not be able to crowd worker memory
    nor the artifact service's shared bandwidth."""
    from nexus.runner.hooks.output_fetch import DEFAULT_MAX_FILE_BYTES, OutputFetchHook

    hook = OutputFetchHook(dest_dir='/tmp/x', globs=['/tmp/work/*/out.pptx'])
    assert hook.artifact_upload is False
    assert hook.artifact_folder is None
    assert hook.storage_pool is None
    assert DEFAULT_MAX_FILE_BYTES == 64 * 1024 * 1024


@pytest.mark.asyncio
async def test_upload_success_records_ref_in_manifest_and_result(tmp_path: Path, fake_artifact_client):
    """artifact_upload with credentials: the verified disk copy is uploaded by
    the hook through the standard ArtifactClient, with the sandbox sha pinned
    into the per-run idempotency key; the manifest and hook summary carry the
    returned artifact reference plus the fetch's own sha verification."""
    payload = b'PK\x03\x04 deck'
    runtime = _runtime_with_listing([('/tmp/work/a/out.pptx', payload)])
    sha = hashlib.sha256(payload).hexdigest()
    manifest = await fetch_runtime_outputs(
        _provider(runtime),
        dest_dir=str(tmp_path),
        globs=['/tmp/work/*/out.pptx'],
        run_id='run-u1',
        artifact_upload=True,
        artifact_folder='/runs/run-u1',
        storage_pool='pool-a',
    )
    client = _FakeArtifactClient.instances[0]
    assert client.resolved == ('/runs/run-u1', 'folder')
    expected_key = 'task-output:' + hashlib.sha256(f'run-u1:main:/tmp/work/a/out.pptx:{sha}'.encode()).hexdigest()
    expected_name = hashlib.sha256(b'run-u1:main:/tmp/work/a/out.pptx').hexdigest()[:16] + '-out.pptx'
    assert client.uploads == [
        {
            'local_path': str(tmp_path / 'out.pptx'),
            'name': expected_name,
            'parent_folder_id': 'folder-123',
            'storage_pool': 'pool-a',
            'idempotency_key': expected_key,
        }
    ]
    assert client.closed is True
    record = manifest['files'][0]
    assert record['artifact_uploaded'] is True
    assert record['artifact'] == {
        'artifact_id': 'art_001',
        'name': expected_name,
        'size_bytes': len(payload),
        'status': 'ready',
    }
    assert manifest['artifact_upload'] == {'mode': 'artifact_service', 'uploaded': 1, 'failed': 0}
    assert manifest['errors'] == []

    hook = OutputFetchHook(dest_dir=str(tmp_path), globs=['/tmp/work/*/out.pptx'], artifact_upload=True)
    hook._result = manifest
    assert hook.get_result() == {
        'dest_dir': str(tmp_path),
        'n_files': 1,
        'n_sha256_match': 1,
        'artifact_mode': 'artifact_service',
        'artifacts': [
            {
                'artifact_id': 'art_001',
                'name': expected_name,
                'size_bytes': len(payload),
                'status': 'ready',
                'sha256': sha,
                'sha256_match': True,
                'remote_path': '/tmp/work/a/out.pptx',
                'disambiguated': None,
            }
        ],
        'errors': [],
    }


@pytest.mark.asyncio
async def test_upload_request_without_credentials_stays_disk_only(tmp_path: Path, monkeypatch):
    """artifact_upload=True but the environment provides no Artifact
    credentials (no PAT, no dev identity): byte-identical disk-only behaviour,
    zero ArtifactClient construction, and the skip is recorded as a note --
    not an error, so nothing raises."""
    from nexus.runner.hooks import output_fetch as output_fetch_module

    monkeypatch.setattr(output_fetch_module, 'load_taihu_pat', lambda: None)
    monkeypatch.delenv('ARTIFACT_DEV_EMPLOYEE', raising=False)
    payload = b'deck'
    runtime = _runtime_with_listing([('/tmp/work/a/out.pptx', payload)])
    manifest = await fetch_runtime_outputs(
        _provider(runtime),
        dest_dir=str(tmp_path),
        globs=['/tmp/work/*/out.pptx'],
        run_id='run-u2',
        artifact_upload=True,
    )
    assert (tmp_path / 'out.pptx').read_bytes() == payload
    assert manifest['files'][0]['sha256_match'] is True
    assert 'artifact' not in manifest['files'][0]
    assert manifest['artifact_upload']['mode'] == 'disk_only'
    assert 'skipped' in manifest['artifact_upload']
    assert manifest['errors'] == []


@pytest.mark.asyncio
async def test_upload_failure_records_error_and_keeps_manifest(tmp_path: Path, fake_artifact_client):
    """A failed upload (ArtifactClient has exhausted its own transient
    retries) is a fetch error: it lands in manifest['errors'] so the hook's
    required/optional policy decides, while the disk copy and the on-disk
    manifest survive the failure."""
    fake_artifact_client.failure = ArtifactUploadError('upload failed (HTTP 503): cos exploded')
    payload = b'deck'
    runtime = _runtime_with_listing([('/tmp/work/a/out.pptx', payload)])
    with pytest.raises(OutputFetchError) as exc_info:
        await fetch_runtime_outputs(
            _provider(runtime),
            dest_dir=str(tmp_path),
            globs=['/tmp/work/*/out.pptx'],
            run_id='run-u3',
            artifact_upload=True,
        )
    manifest = exc_info.value.manifest
    record = manifest['files'][0]
    assert record['artifact_uploaded'] is False
    assert 'cos exploded' in record['upload_error']
    assert manifest['artifact_upload'] == {'mode': 'artifact_service', 'uploaded': 0, 'failed': 1}
    assert any('artifact upload' in e for e in manifest['errors'])
    # disk copy and manifest survive the upload failure
    assert (tmp_path / 'out.pptx').read_bytes() == payload
    on_disk = json.loads((tmp_path / 'fetch_manifest.json').read_text(encoding='utf-8'))
    assert on_disk['files'][0]['upload_error']


@pytest.mark.asyncio
async def test_returned_size_mismatch_is_an_upload_failure(tmp_path: Path, fake_artifact_client):
    """The hook asserts the returned ArtifactInfo against the verified size:
    a server answer describing different bytes is a failed upload, never a
    reported success."""
    fake_artifact_client.size_offset = 1
    payload = b'deck'
    runtime = _runtime_with_listing([('/tmp/work/a/out.pptx', payload)])
    with pytest.raises(OutputFetchError) as exc_info:
        await fetch_runtime_outputs(
            _provider(runtime),
            dest_dir=str(tmp_path),
            globs=['/tmp/work/*/out.pptx'],
            artifact_upload=True,
        )
    record = exc_info.value.manifest['files'][0]
    assert record['artifact_uploaded'] is False
    assert 'expected' in record['upload_error']
    assert (tmp_path / 'out.pptx').read_bytes() == payload


@pytest.mark.asyncio
async def test_sha_mismatch_copy_is_never_uploaded(tmp_path: Path, fake_artifact_client):
    """A file whose host copy does not match the sandbox sha must not be
    shipped anywhere; the skip is recorded as data, not an error."""
    payload = b'deck bytes'
    sandbox_sha = hashlib.sha256(b'completely different bytes').hexdigest()
    rows = [f'/tmp/work/a/out.pptx\t{len(payload)}\t{sandbox_sha}']

    async def run_command(command, timeout):
        return RunCommandResponse(status=CommandRunStatus.Finished, stdout='\n'.join(rows))

    async def download_file(files, **kwargs):
        return SimpleNamespace(
            errors=[],
            files=[SimpleNamespace(content=base64.b64encode(payload).decode('ascii'))],
        )

    runtime = SimpleNamespace(run_command=run_command, download_file=download_file)
    manifest = await fetch_runtime_outputs(
        _provider(runtime),
        dest_dir=str(tmp_path),
        globs=['/tmp/work/*/out.pptx'],
        artifact_upload=True,
    )
    assert _FakeArtifactClient.instances == []  # never constructed: nothing to upload
    record = manifest['files'][0]
    assert record['sha256_match'] is False
    assert record['artifact_uploaded'] is False
    assert 'sha256 mismatch' in record['upload_error']
    assert manifest['errors'] == []  # the verdict is data, not an error


@pytest.mark.asyncio
async def test_sha_mismatch_copy_is_quarantined_not_at_final_path(tmp_path: Path):
    """A mismatched copy must never occupy the standard final path.

    The download decoded fine but the bytes are not what the sandbox listed,
    so the file a path-reading consumer would pick up (fetch/out.pptx) must
    not hold them: only a verified copy may take the final name, and the
    mismatched bytes are quarantined under <stem>.sha_mismatch<ext>.
    """
    payload = b'corrupted transfer bytes'
    sandbox_sha = hashlib.sha256(b'the real sandbox bytes').hexdigest()
    rows = [f'/tmp/work/a/out.pptx\t{len(payload)}\t{sandbox_sha}']

    async def run_command(command, timeout):
        return RunCommandResponse(status=CommandRunStatus.Finished, stdout='\n'.join(rows))

    async def download_file(files, **kwargs):
        return SimpleNamespace(
            errors=[],
            files=[SimpleNamespace(content=base64.b64encode(payload).decode('ascii'))],
        )

    runtime = SimpleNamespace(run_command=run_command, download_file=download_file)
    manifest = await fetch_runtime_outputs(
        _provider(runtime),
        dest_dir=str(tmp_path),
        globs=['/tmp/work/*/out.pptx'],
    )
    assert not (tmp_path / 'out.pptx').exists()  # the standard name holds nothing wrong
    quarantined = tmp_path / 'out.sha_mismatch.pptx'
    assert quarantined.read_bytes() == payload
    assert not list(tmp_path.glob('*.fetching'))  # no temp leftovers
    record = manifest['files'][0]
    assert record['sha256_match'] is False
    assert record['local_path'] == str(quarantined)
    assert record['quarantined_as'] == 'out.sha_mismatch.pptx'
    assert manifest['errors'] == []  # the verdict is data, not an error


@pytest.mark.asyncio
async def test_sha_mismatch_and_verified_copy_coexist(tmp_path: Path):
    """A mismatch of one copy does not quarantine its verified sibling: the
    good copy still lands (on its disambiguated name, since the claim set
    reserved the standard name first), and each record tells the truth about
    its own bytes."""
    good = b'verified deck bytes'
    corrupted = b'corrupted bytes'
    sandbox_sha_of = {
        '/tmp/work/a/out.pptx': hashlib.sha256(b'not what arrived').hexdigest(),
        '/tmp/work/b/out.pptx': hashlib.sha256(good).hexdigest(),
    }
    payloads = {'/tmp/work/a/out.pptx': corrupted, '/tmp/work/b/out.pptx': good}
    rows = [f'{p}\t{len(payloads[p])}\t{sandbox_sha_of[p]}' for p in payloads]

    async def run_command(command, timeout):
        return RunCommandResponse(status=CommandRunStatus.Finished, stdout='\n'.join(rows))

    async def download_file(files, **kwargs):
        target = files[0].path
        return SimpleNamespace(
            errors=[],
            files=[SimpleNamespace(content=base64.b64encode(payloads[target]).decode('ascii'))],
        )

    runtime = SimpleNamespace(run_command=run_command, download_file=download_file)
    manifest = await fetch_runtime_outputs(
        _provider(runtime),
        dest_dir=str(tmp_path),
        globs=['/tmp/work/*/out.pptx'],
    )
    assert (tmp_path / 'out.sha_mismatch.pptx').read_bytes() == corrupted
    assert (tmp_path / 'out_2.pptx').read_bytes() == good
    by_remote = {f['remote_path']: f for f in manifest['files']}
    assert by_remote['/tmp/work/a/out.pptx']['sha256_match'] is False
    assert by_remote['/tmp/work/b/out.pptx']['sha256_match'] is True
    assert by_remote['/tmp/work/b/out.pptx']['local_path'] == str(tmp_path / 'out_2.pptx')


@pytest.mark.asyncio
async def test_oversize_file_skipped_before_download(tmp_path: Path):
    """A file whose sandbox-listed size exceeds max_file_bytes is skipped
    before download_file is ever called: a wide glob must not be able to pull
    an arbitrarily large file into worker memory. The skip is an error (the
    required/optional policy decides what it means for the run)."""
    payload = b'x' * 200
    listing_runtime = _runtime_with_listing([('/tmp/work/a/huge.bin', payload)])
    download = AsyncMock(side_effect=AssertionError('oversize file must not be downloaded'))
    runtime = SimpleNamespace(run_command=listing_runtime.run_command, download_file=download)
    with pytest.raises(OutputFetchError) as exc_info:
        await fetch_runtime_outputs(
            _provider(runtime),
            dest_dir=str(tmp_path),
            globs=['/tmp/work/*/huge.bin'],
            max_file_bytes=100,
        )
    download.assert_not_awaited()
    assert not (tmp_path / 'huge.bin').exists()
    manifest = exc_info.value.manifest
    assert any('max_file_bytes' in e for e in manifest['errors'])
    assert 'max_file_bytes' in manifest['files'][0]['error']
    # the manifest is still persisted before the raise
    on_disk = json.loads((tmp_path / 'fetch_manifest.json').read_text(encoding='utf-8'))
    assert any('max_file_bytes' in e for e in on_disk['errors'])


@pytest.mark.asyncio
async def test_file_grown_past_cap_after_download_is_skipped(tmp_path: Path):
    """The listing size passed the cap but the downloaded copy outgrew it:
    the decode guard catches it and the oversized bytes are never written."""
    payload = b'x' * 300
    rows = [f'/tmp/work/a/grew.bin\t10\t{hashlib.sha256(payload).hexdigest()}']  # listed size 10

    async def run_command(command, timeout):
        return RunCommandResponse(status=CommandRunStatus.Finished, stdout='\n'.join(rows))

    async def download_file(files, **kwargs):
        return SimpleNamespace(
            errors=[],
            files=[SimpleNamespace(content=base64.b64encode(payload).decode('ascii'))],
        )

    runtime = SimpleNamespace(run_command=run_command, download_file=download_file)
    with pytest.raises(OutputFetchError) as exc_info:
        await fetch_runtime_outputs(
            _provider(runtime),
            dest_dir=str(tmp_path),
            globs=['/tmp/work/*/grew.bin'],
            max_file_bytes=100,
        )
    assert not (tmp_path / 'grew.bin').exists()
    assert any('max_file_bytes' in e for e in exc_info.value.manifest['errors'])


@pytest.mark.asyncio
async def test_max_files_caps_fetch_with_skip_error(tmp_path: Path):
    """max_files bounds the number of downloads per fetch (across runtimes);
    excess matches are skipped with a recorded error instead of silently
    flooding dest_dir."""
    paths = [(f'/tmp/work/a/out_{i}.pptx', f'deck {i}'.encode()) for i in range(3)]
    runtime = _runtime_with_listing(paths)
    with pytest.raises(OutputFetchError) as exc_info:
        await fetch_runtime_outputs(
            _provider(runtime),
            dest_dir=str(tmp_path),
            globs=['/tmp/work/a/*.pptx'],
            max_files=2,
        )
    manifest = exc_info.value.manifest
    written = sorted(p.name for p in tmp_path.glob('*.pptx'))
    assert written == ['out_0.pptx', 'out_1.pptx']
    assert any('max_files' in e for e in manifest['errors'])
    skipped = [f for f in manifest['files'] if 'error' in f]
    assert len(skipped) == 1
    assert 'max_files' in skipped[0]['error']


@pytest.mark.asyncio
async def test_sanitize_maps_dot_and_dotdot(tmp_path: Path):
    """``.`` and ``..`` must not pass through as path components: joined into
    a path they resolve to the current / parent directory and would let a
    hostile run_id or runtime name write outside dest_dir. They map to
    underscore-prefixed literal names instead."""
    from nexus.runner.hooks.output_fetch import _sanitize

    assert _sanitize('..') == '_..'
    assert _sanitize('.') == '_.'
    assert _sanitize('') == 'runtime'
    assert _sanitize('a/b') == 'a_b'
    assert _sanitize('normal-runtime') == 'normal-runtime'


@pytest.mark.asyncio
async def test_dotdot_run_id_and_runtime_stay_inside_dest_dir(tmp_path: Path):
    """End to end: run_id='..' with run_id_subdir and a runtime literally
    named '..' with per_runtime_subdir both land in '_..' directories inside
    dest_dir -- nothing is written outside it."""
    payload = b'deck'
    runtime = _runtime_with_listing([('/tmp/work/a/out.pptx', payload)])
    outside = tmp_path.parent
    before = set(outside.iterdir())
    await fetch_runtime_outputs(
        _provider_runtimes({'..': runtime}),
        dest_dir=str(tmp_path),
        globs=['/tmp/work/*/out.pptx'],
        run_id_subdir=True,
        run_id='..',
        per_runtime_subdir=True,
    )
    assert (tmp_path / '_..' / '_..' / 'out.pptx').read_bytes() == payload
    assert set(outside.iterdir()) == before  # no escape into the parent


@pytest.mark.asyncio
async def test_unwritable_out_dir_keeps_manifest_first_contract(tmp_path: Path):
    """os.makedirs failing (unwritable dest_dir, full disk) must not escape
    as a raw OSError: it would skip _finalize() and lose both the on-disk
    manifest and OutputFetchError.manifest. The failure is recorded per file,
    the manifest is still persisted, and the raise is the standard
    OutputFetchError with the manifest attached."""
    payload = b'deck'
    runtime = _runtime_with_listing([('/tmp/work/a/out.pptx', payload)])
    # a regular file squatting on the runtime subdir name: makedirs(out_dir)
    # fails with FileExistsError/NotADirectoryError even as root.
    (tmp_path / 'main').write_text('not a directory', encoding='utf-8')
    with pytest.raises(OutputFetchError) as exc_info:
        await fetch_runtime_outputs(
            _provider(runtime),
            dest_dir=str(tmp_path),
            globs=['/tmp/work/*/out.pptx'],
            per_runtime_subdir=True,
        )
    manifest = exc_info.value.manifest
    assert manifest['files'], 'the per-file failure must be recorded'
    assert any('out.pptx' in e for e in manifest['errors'])
    assert 'error' in manifest['files'][0]
    # the manifest-first contract: the partial record is on disk too
    on_disk = json.loads((tmp_path / 'fetch_manifest.json').read_text(encoding='utf-8'))
    assert any('out.pptx' in e for e in on_disk['errors'])


@pytest.mark.asyncio
async def test_required_upload_failure_reaches_composed_hook_channels(tmp_path: Path, fake_artifact_client):
    """Upload failures ride the same required/optional channel as fetch
    failures: a required hook re-raises, so the driver's raise_if_failed()
    judges the run; the disk copy and summary still exist."""
    fake_artifact_client.failure = ArtifactUploadError('rejected: HTTP 500: nope')
    payload = b'deck'
    runtime = _runtime_with_listing([('/tmp/work/a/out.pptx', payload)])
    hook = OutputFetchHook(dest_dir=str(tmp_path), globs=['/tmp/work/*/out.pptx'], artifact_upload=True)
    hook.required = True
    composed = ComposedHook({'output_fetch': hook})
    ctx = SimpleNamespace(runtime_provider=_provider(runtime), run_id='run-u4')
    output = RunOutput(success=True)

    await composed.on_context_ready(ctx)
    with pytest.raises(OutputFetchError):
        await composed.on_run_output(ctx, output)
    assert 'artifact upload' in output.hook_errors['output_fetch.on_run_output']
    await composed.on_context_closing(ctx)
    summary = output.hook_results['output_fetch']
    assert summary['artifact_mode'] == 'artifact_service'
    assert any('artifact upload' in e for e in summary['errors'])
    assert (tmp_path / 'out.pptx').read_bytes() == payload


@pytest.mark.asyncio
async def test_optional_upload_failure_does_not_fail_the_run(tmp_path: Path, fake_artifact_client):
    """required=false keeps the run green while hook_errors still shows the
    upload failure -- the never-fail default for best-effort uploads."""
    fake_artifact_client.failure = ArtifactUploadError('rejected: HTTP 500: nope')
    payload = b'deck'
    runtime = _runtime_with_listing([('/tmp/work/a/out.pptx', payload)])
    hook = OutputFetchHook(dest_dir=str(tmp_path), globs=['/tmp/work/*/out.pptx'], artifact_upload=True)
    hook.required = False
    composed = ComposedHook({'output_fetch': hook})
    ctx = SimpleNamespace(runtime_provider=_provider(runtime), run_id='run-u5')
    output = RunOutput(success=True)

    await composed.on_context_ready(ctx)
    await composed.on_run_output(ctx, output)  # does not raise
    assert 'artifact upload' in output.hook_errors['output_fetch.on_run_output']
    await composed.on_context_closing(ctx)
    assert output.success is True
    assert output.hook_results['output_fetch']['artifact_mode'] == 'artifact_service'


@pytest.mark.asyncio
async def test_result_artifact_list_is_capped_but_manifest_keeps_all(tmp_path: Path, fake_artifact_client):
    """hook_results has a hard JSON budget when it travels inside the run
    result, so the artifacts list is capped; the manifest is not."""
    from nexus.runner.hooks.output_fetch import MAX_ARTIFACT_REFS_IN_RESULT

    paths = [(f'/tmp/work/a/out_{i:02d}.pptx', f'deck {i}'.encode()) for i in range(MAX_ARTIFACT_REFS_IN_RESULT + 1)]
    runtime = _runtime_with_listing(paths)
    manifest = await fetch_runtime_outputs(
        _provider(runtime),
        dest_dir=str(tmp_path),
        globs=['/tmp/work/a/*.pptx'],
        artifact_upload=True,
    )
    # one client per file: every file was uploaded exactly once
    assert len(_FakeArtifactClient.instances) == len(paths)
    assert sum(len(c.uploads) for c in _FakeArtifactClient.instances) == len(paths)
    assert manifest['artifact_upload']['uploaded'] == len(paths)
    hook = OutputFetchHook(dest_dir=str(tmp_path), globs=['/tmp/work/a/*.pptx'], artifact_upload=True)
    hook._result = manifest
    result = hook.get_result()
    assert len(result['artifacts']) == MAX_ARTIFACT_REFS_IN_RESULT
    assert result['n_files'] == len(paths)
    assert len(manifest['files']) == len(paths)


@pytest.mark.asyncio
async def test_disambiguated_file_uploads_under_its_final_name(tmp_path: Path, fake_artifact_client):
    """The stored name is derived from the (run, runtime, remote path)
    identity, so it stays the same across retries even when the local
    disambiguation suffix shifts (dest_dir already holding a file): re-running
    the upload from the manifest is reproducible, and two same-named sandbox
    paths still land at distinct stored names."""
    (tmp_path / 'out.pptx').write_bytes(b'previous run')
    payload = b'new run'
    runtime = _runtime_with_listing([('/tmp/work/a/out.pptx', payload)])
    manifest = await fetch_runtime_outputs(
        _provider(runtime),
        dest_dir=str(tmp_path),
        globs=['/tmp/work/*/out.pptx'],
        run_id='run-d1',
        artifact_upload=True,
    )
    client = _FakeArtifactClient.instances[0]
    expected = hashlib.sha256(b'run-d1:main:/tmp/work/a/out.pptx').hexdigest()[:16] + '-out.pptx'
    assert client.uploads[0]['name'] == expected
    assert manifest['files'][0]['disambiguated'] == 'out_2.pptx'
