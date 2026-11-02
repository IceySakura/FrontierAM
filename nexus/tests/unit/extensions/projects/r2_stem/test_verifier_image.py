import base64
import subprocess

from pathlib import Path
from types import SimpleNamespace

import pytest

from tests.unit.extensions.projects.r2_stem.verifier_outputs import collected_stdout

import nexus.extensions.projects.r2_stem.evaluator.verifier_image as verifier_module

from nexus.extensions.projects.r2_stem.evaluator.common import resolve_verifier_env
from nexus.extensions.projects.r2_stem.evaluator.verifier_image import (
    _stage_verifier_artifacts,
    _validate_artifact_mounts,
    _validate_verifier_runtime_topology,
    verifier_image_evaluate,
)
from nexus.runtime.executors.models.file import FileItem


IMAGE = 'registry.example/verifier@sha256:' + 'a' * 64


class FakeProvider:
    def __init__(self, *, enable_dind: bool = False):
        self.params = {
            'image': IMAGE,
            'enable_dind': enable_dind,
            'network_policy': {'default_action': 'deny', 'egress': []},
        }

    def get_params(self, name):
        assert name == 'verifier'
        return dict(self.params)


def test_verifier_env_is_resolved_only_from_declared_host_refs(monkeypatch) -> None:
    monkeypatch.setenv('HOST_JUDGE_TOKEN', 'secret-value')

    resolved = resolve_verifier_env(
        {'STATIC_VALUE': 'fixed', 'EXPANDED': '${HOST_JUDGE_TOKEN}'},
        {'JUDGE_TOKEN': 'HOST_JUDGE_TOKEN'},
    )

    assert resolved == {
        'STATIC_VALUE': 'fixed',
        'EXPANDED': 'secret-value',
        'JUDGE_TOKEN': 'secret-value',
    }


REGISTRY_RULE = {'action': 'allow', 'target': 'registry.example:443'}


def test_verifier_topology_requires_digest_match_dind_match_and_explicit_network_mode() -> None:
    provider = FakeProvider()
    ctx = SimpleNamespace(runtime_provider=provider, runtime_configs={'verifier': dict(provider.params)})

    def validate(*, image=IMAGE, dind=False, network_access=False):
        _validate_verifier_runtime_topology(
            ctx,
            verifier_runtime_name='verifier',
            verifier_image=image,
            verifier_dind=dind,
            verifier_network_access=network_access,
        )

    validate()

    with pytest.raises(ValueError, match='pinned by sha256 digest'):
        validate(image='registry.example/verifier:latest')

    with pytest.raises(ValueError, match='enable_dind does not match'):
        validate(dind=True)

    provider.params['network_policy'] = {'default_action': 'allow', 'egress': []}
    with pytest.raises(ValueError, match='requires verifier_network_access=true'):
        validate()
    validate(network_access=True)

    # Direct execution never pulls, so even the image registry is not exempt.
    provider.params['network_policy'] = {'default_action': 'deny', 'egress': [dict(REGISTRY_RULE)]}
    with pytest.raises(ValueError, match='forbids allow rules'):
        validate()

    # The nested runner needs that exact rule, and nothing beyond it.
    provider.params['enable_dind'] = True
    validate(dind=True)

    provider.params['network_policy'] = {'default_action': 'deny', 'egress': []}
    with pytest.raises(ValueError, match='exact image registry'):
        validate(dind=True)

    provider.params['network_policy'] = {
        'default_action': 'deny',
        'egress': [dict(REGISTRY_RULE), {'action': 'allow', 'target': 'unrelated.example:443'}],
    }
    with pytest.raises(ValueError, match='permits only'):
        validate(dind=True)


class LocalAgentRuntime:
    """Runs the real snapshot program and serves file content the way the executor does."""

    async def run_command(self, *, command, timeout):
        completed = subprocess.run(['bash', '-c', command], capture_output=True, text=True)  # noqa: S603
        return SimpleNamespace(return_code=completed.returncode, stdout=completed.stdout, stderr=completed.stderr)

    async def download_file(self, *, files):
        downloaded = []
        for item in files:
            if item.encoding == 'utf8':
                # The executor reads text in universal-newline mode, so CRLF does not survive.
                content = Path(item.path).read_text(encoding='utf-8')
            else:
                content = base64.b64encode(Path(item.path).read_bytes()).decode('ascii')
            downloaded.append(FileItem(path=item.path, content=content, encoding=item.encoding))
        return SimpleNamespace(files=downloaded)


class RecordingVerifierRuntime:
    def __init__(self):
        self.uploads = []
        self.commands = []

    async def run_command(self, *, command, timeout):
        self.commands.append(command)
        return SimpleNamespace(return_code=0, stdout='', stderr='')

    async def upload_file(self, *, files, overwrite):
        self.uploads.extend(files)
        return SimpleNamespace(results=[SimpleNamespace(success=True) for _ in files])


@pytest.fixture
def local_artifact_roots(tmp_path, monkeypatch):
    monkeypatch.setattr(verifier_module, 'AGENT_ARTIFACT_ROOT', str(tmp_path / 'agent-snapshot'))
    monkeypatch.setattr(verifier_module, 'VERIFIER_ARTIFACT_ROOT', str(tmp_path / 'verifier-staging'))
    return tmp_path


@pytest.mark.asyncio
async def test_staged_artifacts_travel_byte_exact_whatever_the_content(local_artifact_roots) -> None:
    source = local_artifact_roots / 'submission'
    source.mkdir()
    (source / 'answer.json').write_text('{"value": "\u00fc"}', encoding='utf-8')
    (source / 'weights.npz').write_bytes(b'PK\x03\x04\x00\xff\xfe not text')
    # csv.writer defaults to CRLF, which a text-mode transfer would silently rewrite.
    (source / 'modes.csv').write_bytes(b'mode,value\r\nq01,1.5\r\n')

    verifier_runtime = RecordingVerifierRuntime()
    mount_pairs = await _stage_verifier_artifacts(
        LocalAgentRuntime(), verifier_runtime, [str(source)], [str(source)], 'sample-1'
    )

    staging_root = f'{verifier_module.VERIFIER_ARTIFACT_ROOT}/sample-1'
    assert mount_pairs == [(staging_root + str(source), str(source))]
    by_name = {Path(item.path).name: item for item in verifier_runtime.uploads}
    assert all(item.encoding == 'base64' for item in verifier_runtime.uploads)
    assert base64.b64decode(by_name['answer.json'].content) == '{"value": "\u00fc"}'.encode()
    assert base64.b64decode(by_name['weights.npz'].content) == b'PK\x03\x04\x00\xff\xfe not text'
    assert base64.b64decode(by_name['modes.csv'].content) == b'mode,value\r\nq01,1.5\r\n'


@pytest.mark.asyncio
async def test_oversized_artifacts_are_split_into_chunks_and_rejoined(local_artifact_roots, monkeypatch) -> None:
    monkeypatch.setattr(verifier_module, 'ARTIFACT_CHUNK_BYTES', 8)
    source = local_artifact_roots / 'blob.bin'
    source.write_bytes(b'\xff' * 30)

    verifier_runtime = RecordingVerifierRuntime()
    await _stage_verifier_artifacts(LocalAgentRuntime(), verifier_runtime, [str(source)], [str(source)], 'sample-1')

    # 30 bytes at 8 bytes per chunk: four parts, none of them larger than the chunk limit.
    assert [Path(item.path).name for item in verifier_runtime.uploads] == [
        'blob.bin.part00000',
        'blob.bin.part00001',
        'blob.bin.part00002',
        'blob.bin.part00003',
    ]
    assert all(item.encoding == 'base64' for item in verifier_runtime.uploads)
    assert b''.join(base64.b64decode(item.content) for item in verifier_runtime.uploads) == b'\xff' * 30
    rejoin = next(command for command in verifier_runtime.commands if command.startswith('set -e; cat '))
    assert rejoin.count('blob.bin.part') == 8  # four parts, named once to cat and once to rm


@pytest.mark.asyncio
@pytest.mark.parametrize(('length', 'expected_parts'), [(0, 1), (8, 1), (9, 2), (16, 2), (17, 3)])
async def test_chunking_handles_empty_and_exact_multiple_sizes(
    local_artifact_roots, monkeypatch, length, expected_parts
) -> None:
    monkeypatch.setattr(verifier_module, 'ARTIFACT_CHUNK_BYTES', 8)
    source = local_artifact_roots / 'blob.bin'
    source.write_bytes(b'\xff' * length)

    verifier_runtime = RecordingVerifierRuntime()
    await _stage_verifier_artifacts(LocalAgentRuntime(), verifier_runtime, [str(source)], [str(source)], 'sample-1')

    assert len(verifier_runtime.uploads) == expected_parts
    assert b''.join(base64.b64decode(item.content) for item in verifier_runtime.uploads) == b'\xff' * length


@pytest.mark.asyncio
async def test_staging_rejects_an_artifact_that_changed_mid_transfer(local_artifact_roots) -> None:
    source = local_artifact_roots / 'answer.json'
    source.write_text('{"value": 1}', encoding='utf-8')

    truncated = base64.b64encode(b'{}').decode('ascii')

    class TruncatingAgentRuntime(LocalAgentRuntime):
        async def download_file(self, *, files):
            return SimpleNamespace(
                files=[FileItem(path=item.path, content=truncated, encoding=item.encoding) for item in files]
            )

    with pytest.raises(ValueError, match='artifact changed during transfer'):
        await _stage_verifier_artifacts(
            TruncatingAgentRuntime(), RecordingVerifierRuntime(), [str(source)], [str(source)], 'sample-1'
        )


@pytest.mark.asyncio
async def test_staging_rejects_an_oversized_artifact_set_before_transferring_anything(
    local_artifact_roots, monkeypatch
) -> None:
    monkeypatch.setattr(verifier_module, 'MAX_VERIFIER_ARTIFACT_BYTES', 8)
    source = local_artifact_roots / 'answer.json'
    source.write_text('{"value": 1}', encoding='utf-8')

    verifier_runtime = RecordingVerifierRuntime()
    with pytest.raises(ValueError, match='exceed the 8 byte limit'):
        await _stage_verifier_artifacts(LocalAgentRuntime(), verifier_runtime, [str(source)], [str(source)], 'sample-1')
    assert verifier_runtime.uploads == []


def test_verifier_mounts_reject_protected_and_overlapping_paths() -> None:
    assert _validate_artifact_mounts([{'source': '/app/output', 'target': '/submission', 'exclude': ['*.tmp']}]) == [
        ('/app/output', '/submission', ['*.tmp'])
    ]

    with pytest.raises(ValueError, match='protected path'):
        _validate_artifact_mounts(['/tests'])

    with pytest.raises(ValueError, match='overlapping'):
        _validate_artifact_mounts(['/app/output', '/app/output/result.json'])


class FakeVerifierRuntime:
    def __init__(self, events):
        self.events = events

    async def run_command(self, *, command, timeout):
        self.events.append(('verifier_command', command, timeout))
        if 'cd /logs/verifier' in command:
            stdout = collected_stdout(reward='0.75\n', ctrf={'summary': {'tests': 4, 'passed': 3}})
            return SimpleNamespace(return_code=0, stdout=stdout, stderr='')
        return SimpleNamespace(return_code=0, stdout='', stderr='')

    async def invoke(self, import_path, payload, *, request_timeout):
        self.events.append(('verifier_invoke', import_path, payload, request_timeout))
        return SimpleNamespace(status=SimpleNamespace(value='Finished'), return_code=0, stdout='', stderr='')


class FakeContext:
    def __init__(self, *, network_access: bool, events, dind: bool = False):
        self.runtime_provider = FakeProvider(enable_dind=dind)
        if dind:
            self.runtime_provider.params['network_policy'] = {
                'default_action': 'deny',
                'egress': [{'action': 'allow', 'target': 'registry.example:443'}],
            }
        if network_access:
            self.runtime_provider.params['network_policy'] = {'default_action': 'allow', 'egress': []}
        self.runtime_configs = {'verifier': dict(self.runtime_provider.params)}
        self.run_id = 'run-1'
        self.events = events
        self.main = SimpleNamespace()
        self.verifier = FakeVerifierRuntime(events)

    async def get_runtime(self, name):
        self.events.append(('get_runtime', name))
        return self.main if name == 'main' else self.verifier

    def get_runtime_id(self, name):
        return 'verifier-runtime-id' if name == 'verifier' else 'main-runtime-id'


@pytest.mark.asyncio
@pytest.mark.parametrize(('dind', 'network_access'), [(False, False), (False, True), (True, False), (True, True)])
async def test_verifier_image_scores_image_owned_tests_and_preserves_fractional_reward(
    monkeypatch, dind, network_access
) -> None:
    events = []
    ctx = FakeContext(dind=dind, network_access=network_access, events=events)

    async def fake_stage_artifacts(agent_runtime, verifier_runtime, mounts, allowed, sample_id):
        assert agent_runtime is ctx.main
        assert verifier_runtime is ctx.verifier
        assert mounts == ['/app/output.patch']
        assert allowed == []
        events.append(('stage_artifacts', sample_id))
        return [('/tmp/nexus_verifier_artifacts/sample-1/app/output.patch', '/app/output.patch')]

    async def fake_pull(runtime, image, timeout):
        assert runtime is ctx.verifier
        events.append(('pull_verifier_image', image, timeout))

    monkeypatch.setattr(verifier_module, 'require_context', lambda: ctx)
    monkeypatch.setattr(verifier_module, '_stage_verifier_artifacts', fake_stage_artifacts)
    monkeypatch.setattr(verifier_module, '_pull_docker_image', fake_pull)

    reward, ctrf = await verifier_image_evaluate(
        sample_id='sample-1',
        verifier_image=IMAGE,
        verifier_dind=dind,
        verifier_network_access=network_access,
        verifier_mounts=['/app/output.patch'],
        test_timeout=90,
        docker_timeout=30,
    )

    assert reward == 0.75
    assert ctrf['summary'] == {'tests': 4, 'passed': 3}
    assert ctrf['nexus_verifier']['return_code'] == 0
    assert ctrf['nexus_verifier']['log_tail'] == ''

    invoke = next(event for event in events if event[0] == 'verifier_invoke')
    command = invoke[2]['command']
    assert any(event[0] == 'pull_verifier_image' for event in events) is dind
    # Neither mode may mount over /tests: the image alone owns the trusted tests.
    assert '/tests:' not in command
    if dind:
        assert 'docker run' in command
        assert ('docker network create --internal' in command) is (not network_access)
        assert '/app/output.patch:ro' in command
        # The inner daemon is what dind buys; images use it to isolate the scoring run further.
        assert '-v /var/run/docker.sock:/var/run/docker.sock' in command
        # `docker run` honours the image WORKDIR, so no explicit cd is needed.
        assert 'cd /tests' not in command
        assert 'bash /tests/test.sh' in command
        # Whatever the image left on the inner daemon still holds /logs/verifier writable, so it
        # is swept before the rewards are read back.
        commands = [event[1] for event in events if event[0] == 'verifier_command']
        swept = next(i for i, entry in enumerate(commands) if 'docker ps -aq | xargs -r docker rm -f' in entry)
        assert swept < next(i for i, entry in enumerate(commands) if 'cd /logs/verifier' in entry)
    else:
        assert 'docker' not in command
        assert 'cd /tests' in command
        assert 'bash /tests/test.sh' in command
        assert 'chmod -R a-w /app/output.patch' in command
    assert events[-1][0] == 'verifier_command'
    assert events[-1][1] == 'rm -rf /tmp/nexus_verifier_artifacts/sample-1'


@pytest.mark.asyncio
async def test_verifier_image_fails_when_image_does_not_ship_the_tests(monkeypatch) -> None:
    events = []
    ctx = FakeContext(network_access=False, events=events)

    async def missing_test_script(*, command, timeout):
        events.append(('verifier_command', command, timeout))
        return SimpleNamespace(return_code=1 if command.startswith('test -f /tests/test.sh') else 0, stdout='')

    ctx.verifier.run_command = missing_test_script
    monkeypatch.setattr(verifier_module, 'require_context', lambda: ctx)

    with pytest.raises(FileNotFoundError, match='/tests/test.sh'):
        await verifier_image_evaluate(sample_id='sample-1', verifier_image=IMAGE)
