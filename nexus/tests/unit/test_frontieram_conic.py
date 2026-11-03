import copy
import json
import time

from types import SimpleNamespace

import pytest

from nexus.extensions.tasks.frontieram_conic.assets import IMAGE_ID, ConicAssets
from nexus.extensions.tasks.frontieram_conic.grading import interpret_summary, verifier_command, verify_verifier
from nexus.extensions.tasks.frontieram_conic.mcp import DeadlineWorkspace
from nexus.extensions.tasks.frontieram_conic.runtime import ConicRuntime
from nexus.extensions.tasks.frontieram_offline.assets import AssetDriftError, file_sha256


@pytest.fixture
def profile(tmp_path, monkeypatch):
    monkeypatch.setattr(
        'nexus.extensions.tasks.frontieram_conic.assets.LockedAssets',
        lambda root: SimpleNamespace(source_root=root, verify=lambda: {}),
    )
    public = tmp_path / 'public.tar'
    public.write_bytes(b'public')
    task = tmp_path / 'task'
    task.mkdir()
    (task / 'grade.py').write_bytes(b'protected grader')
    path = tmp_path / 'profile.json'
    data = {
        'task_version': '0.3.0',
        'scoring_version': 'success-v2',
        'scoring_mode': 'pilot',
        'image_id': IMAGE_ID,
        'case_ids': [f'case{i}' for i in range(20)],
        'repeats': 3,
        'task_root': str(task),
        'public_tar': str(public),
        'public_tar_sha256': file_sha256(public),
        'locked_files': {'grade.py': file_sha256(task / 'grade.py')},
    }
    path.write_text(json.dumps(data))
    return path, data


def test_profile_and_private_assets_cannot_drift(profile):
    path, data = profile
    assets = ConicAssets(path.parent, path, file_sha256(path))
    assets.verify_private_data()
    (assets.task_root / 'grade.py').write_bytes(b'changed')
    with pytest.raises(AssetDriftError):
        assets.verify_private_data()
    data['repeats'] = 1
    path.write_text(json.dumps(data))
    with pytest.raises(AssetDriftError):
        assets.verify()


@pytest.mark.parametrize(
    ('key', 'value'),
    [
        ('scoring_version', 'scoring-v1'),
        ('image_id', 'mutable:latest'),
        ('repeats', 1),
        ('case_ids', ['duplicate'] * 20),
    ],
)
def test_profile_rejects_mixed_contract(profile, key, value):
    path, data = profile
    data[key] = value
    path.write_text(json.dumps(data))
    with pytest.raises(AssetDriftError):
        ConicAssets(path.parent, path, file_sha256(path))


def test_conic_tool_rejects_exponential_lease(tmp_path):
    lease = tmp_path / 'lease.json'
    lease.write_text(json.dumps({'version': 'frontieram-offline-runtime-v1', 'image_id': IMAGE_ID}))
    lease.chmod(0o600)
    with pytest.raises(ValueError, match='contract'):
        DeadlineWorkspace.from_lease(lease)


@pytest.mark.asyncio
async def test_runtime_binds_the_verified_conic_profile_to_its_tool(profile, tmp_path, monkeypatch):
    path, _ = profile
    assets = ConicAssets(tmp_path, path, file_sha256(path))
    runtime = ConicRuntime(
        assets=assets,
        docker=SimpleNamespace(prefix=['docker', '--host', 'unix:///local/socket'], cpuset='0', cgroup_parent=None),
        container='nexus-frontieram-' + 'a' * 32,
        output_dir=tmp_path,
        configuration_checks={},
    )

    async def inspect():
        return {'Id': 'b' * 64}

    monkeypatch.setattr(runtime, 'inspect', inspect)
    await runtime.initialize_lease()
    config = runtime.mcp_config(deadline_monotonic=time.monotonic() + 60)
    lease = json.loads(runtime._lease_path.read_text())
    assert runtime.assets is runtime.conic_assets is assets
    assert lease['image_id'] == IMAGE_ID
    assert lease['profile_sha256'] == file_sha256(path)
    assert config['args'][1] == 'nexus.extensions.tasks.frontieram_conic.mcp'


@pytest.fixture
def summary():
    return {
        'scoring_version': 'success-v2',
        'mode': 'pilot',
        'valid_assessment': True,
        'S': None,
        'formal_score': False,
        'p': 0.5,
        'cases': [{'case_id': 'a'}, {'case_id': 'b'}],
    }


def test_pilot_has_no_full_suite_score(summary):
    result = interpret_summary(summary, ['a', 'b'])
    assert result['exit_status'] == 'finished'
    assert result['score'] is None
    assert result['metrics']['p'] == 0.5


@pytest.mark.parametrize(
    'change',
    [
        {'G': 0.1},
        {'S': 5},
        {'formal_score': True},
        {'valid_assessment': False},
        {'cases': [{'case_id': 'a'}]},
        {'scoring_version': 'scoring-v1'},
    ],
)
def test_invalid_or_old_scoring_never_becomes_model_zero(summary, change):
    summary.update(change)
    with pytest.raises(RuntimeError):
        interpret_summary(summary, ['a', 'b'])


def test_invalid_submission_is_distinct_from_missing_evaluation():
    result = interpret_summary({'outcome': 'submission_error', 'reason': 'missing run.sh'}, ['a'])
    assert result['score'] == 0
    assert 'p' not in result['metrics']


def test_verifier_limits_and_network_are_independent_of_development():
    argv = verifier_command(['docker'], IMAGE_ID, 'test', 'owner')
    assert argv[argv.index('--memory') + 1] == str(8 * 2**30)
    assert argv[argv.index('--network') + 1] == 'none'
    assert not any(x in argv for x in ['--privileged', '--volume', '--mount', '--publish'])
    info = {
        'Image': IMAGE_ID,
        'Mounts': [],
        'Config': {'Labels': {'nexus.frontieram.owner': 'owner'}, 'User': '0:0'},
        'HostConfig': {
            'NetworkMode': 'none',
            'NanoCpus': 10**9,
            'CpusetCpus': '0',
            'Memory': 8 * 2**30,
            'MemorySwap': 8 * 2**30,
            'CgroupnsMode': 'private',
            'SecurityOpt': ['no-new-privileges:true'],
            'CapDrop': ['ALL'],
            'CapAdd': ['CHOWN', 'DAC_OVERRIDE', 'FOWNER', 'SETUID', 'SETGID', 'KILL'],
        },
    }
    assert all(verify_verifier(info, IMAGE_ID, 'owner').values())
    for key, value in [
        ('NetworkMode', 'bridge'),
        ('MemorySwap', -1),
        ('Binds', ['/tmp:/host']),
        ('CapAdd', ['SYS_ADMIN']),
        ('PidMode', 'host'),
    ]:
        altered = copy.deepcopy(info)
        altered['HostConfig'][key] = value
        with pytest.raises(RuntimeError):
            verify_verifier(altered, IMAGE_ID, 'owner')
