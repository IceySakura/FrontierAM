"""Run the unchanged success-v2 verifier in its own offline 8 GiB container."""

from __future__ import annotations

import argparse
import asyncio
import io
import json
import subprocess
import sys
import tarfile
import uuid

from pathlib import Path

from nexus.extensions.tasks.frontieram_offline.grading import OWNER_LABEL, OwnedCleanupError, _finish_grading

from .assets import ConicAssets


def verifier_command(prefix, image, name, owner, cpuset='0', cgroup_parent=None):
    placement = ['--cgroup-parent', cgroup_parent] if cgroup_parent is not None else []
    return [
        *prefix,
        'create',
        '--name',
        name,
        '--label',
        f'{OWNER_LABEL}={owner}',
        '--network',
        'none',
        '--cgroupns',
        'private',
        '--cpus',
        '1',
        '--cpuset-cpus',
        str(cpuset),
        *placement,
        '--memory',
        str(8 * 2**30),
        '--memory-swap',
        str(8 * 2**30),
        '--pids-limit',
        '256',
        '--ulimit',
        'core=0:0',
        '--ipc',
        'none',
        '--cap-drop',
        'ALL',
        '--cap-add',
        'CHOWN',
        '--cap-add',
        'DAC_OVERRIDE',
        '--cap-add',
        'FOWNER',
        '--cap-add',
        'SETUID',
        '--cap-add',
        'SETGID',
        '--cap-add',
        'KILL',
        '--security-opt',
        'no-new-privileges:true',
        '--user',
        '0:0',
        '--workdir',
        '/workspace',
        '--log-driver',
        'none',
        '--entrypoint',
        '/usr/local/bin/python3',
        image,
        '-I',
        '/tests/grade.py',
    ]


def verify_verifier(info, image, owner, cpuset='0', cgroup_parent=None):
    h, c = info['HostConfig'], info['Config']
    checks = {
        'owned': (c.get('Labels') or {}).get(OWNER_LABEL) == owner,
        'image': info['Image'] == image,
        'offline': h['NetworkMode'] == 'none',
        'no_host_mounts': not h.get('Binds') and not info.get('Mounts'),
        'cpu': h['NanoCpus'] == 10**9 and h['CpusetCpus'] == str(cpuset),
        'slot_cgroup_parent': (h.get('CgroupParent') or None) == cgroup_parent,
        'memory_no_swap': h['Memory'] == h['MemorySwap'] == 8 * 2**30,
        'private_namespaces': h['CgroupnsMode'] == 'private' and h.get('PidMode') in ('', None),
        'no_devices': not h.get('Privileged') and not h.get('Devices') and not h.get('DeviceRequests'),
        'root_supervisor': c['User'] == '0:0',
        'no_new_privileges': any(s in ('no-new-privileges', 'no-new-privileges:true') for s in h['SecurityOpt']),
        'restricted_caps': {s.removeprefix('CAP_') for s in h['CapDrop']} == {'ALL'}
        and {s.removeprefix('CAP_') for s in h['CapAdd']}
        == {'CHOWN', 'DAC_OVERRIDE', 'FOWNER', 'SETUID', 'SETGID', 'KILL'},
    }
    if not all(checks.values()):
        raise RuntimeError('Unsafe conic verifier configuration: ' + ','.join(k for k, v in checks.items() if not v))
    return checks


def interpret_summary(summary, expected_ids):
    if summary.get('outcome') == 'submission_error':
        return {'exit_status': 'finished', 'score': 0.0, 'reason': summary['reason'], 'metrics': summary}
    cases = summary.get('cases', [])
    if (
        summary.get('scoring_version') != 'success-v2'
        or summary.get('mode') != 'pilot'
        or summary.get('valid_assessment') is not True
        or summary.get('S') is not None
        or summary.get('formal_score') is not False
        or 'G' in summary
        or len(cases) != len(expected_ids)
        or {c.get('case_id') for c in cases} != set(expected_ids)
    ):
        raise RuntimeError('Invalid or mixed-version conic assessment')
    return {'exit_status': 'finished', 'score': None, 'metrics': summary}


def worker(config_path):
    config = json.loads(config_path.read_text())
    root = Path(config['output_dir'])
    assets = ConicAssets(config['source_root'], config['profile_path'], config['profile_sha256'])
    assets.verify_private_data()
    docker = assets.controller.Docker(config['docker_path'], config['docker_host'])
    name = 'nexus-conic-grade-' + config['owner']
    # Record before creation: cleanup can recover a cancellation at any point.
    Path(config['ledger']).write_text(json.dumps({'kind': 'container', 'value': name}) + '\n')
    result = {'exit_status': 'runtime_error_by_env', 'score': None, 'reason': 'grader_not_completed'}
    try:
        placement = (config['cpuset'], config['cgroup_parent'])
        subprocess.run(
            verifier_command(docker.prefix, assets.image_id, name, config['owner'], *placement),
            check=True,
            capture_output=True,
            timeout=30,
        )
        checks = verify_verifier(docker.inspect(name), assets.image_id, config['owner'], *placement)
        (root / 'verifier_configuration.json').write_text(json.dumps(checks, indent=2) + '\n')
        docker.run('cp', str(assets.task_root / 'tests'), name + ':/', timeout=120)
        # Docker cp extracts this trusted tar in a stopped container, never on the host.
        overlay = root / 'verifier_overlay.tar'
        case_ids = config.get('case_ids') or assets.profile['case_ids']
        suite = json.loads((assets.task_root / 'tests/private/data/manifest.json').read_text())['instances']
        if len(set(case_ids)) != len(case_ids) or not set(case_ids) <= {c['id'] for c in suite}:
            raise RuntimeError('Requested cases are not unique members of the held-out suite')
        grading_config = json.loads((assets.task_root / 'tests/config.json').read_text())
        # The original verifier's case-list mode: the 20-case profile subset, or one shard of the full suite.
        grading_config.update(mode='pilot', case_ids=case_ids)
        with tarfile.open(overlay, 'w') as archive:
            archive.add(root / 'source', arcname='workspace/submission', recursive=True)
            payload = (json.dumps(grading_config, indent=2) + '\n').encode()
            member = tarfile.TarInfo('tests/config.json')
            member.mode, member.size = 0o600, len(payload)
            archive.addfile(member, io.BytesIO(payload))
        with overlay.open('rb') as source:
            subprocess.run(
                [*docker.prefix, 'cp', '-', name + ':/'], stdin=source, check=True, capture_output=True, timeout=120
            )
        overlay.unlink()
        with (root / 'verifier_console.log').open('wb') as log:
            completed = subprocess.run(
                [*docker.prefix, 'start', '--attach', name], stdout=log, stderr=log, timeout=config['grading_timeout']
            )
        verifier = root / 'verifier'
        docker.run('cp', name + ':/logs/verifier', str(verifier), timeout=120)
        if completed.returncode:
            result['reason'] = 'original_verifier_failed'
        else:
            result = interpret_summary(json.loads((verifier / 'summary.json').read_text()), case_ids)
            records = list((verifier / 'cases').glob('*/repeat-*/record.json'))
            result.update(
                total_runs=len(records),
                valid_runs=sum(json.loads(p.read_text()).get('outcome') == 'accepted' for p in records),
            )
    except Exception as error:
        result = {
            'exit_status': 'runtime_error_by_env',
            'score': None,
            'reason': type(error).__name__,
            'detail': str(error)[:2000],
        }
    (root / 'grading_result.json').write_text(json.dumps(result, indent=2) + '\n')


async def grade(runtime, output_dir: Path, *, timeout: float = 28800, case_ids: list[str] | None = None) -> dict:
    assets = runtime.assets
    config = {
        'case_ids': list(case_ids) if case_ids else None,
        'source_root': str(assets.source_root),
        'profile_path': str(assets.profile_path),
        'profile_sha256': assets.profile_sha256,
        'output_dir': str(output_dir),
        'docker_path': str(runtime.docker.prefix[0]),
        'docker_host': runtime.docker.prefix[2],
        'cpuset': runtime.docker.cpuset,
        'cgroup_parent': runtime.docker.cgroup_parent,
        'ledger': str(output_dir / 'grading_owned_resources.jsonl'),
        'owner': uuid.uuid4().hex,
        'grading_timeout': timeout - 180,
    }
    config_path = output_dir / 'grading_config.json'
    config_path.write_text(json.dumps(config, indent=2) + '\n')
    child = spawn = None
    failure = None
    result = None
    with (output_dir / 'grading.log').open('wb') as log:
        try:
            spawn = asyncio.create_task(
                asyncio.create_subprocess_exec(
                    sys.executable,
                    '-m',
                    'nexus.extensions.tasks.frontieram_conic.grading',
                    str(config_path),
                    stdout=log,
                    stderr=log,
                    start_new_session=True,
                )
            )
            child = await asyncio.shield(spawn)
            await asyncio.wait_for(child.wait(), timeout=timeout)
            path = output_dir / 'grading_result.json'
            if child.returncode or not path.exists():
                result = {'exit_status': 'runtime_error_by_env', 'score': None, 'reason': 'grading_worker_failed'}
            else:
                result = json.loads(path.read_text())
        except BaseException as error:
            failure = error
        cleanup = asyncio.create_task(_finish_grading(child, spawn, config))
        while not cleanup.done():
            try:
                await asyncio.shield(cleanup)
            except asyncio.CancelledError as error:
                failure = error
            except Exception:
                break
        try:
            cleanup.result()
        except OwnedCleanupError as error:
            (output_dir / 'grading_cleanup_error.json').write_text(json.dumps(error.errors, indent=2) + '\n')
            if not isinstance(failure, asyncio.CancelledError):
                failure = error
    if failure is not None:
        raise failure
    assert result is not None
    return result


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('config', type=Path)
    worker(parser.parse_args().config)
