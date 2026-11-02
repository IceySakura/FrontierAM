"""Portable Nexus entry points: prepare, smoke, calibrate, run and grade."""

import argparse
import asyncio
import base64
import hashlib
import json
import os
import subprocess
import sys
import threading

from pathlib import Path
from types import SimpleNamespace


ROOT = Path(__file__).resolve().parents[1]
MODELS = ['claude-opus-5-5', 'deepseek-v4-pro', 'hy4-preview', 'minimax-m3.1-flash', 'step-5-preview']


def sha(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def write(path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, ensure_ascii=False) + '\n')


def load_runtime():
    root = ROOT / '.runtime'
    if not (root / 'READY').exists():
        raise RuntimeError('Run ./bench prepare first')
    config = json.loads((root / 'runtime-lock.json').read_text())
    for relative, expected in config['code_sha256'].items():
        if sha(root / relative) != expected:
            raise RuntimeError('Prepared integration changed: ' + relative)
    if sha(config['codex_path']) != config['codex_sha256']:
        raise RuntimeError('Codex executable changed after preparation')
    sys.path.insert(0, str(root / 'nexus'))
    os.environ['PYTHONPATH'] = str(root / 'nexus')
    os.environ['PYTHONDONTWRITEBYTECODE'] = '1'
    return config


def configuration(runtime, task, output, model, *, key_file=None, base_url=None, smoke=False, evaluate=True):
    conic = task == 'conic'
    provider = {
        'source_root': runtime['source_root'],
        'docker_host': runtime['docker_host'],
        'docker_path': runtime['docker_path'],
    }
    if conic:
        provider.update(profile_path=runtime['profile_path'], profile_sha256=runtime['profile_sha256'])
    instruction = (
        'Read /workspace/PILOT.md and /workspace/INSTRUCTION.md in full, then inspect /workspace/resources.yaml '
        'and public-dev. Implement the updated conic solver.'
        if conic
        else 'Read /workspace/task/PILOT.md, INSTRUCTION.md and API.md in full. Implement the exponential propagation task.'
    )
    prompt = (
        instruction
        + (
            ' Use only the offline workspace tool. Put complete source, executable run.sh and README.md in '
            '/workspace/submission. One attempt, at most 1800 wall-clock seconds including requests, waiting and tools; '
            'a final turn or deadline freezes this attempt. No hidden feedback or candidate repair after freezing. '
            'Common inference policy v3: at most 200 actual HTTP attempts including retries, 64000 output tokens '
            'per response including reasoning, 131072 context setting, provider-default reasoning. '
            'The old 429-only wording is superseded: at most three identical-payload retries for retryable '
            '429/500/502/503/504 responses; Retry-After first, otherwise full jitter caps of 1/2/4 seconds. '
            'All retries consume the original deadline. No failover or truncated tool execution. '
        )
        + runtime['architecture_override_prompt']
    )
    return {
        'imports': [
            'nexus.extensions.tasks.frontieram_conic' if conic else 'nexus.extensions.tasks.frontieram_offline'
        ],
        'task_name': 'frontieram_conic_success_offline' if conic else 'frontieram_exponential_offline',
        'task_config': {
            'output_dir': str(output / 'task'),
            'user_prompt': prompt,
            'evaluate': evaluate,
            'verification_only': smoke,
            'grading_timeout': 28800,
        },
        'runtime_provider_name': 'frontieram_conic_offline' if conic else 'frontieram_offline',
        'runtime_provider_config': provider,
        'agent_name': 'frontieram_codex_offline',
        'agent_config': {
            'model': model,
            'base_url': base_url or 'http://api.aihub.woa.com/standard/v1',
            'key_file': str(key_file) if key_file else None,
            'allow_live_endpoint': not smoke,
            'development_seconds': 60 if smoke else 1800,
            'max_http_attempts': 8 if smoke else 200,
            'max_output_tokens': 64000,
        },
        'processors': {'token_counter': {'name': 'token_counter'}},
        'runner': {'cache': {'enabled': False}, 'timeout': 31200},
    }


def fixture_command(task):
    source = ROOT / 'fixtures' / task
    files = {
        str(p.relative_to(source)): base64.b64encode(p.read_bytes()).decode() for p in source.rglob('*') if p.is_file()
    }
    return (
        "python3 - <<'PY'\nimport base64,os,socket\nfrom pathlib import Path\n"
        "assert os.getuid()==1000 and not Path('/tests').exists()\n"
        "assert not Path('/var/run/docker.sock').exists()\n"
        "assert not any('API_KEY' in k or 'AIHUB' in k or 'TAIHU' in k for k in os.environ)\n"
        "s=socket.socket();s.settimeout(1);assert s.connect_ex(('1.1.1.1',53))!=0;s.close()\n"
        "root=Path('/workspace/submission');root.mkdir(exist_ok=True)\n"
        f'files={files!r}\n'
        'for name,value in files.items():\n'
        ' p=root/name;p.parent.mkdir(parents=True,exist_ok=True);p.write_bytes(base64.b64decode(value));p.chmod(0o755)\n'
        "print('FIXTURE_WRITE_OK\\nFIXTURE_ISOLATION_OK')\nPY\n"
    )


async def run_one(runtime, args, task, *, smoke):
    from nexus.extensions.tasks.frontieram_offline.verification import Fixture
    from nexus.runner.single import RunInput, run

    output = Path(args.output).resolve()
    if args.task == 'both':
        output /= task
    output.mkdir(parents=True, exist_ok=False)
    fixture = thread = None
    if smoke:
        fixture = Fixture('basic', fixture_command(task))
        thread = threading.Thread(target=fixture.serve_forever, daemon=True)
        thread.start()
    config = configuration(
        runtime,
        task,
        output,
        args.model,
        key_file=None if smoke else args.key_file,
        base_url=f'http://127.0.0.1:{fixture.server_port}/v1' if fixture is not None else args.base_url,
        smoke=smoke,
        evaluate=not getattr(args, 'develop_only', False),
    )
    write(output / 'input.json', config)
    try:
        result = await run(RunInput.from_dict(config))
        data = result.model_dump(mode='json')
    finally:
        if fixture:
            await asyncio.to_thread(fixture.shutdown)
            fixture.server_close()
            if thread is not None:
                thread.join(timeout=2)
    write(output / 'nexus_result.json', data)
    task_result, agent = data.get('task_output') or {}, data.get('agent_output') or {}
    if smoke:
        assert fixture is not None
        checks = {
            'nexus_completed': data['success'],
            'agent_completed': agent.get('evaluation_eligible') is True,
            'tool_history': not fixture.failures,
            'task_completed': task_result.get('status') == 'success',
        }
        if task == 'exponential':
            checks['numerical_runs'] = task_result.get('valid_runs') == task_result.get('total_runs') == 69
        else:
            checks['numerical_runs'] = task_result.get('total_runs') == 60
            checks['valid_failure_assessment'] = (task_result.get('metrics') or {}).get('p') == 0
            paths = list((output / 'task/verifier/cases').glob('*/repeat-*/stdout.log'))
            checks['all_boundary_probes'] = len(paths) == 60 and all(
                len(p := json.loads(path.read_text())['boundary_fixture']) == 6 and all(p.values()) for path in paths
            )
        report = {
            'passed': all(checks.values()),
            'checks': checks,
            'real_model_calls': 0,
            'real_codex_nexus_docker': True,
            'not_a_model_performance_result': True,
        }
        write(output / 'verification.json', report)
        if not report['passed']:
            raise RuntimeError('End-to-end verification failed; inspect ' + str(output))
    print(
        json.dumps(
            {
                'task': task,
                'status': task_result.get('status'),
                'score': task_result.get('score'),
                'score_status': task_result.get('score_status'),
                'valid_runs': task_result.get('valid_runs'),
                'total_runs': task_result.get('total_runs'),
                'output': str(output),
            },
            ensure_ascii=False,
        ),
        flush=True,
    )


async def grade_existing(runtime, args, *, calibrate=False):
    from nexus.extensions.tasks.frontieram_conic.assets import ConicAssets
    from nexus.extensions.tasks.frontieram_conic.grading import grade as grade_conic
    from nexus.extensions.tasks.frontieram_offline.assets import LockedAssets
    from nexus.extensions.tasks.frontieram_offline.grading import grade as grade_exp
    from nexus.extensions.tasks.frontieram_offline.local_calibration import host_fingerprint

    assets = (
        LockedAssets(runtime['source_root'])
        if args.task == 'exponential'
        else ConicAssets(runtime['source_root'], runtime['profile_path'], runtime['profile_sha256'])
    )
    docker = assets.controller.Docker(runtime['docker_path'], runtime['docker_host'])
    output = Path(args.output).resolve()
    output.mkdir(parents=True, exist_ok=False)
    source = (
        (assets.source_root / 'private_pilot/reference_submission') if calibrate else Path(args.submission).resolve()
    )
    # Reuse the reviewed regular-file archive validator, including its size cap.
    import tarfile

    archive = output / 'incoming.tar'
    with tarfile.open(archive, 'w') as tar:
        tar.add(source, arcname='.')
    manifest = assets.controller.validate_and_extract(archive, output / 'source', output / 'source.tar')
    write(output / 'source_manifest.json', manifest)
    if calibrate and (assets.source_root / 'local_calibration.json').exists():
        raise RuntimeError('Calibration is already frozen; prepare a new profile to recalibrate')
    grader = grade_exp if args.task == 'exponential' else grade_conic
    result = await grader(SimpleNamespace(assets=assets, docker=docker), output, timeout=28800)
    if calibrate:
        assert isinstance(assets, LockedAssets)
        if result.get('valid_runs') != 69 or result.get('total_runs') != 69:
            raise RuntimeError('Reference qualification failed; no score anchor was installed')
        report = output / 'evaluation.json'
        write(
            assets.source_root / 'local_calibration.json',
            {
                'host_fingerprint': host_fingerprint(docker, assets.image_id),
                'report': str(report),
                'report_sha256': sha(report),
                'manifest_sha256': sha(assets.manifest),
                'reference_source_sha256': manifest['sha256'],
                'policy': 'frozen author KIOPS, multiplier 0.01',
            },
        )
    print(json.dumps({'calibrated': calibrate, **result}, ensure_ascii=False), flush=True)


def main():
    if len(sys.argv) > 1 and sys.argv[1] == 'prepare':
        raise SystemExit(subprocess.call([sys.executable, str(ROOT / 'scripts/prepare.py'), *sys.argv[2:]]))
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest='command', required=True)
    for name in ('smoke', 'run', 'grade', 'calibrate'):
        p = sub.add_parser(name)
        p.add_argument(
            '--task',
            choices=['exponential']
            if name == 'calibrate'
            else ['exponential', 'conic', 'both']
            if name == 'smoke'
            else ['exponential', 'conic'],
            default='both' if name == 'smoke' else 'exponential',
        )
        p.add_argument('--output', required=True)
        if name in ('smoke', 'run'):
            p.add_argument('--model', choices=MODELS, default=MODELS[0])
        if name == 'run':
            p.add_argument('--key-file', type=Path, default=os.environ.get('AIHUB_KEY_FILE'))
            p.add_argument('--base-url', default='http://api.aihub.woa.com/standard/v1')
            p.add_argument('--develop-only', action='store_true', help='Freeze source now; grade serially later')
        if name == 'grade':
            p.add_argument('--submission', required=True)
    args = parser.parse_args()
    if args.command == 'run' and not args.key_file:
        parser.error('Supply --key-file PATH or AIHUB_KEY_FILE (a file path, not the key itself)')
    runtime = load_runtime()
    if args.command in ('grade', 'calibrate'):
        asyncio.run(grade_existing(runtime, args, calibrate=args.command == 'calibrate'))
    else:
        for task in ('exponential', 'conic') if args.task == 'both' else (args.task,):
            asyncio.run(run_one(runtime, args, task, smoke=args.command == 'smoke'))


if __name__ == '__main__':
    main()
