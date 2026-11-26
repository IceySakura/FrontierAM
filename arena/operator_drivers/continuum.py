"""Grading of continuum-estimation on a grading slot with the author's verifier (task/tests/grade.py, ported).

One verifier container per job, as the author's operator/run_eval.py runs it, with the slot's CPU and cgroup parent:
1 CPU, 2 GiB RAM + 2 GiB swap, 256 processes, no network, root verifier (the candidate runs as UID 1000 under
Landlock and seccomp inside it). The public materials and resources.yaml are copied to /workspace, the private tests
to /tests and the submission to /workspace/submission; /workspace/runtime is the image's x86-64 runtime, and the
runtime lock is that runtime's own SHA256.json (the author's lock lists the ARM64 files). The public partition
(--partition public) stages the 8 public examples as /tests/private/public with their hashes in the manifest.

Held-out feedback (user decision 2026-11-26): the 8 public examples carry their truth in the agent's workspace, so
feedback on them taught nothing that generalizes. --split names a file dividing the 60 primary cases into 45 scored
and 15 feedback cases; --partition scored grades the 45 (the private score), --partition feedback stages the 15 as the
grader's public partition. Only the staged copy of the tests changes: its manifest and the partition sizes that
grade.py checks.

Usage: continuum.py OPERATOR_ROOT WORKSPACE SUBMISSION OUT --image ID --cpuset C --cgroup-parent P [--partition P]
       [--split FILE]
"""

import argparse
import json
import os
import secrets
import subprocess
import tempfile

from pathlib import Path


def run(command, timeout=600, **kwargs):
    return subprocess.run(command, check=True, capture_output=True, text=True, timeout=timeout, **kwargs)


def stage_public(workspace: Path, tests: Path) -> None:
    import hashlib

    manifest = json.loads((tests / 'private/manifest.json').read_text())
    dev = workspace / 'public/dev'
    cases = sorted(p.name.split('.')[0] for p in dev.glob('*.input.json'))
    manifest['public'] = cases
    for case in cases:
        folder = tests / 'private/public' / case
        folder.mkdir(parents=True)
        for kind in ('input', 'truth'):
            data = (dev / f'{case}.{kind}.json').read_bytes()
            (folder / f'{kind}.json').write_bytes(data)
            manifest['sha256'][f'public/{case}/{kind}.json'] = hashlib.sha256(data).hexdigest()
    (tests / 'private/manifest.json').write_text(json.dumps(manifest, indent=2) + '\n')


SIZES = "{'primary': 60, 'audit': 15, 'public': 8}"


def stage_split(tests: Path, split_file: Path, partition: str) -> None:
    """scored: the primary partition reduced to the scored cases; feedback: the feedback cases as the public partition."""
    import shutil
    split = json.loads(split_file.read_text())
    manifest = json.loads((tests / 'private/manifest.json').read_text())
    if sorted(split['scored'] + split['feedback']) != sorted(manifest['primary']):
        raise SystemExit('continuum split does not partition the primary cases')
    if partition == 'scored':
        manifest['primary'] = split['scored']
    else:
        manifest['public'] = split['feedback']
        for case in split['feedback']:
            shutil.copytree(tests / 'private/primary' / case, tests / 'private/public' / case)
            for kind in ('input', 'truth'):
                manifest['sha256'][f'public/{case}/{kind}.json'] = manifest['sha256'][f'primary/{case}/{kind}.json']
    (tests / 'private/manifest.json').write_text(json.dumps(manifest, indent=2) + '\n')
    grade = tests / 'grade.py'
    text = grade.read_text()
    if SIZES not in text:
        raise SystemExit('continuum grade.py partition sizes not found')
    sizes = (f"{{'primary': {len(split['scored'])}, 'audit': 15, 'public': 8}}" if partition == 'scored'
             else f"{{'primary': 60, 'audit': 15, 'public': {len(split['feedback'])}}}")
    grade.write_text(text.replace(SIZES, sizes))


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('operator', type=Path)
    parser.add_argument('workspace', type=Path)
    parser.add_argument('submission', type=Path)
    parser.add_argument('out', type=Path)
    parser.add_argument('--image', required=True)
    parser.add_argument('--cpuset', required=True)
    parser.add_argument('--cgroup-parent', required=True)
    parser.add_argument('--partition', default='primary', choices=['primary', 'public', 'scored', 'feedback'])
    parser.add_argument('--split', type=Path)
    args = parser.parse_args()
    grader_partition = {'scored': 'primary', 'feedback': 'public'}.get(args.partition, args.partition)
    args.out.mkdir(parents=True, exist_ok=True)
    docker = ['docker', '-H', os.environ['DOCKER_HOST']]
    name = f'frontieram-continuum-{secrets.token_hex(6)}'
    error = None
    try:
        with tempfile.TemporaryDirectory(prefix='continuum-tests-') as scratch:
            tests = Path(scratch) / 'tests'
            subprocess.run(['cp', '-a', str(args.operator / 'task/tests'), str(tests)], check=True)
            if args.partition == 'public':
                stage_public(args.workspace, tests)
            elif args.partition in ('scored', 'feedback'):
                stage_split(tests, args.split, args.partition)
            run([*docker, 'create', '--name', name, '--network', 'none', '--cpus', '1', '--cpuset-cpus', args.cpuset,
                 '--cgroup-parent', args.cgroup_parent, '--memory', '2g', '--memory-swap', '4g', '--pids-limit', '256',
                 '--workdir', '/workspace', '--user', 'root', '--cap-drop', 'NET_RAW',
                 '--security-opt', 'no-new-privileges:true', '--log-driver', 'none', args.image, 'sleep', 'infinity'])
            run([*docker, 'start', name])
            run([*docker, 'exec', name, 'mkdir', '-p', '/tests', '/workspace/submission', '/logs/verifier'])
            run([*docker, 'cp', str(args.workspace / 'public'), f'{name}:/workspace/'])
            run([*docker, 'cp', str(args.workspace / 'resources.yaml'), f'{name}:/workspace/resources.yaml'])
            run([*docker, 'cp', f'{tests}/.', f'{name}:/tests/'])
            run([*docker, 'exec', name, 'cp', '/opt/continuum-runtime/SHA256.json', '/tests/runtime-lock.json'])
            run([*docker, 'cp', f'{args.submission}/.', f'{name}:/workspace/submission/'])
            command = ['/usr/local/bin/python3', '-I', '/tests/grade.py'] + (
                ['--partition', 'public'] if grader_partition == 'public' else [])
            done = subprocess.run([*docker, 'exec', name, *command], capture_output=True, text=True, timeout=3900)
            (args.out / 'stdout.txt').write_text(done.stdout)
            (args.out / 'stderr.txt').write_text(done.stderr)
    except Exception as exc:
        error = f'{type(exc).__name__}: {exc}'
    finally:
        subprocess.run([*docker, 'cp', f'{name}:/logs/verifier/.', str(args.out)], capture_output=True, text=True)
        subprocess.run([*docker, 'rm', '-f', name], capture_output=True, text=True)
    summary = json.loads((args.out / 'summary.json').read_text()) if (args.out / 'summary.json').exists() else None
    if summary is None:
        verdict = {'status': 'infrastructure_failure',
                   'reason': error or ((args.out / 'error.json').read_text()[:500] if (args.out / 'error.json').exists() else 'no summary')}
    else:
        failed = [r for r in summary['per_case'] if r['status'] != 'ok']
        verdict = {'status': 'submission_failure' if failed else 'ok', 'raw_loss': summary['raw_loss'],
                   'cases': summary['cases'], 'failures': summary['failures'], 'partition': args.partition,
                   'reason': f'{len(failed)} of {summary["cases"]} cases failed; first {failed[0]["case_id"]}: '
                             f'{failed[0].get("reason")}' if failed else None,
                   'max_wall_s': max((r.get('elapsed_seconds') or 0 for r in summary['per_case']), default=None)}
    (args.out / 'verdict.json').write_text(json.dumps(verdict, indent=1) + '\n')


if __name__ == '__main__':
    main()
