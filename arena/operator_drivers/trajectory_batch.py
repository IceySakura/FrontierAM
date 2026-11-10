"""Grade one trajectory-covariance submission with the maintainer package's own verifier, on one grading slot.

The maintainer package (mcmc-long-run-covariance-maintainer-v1) runs task/tests/grade.py inside a separate Harbor
verifier container: it builds the submission once, then runs every case serially in fresh user/PID namespaces under
Landlock and seccomp, scoring each output with tasks/.../pilot/score.py. Harbor itself is only orchestration, and its
shard mode refuses submissions with build.sh, so this driver assembles the same verifier task directory as
harbor/mcmc-long-run-covariance/run_trial.py and starts the container directly with the settings of the task's
docker-compose.yaml and task.toml: no network, the package's seccomp profile, one CPU (quota = period), 2 GiB with no
swap, root verifier. Placement comes from the grading slot: its CPU, its slice and its Docker daemon.

The only change to package code is the declared portability edit (the verifier accepts x86_64 besides aarch64). The
private data are mounted read-only at /tests/private/data; the manifest copy lists the selected cases with those paths.

--case-limit N replaces the verifier's 30-second per-input wall limit (protocol v6.3 versions,
results/EVAL_TIME_LIMITS.md), matching exactly once like the portability edit.

Usage: trajectory_batch.py MAINTAINER SOURCE OUTPUT --cases CASES.json --image ID --cpuset N --cgroup-parent SLICE
       [--case-limit N]
"""

import argparse
import collections
import json
import math
import os
import shutil
import subprocess
import uuid

from pathlib import Path

FORMAL = 'results/mcmc-long-run-covariance/formal-v1/private'
EDIT = ("os.uname().machine != 'aarch64'", "os.uname().machine not in ('aarch64', 'x86_64')")


LIMIT = 'limit = 30 if inp else 300'


def assemble(maintainer: Path, cases: list[dict], tests: Path, case_limit: int = 30) -> None:
    harbor = maintainer / 'harbor/mcmc-long-run-covariance/task/tests'
    shared = maintainer / 'harbor/xtrace/task/tests'
    tests.mkdir(parents=True)
    for name in ('grade.py', 'test.sh'):
        shutil.copy(harbor / name, tests / name)
    if case_limit != 30:
        grader = (tests / 'grade.py').read_text()
        if grader.count(LIMIT) != 1:
            raise RuntimeError('case-limit edit does not match the verifier exactly once')
        (tests / 'grade.py').write_text(grader.replace(LIMIT, f'limit = {case_limit} if inp else 300'))
    isolation = (shared / 'grade.py').read_text()
    if isolation.count(EDIT[0]) != 1:
        raise RuntimeError('portability edit does not match the verifier exactly once')
    (tests / 'isolation.py').write_text(isolation.replace(EDIT[0], EDIT[1]))
    for name in ('drop_exec.py', 'runtime_sandbox.py', 'seccomp.json'):
        shutil.copy(shared / name, tests / name)
    shutil.copy(maintainer / 'tasks/mcmc-long-run-covariance/pilot/score.py', tests / 'score.py')
    rows = [dict(c, input=f"data/{c['input']}", reference=f"data/{c['reference']}") for c in cases]
    (tests / 'private').mkdir()
    (tests / 'private/manifest.json').write_text(json.dumps({'cases': rows}, indent=1) + '\n')
    (tests / 'private/data').mkdir()


def summarize(receipt: dict, cases: list[dict]) -> dict:
    """The evaluation plan's total: mean loss per (model, length) cell, then the mean over cells."""
    records = {r['id']: r for r in receipt.get('cases', [])}
    if set(records) != {c['id'] for c in cases}:
        return {'status': 'infrastructure_failure', 'reason': 'verifier did not report every case'}
    cells = collections.defaultdict(list)
    for case in cases:
        loss = records[case['id']].get('loss')
        cells[(case['model'], case['length'])].append(loss if loss is not None else math.sqrt(case['d']))
    means = {f'{m}/{n}': sum(v) / len(v) for (m, n), v in sorted(cells.items())}
    sums = {f'{m}/{n}': [sum(v), len(v)] for (m, n), v in sorted(cells.items())}  # shards merge these
    failures = collections.Counter(r['status'] for r in records.values() if r['status'] != 'ok')
    return {'status': 'ok', 'raw_error': sum(means.values()) / len(means), 'cells': means, 'cell_sums': sums,
            'cases': len(records),
            'failures': dict(failures),
            'max_wall_s': max(r['wall_s'] for r in records.values()),
            'max_cpu_s': max(r['cpu_s'] for r in records.values()),
            'build': receipt.get('build', {}).get('status')}


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('maintainer', type=Path)
    parser.add_argument('source', type=Path)
    parser.add_argument('output', type=Path)
    parser.add_argument('--cases', type=Path, required=True, help='JSON list of the case IDs to grade')
    parser.add_argument('--image', required=True)
    parser.add_argument('--cpuset', required=True)
    parser.add_argument('--cgroup-parent', required=True)
    parser.add_argument('--case-limit', type=int, default=30)
    args = parser.parse_args()
    maintainer, out = args.maintainer.resolve(), args.output.resolve()
    out.mkdir(parents=True, exist_ok=False)
    data = maintainer / FORMAL
    wanted = json.loads(args.cases.read_text())
    every = {c['id']: c for c in json.loads((data / 'manifest.json').read_text())['cases']}
    cases = [every[i] for i in wanted]
    tests, logs, source = out / 'tests', out / 'verifier', out / 'submission'
    assemble(maintainer, cases, tests, args.case_limit)
    logs.mkdir()
    shutil.copytree(args.source.resolve(), source, symlinks=True)
    os.chmod(logs, 0o777)
    timeout = 600 + args.case_limit * len(cases) + 300
    command = ['docker', 'run', '--rm', '--name', f'trajectory-verifier-{uuid.uuid4().hex[:12]}', '--network', 'none',
               '--cpuset-cpus', args.cpuset, '--cpus', '1', '--memory', '2147483648', '--memory-swap', '2147483648',
               '--cgroup-parent', args.cgroup_parent, '--security-opt', f'seccomp={tests / "seccomp.json"}',
               '--user', '0:0', '--workdir', '/workspace',
               '-v', f'{tests}:/tests:ro', '-v', f'{data}:/tests/private/data:ro',
               '-v', f'{source}:/workspace/submission:ro', '-v', f'{logs}:/logs/verifier',
               '--entrypoint', '/bin/sh', args.image, '/tests/test.sh']
    (out / 'docker-command.json').write_text(json.dumps(command, indent=1) + '\n')
    with (out / 'verifier.log').open('wb') as log:
        done = subprocess.run(command, stdout=log, stderr=subprocess.STDOUT, timeout=timeout)
    receipt_path = logs / 'receipt.json'
    if not receipt_path.exists():
        summary = {'status': 'infrastructure_failure', 'reason': f'no receipt (exit {done.returncode})'}
    else:
        receipt = json.loads(receipt_path.read_text())
        if receipt.get('status') == 'invalid_submission':
            summary = {'status': 'submission_failure', 'reason': receipt.get('reason')}
        else:
            summary = summarize(receipt, cases)
        summary['sandbox_checks'] = receipt.get('runtime', {}).get('sandbox_checks')
        summary['runtime'] = {k: v for k, v in receipt.get('runtime', {}).items() if k != 'sandbox_checks'}
    (out / 'summary.json').write_text(json.dumps(summary, indent=2) + '\n')
    print(json.dumps({k: v for k, v in summary.items() if k not in ('cells', 'sandbox_checks', 'runtime')}))


if __name__ == '__main__':
    main()
