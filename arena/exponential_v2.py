"""exponential-v2: dataset v2 graded against the frozen human-SOTA panel (tasks/exponential/sota_v1).

The v1 evaluator, tools, image and scorer run unchanged on a sealed copy (.runtime-exp2, scripts/prepare_exponential_v2.py);
only the manifest, the cases and the reference differ. A baseline grades every panel member in one job on one slot;
a candidate is then scored against the per-case virtual best: for each case the member with the shortest median time
among those whose three repetitions all passed. The v1 scorer turns those ratios into 100 / (1 + R), so the virtual
best itself is 50 and the 50/100 convention holds with the panel as the reference.
"""

from __future__ import annotations

import hashlib
import importlib.util
import json
import shlex
import subprocess
import sys

from pathlib import Path
from types import SimpleNamespace

from .tasks import GIB, Task, tar_directory

ROOT = Path(__file__).resolve().parents[1]
SEAL = ROOT / '.runtime-exp2'
PANEL_MARKER = 'FRONTIERAM_PANEL.json'


def _sha(path: Path) -> str:
    with path.open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


class ExponentialV2(Task):
    name = 'exponential-v2'
    instruction = ('Read /workspace/task/V2.md first, then PILOT.md, INSTRUCTION.md and API.md in full. V2.md overrides '
                   'the reference, data and timing clauses of the other files. Implement the exponential propagation task.')
    grade_memory = 4 * GIB + GIB // 2
    # Five panel members and the candidate each run every case three times on the slot's one CPU.
    grade_timeout = 24 * 3600
    # Timing ratios against a baseline measured earlier on the same slot: no other grader may share the machine.
    grade_isolation = 'grade_pool'
    paired = False

    # Sealed material -------------------------------------------------------------------------------------------
    def lock(self) -> dict:
        return json.loads((SEAL / 'lock.json').read_text())

    def verify(self) -> dict:
        lock = self.lock()
        for relative, expected in lock['files'].items():
            path = SEAL / relative
            if path.is_symlink() or not path.is_file() or _sha(path) != expected:
                raise RuntimeError(f'exponential-v2: sealed file changed or missing: {relative}')
        return lock

    def assets(self, runtime: dict):
        from .tasks import TASKS

        lock = self.lock()
        public = SEAL / 'public.tar'
        if _sha(public) != lock['public_tar_sha256']:
            raise RuntimeError('exponential-v2: sealed public package changed')
        return SimpleNamespace(controller=TASKS['exponential'].assets(runtime).controller, public_tar=public, lock=lock)

    def scoring(self):
        spec = importlib.util.spec_from_file_location('frontieram_exp2_scoring', SEAL / 'public/tools/scoring.py')
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        return module

    def cases(self) -> list[dict]:
        return json.loads((SEAL / 'data/manifest.json').read_text())['cases']

    def dev_config(self, *args, **kwargs):
        raise ValueError('exponential-v2 is developed with --protocol v6 or v6.1')

    # Reference -------------------------------------------------------------------------------------------------
    def reference_archive(self, runtime: dict, output: Path) -> Path:
        lock = self.verify()
        output.mkdir(parents=True, exist_ok=True)
        marker = json.dumps({'task': self.name, 'panel': lock['panel']}).encode() + b'\n'
        return tar_directory(SEAL / 'panel', output / 'incoming.tar', {PANEL_MARKER: (marker, 0o644)})

    # Grading ---------------------------------------------------------------------------------------------------
    def evaluate(self, runtime, placement, run: Path) -> dict:
        """Build (if build.sh) and evaluate run/source on every graded case; returns the part summary."""
        c = self.assets(runtime).controller
        image = self.lock()['image']['id']
        docker = c.Docker(runtime['docker_path'], placement.docker_host, placement.cpuset, placement.cgroup_parent)
        try:
            build = c.offline_build(docker, image, run)
        except c.SubmissionFailure as error:
            return {'status': 'submission_failure', 'reason': f'build: {error}'}
        wrapper = run / 'docker-endpoint'
        wrapper.write_text('#!/bin/sh\nexec ' + ' '.join(shlex.quote(x) for x in docker.prefix) + ' "$@"\n')
        wrapper.chmod(0o700)
        report_path = run / 'evaluation.json'
        command = [sys.executable, str(SEAL / 'docker_eval/evaluator/run_docker.py'), '--image', image,
                   '--docker', str(wrapper), '--submission', build['artifact'], '--command', './run.sh',
                   '--manifest', str(SEAL / 'data/manifest.json'), '--output', str(report_path), '--repeats', '3',
                   '--cpuset', placement.cpuset]
        if placement.cgroup_parent:
            command += ['--cgroup-parent', placement.cgroup_parent]
        with (run / 'evaluator.log').open('wb') as log:
            done = subprocess.run(command, stdout=log, stderr=subprocess.STDOUT, timeout=self.grade_timeout)
        if done.returncode or not report_path.exists():
            return {'status': 'infrastructure_failure', 'reason': f'evaluator exit {done.returncode}; see {run}/evaluator.log'}
        report = json.loads(report_path.read_text())
        expected = {case['id'] for case in self.cases()}
        if set(report['aggregates']) != expected or len(report['runs']) != 3 * len(expected):
            return {'status': 'infrastructure_failure', 'reason': 'incomplete evaluation'}
        valid = sum(a['valid'] for a in report['aggregates'].values())
        return {'status': 'ok', 'valid_cases': valid, 'total_cases': len(expected), 'report': str(report_path),
                'build': build.get('status')}

    async def grade(self, runtime, placement, output):
        self.verify()
        source = output / 'source'
        if (source / PANEL_MARKER).exists():
            c = self.assets(runtime).controller
            parts = {}
            for member in json.loads((source / PANEL_MARKER).read_text())['panel']:
                run = output / 'panel' / member
                run.mkdir(parents=True)
                c.validate_and_extract(tar_directory(source / member, run / 'incoming.tar'), run / 'source',
                                       run / 'source.tar')
                parts[member] = self.evaluate(runtime, placement, run)
            return {'exit_status': 'finished', 'kind': 'panel', 'parts': parts}
        return {'exit_status': 'finished', 'kind': 'candidate', 'parts': {'candidate': self.evaluate(runtime, placement, output)}}

    def metrics(self, output, raw):
        parts = {}
        for name, part in raw.get('parts', {}).items():
            aggregates = json.loads(Path(part['report']).read_text())['aggregates'] if part.get('report') else None
            parts[name] = {**part, 'aggregates': aggregates}
        return {'raw': raw, 'kind': raw.get('kind'), 'parts': parts}

    def virtual_best(self, metrics: dict) -> tuple[dict, dict, list[str]]:
        """Per case: the fastest member whose aggregate passes the scorer's own validation."""
        scoring = self.scoring()
        best, winners, uncovered = {}, {}, []
        for case in self.cases():
            options = []
            for member, part in metrics['parts'].items():
                record = (part.get('aggregates') or {}).get(case['id'])
                if record and scoring.validate_measurement(record, scenario=case['scenario'])['valid']:
                    options.append((record['time_seconds'], member, record))
            if not options:
                uncovered.append(case['id'])
                continue
            _, member, record = min(options, key=lambda item: item[0])
            best[case['id']], winners[case['id']] = record, member
        return best, winners, uncovered

    def baseline_ok(self, metrics):
        if metrics.get('kind') != 'panel':
            return False, 'not a panel baseline'
        _, winners, uncovered = self.virtual_best(metrics)
        if uncovered:
            return False, f'no valid panel member on {len(uncovered)} cases: {uncovered[:5]}'
        counts = {m: list(winners.values()).count(m) for m in metrics['parts']}
        return True, 'virtual best covers all %d cases; wins per member %s' % (len(winners), counts)

    def score(self, runtime, metrics, baseline):
        part = metrics['parts'].get('candidate') or {}
        if part.get('status') in (None, 'infrastructure_failure'):
            return {'status': 'unmeasured', 'score': None, 'reason': part.get('reason')}
        if part['status'] != 'ok':
            return {'status': 'invalid_submission', 'score': 0.0, 'reason': part.get('reason')}
        if baseline is None:
            return {'status': 'no_baseline', 'score': None}
        reference, winners, uncovered = self.virtual_best(baseline['metrics'])
        if uncovered:
            return {'status': 'unmeasured', 'score': None, 'reason': 'panel baseline does not cover every case'}
        result = self.scoring().score_submission(self.cases(), part['aggregates'], reference)
        return {'status': result['status'], 'score': result['score'], 'relative_time': result['relative_time'],
                'failed_cases': result['failed_cases'], 'reference': 'panel virtual best', 'winners': winners,
                'detail': result}
