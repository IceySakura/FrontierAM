"""exponential-v2: dataset v2 graded against the frozen human-SOTA panel (tasks/exponential/references/r1).

The v1 evaluator, tools, image and scorer run unchanged on a sealed copy (.runtime-exp2, scripts/seal/prepare_exponential_v2.py);
only the manifest, the cases and the reference differ. A baseline grades every panel member in one job on one slot;
a candidate is then scored against the per-case virtual best: for each case the member with the shortest median time
among those whose three repetitions all passed. The v1 scorer turns those ratios into 100 / (1 + R), so the virtual
best itself is 50 and the 50/100 convention holds with the panel as the reference.
"""

from __future__ import annotations

import hashlib
import math
import importlib.util
import os
import json
import shlex
import subprocess
import sys

from pathlib import Path
from types import SimpleNamespace

from .tasks import GIB, Task, tar_directory

ROOT = Path(__file__).resolve().parents[1]
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
    # Timing ratios against the panel's earlier measurement; concurrency allowed, the load probe flags anomalies.
    grade_isolation = 'shared'
    paired = False
    # The public-feedback grader is the same task sealed elsewhere (public cases, one repeat), named by the variable.
    seal_env, default_seal = 'FRONTIERAM_EXP2_SEAL', ROOT / '.runtime-exp2'

    @property
    def seal(self) -> Path:
        return Path(os.environ.get(self.seal_env) or self.default_seal)

    # Sealed material -------------------------------------------------------------------------------------------
    def lock(self) -> dict:
        return json.loads((self.seal / 'lock.json').read_text())

    def verify(self) -> dict:
        lock = self.lock()
        for relative, expected in lock['files'].items():
            path = self.seal / relative
            if path.is_symlink() or not path.is_file() or _sha(path) != expected:
                raise RuntimeError(f'exponential-v2: sealed file changed or missing: {relative}')
        present = {str(p.relative_to(self.seal)) for p in (self.seal / 'panel').rglob('*') if not p.is_dir()}
        if present != {r for r in lock['files'] if r.startswith('panel/')}:
            raise RuntimeError('exponential-v2: sealed panel holds files outside the lock')
        return lock

    def assets(self, runtime: dict):
        from .tasks import TASKS

        lock = self.lock()
        public = self.seal / 'public.tar'
        if _sha(public) != lock['public_tar_sha256']:
            raise RuntimeError('exponential-v2: sealed public package changed')
        return SimpleNamespace(controller=TASKS['exponential'].assets(runtime).controller, public_tar=public, lock=lock)

    def scoring(self):
        spec = importlib.util.spec_from_file_location('frontieram_exp2_scoring', self.seal / 'public/tools/scoring.py')
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        return module

    def cases(self) -> list[dict]:
        return json.loads((self.seal / 'data/manifest.json').read_text())['cases']

    def dev_config(self, *args, **kwargs):
        raise ValueError('exponential-v2 is developed with --protocol v6 or v6.1')

    # Reference -------------------------------------------------------------------------------------------------
    def reference_archive(self, runtime: dict, output: Path) -> Path:
        lock = self.verify()
        output.mkdir(parents=True, exist_ok=True)
        # The panel together exceeds one submission's size bound, so the reference carries only the membership;
        # grade() reads each member from the hash-verified seal and validates it as its own submission.
        marker = json.dumps({'task': self.name, 'panel': lock['panel']}).encode() + b'\n'
        empty = output / 'panel-reference'
        empty.mkdir(exist_ok=True)
        return tar_directory(empty, output / 'incoming.tar', {PANEL_MARKER: (marker, 0o644)})

    # Grading ---------------------------------------------------------------------------------------------------
    def suite(self, runtime) -> list[dict]:
        """Every graded case with a cost proxy (nonzeros x requests) for balancing shards across slots."""
        return [dict(c, cost=float(c.get('nnz', 1)) * c.get('request_count', 1)) for c in self.cases()]

    def manifest_for(self, out: Path, case_ids) -> Path:
        """The sealed manifest, or a copy restricted to case_ids with absolute case paths (one shard of a suite)."""
        sealed = self.seal / 'data/manifest.json'
        if not case_ids:
            return sealed
        manifest = json.loads(sealed.read_text())
        wanted = set(case_ids)
        manifest['cases'] = [dict(c, file=str(self.seal / 'data' / c['file'])) for c in manifest['cases'] if c['id'] in wanted]
        if self.lock().get('repeats', 3) == 1:  # public-feedback seal: cheapest sessions first for capped tests
            manifest['cases'].sort(key=lambda c: (float(c.get('nnz', 1)) * c.get('request_count', 1), c['id']))
        if len(manifest['cases']) != len(wanted):
            raise ValueError('case_ids name cases outside the sealed manifest')
        path = out / 'shard-manifest.json'
        path.write_text(json.dumps(manifest, indent=1) + '\n')
        return path

    def evaluate(self, runtime, placement, run: Path, manifest: Path | None = None) -> dict:
        """Build (if build.sh) and evaluate run/source on the graded cases (or a shard); returns the part summary."""
        manifest = manifest or self.seal / 'data/manifest.json'
        repeats = self.lock().get('repeats', 3)
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
        command = [sys.executable, str(self.seal / 'docker_eval/evaluator/run_docker.py'), '--image', image,
                   '--docker', str(wrapper), '--submission', build['artifact'], '--command', './run.sh',
                   '--manifest', str(manifest), '--output', str(report_path), '--repeats', str(repeats),
                   '--cpuset', placement.cpuset]
        if placement.cgroup_parent:
            command += ['--cgroup-parent', placement.cgroup_parent]
        with (run / 'evaluator.log').open('wb') as log:
            done = subprocess.run(command, stdout=log, stderr=subprocess.STDOUT, timeout=self.grade_timeout)
        if done.returncode or not report_path.exists():
            return {'status': 'infrastructure_failure', 'reason': f'evaluator exit {done.returncode}; see {run}/evaluator.log'}
        report = json.loads(report_path.read_text())
        expected = {case['id'] for case in json.loads(manifest.read_text())['cases']}
        if set(report['aggregates']) != expected or len(report['runs']) != repeats * len(expected):
            return {'status': 'infrastructure_failure', 'reason': 'incomplete evaluation'}
        valid = sum(a['valid'] for a in report['aggregates'].values())
        return {'status': 'ok', 'valid_cases': valid, 'total_cases': len(expected), 'report': str(report_path),
                'build': build.get('status')}

    async def grade(self, runtime, placement, output, case_ids=None):
        self.verify()
        source = output / 'source'
        manifest = self.manifest_for(output, case_ids)
        if (source / PANEL_MARKER).exists():
            c = self.assets(runtime).controller
            parts = {}
            for member in json.loads((source / PANEL_MARKER).read_text())['panel']:
                run = output / 'panel' / member
                run.mkdir(parents=True)
                c.validate_and_extract(tar_directory(self.seal / 'panel' / member, run / 'incoming.tar'), run / 'source',
                                       run / 'source.tar')
                parts[member] = self.evaluate(runtime, placement, run, manifest)
            return {'exit_status': 'finished', 'kind': 'panel', 'parts': parts}
        return {'exit_status': 'finished', 'kind': 'candidate', 'parts': {'candidate': self.evaluate(runtime, placement, output, manifest)}}

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
        speed = speed_ratio(self.cases(), part, baseline['metrics']) if result['status'] == 'ok' else None
        return {'status': result['status'], 'score': result['score'], 'relative_time': result['relative_time'],
                'failed_cases': result['failed_cases'], 'reference': 'panel virtual best', 'winners': winners,
                'speed_ratio': speed, 'detail': result}


def _runs(part: dict) -> dict[str, list[dict]]:
    by_case: dict[str, list[dict]] = {}
    for run in json.loads(Path(part['report']).read_text())['runs']:
        by_case.setdefault(run['id'], []).append(run)
    return by_case


def speed_ratio(cases: list[dict], candidate: dict, panel_metrics: dict, replicates: int = 2000, seed: int = 0) -> dict:
    """Supplementary to 100/(1+R) (DESIGN section 4): S = exp(-mean log(T_candidate / T_virtual_best)), mothers equally
    weighted (cases equally within a mother), with a hierarchical bootstrap: resample mothers, cases within mothers,
    and repetitions within each case for the candidate and every panel member, recomputing medians and the virtual
    best inside every replicate. S > 1 means faster than the panel. Percentile 95% interval."""
    import random
    import statistics

    rng = random.Random(seed)
    scored = [c for c in cases if c['role'] == 'scored']
    mine = _runs(candidate)
    members = {m: _runs(p) for m, p in panel_metrics['parts'].items() if p.get('report')}

    def valid_times(runs):
        return [r['time_seconds'] for r in runs] if runs and all(r['valid'] for r in runs) else None

    def log_ratio(case, resample: bool) -> float | None:
        pick = (lambda xs: [rng.choice(xs) for _ in xs]) if resample else (lambda xs: xs)
        own = valid_times(mine.get(case['id']))
        options = [valid_times(runs.get(case['id'])) for runs in members.values()]
        options = [statistics.median(pick(t)) for t in options if t]
        if not own or not options:
            return None
        return math.log(statistics.median(pick(own)) / min(options))

    by_mother: dict[str, list[dict]] = {}
    for case in scored:
        by_mother.setdefault(case['mother'], []).append(case)

    def estimate(resample: bool) -> float | None:
        mothers = list(by_mother)
        chosen = [rng.choice(mothers) for _ in mothers] if resample else mothers
        means = []
        for mother in chosen:
            group = by_mother[mother]
            group = [rng.choice(group) for _ in group] if resample else group
            logs = [log_ratio(c, resample) for c in group]
            if any(v is None for v in logs):
                return None
            means.append(sum(logs) / len(logs))
        return math.exp(-sum(means) / len(means))

    point = estimate(False)
    if point is None:
        return None
    draws = sorted(v for v in (estimate(True) for _ in range(replicates)) if v is not None)
    return {'S': point, 'ci95': [draws[int(0.025 * (len(draws) - 1))], draws[int(0.975 * (len(draws) - 1))]],
            'replicates': len(draws), 'meaning': 'geometric speed ratio vs panel virtual best; >1 is faster'}


class ExponentialV3(ExponentialV2):
    """Protocol v6.3 version (scripts/seal/prepare_exp_v3.py): per-session limits for the evaluation time caps
    (results/EVAL_TIME_LIMITS.md) and a samples-only workspace; same data, panel, evaluator and scorer otherwise."""

    name = 'exponential-v3'
    instruction = ExponentialV2.instruction
    seal_env, default_seal = 'FRONTIERAM_EXP3_SEAL', ROOT / '.runtime-exp3'
