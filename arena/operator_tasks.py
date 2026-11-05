"""Operator-delivered tasks (tasks/operator/<name>/task.json, sealed by scripts/prepare_operator_tasks.py).

Each package brings its own host grader and per-case workers. They run here unchanged apart from the declared
portability edits, inside a grading slot: a docker shim first on the grader's PATH maps the package's ARM64
platform, CPU index and image IDs onto the slot's CPU, cgroup slice and sealed amd64 image, and DOCKER_HOST names the
slot's daemon. A baseline grades the package's weak and strong references in one job; their raw errors measured on
this machine are the 0- and 50-point anchors of the 100-point score (the package's own scale is 0/5/10).
"""

from __future__ import annotations

import json
import math
import os
import shutil
import subprocess
import sys

from pathlib import Path
from types import SimpleNamespace

from .tasks import GIB, Placement, Task, tar_directory, write

ROOT = Path(__file__).resolve().parents[1]
OPS = ROOT / '.runtime-ops'
SPECS = ROOT / 'tasks/operator'
PAIR = 'FRONTIERAM_REFERENCE_PAIR.json'
# Host-side graders use the packages' pinned host NumPy (2.3.5), separate from the orchestration environment.
GRADER_PYTHON = ROOT / '.venv-ops/bin/python'


def score_100(raw: float, weak: float, strong: float) -> float:
    """The packages' 0/5/10 piecewise-linear mapping, times 10: weak reference 0, strong reference 50, exact 100."""
    if raw >= weak:
        return 0.0
    if raw >= strong:
        return 50.0 * (weak - raw) / (weak - strong)
    return 100.0 - 50.0 * raw / strong


class OperatorTask(Task):
    grade_isolation = 'grade_pool'
    paired = False
    grade_timeout = 6 * 3600

    def __init__(self, spec: dict):
        self.spec = spec
        self.name = spec['name']
        self.instruction = spec['instruction']
        self.submission_note = spec['submission_note']
        self.status = spec['status']

    # Sealed material -------------------------------------------------------------------------------------------
    @property
    def sealed(self) -> Path:
        return OPS / self.name

    def lock(self) -> dict:
        return json.loads((self.sealed / 'lock.json').read_text())

    def root(self, package: str = 'operator') -> Path:
        return self.sealed / self.lock()['roots'][package]

    def assets(self, runtime: dict):
        from .tasks import TASKS

        lock = self.lock()
        public = self.sealed / 'public.tar'
        if _sha(public) != lock['public_tar_sha256']:
            raise RuntimeError(f'{self.name}: sealed public package changed')
        return SimpleNamespace(controller=TASKS['exponential'].assets(runtime).controller, public_tar=public, lock=lock)

    def dev_config(self, *args, **kwargs):
        raise ValueError(f'{self.name} is an operator task; develop it with --protocol v6 or v6.1')

    # References ------------------------------------------------------------------------------------------------
    def reference_dir(self, which: str) -> Path:
        return self.root() / self.spec['references'][which]

    def reference_archive(self, runtime: dict, output: Path) -> Path:
        output.mkdir(parents=True, exist_ok=True)
        pair = output / 'pair'
        for which in ('weak', 'strong'):
            shutil.copytree(self.reference_dir(which), pair / which)
        (pair / PAIR).write_text(json.dumps({'task': self.name, 'references': self.spec['references']}) + '\n')
        return tar_directory(pair, output / 'incoming.tar')

    # Grading ---------------------------------------------------------------------------------------------------
    def shim_environment(self, placement: Placement, log: Path) -> dict:
        lock = self.lock()
        if not lock['image']['id']:
            raise RuntimeError(f'{self.name}: amd64 image not built; run prepare_operator_tasks.py --build-images')
        shim_dir = log.parent / 'shim'
        shim_dir.mkdir(exist_ok=True)
        wrapper = shim_dir / 'docker'
        wrapper.write_text(f'#!/bin/sh\nexec {sys.executable} -IS {ROOT / "arena/docker_shim.py"} "$@"\n')
        wrapper.chmod(0o755)
        config = {'platform': 'linux/amd64', 'cpuset': placement.cpuset, 'cgroup_parent': placement.cgroup_parent,
                  'images': {old: lock['image']['id'] for old in lock['image']['replaces']}, 'log': str(log)}
        return {**os.environ, 'PATH': f'{shim_dir}:{os.environ.get("PATH", "/usr/bin:/bin")}',
                'DOCKER_HOST': placement.docker_host, 'FRONTIERAM_REAL_DOCKER': shutil.which('docker'),
                'FRONTIERAM_DOCKER_SHIM': json.dumps(config), 'PYTHONDONTWRITEBYTECODE': '1'}

    def run_grader(self, source: Path, out: Path, placement: Placement) -> dict:
        raise NotImplementedError

    async def grade(self, runtime, placement, output):
        source = output / 'source'
        if (source / PAIR).exists():
            parts = {which: self.run_grader(source / which, output / which, placement) for which in ('weak', 'strong')}
            return {'exit_status': 'finished', 'kind': 'reference_pair', 'parts': parts}
        return {'exit_status': 'finished', 'kind': 'candidate', 'parts': {'candidate': self.run_grader(source, output / 'candidate', placement)}}

    def metrics(self, output, raw):
        return {'raw': raw, 'kind': raw.get('kind'), 'parts': raw.get('parts', {})}

    def baseline_ok(self, metrics):
        parts = metrics['parts']
        weak, strong = parts.get('weak', {}), parts.get('strong', {})
        ok = (weak.get('status') == strong.get('status') == 'ok' and _finite(weak.get('raw_error'))
              and _finite(strong.get('raw_error')) and weak['raw_error'] > strong['raw_error'] > 0)
        reason = ('both references valid; local anchors weak %.6g > strong %.6g' % (weak['raw_error'], strong['raw_error'])
                  if ok else f'reference pair not valid: weak={weak.get("status")} strong={strong.get("status")}')
        return ok, reason

    def score(self, runtime, metrics, baseline):
        part = metrics['parts'].get('candidate') or {}
        status = part.get('status')
        if status == 'infrastructure_failure' or status is None:
            return {'status': 'unmeasured', 'score': None, 'reason': part.get('reason')}
        if status != 'ok':
            return {'status': 'submission_failure', 'score': 0.0, 'reason': part.get('reason'), 'raw_error': part.get('raw_error')}
        result = {'status': 'ok', 'raw_error': part['raw_error'], 'package_score': part.get('package_score')}
        if baseline is not None:
            weak = baseline['metrics']['parts']['weak']['raw_error']
            strong = baseline['metrics']['parts']['strong']['raw_error']
            result.update(score=score_100(part['raw_error'], weak, strong), anchors={'weak': weak, 'strong': strong})
        else:
            result['score'] = None
        return result


class GalaxyGravity(OperatorTask):
    grade_memory = 4 * GIB

    def run_grader(self, source, out, placement):
        root = self.root()
        official = json.loads((root / 'private/scoring-config.json').read_text())
        cases = [dict(c, input=str(root / c['input']), truth=str(root / c['truth'])) for c in official['cases']]
        config = {'image': self.lock()['image']['id'], 'cpu': int(placement.cpuset), 'cpu_limit_s': official['cpu_limit_s'],
                  'repeats': official['repeats'], 'mode': 'calibration', 'cases': cases}
        out.parent.mkdir(parents=True, exist_ok=True)
        config_path = out.parent / f'{out.name}-config.json'
        config_path.write_text(json.dumps(config, indent=2) + '\n')
        grader = root / 'tasks/galaxy-gravity/harbor/tests/grade.py'
        env = self.shim_environment(placement, out.parent / f'{out.name}-docker.jsonl')
        done = subprocess.run([str(GRADER_PYTHON), '-IB', str(grader), '--submission', str(source), '--output', str(out),
                               '--config', str(config_path)], env=env, capture_output=True, text=True, timeout=self.grade_timeout)
        (out.parent / f'{out.name}-grader.log').write_text(done.stdout + done.stderr)
        summary = json.loads((out / 'summary.json').read_text()) if (out / 'summary.json').exists() else {}
        status = summary.get('status', 'infrastructure_failure')
        raw = summary.get('raw_error')
        records = summary.get('records', [])
        return {'status': status, 'raw_error': float(raw) if raw is not None else None, 'reason': summary.get('reason'),
                'package_score': None, 'calls': len(records),
                'max_cpu_s': max((r.get('cpu_s') or 0 for r in records), default=None),
                'cpu_limit_s': official['cpu_limit_s']}


class PolarizationDemosaicking(OperatorTask):
    grade_memory = 10 * GIB
    # Each scene has a 1.000-second wall limit including interpreter start-up, so no other grader runs meanwhile.
    grade_isolation = 'grade_pool'

    def run_grader(self, source, out, placement):
        root = self.root()
        grader = root / 'harbor/polarization-demosaicking/grade_submission.py'
        out.parent.mkdir(parents=True, exist_ok=True)
        env = self.shim_environment(placement, out.parent / f'{out.name}-docker.jsonl')
        done = subprocess.run([str(GRADER_PYTHON), '-B', str(grader), '--submission', str(source), '--output', str(out),
                               '--mode', 'formal'], cwd=grader.parent, env=env, capture_output=True, text=True,
                              timeout=self.grade_timeout)
        (out.parent / f'{out.name}-grader.log').write_text(done.stdout + done.stderr)
        summary = json.loads((out / 'summary.json').read_text()) if (out / 'summary.json').exists() else {}
        status = summary.get('status', 'infrastructure_failure')
        if status not in ('ok', 'infrastructure_failure'):
            status = 'submission_failure'
        rows = summary.get('scene_rows', [])
        return {'status': status, 'raw_error': summary.get('mean_rmse'), 'reason': summary.get('reason'),
                'package_score': summary.get('formal_score'), 'cases_passed': summary.get('cases_passed'),
                'expected_cases': summary.get('expected_cases'),
                'max_wall_s': max((r.get('wall_s') or 0 for r in rows), default=None),
                'scene_statuses': {r['scene']: r['status'] for r in rows}}


class SpeciesTree(OperatorTask):
    grade_memory = 6 * GIB

    def reference_dir(self, which: str) -> Path:
        method = self.spec['references'][which].split(':', 1)[1]
        target = self.sealed / 'references' / method
        if not target.exists():
            subprocess.run([sys.executable, '-B', str(self.root() / 'operator/references/materialize.py'), method,
                            str(target)], check=True, capture_output=True, timeout=120)
        return target

    def run_grader(self, source, out, placement):
        driver = ROOT / 'arena/operator_drivers/species_tree_batch.py'
        out.parent.mkdir(parents=True, exist_ok=True)
        env = self.shim_environment(placement, out.parent / f'{out.name}-docker.jsonl')
        done = subprocess.run([str(GRADER_PYTHON), '-B', str(driver), str(self.root() / 'operator'), str(source), str(out)],
                              env=env, capture_output=True, text=True, timeout=self.grade_timeout)
        (out.parent / f'{out.name}-grader.log').write_text(done.stdout + done.stderr)
        summary = json.loads((out / 'summary.json').read_text()) if (out / 'summary.json').exists() else {}
        status = summary.get('status')
        # Per-case algorithm failures count as 98 missed splits and still give a valid batch error.
        mapped = {'ok': 'ok', 'algorithm_failures': 'ok', 'submission_failure': 'submission_failure'}.get(status, 'infrastructure_failure')
        reason = summary.get('reason') or (done.stderr.strip().splitlines()[-1] if done.returncode and done.stderr.strip() else None)
        return {'status': mapped, 'raw_error': summary.get('mean_nrf'), 'reason': reason, 'package_score': None,
                'missed_splits': summary.get('missed_splits'), 'failed_cases': summary.get('failed_cases'),
                'scope': 'public development cases (not the official private set)'}


class DevelopmentOnlyTask(OperatorTask):
    """Development wiring only: the delivered material cannot be graded on this machine (see task.json status)."""

    grade_memory = 4 * GIB

    def run_grader(self, source, out, placement):
        return {'status': 'infrastructure_failure', 'raw_error': None,
                'reason': f'{self.name}: {self.spec.get("status_reason", "no grader available")}'}


def _sha(path: Path) -> str:
    import hashlib

    with path.open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def _finite(value) -> bool:
    return isinstance(value, (int, float)) and math.isfinite(value)


CLASSES = {'galaxy-gravity': GalaxyGravity, 'polarization-demosaicking': PolarizationDemosaicking,
           'species-tree': SpeciesTree}


def operator_tasks() -> dict[str, OperatorTask]:
    tasks = {}
    for path in sorted(SPECS.glob('*/task.json')):
        spec = json.loads(path.read_text())
        tasks[spec['name']] = CLASSES.get(spec['name'], DevelopmentOnlyTask)(spec)
    return tasks
