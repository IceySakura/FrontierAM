"""Operator-delivered tasks (tasks/operator/<name>/task.json, sealed by scripts/seal/prepare_operator_tasks.py).

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
import time

from pathlib import Path
from types import SimpleNamespace

from .tasks import GIB, Placement, Task, tar_directory, write

ROOT = Path(__file__).resolve().parents[1]
OPS = ROOT / '.runtime-ops'
SPECS = ROOT / 'tasks'  # tasks/<family>/versions/<version>/task.json
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
        path = self.spec['references'][which]
        # 'repo:<path>' names a reference kept in this repository (tasks/<family>/references/r<k>/).
        return ROOT / path[len('repo:'):] if path.startswith('repo:') else self.root() / path

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
        mode, repeats = 'calibration', official['repeats']
        # FRONTIERAM_GALAXY_DATA=public grades the public snapshots, every call even after a failure (v6.3 feedback).
        if os.environ.get('FRONTIERAM_GALAXY_DATA') == 'public':
            public = self.root('public') / 'public-dev'
            cases = [dict(c, input=str(public / c['input']), truth=str(public / c['truth']))
                     for c in json.loads((public / 'manifest.json').read_text())['cases']]
            mode, repeats = 'diagnostic', 1
        config = {'image': self.lock()['image']['id'], 'cpu': int(placement.cpuset), 'cpu_limit_s': official['cpu_limit_s'],
                  'repeats': repeats, 'mode': mode, 'cases': cases}
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


class GalaxyGravityV2(GalaxyGravity):
    """galaxy-gravity at 131,072 particles (scripts/seal/prepare_galaxy_v2.py); same grader, references and budget."""


class GalaxyGravityV3(GalaxyGravityV2):
    """Protocol v6.3 version (scripts/seal/prepare_v63.py): the task author's references (r1, two Treecode2 settings) with
    a log-scale mapping, since accuracies span many orders of magnitude (results/BASELINE_POLICY.md)."""

    def score(self, runtime, metrics, baseline):
        result = super().score(runtime, metrics, baseline)
        if result.get('status') == 'ok' and baseline is not None:
            from .public_v63 import galaxy_score
            result['score'] = galaxy_score(result['raw_error'], result['anchors']['weak'], result['anchors']['strong'])
        return result


class PolarizationDemosaicking(OperatorTask):
    grade_memory = 10 * GIB
    # Each scene has a 1.000-second wall limit including interpreter start-up, about 0.6 s of which a typical
    # submission uses; with development sessions running on the L3-sharing dev cores, scenes spiked to 0.9-1.0 s
    # (tasks/polarization-demosaicking/analysis/polarization-first-scene.json), so no development may run while it grades;
    # other gradings may (grade-to-grade interference is negligible on the paired layout, docs/platform/CONCURRENCY.md 7).
    grade_isolation = 'dev_idle'

    # FRONTIERAM_POLAR_DATA=wen grades the public-feedback scenes instead (protocol v6.2; results/OPEN_ITEMS.md).
    feedback = OPS / 'polarization-demosaicking-feedback'

    def warm_image(self, placement, record: Path) -> None:
        """Read the task image's layer files on the slot's daemon into the page cache before any timed scene.

        A cold cache adds 0.2-0.5 s to the first scene's interpreter start-up and imports (1.000 s limit), while warm
        runs show no first-scene penalty (tasks/polarization-demosaicking/analysis/polarization-first-scene.json). Untimed, and the submission is not run.
        """
        start, files, size = time.monotonic(), 0, 0
        inspect = subprocess.run(['docker', '-H', placement.docker_host, 'image', 'inspect', self.lock()['image']['id'],
                                  '--format', '{{index .GraphDriver.Data "LowerDir"}}:{{index .GraphDriver.Data "UpperDir"}}'],
                                 capture_output=True, text=True, check=True)
        for layer in filter(None, inspect.stdout.strip().split(':')):
            for base, _, names in os.walk(layer):
                for name in names:
                    path = os.path.join(base, name)
                    if os.path.islink(path) or not os.path.isfile(path):
                        continue
                    with open(path, 'rb', buffering=0) as stream:
                        while chunk := stream.read(1 << 20):
                            size += len(chunk)
                    files += 1
        record.write_text(json.dumps({'files': files, 'bytes': size, 'seconds': time.monotonic() - start}) + '\n')

    def run_grader(self, source, out, placement):
        root = self.root()
        out.parent.mkdir(parents=True, exist_ok=True)
        self.warm_image(placement, out.parent / f'{out.name}-warmup.json')
        if os.environ.get('FRONTIERAM_POLAR_DATA') == 'wen':
            return self.run_feedback(source, out, placement)
        grader = root / 'harbor/polarization-demosaicking/grade_submission.py'
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


    def run_feedback(self, source, out, placement):
        driver = ROOT / 'arena/operator_drivers/polarization_feedback.py'
        out.parent.mkdir(parents=True, exist_ok=True)
        env = self.shim_environment(placement, out.parent / f'{out.name}-docker.jsonl')
        done = subprocess.run([str(GRADER_PYTHON), '-B', str(driver), str(self.root() / 'harbor/polarization-demosaicking'),
                               str(source), str(out), str(self.feedback)], env=env, capture_output=True, text=True,
                              timeout=self.grade_timeout)
        (out.parent / f'{out.name}-grader.log').write_text(done.stdout + done.stderr)
        summary = json.loads((out / 'summary.json').read_text()) if (out / 'summary.json').exists() else {}
        rows = summary.get('scene_rows', [])
        return {'status': summary.get('status', 'infrastructure_failure'), 'raw_error': summary.get('mean_rmse'),
                'reason': summary.get('reason'), 'cases_passed': summary.get('cases_passed'),
                'expected_cases': summary.get('expected_cases'), 'max_wall_s': max((r.get('wall_s') or 0 for r in rows), default=None),
                'scene_rmse': {r['scene']: r.get('mean_rmse') for r in rows}, 'scene_statuses': {r['scene']: r['status'] for r in rows}}


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
        private = self.root() / 'private-data'
        command = [str(GRADER_PYTHON), '-B', str(driver), str(self.root() / 'operator'), str(source), str(out)]
        done = subprocess.run(command + ([str(private)] if private.is_dir() else []),
                              env=env, capture_output=True, text=True, timeout=self.grade_timeout)
        (out.parent / f'{out.name}-grader.log').write_text(done.stdout + done.stderr)
        summary = json.loads((out / 'summary.json').read_text()) if (out / 'summary.json').exists() else {}
        status = summary.get('status')
        # Per-case algorithm failures count as 98 missed splits and still give a valid batch error.
        mapped = {'ok': 'ok', 'algorithm_failures': 'ok', 'submission_failure': 'submission_failure'}.get(status, 'infrastructure_failure')
        reason = summary.get('reason') or (done.stderr.strip().splitlines()[-1] if done.returncode and done.stderr.strip() else None)
        return {'status': mapped, 'raw_error': summary.get('mean_nrf'), 'reason': reason, 'package_score': None,
                'missed_splits': summary.get('missed_splits'), 'failed_cases': summary.get('failed_cases'),
                'scope': {'formal': '50 formal hidden cases', 'public': 'public development cases (pipeline check)'}
                .get(summary.get('scope'), summary.get('scope'))}


class SpeciesTreeV2(SpeciesTree):
    """species-tree rebuilt at 250 taxa (scripts/seal/prepare_species_v2.py), graded in shards like the conic suite."""
    # FRONTIERAM_SPECIES_V2_DATA=public grades the shipped public cases instead (protocol v6.2 public feedback).
    grade_timeout = 8 * 3600

    def reference_dir(self, which: str) -> Path:
        return SpeciesTree.reference_dir(self, which)

    def data(self) -> Path:
        public = os.environ.get('FRONTIERAM_SPECIES_V2_DATA') == 'public'
        return self.root('public') / 'public-dev' if public else self.sealed / 'private'

    def suite(self, runtime=None) -> list[dict]:
        data = self.data()
        limit = self.lock().get('case_limit_s', 120)
        if data.name == 'public-dev' and not (data / 'manifest.json').exists():
            return [{'id': d.name, 'group': 'all', 'time_limit_seconds': limit}
                    for d in sorted((data / 'cases').iterdir())]
        return [{'id': c['case'], 'group': 'all', 'time_limit_seconds': limit}
                for c in json.loads((data / 'manifest.json').read_text())['cases']]

    async def grade(self, runtime, placement, output, case_ids=None):
        source = output / 'source'
        if (source / PAIR).exists():
            parts = {which: self.run_grader(source / which, output / which, placement, case_ids)
                     for which in ('weak', 'strong')}
            return {'exit_status': 'finished', 'kind': 'reference_pair', 'parts': parts}
        return {'exit_status': 'finished', 'kind': 'candidate',
                'parts': {'candidate': self.run_grader(source, output / 'candidate', placement, case_ids)}}

    def run_grader(self, source, out, placement, case_ids=None):
        driver = ROOT / 'arena/operator_drivers/species_tree_v2_batch.py'
        out.parent.mkdir(parents=True, exist_ok=True)
        env = self.shim_environment(placement, out.parent / f'{out.name}-docker.jsonl')
        data = self.data()
        if data.name == 'public-dev' and not (data / 'manifest.json').exists():
            raise RuntimeError('public-feedback grading needs the public manifest (prepare_species_v2.py)')
        command = [str(GRADER_PYTHON), '-B', str(driver), str(self.root() / 'operator'), str(source), str(out), str(data)]
        if case_ids:
            ids = out.parent / f'{out.name}-cases.json'
            ids.write_text(json.dumps(list(case_ids)) + '\n')
            command += ['--cases', str(ids)]
        done = subprocess.run(command, env=env, capture_output=True, text=True, timeout=self.grade_timeout)
        (out.parent / f'{out.name}-grader.log').write_text(done.stdout + done.stderr)
        summary = json.loads((out / 'summary.json').read_text()) if (out / 'summary.json').exists() else {}
        mapped = {'ok': 'ok', 'algorithm_failures': 'ok', 'submission_failure': 'submission_failure'}.get(
            summary.get('status'), 'infrastructure_failure')
        reason = summary.get('reason') or (done.stderr.strip().splitlines()[-1] if done.returncode and done.stderr.strip() else None)
        return {'status': mapped, 'raw_error': summary.get('mean_nrf'), 'reason': reason, 'package_score': None,
                'missed_splits': summary.get('missed_splits'), 'denominator': summary.get('denominator'),
                'failed_cases': summary.get('failed_cases'), 'cases': summary.get('expected_cases'),
                'scope': summary.get('scope')}


class TrajectoryCovariance(OperatorTask):
    grade_memory = 4 * GIB
    grade_timeout = 12 * 3600
    # Parent chains per setting in the default case set (all five prefixes each): 100 of 1,000 gives 3,000 of 30,000.
    subset_parents = 100

    # Protocol v6.2 public feedback (results/OPEN_ITEMS.md): FRONTIERAM_TRAJ_POOL=feedback selects the last 10 parents
    # per setting (all five prefixes, 300 inputs). They are never part of the formal subset, so feedback reveals
    # nothing about graded inputs; agents see per-case errors only, never the trajectories.
    feedback_parents = 10

    def case_ids(self) -> list[str]:
        """Fixed stratified subset: per setting, the first parents in ID order, with every prefix of each."""
        manifest = self.root() / 'results/mcmc-long-run-covariance/formal-v1/private/manifest.json'
        cases = json.loads(manifest.read_text())['cases']
        feedback = os.environ.get('FRONTIERAM_TRAJ_POOL') == 'feedback'
        lock = self.lock()
        subset, pool = lock.get('subset_parents', self.subset_parents), lock.get('feedback_parents', self.feedback_parents)
        chosen = set()
        for model in sorted({c['model'] for c in cases}):
            parents = sorted({c['parent'] for c in cases if c['model'] == model})
            if len(parents) < subset + pool:
                raise RuntimeError('trajectory: too few parents to keep the feedback pool disjoint from the formal subset')
            chosen.update(parents[-pool:] if feedback else parents[:subset])
        return [c['id'] for c in cases if c['parent'] in chosen]

    def suite(self, runtime=None) -> list[dict]:
        """The formal subset as shardable cases (v6.3 versions grade it on several slots; scripts/grade/trajectory-suite-score.py)."""
        manifest = self.root() / 'results/mcmc-long-run-covariance/formal-v1/private/manifest.json'
        models = {c['id']: c['model'] for c in json.loads(manifest.read_text())['cases']}
        limit = self.lock().get('case_limit_s', 30)
        return [{'id': i, 'group': models[i], 'time_limit_seconds': limit} for i in self.case_ids()]

    async def grade(self, runtime, placement, output, case_ids=None):
        source = output / 'source'
        if (source / PAIR).exists():
            parts = {which: self.run_grader(source / which, output / which, placement, case_ids) for which in ('weak', 'strong')}
            return {'exit_status': 'finished', 'kind': 'reference_pair', 'parts': parts}
        return {'exit_status': 'finished', 'kind': 'candidate',
                'parts': {'candidate': self.run_grader(source, output / 'candidate', placement, case_ids)}}

    def run_grader(self, source, out, placement, case_ids=None):
        out.parent.mkdir(parents=True, exist_ok=True)
        cases = out.parent / f'{out.name}-cases.json'
        cases.write_text(json.dumps(list(case_ids) if case_ids else self.case_ids()) + '\n')
        driver = ROOT / 'arena/operator_drivers/trajectory_batch.py'
        env = {**os.environ, 'DOCKER_HOST': placement.docker_host}
        done = subprocess.run([str(GRADER_PYTHON), '-B', str(driver), str(self.root()), str(source), str(out),
                               '--cases', str(cases), '--image', self.lock()['image']['id'], '--cpuset', placement.cpuset,
                               '--cgroup-parent', placement.cgroup_parent,
                               '--case-limit', str(self.lock().get('case_limit_s', 30))],
                              env=env, capture_output=True, text=True, timeout=self.grade_timeout)
        (out.parent / f'{out.name}-grader.log').write_text(done.stdout + done.stderr)
        summary = json.loads((out / 'summary.json').read_text()) if (out / 'summary.json').exists() else {}
        status = summary.get('status', 'infrastructure_failure')
        return {'status': status, 'raw_error': summary.get('raw_error'), 'reason': summary.get('reason'),
                'package_score': None, 'cases': summary.get('cases'), 'failures': summary.get('failures'),
                'cell_sums': summary.get('cell_sums'),
                'max_wall_s': summary.get('max_wall_s'), 'max_cpu_s': summary.get('max_cpu_s'),
                'scope': f'{len(json.loads(cases.read_text()))} of 30,000 formal inputs (fixed stratified subset)'}


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


CLASSES = {'galaxy-gravity': GalaxyGravity, 'galaxy-gravity-v2': GalaxyGravityV2, 'galaxy-gravity-v3': GalaxyGravityV3,
           'polarization-demosaicking': PolarizationDemosaicking, 'polarization-demosaicking-v2': PolarizationDemosaicking,
           'species-tree': SpeciesTree, 'species-tree-v2': SpeciesTreeV2, 'species-tree-v3': SpeciesTreeV2,
           'trajectory-covariance': TrajectoryCovariance, 'trajectory-covariance-v2': TrajectoryCovariance}


def operator_tasks() -> dict[str, OperatorTask]:
    # Repository-native draft tasks do not pretend to have a sealed operator
    # package or a calibrated deployment on this host.
    from .operator_drivers.ihvp_sequence import IHVPSequence
    from .operator_drivers.ihvp_hard_v3 import IHVPHardV3
    handlers = {**CLASSES, 'ihvp-sequence': IHVPSequence, 'ihvp-sequence-v2': IHVPSequence,
                'ihvp-sequence-v3': IHVPHardV3}
    tasks = {}
    for path in sorted(SPECS.glob('*/versions/*/task.json')):
        spec = json.loads(path.read_text())
        tasks[spec['name']] = handlers.get(spec['name'], DevelopmentOnlyTask)(spec)
    return tasks
