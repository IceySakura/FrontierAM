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
    grade_isolation = 'shared'
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
            # Graders require a README.md beside solution.py; maintenance references may ship without one.
            if not (pair / which / 'README.md').exists():
                (pair / which / 'README.md').write_text(f'{which} reference of {self.name}, graded for the local anchors.\n')
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


class GalaxyGravityV4(GalaxyGravityV3):
    """v3 with the author's B.4 data (scripts/seal/prepare_galaxy_v4.py): 80 snapshots at four scales, each with its own
    CPU limit, graded in shards with the B.4 grader's diagnostic mode (any case subset, per-case limits, three calls
    each); scripts/grade/galaxy-v4-suite-score.py averages the four snapshots of each scale x family cell and the 20
    cells with equal weight, then applies the v3 log mapping.

    FRONTIERAM_GALAXY_DATA=public grades the feedback pool. FRONTIERAM_GALAXY_V4_TIMING=<seconds> replaces every case
    limit by that value (limit calibration, scripts/analyze/galaxy-v4-calibrate.py)."""
    grade_timeout = 8 * 3600

    def cases(self) -> list[dict]:
        if os.environ.get('FRONTIERAM_GALAXY_DATA') == 'public':
            base = self.root('public') / 'public-dev'
            cases = json.loads((base / 'manifest.json').read_text())['cases']
        else:
            base = self.root()
            cases = json.loads((base / 'private/scoring-config.json').read_text())['cases']
        timing = os.environ.get('FRONTIERAM_GALAXY_V4_TIMING')
        out = []
        for c in cases:
            limit = float(timing) if timing else c.get('cpu_limit_s')
            if limit is None:
                raise RuntimeError(f'{self.name}: {c["case"]} has no calibrated CPU limit (run galaxy-v4-calibrate.py)')
            out.append({k: c[k] for k in ('case', 'family', 'n', 'input_sha256', 'truth_sha256')} |
                       {'input': str(base / c['input']), 'truth': str(base / c['truth']), 'cpu_limit_s': limit})
        return out

    def suite(self, runtime=None) -> list[dict]:
        return [{'id': c['case'], 'group': f'{c["n"]}/{c["family"]}', 'time_limit_seconds': c['cpu_limit_s'],
                 'cost': 3 * (c['cpu_limit_s'] + 2)} for c in self.cases()]

    async def grade(self, runtime, placement, output, case_ids=None):
        source = output / 'source'
        if (source / PAIR).exists():
            parts = {which: self.run_grader(source / which, output / which, placement, case_ids) for which in ('weak', 'strong')}
            return {'exit_status': 'finished', 'kind': 'reference_pair', 'parts': parts}
        return {'exit_status': 'finished', 'kind': 'candidate',
                'parts': {'candidate': self.run_grader(source, output / 'candidate', placement, case_ids)}}

    def run_grader(self, source, out, placement, case_ids=None):
        root = self.root()
        cases = self.cases()
        if case_ids:
            wanted = set(case_ids)
            cases = [c for c in cases if c['case'] in wanted]
        config = {'image': self.lock()['image']['id'], 'cpu': int(placement.cpuset), 'repeats': 3, 'mode': 'diagnostic',
                  'cases': cases}
        out.parent.mkdir(parents=True, exist_ok=True)
        config_path = out.parent / f'{out.name}-config.json'
        config_path.write_text(json.dumps(config, indent=2) + '\n')
        grader = root / 'tasks/galaxy-gravity/harbor/tests/grade.py'
        env = self.shim_environment(placement, out.parent / f'{out.name}-docker.jsonl')
        done = subprocess.run([str(GRADER_PYTHON), '-IB', str(grader), '--submission', str(source), '--output', str(out),
                               '--config', str(config_path)], env=env, capture_output=True, text=True, timeout=self.grade_timeout)
        (out.parent / f'{out.name}-grader.log').write_text(done.stdout + done.stderr)
        summary = json.loads((out / 'summary.json').read_text()) if (out / 'summary.json').exists() else {}
        by_case = (summary.get('strata') or {}).get('by_case', {})
        return {'status': summary.get('status', 'infrastructure_failure'), 'reason': summary.get('reason'),
                'raw_error': None, 'package_score': None, 'calls': len(summary.get('records', [])),
                'cases': [{'case': k, 'n': v['n'], 'family': v['family'], 'complete': v['complete'],
                           'raw_error': v['raw_error'], 'cpu_s': v['cpu_s'], 'peak_bytes': v['peak_bytes'],
                           'statuses': v['statuses'], 'cpu_limit_s': next(c['cpu_limit_s'] for c in cases if c['case'] == k)}
                          for k, v in by_case.items()],
                'max_cpu_s': max((r.get('cpu_s') or 0 for r in summary.get('records', [])), default=None)}


class PolarizationDemosaicking(OperatorTask):
    grade_memory = 10 * GIB
    # Each scene has a 1.000-second wall limit including interpreter start-up, about 0.6 s of which a typical
    # submission uses; with development sessions running on the L3-sharing dev cores, scenes spiked to 0.9-1.0 s
    # (tasks/polarization-demosaicking/analysis/polarization-first-scene.json); those spikes came from evicted image files
    # (warm_image below), and real development adds about 1.5% (docs/platform/CONCURRENCY.md 7, 2026-11-12).
    grade_isolation = 'shared'

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


class PolarizationDemosaickingV3(PolarizationDemosaicking):
    """The author's vNext 1.3.0-dev package: 132 formal scenes (Qiu32 + Wen100), graded by grade_vnext.py; the Wen5
    development scenes (--mode dev) are the public feedback set (FRONTIERAM_POLAR_DATA=wen). Same per-scene contract
    as v2 (1 CPU, 8 GiB, 1.000 s wall); the local reference pair gives the 0- and 50-point anchors."""

    def run_grader(self, source, out, placement):
        import csv
        out.parent.mkdir(parents=True, exist_ok=True)
        self.warm_image(placement, out.parent / f'{out.name}-warmup.json')
        mode = 'dev' if os.environ.get('FRONTIERAM_POLAR_DATA') == 'wen' else 'formal'
        grader = self.root() / 'harbor/polarization-demosaicking/grade_vnext.py'
        env = self.shim_environment(placement, out.parent / f'{out.name}-docker.jsonl')
        done = subprocess.run([str(GRADER_PYTHON), '-B', str(grader), str(source), str(out), '--mode', mode],
                              cwd=grader.parent, env=env, capture_output=True, text=True, timeout=self.grade_timeout)
        (out.parent / f'{out.name}-grader.log').write_text(done.stdout + done.stderr)
        summary = json.loads((out / 'summary.json').read_text()) if (out / 'summary.json').exists() else {}
        scored = summary.get('score') or {}
        rows = list(csv.DictReader((out / 'per_scene.csv').open())) if (out / 'per_scene.csv').exists() else []
        status = scored.get('status', 'infrastructure_failure')
        if status not in ('ok', 'infrastructure_failure'):
            status = 'submission_failure'
        walls = [float(r['wall_s']) for r in rows if r.get('wall_s')]
        raw = scored.get('E') if mode == 'formal' else scored.get('mean_rmse')
        return {'status': status, 'raw_error': raw, 'reason': scored.get('reason') or (summary.get('build') or {}).get('reason'),
                'package_score': scored.get('score'), 'cases_passed': sum(r.get('status') == 'ok' for r in rows),
                'expected_cases': summary.get('scene_count'), 'max_wall_s': max(walls, default=None), 'mode': mode,
                'scene_statuses': {f'{r["dataset"]}/{r["scene"]}': r['status'] for r in rows},
                'scene_rmse': {f'{r["dataset"]}/{r["scene"]}': float(r['mean_rmse']) for r in rows if r.get('mean_rmse')}}


class FluorescenceDeconvolution(PolarizationDemosaicking):
    """The author's ci_3d_fluorescence_deconvolution 1.1.0 (same author and runner design as polarization vNext),
    graded by arena/operator_drivers/fluorescence.py: 96 private cases, 0.500 s per case including start-up; public
    feedback (FRONTIERAM_FLUOR_DATA=public): the 12 public examples. The image is reconstructed for amd64 and warmed
    before the timed cases; the references rerun here are the 0- and 50-point anchors (package 0/5/10, x10)."""

    def run_grader(self, source, out, placement):
        out.parent.mkdir(parents=True, exist_ok=True)
        self.warm_image(placement, out.parent / f'{out.name}-warmup.json')
        public = os.environ.get('FRONTIERAM_FLUOR_DATA') == 'public'
        command = [str(GRADER_PYTHON), '-B', str(ROOT / 'arena/operator_drivers/fluorescence.py'), str(self.root() / 'runtime'),
                   str(self.root() / 'private-data'), str(source), str(out)]
        if public:
            command += ['--public', str(self.root('public') / 'public-dev')]
        env = self.shim_environment(placement, out.parent / f'{out.name}-docker.jsonl')
        done = subprocess.run(command, env=env, capture_output=True, text=True, timeout=self.grade_timeout)
        (out.parent / f'{out.name}-grader.log').write_text(done.stdout + done.stderr)
        summary = json.loads((out / 'summary.json').read_text()) if (out / 'summary.json').exists() else {}
        rows = json.loads((out / 'records.json').read_text()) if (out / 'records.json').exists() else []
        failed = [r for r in rows if r.get('status') != 'ok']
        reason = summary.get('reason') or (f'{len(failed)} cases failed; first {failed[0]["case_id"]}: {failed[0].get("reason")}'
                                           if failed else (summary.get('build') or {}).get('reason'))
        return {'status': summary.get('status', 'infrastructure_failure'), 'raw_error': summary.get('mean_nmse'),
                'reason': reason, 'package_score': None, 'cases': summary.get('case_count'),
                'failures': summary.get('failure_cases'), 'scope': summary.get('scope'),
                'max_wall_s': max((r.get('wall_s') or 0 for r in rows), default=None),
                'stratum_means': summary.get('stratum_means')}


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


class BlackboxCubature(OperatorTask):
    """Interactive cubature oracle (arena/operator_drivers/blackbox_cubature.py). Private: the author's splits B and C
    (144 cases); public feedback (FRONTIERAM_CUBATURE_DATA=public): suite H1 of split A (24 cases). Both are balanced
    over family x dimension; the 30 s case limit keeps public within 15 minutes on one slot (results/OPEN_ITEMS.md)."""
    grade_memory = 4 * GIB
    grade_timeout = 3 * 3600
    SPLITS = {'private': {'B': ('cross_splits/split_B/cases', ('H1', 'H2', 'H3')),
                          'C': ('cross_splits/split_C_20261114/cases', ('H1', 'H2', 'H3'))},
              'public': {'A': ('task/private_hidden/cases', ('H1',))}}

    guard = frozenset({'polynomial_ridge'})

    def data_set(self) -> str:
        return 'public' if os.environ.get('FRONTIERAM_CUBATURE_DATA') == 'public' else 'private'

    def oracle(self) -> Path:
        return self.root() / 'task/private_hidden/grade_hidden.py'

    def cases(self) -> list[dict]:
        cases = []
        for split, (folder, suites) in self.SPLITS[self.data_set()].items():
            for suite in suites:
                for case in json.loads((self.root() / folder / f'{suite}.json').read_text())['cases']:
                    cases.append({**case, 'split': split, 'suite': suite})
        return cases

    def suite(self, runtime=None) -> list[dict]:
        limit = self.spec.get('case_limit_s', 30)
        return [{'id': f'{c["split"]}/{c["case_id"]}', 'group': c['family'], 'time_limit_seconds': limit}
                for c in self.cases()]

    async def grade(self, runtime, placement, output, case_ids=None):
        source = output / 'source'
        if (source / PAIR).exists():
            parts = {which: self.run_grader(source / which, output / which, placement, case_ids) for which in ('weak', 'strong')}
            return {'exit_status': 'finished', 'kind': 'reference_pair', 'parts': parts}
        return {'exit_status': 'finished', 'kind': 'candidate',
                'parts': {'candidate': self.run_grader(source, output / 'candidate', placement, case_ids)}}

    def run_grader(self, source, out, placement, case_ids=None):
        out.parent.mkdir(parents=True, exist_ok=True)
        cases = self.cases()
        if case_ids:
            wanted = set(case_ids)
            cases = [c for c in cases if f'{c["split"]}/{c["case_id"]}' in wanted]
            if len(cases) != len(wanted):
                raise ValueError('case_ids name cases outside the data set')
        listing = out.parent / f'{out.name}-cases.json'
        listing.write_text(json.dumps(cases) + '\n')
        driver = ROOT / 'arena/operator_drivers/blackbox_cubature.py'
        done = subprocess.run([str(GRADER_PYTHON), '-B', str(driver), str(self.oracle()),
                               str(listing), str(source), str(out), '--image', self.lock()['image']['id'],
                               '--cpuset', placement.cpuset, '--cgroup-parent', placement.cgroup_parent,
                               '--case-limit', str(self.spec.get('case_limit_s', 30))],
                              env={**os.environ, 'DOCKER_HOST': placement.docker_host},
                              capture_output=True, text=True, timeout=self.grade_timeout)
        (out.parent / f'{out.name}-grader.log').write_text(done.stdout + done.stderr)
        summary = json.loads((out / 'summary.json').read_text()) if (out / 'summary.json').exists() else {}
        reason = summary.get('reason') or (done.stderr.strip().splitlines()[-1] if done.returncode and done.stderr.strip() else None)
        return {'status': summary.get('status', 'infrastructure_failure'), 'raw_error': None, 'reason': reason,
                'package_score': None, 'cases': summary.get('cases'), 'failures': summary.get('failures'),
                'max_wall_s': summary.get('max_wall_s'), 'rows': summary.get('rows', []), 'data': self.data_set()}

    def baseline_ok(self, metrics):
        from .operator_drivers.blackbox_cubature import gain

        parts = metrics['parts']
        weak, strong = parts.get('weak', {}), parts.get('strong', {})
        if not weak.get('status') == strong.get('status') == 'ok':
            return False, f'reference pair not valid: weak={weak.get("status")} strong={strong.get("status")}'
        g0 = gain(weak['rows'], strong['rows'], self.guard)['gain']
        return g0 < 0, f'Monte Carlo gain over HumanPortfolio-v1 G0 = {g0:.4f} (must be negative)'

    def score(self, runtime, metrics, baseline):
        from .operator_drivers.blackbox_cubature import gain, score_100 as cubature_score

        part = metrics['parts'].get('candidate') or {}
        status = part.get('status')
        if status == 'infrastructure_failure' or status is None:
            return {'status': 'unmeasured', 'score': None, 'reason': part.get('reason')}
        if status != 'ok':
            return {'status': 'submission_failure', 'score': 0.0, 'reason': part.get('reason')}
        if baseline is None:
            return {'status': 'ok', 'score': None}
        anchors = baseline['metrics']['parts']
        ids = {row['id'] for row in part['rows']}
        strong = anchors['strong']['rows']
        candidate = gain(part['rows'], strong, self.guard)
        mc = gain([row for row in anchors['weak']['rows'] if row['id'] in ids], strong, self.guard)
        return {'status': 'ok', 'raw_error': candidate['gain'], 'score': cubature_score(candidate, mc),
                'families': candidate['families'], 'anchors': {'weak': mc['gain'], 'strong': 0.0}}


class BlackboxCubatureV2(BlackboxCubature):
    """v2: polynomial_ridge replaced by kinked_ridge_sum (no guard family); cases and oracle are kept in this repository
    (tasks/blackbox-cubature/versions/v2/{data,grader,generation}). Public: 2 cases of split A per (family, dimension)."""
    guard = frozenset()
    VERSION = ROOT / 'tasks/blackbox-cubature/versions/v2'

    def oracle(self) -> Path:
        return self.VERSION / 'grader/oracle.py'

    def cases(self) -> list[dict]:
        return json.loads((self.VERSION / 'data' / self.data_set() / 'cases.json').read_text())['cases']


class ContinuumEstimation(OperatorTask):
    """The author's continuum-estimation 1.0.1 verifier, ported to x86-64 (arena/operator_drivers/continuum.py).
    Private: the 60 primary cases; public feedback (FRONTIERAM_CONTINUUM_DATA=public): the 8 public examples. The
    weak and strong references rerun here are the 0- and 50-point anchors (the package's 0/5/10 mapping, x10)."""
    grade_memory = 4 * GIB
    grade_timeout = 3 * 3600

    def data_set(self) -> str:
        return 'public' if os.environ.get('FRONTIERAM_CONTINUUM_DATA') == 'public' else 'primary'

    def run_grader(self, source, out, placement):
        out.parent.mkdir(parents=True, exist_ok=True)
        driver = ROOT / 'arena/operator_drivers/continuum.py'
        done = subprocess.run([str(GRADER_PYTHON), '-B', str(driver), str(self.root()), str(self.sealed / 'workspace'),
                               str(source), str(out), '--image', self.lock()['image']['id'], '--cpuset', placement.cpuset,
                               '--cgroup-parent', placement.cgroup_parent, '--partition', self.data_set()],
                              env={**os.environ, 'DOCKER_HOST': placement.docker_host},
                              capture_output=True, text=True, timeout=self.grade_timeout)
        (out.parent / f'{out.name}-grader.log').write_text(done.stdout + done.stderr)
        verdict = json.loads((out / 'verdict.json').read_text()) if (out / 'verdict.json').exists() else {}
        return {'status': verdict.get('status', 'infrastructure_failure'), 'raw_error': verdict.get('raw_loss'),
                'reason': verdict.get('reason') or (done.stderr.strip().splitlines()[-1] if done.returncode and done.stderr.strip() else None),
                'package_score': None, 'cases': verdict.get('cases'), 'failures': verdict.get('failures'),
                'max_wall_s': verdict.get('max_wall_s'), 'data': verdict.get('partition')}


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
           'galaxy-gravity-v4': GalaxyGravityV4,
           'polarization-demosaicking': PolarizationDemosaicking, 'polarization-demosaicking-v2': PolarizationDemosaicking,
           'polarization-demosaicking-v3': PolarizationDemosaickingV3,
           'species-tree': SpeciesTree, 'species-tree-v2': SpeciesTreeV2, 'species-tree-v3': SpeciesTreeV2,
           'trajectory-covariance': TrajectoryCovariance, 'trajectory-covariance-v2': TrajectoryCovariance,
           'blackbox-cubature': BlackboxCubature, 'blackbox-cubature-v2': BlackboxCubatureV2,
           'continuum-estimation': ContinuumEstimation, 'fluorescence-deconvolution': FluorescenceDeconvolution}


def operator_tasks() -> dict[str, OperatorTask]:
    # Repository-native draft tasks do not pretend to have a sealed operator
    # package or a calibrated deployment on this host.
    from .operator_drivers.ihvp_sequence import IHVPSequence
    from .operator_drivers.ihvp_hard_v3 import IHVPHardV3
    from .operator_drivers.ihvp_hard_v4 import IHVPHardV4
    from .operator_drivers.pdhcg_qp import PDHCGQP
    handlers = {**CLASSES, 'ihvp-sequence': IHVPSequence, 'ihvp-sequence-v2': IHVPSequence,
                'ihvp-sequence-v3': IHVPHardV3, 'ihvp-sequence-v4': IHVPHardV4,
                'pdhcg-qp-v1': PDHCGQP}
    tasks = {}
    for path in sorted(SPECS.glob('*/versions/*/task.json')):
        spec = json.loads(path.read_text())
        tasks[spec['name']] = handlers.get(spec['name'], DevelopmentOnlyTask)(spec)
    return tasks
