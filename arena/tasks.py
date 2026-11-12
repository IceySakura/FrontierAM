"""Task plugins: what the orchestration layer needs to develop, grade, baseline and score one task.

A plugin never decides placement or concurrency. It receives a Placement (one leased CPU, its isolated slice and the
Docker daemon for that flow) and must pass it to every container it creates. Grading receives only a frozen,
re-validated source tree; development evidence is never visible to it.

To add a task, implement Task and register it in TASKS. Its grade() must produce per-case raw measurements, and
score() must compare a candidate's measurements with a baseline measured by the same grade() on this machine.
"""

from __future__ import annotations

import io
import json
import shlex
import tarfile

from dataclasses import asdict, dataclass
from pathlib import Path
from types import SimpleNamespace
from typing import Any


GIB = 2**30


@dataclass(frozen=True)
class Placement:
    cpuset: str
    cgroup_parent: str | None
    docker_host: str
    slot: str


def write(path: Path, data) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, ensure_ascii=False) + '\n')


DEV_POLICY = (
    ' Use only the offline workspace tool. Put complete source, executable run.sh and README.md in '
    '/workspace/submission. One attempt, at most 7200 wall-clock seconds including requests, waiting and tools; '
    'a final turn or deadline freezes this attempt. No hidden feedback or candidate repair after freezing. '
    'Common inference policy v4: at most 800 actual HTTP attempts including retries, 64000 output tokens '
    'per response including reasoning, 131072 context setting, provider-default reasoning. Codex compacts '
    'the conversation automatically near 110000 tokens; compaction requests use the same time and HTTP budget. '
    'The old 429-only wording is superseded: at most three identical-payload retries for retryable '
    '429/500/502/503/504 responses; Retry-After first, otherwise full jitter caps of 1/2/4 seconds. '
    'All retries consume the original deadline. No failover or truncated tool execution. '
)


# Models served outside AIHub: their native Responses route, credential file and fixed headers (protocol v5 only).
MODEL_UPSTREAMS = {
    'hy4': {
        'base_url': 'http://api.taiji.woa.com/openapi/v2',
        'key_file': '/root/.frontieram/hy4-taiji.token',
        'upstream_headers': {'Sensitive-Business': 'true'},
    },
}


class Task:
    name: str
    nexus_import: str
    nexus_task: str
    provider: str
    instruction: str
    dev_memory: int = 4 * GIB + GIB // 2
    grade_memory: int
    grade_timeout: float = 28800
    # How much of the machine must stay idle while this task is graded. Full concurrency by default (2026-11-12): with
    # real development and concurrent grading the measured load stays within a few percent (docs/platform/CONCURRENCY.md
    # section 7); the per-job load probe raises an alert instead (arena/load_probe.py). 'grade_pool' and 'quiet' remain
    # for explicit, case-by-case regrades.
    grade_isolation: str = 'shared'
    # Load factor (paired reference time / baseline time) above which the result is flagged with an alert.
    load_gate: float | None = None
    # UID whose host-wide task count the grader's RLIMIT_NPROC budget depends on; grading waits for headroom.
    grader_uid: int | None = None
    grader_uid_headroom: int = 0
    # Whether grading measures the active baseline's frozen reference beside each submission.
    paired: bool = False

    def provider_config(self, runtime: dict, placement: Placement) -> dict:
        return {
            'source_root': runtime['source_root'],
            'docker_host': placement.docker_host,
            'docker_path': runtime['docker_path'],
            'cpuset': placement.cpuset,
            'cgroup_parent': placement.cgroup_parent,
        }

    def dev_config(
        self,
        runtime: dict,
        output: Path,
        placement: Placement,
        *,
        model: str,
        base_url: str,
        key_file: str | None,
        smoke: bool,
        protocol: str = 'v4',
        development_seconds: float | None = None,
    ) -> dict:
        """One Nexus run that develops and freezes only; grading is a separate job."""
        seconds = development_seconds or (60 if smoke else 7200)
        upstream = MODEL_UPSTREAMS.get(model) if not smoke else None
        if upstream is not None:
            if protocol not in ('v5', 'v5.1'):
                raise ValueError(f'{model} is served only through its native Responses route (protocol v5)')
            base_url, key_file = upstream['base_url'], upstream['key_file']
        return {
            'imports': [self.nexus_import],
            'task_name': self.nexus_task,
            'task_config': {
                'output_dir': str(output / 'task'),
                'user_prompt': self.instruction + DEV_POLICY + runtime['architecture_override_prompt'],
                'evaluate': False,
                'verification_only': smoke,
                'grading_timeout': self.grade_timeout,
            },
            'runtime_provider_name': self.provider,
            'runtime_provider_config': self.provider_config(runtime, placement),
            'agent_name': 'frontieram_codex_offline',
            'agent_config': {
                'model': model,
                'base_url': base_url,
                'key_file': key_file,
                'allow_live_endpoint': not smoke,
                'development_seconds': seconds,
                'max_http_attempts': 8 if smoke else 800,
                'max_output_tokens': 64000,
                'auto_compact_token_limit': 110000,
                'protocol': protocol,
                'upstream_headers': (upstream or {}).get('upstream_headers', {}),
            },
            'processors': {'token_counter': {'name': 'token_counter'}},
            'runner': {'cache': {'enabled': False}, 'timeout': 36600},
        }

    def assets(self, runtime: dict):
        raise NotImplementedError

    def docker(self, runtime: dict, placement: Placement):
        return self.assets(runtime).controller.Docker(
            runtime['docker_path'], placement.docker_host, placement.cpuset, placement.cgroup_parent
        )

    def stage(self, runtime: dict, archive: Path, output: Path) -> dict:
        """Re-validate an untrusted tar with the reviewed extractor into output/source and output/source.tar."""
        manifest = self.assets(runtime).controller.validate_and_extract(
            archive, output / 'source', output / 'source.tar'
        )
        write(output / 'source_manifest.json', manifest)
        return manifest

    def reference_archive(self, runtime: dict, output: Path) -> Path:
        raise NotImplementedError

    async def grade(self, runtime: dict, placement: Placement, output: Path) -> dict:
        raise NotImplementedError

    def metrics(self, output: Path, raw: dict) -> dict:
        raise NotImplementedError

    def score(self, runtime: dict, metrics: dict, baseline: dict | None) -> dict:
        raise NotImplementedError

    def baseline_ok(self, metrics: dict) -> tuple[bool, str]:
        raise NotImplementedError


def tar_directory(source: Path, archive: Path, extra: dict[str, tuple[bytes, int]] | None = None) -> Path:
    with tarfile.open(archive, 'w') as tar:
        tar.add(source, arcname='.')
        for name, (payload, mode) in (extra or {}).items():
            member = tarfile.TarInfo(f'./{name}')
            member.size, member.mode = len(payload), mode
            tar.addfile(member, io.BytesIO(payload))
    return archive


class Exponential(Task):
    name = 'exponential'
    nexus_import = 'nexus.extensions.tasks.frontieram_offline'
    nexus_task = 'frontieram_exponential_offline'
    provider = 'frontieram_offline'
    instruction = 'Read /workspace/task/PILOT.md, INSTRUCTION.md and API.md in full. Implement the exponential propagation task.'
    grade_memory = 4 * GIB + GIB // 2
    grade_timeout = 8 * 3600
    # Scored against the reference measured beside each candidate, so graders may share the machine. Pairing cancels
    # load only as far as both respond alike: at load factor 1.095 a real candidate drifted 1.1 points, so any grade
    # measured above 1.04 is redone with the machine quiet (docs/platform/CONCURRENCY.md).
    grade_isolation = 'shared'
    load_gate = 1.04
    # Reference measured for new baselines. The KIOPS/Octave pilot anchor (private_pilot/reference_submission) is
    # dominated by interpreter overhead at pilot sizes; Expokit is the published author code, compiled.
    reference = 'private_pilot/reference_expokit'
    paired = True

    def assets(self, runtime):
        from nexus.extensions.tasks.frontieram_offline.assets import LockedAssets

        return LockedAssets(runtime['source_root'])

    async def grade(self, runtime, placement, output, paired_archive=None):
        """paired_archive is the frozen source.tar of the active baseline, so pairing and anchoring share one reference."""
        from nexus.extensions.tasks.frontieram_offline.grading import grade

        if paired_archive is None:
            raise ValueError('exponential grading needs the frozen reference archive of the active baseline')
        assets = self.assets(runtime)
        paired = output / 'paired_reference'
        assets.controller.validate_and_extract(Path(paired_archive), paired, output / 'paired_reference.tar')
        return await grade(
            SimpleNamespace(assets=assets, docker=self.docker(runtime, placement)),
            output,
            timeout=self.grade_timeout,
            paired_reference=paired,
        )

    def reference_archive(self, runtime, output):
        output.mkdir(parents=True, exist_ok=True)
        source = Path(runtime['source_root']) / self.reference
        if (source / 'run.sh').exists():
            return tar_directory(source, output / 'incoming.tar')
        # The KIOPS reference has no run.sh; the evaluator only starts ./run.sh.
        command = json.loads((source / 'PROVENANCE.json').read_text())['command']
        entry = f'#!/bin/sh\nexec {shlex.join(command)}\n'.encode()
        return tar_directory(source, output / 'incoming.tar', {'run.sh': (entry, 0o755)})

    def metrics(self, output, raw):
        report_path = output / 'evaluation.json'
        report = json.loads(report_path.read_text()) if report_path.exists() else None
        return {
            'raw': raw,
            'valid_runs': raw.get('valid_runs'),
            'total_runs': raw.get('total_runs'),
            'aggregates': report['aggregates'] if report else None,
            'paired_aggregates': report.get('paired_aggregates') if report else None,
            'report': str(report_path) if report else None,
        }

    def baseline_ok(self, metrics):
        ok = metrics['valid_runs'] == metrics['total_runs'] == 69
        return ok, 'all 69 reference runs verified' if ok else 'reference did not pass all 69 runs'

    def score(self, runtime, metrics, baseline):
        """Primary score: candidate against the reference measured beside it; anchored score is diagnostic."""
        raw = metrics['raw']
        if raw.get('exit_status') != 'finished':
            return {'status': 'unmeasured', 'score': None, 'reason': raw.get('reason')}
        if metrics['aggregates'] is None:
            # Build or archive failure: the frozen protocol scores an invalid submission as zero.
            return {'status': 'invalid_submission', 'score': 0.0, 'reason': raw.get('reason')}
        assets = self.assets(runtime)
        cases = json.loads(assets.manifest.read_text())['cases']
        result: dict[str, Any] = {}
        if metrics.get('paired_aggregates'):
            paired = assets.scoring.score_submission(cases, metrics['aggregates'], metrics['paired_aggregates'])
            result.update(status=paired.get('status'), score=paired['score'], scoring='paired', detail=paired)
        if baseline is not None:
            anchored = assets.scoring.score_submission(cases, metrics['aggregates'], baseline['metrics']['aggregates'])
            result.update(anchored_score=anchored['score'])
            if metrics.get('paired_aggregates'):
                reference = baseline['metrics']['aggregates']
                ratios = [metrics['paired_aggregates'][k]['time_seconds'] / reference[k]['time_seconds']
                          for k in reference]
                result.update(load_factor=sorted(ratios)[len(ratios) // 2])
            else:
                result.update(status=anchored.get('status'), score=anchored['score'], scoring='anchored',
                              detail=anchored)
        if 'score' not in result:
            return {'status': 'baseline_required', 'score': None}
        return result


class Conic(Task):
    name = 'conic'
    nexus_import = 'nexus.extensions.tasks.frontieram_conic'
    nexus_task = 'frontieram_conic_success_offline'
    provider = 'frontieram_conic_offline'
    instruction = (
        'Read /workspace/PILOT.md and /workspace/INSTRUCTION.md in full, then inspect /workspace/resources.yaml '
        'and public-dev. Implement the updated conic solver.'
    )
    grade_memory = 8 * GIB + GIB // 2
    # Success is judged against fixed per-case deadlines, so pairing cannot correct it, and four pilot cases finish
    # within 1% of their deadline. Under 4 graders plus full development load the reference lost 3 of 10 successes;
    # with no other grader and 9 real agents developing, every case stayed within 1% (docs/platform/CONCURRENCY.md).
    # Each result records the slots that were leased while it ran; full concurrency since 2026-11-12 (base class note).
    grade_isolation = 'shared'
    # The verifier sets RLIMIT_NPROC=128 for candidate UID 1000, counted across the host; keep 96 of it free.
    grader_uid = 1000
    grader_uid_headroom = 96

    def provider_config(self, runtime, placement):
        return {
            **super().provider_config(runtime, placement),
            'profile_path': runtime['profile_path'],
            'profile_sha256': runtime['profile_sha256'],
        }

    def assets(self, runtime):
        from nexus.extensions.tasks.frontieram_conic.assets import ConicAssets

        return ConicAssets(runtime['source_root'], runtime['profile_path'], runtime['profile_sha256'])

    def reference_archive(self, runtime, output):
        return tar_directory(Path(runtime['conic_reference_root']), output / 'incoming.tar')

    def suite(self, runtime) -> list[dict]:
        """The 650 held-out instances with their frozen per-case limits."""
        manifest = Path(self.assets(runtime).task_root) / 'tests/private/data/manifest.json'
        return json.loads(manifest.read_text())['instances']

    async def grade(self, runtime, placement, output, case_ids=None):
        from nexus.extensions.tasks.frontieram_conic.grading import grade

        assets = self.assets(runtime)
        extra = {'timeout': self.grade_timeout}
        if case_ids:
            limits = {c['id']: c['time_limit_seconds'] for c in self.suite(runtime)}
            # Three repeats at every case's limit, the 1800-second build budget and per-case process overhead.
            extra = {'timeout': 3 * sum(limits[i] for i in case_ids) + 1800 + 15 * len(case_ids) + 900,
                     'case_ids': case_ids}
        return await grade(SimpleNamespace(assets=assets, docker=self.docker(runtime, placement)), output, **extra)

    def metrics(self, output, raw):
        summary = raw.get('metrics') or {}
        cases = summary.get('cases') or []
        return {
            'raw': {k: v for k, v in raw.items() if k != 'metrics'},
            'valid_runs': raw.get('valid_runs'),
            'total_runs': raw.get('total_runs'),
            'success_rate': summary.get('p'),
            'groups': summary.get('groups'),
            'cases': [
                {
                    'case_id': c.get('case_id'),
                    'group': c.get('group'),
                    'success': c.get('success'),
                    'status': c.get('status'),
                    'elapsed_seconds': c.get('elapsed_seconds'),
                    'time_limit_seconds': c.get('time_limit_seconds'),
                }
                for c in cases
            ],
            'summary': summary,
        }

    def baseline_ok(self, metrics):
        ok = metrics['success_rate'] is not None and metrics['total_runs'] == 60
        return ok, 'reference completed 20 cases x 3 repeats' if ok else 'reference assessment incomplete'

    def score(self, runtime, metrics, baseline):
        raw = metrics['raw']
        if raw.get('exit_status') != 'finished':
            return {'status': 'unmeasured', 'score': None, 'reason': raw.get('reason')}
        if raw.get('score') == 0.0 and metrics['success_rate'] is None:
            return {'status': 'invalid_submission', 'score': 0.0, 'reason': raw.get('reason')}
        p = metrics['success_rate']
        result = {'status': 'pilot_success_rate', 'score': p}
        if baseline is not None:
            reference = baseline['metrics']['success_rate']
            by_case = {c['case_id']: c['success'] for c in baseline['metrics']['cases']}
            result.update(
                baseline_success_rate=reference,
                delta_vs_baseline=None if p is None else p - reference,
                solved_where_baseline_failed=sorted(
                    c['case_id'] for c in metrics['cases'] if c['success'] and not by_case.get(c['case_id'])
                ),
                failed_where_baseline_solved=sorted(
                    c['case_id'] for c in metrics['cases'] if not c['success'] and by_case.get(c['case_id'])
                ),
            )
        return result


TASKS: dict[str, Task] = {task.name: task for task in (Exponential(), Conic())}

from .operator_tasks import operator_tasks  # noqa: E402  (operator tasks subclass Task)

TASKS.update(operator_tasks())

from .exponential_v2 import ExponentialV2, ExponentialV3  # noqa: E402

TASKS[ExponentialV2.name] = ExponentialV2()
TASKS[ExponentialV3.name] = ExponentialV3()


def placement_record(placement: Placement) -> dict[str, Any]:
    return asdict(placement)
