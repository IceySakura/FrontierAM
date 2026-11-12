"""ihvp-sequence: the author's evaluator (versions/v1/grader/evaluate.py) run inside the sealed task image on one grade core.

Isolation: no network; the evaluator runs as root next to the root-only teacher data (data directory mode 0700) and
starts the submission's worker as UID 1000 (IHVP_WORKER_UID, runtime/api.py), so the submission reaches only the
current right-hand side and HVP replies over its pipes. The worker's 256 MiB peak-RSS limit is the evaluator's own
check; the container memory limit is a backstop for the evaluator and teacher data.

Data: FRONTIERAM_IHVP_DATA = private (default) | public (v6.3 feedback) | samples (the two full-size samples, for
infrastructure checks) | public_all (v3: all 30 feedback-pool sequences, for reference and replay reports). Sealed workspace material and data roots come from scripts/seal/seal_ihvp_sequence.py.

Explicit local diagnostics without a slot: python arena/operator_drivers/ihvp_sequence.py --smoke --out DIR
"""
from __future__ import annotations
import json
import os
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[2]
TASK = ROOT / 'tasks' / 'ihvp-sequence'
VERSION = TASK / 'versions' / 'v1'
WORKER_UID = 1000
CONTAINER_MEMORY = 4 * 2**30
BUILD_SECONDS = 600

if __name__ != '__main__':
    from ..operator_tasks import OperatorTask
    from ..tasks import GIB

    class IHVPSequence(OperatorTask):
        grader_version = 'v1'  # versions/<v>/grader evaluates this task version (v1 and v2 share it)
        grade_memory = 5 * GIB
        grade_timeout = 3600
        grade_isolation = 'quiet'  # wall-clock limits (30 s per sequence), as polarization-demosaicking

        def data_dir(self) -> Path | None:
            split = os.environ.get('FRONTIERAM_IHVP_DATA', 'private')
            if split not in ('private', 'public', 'samples', 'public_all'):
                raise ValueError(f'FRONTIERAM_IHVP_DATA={split}')
            if not (self.sealed / 'lock.json').exists():
                return None
            return ROOT / self.lock()['data'][split]

        def baseline_ok(self, metrics):
            if self.status == 'draft':
                return False, 'draft task: anchors are measured but not activated (task.json status)'
            return super().baseline_ok(metrics)

        def score(self, runtime, metrics, baseline):
            if self.status == 'draft':
                return {'status': 'unmeasured', 'score': None,
                        'raw_error': (metrics.get('parts', {}).get('candidate') or {}).get('raw_error'),
                        'reason': 'draft task: no formal score before activation (task.json status)'}
            return super().score(runtime, metrics, baseline)

        def dev_service(self, output: Path):
            """ihvp-sequence-v2: the samples' HVPs come from a host-side service; v1 ships them in the workspace."""
            cases = self.lock().get('sample_service') if (self.sealed / 'lock.json').exists() else None
            if not cases:
                return [], None
            from ..operator_tasks import GRADER_PYTHON
            directory = Path('/run/frontieram/ihvp-samples') / output.name
            directory.mkdir(parents=True)
            directory.chmod(0o755)
            paths = json.dumps({name: str(ROOT / path) for name, path in cases.items()})
            grader = TASK / 'versions' / self.grader_version / 'grader'
            process = subprocess.Popen([str(GRADER_PYTHON), '-B', '-m', 'arena.operator_drivers.ihvp_oracle', str(directory), paths,
                                        str(grader)],
                                       cwd=ROOT, env={**os.environ, 'PYTHONPATH': str(ROOT)}, stdout=subprocess.PIPE,
                                       stderr=(output / 'sample-service.log').open('wb'), text=True)
            if process.stdout.readline().strip() != 'ready':
                process.kill()
                raise RuntimeError(f'ihvp sample service did not start ({output / "sample-service.log"})')

            def stop():
                process.terminate()
                try:
                    process.wait(timeout=10)
                except subprocess.TimeoutExpired:
                    process.kill()
            return ['--mount', f'type=bind,src={directory},dst=/run/ihvp-samples'], stop

        def run_grader(self, source, out, placement):
            data = self.data_dir()
            if data is None or not (data / 'manifest.json').is_file():
                return {'status': 'infrastructure_failure', 'raw_error': None,
                        'reason': 'not sealed (scripts/seal/seal_ihvp_sequence.py)' if data is None else f'no data manifest in {data}'}
            if data.stat().st_mode & 0o077:
                raise RuntimeError(f'{data} must be mode 0700 (teacher data is root-only inside the container)')
            out.mkdir(parents=True, exist_ok=True)
            grader = f'/opt/repo/tasks/ihvp-sequence/versions/{self.grader_version}/grader'
            evaluate = f'{grader}/evaluate.py'
            # The task directory is mounted under /root (mode 0700). The evaluator runs from a root-only copy of it
            # (generators, references, analysis) in which only the worker's protocol files are readable by its UID.
            readable = ' '.join(['/opt/repo', '/opt/repo/tasks', '/opt/repo/tasks/ihvp-sequence', '/opt/repo/tasks/ihvp-sequence/versions',
                                 f'/opt/repo/tasks/ihvp-sequence/versions/{self.grader_version}', grader])
            # An optional ./build.sh runs once, offline, as the worker UID and outside the sequence clock.
            script = ('mkdir -p /opt/repo/tasks && cp -r /root/task-src /opt/repo/tasks/ihvp-sequence && chmod -R go-rwx /opt/repo && '
                      f'chmod go+x {readable} && chmod go+rx {grader}/runtime && chmod go+r {grader}/runtime/worker.py {grader}/runtime/api.py && '
                      'cp -r /submission /tmp/candidate && chown -R 1000:1000 /tmp/candidate && cd /tmp/candidate && '
                      f'if [ -f build.sh ]; then timeout {BUILD_SECONDS} setpriv --reuid 1000 --regid 1000 --clear-groups '
                      'sh build.sh > /out/build.log 2>&1 || { touch /out/build_failed; exit 0; }; fi && '
                      f'exec python3 -B {evaluate} --submission /tmp/candidate --manifest /data/manifest.json --out /out/run')
            command = ['docker', '-H', placement.docker_host, 'run', '--rm', '--network', 'none',
                       '--cpus', '1', '--cpuset-cpus', placement.cpuset, '--cgroup-parent', placement.cgroup_parent,
                       '--memory', str(CONTAINER_MEMORY), '--memory-swap', str(CONTAINER_MEMORY), '--pids-limit', '256',
                       '--cap-drop', 'ALL', '--cap-add', 'SETUID', '--cap-add', 'SETGID', '--cap-add', 'CHOWN',
                       '--security-opt', 'no-new-privileges:true', '--user', '0:0', '--log-driver', 'none',
                       '--tmpfs', '/tmp:rw,exec,nosuid,nodev,size=1g,mode=1777',
                       '-e', f'IHVP_WORKER_UID={WORKER_UID}',
                       '-v', f'{TASK}:/root/task-src:ro', '-v', f'{Path(source).resolve()}:/submission:ro',
                       '-v', f'{data}:/data:ro', '-v', f'{out.resolve()}:/out',
                       '--entrypoint', 'sh', self.lock()['image']['id'], '-c', script]
            done = subprocess.run(command, capture_output=True, text=True, timeout=self.grade_timeout)
            (out.parent / f'{out.name}-grader.log').write_text(done.stdout + done.stderr)
            if (out / 'build_failed').exists():
                return {'status': 'submission_failure', 'raw_error': None,
                        'reason': f'build.sh failed or exceeded {BUILD_SECONDS} s (build.log)'}
            result_path = out / 'run' / 'result.json'
            if not result_path.exists():
                return {'status': 'infrastructure_failure', 'raw_error': None,
                        'reason': f'evaluator exited {done.returncode} without result.json'}
            result = json.loads(result_path.read_text())['result']
            families = {c['id']: c.get('family', 'all') for c in json.loads((data / 'manifest.json').read_text())['cases']}
            records = []
            for path in sorted((out / 'run').glob('*__candidate__r0.json')):
                row = json.loads(path.read_text())
                records.append({'case': row['case'], 'family': families.get(row['case'], 'all'), 'status': row['status'],
                                'reason': row.get('reason'), 'energy_loss': row.get('raw_error'),
                                'wall_s': row.get('wall_seconds'), 'peak_rss_mib': row.get('peak_rss_mib')})
            (out / 'summary.json').write_text(json.dumps({**result, 'records': records}, indent=1) + '\n')
            return {'status': result.get('status', 'infrastructure_failure'), 'raw_error': result.get('raw_error'),
                    'reason': result.get('reason'), 'package_score': None, 'sequences': result.get('sequences'),
                    'max_wall_s': result.get('max_wall_seconds'), 'mean_wall_s': result.get('mean_wall_seconds'),
                    'max_peak_rss_mib': result.get('max_peak_rss_mib')}


def main():
    # Explicit local diagnostics only; no hidden data is copied into a worker.
    grader = VERSION / 'grader' / 'evaluate.py'
    os.execv(sys.executable, [sys.executable, str(grader), *sys.argv[1:]])


if __name__ == '__main__':
    main()
