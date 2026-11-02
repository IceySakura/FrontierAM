"""Private Docker evaluator. Requires a qualified image containing supervisor.py.

No reference/manifest/answer path is mounted. A frozen regular-file submission
is copied before startup. Candidate stdout is the numerical stream; trusted
supervisor stderr supplies cgroup accounting. Docker CLI rusage is NEVER used.
"""
import argparse
from contextlib import contextmanager
import hashlib
import importlib.util
import io
import json
import math
import os
from pathlib import Path
import shlex
import statistics
import subprocess
import tarfile
import tempfile
import threading
import time
import uuid

import numpy as np

TASK = Path(__file__).resolve().parents[2]
spec = importlib.util.spec_from_file_location('frontieram_public_local_runner', TASK / 'public/tools/run_local.py')
local = importlib.util.module_from_spec(spec)
spec.loader.exec_module(local)
wire = local.wire

MEMORY_BYTES = 4 * 1024**3
ARTIFACT_BYTES = 1024**3
SCRATCH_BYTES = 256 * 1024**2


def freeze_submission(directory, tar_path, *, max_bytes=ARTIFACT_BYTES):
    """Copy exact verified bytes into a root-owned, read-only regular-file tar.

    The archive, not a mutable source directory, is copied to every fresh run.
    Symlinks/hardlinks/device files are rejected; source mutation during the
    snapshot is an error. Every path is relative and hash-covered.
    """
    root = Path(directory)
    if root.is_symlink() or not root.is_dir():
        raise ValueError('submission root must be a real directory')
    root = root.resolve()
    hashes, total = {}, 0
    paths = sorted(root.rglob('*'))
    with tarfile.open(tar_path, 'w') as archive:
        for path in paths:
            if path.is_symlink():
                raise ValueError('submission symlinks are forbidden')
            if path.is_dir():
                continue
            before = path.stat()
            if not path.is_file() or before.st_nlink != 1:
                raise ValueError('only non-hardlinked regular submission files allowed')
            relative = path.relative_to(root).as_posix()
            if relative.startswith('/') or '..' in Path(relative).parts:
                raise ValueError('unsafe archive path')
            total += before.st_size
            if total > max_bytes:
                raise ValueError('submission exceeds artifact budget')
            # O_NOFOLLOW prevents a last-moment symlink swap from reading a
            # different host file; matching inode/mtime/size detects mutation.
            fd = os.open(path, os.O_RDONLY | os.O_NOFOLLOW)
            with os.fdopen(fd, 'rb') as stream:
                actual = os.fstat(stream.fileno())
                if (actual.st_ino, actual.st_dev, actual.st_size, actual.st_mtime_ns) != (before.st_ino, before.st_dev, before.st_size, before.st_mtime_ns):
                    raise ValueError('submission changed while freezing')
                digest = hashlib.sha256()
                class Reader:
                    def read(self, count=-1):
                        block = stream.read(count)
                        digest.update(block)
                        return block
                info = tarfile.TarInfo(relative)
                info.size = before.st_size
                info.mode = 0o555 if before.st_mode & 0o111 else 0o444
                info.uid = info.gid = 0
                info.mtime = 0
                archive.addfile(info, Reader())
                after = os.fstat(stream.fileno())
                if (after.st_size, after.st_mtime_ns) != (before.st_size, before.st_mtime_ns):
                    raise ValueError('submission changed while freezing')
                hashes[relative] = digest.hexdigest()
    if not hashes:
        raise ValueError('empty submission')
    content_digest = hashlib.sha256(json.dumps(hashes, sort_keys=True, separators=(',', ':')).encode()).hexdigest()
    return {'sha256': content_digest, 'bytes': total, 'files': hashes}


def create_command(docker, image, name, command, cpu_seconds, *, memory_bytes=MEMORY_BYTES,
                   scratch_bytes=SCRATCH_BYTES, pids=32, stderr_bytes=4194304):
    if not command or any(not isinstance(x, str) or '\x00' in x for x in command):
        raise ValueError('invalid command argv')
    for value, upper, name_ in ((memory_bytes, MEMORY_BYTES, 'memory'), (scratch_bytes, SCRATCH_BYTES, 'scratch'),
                                (pids, 32, 'pids'), (stderr_bytes, 4194304, 'stderr')):
        if type(value) is not int or value <= 0 or value > upper:
            raise ValueError(f'invalid {name_} limit')
    if isinstance(cpu_seconds, bool) or not math.isfinite(cpu_seconds) or not 0 < cpu_seconds <= 240:
        raise ValueError('invalid CPU cap')
    return [docker, 'create', '--name', name, '--interactive', '--network', 'none',
            '--read-only', '--cap-drop', 'ALL', '--cap-add', 'SETUID', '--cap-add', 'SETGID', '--cap-add', 'KILL',
            '--security-opt', 'no-new-privileges:true', '--cgroupns', 'private',
            '--cpus', '1', '--cpuset-cpus', '0', '--memory', str(memory_bytes), '--memory-swap', str(memory_bytes),
            '--pids-limit', str(pids), '--ulimit', 'nofile=128:128', '--ulimit', 'core=0:0',
            '--ipc', 'none', '--tmpfs', f'/tmp:rw,nosuid,nodev,size={scratch_bytes},mode=1777',
            '--log-driver', 'none', '--workdir', '/submission', '--user', '0:0',
            '--entrypoint', 'python3', image, '/opt/frontieram/evaluator/supervisor.py',
            '--cpu-seconds', str(cpu_seconds), '--stderr-bytes', str(stderr_bytes), '--', *command]


def _docker(docker, *args, timeout=30, **kwargs):
    result = subprocess.run([docker, *args], capture_output=True, timeout=timeout, **kwargs)
    if result.returncode:
        raise RuntimeError(f'docker {args[0]} failed: {result.stderr.decode(errors="replace")[:1500]}')
    return result.stdout


def validate_frozen_archive(path):
    """Reject paths, ownership or links that could alter the trusted image."""
    names, total = set(), 0
    with tarfile.open(path, 'r:') as archive:
        for item in archive:
            parts = item.name.split('/')
            if (not item.isfile() or any(p in ('', '.', '..') for p in parts)
                    or item.name in names or item.uid != 0 or item.gid != 0
                    or item.mode not in (0o444, 0o555) or item.size < 0):
                raise ValueError('unsafe frozen submission archive')
            names.add(item.name)
            total += item.size
            if total > ARTIFACT_BYTES:
                raise ValueError('submission exceeds artifact budget')
    if not names:
        raise ValueError('empty frozen submission archive')


@contextmanager
def frozen_submission_image(docker, image, submission_tar):
    """Populate a never-started container, then execute only its frozen image.

    Docker refuses cp into a read-only container. No candidate process ever
    runs while the filesystem is writable, and no host directory is mounted.
    """
    validate_frozen_archive(submission_tar)
    base_id = json.loads(_docker(docker, 'image', 'inspect', image))[0]['Id']
    name = 'frontieram-stage-' + uuid.uuid4().hex
    created, frozen = False, None
    try:
        _docker(docker, 'create', '--name', name, '--network', 'none', '--cap-drop', 'ALL',
                '--security-opt', 'no-new-privileges:true', '--entrypoint', '/bin/false', base_id)
        created = True
        with open(submission_tar, 'rb') as stream:
            _docker(docker, 'cp', '-', name + ':/submission', stdin=stream, timeout=60)
        frozen = _docker(docker, 'commit', name, timeout=60).decode().strip()
        yield frozen, base_id
    finally:
        if created:
            _docker(docker, 'rm', '--force', name)
        if frozen:
            _docker(docker, 'image', 'rm', frozen)


def inspect_configuration(info):
    h = info['HostConfig']
    checks = {
        'network_none': h.get('NetworkMode') == 'none', 'readonly_root': h.get('ReadonlyRootfs') is True,
        'no_binds': not h.get('Binds') and not info.get('Mounts'),
        'memory_swap_equal': h.get('Memory') == h.get('MemorySwap') and 0 < h.get('Memory') <= MEMORY_BYTES,
        'one_cpu': h.get('NanoCpus') == 1_000_000_000 and h.get('CpusetCpus') == '0',
        'pids_bounded': 0 < h.get('PidsLimit', 0) <= 32,
        'drop_all_caps': set(h.get('CapDrop') or []) == {'ALL'},
        'supervisor_only_caps': {x.removeprefix('CAP_') for x in h.get('CapAdd') or []} == {'SETUID', 'SETGID', 'KILL'},
        'no_new_privileges': any(x.split(':')[0] == 'no-new-privileges' for x in h.get('SecurityOpt') or []),
        'private_cgroup': h.get('CgroupnsMode') == 'private', 'ipc_none': h.get('IpcMode') == 'none',
        'no_devices': not h.get('Devices') and not h.get('DeviceRequests') and not h.get('Privileged'),
    }
    if not all(checks.values()):
        raise RuntimeError('evaluator_error:unsafe_container_config:' + ','.join(k for k, v in checks.items() if not v))
    return checks


def run_session(command, case_path, *, image, submission_tar, docker='docker', **limits):
    with frozen_submission_image(docker, image, submission_tar) as (frozen_image, base_id):
        record = _run_session(command, case_path, image=frozen_image, docker=docker, **limits)
        record['submission_image_id'] = record['image_id']
        record['image_id'] = base_id
        return record


def _run_session(command, case_path, *, image, docker='docker', repeat=1,
                startup_timeout=15, wall_timeout=None, cpu_timeout=None,
                memory_bytes=MEMORY_BYTES, scratch_bytes=SCRATCH_BYTES, stderr_bytes=4194304):
    metadata, header, b0, b1, reference = local._load_case(case_path)
    argv = shlex.split(command) if isinstance(command, str) else list(command)
    cap = 60 if metadata['requests'] == 1 else 240
    wall_timeout = cap if wall_timeout is None else wall_timeout
    cpu_timeout = cap if cpu_timeout is None else cpu_timeout
    for val, upper in ((wall_timeout, cap), (cpu_timeout, cap), (startup_timeout, 15)):
        if not isinstance(val, (int, float)) or isinstance(val, bool) or not math.isfinite(val) or not 0 < val <= upper:
            raise ValueError('invalid deadline')
    name = 'frontieram-final-' + uuid.uuid4().hex
    proc = None
    created = False
    state = {'ready_at': None, 'started': None, 'candidate_exit_at': None, 'last_output_at': None,
             'final': None, 'failure': None, 'io_done': False, 'telemetry_done': False, 'raw': []}
    lock = threading.Lock()
    def fail(reason):
        with lock:
            state['failure'] = state['failure'] or reason
    try:
        result = subprocess.run(create_command(docker, image, name, argv, cpu_timeout,
                                               memory_bytes=memory_bytes, scratch_bytes=scratch_bytes,
                                               stderr_bytes=stderr_bytes), capture_output=True, timeout=30)
        if result.returncode:
            raise RuntimeError('evaluator_error:container_create:' + result.stderr.decode(errors='replace')[:1000])
        created = True
        configuration = inspect_configuration(json.loads(_docker(docker, 'inspect', name))[0])
        launched = time.perf_counter()
        proc = subprocess.Popen([docker, 'start', '--attach', '--interactive', name],
                                stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE, bufsize=0)
        def exchange():
            try:
                if wire._read_exact(proc.stdout, len(wire.READY)) != wire.READY:
                    fail('invalid_ready'); return
                state['started'] = time.perf_counter()
                for block in header: wire._write_all(proc.stdin, block)
                for index in range(metadata['requests']):
                    wire._write_all(proc.stdin, np.asarray(b0[index], dtype='<f8').tobytes())
                    wire._write_all(proc.stdin, np.asarray(b1[index], dtype='<f8').tobytes())
                    proc.stdin.flush()
                    state['raw'].append(wire._read_exact(proc.stdout, 8 * metadata['n']))
                    state['last_output_at'] = time.perf_counter()
                proc.stdin.close()
                if proc.stdout.read(1): fail('extra_stdout')
            except wire.ProtocolError:
                fail('short_output' if state['started'] else 'incomplete_ready')
            except (OSError, ValueError): fail('stream_io_failure')
            finally: state['io_done'] = True
        def telemetry():
            try:
                seen = set()
                while True:
                    line = proc.stderr.readline(524289)
                    if not line: break
                    if len(line) > 524288 or not line.endswith(b'\n'):
                        fail('invalid_supervisor_telemetry'); return
                    event = json.loads(line)
                    kind = event.get('event')
                    if kind in seen or kind not in ('supervisor_ready', 'candidate_exit', 'final', 'supervisor_error'):
                        fail('invalid_supervisor_telemetry'); return
                    seen.add(kind)
                    if kind == 'supervisor_ready':
                        if event.get('protocol') != 'FAMEXP2-docker-v1': fail('wrong_supervisor_protocol')
                        state['ready_at'] = time.perf_counter()
                    elif kind == 'candidate_exit':
                        state['candidate_exit_at'] = time.perf_counter()
                    elif kind == 'final': state['final'] = event
                    else: fail('supervisor_error')
            except (ValueError, OSError, TypeError): fail('invalid_supervisor_telemetry')
            finally: state['telemetry_done'] = True
        threads = [threading.Thread(target=f, daemon=True) for f in (exchange, telemetry)]
        for thread in threads: thread.start()
        while True:
            now = time.perf_counter()
            if state['started'] is None:
                base = state['ready_at']
                if base is not None and now - base > startup_timeout: fail('startup_timeout')
                elif base is None and now - launched > 30: fail('container_start_timeout')
            elif now - state['started'] > wall_timeout and not (state['candidate_exit_at'] and state['io_done']):
                fail('wall_timeout')
            if (state['candidate_exit_at'] is not None and state['io_done'] and state['telemetry_done']
                    and now - state['candidate_exit_at'] > 5):
                fail('docker_teardown_timeout')
            if state['failure']:
                subprocess.run([docker, 'kill', name], capture_output=True, timeout=10)
                break
            if proc.poll() is not None and state['io_done'] and state['telemetry_done']: break
            time.sleep(0.01)
        try: proc.wait(timeout=10)
        except subprocess.TimeoutExpired:
            proc.kill(); proc.wait(timeout=5); fail('docker_attach_did_not_exit')
        for thread in threads: thread.join(timeout=1)
        info = json.loads(_docker(docker, 'inspect', name))[0]
        final = state['final']
        status = state['failure']
        if final is None:
            status = 'memory_limit' if info['State'].get('OOMKilled') else status or 'missing_supervisor_measurement'
        else:
            if final.get('status') in ('cpu_limit', 'memory_limit', 'stderr_limit'):
                status = final['status']
            elif final.get('status') != 'ok': status = status or final.get('status', 'invalid_final_status')
            if proc.returncode or info['State'].get('ExitCode'): status = status or 'container_nonzero_exit'
            for name_, positive in (('cpu_seconds', False), ('memory_bytes', True)):
                val = final.get(name_)
                if isinstance(val, bool) or not isinstance(val, (int, float)) or not math.isfinite(val) or val < 0 or (positive and val == 0):
                    status = status or 'invalid_resource_measurement'
            if status is None and final['cpu_seconds'] > cpu_timeout: status = 'cpu_limit'
            if status is None and final['memory_bytes'] > memory_bytes: status = 'memory_limit'
        end = max((v for v in (state['candidate_exit_at'], state['last_output_at']) if v is not None), default=None)
        elapsed = end - state['started'] if end is not None and state['started'] is not None else None
        if status is None and (elapsed is None or elapsed <= 0 or elapsed > wall_timeout): status = 'wall_timeout'
        outputs = [np.frombuffer(raw, dtype='<f8') for raw in state['raw']]
        errors = [local._check_output(output, reference[i], metadata['atol'], metadata['rtol']) for i, output in enumerate(outputs)]
        if status is None and (len(errors) != metadata['requests'] or not all(e['passed'] for e in errors)): status = 'accuracy_failure'
        status = status or 'ok'
        return {'valid': status == 'ok', 'status': status, 'reason': status, 'repeat': repeat,
                'scenario': metadata['scenario'], 'request_count': metadata['requests'],
                'time_seconds': elapsed, 'cpu_seconds': final.get('cpu_seconds') if final else None,
                'memory_bytes': final.get('memory_bytes') if final else None,
                'remote_session_seconds': final.get('remote_session_seconds') if final else None,
                'startup_seconds': state['started'] - state['ready_at'] if state['started'] and state['ready_at'] else None,
                'stderr_bytes': final.get('stderr_bytes') if final else None,
                'stderr': final.get('stderr') if final else None, 'errors': errors,
                'configuration_checks': configuration,
                'image_id': info['Image'], 'container_exit_code': info['State'].get('ExitCode'),
                'container_oom_killed': info['State'].get('OOMKilled'),
                'cgroup_oom_kill': final.get('oom_kill') if final else None,
                'measurement_scope': 'whole-container cgroup v2 CPU+memory.peak; host stream wall including VM transport, excluding docker teardown'}
    finally:
        if proc is not None:
            if proc.poll() is None:
                subprocess.run([docker, 'kill', name], capture_output=True, timeout=10)
                proc.kill()
                proc.wait(timeout=5)
            for stream in (proc.stdin, proc.stdout, proc.stderr):
                if stream and not stream.closed: stream.close()
        if created:
            subprocess.run([docker, 'rm', '--force', name], capture_output=True, timeout=15)


def aggregate(records):
    valid = bool(records) and all(x['valid'] for x in records)
    values = {k: [x[k] for x in records if x.get(k) is not None] for k in ('time_seconds', 'memory_bytes', 'cpu_seconds')}
    return {'valid': valid, 'reason': 'ok' if valid else ';'.join(sorted({x['reason'] for x in records if not x['valid']})),
            'repeat_count': len(records), 'all_repeats_valid': valid,
            'time_seconds': statistics.median(values['time_seconds']) if values['time_seconds'] else None,
            'memory_bytes': max(values['memory_bytes'], default=None),
            'max_wall_seconds': max(values['time_seconds'], default=None),
            'max_cpu_seconds': max(values['cpu_seconds'], default=None),
            'request_count': records[0]['request_count'] if records else None}


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--image', required=True)
    p.add_argument('--docker', default='docker')
    p.add_argument('--submission', type=Path, required=True)
    p.add_argument('--command', required=True)
    p.add_argument('--manifest', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--repeats', type=int, default=3)
    args = p.parse_args(argv)
    if not 1 <= args.repeats <= 3: p.error('repeats must be 1..3; official aggregates require 3')
    manifest = json.loads(args.manifest.read_text())
    cases = manifest['cases']
    if len({c['id'] for c in cases}) != len(cases): p.error('duplicate case IDs')
    report = {'scope': 'Docker offline evaluation; no score without independently calibrated reference',
              'score': None, 'runs': [], 'aggregates': {}}
    with tempfile.TemporaryDirectory(prefix='frontieram-frozen-') as tmp:
        frozen = Path(tmp) / 'submission.tar'
        report['submission'] = freeze_submission(args.submission, frozen)
        for case in cases:
            file = args.manifest.parent / case.get('file', case.get('path'))
            if case.get('sha256') and hashlib.sha256(file.read_bytes()).hexdigest() != case['sha256']:
                raise ValueError('evaluator_error:case_hash_mismatch')
            rows = []
            for repeat in range(1, args.repeats + 1):
                record = run_session(args.command, file, image=args.image, submission_tar=frozen, docker=args.docker, repeat=repeat)
                record['id'] = case['id']; rows.append(record); report['runs'].append(record)
            report['aggregates'][case['id']] = aggregate(rows)
            args.output.parent.mkdir(parents=True, exist_ok=True)
            args.output.write_text(json.dumps(report, indent=2, allow_nan=False) + '\n')
    print(f'Wrote {len(report["runs"])} isolated runs to {args.output}')


if __name__ == '__main__':
    main()
