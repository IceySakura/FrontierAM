"""V2 one-attempt orchestration with explicit harness termination attribution.

Original campaign_runner.py and previous campaign records remain immutable.
Never execute candidate code on the host.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import importlib.util
import io
import json
import math
import os
from pathlib import Path, PurePosixPath
import re
import shlex
import signal
import subprocess
import sys
import tarfile
import threading
import time
import uuid

ROOT = Path(__file__).resolve().parent
MEMORY = 4 * 1024**3
ARTIFACT = 1024**3
SCRATCH = 256 * 1024**2
RAW_TAR_CAP = ARTIFACT + 32 * 1024**2
DEFAULT_DOCKER = ROOT / 'runtime/bin/docker'
DEFAULT_HOST = 'unix:///Users/iceysakura/.lima/frontieram/sock/docker.sock'
ALIASES = ('opus5.5', 'dpsk pro', 'hy4 preview', 'minimax m3.1 flash', 'step5')
MODEL_IDS = dict(zip(ALIASES, ('claude-opus-5-5', 'deepseek-v4-pro', 'hy4-preview',
                              'minimax-m3.1-flash', 'step-5-preview')))
DEADLINE_CLOCK_TOLERANCE_SECONDS = 0.25
DEFAULT_PACKAGE = ROOT / 'public_pilot_v2'
HARNESS_ENTRY = ROOT / 'harness_recovery/run_codex.py'
# This trusted sidecar sees only the paused container PID namespace. It must
# check link counts itself: tar has no st_nlink field for links outside the tree.
EXPORT_SCRIPT = '''import io,os,stat,sys,tarfile
from pathlib import Path
root=Path('/proc/1/root/workspace/submission')
try:
    if not stat.S_ISDIR(root.lstat().st_mode): raise ValueError('submission root is not a real directory')
    paths=sorted(root.rglob('*'))
    if len(paths)>20000: raise ValueError('too many paths')
    total=0
    for path in paths:
        info=path.lstat()
        if stat.S_ISDIR(info.st_mode): continue
        if not stat.S_ISREG(info.st_mode) or info.st_nlink!=1: raise ValueError('links or special files forbidden')
        total+=info.st_size
        if total>1073741824: raise ValueError('artifact exceeds 1 GiB')
    with tarfile.open(fileobj=sys.stdout.buffer,mode='w|',format=tarfile.PAX_FORMAT) as out:
        for path in paths:
            out.add(path,arcname=path.relative_to(root).as_posix(),recursive=False)
except (ValueError,OSError) as exc:
    sys.stderr.write('invalid_submission:'+str(exc)+'\\n')
    sys.exit(42)
'''


class InfrastructureFailure(RuntimeError):
    pass


class SubmissionFailure(ValueError):
    pass


def now():
    return datetime.now(timezone.utc).isoformat()


def sha(path):
    result = hashlib.sha256()
    with Path(path).open('rb') as source:
        for block in iter(lambda: source.read(1024 * 1024), b''):
            result.update(block)
    return result.hexdigest()


def save(path, data):
    path = Path(path)
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(json.dumps(data, indent=2, allow_nan=False) + '\n')
    temporary.replace(path)


def placement_options(cpuset='0', cgroup_parent=None):
    """One exclusive CPU from the host slot broker; '0' is the original single-slot layout."""
    if not re.fullmatch(r'[0-9]{1,4}', str(cpuset)):
        raise ValueError('a slot must pin exactly one CPU')
    options = ['--cpus', '1', '--cpuset-cpus', str(cpuset)]
    if cgroup_parent is not None:
        if not re.fullmatch(r'[a-z][a-z0-9-]{0,62}\.slice', cgroup_parent):
            raise ValueError('cgroup parent must be a systemd slice name')
        options += ['--cgroup-parent', cgroup_parent]
    return options


def development_command(prefix, image, name, cpuset='0', cgroup_parent=None):
    if not re.fullmatch(r'sha256:[0-9a-f]{64}', image):
        raise ValueError('use immutable image ID')
    return list(prefix) + ['create', '--name', name, '--network', 'none', '--read-only',
            '--cap-drop', 'ALL', '--security-opt', 'no-new-privileges:true', '--cgroupns', 'private',
            *placement_options(cpuset, cgroup_parent), '--memory', str(MEMORY), '--memory-swap', str(MEMORY),
            '--pids-limit', '32', '--ulimit', 'nofile=128:128', '--ulimit', 'core=0:0', '--ipc', 'none',
            '--tmpfs', f'/tmp:rw,nosuid,nodev,size={SCRATCH},mode=1777',
            '--tmpfs', f'/workspace:rw,exec,nosuid,nodev,size={ARTIFACT},uid=1000,gid=1000,mode=0755',
            '--log-driver', 'none', '--user', '1000:1000', '--workdir', '/workspace',
            '--entrypoint', '/bin/sleep', image, 'infinity']


def verify_configuration(info, image, cpuset='0', cgroup_parent=None):
    h, config = info['HostConfig'], info['Config']
    caps = lambda value: {item.removeprefix('CAP_') for item in value or []}
    ulimits = {row['Name']: (row['Soft'], row['Hard']) for row in h.get('Ulimits') or []}
    tmpfs = h.get('Tmpfs') or {}
    checks = {
        'immutable_image': info['Image'] == image,
        'network_none': h.get('NetworkMode') == 'none',
        'readonly_root': h.get('ReadonlyRootfs') is True,
        'no_host_mounts': not h.get('Binds') and not info.get('Mounts'),
        'memory_no_swap': h.get('Memory') == MEMORY and h.get('MemorySwap') == MEMORY,
        'one_cpu_pinned': h.get('NanoCpus') == 1000000000 and h.get('CpusetCpus') == str(cpuset),
        'slot_cgroup_parent': (h.get('CgroupParent') or None) == cgroup_parent,
        'pids_32': h.get('PidsLimit') == 32,
        'fd_128': ulimits.get('nofile') == (128, 128),
        'no_caps': caps(h.get('CapDrop')) == {'ALL'} and not h.get('CapAdd'),
        'no_new_privileges': any(x in ('no-new-privileges', 'no-new-privileges:true') for x in h.get('SecurityOpt') or []),
        'uid_1000': config.get('User') == '1000:1000',
        'private_cgroup': h.get('CgroupnsMode') == 'private',
        'ipc_none': h.get('IpcMode') == 'none',
        'no_devices': not h.get('Devices') and not h.get('DeviceRequests') and not h.get('Privileged'),
        'private_pid_namespace': h.get('PidMode') in ('', None),
        'bounded_tmpfs': set(tmpfs) == {'/tmp', '/workspace'}
            and f'size={SCRATCH}' in tmpfs.get('/tmp', '') and f'size={ARTIFACT}' in tmpfs.get('/workspace', ''),
    }
    if not all(checks.values()):
        raise InfrastructureFailure('unsafe workspace configuration: ' + ', '.join(k for k, v in checks.items() if not v))
    return checks


class Docker:
    def __init__(self, binary, host, cpuset='0', cgroup_parent=None):
        self.prefix = [str(Path(binary).resolve()), '--host', host]
        placement_options(cpuset, cgroup_parent)
        self.cpuset, self.cgroup_parent = str(cpuset), cgroup_parent

    def verify(self, info, image):
        return verify_configuration(info, image, self.cpuset, self.cgroup_parent)

    def run(self, *args, timeout=30, input=None):
        result = subprocess.run(self.prefix + list(args), input=input, capture_output=True, timeout=timeout)
        if result.returncode:
            raise InfrastructureFailure('docker ' + str(args[0]) + ' failed: ' + result.stderr.decode(errors='replace')[:1000])
        return result.stdout

    def inspect(self, name):
        return json.loads(self.run('inspect', name))[0]

    def create(self, image, name, public_tar=None, source_tar=None):
        subprocess.run(development_command(self.prefix, image, name, self.cpuset, self.cgroup_parent),
                       check=True, capture_output=True, timeout=30)
        try:
            self.run('start', name)
            checks = self.verify(self.inspect(name), image)
            self.run('exec', name, '/bin/mkdir', '/workspace/submission')
            archive, directory = (public_tar, '/workspace') if public_tar else (source_tar, '/workspace/submission')
            if archive:
                with Path(archive).open('rb') as stream:
                    result = subprocess.run(self.prefix + ['exec', '-i', '--workdir', directory, name,
                                                          '/bin/tar', '--no-same-owner', '-xf', '-'],
                                            stdin=stream, capture_output=True, timeout=90)
                if result.returncode:
                    raise InfrastructureFailure('copy verified archive into workspace failed')
            return checks
        except BaseException:
            self.remove(name)
            raise

    def pause(self, name):
        if not self.inspect(name)['State'].get('Paused'):
            self.run('pause', name)
        if not self.inspect(name)['State'].get('Paused'):
            raise InfrastructureFailure('workspace did not pause')

    def remove(self, name):
        self.run('rm', '--force', name)


def _export_members(archive, *, max_bytes):
    seen, files, total, count = set(), [], 0, 0
    for member in archive:
        count += 1
        if count > 20000:
            raise SubmissionFailure('too many submission paths')
        name = member.name
        if '\\' in name or '\x00' in name or len(name.encode()) > 4096 or name.startswith('/'):
            raise SubmissionFailure('unsafe submission path')
        parts = name.split('/')
        if '..' in parts:
            raise SubmissionFailure('submission path traversal')
        while parts and parts[0] == '.':
            parts.pop(0)
        normalized = '/'.join(x for x in parts if x not in ('', '.'))
        if not normalized:
            if not member.isdir():
                raise SubmissionFailure('submission root must be a directory')
            continue
        if normalized in seen:
            raise SubmissionFailure('duplicate submission path')
        seen.add(normalized)
        if not (member.isdir() or member.isfile()) or member.issym() or member.islnk() or member.sparse is not None:
            raise SubmissionFailure('submission contains a link, sparse or special file')
        if member.size < 0 or (member.isdir() and member.size != 0):
            raise SubmissionFailure('invalid member size')
        total += member.size
        if total > max_bytes:
            raise SubmissionFailure('submission exceeds artifact limit')
        files.append((normalized, member))
    if not any(member.isfile() for _, member in files):
        raise SubmissionFailure('empty submission')
    directories = {name for name, member in files if member.isdir()}
    regular = {name for name, member in files if member.isfile()}
    if directories & regular or any(parent.as_posix() in regular for name in seen for parent in PurePosixPath(name).parents if str(parent) != '.'):
        raise SubmissionFailure('submission file/directory collision')
    return files, total


def validate_and_extract(source_tar, destination, canonical_tar, *, max_bytes=ARTIFACT):
    """Validate the entire untrusted archive before manually creating regular files.

    No tar extraction API is called. Input must be an uncompressed Docker archive.
    Output directories are newly created and cannot contain links from prior runs.
    """
    source_tar, destination, canonical_tar = map(Path, (source_tar, destination, canonical_tar))
    if source_tar.stat().st_size > max_bytes + 32 * 1024**2:
        raise SubmissionFailure('export archive exceeds bounded overhead')
    if destination.exists() or canonical_tar.exists():
        raise InfrastructureFailure('snapshot destination must be new')
    try:
        with tarfile.open(source_tar, 'r:') as archive:
            files, total = _export_members(archive, max_bytes=max_bytes)
            # Force all data reads before writing any host path; this catches
            # truncated member bodies as well as malicious headers.
            for _, member in files:
                if member.isfile():
                    stream = archive.extractfile(member)
                    remaining = member.size
                    while remaining:
                        block = stream.read(min(remaining, 1024 * 1024))
                        if not block:
                            raise SubmissionFailure('truncated submission archive')
                        remaining -= len(block)
            destination.mkdir(parents=True)
            hashes = {}
            with tarfile.open(canonical_tar, 'w') as canonical:
                for name, member in sorted(files):
                    path = destination / name
                    if member.isdir():
                        path.mkdir(parents=True, exist_ok=True)
                        continue
                    path.parent.mkdir(parents=True, exist_ok=True)
                    stream = archive.extractfile(member)
                    digest = hashlib.sha256()
                    with path.open('xb') as output:
                        for block in iter(lambda: stream.read(1024 * 1024), b''):
                            output.write(block)
                            digest.update(block)
                    mode = 0o555 if member.mode & 0o111 else 0o444
                    path.chmod(mode)
                    info = tarfile.TarInfo(name)
                    info.size, info.mode, info.mtime = member.size, mode, 0
                    with path.open('rb') as content:
                        canonical.addfile(info, content)
                    hashes[name] = digest.hexdigest()
    except (tarfile.TarError, EOFError, OSError) as exc:
        raise SubmissionFailure('invalid or unreadable submission archive') from exc
    source_hash = hashlib.sha256(json.dumps(hashes, sort_keys=True, separators=(',', ':')).encode()).hexdigest()
    return {'sha256': source_hash, 'bytes': total, 'files': hashes, 'archive_sha256': sha(canonical_tar)}


def freeze(docker, name, output, label):
    """Pause first; stream a bounded archive; validate; only then delete container."""
    docker.pause(name)
    output = Path(output)
    raw = output / (label + '.docker.tar')
    image = docker.inspect(name)['Image']
    exporter = 'frontieram-export-' + uuid.uuid4().hex
    process = subprocess.Popen(docker.prefix + ['run', '--rm', '--name', exporter, '--network', 'none',
                               '--pid', 'container:' + name, '--read-only', '--cap-drop', 'ALL',
                               '--security-opt', 'no-new-privileges:true', '--user', '1000:1000',
                               *placement_options(docker.cpuset, docker.cgroup_parent), '--memory', str(MEMORY),
                               '--memory-swap', str(MEMORY), '--pids-limit', '32', '--ulimit', 'nofile=128:128',
                               '--ipc', 'none', '--log-driver', 'none', '--entrypoint', 'python3',
                               image, '-c', EXPORT_SCRIPT],
                               stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    expired = threading.Event()
    def timeout():
        expired.set()
        process.kill()
    watchdog = threading.Timer(90, timeout)
    watchdog.start()
    total = 0
    try:
        with raw.open('xb') as stream:
            while True:
                block = process.stdout.read(1024 * 1024)
                if not block:
                    break
                total += len(block)
                if total > RAW_TAR_CAP:
                    raise SubmissionFailure('export exceeds artifact plus archive overhead limit')
                stream.write(block)
        error = process.stderr.read(4096)
        code = process.wait(timeout=5)
        if code == 42:
            raise SubmissionFailure(error.decode(errors='replace')[:500])
        if code or expired.is_set():
            raise InfrastructureFailure('paused workspace export failed: ' + error.decode(errors='replace')[:500])
        result = validate_and_extract(raw, output / label, output / (label + '.tar'))
        result.update({'paused_before_export': True, 'frozen_at_utc': now(), 'raw_archive_bytes': total})
        save(output / (label + '_manifest.json'), result)
        raw.unlink()
        docker.remove(name)
        return result
    finally:
        watchdog.cancel()
        if process.poll() is None:
            process.kill()
            process.wait(timeout=5)
        process.stdout.close()
        process.stderr.close()
        subprocess.run(docker.prefix + ['rm', '--force', exporter], capture_output=True, timeout=15)


def cpu_usage(docker, name):
    data = docker.run('exec', name, '/bin/cat', '/sys/fs/cgroup/cpu.stat', timeout=5).decode()
    return int(dict(line.split() for line in data.splitlines())['usage_usec']) / 1000000


def offline_build(docker, image, run):
    source = run / 'source'
    if not (source / 'build.sh').exists():
        return {'status': 'not_requested', 'artifact': str(source), 'archive': str(run / 'source.tar')}
    name = 'frontieram-build-' + uuid.uuid4().hex
    checks = docker.create(image, name, source_tar=run / 'source.tar')
    process = None
    frozen = False
    try:
        # This is an isolated working copy; the preserved source snapshot stays immutable.
        docker.run('exec', '--workdir', '/workspace/submission', name, '/bin/chmod', '-R', 'u+rwX', '.')
        cpu_start = cpu_usage(docker, name)
        started = time.monotonic()
        status = 'ok'
        # Keep build stdout bounded in the container's scratch filesystem;
        # trusted exporter snapshots only submission, never this log.
        with (run / 'build_driver.log').open('wb') as log:
            process = subprocess.Popen(docker.prefix + ['exec', '--workdir', '/workspace/submission', name,
                                       '/bin/bash', '-c', 'bash ./build.sh > /tmp/build.log 2>&1'],
                                       stdin=subprocess.DEVNULL, stdout=log, stderr=log)
            while process.poll() is None:
                if time.monotonic() - started >= 600:
                    status = 'build_wall_limit'
                    break
                if cpu_usage(docker, name) - cpu_start > 600:
                    status = 'build_cpu_limit'
                    break
                time.sleep(.2)
        # Freeze all descendants even when build.sh returned with background work.
        cpu = cpu_usage(docker, name) - cpu_start
        docker.pause(name)
        elapsed = time.monotonic() - started
        if status == 'ok' and (cpu > 600 or elapsed > 600):
            status = 'build_cpu_limit' if cpu > 600 else 'build_wall_limit'
        if status == 'ok' and process.returncode != 0:
            status = 'build_failure'
        info = docker.inspect(name)
        if info['State'].get('OOMKilled'):
            status = 'build_memory_limit'
        result = {'status': status, 'wall_seconds': elapsed, 'cpu_seconds': cpu,
                  'configuration_checks': checks, 'exit_code': process.returncode}
        if status != 'ok':
            save(run / 'build_result.json', result)
            raise SubmissionFailure(status)
        result['submission'] = freeze(docker, name, run, 'built')
        frozen = True
        result.update({'artifact': str(run / 'built'), 'archive': str(run / 'built.tar')})
        save(run / 'build_result.json', result)
        return result
    finally:
        if not frozen:
            # Failed artifacts are not evaluated; leave an audit record then discard.
            docker.remove(name)
        if process is not None and process.poll() is None:
            process.kill()
            process.wait(timeout=5)


def configuration(path):
    value = json.loads(Path(path).read_text())
    models = value.get('models')
    if not isinstance(models, list) or len(models) != 5:
        raise ValueError('config must contain exactly the five authorized model labels, once each')
    for model in models:
        if not isinstance(model, dict) or not isinstance(model.get('label', model.get('alias')), str):
            raise ValueError('model requires a label')
        label = model.get('label', model.get('alias'))
        model['label'] = label
        model['alias'] = label.lower()
    if {x['alias'] for x in models} != set(ALIASES):
        raise ValueError('model labels must be: ' + ', '.join(ALIASES))
    if (value.get('development_seconds', 1800) != 1800
            or value.get('development_seconds_per_model', 1800) != 1800
            or value.get('attempts_per_model', 1) != 1):
        raise ValueError('pilot requires one attempt per model and 1800 seconds')
    for model in models:
        if (not isinstance(model.get('model_id'), str) or not model['model_id'].strip()
                or '\n' in model['model_id'] or model['model_id'] == 'EXACT_VERIFIED_ID'):
            raise ValueError('every model must have a verified exact model_id')
        if model.get('resolved') is False:
            raise ValueError('unresolved model ID')
        if model['model_id'] != MODEL_IDS[model['alias']]:
            raise ValueError('model ID differs from the five authorized verified models')
    return value


def baseline_preflight(path, manifest, image):
    """Fail before model development if the supposed frozen reference is incomplete."""
    baseline = json.loads(Path(path).read_text())
    cases = json.loads(Path(manifest).read_text())['cases']
    if len(baseline.get('runs', [])) != 3 * len(cases):
        raise InfrastructureFailure('baseline must contain all three repetitions of every case')
    if any(row.get('image_id') != image for row in baseline['runs']):
        raise InfrastructureFailure('baseline was not measured in the locked image')
    spec = importlib.util.spec_from_file_location('pilot_reference_validation', ROOT.parent / 'public/tools/scoring.py')
    scoring = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(scoring)
    scoring.score_submission(cases, baseline['aggregates'], baseline['aggregates'])


def package_preflight(package, image_lock):
    audit = json.loads((package / 'package_audit.json').read_text())
    lock = json.loads(image_lock.read_text())
    if audit['image']['image_id'] != lock['image_id'] or audit['archive_sha256'] != sha(package / 'public_pilot.tar'):
        raise InfrastructureFailure('public package/image lock hash mismatch')
    if audit.get('private_files') != 0 or audit.get('development_seconds') != 1800:
        raise InfrastructureFailure('public package policy mismatch')
    return audit, lock


def harness_prompt():
    return ('Read /workspace/task/PILOT.md first, then INSTRUCTION.md, API.md and ENVIRONMENT.md. '
            'Solve the published numerical optimization task using only workspace.exec. '
            'The PILOT.md 30-minute development limit supersedes older two-hour text. '
            'Create the complete source and executable run.sh in /workspace/submission. '
            'The evaluator runs ./run.sh with the submission directory as working directory; use relative paths. '
            'You may provide build.sh for the separate clean offline build stage. '
            'Use public development tests and installed general libraries. No hidden-test query is available. '
            'When ready, finish your turn; this irrevocably freezes your one submission. '
            'All programs and tools share the published CPU, memory and wall-clock budgets.')


def terminate(process):
    if process.poll() is not None:
        return
    try:
        os.killpg(process.pid, signal.SIGTERM)
    except ProcessLookupError:
        return
    try:
        process.wait(timeout=5)
    except subprocess.TimeoutExpired:
        os.killpg(process.pid, signal.SIGKILL)
        process.wait(timeout=5)


def classify_development(result, *, process_returncode, audit_rows=(), shared_deadline_unix=None):
    """Attribute termination before inspecting the candidate submission.

    Only a verified normal final turn or a verified global deadline may proceed
    to build/evaluation. A timeout in the request that straddles the deadline is
    expected cancellation, provided timestamp evidence establishes that boundary.
    Earlier errors, output truncation, request caps and unexplained exits remain
    infrastructure/inference failures even when the source snapshot is empty.
    """
    tolerance = DEADLINE_CLOCK_TOLERANCE_SECONDS
    decision = {'version': 2, 'evaluation_eligible': False,
                'classification': 'infrastructure_failure', 'reason': None,
                'process_returncode': process_returncode,
                'deadline_clock_tolerance_seconds': tolerance,
                'expected_deadline_errors': [], 'terminal_errors': []}

    def fail(reason):
        decision['reason'] = reason
        return decision

    def finite(value):
        return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value)

    if not isinstance(result, dict) or not result:
        return fail('harness result missing')
    decision.update(execution_status=result.get('execution_status'),
                    codex_exit_code=result.get('exit_code'),
                    deadline_reached=result.get('deadline_reached'),
                    stop_reason=result.get('stop_reason'))
    freeze_record = result.get('workspace_freeze')
    if not isinstance(freeze_record, dict) or freeze_record.get('success') is not True:
        return fail('harness did not confirm a successful boundary freeze')
    if type(process_returncode) is not int or process_returncode != 0:
        return fail('harness wrapper did not exit successfully')
    if type(result.get('exit_code')) is not int:
        return fail('Codex exit status missing or invalid')

    deadline = result.get('development_deadline_at_unix')
    started = result.get('started_at_unix')
    stopped = result.get('stop_detected_at_unix')
    if not all(finite(value) for value in (shared_deadline_unix, deadline, started, stopped)):
        return fail('harness boundary timestamps missing or invalid')
    decision.update(shared_deadline_at_unix=shared_deadline_unix,
                    development_deadline_at_unix=deadline,
                    deadline_clock_difference_seconds=deadline - shared_deadline_unix,
                    stop_relative_to_deadline_seconds=stopped - deadline)
    if abs(deadline - shared_deadline_unix) > tolerance or not started < deadline:
        return fail('harness and controller development deadlines do not agree')
    if stopped < started:
        return fail('harness stop timestamp precedes its start')

    # Usage summaries duplicate final bridge errors but may omit wall timestamps.
    # Prefer the matching audit row, retaining unpaired summary errors fail-closed.
    usage = result.get('usage_accounting', {})
    if not isinstance(usage, dict) or not isinstance(usage.get('errors', []), list):
        return fail('harness usage/error record malformed')
    if any(not isinstance(row, dict) for row in list(usage.get('errors', [])) + list(audit_rows)):
        return fail('harness audit record malformed')
    errors = {}
    for row in list(usage.get('errors', [])) + [
            row for row in audit_rows if row.get('event') == 'error']:
        if (row.get('request') is not None and type(row.get('request')) is not int
                or not isinstance(row.get('category'), str)):
            return fail('harness terminal error metadata malformed')
        key = (row.get('request'), row.get('category'))
        previous = errors.get(key)
        if previous is None or finite(row.get('time_unix')):
            errors[key] = row
    requests = {row.get('request'): row.get('time_unix') for row in audit_rows
                if row.get('event') == 'request' and type(row.get('request')) is int}
    normal_final = (result.get('execution_status') == 'model_turn_completed'
                    and result.get('finished_turn') is True
                    and result.get('deadline_reached') is False
                    and result.get('stop_reason') == 'turn_completed'
                    and result['exit_code'] == 0
                    and stopped <= deadline + tolerance)
    normal_deadline = (result.get('execution_status') == 'development_deadline'
                       and result.get('deadline_reached') is True
                       and result.get('stop_reason') == 'development_deadline'
                       and stopped >= deadline - tolerance)
    for error in errors.values():
        happened = error.get('time_unix')
        request_started = requests.get(error.get('request'))
        summary = {'request': error.get('request'), 'category': error.get('category'),
                   'error_at_unix': happened, 'request_started_at_unix': request_started}
        is_deadline_cancellation = (normal_deadline and error.get('category') in (
                                       'upstream_timeout', 'upstream_development_deadline', 'upstream_cancelled')
                                   and finite(happened) and finite(request_started)
                                   and started - tolerance <= request_started < deadline
                                   and happened >= deadline - tolerance)
        decision['expected_deadline_errors' if is_deadline_cancellation else 'terminal_errors'].append(summary)
    if decision['terminal_errors']:
        categories = sorted({str(row['category']) for row in decision['terminal_errors']})
        return fail('harness or inference terminated abnormally: ' + ', '.join(categories))
    if normal_final:
        decision.update(evaluation_eligible=True, classification='normal_final_turn',
                        reason='normal final turn and successful boundary freeze')
        return decision
    if normal_deadline:
        # SIGTERM/SIGKILL are expected watchdog shutdowns. A code of 1 is also
        # attributable when the in-flight request timed out at this exact bound.
        expected_codes = {0, -signal.SIGTERM, -signal.SIGKILL, 128 + signal.SIGTERM,
                          128 + signal.SIGKILL}
        if decision['expected_deadline_errors']:
            expected_codes.add(1)
        if result['exit_code'] not in expected_codes:
            return fail('unexpected Codex exit at the development deadline')
        decision.update(evaluation_eligible=True, classification='normal_development_deadline',
                        reason='global development deadline and successful boundary freeze')
        return decision
    return fail('coding harness did not record a normal final turn or global deadline')


def develop(docker, image, model, run, args):
    name = 'frontieram-dev-' + uuid.uuid4().hex
    checks = docker.create(image, name, public_tar=args.package / 'public_pilot.tar')
    save(run / 'workspace_preflight.json', {'checks': checks, 'container': name, 'image_id': image})
    prompt_path = run / 'prompt.txt'
    prompt_path.write_text(harness_prompt())
    command = [sys.executable, str(HARNESS_ENTRY), '--model', model['model_id'],
               '--key-file', str(args.key_file), '--docker', str(args.docker), '--docker-host', args.docker_host,
               '--container', name, '--run-dir', str(run / 'harness'), '--minutes', '30', '--freeze-at-stop',
               '--prompt-file', str(prompt_path)]
    if args.codex:
        command += ['--codex', args.codex]
    started = time.monotonic()
    process = None
    exported = False
    shared_deadline_unix = None
    try:
        with (run / 'harness_stdout.log').open('wb') as stdout, (run / 'harness_stderr.log').open('wb') as stderr:
            shared_deadline_unix = time.time() + 1800.0
            command += ['--deadline-unix', str(shared_deadline_unix)]
            save(run / 'development_boundary.json', {
                'development_deadline_at_unix': shared_deadline_unix,
                'development_seconds': 1800,
                'deadline_clock_tolerance_seconds': DEADLINE_CLOCK_TOLERANCE_SECONDS,
                'harness_entry': str(HARNESS_ENTRY),
            })
            process = subprocess.Popen(command, stdout=stdout, stderr=stderr, stdin=subprocess.DEVNULL,
                                       start_new_session=True)
            # The harness freezes at its precise budget/turn boundary. This outer
            # independent watchdog also fails closed if the harness itself hangs.
            try:
                process.wait(timeout=1800)
            except subprocess.TimeoutExpired:
                docker.pause(name)
                # Give the inner watchdog enough time to persist its result.
                # The workspace is already frozen, so this is no extra work time.
                try:
                    process.wait(timeout=15)
                except subprocess.TimeoutExpired:
                    terminate(process)
        docker.pause(name)
        result_file = run / 'harness/run_result.json'
        result = json.loads(result_file.read_text()) if result_file.exists() else {}
        audit_file = run / 'harness/bridge_audit.jsonl'
        audit_rows = [json.loads(line) for line in audit_file.read_text().splitlines()] if audit_file.exists() else []
        classification = classify_development(
            result, process_returncode=process.returncode, audit_rows=audit_rows,
            shared_deadline_unix=shared_deadline_unix)
        save(run / 'development_classification.json', classification)
        problem = None if classification['evaluation_eligible'] else classification['reason']
        try:
            source = freeze(docker, name, run, 'source')
            exported = True
        except SubmissionFailure as exc:
            if problem:
                raise InfrastructureFailure(problem + '; no accepted source snapshot') from exc
            raise
        if problem:
            raise InfrastructureFailure(problem + '; frozen source retained')
        return {'source': source, 'harness': result, 'elapsed_seconds': time.monotonic() - started,
                'container': name, 'configuration_checks': checks, 'classification': classification}
    finally:
        if not exported:
            # Never resume: preserve the paused container when safe export fails,
            # so inspection cannot accidentally alter the claimed snapshot.
            try:
                docker.pause(name)
                save(run / 'retained_paused_container.json', {'container': name, 'reason': 'export or infrastructure failure'})
            except Exception:
                pass
        if process is not None:
            terminate(process)


def evaluation(args, docker, image, run, artifact):
    wrapper = run / 'docker-endpoint'
    wrapper.write_text('#!/bin/sh\nexec ' + ' '.join(shlex.quote(x) for x in docker.prefix) + ' "$@"\n')
    wrapper.chmod(0o700)
    output = run / 'evaluation.json'
    command = [sys.executable, str(ROOT / 'evaluator/run_docker.py'), '--image', image,
               '--docker', str(wrapper), '--submission', str(artifact), '--command', './run.sh',
               '--manifest', str(args.manifest), '--output', str(output), '--repeats', '3',
               '--cpuset', docker.cpuset]
    if docker.cgroup_parent is not None:
        command += ['--cgroup-parent', docker.cgroup_parent]
    with (run / 'evaluator_stdout.log').open('wb') as stdout, (run / 'evaluator_stderr.log').open('wb') as stderr:
        result = subprocess.run(command, stdout=stdout, stderr=stderr, timeout=8 * 3600)
    if result.returncode or not output.exists():
        raise InfrastructureFailure('private evaluator failed; inspect evaluator logs')
    report = json.loads(output.read_text())
    manifest = json.loads(args.manifest.read_text())
    expected = {case['id'] for case in manifest['cases']}
    if set(report['aggregates']) != expected or len(report['runs']) != 3 * len(expected):
        raise InfrastructureFailure('incomplete private evaluation')
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', type=Path, required=True)
    parser.add_argument('--key-file', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--docker', type=Path, default=DEFAULT_DOCKER)
    parser.add_argument('--docker-host', default=DEFAULT_HOST)
    parser.add_argument('--image-lock', type=Path, default=ROOT / 'image_lock.json')
    parser.add_argument('--package', type=Path, default=DEFAULT_PACKAGE)
    parser.add_argument('--manifest', type=Path, default=ROOT / 'private_pilot/dataset_v1/manifest.json')
    parser.add_argument('--baseline', type=Path)
    parser.add_argument('--codex')
    parser.add_argument('--preflight-only', action='store_true')
    args = parser.parse_args()
    if args.codex:
        args.codex = str(Path(args.codex).resolve())
    config = configuration(args.config)
    if not args.preflight_only and config.get('user_authorized') is not True:
        parser.error('this proposed campaign is not authorized for model generation; use preflight-only')
    args.output = args.output.resolve()
    audit, lock = package_preflight(args.package, args.image_lock)
    docker = Docker(args.docker, args.docker_host)
    image = lock['image_id']
    if json.loads(docker.run('image', 'inspect', image))[0]['Id'] != image:
        raise InfrastructureFailure('frozen image unavailable')
    if not args.key_file.is_file() or not args.manifest.is_file():
        parser.error('key file and private manifest must exist')
    distribution = args.manifest.parent / 'PUBLIC_DISTRIBUTION.json'
    if not distribution.exists() or sha(distribution) != audit['distribution_sha256']:
        raise InfrastructureFailure('private dataset does not match the published pilot distribution')
    if args.baseline:
        baseline_preflight(args.baseline, args.manifest, image)
    source_files = ['campaign_runner_v2.py', 'build_public_pilot_v2.py', 'harness_recovery/run_codex.py',
                    'harness_recovery/docker_mcp.py', 'harness_recovery/bridge.py',
                    'harness_recovery/metadata/retry_policies.json',
                    'evaluator/run_docker.py', 'evaluator/supervisor.py']
    source_files += [str(path.relative_to(ROOT)) for path in sorted((ROOT / 'harness_recovery/metadata').glob('*.protocol.json'))]
    source_files += [str(path.relative_to(ROOT)) for path in sorted((ROOT / 'harness_recovery/metadata').glob('*.detail.json'))]
    if args.output.exists():
        parser.error('output must be new: one-attempt runs cannot be resumed or overwritten')
    args.output.mkdir(parents=True)
    record = {'controller_version': 'v2-harness-attribution',
              'deadline_clock_tolerance_seconds': DEADLINE_CLOCK_TOLERANCE_SECONDS,
              'created_at_utc': now(), 'status': 'preflight' if args.preflight_only else 'running',
              'configuration': config, 'image': lock, 'public_package_sha256': audit['archive_sha256'],
              'controller_source_sha256': {name: sha(ROOT / name) for name in source_files},
              'environment': audit['environment'],
              'manifest_sha256': sha(args.manifest), 'baseline_sha256': sha(args.baseline) if args.baseline else None,
              'resources': {'cpu': 1, 'cpuset': '0', 'memory_bytes': MEMORY, 'swap_bytes': 0,
                            'scratch_bytes': SCRATCH, 'workspace_bytes': ARTIFACT, 'pids': 32, 'nofile': 128},
              'execution_order': 'one model development, freeze, build, all evaluation; then next model',
              'attempts_per_model': 1, 'development_seconds': 1800, 'runs': []}
    save(args.output / 'campaign.json', record)
    if args.preflight_only:
        name = 'frontieram-preflight-' + uuid.uuid4().hex
        docker.create(image, name, public_tar=args.package / 'public_pilot.tar')
        try:
            docker.run('exec', '--workdir', '/workspace/submission', name, '/bin/bash', '-c',
                       'printf "#!/bin/sh\\nexit 0\\n" > run.sh; chmod +x run.sh')
            snapshot = freeze(docker, name, args.output, 'preflight_source')
            save(args.output / 'preflight.json', {'passed': True, 'snapshot': snapshot, 'model_requests': 0})
            record.update(status='preflight_passed', finished_at_utc=now())
            save(args.output / 'campaign.json', record)
        except BaseException:
            docker.remove(name)
            raise
        print('Preflight passed; no model generation was requested.')
        return
    for index, model in enumerate(config['models'], 1):
        run = args.output / f'{index:02d}-{re.sub(r"[^a-z0-9]+", "-", model["alias"].lower()).strip("-")}'
        run.mkdir()
        row = {'alias': model['alias'], 'model_id': model['model_id'], 'attempt': 1,
               'started_at_utc': now(), 'status': 'developing', 'directory': str(run), 'score': None,
               'traces': {'events': str(run / 'harness/codex_events.jsonl'),
                          'bridge': str(run / 'harness/bridge_audit.jsonl'),
                          'source': str(run / 'source.tar'), 'evaluation': str(run / 'evaluation.json')}}
        record['runs'].append(row)
        save(args.output / 'campaign.json', record)
        try:
            row['development'] = develop(docker, image, model, run, args)
            row['source_sha256'] = row['development']['source']['sha256']
            row['build'] = offline_build(docker, image, run)
            artifact = Path(row['build']['artifact'])
            if not (artifact / 'run.sh').is_file() or not os.access(artifact / 'run.sh', os.X_OK):
                raise SubmissionFailure('missing executable run.sh')
            row['status'] = 'evaluating'
            save(args.output / 'campaign.json', record)
            report = evaluation(args, docker, image, run, artifact)
            row['valid_cases'] = sum(item['valid'] for item in report['aggregates'].values())
            row['total_cases'] = len(report['aggregates'])
            row['status'] = 'valid_submission' if row['valid_cases'] == row['total_cases'] else 'invalid_submission'
            if args.baseline:
                spec = importlib.util.spec_from_file_location('pilot_scoring', ROOT.parent / 'public/tools/scoring.py')
                scoring = importlib.util.module_from_spec(spec)
                spec.loader.exec_module(scoring)
                reference = json.loads(args.baseline.read_text())['aggregates']
                score = scoring.score_submission(json.loads(args.manifest.read_text())['cases'], report['aggregates'], reference)
                save(run / 'score.json', score)
                row['score'] = score['score']
        except SubmissionFailure as exc:
            row.update(status='invalid_submission', reason=str(exc), score=0.0)
        except (InfrastructureFailure, subprocess.SubprocessError, OSError, ValueError, KeyError) as exc:
            row.update(status='infrastructure_failure', reason=str(exc), score=None)
        finally:
            row['finished_at_utc'] = now()
            save(run / 'result.json', row)
            save(args.output / 'campaign.json', record)
    record.update(status='complete', finished_at_utc=now())
    save(args.output / 'campaign.json', record)
    print(json.dumps({'output': str(args.output), 'runs': [{k: row.get(k) for k in ('alias', 'status', 'score')} for row in record['runs']]}, indent=2))


if __name__ == '__main__':
    main()
