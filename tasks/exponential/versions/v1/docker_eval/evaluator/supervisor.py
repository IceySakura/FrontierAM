#!/usr/bin/env python3
"""Trusted Linux PID 1. This file is baked into a read-only image, never submitted.

Only this root process has SETUID, SETGID and KILL. Solver uid/gid is 65534 and
all effective/permitted/inheritable/ambient capability sets must be empty.
stdout carries solver bytes; stderr carries supervisor-authored JSON only.
Candidate stderr is captured, bounded and JSON-escaped, never passed through.
"""
import argparse
import ctypes
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import threading
import time


CGROUP = Path('/sys/fs/cgroup')
CAP_FIELDS = ('CapEff', 'CapPrm', 'CapInh', 'CapAmb')


def status_fields(path='/proc/self/status'):
    return dict(line.split(':', 1) for line in Path(path).read_text().splitlines() if ':' in line)


def validate_unprivileged(fields):
    if any(int(fields.get(name, '-1').strip(), 16) != 0 for name in CAP_FIELDS):
        raise RuntimeError('candidate capabilities not empty')
    if fields.get('NoNewPrivs', '').strip() != '1':
        raise RuntimeError('candidate no_new_privs missing')
    if any(int(x) != 65534 for x in fields.get('Uid', '').split()):
        raise RuntimeError('candidate UID not isolated')
    if len(fields.get('Uid', '').split()) != 4:
        raise RuntimeError('candidate UID fields missing')


def drop_identity():
    os.setgroups([])
    os.setgid(65534)
    os.setuid(65534)
    libc = ctypes.CDLL(None, use_errno=True)
    if libc.prctl(38, 1, 0, 0, 0) != 0:  # PR_SET_NO_NEW_PRIVS
        raise OSError(ctypes.get_errno(), 'prctl no_new_privs')
    validate_unprivileged(status_fields())


def cgroup_metrics(root=CGROUP):
    cpu = dict(line.split() for line in (root / 'cpu.stat').read_text().splitlines())
    events = dict(line.split() for line in (root / 'memory.events').read_text().splitlines())
    return {'cpu_seconds': int(cpu['usage_usec']) / 1_000_000,
            'memory_bytes': int((root / 'memory.peak').read_text()),
            'oom_kill': int(events.get('oom_kill', 0))}


def emit(event, **payload):
    sys.stderr.write(json.dumps({'event': event, **payload}, separators=(',', ':'), allow_nan=False) + '\n')
    sys.stderr.flush()


def kill_namespace_children():
    # Dedicated PID namespace: no host processes are visible here. KILL is
    # needed because the untrusted children have a different UID.
    for path in Path('/proc').iterdir():
        if path.name.isdigit() and int(path.name) != os.getpid():
            try:
                os.kill(int(path.name), signal.SIGKILL)
            except ProcessLookupError:
                pass


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument('--cpu-seconds', type=float, required=True)
    parser.add_argument('--stderr-bytes', type=int, default=4194304)
    parser.add_argument('command', nargs=argparse.REMAINDER)
    args = parser.parse_args(argv)
    command = args.command[1:] if args.command[:1] == ['--'] else args.command
    if not command or not (0 < args.cpu_seconds <= 240) or not (0 < args.stderr_bytes <= 4194304):
        parser.error('invalid supervisor limits/command')
    if os.getuid() != 0 or os.getpid() != 1:
        raise RuntimeError('trusted supervisor must be root PID 1 in its own container')
    before = cgroup_metrics()  # Fail closed if cgroup v2 accounting unavailable.
    environment = {'PATH': os.environ.get('PATH', '/usr/local/bin:/usr/bin:/bin'),
                   'HOME': '/tmp', 'TMPDIR': '/tmp', 'LANG': 'C.UTF-8',
                   'OMP_NUM_THREADS': '1', 'OPENBLAS_NUM_THREADS': '1', 'MKL_NUM_THREADS': '1',
                   'NUMEXPR_NUM_THREADS': '1', 'VECLIB_MAXIMUM_THREADS': '1',
                   'PYTHONDONTWRITEBYTECODE': '1'}
    try:
        child = subprocess.Popen(command, stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                                 stderr=subprocess.PIPE, cwd='/submission', env=environment,
                                 preexec_fn=drop_identity, start_new_session=True, bufsize=0)
    except Exception as exc:
        emit('final', status='launch_failure', detail=type(exc).__name__, returncode=None,
             stderr_bytes=0, stderr='', **cgroup_metrics())
        return 125
    emit('supervisor_ready', protocol='FAMEXP2-docker-v1', supervisor_uid=0,
         candidate_uid=65534, no_new_privs=True, candidate_capabilities_empty=True,
         cgroup_v2=True, measurement='whole-container cgroup, including supervisor')
    state = {'first_input': None, 'last_output': None, 'stdout_done': False,
             'stderr_done': False, 'stderr_bytes': 0, 'stderr': bytearray(), 'failure': None}

    def write_all(fd, data):
        view = memoryview(data)
        while view:
            count = os.write(fd, view)
            view = view[count:]

    def input_relay():
        try:
            while True:
                data = os.read(0, 65536)
                if not data:
                    break
                if state['first_input'] is None:
                    state['first_input'] = time.monotonic()
                write_all(child.stdin.fileno(), data)
        except (OSError, ValueError):
            pass
        finally:
            try: child.stdin.close()
            except OSError: pass

    def output_relay():
        try:
            while True:
                data = os.read(child.stdout.fileno(), 65536)
                if not data:
                    break
                write_all(1, data)
                state['last_output'] = time.monotonic()
        except (OSError, ValueError):
            state['failure'] = state['failure'] or 'stdout_io_failure'
        finally:
            state['stdout_done'] = True

    def stderr_relay():
        try:
            while True:
                data = os.read(child.stderr.fileno(), 65536)
                if not data:
                    break
                state['stderr_bytes'] += len(data)
                available = max(0, 65536 - len(state['stderr']))
                state['stderr'].extend(data[:available])
                if state['stderr_bytes'] > args.stderr_bytes:
                    state['failure'] = state['failure'] or 'stderr_limit'
        finally:
            state['stderr_done'] = True

    for target in (input_relay, output_relay, stderr_relay):
        threading.Thread(target=target, daemon=True).start()
    exited = None
    while True:
        measured = cgroup_metrics()
        if measured['cpu_seconds'] > args.cpu_seconds:
            state['failure'] = state['failure'] or 'cpu_limit'
        if measured['oom_kill'] > before['oom_kill']:
            state['failure'] = state['failure'] or 'memory_limit'
        if child.poll() is not None and exited is None:
            exited = time.monotonic()
            # Clock on this event's HOST reception supplies the controller's
            # process-exit endpoint without Docker teardown/CLI cleanup time.
            emit('candidate_exit', returncode=child.returncode)
        if exited is not None and state['stdout_done'] and state['stderr_done']:
            break
        if exited is not None and time.monotonic() - exited > 0.5:
            state['failure'] = state['failure'] or 'descendant_holds_pipe'
        if state['failure']:
            kill_namespace_children()
            child.wait(timeout=2)
            exited = exited or time.monotonic()
            break
        time.sleep(0.01)
    remaining = [p for p in Path('/proc').iterdir() if p.name.isdigit() and int(p.name) != os.getpid()]
    if remaining:
        state['failure'] = state['failure'] or 'descendant_remains'
        kill_namespace_children()
    measured = cgroup_metrics()
    status = state['failure'] or ('ok' if child.returncode == 0 else 'nonzero_exit')
    if measured['cpu_seconds'] > args.cpu_seconds:
        status = 'cpu_limit'
    if measured['oom_kill'] > before['oom_kill']:
        status = 'memory_limit'
    end = max(exited, state['last_output'] or exited)
    elapsed = end - state['first_input'] if state['first_input'] is not None else None
    emit('final', status=status, returncode=child.returncode,
         stderr_bytes=state['stderr_bytes'], stderr=bytes(state['stderr']).decode('utf-8', errors='replace'),
         remote_session_seconds=elapsed, **measured)
    return 0 if status == 'ok' else 125


if __name__ == '__main__':
    try:
        raise SystemExit(main())
    except Exception as exc:
        emit('supervisor_error', error=type(exc).__name__, detail=str(exc))
        raise SystemExit(126)
