"""Runtime-only Judge launcher, dependency preflight and compact progress records.

This file is copied next to test.sh and uses only the standard library until
preflight has verified the image's packages. It never installs dependencies.
"""

import argparse
import contextlib
import importlib
import importlib.metadata
import json
import os
import shutil
import signal
import subprocess
import sys
import time

from pathlib import Path


TIMELINE = Path('/logs/verifier/judge-timing.jsonl')


def mark(stage: str, event: str, **fields: object) -> None:
    TIMELINE.parent.mkdir(parents=True, exist_ok=True)
    record = {'stage': stage, 'event': event, 'time': time.time(), **fields}
    # One append syscall: CLI invocations may share the timeline.
    fd = os.open(TIMELINE, os.O_WRONLY | os.O_CREAT | os.O_APPEND, 0o600)
    try:
        os.write(fd, (json.dumps(record) + '\n').encode())
    finally:
        os.close(fd)
    print('[judge-stage] ' + json.dumps(record), file=sys.stderr, flush=True)


def preflight() -> None:
    from packaging.version import Version

    expected = {'harbor-rewardkit': '0.1.7'}
    for package, version in expected.items():
        installed = importlib.metadata.version(package)
        if installed != version:
            raise RuntimeError(f'Image dependency {package}=={version} required, found {installed}; rebuild the image')
    if Version(importlib.metadata.version('mcp')) >= Version('2'):
        raise RuntimeError('Image dependency mcp<2 required; rebuild the image')
    for module in ('rewardkit', 'mcp', 'litellm'):
        importlib.import_module(module)
    if shutil.which('opencode') is None:
        raise RuntimeError('Image dependency opencode missing; rebuild the image')
    subprocess.run(['opencode', '--version'], check=True, timeout=30)


def process_identity(pid: int) -> str:
    # comm may contain spaces/parentheses; starttime is field 22.
    return Path(f'/proc/{pid}/stat').read_text().rsplit(')', 1)[1].split()[19]


def stop_command(pid_path: str) -> bool:
    """Kill only our recorded session, checking PID reuse before signalling."""
    path = Path(pid_path)
    if not path.exists():
        return False
    record = json.loads(path.read_text())
    pid = int(record['pid'])
    try:
        if process_identity(pid) != record['starttime'] or os.getpgid(pid) != pid:
            return False
        os.killpg(pid, signal.SIGTERM)
        # Kill surviving children even if their group leader exits after TERM.
        time.sleep(0.5)
        with contextlib.suppress(ProcessLookupError):
            os.killpg(pid, signal.SIGKILL)
        return True
    except ProcessLookupError:
        return False
    except FileNotFoundError:
        return False


def read_progress(verifier_dir: str = '/logs/verifier', *, compact: bool = False) -> dict:
    timeline = Path(verifier_dir) / 'judge-timing.jsonl'
    stages = []
    if timeline.exists():
        for line in timeline.read_text().splitlines():
            with contextlib.suppress(json.JSONDecodeError):
                stages.append(json.loads(line))
    tool_events = []
    for path in Path(verifier_dir).glob('opencode_judge/*.latest.jsonl'):
        for line in path.read_text(errors='replace').splitlines():
            try:
                event = json.loads(line)
            except json.JSONDecodeError:
                continue
            part = event.get('part') or {}
            state = part.get('state') or {}
            if event.get('type') == 'tool_use' and state.get('status') in {'completed', 'error'}:
                timing = state.get('time') or {}
                tool_events.append({
                    'trace_file': path.name,
                    'call_id': part.get('callID'),
                    'tool': part.get('tool'),
                    'status': state.get('status'),
                    'start_ms': timing.get('start'),
                    'end_ms': timing.get('end'),
                })
    progress = {
        'stages': stages,
        'tools': tool_events,
        'completed_stages': sum(s.get('event') == 'end' for s in stages),
        'completed_tools': [(t['trace_file'], t['call_id'], t['end_ms']) for t in tool_events],
    }
    direct_status = Path(verifier_dir) / 'direct_responses/status.json'
    direct = json.loads(direct_status.read_text()) if direct_status.exists() else {}
    direct_progress = {
        'direct_successes': direct.get('successful_responses', 0),
        'direct_requests': direct.get('requests', 0),
        'direct_first_request_seconds': direct.get('first_request_seconds'),
        'direct_abort': direct.get('abort_reason'),
    }
    progress.update(direct_progress)
    if compact:
        return {
            **direct_progress,
            'completed_stages': progress['completed_stages'],
            'completed_tools': progress['completed_tools'],
            'last_stage': stages[-1] if stages else None,
        }
    return progress


def run_command(pid_path: str, command: list[str]) -> int:
    def interrupted(signum: int, _frame: object) -> None:
        raise SystemExit(128 + signum)

    signal.signal(signal.SIGTERM, interrupted)
    signal.signal(signal.SIGINT, interrupted)
    child = subprocess.Popen(command, start_new_session=True)
    path = Path(pid_path)
    try:
        temporary = path.with_suffix('.tmp')
        temporary.write_text(json.dumps({'pid': child.pid, 'starttime': process_identity(child.pid)}))
        temporary.replace(path)
        return child.wait()
    finally:
        # The parent stays alive until wait() returns so cancellation from host
        # can terminate the whole child group, not just a remote shell wrapper.
        if child.poll() is None:
            stop_command(pid_path)
            child.wait()
        # The direct child can exit while its shell/OpenCode descendants remain.
        with contextlib.suppress(ProcessLookupError):
            os.killpg(child.pid, signal.SIGKILL)
        path.unlink(missing_ok=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument('action', choices=['stage', 'run'])
    parser.add_argument('name')
    parser.add_argument('command', nargs=argparse.REMAINDER)
    args = parser.parse_args()
    if args.action == 'run':
        raise SystemExit(run_command(args.name, args.command))
    started = time.monotonic()
    mark(args.name, 'start')
    try:
        if args.name == 'dependencies':
            preflight()
        else:
            subprocess.run(args.command, check=True)
    except BaseException as exc:
        mark(args.name, 'error', duration_seconds=time.monotonic() - started, error_type=type(exc).__name__)
        raise
    else:
        mark(args.name, 'end', duration_seconds=time.monotonic() - started)


if __name__ == '__main__':
    main()
