"""Offline cold-start audit: load each pinned task and construct its official Controller.

Run this with an isolated interpreter containing only Nexus and os_world_v2 host
dependencies. No task setup/evaluate, sandbox allocation, or model call is made.
Each task gets its own subprocess; all outbound socket connections are blocked.
GitLab values below are dummy import-only configuration, not deployment credentials.
"""

from __future__ import annotations

import argparse
import concurrent.futures
import json
import os
import subprocess
import sys
import tempfile
import traceback

from pathlib import Path


def _child(root: Path, selected: str) -> dict:
    import socket

    import requests

    from nexus.extensions.tasks.os_world_v2.adapter import loader, support
    from nexus.extensions.tasks.os_world_v2.adapter.worker import Environment

    def offline_request(*args, **kwargs):
        raise requests.ConnectionError('Offline dependency audit: requests disabled')

    def offline_connect(*args, **kwargs):
        raise OSError('Offline dependency audit: sockets disabled')

    for target, name, replacement in (
        (requests.Session, 'request', offline_request),
        (socket.socket, 'connect', offline_connect),
        (socket.socket, 'connect_ex', offline_connect),
    ):
        setattr(target, name, replacement)
    os.environ.update(
        WEBSITE_HOST_SUFFIX='example.invalid',
        GITLAB_URL='https://gitlab.example.invalid',
        GITLAB_PRIVATE_TOKEN='offline-import-fixture',
        PYTHON_DOTENV_DISABLED='1',
        OSWORLD_FILE_BASE_URL=str(root / 'cache/osworld_v2_assets_v2.1'),
    )
    support.install({})
    try:
        task = loader.load(root, selected)
        with tempfile.TemporaryDirectory(prefix='v2-import-audit-') as cache:
            Environment(task, '127.0.0.1', cache, 'offline-fixture')
        return {'task_id': selected, 'loaded': True}
    except Exception as exc:
        return {
            'task_id': selected,
            'loaded': False,
            'error_type': type(exc).__name__,
            'missing_module': exc.name if isinstance(exc, ModuleNotFoundError) else None,
            'frames': [
                f'{Path(frame.filename).name}:{frame.lineno}:{frame.name}'
                for frame in traceback.extract_tb(exc.__traceback__)
            ],
        }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--repo', type=Path, required=True)
    parser.add_argument('--report', type=Path)
    parser.add_argument('--tasks', default=','.join(f'{i:03d}' for i in range(1, 109)))
    parser.add_argument('--workers', type=int, default=4)
    parser.add_argument('--child', help=argparse.SUPPRESS)
    args = parser.parse_args()
    root = args.repo.resolve()
    if args.child:
        print('NEXUS_V2_AUDIT=' + json.dumps(_child(root, args.child)))
        return 0
    from nexus.extensions.tasks.os_world_v2.adapter.source import task_id, verify_files

    verify_files(root)
    selected = [task_id(value.strip()) for value in args.tasks.split(',')]
    if args.workers < 1:
        parser.error('--workers must be positive')

    def run_one(value):
        with tempfile.TemporaryDirectory(prefix='v2-import-home-') as home:
            env = {key: os.environ[key] for key in ('PATH', 'LANG') if key in os.environ}
            env.update(HOME=home, PYTHONDONTWRITEBYTECODE='1', PYTHON_DOTENV_DISABLED='1')
            command = [
                sys.executable,
                '-m',
                'nexus.extensions.tasks.os_world_v2.scripts.check_host_dependencies',
                '--repo',
                str(root),
                '--child',
                value,
            ]
            try:
                result = subprocess.run(command, env=env, capture_output=True, text=True, timeout=120)
                responses = [
                    line.partition('=')[2] for line in result.stdout.splitlines() if line.startswith('NEXUS_V2_AUDIT=')
                ]
                if len(responses) == 1:
                    return json.loads(responses[0])
                return {
                    'task_id': value,
                    'loaded': False,
                    'error_type': 'WorkerStartupError',
                    'returncode': result.returncode,
                }
            except subprocess.TimeoutExpired:
                return {'task_id': value, 'loaded': False, 'error_type': 'WorkerTimeout'}

    with concurrent.futures.ThreadPoolExecutor(max_workers=args.workers) as pool:
        results = list(pool.map(run_one, selected))
    passed = sum(result['loaded'] for result in results)
    report = {
        'python': sys.version,
        'passed': passed,
        'total': len(results),
        'scope': 'offline task imports and Controller construction only',
        'tasks': results,
    }
    if args.report:
        args.report.write_text(json.dumps(report, indent=2))
    print(
        json.dumps({
            'passed': passed,
            'total': len(results),
            'failures': [result for result in results if not result['loaded']],
        })
    )
    return 0 if passed == len(results) else 1


if __name__ == '__main__':
    raise SystemExit(main())
