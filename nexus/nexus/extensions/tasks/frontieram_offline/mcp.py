"""The sole candidate-facing tool: execute in a leased, networkless container."""

from __future__ import annotations

import argparse
import json
import math
import os
import re
import stat
import sys
import threading
import time

from pathlib import Path
from typing import Any

from .assets import IMAGE_ID, LockedAssets


LEASE_VERSION = 'frontieram-offline-runtime-v1'


class DeadlineWorkspace:
    """Use the reviewed workspace tool, bounded by an immutable monotonic deadline."""

    def __init__(self, assets: LockedAssets, lease: dict[str, Any]) -> None:
        self.assets = assets
        self.lease = lease
        self.deadline = lease['deadline_monotonic']
        self.module = assets.docker_mcp
        self.workspace = self.module.DockerWorkspace(
            lease['docker_path'],
            lease['docker_host'],
            lease['container'],
        )
        self._timer: threading.Timer | None = None

    @classmethod
    def from_lease(cls, path: str | Path) -> DeadlineWorkspace:
        path = Path(path)
        info = path.lstat()
        if not stat.S_ISREG(info.st_mode) or stat.S_IMODE(info.st_mode) & 0o077 or info.st_uid != os.getuid():
            raise ValueError('The workspace lease must be a private, owned regular file')
        if info.st_size > 16384:
            raise ValueError('Oversized workspace lease')
        lease = json.loads(path.read_text())
        if lease.get('version') != LEASE_VERSION or lease.get('image_id') != IMAGE_ID:
            raise ValueError('Unknown workspace lease contract')
        if not re.fullmatch(r'nexus-frontieram-[0-9a-f]{32}', lease.get('container', '')):
            raise ValueError('The workspace is not owned by this provider')
        if not re.fullmatch(r'[0-9a-f]{64}', lease.get('container_id', '')):
            raise ValueError('The workspace lease has no full container identity')
        deadline = lease.get('deadline_monotonic')
        if isinstance(deadline, bool) or not isinstance(deadline, (int, float)) or not math.isfinite(deadline):
            raise ValueError('The workspace lease requires a finite monotonic deadline')
        if not str(lease.get('docker_host', '')).startswith('unix://'):
            raise ValueError('Only a local Docker daemon socket is permitted')
        result = cls(LockedAssets(lease['source_root']), lease)
        result.verify()
        return result

    def verify(self) -> dict[str, Any]:
        controller = self.assets.controller
        docker = controller.Docker(
            self.lease['docker_path'], self.lease['docker_host'], self.lease['cpuset'], self.lease['cgroup_parent']
        )
        info = docker.inspect(self.lease['container'])
        if info['Id'] != self.lease['container_id']:
            raise RuntimeError('The named container no longer matches its owned lease')
        checks = docker.verify(info, self.assets.image_id)
        self.workspace.verify()
        return checks

    def _pause_at_deadline(self) -> None:
        try:
            self.workspace.freeze()
        except Exception as error:
            print(f'Workspace deadline freeze failed: {type(error).__name__}', file=sys.stderr, flush=True)

    def start_watchdog(self) -> None:
        if self._timer is not None:
            raise RuntimeError('The workspace watchdog was already started')
        self._timer = threading.Timer(max(0.0, self.deadline - time.monotonic()), self._pause_at_deadline)
        self._timer.daemon = True
        self._timer.start()

    def stop_watchdog(self) -> None:
        if self._timer is not None:
            self._timer.cancel()
            self._timer.join(timeout=25)

    def execute(self, arguments: dict[str, Any]) -> dict[str, Any]:
        command, timeout, limit = self.module.validate_arguments(arguments)
        remaining = self.deadline - time.monotonic()
        if remaining <= 0:
            raise RuntimeError('The development deadline has expired; no command was dispatched')
        budget = min(float(timeout), remaining)
        # The fixed Docker exec destination comes from the provider lease. No
        # arguments from the model can select a container, UID, mount or host.
        argv = self.workspace.prefix + [
            'exec',
            '-i',
            '--workdir',
            '/workspace',
            self.workspace.container,
            '/usr/bin/timeout',
            '--signal=KILL',
            '--kill-after=1',
            str(budget),
            '/bin/bash',
            '-lc',
            command,
        ]
        result = self.module.bounded_run(argv, budget + 1.0, limit)
        if time.monotonic() >= self.deadline:
            self._pause_at_deadline()
            result['timed_out'] = True
        result['development_deadline_monotonic'] = self.deadline
        return result


def main() -> None:
    parser = argparse.ArgumentParser(description='Nexus offline workspace MCP')
    parser.add_argument('--lease-file', type=Path, required=True)
    args = parser.parse_args()
    workspace = DeadlineWorkspace.from_lease(args.lease_file)
    workspace.start_watchdog()
    try:
        workspace.module.serve(workspace)
    finally:
        workspace.stop_watchdog()


if __name__ == '__main__':
    main()
