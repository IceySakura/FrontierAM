"""The sole candidate-facing tool: execute in a leased, networkless container."""

from __future__ import annotations

import argparse
import json
import math
import os
import re
import stat

from pathlib import Path
from typing import cast

from nexus.extensions.tasks.frontieram_offline.assets import LockedAssets
from nexus.extensions.tasks.frontieram_offline.mcp import DeadlineWorkspace as BaseDeadlineWorkspace

from .assets import IMAGE_ID, ConicAssets


LEASE_VERSION = 'frontieram-conic-runtime-v1'


class DeadlineWorkspace(BaseDeadlineWorkspace):
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
        assets = ConicAssets(lease['source_root'], lease['profile_path'], lease['profile_sha256'])
        # The inherited command boundary uses the same verified asset interface.
        result = cls(cast(LockedAssets, assets), lease)
        result.verify()
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
