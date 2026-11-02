"""Add the macOSWorld guest entry to a private copy of the shared bridge artifact."""

import io
import json
import tarfile

from pathlib import Path


def build_macosworld_bundle(source: Path, output: Path) -> None:
    launcher_name = 'nexus-llm-bridge/bin/launch_bridge.sh'
    found_launcher = False
    with tarfile.open(source, 'r:gz') as original, tarfile.open(output, 'w:gz') as target:
        for member in original:
            content = original.extractfile(member) if member.isfile() else None
            if member.name == launcher_name:
                assert content is not None
                data = content.read()
                old = b'-m nexus.runtime.bridge '
                if data.count(old) != 1:
                    raise ValueError('Unexpected shared bridge launcher; refusing to patch its entry point')
                data = data.replace(old, b'-m macosworld_bridge ')
                content = io.BytesIO(data)
                member.size = len(data)
                found_launcher = True
            if member.name == 'nexus-llm-bridge/manifest.json':
                assert content is not None
                manifest = json.loads(content.read())
                manifest['entrypoint'] = 'python -m macosworld_bridge'
                data = json.dumps(manifest, indent=2).encode()
                content = io.BytesIO(data)
                member.size = len(data)
            target.addfile(member, content)
        if not found_launcher:
            raise ValueError('Shared bridge artifact is missing its POSIX launcher')
        for path in sorted(Path(__file__).with_name('bridge_guest').glob('*.py')):
            target.add(path, arcname=f'nexus-llm-bridge/src/macosworld_bridge/{path.name}')
