"""The private guest overlay must work without host source or host-only dependencies."""

import hashlib
import json
import subprocess
import sys
import tarfile

import pytest

from nexus.runtime.bridge.artifacts import build_app_bundle
from nexus.runtime.providers.macosworld_devcloud.bridge_bundle import build_macosworld_bundle


@pytest.mark.parametrize('protocol', ['anthropic', 'responses', 'responses-chat'])
def test_private_bundle_is_standalone_and_leaves_shared_artifact_untouched(tmp_path, protocol):
    shared = tmp_path / 'shared.tar.gz'
    private = tmp_path / 'private.tar.gz'
    build_app_bundle(output=shared, target='macos-x86_64-py311', default_port=8765, include_vendor=False)
    original_hash = hashlib.sha256(shared.read_bytes()).hexdigest()
    build_macosworld_bundle(shared, private)
    assert hashlib.sha256(shared.read_bytes()).hexdigest() == original_hash
    with tarfile.open(shared) as archive:
        assert not any('macosworld' in name for name in archive.getnames())
        launcher = archive.extractfile('nexus-llm-bridge/bin/launch_bridge.sh').read()
        assert b'-m nexus.runtime.bridge ' in launcher
    with tarfile.open(private) as archive:
        launcher = archive.extractfile('nexus-llm-bridge/bin/launch_bridge.sh').read()
        assert b'-m macosworld_bridge ' in launcher
        manifest = json.load(archive.extractfile('nexus-llm-bridge/manifest.json'))
        assert manifest['entrypoint'] == 'python -m macosworld_bridge'
        archive.extractall(tmp_path / 'unpacked', filter='data')
    source = tmp_path / 'unpacked/nexus-llm-bridge/src'
    result = subprocess.run(
        [
            sys.executable,
            '-I',
            '-c',
            f'import sys; sys.path.insert(0, {str(source)!r}); '
            'from macosworld_bridge import create_bridge_asgi_app; '
            f'create_bridge_asgi_app(api_protocol={protocol!r}); '
            'assert "nexus.runtime.providers.macosworld_devcloud" not in sys.modules',
        ],
        cwd=tmp_path,
        capture_output=True,
        text=True,
        timeout=30,
    )
    assert result.returncode == 0, result.stderr


def test_shared_app_rejects_mac_protocol_unless_explicitly_wrapped():
    from nexus.runtime.bridge.app import create_bridge_fastapi_app

    with pytest.raises(ValueError, match='Unsupported bridge API protocol'):
        create_bridge_fastapi_app(api_protocol='responses-chat')
