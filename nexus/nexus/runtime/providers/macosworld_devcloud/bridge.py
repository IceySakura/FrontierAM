from __future__ import annotations

import asyncio
import tempfile

from pathlib import Path

from nexus.runtime.bridge.adapters.devcloud_macos import DevCloudMacOSBridgeAdapter


class MacOSWorldBridgeAdapter(DevCloudMacOSBridgeAdapter):
    """Add the macOSWorld protocol overlay; reuse DevCloud upload and transport."""

    async def upload_artifact(self, artifact_path: Path) -> None:
        from .bridge_bundle import build_macosworld_bundle

        # Only this adapter uploads the private overlay. The shared cached archive is immutable.
        with tempfile.TemporaryDirectory(prefix='macosworld-bridge-') as directory:
            private_artifact = Path(directory) / 'bridge.tar.gz'
            await asyncio.to_thread(build_macosworld_bundle, artifact_path, private_artifact)
            await super().upload_artifact(private_artifact)
