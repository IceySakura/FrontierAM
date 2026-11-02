"""Conic task lifecycle using the same strict offline development boundary."""

from __future__ import annotations

import contextlib
import uuid

from pathlib import Path
from typing import cast

from nexus.extensions.tasks.frontieram_offline.assets import LockedAssets
from nexus.extensions.tasks.frontieram_offline.runtime import (
    FrontierAMOfflineProvider,
    OfflineDockerRuntime,
    _blocking,
    _write_lease,
)
from nexus.runtime.models import InvokeRequest
from nexus.runtime.providers.registry import runtime_provider_registry

from .assets import ConicAssets


class ConicRuntime(OfflineDockerRuntime):
    def __init__(self, *, assets: ConicAssets, **kwargs):
        self.conic_assets = assets
        # The unchanged lifecycle consumes only the verified shared asset API.
        super().__init__(assets=cast(LockedAssets, assets), **kwargs)

    async def initialize_lease(self):
        await super().initialize_lease()
        self._lease.update(
            version='frontieram-conic-runtime-v1',
            profile_path=str(self.conic_assets.profile_path),
            profile_sha256=self.conic_assets.profile_sha256,
        )
        _write_lease(self._lease_path, self._lease)

    def mcp_config(self, **kwargs):
        config = super().mcp_config(**kwargs)
        config['args'][1] = 'nexus.extensions.tasks.frontieram_conic.mcp'
        return config

    async def _invoke(self, request: InvokeRequest):
        raise ValueError('Conic candidate commands must use the leased workspace MCP tool')


@runtime_provider_registry.register('frontieram_conic_offline')
class ConicOfflineProvider(FrontierAMOfflineProvider):
    def __init__(self, source_root, profile_path, profile_sha256, docker_host, docker_path=None, **config):
        super().__init__(source_root, docker_host, docker_path, **config)
        self.conic_assets = ConicAssets(source_root, profile_path, profile_sha256)
        self.assets = cast(LockedAssets, self.conic_assets)

    async def acquire(self, name, **params):
        if name in self._active_runtimes or name in self._acquiring_names:
            raise ValueError('Offline workspaces cannot be reused across attempts')
        merged = self.get_params(name, **params)
        if set(merged) - {'output_dir', 'stage_public', 'run_id'}:
            raise ValueError('Unknown conic workspace settings')
        output_dir = Path(merged['output_dir']).expanduser().resolve()
        for protected in (
            self.conic_assets.source_root,
            self.conic_assets.task_root,
            self.conic_assets.public_tar.parent,
        ):
            if output_dir == protected or protected in output_dir.parents:
                raise ValueError('Runtime evidence must stay outside the frozen source tree')
        output_dir.mkdir(parents=True, exist_ok=True)
        container = 'nexus-frontieram-' + uuid.uuid4().hex
        runtime = None
        self._acquiring_names.add(name)
        try:
            checks = await _blocking(self.docker.create, self.assets.image_id, container)
            runtime = ConicRuntime(
                assets=self.conic_assets,
                docker=self.docker,
                container=container,
                output_dir=output_dir,
                configuration_checks=checks,
            )
            await runtime.initialize_lease()
            if merged.get('stage_public', True):
                await runtime.stage_public()
            self._active_runtimes[name] = runtime
            return runtime
        except BaseException:
            if runtime is not None:
                await runtime.close()
            else:
                with contextlib.suppress(Exception):
                    await _blocking(self.docker.remove, container)
            raise
        finally:
            self._acquiring_names.discard(name)
