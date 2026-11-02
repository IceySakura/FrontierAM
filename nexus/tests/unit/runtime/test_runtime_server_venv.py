import asyncio

from datetime import datetime, timezone
from pathlib import Path
from types import SimpleNamespace

import pytest

from nexus.runtime.server.venv import manager as venv_manager_module
from nexus.runtime.server.venv.manager import VenvManager
from nexus.runtime.server.venv.models import VenvSpec, VenvStatus


def test_base_site_packages_prefers_actual_site_packages(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    venv_root = tmp_path / '.venv'
    site_packages = venv_root / 'Lib' / 'site-packages'
    site_packages.mkdir(parents=True)
    monkeypatch.setattr('site.getsitepackages', lambda: [str(venv_root), str(site_packages)])
    monkeypatch.setattr(venv_manager_module, '_BASE_SITE_PACKAGES', None)

    assert venv_manager_module._get_base_site_packages() == site_packages


def test_find_top_level_entries_merges_record_with_top_level_txt(tmp_path: Path) -> None:
    site_packages = tmp_path / 'site-packages'
    dist_info = site_packages / 'example-1.0.dist-info'
    package_dir = site_packages / 'example'
    binary_dir = site_packages / 'example_system32'
    dist_info.mkdir(parents=True)
    package_dir.mkdir()
    binary_dir.mkdir()
    (dist_info / 'top_level.txt').write_text('example\nmissing_import_name\n', encoding='utf-8')
    (dist_info / 'RECORD').write_text(
        'example/__init__.py,,\nexample_system32/example.dll,,\nexample-1.0.dist-info/METADATA,,\n',
        encoding='utf-8',
    )

    entries = venv_manager_module._find_top_level_entries(site_packages, dist_info)

    assert set(entries) == {package_dir, binary_dir}


def _runtime_app(monkeypatch: pytest.MonkeyPatch, tmp_path: Path):  # noqa: ANN202
    pytest.importorskip('fastapi')
    from nexus.core import executor_registry
    from nexus.utils import log as nexus_log

    monkeypatch.setattr(nexus_log, 'setup_logging', lambda **_kwargs: None)
    monkeypatch.setattr(executor_registry, 'get_all', lambda: {})
    monkeypatch.setattr(venv_manager_module, 'RUNTIME_DIR', tmp_path / 'runtime')
    from nexus.runtime.server import app as runtime_app

    return runtime_app


def _installing_manager(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> VenvManager:
    manager = VenvManager(root=tmp_path / 'venvs')

    async def create(
        venv_id: str,
        spec: VenvSpec,
        *,
        wheelhouse: Path | None = None,
        defer_ready: bool = False,
    ) -> VenvStatus:
        assert wheelhouse is None
        assert defer_ready is True
        status = VenvStatus(
            venv_id=venv_id,
            state='creating',
            python_version=spec.python_version,
            artifacts=[artifact.name for artifact in spec.artifacts],
            extra_requirements=spec.extra_requirements,
            created_at=datetime.now(timezone.utc),
        )
        manager._venvs[venv_id] = status
        return status

    monkeypatch.setattr(manager, 'create', create)
    return manager


async def test_venv_manager_can_defer_ready_until_worker_connects(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    manager = VenvManager(root=tmp_path / 'venvs')

    async def create_bare_venv(_python: str, _venv_dir: Path) -> None:
        return None

    monkeypatch.setattr(manager, '_create_bare_venv', create_bare_venv)
    monkeypatch.setattr(
        manager,
        '_symlink_from_base',
        lambda *_args: venv_manager_module._SymlinkResult(artifact_missing=set(), module_missing=[]),
    )

    status = await manager.create('venv-1', VenvSpec(modules=[]), defer_ready=True)

    assert status.state == 'creating'
    assert manager.get('venv-1') == status


async def test_create_venv_stays_creating_until_worker_connects(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    runtime_app = _runtime_app(monkeypatch, tmp_path)
    manager = _installing_manager(tmp_path, monkeypatch)
    worker_started = asyncio.Event()
    launch_args = {}

    async def start_worker(venv_id, venv_dir, *, base_url, env_vars):  # noqa: ANN001, ANN202
        launch_args.update(
            venv_id=venv_id,
            venv_dir=venv_dir,
            base_url=base_url,
            env_vars=env_vars,
        )
        worker_started.set()

    monkeypatch.setattr(runtime_app, '_venv_manager', manager)
    monkeypatch.setattr(runtime_app, '_worker_launcher', SimpleNamespace(start_worker=start_worker))
    monkeypatch.setattr(runtime_app, '_worker_ready_events', {})
    monkeypatch.setattr(runtime_app, '_worker_sids', {})
    monkeypatch.setattr(runtime_app, '_worker_capabilities', {})
    monkeypatch.setattr(runtime_app, '_server_base_url', 'http://runtime.test:8000')
    spec = VenvSpec(modules=[], env_vars={'WORKER_ENV': 'value'})

    creation = asyncio.create_task(runtime_app._create_venv_full('venv-1', spec))
    await asyncio.wait_for(worker_started.wait(), timeout=1)
    await asyncio.sleep(0)

    assert creation.done() is False
    assert manager.get('venv-1').state == 'creating'
    assert runtime_app._worker_sids == {}

    await runtime_app.worker_connect(
        'worker-sid',
        {},
        {'worker_id': 'worker-venv-1', 'venv_id': 'venv-1'},
    )
    status = await asyncio.wait_for(creation, timeout=1)

    assert status.state == 'ready'
    assert manager.get('venv-1') == status
    assert runtime_app._worker_sids == {'worker-venv-1': 'worker-sid'}
    assert runtime_app._worker_ready_events == {}
    assert launch_args == {
        'venv_id': 'venv-1',
        'venv_dir': tmp_path / 'venvs' / 'venv-1',
        'base_url': 'http://runtime.test:8000',
        'env_vars': {'WORKER_ENV': 'value'},
    }


async def test_create_venv_persists_worker_launch_failure(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    runtime_app = _runtime_app(monkeypatch, tmp_path)
    manager = _installing_manager(tmp_path, monkeypatch)

    async def start_worker(*_args, **_kwargs) -> None:
        raise OSError('worker launch failed')

    monkeypatch.setattr(runtime_app, '_venv_manager', manager)
    monkeypatch.setattr(runtime_app, '_worker_launcher', SimpleNamespace(start_worker=start_worker))
    monkeypatch.setattr(runtime_app, '_worker_ready_events', {})
    monkeypatch.setattr(runtime_app, '_worker_sids', {})

    status = await runtime_app._create_venv_full('venv-1', VenvSpec(modules=[]))
    queried = await runtime_app.get_venv('venv-1')

    assert status.state == 'failed'
    assert status.error == 'worker launch failed'
    assert queried == status
    assert runtime_app._worker_ready_events == {}
