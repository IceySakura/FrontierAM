from __future__ import annotations

import asyncio
import threading
import zipfile

from pathlib import Path

import pytest

from nexus.runtime.client import packaging
from nexus.runtime.client.packaging import build_artifact


@pytest.fixture(autouse=True)
def clear_packaging_caches():
    packaging._result_cache = None
    packaging._inflight = None
    yield
    packaging._result_cache = None
    packaging._inflight = None


def _write_dist_info(site_packages: Path, *, top_level_txt: str | None = None, record: str | None = None) -> None:
    dist_info = site_packages / 'nexus_agent-1.2.3.dist-info'
    dist_info.mkdir()
    (dist_info / 'METADATA').write_text('Name: nexus-agent\nVersion: 1.2.3\n')
    (dist_info / 'WHEEL').write_text('Wheel-Version: 1.0\nTag: py3-none-any\n')
    if top_level_txt is not None:
        (dist_info / 'top_level.txt').write_text(top_level_txt)
    if record is not None:
        (dist_info / 'RECORD').write_text(record)


def _write_nexus_package(tmp_path: Path) -> Path:
    site_packages = tmp_path / 'site-packages'
    package_dir = site_packages / 'nexus'
    package_dir.mkdir(parents=True)
    (package_dir / '__init__.py').write_text('__version__ = "1.2.3"\n')
    return package_dir


def test_build_artifact_reuses_dist_info_by_top_level_txt(tmp_path: Path) -> None:
    package_dir = _write_nexus_package(tmp_path)
    _write_dist_info(package_dir.parent, top_level_txt='nexus\n')

    whl_path = build_artifact(package_dir)

    assert whl_path.name == 'nexus_agent-1.2.3-py3-none-any.whl'
    with zipfile.ZipFile(whl_path) as whl:
        names = set(whl.namelist())
    assert 'nexus/__init__.py' in names
    assert 'nexus_agent-1.2.3.dist-info/METADATA' in names


def test_build_artifact_reuses_dist_info_by_record(tmp_path: Path) -> None:
    package_dir = _write_nexus_package(tmp_path)
    _write_dist_info(package_dir.parent, record='nexus/__init__.py,,\nnexus_agent-1.2.3.dist-info/METADATA,,\n')

    whl_path = build_artifact(package_dir)

    assert whl_path.name == 'nexus_agent-1.2.3-py3-none-any.whl'


def test_build_artifact_preserves_nexus_distribution_name_without_project_metadata(tmp_path: Path) -> None:
    package_dir = _write_nexus_package(tmp_path)
    client_dir = package_dir / 'runtime' / 'client'
    client_dir.mkdir(parents=True)
    (client_dir / 'packaging.py').write_text('')

    whl_path = build_artifact(package_dir)

    assert whl_path.name == 'nexus_agent-0.0.0-py3-none-any.whl'
    with zipfile.ZipFile(whl_path) as whl:
        names = set(whl.namelist())
    assert 'nexus/__init__.py' in names
    assert 'nexus_agent-0.0.0.dist-info/METADATA' in names


def test_content_hash_includes_installed_dist_info(tmp_path: Path) -> None:
    package_dir = _write_nexus_package(tmp_path)
    _write_dist_info(package_dir.parent, top_level_txt='nexus\n')

    first = packaging._content_hash(package_dir)

    (package_dir.parent / 'nexus_agent-1.2.3.dist-info' / 'METADATA').write_text('Name: nexus-agent\nVersion: 1.2.4\n')
    second = packaging._content_hash(package_dir)

    assert first != second


def test_generate_metadata_combines_dependency_and_extra_markers() -> None:
    metadata = packaging._generate_metadata(
        'example',
        '1.0.0',
        optional_dependencies={
            'distributed': [
                "example-dependency>=1.0; python_version < '3.14'",
                'sqlalchemy>=2.0.0',
            ]
        },
    )

    assert 'Requires-Dist: example-dependency>=1.0; (python_version < \'3.14\') and extra == "distributed"' in metadata
    assert 'Requires-Dist: sqlalchemy>=2.0.0; extra == "distributed"' in metadata


async def test_build_artifact_cached_computes_content_hash_off_event_loop(monkeypatch, tmp_path: Path) -> None:
    package_dir = _write_nexus_package(tmp_path)
    main_thread = threading.current_thread()
    hash_threads: list[threading.Thread] = []
    wheel = tmp_path / 'cached.whl'
    wheel.write_bytes(b'wheel')

    def fake_content_hash(source: Path) -> str:
        hash_threads.append(threading.current_thread())
        assert source == package_dir.resolve()
        assert threading.current_thread() is not main_thread
        return 'content-hash'

    def fake_locked_build(source: Path, *, exclude_patterns: list[str] | None = None) -> Path:
        source = source.resolve()
        packaging._content_hash(source)
        assert exclude_patterns is None
        return wheel

    monkeypatch.setattr(packaging, '_content_hash', fake_content_hash)
    monkeypatch.setattr(packaging, '_locked_build', fake_locked_build)

    assert await packaging.build_artifact_cached(package_dir) == wheel
    assert hash_threads
    assert all(thread is not main_thread for thread in hash_threads)


async def test_build_artifact_cached_coalesces_concurrent_builds_without_holding_lock(
    monkeypatch, tmp_path: Path
) -> None:
    package_dir = _write_nexus_package(tmp_path)
    loop = asyncio.get_running_loop()
    build_started = asyncio.Event()
    release_build = threading.Event()
    build_calls = 0
    wheel = tmp_path / 'cached.whl'
    wheel.write_bytes(b'wheel')

    def fake_locked_build(source: Path, *, exclude_patterns: list[str] | None = None) -> Path:
        nonlocal build_calls
        build_calls += 1
        loop.call_soon_threadsafe(build_started.set)
        assert release_build.wait(timeout=5)
        return wheel

    monkeypatch.setattr(packaging, '_locked_build', fake_locked_build)

    first = asyncio.create_task(packaging.build_artifact_cached(package_dir))
    await build_started.wait()
    second = asyncio.create_task(packaging.build_artifact_cached(package_dir))
    await asyncio.sleep(0)
    release_build.set()

    assert await asyncio.gather(first, second) == [wheel, wheel]
    assert build_calls == 1
    assert packaging._inflight is None
