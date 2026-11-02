from __future__ import annotations

import shutil
import subprocess
import sys

from pathlib import Path

import pytest


try:
    import tomllib
except ModuleNotFoundError:
    import tomli as tomllib


PROJECT_ROOT = Path(__file__).parents[3]
REDIS_RESOURCE_PREFIX = 'nexus/gateway/trajectory_store_backends/redis_store/'
RESOURCE_PREFIXES = ('nexus/_bundle/', REDIS_RESOURCE_PREFIX)
EXPECTED_BUNDLE_INCLUSION_MAP = {
    'README.md': 'nexus/_bundle/README.md',
    'deployment/docker/Dockerfile.runtime': 'nexus/_bundle/deployment/docker/Dockerfile.runtime',
    # Directory inclusion: every file below it is staged under the destination prefix.
    'deployment/docker/runtime-bin': 'nexus/_bundle/deployment/docker/runtime-bin',
    REDIS_RESOURCE_PREFIX + 'redis_server_cluster.json': REDIS_RESOURCE_PREFIX + 'redis_server_cluster.json',
    REDIS_RESOURCE_PREFIX + 'redis_server_launcher_prepare.sh': REDIS_RESOURCE_PREFIX
    + 'redis_server_launcher_prepare.sh',
    'deployment/runtime/bash/bc_common.c': 'nexus/_bundle/deployment/runtime/bash/bc_common.c',
    'deployment/runtime/bash/bc_common.h': 'nexus/_bundle/deployment/runtime/bash/bc_common.h',
    'deployment/runtime/bash/cmd_wrap.c': 'nexus/_bundle/deployment/runtime/bash/cmd_wrap.c',
    'deployment/runtime/bash/readline_wrap.c': 'nexus/_bundle/deployment/runtime/bash/readline_wrap.c',
    'deployment/runtime/dind-runc.sh': 'nexus/_bundle/deployment/runtime/dind-runc.sh',
    'deployment/runtime/docker-bootstrap.sh': 'nexus/_bundle/deployment/runtime/docker-bootstrap.sh',
    'deployment/runtime/flake.lock': 'nexus/_bundle/deployment/runtime/flake.lock',
    'deployment/runtime/flake.nix': 'nexus/_bundle/deployment/runtime/flake.nix',
    'deployment/runtime/gui_stack.sh': 'nexus/_bundle/deployment/runtime/gui_stack.sh',
    'deployment/runtime/start.sh': 'nexus/_bundle/deployment/runtime/start.sh',
    'pyproject.toml': 'nexus/_bundle/pyproject.toml',
    'uv.lock': 'nexus/_bundle/uv.lock',
}


def _bundle_inclusion_map() -> dict[str, str]:
    with (PROJECT_ROOT / 'pyproject.toml').open('rb') as file:
        project = tomllib.load(file)
    force_include = project['tool']['hatch']['build']['targets']['wheel']['force-include']
    return {
        source: destination
        for source, destination in force_include.items()
        if destination.startswith(RESOURCE_PREFIXES)
    }


def _expected_staged_files() -> dict[str, Path]:
    """Map every staged bundle file to its project source, expanding directory inclusions."""
    staged: dict[str, Path] = {}
    for source, destination in EXPECTED_BUNDLE_INCLUSION_MAP.items():
        source_path = PROJECT_ROOT / source
        if source_path.is_dir():
            for path in sorted(source_path.rglob('*')):
                if path.is_file():
                    staged[f'{destination}/{path.relative_to(source_path).as_posix()}'] = path
        else:
            staged[destination] = source_path
    return staged


def test_hatch_wheel_bundle_contains_only_runtime_build_inputs() -> None:
    assert _bundle_inclusion_map() == EXPECTED_BUNDLE_INCLUSION_MAP


def test_nuitka_builder_stages_bundle_from_hatch_manifest(tmp_path: Path) -> None:
    staging_dir = tmp_path / 'staging'
    subprocess.run(
        [
            sys.executable,
            str(PROJECT_ROOT / 'deployment/pypi/stage_wheel_bundle.py'),
            '--project-root',
            str(PROJECT_ROOT),
            '--staging-dir',
            str(staging_dir),
        ],
        check=True,
    )

    expected_files = _expected_staged_files()
    actual_files = {path.relative_to(staging_dir).as_posix() for path in staging_dir.rglob('*') if path.is_file()}
    assert actual_files == set(expected_files)
    for destination, source_path in expected_files.items():
        destination_path = staging_dir / destination
        assert destination_path.read_bytes() == source_path.read_bytes()
        assert destination_path.stat().st_mode & 0o777 == source_path.stat().st_mode & 0o777


def test_nuitka_stager_includes_packaged_redis_resources_only(tmp_path: Path) -> None:
    project_root = tmp_path / 'project'
    redis_prefix = REDIS_RESOURCE_PREFIX
    resources = {
        'README.md': 'nexus/_bundle/README.md',
        redis_prefix + 'redis_server_cluster.json': redis_prefix + 'redis_server_cluster.json',
        redis_prefix + 'redis_server_launcher_prepare.sh': redis_prefix + 'redis_server_launcher_prepare.sh',
        'nexus/unrelated.json': 'nexus/unrelated.json',
    }
    for source in resources:
        path = project_root / source
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text('resource: ' + source)
        path.chmod(0o755 if source.endswith('.sh') else 0o644)
    (project_root / 'pyproject.toml').write_text(
        '[tool.hatch.build.targets.wheel.force-include]\n'
        + ''.join(f'"{source}" = "{destination}"\n' for source, destination in resources.items())
    )
    staging_dir = tmp_path / 'staging'
    for prefix in RESOURCE_PREFIXES:
        stale = staging_dir / prefix / 'stale-resource.json'
        stale.parent.mkdir(parents=True, exist_ok=True)
        stale.write_text('old resource')
    subprocess.run(
        [
            sys.executable,
            str(PROJECT_ROOT / 'deployment/pypi/stage_wheel_bundle.py'),
            '--project-root',
            str(project_root),
            '--staging-dir',
            str(staging_dir),
        ],
        check=True,
    )

    expected = {destination for source, destination in resources.items() if source != 'nexus/unrelated.json'}
    actual = {path.relative_to(staging_dir).as_posix() for path in staging_dir.rglob('*') if path.is_file()}
    assert actual == expected
    for source, destination in resources.items():
        if destination not in expected:
            continue
        source_path, destination_path = project_root / source, staging_dir / destination
        assert destination_path.read_bytes() == source_path.read_bytes()
        assert destination_path.stat().st_mode & 0o777 == source_path.stat().st_mode & 0o777


@pytest.mark.parametrize('destination', [REDIS_RESOURCE_PREFIX.rstrip('/'), REDIS_RESOURCE_PREFIX + '../outside.json'])
def test_nuitka_stager_rejects_invalid_redis_destinations(tmp_path: Path, destination: str) -> None:
    (tmp_path / 'config.json').write_text('{}')
    (tmp_path / 'pyproject.toml').write_text(
        f'[tool.hatch.build.targets.wheel.force-include]\n"config.json" = "{destination}"\n'
    )
    result = subprocess.run(
        [
            sys.executable,
            str(PROJECT_ROOT / 'deployment/pypi/stage_wheel_bundle.py'),
            '--project-root',
            str(tmp_path),
            '--staging-dir',
            str(tmp_path / 'staging'),
        ],
        capture_output=True,
        text=True,
    )
    assert result.returncode != 0
    assert 'Invalid wheel bundle destination' in result.stderr


def test_nuitka_builder_uses_shared_bundle_stager() -> None:
    script = (PROJECT_ROOT / 'deployment/pypi/build_nuitka_binary_wheel.sh').read_text()
    assert 'deployment/pypi/stage_wheel_bundle.py --staging-dir "$STAGING_DIR"' in script
    assert 'cp deployment/runtime/' not in script


def test_bundled_lock_exports_runtime_constraints_without_workspace_members(tmp_path: Path) -> None:
    uv = shutil.which('uv')
    if uv is None:
        pytest.skip('uv is required to export runtime build constraints')
    # Installed wheels contain the root manifests, but not the services workspace.
    for name in ('pyproject.toml', 'uv.lock'):
        shutil.copy2(PROJECT_ROOT / name, tmp_path / name)
    output = tmp_path / 'core-constraints.txt'
    subprocess.run(
        [
            uv,
            'export',
            '--frozen',
            '--no-dev',
            '--extra',
            'runtime',
            '--extra',
            'otel',
            '--no-emit-project',
            '--no-hashes',
            '--output-file',
            str(output),
        ],
        cwd=tmp_path,
        capture_output=True,
        text=True,
        check=True,
    )
    with (PROJECT_ROOT / 'uv.lock').open('rb') as file:
        lock = tomllib.load(file)
    openai_version = next(package['version'] for package in lock['package'] if package['name'] == 'openai')
    requirements = [line for line in output.read_text().splitlines() if line and not line.startswith('#')]
    assert f'openai=={openai_version}' in requirements
    assert any(line.startswith('uvicorn==') for line in requirements)
    assert any(line.startswith('opentelemetry-sdk==') for line in requirements)
    assert not any(line.startswith(('nexus-agent', '-e ', './', '../', 'pytest==')) for line in requirements)
