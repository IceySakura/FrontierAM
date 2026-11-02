from pathlib import Path

from nexus.runtime.providers import docker as docker_provider


def test_read_package_version_from_pyproject_supports_bundle_metadata(tmp_path: Path) -> None:
    pyproject = tmp_path / 'pyproject.toml'
    pyproject.write_text('[project]\nname = "nexus-agent"\nversion = "9.8.7"\n')

    assert docker_provider._read_package_version_from_pyproject(pyproject) == '9.8.7'


def test_copy_compiled_package_artifacts_into_runtime_build_context(tmp_path: Path) -> None:
    source = tmp_path / 'dist'
    source.mkdir()
    binary = source / 'nexus.cpython-310-x86_64-linux-gnu.so'
    stub = source / 'nexus.pyi'
    binary.write_bytes(b'binary-nexus')
    stub.write_text('from typing import Any\n')

    dest = tmp_path / 'context'
    dest.mkdir()

    docker_provider._copy_compiled_package_artifacts(source, dest)

    assert (dest / binary.name).read_bytes() == b'binary-nexus'
    assert (dest / 'nexus.pyi').read_text() == 'from typing import Any\n'


def test_copy_package_tree_preserves_executable_mode_for_filesystem_resources(tmp_path: Path) -> None:
    source = tmp_path / 'bundle'
    script = source / 'deployment' / 'runtime' / 'start.sh'
    script.parent.mkdir(parents=True)
    script.write_text('#!/bin/sh\n')
    script.chmod(0o755)

    dest = tmp_path / 'context'
    docker_provider._copy_package_tree(source, dest)

    copied = dest / 'deployment' / 'runtime' / 'start.sh'
    assert copied.read_text() == '#!/bin/sh\n'
    assert copied.stat().st_mode & 0o777 == 0o755
