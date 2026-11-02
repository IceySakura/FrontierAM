from __future__ import annotations

import subprocess
import tarfile
import zipfile

from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[5]
CONFINE_SOURCE_ROOT = 'nexus/extensions/tasks/aidata/engine/confine'
CONFINE_WHEEL_ROOT = 'nexus/extensions/tasks/aidata/mock_servers/confine'
CONFINE_FILES = ('compile.sh', 'libconfine.c')


def test_sdist_builds_wheel_with_confine_kit_at_runtime_path(tmp_path: Path) -> None:
    subprocess.run(
        ['uv', 'build', '--out-dir', str(tmp_path)],
        cwd=REPO_ROOT,
        check=True,
        capture_output=True,
        text=True,
    )

    sdist = next(tmp_path.glob('*.tar.gz'))
    with tarfile.open(sdist, 'r:gz') as archive:
        sdist_names = {name.partition('/')[2] for name in archive.getnames()}

    wheel = next(tmp_path.glob('*.whl'))
    with zipfile.ZipFile(wheel) as archive:
        wheel_names = set(archive.namelist())

    for filename in CONFINE_FILES:
        assert f'{CONFINE_SOURCE_ROOT}/{filename}' in sdist_names
        assert f'{CONFINE_WHEEL_ROOT}/{filename}' in wheel_names
