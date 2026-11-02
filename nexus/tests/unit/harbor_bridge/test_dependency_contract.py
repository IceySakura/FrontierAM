from __future__ import annotations

from pathlib import Path

import tomllib

from packaging.specifiers import SpecifierSet
from packaging.version import Version


def test_root_lock_keeps_harbor_host_realtime_compatible() -> None:
    """The Harbor host extra currently requires realtime's websockets ceiling."""

    repository_root = Path(__file__).resolve().parents[3]
    lock = tomllib.loads((repository_root / 'uv.lock').read_text(encoding='utf-8'))
    websockets = next(package for package in lock['package'] if package['name'] == 'websockets')

    assert SpecifierSet('>=11,<16').contains(Version(websockets['version']))
