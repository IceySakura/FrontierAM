"""Run the frontend media unit tests through pytest."""

import shutil
import subprocess

from pathlib import Path

import pytest


def test_media_javascript():
    node = shutil.which('node')
    if node is None:
        pytest.skip('Node.js is not installed; media JavaScript tests require Node.js')

    result = subprocess.run(
        [node, '--test', str(Path(__file__).resolve().with_suffix('.cjs'))],
        capture_output=True,
        text=True,
        encoding='utf-8',
        timeout=30,
        check=False,
    )
    assert result.returncode == 0, f'Media JavaScript tests failed:\n{result.stdout}\n{result.stderr}'
