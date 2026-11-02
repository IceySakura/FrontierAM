"""Declare the dependencies used by the isolated official worker, not just the host developer venv."""

import subprocess
import sys

from pathlib import Path

import pytest

from packaging.requirements import Requirement
from packaging.utils import canonicalize_name


try:
    import tomllib
except ImportError:
    import tomli as tomllib


ROOT = Path(__file__).resolve().parents[5]
METADATA = ROOT / 'nexus/extensions/tasks/os_world_v2/pyproject.toml'


@pytest.mark.parametrize(
    'package',
    [
        'xmltodict',
        'tldextract',
        'odfpy',
        'opencv-python-headless',
        'scikit-image',
        'imageio-ffmpeg',
        'imagehash',
        'scipy',
        'beautifulsoup4',
        'pypdf',
        'mutagen',
        'pyacoustid',
        'librosa',
        'fastdtw',
        'soundfile',
        'websocket-client',
        'pyparsing',
        'pytz',
        'borb',
        'anthropic',
        'google-genai',
        'scikit-learn',
        'chardet',
        'easyocr',
    ],
)
def test_official_worker_import_dependencies_are_declared(package):
    config = tomllib.loads(METADATA.read_text())
    packages = {
        canonicalize_name(Requirement(value).name) for value in config['project']['optional-dependencies']['host']
    }
    assert canonicalize_name(package) in packages
    assert 'opencv-python' not in packages  # headless avoids host libGL requirements


def test_upstream_constraints_and_local_ffmpeg_source_are_not_lost():
    config = tomllib.loads(METADATA.read_text())
    requirements = {
        canonicalize_name(Requirement(value).name): Requirement(value)
        for value in config['project']['optional-dependencies']['host']
    }
    assert str(requirements['imageio-ffmpeg'].specifier) == '==0.6.0'
    assert '3.0' not in requirements['borb'].specifier
    assert '6.3' in requirements['icalendar'].specifier
    assert '8.0' not in requirements['icalendar'].specifier
    assert 'osworld-ffmpeg' not in requirements
    for unrelated in ('pyautogui', 'docker', 'gymnasium', 'zhipuai', 'groq', 'black', 'pyobjc', 'pywin32'):
        assert unrelated not in requirements


PROBE = '''
import sys

BLOCKED = ('e2b_code_interpreter', 'e2b', 'tencentcloud')


class Blocker:
    def find_spec(self, name, path=None, target=None):
        root = name.split('.')[0]
        if root in BLOCKED:
            raise ModuleNotFoundError(f"No module named {root!r} (blocked)", name=root)
        return None


for module in list(sys.modules):
    if module.split('.')[0] in BLOCKED:
        del sys.modules[module]
sys.meta_path.insert(0, Blocker())

for target in (
    'nexus.extensions.tasks.os_world_v2.task',
    'nexus.extensions.tasks.os_world_v2.adapter.adapter',
    'nexus.extensions.tasks.os_world_v2.adapter.pdf_grader',
):
    __import__(target)
print('imported')
'''


def test_task_schema_load_does_not_require_ags_host_dependencies(tmp_path):
    """Only the AGS extra supplies e2b-code-interpreter and tencentcloud-sdk-python.

    Installing just os_world_v2 (per the runbook) must still register the task; the
    provider modules may only be imported when a grader or runtime is actually built.
    """
    probe = tmp_path / 'probe.py'
    probe.write_text(PROBE)
    result = subprocess.run(
        [sys.executable, '-I', str(probe)],
        capture_output=True,
        text=True,
        timeout=300,
        check=False,
    )
    assert result.returncode == 0, result.stderr
    assert 'imported' in result.stdout
