"""Tests for artifact archiving exclusions (``_tar_working_dir``).

Offline: builds a real temp working_dir on disk and tars it in-process. No
runtime, no sandbox, no network.

Why these assertions matter — web-sandbox dispatches on *file count*:

* exactly 1 file  → ``static_single``, that file becomes the homepage whatever
  its name is (``mobius.html`` deploys fine)
* 2 or more files → nginx static site, which **requires** ``index.html`` at the
  project root or every request to ``/`` 403s

So any junk directory that survives archiving can flip a perfectly good
single-file artifact into a multi-file one and fail its deploy, which the
reward path then scores as ``ARTIFACT_UNDEPLOYABLE`` (−0.2). Tool state dirs
must never reach the tar.
"""

from __future__ import annotations

import base64
import io
import os
import tarfile

from pathlib import Path

import pytest

from nexus.extensions.tasks.artifact.judges.capabilities import deploy


def _write(root: Path, rel: str, body: str = 'x') -> None:
    path = root / rel
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(body, encoding='utf-8')


def _tar(root: Path, **kwargs):
    """Run the real archiver and return ``(file_list, names_inside_tar)``."""
    archive_b64, file_list = deploy._tar_working_dir(str(root), **kwargs)
    with tarfile.open(fileobj=io.BytesIO(base64.b64decode(archive_b64)), mode='r:gz') as tar:
        names = sorted(m.name for m in tar.getmembers() if m.isfile())
    return sorted(file_list), names


def test_playwright_cli_state_dir_is_excluded(tmp_path: Path) -> None:
    """A single-file artifact stays single-file despite @playwright/mcp state.

    ``@playwright/mcp`` creates ``.playwright-cli/`` in its own process cwd,
    which *is* ``working_dir``. Before this exclusion the dir rode along in the
    tar, turning ``['mobius.html']`` into ``['mobius.html', '.playwright-cli']``
    — a 2-entry root that nginx 403s for lack of ``index.html``.
    """
    _write(tmp_path, 'mobius.html', '<html></html>')
    _write(tmp_path, '.playwright-cli/traces/trace.zip')
    _write(tmp_path, '.playwright-cli/settings.json', '{}')

    file_list, names = _tar(tmp_path)

    assert file_list == ['mobius.html']
    assert names == ['mobius.html']
    # The count feeding artifact_file_count / empty-cause attribution must be 1,
    # otherwise the sandbox picks nginx over static_single.
    assert len(file_list) == 1


def test_excluded_dirs_cover_known_tool_state(tmp_path: Path) -> None:
    """Junk from deps, build output and tool state is all dropped together."""
    _write(tmp_path, 'index.html', '<html></html>')
    for junk in ('node_modules/react/index.js', 'dist/bundle.js', '.git/HEAD',
                 '.cache/x', 'playwright-report/index.html', '.playwright-cli/state'):
        _write(tmp_path, junk)

    file_list, names = _tar(tmp_path)

    assert file_list == ['index.html']
    assert names == ['index.html']


def test_real_artifact_files_survive_archiving(tmp_path: Path) -> None:
    """The exclusion list must not eat genuine model output.

    Guards against a too-broad rule: dotted *files*, nested source dirs and
    assets are all real deliverables even though some sit under dot-names.
    """
    wanted = ['index.html', 'app.js', 'style.css', 'src/main.ts',
              'assets/logo.svg', '.env.example', 'public/favicon.ico']
    for rel in wanted:
        _write(tmp_path, rel)
    _write(tmp_path, '.playwright-cli/state')

    file_list, _ = _tar(tmp_path)

    assert file_list == sorted(wanted)


def test_gitignore_file_is_excluded_but_other_dotfiles_are_not(tmp_path: Path) -> None:
    """``_EXCLUDED_FILES`` is exact-match, not a dotfile wildcard."""
    _write(tmp_path, 'index.html')
    _write(tmp_path, '.gitignore')
    _write(tmp_path, '.npmrc')

    file_list, _ = _tar(tmp_path)

    assert file_list == ['.npmrc', 'index.html']


def test_extra_exclude_paths_and_extensions_still_apply(tmp_path: Path) -> None:
    """Per-row excludes compose with the hardcoded list (OR, not override)."""
    _write(tmp_path, 'index.html')
    _write(tmp_path, 'task_inputs/spec.md')
    _write(tmp_path, 'shot.PNG')
    _write(tmp_path, '.playwright-cli/state')

    file_list, _ = _tar(
        tmp_path,
        extra_exclude_paths=['task_inputs'],
        extra_exclude_extensions=['png'],
    )

    assert file_list == ['index.html']


@pytest.mark.parametrize('junk_dir', sorted(deploy._EXCLUDED_DIRS))
def test_every_excluded_dir_is_dropped_at_any_depth(tmp_path: Path, junk_dir: str) -> None:
    """Exclusion is by directory *name*, so it applies to nested copies too."""
    _write(tmp_path, 'index.html')
    _write(tmp_path, f'{junk_dir}/top-level-junk')
    _write(tmp_path, f'src/{junk_dir}/nested-junk')

    file_list, _ = _tar(tmp_path)

    assert file_list == ['index.html'], f'{junk_dir} leaked into the archive'


def test_archive_roundtrip_preserves_relative_paths(tmp_path: Path) -> None:
    """file_list matches the tar member names, both relative to working_dir.

    Downstream code (``archive_to_code_content_dict`` → sandbox ``value`` dict)
    keys off these paths, so an absolute or ``./``-prefixed name would land the
    file outside the project root on the sandbox side.
    """
    _write(tmp_path, 'index.html', '<html>hi</html>')
    _write(tmp_path, 'src/deep/nested/mod.js', 'export {}')

    file_list, names = _tar(tmp_path)

    assert file_list == names == ['index.html', 'src/deep/nested/mod.js']
    assert not any(n.startswith(('/', './')) for n in names)
    assert not any(os.path.isabs(n) for n in names)
