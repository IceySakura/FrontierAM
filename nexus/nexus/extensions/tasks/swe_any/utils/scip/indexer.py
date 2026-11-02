"""SCIP indexing helpers — provision ``scip-python`` and produce ``index.scip``.

Kept free of any nexus/ExecutionContext coupling so it can run unchanged either
on the host or inside a sandbox worker (invoked from ``runtime_setup``). All
work is done with plain ``subprocess`` — no runtime executors — so the module
is importable and callable anywhere the ``scip-python`` CLI is reachable.

``scip-python`` is a Node CLI (``@sourcegraph/scip-python``). It is expected to
be preinstalled in a baked repo image; when it is missing we can ``npm install``
it on demand (see ``ensure_scip_python``). Only Python is supported — other
languages need their own SCIP indexer (scip-typescript / scip-java / …).
"""

from __future__ import annotations

import shutil
import subprocess

from collections.abc import Sequence
from pathlib import Path


SCIP_PYTHON_PKG = '@sourcegraph/scip-python'
DEFAULT_NPM_REGISTRY = 'http://mirrors.tencent.com/npm/'


def find_scip_python(
    install_dir: str | Path | None = None,
    *,
    extra_bin_dirs: Sequence[str | Path] = (),
) -> str | None:
    """Locate the ``scip-python`` binary.

    Checks, in order: the ``PATH``; each of *extra_bin_dirs* (e.g. the nexus
    runtime's nix bin dir); the local ``node_modules/.bin`` under *install_dir*
    (where ``npm install`` drops it). Returns the resolved path or ``None``.
    """
    on_path = shutil.which('scip-python')
    if on_path:
        return on_path
    candidates = [Path(d) / 'scip-python' for d in extra_bin_dirs]
    if install_dir is not None:
        candidates.append(Path(install_dir) / 'node_modules' / '.bin' / 'scip-python')
    for candidate in candidates:
        if candidate.is_file():
            return str(candidate)
    return None


def ensure_scip_python(
    *,
    install_dir: str | Path,
    npm_bin: str = 'npm',
    npm_registry: str | None = DEFAULT_NPM_REGISTRY,
    timeout: float = 600.0,
    env: dict[str, str] | None = None,
    extra_bin_dirs: Sequence[str | Path] = (),
) -> str:
    """Return a usable ``scip-python`` path, ``npm install``-ing it if missing.

    *install_dir* is the directory under which ``node_modules`` is created.
    *npm_bin* is the npm executable (resolve it from the runtime toolchain — a
    baked user image often has no ``npm`` on PATH). *env* is the subprocess
    environment (set ``PATH`` so ``npm``/``node`` resolve). Installs with
    ``--no-audit --no-fund`` against *npm_registry*.

    Raises ``FileNotFoundError`` if the binary is still absent after install and
    ``RuntimeError`` on a failing ``npm install``.
    """
    found = find_scip_python(install_dir, extra_bin_dirs=extra_bin_dirs)
    if found:
        return found

    install_dir = Path(install_dir)
    install_dir.mkdir(parents=True, exist_ok=True)
    cmd = [npm_bin, 'install', '--no-audit', '--no-fund', SCIP_PYTHON_PKG]
    if npm_registry:
        cmd += ['--registry', npm_registry]
    proc = subprocess.run(cmd, cwd=str(install_dir), capture_output=True, text=True, timeout=timeout, env=env)
    if proc.returncode != 0:
        raise RuntimeError(
            f'npm install {SCIP_PYTHON_PKG} failed (exit {proc.returncode}):\n'
            f'{(proc.stdout or "")}\n{(proc.stderr or "")}'
        )

    binary = find_scip_python(install_dir, extra_bin_dirs=extra_bin_dirs)
    if not binary:
        raise FileNotFoundError(f'npm install finished but scip-python is missing under {install_dir}/node_modules')
    return binary


def run_scip_index(
    repo_dir: str | Path,
    *,
    scip_python: str,
    project_name: str | None = None,
    project_version: str = '0.1.0',
    output: str = 'index.scip',
    environment: str | None = None,
    timeout: float = 3600.0,
    env: dict[str, str] | None = None,
) -> Path:
    """Run ``scip-python index`` inside *repo_dir* and return the index path.

    Mirrors the original SWE-Any CLI invocation. *output* is resolved relative
    to *repo_dir*. *env* is the subprocess environment — set ``PATH`` so the
    ``scip-python`` node shebang resolves ``node``. Raises ``RuntimeError`` on a
    non-zero exit (stdout/stderr captured for diagnosis).
    """
    repo_dir = Path(repo_dir)
    project_name = project_name or repo_dir.name
    cmd = [
        scip_python,
        'index',
        '--project-name',
        project_name,
        '--project-version',
        project_version,
        '--output',
        output,
    ]
    if environment:
        cmd += ['--environment', environment]
    cmd.append('.')
    result = subprocess.run(
        cmd,
        cwd=str(repo_dir),
        check=False,
        timeout=timeout,
        env=env,
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        raise RuntimeError(
            f'scip-python index failed (exit {result.returncode}):\nstdout:\n{result.stdout}\nstderr:\n{result.stderr}'
        )
    return repo_dir / output
