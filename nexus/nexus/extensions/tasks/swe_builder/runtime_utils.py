"""Runtime utility functions for SWE Builder.

These functions are serialized and executed inside the runtime container.
"""

from __future__ import annotations

import contextlib
import subprocess

from typing import Any


def _git_cmd_env() -> tuple[str, dict]:
    """Return (git_binary, env) that forces OpenSSL libcurl, bypassing nix git.

    The nexus-runtime server runs with PATH=/nix/runtime/nix-env/bin:... so the
    nix git is found first.  That nix git links against nix-bundled GnuTLS which
    fails TLS handshakes in CN container networks.  We bypass it by calling
    /usr/bin/git (system git) directly and preloading libcurl.so.4 (OpenSSL),
    which coexists with libcurl-gnutls.so.4 in Ubuntu/Debian images.

    start.sh also exports LD_LIBRARY_PATH=/nix/runtime/nix-env/lib which contains
    nix-bundled libcurl (GnuTLS).  We strip nix paths from LD_LIBRARY_PATH so the
    LD_PRELOAD (OpenSSL libcurl) is not overridden by the nix one — same filtering
    as run_command/run_script.
    """
    import os

    env = os.environ.copy()
    nix_lib_prefix = '/nix/'
    raw_ldpath = env.get('LD_LIBRARY_PATH', '')
    if raw_ldpath:
        filtered = ':'.join(p for p in raw_ldpath.split(':') if p and not p.startswith(nix_lib_prefix))
        if filtered:
            env['LD_LIBRARY_PATH'] = filtered
        else:
            env.pop('LD_LIBRARY_PATH', None)
    git_bin = '/usr/bin/git' if os.path.exists('/usr/bin/git') else 'git'
    libcurl = '/usr/lib/x86_64-linux-gnu/libcurl.so.4'
    if os.path.exists(libcurl):
        env['LD_PRELOAD'] = libcurl
    return git_bin, env


def git_clone(repo_url: str, dest: str) -> dict[str, Any]:
    """Clone git repository, or skip if valid repo already exists.

    Args:
        repo_url: Repository URL (https://github.com/owner/repo)
        dest: Destination directory

    Returns:
        Dict with exit_code, stdout, stderr
    """
    import os
    import shutil

    # Check if directory exists and is a valid git repository
    git_dir = os.path.join(dest, '.git')
    if os.path.exists(git_dir):
        # Valid git repository already exists, skip clone
        print(f'[git_clone] Valid git repository already exists at {dest}, skipping clone', flush=True)
        return {'exit_code': 0, 'stdout': f'Repository already exists at {dest}', 'stderr': ''}

    # Check if directory exists but is not a valid git repo
    if os.path.exists(dest):
        try:
            shutil.rmtree(dest)
            print(f'[git_clone] Removed invalid directory: {dest}', flush=True)
        except Exception as e:
            print(f'[git_clone] Failed to remove {dest}: {e}', flush=True)
            return {'exit_code': 1, 'stdout': '', 'stderr': f'Failed to remove invalid directory: {e}'}

    # Ensure dest directory exists before clone so the runtime server's CWD (which may
    # be a path that does not exist in the user container) never affects subprocess calls.
    import os as _os

    _os.makedirs(dest, exist_ok=True)

    git_bin, env = _git_cmd_env()
    # Use blobless clone (--filter=blob:none) to dramatically reduce clone time for large
    # repos (e.g. google-cloud-cpp, velox, LLVM). Only commit/tree objects are fetched
    # upfront; file blobs are fetched lazily on checkout. Compatible with git reset --hard.
    result = subprocess.run(
        [git_bin, 'clone', '--filter=blob:none', repo_url, dest],
        capture_output=True,
        text=True,
        env=env,
        cwd=dest,
    )
    return {'exit_code': result.returncode, 'stdout': result.stdout, 'stderr': result.stderr}


def git_reset_and_clean(testbed_dir: str, commit: str) -> dict[str, Any]:
    """Reset git repository to specific commit and clean working directory.

    Args:
        testbed_dir: Testbed directory path
        commit: Git commit hash

    Returns:
        Dict with exit_code, stdout, stderr
    """
    import os

    os.chdir(testbed_dir)
    git_bin, env = _git_cmd_env()

    # Deinit all submodules before reset.
    # git reset --hard / git clean -fd do NOT remove initialized submodule working
    # trees — if install.sh ran `git submodule update --init`, the submodule stays
    # "dirty" (shows as Staged in git status) and blocks golden_patch application.
    # deinit -f --all removes all submodule working trees and .git/config entries
    # so the parent repo is fully clean.  Errors are ignored (no submodules = ok).
    subprocess.run(
        [git_bin, 'submodule', 'deinit', '-f', '--all'],
        capture_output=True,
        text=True,
        env=env,
    )

    # Reset to commit
    result1 = subprocess.run(
        [git_bin, 'reset', '--hard', commit],
        capture_output=True,
        text=True,
        env=env,
    )

    if result1.returncode != 0:
        # Blobless clones (--filter=blob:none) may not include commits that were on deleted
        # branches — the server only sends reachable objects. Try fetching the specific commit
        # directly (GitHub supports fetching by hash) before giving up.
        if 'Could not parse object' in result1.stderr or 'unknown revision' in result1.stderr:
            print(
                f'[git_reset] Commit {commit[:12]} not found locally, trying git fetch origin {commit[:12]}...',
                flush=True,
            )
            fetch_result = subprocess.run(
                [git_bin, 'fetch', 'origin', commit],
                capture_output=True,
                text=True,
                env=env,
                timeout=120,
            )
            if fetch_result.returncode == 0:
                result1 = subprocess.run(
                    [git_bin, 'reset', '--hard', commit],
                    capture_output=True,
                    text=True,
                    env=env,
                )
        if result1.returncode != 0:
            return {'exit_code': result1.returncode, 'stdout': result1.stdout, 'stderr': result1.stderr}

    # Clean working directory
    result2 = subprocess.run(
        [git_bin, 'clean', '-fd'],
        capture_output=True,
        text=True,
        env=env,
    )

    return {
        'exit_code': result2.returncode,
        'stdout': result1.stdout + '\n' + result2.stdout,
        'stderr': result1.stderr + '\n' + result2.stderr,
    }


def git_apply_patch(patch_path: str, testbed_dir: str) -> dict[str, Any]:
    """Apply git patch.

    Args:
        patch_path: Path to patch file
        testbed_dir: Testbed directory path

    Returns:
        Dict with exit_code, stdout, stderr
    """
    import os

    os.chdir(testbed_dir)
    git_bin, env = _git_cmd_env()

    result = subprocess.run(
        [git_bin, 'apply', '-v', patch_path],
        capture_output=True,
        text=True,
        env=env,
    )

    return {'exit_code': result.returncode, 'stdout': result.stdout, 'stderr': result.stderr}


def git_apply_patch_from_content(content: str, testbed_dir: str) -> dict[str, Any]:
    """Apply a git patch from string content with fallback chain.

    Fallback order:
      1. git apply --verbose (strict)
      2. git apply --verbose --ignore-whitespace
      3. git reset --hard + patch --fuzz=5

    Args:
        content: Patch content as string
        testbed_dir: Testbed directory path

    Returns:
        Dict with exit_code, stdout, stderr
    """
    import os
    import re as _re
    import shutil
    import tempfile

    if not content or not content.strip():
        return {'exit_code': 0, 'stdout': '(empty patch, skipped)', 'stderr': ''}

    content = _re.sub(r'(rename (?:from|to)) tmp/tmp[^/\s]+/(?:old|new)/', r'\1 ', content)
    content = _re.sub(r'(diff --git a/)tmp/tmp[^/\s]+/old/', r'\1', content)
    content = _re.sub(r'( b/)tmp/tmp[^/\s]+/new/', r'\1', content)

    fd, tmp_path = tempfile.mkstemp(suffix='.diff', prefix='_swe_patch_')
    try:
        with os.fdopen(fd, 'w') as f:
            f.write(content)

        result = subprocess.run(
            ['git', 'apply', '--verbose', tmp_path],
            capture_output=True,
            text=True,
            cwd=testbed_dir,
        )
        if result.returncode == 0:
            return {'exit_code': 0, 'stdout': result.stdout, 'stderr': result.stderr}
        first_stderr = result.stderr

        result2 = subprocess.run(
            ['git', 'apply', '--verbose', '--ignore-whitespace', tmp_path],
            capture_output=True,
            text=True,
            cwd=testbed_dir,
        )
        if result2.returncode == 0:
            return {
                'exit_code': 0,
                'stdout': result2.stdout,
                'stderr': '[applied with --ignore-whitespace fallback]\n' + result2.stderr,
            }

        subprocess.run(['git', 'reset', '--hard', 'HEAD'], cwd=testbed_dir, capture_output=True)
        subprocess.run(['git', 'clean', '-fdx'], cwd=testbed_dir, capture_output=True)
        if not shutil.which('patch'):
            if shutil.which('apk'):
                subprocess.run(['apk', 'add', '--no-cache', 'patch'], capture_output=True)
            elif shutil.which('apt-get'):
                subprocess.run(['apt-get', 'install', '-y', '--no-install-recommends', 'patch'], capture_output=True)
        try:
            result3 = subprocess.run(
                ['patch', '--batch', '--fuzz=5', '-p1', '-i', tmp_path],
                capture_output=True,
                text=True,
                cwd=testbed_dir,
            )
            if result3.returncode == 0:
                return {
                    'exit_code': 0,
                    'stdout': result3.stdout,
                    'stderr': '[applied with patch --fuzz=5 fallback]\n' + result3.stderr,
                }
        except FileNotFoundError:
            pass

        return {'exit_code': result.returncode, 'stdout': result.stdout, 'stderr': first_stderr}
    finally:
        with contextlib.suppress(OSError):
            os.unlink(tmp_path)


def git_remove_remote(testbed_dir: str) -> dict[str, Any]:
    """Remove git remote 'origin'.

    Args:
        testbed_dir: Testbed directory path

    Returns:
        Dict with exit_code, stdout, stderr
    """
    import os

    os.chdir(testbed_dir)
    git_bin, env = _git_cmd_env()

    result = subprocess.run(
        [git_bin, 'remote', 'remove', 'origin'],
        capture_output=True,
        text=True,
        env=env,
    )

    return {'exit_code': result.returncode, 'stdout': result.stdout, 'stderr': result.stderr}


def git_status(testbed_dir: str) -> dict[str, Any]:
    """Get git status of working directory.

    Args:
        testbed_dir: Testbed directory path

    Returns:
        Dict with untracked_files, modified_files, staged_files, is_clean
    """
    import os

    os.chdir(testbed_dir)
    git_bin, env = _git_cmd_env()

    # Get status in porcelain format
    result = subprocess.run(
        [git_bin, 'status', '--porcelain'],
        capture_output=True,
        text=True,
        env=env,
    )

    if result.returncode != 0:
        return {'error': result.stderr, 'is_clean': False}

    untracked_files = []
    modified_files = []
    staged_files = []

    for line in result.stdout.strip().split('\n'):
        if not line:
            continue

        status = line[:2]
        file_path = line[3:]

        if status == '??':
            untracked_files.append(file_path)
        elif status[0] in ['M', 'A', 'D', 'R', 'C']:
            staged_files.append(file_path)
        elif status[1] in ['M', 'D']:
            modified_files.append(file_path)

    return {
        'untracked_files': untracked_files,
        'modified_files': modified_files,
        'staged_files': staged_files,
        'is_clean': not (untracked_files or modified_files or staged_files),
    }


def read_file(path: str) -> str:
    """Read file content.

    Args:
        path: File path

    Returns:
        File content as string, or empty string if file doesn't exist
    """
    import os

    if not os.path.exists(path):
        return ''

    try:
        with open(path, encoding='utf-8') as f:
            return f.read()
    except FileNotFoundError:
        return ''


def write_file(path: str, content: str) -> dict[str, Any]:
    """Write content to file.

    Args:
        path: File path
        content: Content to write

    Returns:
        Dict with success status
    """
    import os

    os.makedirs(os.path.dirname(path), exist_ok=True)

    with open(path, 'w', encoding='utf-8') as f:
        f.write(content)

    return {'success': True, 'path': path}


def remove_files(paths: list[str]) -> dict[str, Any]:
    """Remove files.

    Args:
        paths: List of file paths to remove

    Returns:
        Dict with removed files list
    """
    import os

    removed = []
    for path in paths:
        if os.path.exists(path):
            os.remove(path)
            removed.append(path)

    return {'removed': removed}


def run_command(command: str, cwd: str | None = None, timeout: int | None = None) -> dict[str, Any]:
    """Run shell command string inside the runtime container.

    Similar to run_script but takes a command string instead of a script file.
    Useful for running arbitrary test commands (e.g. test_spec) with a timeout.

    Args:
        command: Shell command to run (passed to bash -c)
        cwd: Working directory (optional)
        timeout: Timeout in seconds (optional). If None, no timeout is applied.

    Returns:
        Dict with exit_code, stdout, stderr
    """
    import glob as _glob
    import os

    env = os.environ.copy()

    nix_lib_prefix = '/nix/'
    raw_ldpath = env.get('LD_LIBRARY_PATH', '')
    if raw_ldpath:
        filtered = ':'.join(p for p in raw_ldpath.split(':') if p and not p.startswith(nix_lib_prefix))
        if filtered:
            env['LD_LIBRARY_PATH'] = filtered
        else:
            env.pop('LD_LIBRARY_PATH', None)

    current_path_set = set(env.get('PATH', '').split(':'))
    extra_paths = [
        '/root/.cargo/bin',
        '/usr/local/go/bin',
        '/root/go/bin',
        '/root/.local/bin',
        '/usr/local/bin',
    ]
    nvm_bins = sorted(_glob.glob('/root/.nvm/versions/node/*/bin'), reverse=True)
    extra_paths.extend(nvm_bins)
    if os.path.isdir('/root/.sdkman/candidates/java/current/bin'):
        extra_paths.append('/root/.sdkman/candidates/java/current/bin')
    if os.path.isdir('/root/.rbenv/shims'):
        extra_paths.extend(['/root/.rbenv/shims', '/root/.rbenv/bin'])

    new_paths = [p for p in extra_paths if p and p not in current_path_set]
    if new_paths:
        env['PATH'] = ':'.join(new_paths) + ':' + env.get('PATH', '')

    env.setdefault('CARGO_HOME', '/root/.cargo')
    env.setdefault('RUSTUP_HOME', '/root/.rustup')
    env.setdefault('GOPATH', '/root/go')
    env.setdefault('HOME', '/root')
    env.setdefault('USER', 'root')
    env.setdefault('LOGNAME', 'root')
    env.setdefault('SHELL', '/bin/bash')
    env.setdefault('TMPDIR', '/tmp')
    env.setdefault('TERM', 'xterm-256color')
    env.setdefault('LANG', 'en_US.UTF-8')

    try:
        result = subprocess.run(
            ['bash', '-c', command],
            capture_output=True,
            text=True,
            cwd=cwd,
            timeout=timeout,
            env=env,
            start_new_session=True,  # setsid — allows killing entire process group on timeout
        )
        return {
            'exit_code': result.returncode,
            'stdout': result.stdout,
            'stderr': result.stderr,
        }
    except subprocess.TimeoutExpired as e:
        # Kill entire process group (not just direct child) to clean up orphan subprocesses
        # e.g. pip install spawning gcc, go mod download spawning git, etc.
        import os
        import signal

        try:
            pid = getattr(e, 'pid', None)
            if pid is not None and pid:
                os.killpg(int(pid), signal.SIGKILL)
        except (ProcessLookupError, PermissionError):
            pass
        # Preserve partial stdout/stderr captured before timeout (aids diagnosis)
        partial_stdout = (e.stdout or '') if hasattr(e, 'stdout') else ''
        partial_stderr = (e.stderr or '') if hasattr(e, 'stderr') else ''
        if isinstance(partial_stdout, bytes):
            partial_stdout = partial_stdout.decode('utf-8', errors='replace')
        if isinstance(partial_stderr, bytes):
            partial_stderr = partial_stderr.decode('utf-8', errors='replace')
        return {
            'exit_code': -1,
            'stdout': partial_stdout,
            'stderr': f'Command timed out after {timeout}s\n{partial_stderr}',
        }


def run_script(script_path: str, cwd: str | None = None, timeout: int | None = None) -> dict[str, Any]:
    """Run shell script.

    Args:
        script_path: Path to script file
        cwd: Working directory (optional)
        timeout: Timeout in seconds (optional). If None, no timeout is applied.

    Returns:
        Dict with exit_code, stdout, stderr
    """
    import os

    # Start from the container's own environment (inherited from the image's Dockerfile ENV).
    # The FastAPI server process is started at container init, so os.environ reflects the
    # image's ENV PATH — e.g. /root/.cargo/bin for rust-git, /usr/local/go/bin for golang-local.
    env = os.environ.copy()

    # Remove Nix paths from both LD_LIBRARY_PATH and PATH.
    #
    # The FastAPI server process runs inside Nix, so its environment has:
    #   LD_LIBRARY_PATH = /nix/runtime/nix-env/lib:...  (Nix-bundled glibc/libstdc++)
    #   PATH            = /nix/runtime/nix-env/bin:...  (Nix git, curl, python3, ...)
    #
    # Problems if we leave these:
    #   LD_LIBRARY_PATH: Nix libgcc_s/libstdc++ require GLIBC_2.34+/2.38+; causes
    #                    "GLIBC_X.Y not found" errors in node/java/python on Ubuntu 20/22.
    #   PATH:            Nix python3/git/curl come first; eval.sh may accidentally use
    #                    Nix python3 (different version) instead of the system one.
    #                    The offline evaluator runs `docker run fresh_image bash eval.sh`
    #                    where PATH has NO Nix entries — only the image's Dockerfile ENV.
    #                    Leaving Nix in PATH here causes in-situ/offline divergence.
    #
    # Stripping both /nix/ prefixes ensures run_script faithfully reproduces the
    # `docker run fresh_image` environment and in-situ validation matches offline.
    nix_prefix = '/nix/'
    raw_ldpath = env.get('LD_LIBRARY_PATH', '')
    if raw_ldpath:
        filtered = ':'.join(p for p in raw_ldpath.split(':') if p and not p.startswith(nix_prefix))
        if filtered:
            env['LD_LIBRARY_PATH'] = filtered
        else:
            env.pop('LD_LIBRARY_PATH', None)
    raw_path = env.get('PATH', '')
    if raw_path:
        filtered_path = ':'.join(p for p in raw_path.split(':') if p and not p.startswith(nix_prefix))
        if filtered_path:
            env['PATH'] = filtered_path
        else:
            env.pop('PATH', None)

    # Do NOT enrich PATH here.
    #
    # This function is used for fail2pass validation (submit_eval).  The goal is to
    # reproduce the same environment that the offline evaluator sees when it runs
    # eval.sh inside a FRESH docker container started from the committed image:
    #
    #   docker run --rm <committed-image> bash -c "bash /tmp/eval.sh"
    #
    # In a fresh container, PATH comes only from the image's Dockerfile ENV.
    # Tools installed by the agent at runtime (NVM, rbenv, rustup, Go tarball, sdkman)
    # exist on the committed filesystem but are NOT in the default PATH unless eval.sh
    # explicitly exports them.
    #
    # If we add those paths here, in-situ validation passes for scripts that forget to
    # source/export them — producing a false positive that will fail in offline evaluation.
    #
    # eval.sh MUST be fully self-contained: it must source or export every tool PATH it
    # needs (nvm, rbenv, cargo, GOPATH/bin, JAVA_HOME, etc.).  See Section 6 in base.md.
    #
    # os.environ already contains the image's Dockerfile ENV paths (cargo/go for rust-git
    # and golang-local images) because the FastAPI server process inherits the container's
    # environment at startup.  No manual enrichment is needed for image-level tools.

    # Language-specific env defaults (safe to set — eval.sh can override if needed)
    env.setdefault('CARGO_HOME', '/root/.cargo')
    env.setdefault('RUSTUP_HOME', '/root/.rustup')
    env.setdefault('GOPATH', '/root/go')

    # Shell/system defaults: eval.sh may use set -u which treats unset vars as errors.
    # NVM in particular accesses $TMPDIR and $TERM without always using ${VAR:-} defaults.
    env.setdefault('HOME', '/root')
    env.setdefault('USER', 'root')
    env.setdefault('LOGNAME', 'root')
    env.setdefault('SHELL', '/bin/bash')
    env.setdefault('TMPDIR', '/tmp')
    env.setdefault('TERM', 'xterm-256color')
    env.setdefault('LANG', 'en_US.UTF-8')
    try:
        result = subprocess.run(
            ['bash', script_path],
            capture_output=True,
            text=True,
            cwd=cwd,
            timeout=timeout,
            env=env,
        )
        return {
            'exit_code': result.returncode,
            'stdout': result.stdout,
            'stderr': result.stderr,
        }
    except subprocess.TimeoutExpired:
        return {
            'exit_code': -1,
            'stdout': '',
            'stderr': f'Script timed out after {timeout}s',
        }
