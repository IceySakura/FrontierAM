"""Inference operations for SWE-bench tasks.

This module contains functions for:
- Pre-inference setup (preparing the runtime environment)
- Post-inference operations (extracting patches from the runtime)
"""

import contextlib
import logging
import shlex

from nexus.runtime import Runtime
from nexus.runtime.executors.models.file import FileItem


logger = logging.getLogger(__name__)


# Default gitignore rules for SWE-bench evaluation
DEFAULT_GITIGNORE_RULES = [
    '*.pyc',
    '__pycache__/',
    '.pytest_cache/',
    '*.egg-info/',
    '.eggs/',
    'build/',
    'dist/',
    '.tox/',
    '.coverage',
    'htmlcov/',
    '.mypy_cache/',
    '.ruff_cache/',
    '*.log',
    '*.orig',
    '*.rej',
    # Agent working directories / runtime state that must never appear in patches.
    # If you add a new scaffold that writes state under /testbed, list it here.
    '.agent_history/',
    '.codex-runtime/',  # codex CLI: config.toml, last_message.txt, skills/, sessions/
    '.workbuddy/',  # wb: the agent writes working memory into the workspace
    'logs/',
    '/.openclaw/',  # openclaw onboard scaffolds its "agent home" into the workspace root
    # Limit scaffold exclusions to dedicated directories; root-level documentation can contain project changes.
]

# Language-specific patch exclusions.
# fmt: off
LANGUAGE_GITIGNORE_RULES: dict[str, list[str]] = {
    'py': [
        '*.pyc',
        '__pycache__/',
        '.pytest_cache/',
        '*.egg-info/',
        '.eggs/',
        'build/',
        'dist/',
        '.tox/',
        '.coverage',
        'htmlcov/',
        '.mypy_cache/',
        '.ruff_cache/',
        'venv/',
        '.venv/',
        # Agent working directories / runtime state that must never appear in patches.
        '.agent_history/',
        '.codex-runtime/',
        'logs/',
    ],
    'js': [
        'node_modules/',
        'dist/',
        'build/',
        '.next/',
        'coverage/',
        '*.log',
    ],
    'java': [
        'target/',
        '*.class',
        '*.jar',
        '.gradle/',
        'build/',
    ],
    # Apply exclusions by language to avoid extending ambiguous directory names to all repositories.
    'c': [
        '*.o', '*.a', '*.so', '*.so.*', '*.lo', '*.la', '*.lai', '*.obj',
        '*.d', '*.Po', '*.Plo', '*.dirstamp',
        '.libs/', '.deps/',
        'CMakeCache.txt', 'CMakeFiles/', 'cmake_install.cmake', 'CTestTestfile.cmake',
        'autom4te.cache/',
    ],
    'go': [
        '*.test', '*.exe',
    ],
    'rust': [
        'target/',
        '*.rlib', '*.rmeta',
    ],
    'c#': [
        'bin/', 'obj/',
        '*.dll', '*.pdb', '*.exe', '*.nupkg',
    ],
    'typescript': [
        'node_modules/',
        'dist/', '.next/', 'coverage/',
        '*.tsbuildinfo', '*.log',
    ],
    'ruby': [
        '.bundle/', 'vendor/bundle/',
        '*.gem', 'coverage/', 'tmp/',
    ],
    'php': [
        '.phpunit.result.cache',
    ],
    'wasm': [
        '*.wasm',
    ],
}
# fmt: on

# Normalize language names to the keys used by the exclusion rules.
LANGUAGE_ALIASES: dict[str, str] = {
    'python': 'py',
    'python3': 'py',
    'javascript': 'js',
    'node': 'js',
    'nodejs': 'js',
    'ts': 'typescript',
    'cpp': 'c',
    'c++': 'c',
    'cxx': 'c',
    'cc': 'c',
    'objective-c': 'c',
    'csharp': 'c#',
    'cs': 'c#',
    'dotnet': 'c#',
    'golang': 'go',
    'rs': 'rust',
    'rb': 'ruby',
}


def _normalize_language(language: object) -> str:
    """Map language names or a language-weight mapping to the corresponding exclusion-rule key."""
    if isinstance(language, dict):
        # Use the language with the greatest weight.
        language = max(language, key=lambda k: language[k]) if language else ''
    key = str(language or '').strip().lower()
    return LANGUAGE_ALIASES.get(key, key)


# Markers for auto-generated gitignore block
GITIGNORE_START_MARKER = '# === NEXUS AUTO-GENERATED START ==='
GITIGNORE_END_MARKER = '# === NEXUS AUTO-GENERATED END ==='


# =============================================================================
# Pre-inference operations
# =============================================================================


async def prepare_runtime_for_inference(
    runtime: Runtime,
    working_dir: str = '/testbed',
) -> None:
    """Prepare the runtime environment before running inference.

    This function can be extended to perform any necessary setup before
    the agent starts working on the problem.

    Args:
        runtime: Runtime instance to prepare.
        working_dir: Working directory in the runtime.
    """
    # Placeholder hook for future shared setup logic.
    del runtime, working_dir


async def run_pre_agent_script(
    runtime: Runtime,
    commands: list[str],
    working_dir: str = '/testbed',
) -> str:
    """Execute pre-agent shell commands inside the runtime."""
    if not commands:
        return ''
    return await runtime.invoke(
        _execute_pre_agent_commands,
        commands,
        working_dir,
    )


async def remove_git_dir(
    runtime: Runtime,
    working_dir: str = '/testbed',
) -> None:
    """Remove git directory from the working directory."""
    await runtime.call_executor('run_command', {'command': f'rm -rf {working_dir}/.git', 'cwd': working_dir})


async def run_git_truncate(
    runtime: Runtime,
    base_commit: str,
    working_dir: str = '/testbed',
) -> str:
    """Truncate git history at base_commit to prevent peeking at future changes.

    This removes all commits after base_commit via git refs cleanup, reflog
    expiration and garbage collection.

    Args:
        runtime: Runtime instance to execute in.
        base_commit: The commit SHA to truncate history at.
        working_dir: Working directory containing the git repository.

    Returns:
        Log output from the truncation script.
    """
    from nexus.extensions.tasks.swe.constants import GIT_TRUNCATE_SCRIPT

    if not base_commit:
        logger.debug('No base_commit provided, skipping git history truncation')
        return ''

    # Build the truncation command with BASE_SHA set
    truncate_cmd = (
        f'cd {working_dir} && '
        f'export FILTER_BRANCH_SQUELCH_WARNING=1 && '
        f'export BASE_SHA={base_commit} && '
        f"bash << 'TRUNCATE_SCRIPT'{GIT_TRUNCATE_SCRIPT}TRUNCATE_SCRIPT"
    )

    logger.debug('Running git history truncation at base_commit=%s', base_commit)
    log_output = await run_init_cmds_in_runtime(runtime, [truncate_cmd], working_dir=working_dir)
    logger.debug('Git truncation output:\n%s', log_output)
    return log_output


async def verify_git_truncation(
    runtime: Runtime,
    base_commit: str,
    working_dir: str = '/testbed',
) -> bool | None:
    """Check whether refs reach commits outside the base commit's ancestry.

    Return True when none are reachable, False when any remain, and None when
    the check cannot determine a result. This checks refs only; it does not
    establish that unreachable objects or reflog entries have been removed.
    """
    if not base_commit:
        return None

    log_output = await run_init_cmds_in_runtime(
        runtime,
        [f'cd {working_dir} && git rev-list --all --not {base_commit} --count'],
        working_dir=working_dir,
    )
    # run_init_cmds_in_runtime wraps the command as `$ cmd\nstdout:\n...\nstderr:\n...`,
    # so the count is not the last line; scan back for the one bare integer.
    for line in reversed(log_output.splitlines()):
        count = line.strip()
        if count.isdigit():
            logger.debug('verify_git_truncation: %s commits outside %s', count, base_commit)
            return int(count) == 0

    logger.warning('verify_git_truncation: could not parse a commit count from output: %s', log_output)
    return None


async def get_head_commit(
    runtime: Runtime,
    working_dir: str = '/testbed',
) -> str:
    """Return HEAD's commit SHA via a dedicated `git rev-parse HEAD`.

    Run as its own command rather than echoing a marker into a larger batch. run_init_cmds_in_runtime wraps each command as `$ cmd\\nstdout:\\n...\\nstderr:\\n...`, so the SHA is not the literal last line; we scan for the bare 40-hex commit hash instead.

    Args:
        runtime: Runtime instance to execute in.
        working_dir: Working directory containing the git repository.

    Returns:
        The HEAD commit SHA, or '' if it could not be parsed.
    """
    log_output = await run_init_cmds_in_runtime(
        runtime, [f'cd {working_dir} && git rev-parse HEAD'], working_dir=working_dir
    )
    for line in reversed(log_output.splitlines()):
        sha = line.strip()
        if len(sha) == 40 and all(c in '0123456789abcdef' for c in sha):
            logger.debug('get_head_commit: HEAD commit %s', sha)
            return sha

    logger.warning('get_head_commit: could not parse HEAD SHA from output: %s', log_output)
    return ''


async def run_git_reinit(
    runtime: Runtime,
    working_dir: str = '/testbed',
) -> str:
    """Drop all git history and reinitialize a single-commit repo, returning the new base commit.

    Runs GIT_REINIT_SCRIPT inside the runtime: it removes every .git (dir or gitlink file) plus .gitmodules, re-inits the repo, and creates one "Initial commit" capturing the current working tree. This prevents an agent from recovering a synthesized fix by walking back history.

    Args:
        runtime: Runtime instance to execute in.
        working_dir: Working directory containing the git repository.

    Returns:
        The new "Initial commit" SHA, or '' if it could not be parsed from the output.
    """
    from nexus.extensions.tasks.swe.constants import GIT_REINIT_SCRIPT

    reinit_cmd = f"cd {working_dir} && bash << 'REINIT_SCRIPT'{GIT_REINIT_SCRIPT}REINIT_SCRIPT"

    logger.debug('Running git reinit in %s', working_dir)
    log_output = await run_init_cmds_in_runtime(runtime, [reinit_cmd], working_dir=working_dir)
    logger.debug('Git reinit output:\n%s', log_output)

    logger.debug('Get new base commit from reinit repo %s', working_dir)
    return await get_head_commit(runtime, working_dir)


# =============================================================================
# Post-inference operations
# =============================================================================


async def get_patch_from_runtime(
    runtime: Runtime,
    working_dir: str = '/testbed',
    language: str = 'py',
    base_commit: str = '',
    extra_ignore_patterns: list[str] | None = None,
) -> str:
    """Get git patch from the runtime by extracting staged changes.

    This function stages all changes in the working directory and generates
    a git diff patch. It also updates .gitignore to exclude common build
    artifacts and cache files.

    Args:
        runtime: Runtime instance to execute the function in.
        working_dir: Working directory in the runtime (default: /testbed).
        language: Programming language for language-specific gitignore rules.
        base_commit: Optional base commit to diff against (default: HEAD).

    Returns:
        The git patch as a string, or empty string if no changes.
    """
    if extra_ignore_patterns:
        return await runtime.invoke(get_git_patch, working_dir, language, base_commit, extra_ignore_patterns)
    return await runtime.invoke(get_git_patch, working_dir, language, base_commit)


def remove_binary_diffs(patch_text: str) -> str:
    """Remove binary file diffs from a git patch.

    Args:
        patch_text: The git patch text.

    Returns:
        The cleaned patch text with binary diffs removed.
    """
    if patch_text.strip() == '':
        return patch_text

    lines = patch_text.splitlines(keepends=True)
    cleaned_lines: list[str] = []
    block: list[str] = []
    is_binary_block = False

    for line in lines:
        if line.startswith('diff --git '):
            if block and not is_binary_block:
                cleaned_lines.extend(block)
            block = [line]
            is_binary_block = False
        elif line.startswith('Binary files '):
            is_binary_block = True
            block.append(line)
        else:
            block.append(line)

    if block and not is_binary_block:
        cleaned_lines.extend(block)

    return ''.join(cleaned_lines)


def get_git_patch(
    working_dir: str = '/testbed',
    language: str = 'py',
    base_commit: str = '',
    extra_ignore_patterns: list[str] | None = None,
) -> str:
    """Extract git patch from the working directory.

    This function is designed to be executed inside the runtime environment.
    It updates .gitignore, stages all changes, and returns the diff.

    Args:
        working_dir: Working directory containing the git repository.
        language: Programming language for language-specific gitignore rules.
        base_commit: Optional base commit to diff against.

    Returns:
        The git patch as a string, or empty string if no changes.
    """
    import subprocess

    from pathlib import Path

    repo_dir = Path(working_dir)
    if not repo_dir.exists():
        return ''

    # Update .gitignore idempotently
    _update_gitignore(repo_dir, language, extra_ignore_patterns)

    # Stage all changes
    with contextlib.suppress(subprocess.SubprocessError):
        subprocess.run(
            ['git', 'add', '-A'],
            cwd=repo_dir,
            capture_output=True,
            timeout=60,
            check=True,
        )

    # Disable external diff drivers when extracting the patch.
    # The --no-ext-diff option belongs to git diff, not the top-level git command.
    diff_ref = base_commit if base_commit else 'HEAD'
    diff_cmd_exclude = ['git', 'diff', '--no-ext-diff', diff_ref, '--', ':!.gitignore']
    diff_cmd_plain = ['git', 'diff', '--no-ext-diff', diff_ref]

    try:
        result = subprocess.run(
            diff_cmd_exclude,
            cwd=repo_dir,
            capture_output=True,
            timeout=120,
        )
        if result.returncode == 0:
            return remove_binary_diffs(result.stdout.decode('utf-8', errors='replace'))
        result = subprocess.run(
            diff_cmd_plain,
            cwd=repo_dir,
            capture_output=True,
            timeout=120,
        )
        # Decode with error handling for non-UTF-8 bytes (e.g., binary files)
        return remove_binary_diffs(result.stdout.decode('utf-8', errors='replace'))
    except subprocess.SubprocessError:
        return ''


def _update_gitignore(repo_dir, language: str, extra_ignore_patterns: list[str] | None = None) -> None:
    """Update .gitignore file idempotently with auto-generated rules.

    Manages a dedicated block in .gitignore marked by special comments.
    This allows updating the rules without duplicating or losing user entries.

    Args:
        repo_dir: Path to the repository root.
        language: Programming language for language-specific rules.
    """
    from pathlib import Path

    # Normalize the language key, combine default and language rules, and deduplicate in order.
    lang_key = _normalize_language(language)
    rules: list[str] = []
    seen: set[str] = set()
    for r in [*DEFAULT_GITIGNORE_RULES, *LANGUAGE_GITIGNORE_RULES.get(lang_key, []), *(extra_ignore_patterns or [])]:
        if r not in seen:
            seen.add(r)
            rules.append(r)

    # Build the auto-generated block
    block_lines = [GITIGNORE_START_MARKER] + rules + [GITIGNORE_END_MARKER]
    generated_block = '\n'.join(block_lines)

    gitignore_path = Path(repo_dir) / '.gitignore'

    # Read existing content
    content = ''
    try:
        if gitignore_path.exists():
            content = gitignore_path.read_text(encoding='utf-8', errors='replace')
    except Exception:
        pass

    # Update or append the auto-generated block
    if GITIGNORE_START_MARKER in content and GITIGNORE_END_MARKER in content:
        # Replace existing block
        start_idx = content.find(GITIGNORE_START_MARKER)
        end_idx = content.find(GITIGNORE_END_MARKER) + len(GITIGNORE_END_MARKER)
        new_content = content[:start_idx] + generated_block + content[end_idx:]
    else:
        # Append new block
        if content and not content.endswith('\n'):
            content += '\n'
        new_content = content + ('\n' if content else '') + generated_block + '\n'

    # Write only if changed
    if new_content != content:
        with contextlib.suppress(Exception):
            gitignore_path.write_text(new_content)


def _execute_pre_agent_commands(
    commands: list[str],
    working_dir: str = '/testbed',
) -> str:
    """Execute shell commands before agent inference and return combined logs."""
    import subprocess

    logs: list[str] = []

    for cmd in commands:
        try:
            proc = subprocess.run(
                cmd,
                shell=True,
                cwd=working_dir,
                capture_output=True,
                text=True,
                timeout=300,
                check=True,
            )
            logs.append(f'$ {cmd}\nstdout:\n{proc.stdout}\nstderr:\n{proc.stderr}')
        except subprocess.CalledProcessError as e:
            logs.append(f'$ {cmd}\nstdout:\n{e.stdout or ""}\nstderr:\n{e.stderr or ""}\nreturncode: {e.returncode}')
            raise
    return '\n'.join(logs)


async def apply_patch_and_commit(
    runtime: Runtime,
    patch_text: str | None,
    patch_file: str,
    working_dir: str,
    message: str | None = None,
) -> str | None:
    """Upload a patch into the sandbox, apply it (git apply with a patch(1) fallback), optionally commit.

    The patch content rides in the upload_file request body and is applied by path, so a large patch never lands in a shell argv. Embedding it inline (e.g. a heredoc) would blow past the kernel's single-argument limit (MAX_ARG_STRLEN, ~128KB) and fail the whole `sh -c ...` with [Errno 7] Argument list too long. No host-side temp file is needed: upload_file carries the content directly, so the file only ever materializes inside the sandbox.

    If message is falsy the patch is only applied (no commit). The caller is responsible for any git identity config the commit needs.

    Returns the command log, or None when patch_text is empty (nothing applied).
    """
    if not patch_text:
        return None

    await runtime.upload_file([FileItem(path=patch_file, content=patch_text, encoding='utf8')])
    apply_cmd = (
        f'git apply --verbose --ignore-whitespace {patch_file} '
        f'|| patch --batch --fuzz=5 -p1 --no-backup-if-mismatch -r /dev/null -i {patch_file}'
    )
    cmds = [apply_cmd, f'rm -f {patch_file}']
    if message:
        cmds.append(
            f'git add -A && git commit --no-gpg-sign --no-verify --allow-empty -m {shlex.quote(message)} || true'
        )
    return await run_init_cmds_in_runtime(runtime, cmds, working_dir=working_dir)


async def run_init_cmds_in_runtime(
    runtime: Runtime,
    init_cmds: list[str],
    working_dir: str = '/testbed',
) -> str:
    """Run initialization commands inside the runtime environment.

    Args:
        runtime: Runtime instance to execute in.
        init_cmds: List of shell commands to execute.
        working_dir: Working directory in the runtime.

    Returns:
        Log output from the commands execution.
    """
    return await runtime.invoke(
        _run_init_cmds,
        init_cmds,
        working_dir,
    )


def _run_init_cmds(
    commands: list[str],
    working_dir: str = '/testbed',
) -> str:
    """Run initialization commands inside the runtime environment.

    This function is executed inside the runtime environment.
    """
    import subprocess

    from pathlib import Path

    from nexus.extensions.tasks.swe.env import strip_dangling_display
    from nexus.utils.shell import get_os_env_without_ld_library_path

    repo_dir = Path(working_dir)
    logs: list[str] = []

    # Strip LD_LIBRARY_PATH inherited from the nix runtime server so that
    # container-native binaries (e.g. node) don't load nix's libstdc++ which
    # may require a newer glibc than the container provides.
    os_envs = get_os_env_without_ld_library_path()
    # Remove DISPLAY only when it points to a missing local X socket.
    os_envs = strip_dangling_display(os_envs)

    for cmd in commands:
        try:
            proc = subprocess.run(
                cmd,
                shell=True,
                cwd=repo_dir,
                capture_output=True,
                text=True,
                timeout=300,
                env=os_envs,
            )
            logs.append(f'$ {cmd}\nstdout:\n{proc.stdout}\nstderr:\n{proc.stderr}')
            if proc.returncode != 0:
                raise RuntimeError(f'Command failed with return code {proc.returncode}: {proc.stderr}')
        except subprocess.TimeoutExpired as e:
            raise RuntimeError(f'Command timed out: {cmd}') from e

    return '\n'.join(logs)


__all__ = [
    'apply_patch_and_commit',
    'run_init_cmds_in_runtime',
    'prepare_runtime_for_inference',
    'run_pre_agent_script',
    'run_git_truncate',
    'run_git_reinit',
    'get_patch_from_runtime',
    'get_git_patch',
]
