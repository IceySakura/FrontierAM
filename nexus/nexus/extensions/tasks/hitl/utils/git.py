"""Runtime-side git helpers for HITL tasks."""

import logging
import shlex


logger = logging.getLogger(__name__)

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
    '.agent_history/',
    '.codex-runtime/',
    'logs/',
]

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
}

GITIGNORE_START_MARKER = '# === NEXUS AUTO-GENERATED START ==='
GITIGNORE_END_MARKER = '# === NEXUS AUTO-GENERATED END ==='

_GIT_ENV = {
    'GIT_AUTHOR_NAME': 'hitl-auto',
    'GIT_AUTHOR_EMAIL': 'hitl-auto@noreply',
    'GIT_COMMITTER_NAME': 'hitl-auto',
    'GIT_COMMITTER_EMAIL': 'hitl-auto@noreply',
}


async def git_status(working_dir: str) -> str:
    """Return porcelain git status for a working directory."""
    stdout, _ = await _run_git(working_dir, 'git status --porcelain', timeout=30)
    return stdout


async def git_has_changes(working_dir: str) -> bool:
    """Return whether the working directory has staged, unstaged, or untracked changes."""
    return bool(await git_status(working_dir))


async def git_diff(working_dir: str, base_ref: str = '', *, staged: bool = False) -> str:
    """Return a git diff, including uncommitted worktree changes by default."""
    if staged:
        command = 'git diff --cached'
    elif base_ref:
        command = f'git diff {shlex.quote(base_ref)}'
    else:
        command = 'git diff'
    stdout, _ = await _run_git(working_dir, command, timeout=120)
    return stdout


async def git_changed_files(working_dir: str, base_ref: str = '') -> list[str]:
    """Return changed file paths from git diff/status."""
    command = f'git diff --name-only {shlex.quote(base_ref)}' if base_ref else 'git status --porcelain'
    stdout, _ = await _run_git(working_dir, command, timeout=60)
    if not stdout:
        return []
    if base_ref:
        return [line for line in stdout.splitlines() if line]
    return [_status_path(line) for line in stdout.splitlines() if _status_path(line)]


async def capture_git_patch(working_dir: str) -> dict[str, object]:
    """Capture all current changes as a patch without committing them."""
    status = await git_status(working_dir)
    if not status:
        return {'patch': '', 'has_changes': False, 'files': []}

    await _run_git(working_dir, 'git add -A', timeout=120)
    await _unstage_binary_files(working_dir)
    patch, code = await _run_git(working_dir, "git diff --cached -- ':!.gitignore'", timeout=120)
    if code != 0:
        patch = await git_diff(working_dir, staged=True)
    from nexus.extensions.tasks.hitl.utils.patch import extract_patch_files

    files = extract_patch_files(patch)
    await _run_git(working_dir, 'git reset HEAD', timeout=120)

    logger.info('[capture_git_patch] Captured patch in %s, patch=%d chars', working_dir, len(patch))
    return {'patch': patch, 'has_changes': bool(patch), 'files': files}


async def cumulative_git_patch(working_dir: str, base_ref: str) -> str:
    """Return cumulative diff from base_ref to current worktree."""
    if not base_ref:
        logger.warning('[cumulative_git_patch] No base_ref provided')
        return ''
    return await git_diff(working_dir, base_ref)


async def ensure_generated_gitignore(working_dir: str, language: str = '') -> bool:
    """Add the Nexus-generated gitignore block for runtime/agent artifacts.

    Returns True when the file was changed.
    """

    from pathlib import Path

    repo_dir = Path(working_dir)
    if not repo_dir.exists():
        return False

    lang_key = language.lower()
    rules: list[str] = []
    seen: set[str] = set()
    for rule in list(DEFAULT_GITIGNORE_RULES) + list(LANGUAGE_GITIGNORE_RULES.get(lang_key, [])):
        if rule not in seen:
            seen.add(rule)
            rules.append(rule)

    generated_block = '\n'.join([GITIGNORE_START_MARKER, *rules, GITIGNORE_END_MARKER])
    gitignore_path = repo_dir / '.gitignore'

    content = ''
    try:
        if gitignore_path.exists():
            content = gitignore_path.read_text(encoding='utf-8', errors='replace')
    except Exception:
        content = ''

    if GITIGNORE_START_MARKER in content and GITIGNORE_END_MARKER in content:
        start_idx = content.find(GITIGNORE_START_MARKER)
        end_idx = content.find(GITIGNORE_END_MARKER) + len(GITIGNORE_END_MARKER)
        new_content = content[:start_idx] + generated_block + content[end_idx:]
    else:
        if content and not content.endswith('\n'):
            content += '\n'
        new_content = content + ('\n' if content else '') + generated_block + '\n'

    if new_content == content:
        return False

    gitignore_path.write_text(new_content, encoding='utf-8')
    return True


async def cleanup_test_file_conflicts(working_dir: str, test_patch: str, base_commit: str = '') -> list[str]:
    """Revert worker edits to files that will be controlled by test_patch.

    SWE-style eval scripts often apply the dataset test patch after the model
    patch. If the worker changed the same test files, the test patch can fail to
    apply even when the solution code is reasonable. This mirrors the original
    HITL cleanup: reset those files to the base state before extracting or
    evaluating the final model patch.
    """

    if not test_patch:
        return []

    from nexus.extensions.tasks.hitl.utils.patch import extract_patch_files

    test_files = extract_patch_files(test_patch)
    if not test_files:
        return []

    base_ref = base_commit or 'HEAD~20'
    cleaned: list[str] = []
    for path in test_files:
        quoted_path = shlex.quote(path)
        diff_out, _ = await _run_git(working_dir, f'git diff {shlex.quote(base_ref)} -- {quoted_path}', timeout=120)
        if not diff_out:
            continue

        _, exists_code = await _run_git(
            working_dir,
            f'git cat-file -e {shlex.quote(base_ref)}:{quoted_path} 2>/dev/null',
            timeout=30,
        )
        if exists_code == 0:
            await _run_git(working_dir, f'git checkout {shlex.quote(base_ref)} -- {quoted_path}', timeout=120)
        else:
            await _run_git(working_dir, f'git rm -f {quoted_path} 2>/dev/null || rm -f {quoted_path}', timeout=120)
        cleaned.append(path)

    if cleaned:
        files_list = ', '.join(cleaned)
        await _run_git(working_dir, 'git add -A', timeout=120)
        await _run_git(
            working_dir,
            f'git commit -m {shlex.quote(f"hitl: revert worker changes to test files ({files_list})")} --allow-empty',
            timeout=120,
        )
        logger.info('[cleanup_test_file_conflicts] Cleaned %d test file conflicts: %s', len(cleaned), cleaned)

    return cleaned


async def _unstage_binary_files(working_dir: str) -> None:
    numstat, _ = await _run_git(working_dir, 'git diff --cached --numstat', timeout=60)
    binary_files = []
    for line in numstat.splitlines():
        if line.startswith('-\t-\t'):
            binary_files.append(line.split('\t', 2)[2])
    if not binary_files:
        return
    files_arg = ' '.join(shlex.quote(path) for path in binary_files)
    await _run_git(working_dir, f'git reset HEAD -- {files_arg}', timeout=60)


async def _run_git(working_dir: str, command: str, *, timeout: int) -> tuple[str, int]:
    from nexus.runtime.executors.command import RunCommandRequest, run_command

    response = await run_command(
        RunCommandRequest(
            command=command,
            cwd=working_dir,
            timeout=timeout,
            extra_env=_GIT_ENV,
        )
    )
    stdout = (response.stdout or '').strip()
    code = response.return_code if hasattr(response, 'return_code') else getattr(response, 'exit_code', -1)
    return stdout, code


def _status_path(line: str) -> str:
    if len(line) < 4:
        return ''
    path = line[3:]
    if ' -> ' in path:
        return path.split(' -> ', 1)[1]
    return path
