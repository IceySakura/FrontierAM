"""Dataset handler for SWE Builder task.

This module provides functions for initializing runtime environments and handling dataset operations.
"""

import asyncio
import logging

from typing import TYPE_CHECKING, Any

from nexus.runtime import Runtime

from .constants import DEFAULT_TESTBED_DIR, DEFAULT_WORKSPACE_DIR, GOLDEN_PATCH_PATH, TEST_PATCH_PATH


logger = logging.getLogger(__name__)

# Use TYPE_CHECKING to avoid circular import
if TYPE_CHECKING:
    pass
else:
    # Will be imported at runtime in functions that need it
    pass

logger.debug('[swe_builder/dataset_handler] ✅ All imports completed')


async def setup_runtime_init(
    instance: dict[str, Any],
    runtime: Runtime,
    task_input: Any,  # SWEBuilderTaskInput, imported at runtime to avoid circular import
    **kwargs: Any,
) -> None:
    """Initialize setup runtime environment.

    This function:
    1. Clones the repository to testbed_dir (inside container)
    2. Checks out to base_commit
    3. Ensures workspace directory exists (unified with testbed_dir)
    4. Injects patch files for switch_tool
    5. Sets up environment variables

    Args:
        instance: Instance data dictionary
        runtime: Runtime instance
        task_input: Task input configuration
        **kwargs: Additional parameters
    """
    repo = instance.get('repo')  # e.g., "owner/repo"
    testbed_dir = task_input.testbed_dir or DEFAULT_TESTBED_DIR
    workspace_dir = task_input.workspace_dir or DEFAULT_WORKSPACE_DIR
    base_commit = instance.get('base_commit')

    if not repo:
        raise ValueError('repo is required in instance data')
    if not base_commit:
        raise ValueError('base_commit is required in instance data')

    logger.debug(f'Initializing runtime for repo: {repo}, base_commit: {base_commit}')
    logger.debug(f'Repository will be cloned to: {testbed_dir} (inside container)')
    logger.debug(f'Workspace directory: {workspace_dir} (unified with testbed)')

    # 0. Check if git is available (ubuntu-git:22.04 image has git pre-installed)
    # If using a different base image without git, this will fail and show clear error
    logger.debug('🔧 Verifying git is available...')

    check_git_cmd = 'command -v git'
    check_result = await runtime.run_command(check_git_cmd, timeout=10)

    if check_result.return_code != 0:
        # Git not found - this shouldn't happen with ubuntu-git:22.04 image
        logger.error('❌ Git is not available in the container')
        logger.error('   Make sure you are using ubuntu-git:22.04 base image')
        logger.error('   Run: cd /path/to/nexus/deployment/docker && bash build-ubuntu-git.sh')
        raise RuntimeError(
            'Git command not found. Please use ubuntu-git:22.04 as base image, '
            'or build it with: cd deployment/docker && bash build-ubuntu-git.sh'
        )

    logger.debug('✅ Git is available')

    # 1. Clone repository inside container to testbed_dir
    # Reference: swe-builder-jiacx uses "git clone -o origin" and checks success
    # Note: This clones directly into the container, not from host filesystem
    logger.debug(f'📦 Starting git clone: https://github.com/{repo}.git -> {testbed_dir}')
    logger.debug('   This may take a while for large repositories (timeout: 600s)...')

    # Check if directory already exists and is a valid git repository
    # Test both: directory exists AND it's a valid git repo
    check_git_cmd = f'test -d {testbed_dir}/.git'
    check_result = await runtime.run_command(check_git_cmd, timeout=10)

    if check_result.return_code == 0:
        logger.debug(f'📦 Valid git repository already exists at {testbed_dir}, skipping clone')
    else:
        # Directory doesn't exist or is not a valid git repo - need to clone
        # First, remove the directory if it exists but is not a valid git repo
        check_dir_cmd = f'test -d {testbed_dir}'
        check_dir_result = await runtime.run_command(check_dir_cmd, timeout=10)
        if check_dir_result.return_code == 0:
            logger.warning(f'⚠️  Directory {testbed_dir} exists but is not a valid git repository, removing...')
            rm_cmd = f'rm -rf {testbed_dir}'
            rm_result = await runtime.run_command(rm_cmd, timeout=60)
            if rm_result.return_code != 0:
                logger.error(f'❌ Failed to remove invalid directory: {rm_result.stderr}')
                raise RuntimeError(f'Failed to remove invalid directory {testbed_dir}: {rm_result.stderr}')
            logger.debug(f'✅ Removed invalid directory {testbed_dir}')

        # Clone with retry: transient network errors (especially on GitHub) are common.
        # Strategy: up to 3 attempts with exponential back-off (30 s, 60 s, 120 s).
        clone_cmd = f'git clone -o origin https://github.com/{repo}.git {testbed_dir}'
        clone_timeout = 1800  # 30 minutes per attempt (large repos like Cataclysm-DDA can exceed 10 min)
        max_clone_attempts = 3
        result = None
        for attempt in range(1, max_clone_attempts + 1):
            logger.debug(f'📦 Git clone attempt {attempt}/{max_clone_attempts}...')
            result = await runtime.run_command(clone_cmd, timeout=clone_timeout)
            logger.info(f'📦 Git clone attempt {attempt} completed (return_code={result.return_code})')
            if result.return_code == 0:
                break
            logger.warning(f'⚠️  Git clone attempt {attempt} failed')
            logger.warning(f'   stderr: {result.stderr[:300] if result.stderr else "N/A"}')
            if attempt < max_clone_attempts:
                wait_seconds = 30 * (2 ** (attempt - 1))  # 30 s, 60 s
                logger.debug(f'   Retrying in {wait_seconds}s...')
                import asyncio as _asyncio

                await _asyncio.sleep(wait_seconds)
                # Remove partial clone directory before retrying
                cleanup_cmd = f'rm -rf {testbed_dir}'
                await runtime.run_command(cleanup_cmd, timeout=60)

        assert result is not None, f'Git clone loop did not execute (max_clone_attempts={max_clone_attempts})'
        if result.return_code != 0:
            logger.error(f'❌ Git clone failed after {max_clone_attempts} attempts')
            logger.error(f'   stdout: {result.stdout[:500] if result.stdout else "N/A"}')
            logger.error(f'   stderr: {result.stderr[:500] if result.stderr else "N/A"}')
            raise RuntimeError(
                f'Failed to clone repository {repo} to {testbed_dir} after {max_clone_attempts} attempts: {result.stderr}'
            )

        logger.debug(f'✅ Repository cloned successfully to {testbed_dir}')

    # 2. Set permissions (reference: swe-builder-jiacx uses chmod -R 777)
    # This ensures non-root users can run tests if needed
    # Use 300s timeout: large TypeScript monorepos (Expensify, amplication) have many files
    chmod_cmd = f'chmod -R 777 {testbed_dir}'
    result = await _run_command_with_retry(runtime, chmod_cmd, timeout=300, label='chmod')
    if result.return_code != 0:
        logger.warning(f'Failed to set permissions on {testbed_dir}: {result.stderr}')
    else:
        logger.debug(f'✅ Set permissions on {testbed_dir}')

    # 3. Reset to base_commit (reference: swe-builder-jiacx uses "git reset --hard")
    # This is more reliable than "git checkout + git clean"
    # If commit not found (e.g., not in default branch), fetch and retry
    reset_cmd = f"""
    cd {testbed_dir} && \
    git reset --hard {base_commit}
    """
    result = await _run_command_with_retry(runtime, reset_cmd, timeout=300, label='git reset')
    if result.return_code != 0:
        # Commit not found - try different strategies to fetch it
        logger.warning(f'⚠️  Commit {base_commit} not found in local repo, attempting to fetch...')

        # Strategy 1: Fetch all branches and tags (most comprehensive)
        logger.debug('   Strategy 1: Fetching all branches and tags...')
        fetch_all_cmd = f'cd {testbed_dir} && git fetch --all --tags --force'
        fetch_result = await runtime.run_command(fetch_all_cmd, timeout=300)
        if fetch_result.return_code != 0:
            logger.warning(f'   ⚠️  fetch --all failed: {fetch_result.stderr}')
        else:
            logger.debug('   ✅ Fetched all branches and tags')
            # Retry reset
            result = await runtime.run_command(reset_cmd, timeout=300)
            if result.return_code == 0:
                logger.debug(f'   ✅ Successfully reset to {base_commit} after fetch --all')
            else:
                logger.warning('   ⚠️  Reset still failed after fetch --all')

        # Strategy 2: Try to fetch the specific commit directly (if Strategy 1 failed)
        if result.return_code != 0:
            logger.debug(f'   Strategy 2: Fetching commit {base_commit} directly from origin...')
            fetch_commit_cmd = f'cd {testbed_dir} && git fetch origin {base_commit}'
            fetch_commit_result = await runtime.run_command(fetch_commit_cmd, timeout=300)
            if fetch_commit_result.return_code != 0:
                logger.warning(f'   ⚠️  fetch commit failed: {fetch_commit_result.stderr}')
            else:
                logger.debug(f'   ✅ Fetched commit {base_commit} directly')
                # Retry reset
                result = await runtime.run_command(reset_cmd, timeout=300)
                if result.return_code == 0:
                    logger.debug(f'   ✅ Successfully reset to {base_commit} after fetch commit')
                else:
                    logger.warning('   ⚠️  Reset still failed after fetch commit')

        # Strategy 3: Try with unshallow (if repo was shallow cloned)
        if result.return_code != 0:
            logger.debug('   Strategy 3: Attempting unshallow (in case repo was shallow cloned)...')
            unshallow_cmd = f'cd {testbed_dir} && git fetch --unshallow'
            unshallow_result = await runtime.run_command(unshallow_cmd, timeout=300)
            if unshallow_result.return_code != 0:
                logger.warning(f'   ⚠️  unshallow failed: {unshallow_result.stderr}')
            else:
                logger.debug('   ✅ Unshallowed repository')
                # Retry reset
                result = await runtime.run_command(reset_cmd, timeout=300)
                if result.return_code == 0:
                    logger.debug(f'   ✅ Successfully reset to {base_commit} after unshallow')

        # Final check
        if result.return_code != 0:
            logger.error(f'❌ All strategies failed to reset to base_commit {base_commit}')
            logger.error(f'   Original error: {result.stderr}')
            logger.error('   This commit might not exist in the repository or might be from a force-pushed branch')
            logger.error(f'   Please verify the commit exists: git log --all --oneline | grep {base_commit[:8]}')
            raise RuntimeError(
                f'Failed to reset to base_commit {base_commit} after trying all strategies.\n'
                f'Original error: {result.stderr}\n'
                f'This commit might not exist in the repository. Please verify the instance data.'
            )

    logger.debug(f'✅ Reset to base_commit: {base_commit}')

    # 4. Remove remote origin (reference: swe-builder-jiacx removes origin)
    # This prevents agent from seeing newer commits
    remove_remote_cmd = f"""
    cd {testbed_dir} && \
    git remote remove origin
    """
    result = await _run_command_with_retry(runtime, remove_remote_cmd, timeout=60, label='git remote remove')
    if result.return_code != 0:
        # Remote might not exist, just log warning
        logger.warning(f'Failed to remove remote origin (may not exist): {result.stderr}')
    else:
        logger.debug('✅ Removed remote origin')

    # 3. Ensure workspace directory exists (usually same as testbed_dir now)
    # If workspace_dir != testbed_dir, create it separately
    if workspace_dir != testbed_dir:
        mkdir_cmd = f'mkdir -p {workspace_dir}'
        result = await runtime.run_command(mkdir_cmd, timeout=30)
        if result.return_code != 0:
            raise RuntimeError(f'Failed to create workspace directory: {result.stderr}')
    else:
        # workspace_dir == testbed_dir, already exists from clone
        logger.debug(f'Workspace directory unified with testbed: {workspace_dir}')

    # 4. Inject patch files (for switch_tool)
    test_patch = instance.get('test_patch', '')
    patch = instance.get('patch', '')

    if test_patch:
        await write_file_to_runtime(runtime, TEST_PATCH_PATH, test_patch)
        logger.debug(f'Injected test_patch to {TEST_PATCH_PATH}')

    if patch:
        await write_file_to_runtime(runtime, GOLDEN_PATCH_PATH, patch)
        logger.debug(f'Injected golden_patch to {GOLDEN_PATCH_PATH}')

    logger.debug('Runtime initialization completed')


async def write_file_to_runtime(runtime: Runtime, file_path: str, content: str) -> None:
    """Write content to a file in the runtime.

    Uses heredoc to avoid escaping issues.
    Splits content into chunks to avoid ARG_MAX limits.

    Args:
        runtime: Runtime instance
        file_path: Target file path in runtime
        content: File content to write
    """
    from .constants import DOCKER_HEREDOC_DELIMITER

    # Split content into chunks (e.g., 50KB) to avoid Argument list too long
    # ARG_MAX is typically 2MB, so 50KB is very safe
    chunk_size = 50 * 1024

    if not content:
        # Handle empty file
        command = f"cat <<'{DOCKER_HEREDOC_DELIMITER}' > {file_path}\n{DOCKER_HEREDOC_DELIMITER}"
        result = await runtime.run_command(command, timeout=30)
        if result.return_code != 0:
            raise RuntimeError(f'Failed to write empty file {file_path}: {result.stderr}')
        return

    # Process in chunks
    total_length = len(content)
    for i in range(0, total_length, chunk_size):
        chunk = content[i : i + chunk_size]
        # First chunk overwrites, subsequent chunks append
        operator = '>' if i == 0 else '>>'

        # Use heredoc to write chunk
        command = f"cat <<'{DOCKER_HEREDOC_DELIMITER}' {operator} {file_path}\n{chunk}\n{DOCKER_HEREDOC_DELIMITER}"

        result = await runtime.run_command(command, timeout=120)
        if result.return_code != 0:
            raise RuntimeError(f'Failed to write file chunk to {file_path}: {result.stderr}')


async def _run_command_with_retry(runtime: Runtime, cmd: str, timeout: int, label: str) -> Any:
    """Run a runtime command with retry on transient network errors.

    Retries up to 6 times with exponential backoff (2s, 4s, 8s, 16s, 32s) for errors like
    'Can not write request body' and 'ServerDisconnected' that occur under
    high concurrency (50+ tasks hitting the same runtime HTTP server).

    Args:
        runtime: Runtime instance
        cmd: Shell command to run
        timeout: Command timeout in seconds
        label: Human-readable label for log messages (e.g. 'Dockerfile')

    Returns:
        Command result object

    Raises:
        RuntimeError: After all retries are exhausted
    """
    _max_retries = 6
    for attempt in range(1, _max_retries + 1):
        try:
            return await runtime.run_command(cmd, timeout=timeout)
        except Exception as e:
            err_str = str(e).lower()
            _is_transient = any(
                kw in err_str
                for kw in (
                    'can not write request body',
                    'server disconnected',
                    'serverdisconnected',
                    'connection reset',
                    'connection refused',
                    'broken pipe',
                    'clientconnectorerror',
                    'eoferror',
                    'eof occurred',
                    'timeout',
                    'command failed to start',
                    'failed to start',
                    'failed to send command',
                    'expecting value',
                    'jsondecode',
                )
            )
            if _is_transient and attempt < _max_retries:
                _backoff = min(2**attempt, 32)  # 2s, 4s, 8s, 16s, 32s
                logger.warning(
                    f'[extract {label}] Transient error (attempt {attempt}/{_max_retries}), '
                    f'retrying in {_backoff}s: {e}'
                )
                await asyncio.sleep(_backoff)
                continue
            error_msg = f'Runtime command failed when extracting {label}: {e}'
            logger.error(error_msg)
            raise RuntimeError(error_msg) from e
    return None


async def extract_dockerfile_from_runtime(runtime: Runtime, workspace_dir: str) -> str:
    """Extract Dockerfile content from runtime.

    Dockerfile is always saved by the agent to /tmp/Dockerfile (not workspace_dir),
    to avoid interference with git operations on /testbed.

    Args:
        runtime: Runtime instance
        workspace_dir: Workspace directory path (unused, kept for API compatibility)

    Returns:
        Dockerfile content as string, empty string if not found

    Raises:
        RuntimeError: If runtime command execution fails after retries
    """
    from .constants import DOCKERFILE_PATH

    # Agent always saves Dockerfile to /tmp/Dockerfile
    read_cmd = f'test -f {DOCKERFILE_PATH} && cat {DOCKERFILE_PATH} || echo "FILE_NOT_FOUND"'

    result = await _run_command_with_retry(runtime, read_cmd, timeout=30, label='Dockerfile')

    if result.return_code != 0:
        logger.warning(f'Failed to read Dockerfile from {DOCKERFILE_PATH}: {result.stderr}')
        return ''

    if result.stdout.strip() == 'FILE_NOT_FOUND':
        logger.debug(f'Dockerfile not found at {DOCKERFILE_PATH}')
        return ''

    return result.stdout


async def extract_eval_script_from_runtime(runtime: Runtime) -> str:
    """Extract eval.sh script content from runtime container.

    Args:
        runtime: Runtime instance

    Returns:
        eval.sh script content as string, empty string if not found

    Raises:
        RuntimeError: If runtime command execution fails after retries
    """
    from .constants import EVAL_SCRIPT_PATH

    read_cmd = f'test -f {EVAL_SCRIPT_PATH} && cat {EVAL_SCRIPT_PATH} || echo "FILE_NOT_FOUND"'

    result = await _run_command_with_retry(runtime, read_cmd, timeout=30, label='eval.sh')

    if result.return_code != 0:
        logger.debug(f'Failed to read {EVAL_SCRIPT_PATH}: {result.stderr}')
        return ''

    if result.stdout.strip() == 'FILE_NOT_FOUND':
        logger.debug(f'eval.sh not found at {EVAL_SCRIPT_PATH}')
        return ''

    return result.stdout


async def extract_eval_dockerfile_script_from_runtime(runtime: Runtime) -> str:
    """Extract eval_dockerfile.sh script content from runtime container.

    Args:
        runtime: Runtime instance

    Returns:
        eval_dockerfile.sh script content as string, empty string if not found

    Raises:
        RuntimeError: If runtime command execution fails after retries
    """
    from .constants import EVAL_DOCKERFILE_SCRIPT_PATH

    read_cmd = f'test -f {EVAL_DOCKERFILE_SCRIPT_PATH} && cat {EVAL_DOCKERFILE_SCRIPT_PATH} || echo "FILE_NOT_FOUND"'

    result = await _run_command_with_retry(runtime, read_cmd, timeout=30, label='eval_dockerfile.sh')

    if result.return_code != 0:
        logger.debug(f'Failed to read {EVAL_DOCKERFILE_SCRIPT_PATH}: {result.stderr}')
        return ''

    if result.stdout.strip() == 'FILE_NOT_FOUND':
        logger.debug(f'eval_dockerfile.sh not found at {EVAL_DOCKERFILE_SCRIPT_PATH}')
        return ''

    return result.stdout
