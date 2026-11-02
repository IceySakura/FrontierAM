"""Evaluation module for SWE Builder task.

This module provides functions for building Docker images and evaluating test cases.
Note: This is framework post-processing (Nexus Eval stage), not executed in Agent.
"""

import asyncio
import logging
import os
import re
import sys
import tempfile
import time
import traceback

from pathlib import Path
from typing import TYPE_CHECKING, Any


logger = logging.getLogger(__name__)
logger.debug('[swe_builder/eval] 🔵 Module import started')

# Docker build concurrency management
_docker_build_semaphore = None
_docker_build_concurrency = None

# Container operations concurrency management
# Separate semaphore to limit concurrent container create/start/exec/stop operations
_container_ops_semaphore = None
_container_ops_concurrency = None

# _active_builds: number of tasks currently inside build_and_evaluate() try block.
# Since asyncio is single-threaded, plain int is safe without a lock.
_active_builds = 0


def _get_docker_build_concurrency() -> int:
    """Get Docker build concurrency limit from environment or use default.

    Priority:
    1. Environment variable DOCKER_BUILD_CONCURRENCY (if set)
    2. Fixed default value (recommended for production)

    To override, set environment variable before running:
        export DOCKER_BUILD_CONCURRENCY=16
    """
    os.getenv('DOCKER_BUILD_CONCURRENCY')
    # if env_value:
    #     try:
    #         value = int(env_value)
    #         if value > 0:
    #             logger.info(f'[Docker Build] Using DOCKER_BUILD_CONCURRENCY from environment: {value}')
    #             return value
    #     except ValueError:
    #         pass

    # Fixed default: 12 concurrent Docker builds (based on disk I/O test)
    # This value is optimized for machines with SATA SSD performance
    # Disk performance: 361 MB/s write, 374 MB/s read (tested on /data/docker/lib)
    # Adjust this value based on your disk performance:
    #   - HDD or slow SSD (< 200 MB/s): 6-8
    #   - SATA SSD (300-500 MB/s): 10-14  ← Current machine
    #   - Fast SSD (> 500 MB/s): 12-16
    #   - NVMe SSD (> 1000 MB/s): 16-20
    #   - High-performance NVMe (> 2000 MB/s): 20-24
    default_concurrency = 12
    logger.debug(f'[Docker Build] Using fixed default concurrency: {default_concurrency}')
    logger.debug('[Docker Build] To override, set: export DOCKER_BUILD_CONCURRENCY=<number>')
    return default_concurrency


def _get_docker_build_semaphore() -> asyncio.Semaphore:
    """Get or create the global Docker build semaphore."""
    global _docker_build_semaphore, _docker_build_concurrency
    if _docker_build_semaphore is None:
        _docker_build_concurrency = _get_docker_build_concurrency()
        _docker_build_semaphore = asyncio.Semaphore(_docker_build_concurrency)
        logger.debug(f'[Docker Build] Initialized build semaphore with limit: {_docker_build_concurrency}')
        logger.debug('[Docker Build] Override with: DOCKER_BUILD_CONCURRENCY=<number>')
    return _docker_build_semaphore


def _get_container_ops_concurrency() -> int:
    """Get container operations concurrency limit.

    This limits concurrent container create/start/exec operations
    to prevent Docker daemon overload during high concurrency.

    CRITICAL: This must account for the ENTIRE lifecycle of container operations,
    not just creation. Each test phase does 10+ exec_run calls, so we need very
    conservative limits to prevent daemon overload.

    Strategy:
    - Build concurrency * 1.0 (very conservative, each instance runs 4 phases sequentially)
    - For build_concurrency=12: container_ops=12

    Rationale:
    - 12 concurrent tasks × 4 phases/task = 48 potential phases
    - Each phase has 10+ exec_run operations
    - If we set limit to 24, we get 24 × 10 = 240 concurrent exec_run operations → daemon crash
    - Setting limit to 12 means max 12 × 10 = 120 concurrent operations → daemon stable
    - Lower limit = stable Docker daemon, no task hangs, results written to DB

    Can be overridden with environment variable DOCKER_CONTAINER_OPS_CONCURRENCY
    """
    env_value = os.getenv('DOCKER_CONTAINER_OPS_CONCURRENCY')
    if env_value:
        try:
            value = int(env_value)
            if value > 0:
                return value
        except ValueError:
            pass

    # Default: build concurrency * 1.0 (very conservative to prevent daemon overload)
    # This accounts for multiple exec_run calls per phase
    return _get_docker_build_concurrency()


def _get_container_ops_semaphore() -> asyncio.Semaphore:
    """Get or create the global container operations semaphore."""
    global _container_ops_semaphore, _container_ops_concurrency
    if _container_ops_semaphore is None:
        _container_ops_concurrency = _get_container_ops_concurrency()
        _container_ops_semaphore = asyncio.Semaphore(_container_ops_concurrency)
        logger.debug(f'[Container Ops] Initialized semaphore with limit: {_container_ops_concurrency}')
        logger.debug('[Container Ops] Override with: DOCKER_CONTAINER_OPS_CONCURRENCY=<number>')
    return _container_ops_semaphore


logger.debug('[swe_builder/eval] 📦 Importing nexus.core...')
from nexus.core import ExecutionContext


logger.debug('[swe_builder/eval] ✅ nexus.core imported')

logger.debug('[swe_builder/eval] 📦 Importing nexus.runtime...')

logger.debug('[swe_builder/eval] ✅ nexus.runtime imported')

logger.debug('[swe_builder/eval] 📦 Importing local constants...')
from .constants import (
    TEST_CATEGORY_FAIL2FAIL,
    TEST_CATEGORY_FAIL2PASS,
    TEST_CATEGORY_PASS2FAIL,
    TEST_CATEGORY_PASS2PASS,
)


logger.debug('[swe_builder/eval] ✅ constants imported')

logger.debug('[swe_builder/eval] 📦 Importing local dataset_handler...')

logger.debug('[swe_builder/eval] ✅ dataset_handler imported')

# Use TYPE_CHECKING to avoid circular import
if TYPE_CHECKING:
    pass

logger.debug('[swe_builder/eval] ✅ All imports completed')

# Try to import docker library (optional, will auto-install if needed)
try:
    import docker
    import docker.errors

    logger.debug('[swe_builder/eval] ✅ docker library available')
except ImportError:
    logger.warning('[swe_builder/eval] ⚠️  docker library not found, will auto-install when needed')
    docker = None  # type: ignore

# Exit code extraction pattern
# Note: eval_script outputs "OMNIGRIL_EXIT_CODE=0", not "echo OMNIGRIL_EXIT_CODE=0"
EXIT_CODE_RE = re.compile(r'OMNIGRIL_EXIT_CODE=(\d+)')
APPLY_PATCH_FAIL = '>>>>> Patch Apply Failed'
APPLY_PATCH_PASS = '>>>>> Patch Apply Passed'


def _make_full_result(
    instance: dict[str, Any],
    success: bool = False,
    resolved: bool = False,
    test_results: dict[str, list] | None = None,
    report: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Create a full result dictionary with all required fields.

    Definitions:
        success: Golden patch passes (targeted-B exit_code == 0)
        resolved: fail2pass in targeted eval (eval_dockerfile.sh A fails, B passes)
        test_results: from comprehensive eval.sh output (all test categories)

    Args:
        instance: Instance data dictionary
        success: Whether golden patch passes (targeted-B exit_code == 0)
        resolved: Whether fail2pass detected in targeted eval (eval_dockerfile.sh)
        test_results: Test results dictionary (empty structure if None)
        report: Evaluation report dictionary

    Returns:
        Full result dictionary with all required fields
    """
    if test_results is None:
        test_results = {
            TEST_CATEGORY_FAIL2PASS: [],
            TEST_CATEGORY_FAIL2FAIL: [],
            TEST_CATEGORY_PASS2PASS: [],
            TEST_CATEGORY_PASS2FAIL: [],
        }

    if report is None:
        report = {}

    return {
        'success': success,
        'resolved': resolved,
        'test_results': test_results,
        'report': report,
        'instance_id': instance.get('instance_id', ''),
        'repo': instance.get('repo', ''),
        'base_commit': instance.get('base_commit', ''),
        'problem_statement': instance.get('problem_statement', ''),
        'test_patch': instance.get('test_patch', ''),
        'patch': instance.get('patch', ''),
        'language': instance.get('language', ''),
    }


async def build_and_evaluate(
    dockerfile_content: str,
    eval_script_content: str,  # eval.sh: comprehensive test suite for test case enumeration
    instance: dict[str, Any],
    task_input: Any,  # SWEBuilderTaskInput, imported at runtime to avoid circular import
    ctx: ExecutionContext,
    cleanup_image: bool = True,  # Whether to cleanup image after evaluation
    eval_dockerfile_script_content: str = '',  # eval_dockerfile.sh: targeted tests for fail2pass judgment
) -> dict[str, Any]:
    """Build Docker image and execute 4-phase evaluation (framework post-processing).

    4-Phase Evaluation Design:
    - Phase targeted-A: eval_dockerfile.sh + test_patch only → expect FAIL (bug exposed)
    - Phase targeted-B: eval_dockerfile.sh + golden_patch + test_patch → expect PASS (bug fixed)
    - Phase full-A:     eval.sh + test_patch only → for test case enumeration
    - Phase full-B:     eval.sh + golden_patch + test_patch → for test case enumeration

    Result Definitions:
    - success: golden patch passes (targeted-B exit_code == 0)
    - correct: fail2pass in targeted (A fails, B passes) OR (test_patch empty AND B passes)
    - test_results: classify test cases from full eval.sh output (fail2pass/fail2fail/pass2pass/pass2fail)

    Fallback when eval_dockerfile.sh is absent:
    - Use eval.sh as targeted script (degrade to original 2-phase behavior)
    - test_results will come from the same A/B run

    Args:
        dockerfile_content: Dockerfile content generated by agent
        eval_script_content: eval.sh script (comprehensive, for test case enumeration)
        instance: Instance data dictionary
        task_input: Task input configuration
        ctx: Execution context
        cleanup_image: Whether to cleanup image after evaluation
        eval_dockerfile_script_content: eval_dockerfile.sh (targeted, for fail2pass judgment)

    Returns:
        Evaluation result dictionary with success, correct, test_results and report
    """
    logger.debug('=' * 80)
    logger.debug('🔨 build_and_evaluate() - Starting')
    logger.debug('=' * 80)
    logger.debug('📋 Build Configuration:')
    instance_id = instance.get('instance_id', 'N/A')
    repo = instance.get('repo', 'N/A')
    logger.debug(f'   Instance ID: {instance_id}')
    logger.debug(f'   Repo: {repo}')
    logger.debug(f'   Dockerfile size: {len(dockerfile_content) if dockerfile_content else 0} chars')
    logger.debug('=' * 80)

    if not dockerfile_content:
        logger.warning('❌ No Dockerfile content provided')
        return _make_full_result(
            instance,
            success=False,
            report={'error': 'No Dockerfile content provided'},
        )

    logger.debug('🚀 Starting evaluation phase')

    # Initialize variables for cleanup (needed in finally block)
    docker_client = None
    image_name = None

    # Get Docker build semaphore before entering the try block.
    # This is a synchronous, non-throwing operation (lazy Semaphore init).
    # Done here so the try/finally below cleanly pairs _active_builds +1/-1.
    docker_build_semaphore = _get_docker_build_semaphore()

    global _active_builds
    try:
        # Increment INSIDE try so that finally always decrements exactly once.
        _active_builds += 1
        # Step 1: Build Docker image (with concurrency control)
        logger.debug('   Step 1: Building Docker image from Dockerfile...')
        image_name = f'swe_builder_eval_{instance_id.lower().replace("/", "_").replace("-", "_")}:latest'
        logger.debug(f'   [Docker Build] Waiting for build slot (max concurrent: {_docker_build_concurrency})...')

        # Ensure docker library is available (auto-install if missing)
        global docker
        if docker is None:
            logger.warning('⚠️  Docker library not installed, attempting to install...')
            try:
                # Try to install docker library using subprocess (async-friendly)
                process = await asyncio.create_subprocess_exec(
                    sys.executable,
                    '-m',
                    'pip',
                    'install',
                    'docker',
                    stdout=asyncio.subprocess.PIPE,
                    stderr=asyncio.subprocess.PIPE,
                )
                stdout, stderr = await process.communicate()

                if process.returncode == 0:
                    logger.info('✅ Docker library installed successfully')
                    # Try importing again
                    import docker
                    import docker.errors

                    logger.debug('✅ Docker library imported successfully')
                else:
                    error_msg = stderr.decode() if stderr else 'Unknown error'
                    logger.error(f'❌ Failed to install docker library: {error_msg}')
                    return _make_full_result(
                        instance,
                        success=False,
                        report={'error': f'Docker library installation failed: {error_msg}'},
                    )
            except Exception as install_error:
                logger.error(f'❌ Failed to install docker library: {install_error}')
                return _make_full_result(
                    instance,
                    success=False,
                    report={'error': f'Docker library installation failed: {str(install_error)}'},
                )

        # Create Docker client
        try:
            docker_client = docker.from_env()
        except Exception as e:
            logger.error(f'❌ Failed to create Docker client: {e}')
            return _make_full_result(
                instance,
                success=False,
                report={'error': f'Docker client creation failed: {str(e)}'},
            )

        # Create temporary build directory
        with tempfile.TemporaryDirectory() as build_dir:
            build_path = Path(build_dir)
            dockerfile_path = build_path / 'Dockerfile'

            # Check if Dockerfile uses ubuntu-git:22.04 and ensure it exists
            if 'FROM ubuntu-git:22.04' in dockerfile_content:
                logger.debug('   🔍 Checking if ubuntu-git:22.04 base image exists...')
                try:
                    docker_client.images.get('ubuntu-git:22.04')
                    logger.debug('   ✅ ubuntu-git:22.04 image found')
                except docker.errors.ImageNotFound:
                    logger.error('   ❌ ubuntu-git:22.04 image not found!')
                    logger.error('   This image must be built first with:')
                    logger.error('   cd /path/to/nexus/deployment/docker && bash build-ubuntu-git.sh')
                    logger.error('')
                    logger.error('   Attempting to build it now...')
                    try:
                        # Try to build the ubuntu-git image
                        nexus_root = Path(__file__).parent.parent.parent.parent
                        build_script = nexus_root / 'deployment' / 'docker' / 'build-ubuntu-git.sh'
                        if build_script.exists():
                            import subprocess

                            result = subprocess.run(
                                ['bash', str(build_script)],
                                capture_output=True,
                                text=True,
                                timeout=300,  # 5 minutes timeout
                            )
                            if result.returncode == 0:
                                logger.info('   ✅ Successfully built ubuntu-git:22.04 image')
                            else:
                                logger.error(f'   ❌ Failed to build ubuntu-git:22.04: {result.stderr}')
                                raise RuntimeError(
                                    'ubuntu-git:22.04 image not found and automatic build failed. '
                                    'Please run: cd deployment/docker && bash build-ubuntu-git.sh'
                                )
                        else:
                            logger.error(f'   Build script not found at: {build_script}')
                            raise RuntimeError(
                                'ubuntu-git:22.04 image not found and build script is missing. '
                                'Please build it manually or check your nexus installation.'
                            )
                    except subprocess.TimeoutExpired as exc:
                        logger.error('   ❌ ubuntu-git:22.04 build timed out')
                        raise RuntimeError('ubuntu-git:22.04 build timed out after 5 minutes') from exc

            # Write Dockerfile
            with open(dockerfile_path, 'w') as f:
                f.write(dockerfile_content)

            logger.debug(f'   Building image {image_name}...')

            # CRITICAL: Use semaphore to limit concurrent Docker builds
            # This prevents resource exhaustion when many builds run simultaneously
            async with docker_build_semaphore:
                build_start_time = time.time()
                logger.debug('   [Docker Build] Acquired build slot, starting build...')

                try:
                    # Build the image
                    build_error_found = False
                    logger.debug('   Starting Docker build (no cache)...')
                    # 30-minute hard limit for the build; asyncio.wait_for gives us a real
                    # cancellation path even if the Docker SDK hangs internally.
                    build_timeout = 1800  # 60 minutes in seconds

                    # Run Docker build in executor to avoid blocking the event loop.
                    # nocache=True: each build starts from scratch, no dangling layer
                    # accumulation and no risk of concurrent builds deleting each other's
                    # cache layers (the "sha256: unable to find image" race condition).
                    loop = asyncio.get_running_loop()

                    def _build_image():
                        return docker_client.images.build(
                            path=str(build_path),
                            tag=image_name,
                            rm=True,
                            forcerm=True,
                            nocache=True,
                            timeout=build_timeout,
                        )

                    try:
                        image, build_logs = await asyncio.wait_for(
                            loop.run_in_executor(None, _build_image),
                            timeout=build_timeout + 60,  # asyncio layer adds a hard upper bound
                        )
                    except asyncio.TimeoutError as exc:
                        raise RuntimeError(f'Docker build timed out after {build_timeout}s (asyncio layer)') from exc

                    # Log build output with more detail
                    for log in build_logs:
                        if 'stream' in log:
                            log_line = log['stream'].strip()
                            if log_line:  # Only log non-empty lines
                                # Highlight important steps (including go/npm/pip commands)
                                if any(
                                    keyword in log_line.lower()
                                    for keyword in [
                                        'apt-get',
                                        'git',
                                        'go mod',
                                        'npm install',
                                        'pip install',
                                        'error',
                                        'failed',
                                    ]
                                ):
                                    logger.debug(f'   [BUILD] {log_line}')
                                else:
                                    logger.debug(f'   [BUILD] {log_line}')
                        elif 'error' in log:
                            logger.error(f'   [BUILD ERROR] {log["error"]}')
                            build_error_found = True
                        elif 'errorDetail' in log:
                            error_detail = log['errorDetail']
                            logger.error(f'   [BUILD ERROR DETAIL] {error_detail}')
                            # Log the full error detail including message
                            if isinstance(error_detail, dict) and 'message' in error_detail:
                                logger.error(f'   Error message: {error_detail["message"]}')
                            build_error_found = True

                    if build_error_found:
                        raise RuntimeError('Docker build encountered errors (see logs above)')

                    build_duration = time.time() - build_start_time
                    logger.info(f'   ✅ Image built successfully: {image_name} (took {build_duration:.1f}s)')
                    logger.debug('   [Docker Build] Released build slot')

                    # Verify git is installed in the built image
                    logger.debug('   🔍 Verifying git is installed in image...')
                    git_check_output = docker_client.containers.run(
                        image=image_name,
                        command=['/bin/sh', '-c', 'which git'],
                        remove=True,
                        detach=False,
                    )
                    git_path = (
                        git_check_output.decode().strip()
                        if isinstance(git_check_output, bytes)
                        else str(git_check_output).strip()
                    )
                    if git_path:
                        logger.debug(f'   ✅ Git is available in image at: {git_path}')
                        # Also check git version
                        git_version_output = docker_client.containers.run(
                            image=image_name,
                            command=['/bin/sh', '-c', 'git --version'],
                            remove=True,
                            detach=False,
                        )
                        git_version = (
                            git_version_output.decode().strip()
                            if isinstance(git_version_output, bytes)
                            else str(git_version_output).strip()
                        )
                        logger.debug(f'   ✅ Git version: {git_version}')
                    else:
                        logger.error('   ❌ Git verification failed - git not found in built image')
                        logger.error("   This suggests the Dockerfile's apt-get install git command failed")
                        logger.error('   Try checking network connectivity and apt repository availability')
                        raise RuntimeError('Git not found in built Docker image despite Dockerfile including it')

                except Exception as e:
                    logger.error(f'   ❌ Failed to build/verify image: {e}')
                    logger.error(traceback.format_exc())
                    logger.debug('   [Docker Build] Released build slot (after error)')
                    # Try to inspect what packages are actually installed (only if image exists)
                    try:
                        # First check if image exists before trying to inspect it
                        logger.debug('   🔍 Checking if image was partially built...')
                        docker_client.images.get(image_name)
                        # Image exists, try to inspect it
                        logger.debug('   🔍 Image exists, attempting to list installed packages...')
                        pkg_list_output = docker_client.containers.run(
                            image=image_name,
                            command=[
                                '/bin/sh',
                                '-c',
                                "dpkg -l | grep -E 'git|curl|wget' || echo 'No matching packages found'",
                            ],
                            remove=True,
                            detach=False,
                        )
                        pkg_list = (
                            pkg_list_output.decode().strip()
                            if isinstance(pkg_list_output, bytes)
                            else str(pkg_list_output).strip()
                        )
                        logger.debug(f'   Installed packages:\n{pkg_list}')
                    except docker.errors.ImageNotFound:
                        logger.warning('   ⚠️  Image does not exist (build failed before completion)')
                    except Exception as inspect_error:
                        logger.warning(f'   Could not inspect packages: {inspect_error}')
                    return _make_full_result(
                        instance,
                        success=False,
                        report={'error': f'Image build/verification failed: {str(e)}'},
                    )

        # Check if test_patch is empty (affects injection strategy)
        test_patch = instance.get('test_patch', '')
        test_patch_is_empty = not test_patch or not test_patch.strip()

        if test_patch_is_empty:
            logger.warning('   ⚠️  test_patch is empty for this instance')
            logger.debug('   Targeted evaluation: Phase A = no patches, Phase B = golden_patch only')

        # ============================================================
        # Determine scripts to use for each phase
        # ============================================================
        # Targeted script: eval_dockerfile.sh (fail2pass judgment)
        #   Fallback: eval.sh; if also absent, use generated fallback
        # Full script: eval.sh (test case enumeration)
        #   Fallback: generated script

        # Whether we have a dedicated targeted script (eval_dockerfile.sh)
        has_dedicated_targeted = bool(eval_dockerfile_script_content)

        # Prepare targeted script (eval_dockerfile.sh preferred, fallback to eval.sh or generated)
        if eval_dockerfile_script_content:
            targeted_script = eval_dockerfile_script_content
            logger.debug('   Targeted script: using eval_dockerfile.sh (dedicated)')
        elif eval_script_content:
            targeted_script = eval_script_content
            logger.debug('   Targeted script: using eval.sh as fallback (no eval_dockerfile.sh)')
        else:
            _instance_eval = instance.get('eval_script', '')
            targeted_script = _instance_eval if _instance_eval else generate_test_eval_script(instance)
            logger.debug('   Targeted script: using generated fallback (no eval_dockerfile.sh or eval.sh)')

        # Prepare full (comprehensive) script (eval.sh preferred, fallback to generated)
        if eval_script_content:
            full_script = eval_script_content
            logger.debug('   Full script: using eval.sh (comprehensive)')
        else:
            _instance_eval = instance.get('eval_script', '')
            full_script = _instance_eval if _instance_eval else generate_test_eval_script(instance)
            logger.debug('   Full script: using generated fallback (no eval.sh)')

        # Skip separate full phases when targeted and full scripts are identical
        # (no dedicated eval_dockerfile.sh AND no separate eval.sh → both would run same script)
        _scripts_are_identical = targeted_script == full_script

        _id_safe = instance_id.lower().replace('/', '_').replace('-', '_')

        # ============================================================
        # Phase targeted-A: targeted script + test_patch only
        # ============================================================
        if test_patch_is_empty:
            logger.debug('   Phase targeted-A: targeted tests, no patches (test_patch is empty)...')
        else:
            logger.debug('   Phase targeted-A: targeted tests, test_patch only...')
        targeted_a_result = await _run_test_phase(
            docker_client=docker_client,
            image_name=image_name,
            instance=instance,
            apply_patch=False,
            container_name=f'swe_eval_{_id_safe}_ta',
            eval_script_content=targeted_script,
            eval_timeout_seconds=task_input.eval_timeout_seconds,
        )

        # ============================================================
        # Phase targeted-B: targeted script + golden_patch + test_patch
        # ============================================================
        if test_patch_is_empty:
            logger.debug('   Phase targeted-B: targeted tests, golden_patch only (test_patch is empty)...')
        else:
            logger.debug('   Phase targeted-B: targeted tests, golden_patch + test_patch...')
        targeted_b_result = await _run_test_phase(
            docker_client=docker_client,
            image_name=image_name,
            instance=instance,
            apply_patch=True,
            container_name=f'swe_eval_{_id_safe}_tb',
            eval_script_content=targeted_script,
            eval_timeout_seconds=task_input.eval_timeout_seconds,
        )

        # ============================================================
        # Phase full-A: eval.sh + test_patch only
        # ============================================================
        if _scripts_are_identical:
            # No separate full script; reuse targeted phase outputs for test case enumeration
            logger.debug('   Phase full-A/B: reusing targeted outputs (targeted == full script)')
            full_a_result = targeted_a_result
            full_b_result = targeted_b_result
        else:
            if test_patch_is_empty:
                logger.debug('   Phase full-A: comprehensive tests, no patches...')
            else:
                logger.debug('   Phase full-A: comprehensive tests, test_patch only...')
            full_a_result = await _run_test_phase(
                docker_client=docker_client,
                image_name=image_name,
                instance=instance,
                apply_patch=False,
                container_name=f'swe_eval_{_id_safe}_fa',
                eval_script_content=full_script,
                eval_timeout_seconds=task_input.eval_timeout_seconds,
            )

            # ============================================================
            # Phase full-B: eval.sh + golden_patch + test_patch
            # ============================================================
            if test_patch_is_empty:
                logger.debug('   Phase full-B: comprehensive tests, golden_patch only...')
            else:
                logger.debug('   Phase full-B: comprehensive tests, golden_patch + test_patch...')
            full_b_result = await _run_test_phase(
                docker_client=docker_client,
                image_name=image_name,
                instance=instance,
                apply_patch=True,
                container_name=f'swe_eval_{_id_safe}_fb',
                eval_script_content=full_script,
                eval_timeout_seconds=task_input.eval_timeout_seconds,
            )

        # ============================================================
        # Determine success and resolved from targeted phases
        # ============================================================
        targeted_a_exit = targeted_a_result.get('exit_code')
        targeted_b_exit = targeted_b_result.get('exit_code')
        targeted_category = _classify_by_exit_codes(targeted_a_exit, targeted_b_exit)

        # success: golden patch passes (targeted-B exit_code == 0)
        # Changed from "docker build succeeded" to "passes golden patch test"
        success = targeted_b_exit == 0

        # resolved: fail2pass in targeted eval (A fails, B passes)
        # When test_patch is empty: pass2pass (A pass, B pass) also counts as resolved
        resolved = targeted_category in ['fail2pass'] if test_patch_is_empty else targeted_category == 'fail2pass'

        # ============================================================
        # Optimization: skip full eval phases when targeted is fail2fail
        # If targeted-B already fails (golden_patch doesn't fix it), full eval
        # will also fail and won't change resolved=False outcome.
        # ============================================================
        skip_full_eval = not resolved
        if skip_full_eval and not _scripts_are_identical:
            logger.info(f'   ⚡ Skipping full eval phases (targeted={targeted_category}, resolved={resolved})')
            full_a_result = targeted_a_result
            full_b_result = targeted_b_result
        elif not _scripts_are_identical:
            logger.info(f'   ✅ Targeted resolved={resolved}, running full eval for test case enumeration...')

        # ============================================================
        # Analyze test results from full eval phases
        # ============================================================
        test_results = analyze_test_results(
            output_a=full_a_result['output'],
            output_b=full_b_result['output'],
            instance=instance,
        )

        logger.info('   ✅ 4-Phase Evaluation completed')
        logger.debug(f'   Targeted-A exit: {targeted_a_exit} | Targeted-B exit: {targeted_b_exit}')
        logger.debug(f'   Targeted category: {targeted_category}')
        logger.debug(f'   test_patch_is_empty: {test_patch_is_empty}')
        logger.debug(f'   has_dedicated_targeted_script: {has_dedicated_targeted}')
        logger.info(f'   success: {success}')
        logger.info(f'   resolved (fail2pass in targeted): {resolved}')
        logger.info(f'   FAIL_TO_PASS: {len(test_results[TEST_CATEGORY_FAIL2PASS])} tests')
        logger.info(f'   FAIL_TO_FAIL: {len(test_results[TEST_CATEGORY_FAIL2FAIL])} tests')
        logger.info(f'   PASS_TO_PASS: {len(test_results[TEST_CATEGORY_PASS2PASS])} tests')
        logger.info(f'   PASS_TO_FAIL: {len(test_results[TEST_CATEGORY_PASS2FAIL])} tests')

        logger.debug('=' * 80)
        logger.info('✅ build_and_evaluate() - Completed')
        logger.debug('=' * 80)

        return _make_full_result(
            instance,
            success=success,
            resolved=resolved,
            test_results=test_results,
            report={
                # Targeted phase results (for fail2pass judgment)
                'category': targeted_category,  # top-level for compute_statistics.py compatibility
                'targeted_exit_a': targeted_a_exit,
                'targeted_exit_b': targeted_b_exit,
                'targeted_category': targeted_category,
                'targeted_output_a': targeted_a_result['output'],
                'targeted_output_b': targeted_b_result['output'],
                'has_dedicated_targeted_script': has_dedicated_targeted,
                # Full phase results (for test case enumeration)
                'full_exit_a': full_a_result.get('exit_code'),
                'full_exit_b': full_b_result.get('exit_code'),
                'full_output_a': full_a_result['output'],
                'full_output_b': full_b_result['output'],
                # Summary
                'test_patch_is_empty': test_patch_is_empty,
                'fail2pass_resolved': resolved,
                'skip_full_eval': skip_full_eval,
            },
        )

    except Exception as e:
        logger.error(f'❌ Evaluation failed: {e}')
        logger.error(traceback.format_exc())
        return _make_full_result(
            instance,
            success=False,
            report={'error': str(e), 'traceback': traceback.format_exc()},
        )
    finally:
        # Decrement active build counter first (before any I/O that could yield)
        _active_builds = max(0, _active_builds - 1)

        # Cleanup: Remove docker image (ensures cleanup even if killed or interrupted)
        # Only cleanup if cleanup_image is True (skip when upload is enabled)
        if cleanup_image and docker_client and image_name:
            try:
                logger.debug(f'🧹 [Cleanup] Removing Docker image: {image_name}')
                docker_client.images.remove(image_name, force=True)
                logger.debug(f'   ✅ Cleaned up image: {image_name}')
            except Exception as e:
                # Handle ImageNotFound and other exceptions gracefully
                error_msg = str(e)
                if 'not found' in error_msg.lower() or 'no such image' in error_msg.lower():
                    logger.debug(f'   ℹ️  Image not found (may have been cleaned up already): {image_name}')
                else:
                    logger.warning(f'   ⚠️  Failed to cleanup image {image_name}: {e}')
        elif not cleanup_image:
            logger.debug(f'🧹 [Cleanup] Skipping image cleanup (will be handled after upload): {image_name}')

        # nocache=True means Docker never creates intermediate cache layers, so there
        # are no dangling images to prune here.  The final tagged image is already
        # removed above.  No further cleanup needed.


async def _run_test_phase(
    docker_client,
    image_name: str,
    instance: dict[str, Any],
    apply_patch: bool,
    container_name: str,
    eval_script_content: str,
    eval_timeout_seconds: int = 1800,
) -> dict[str, Any]:
    """Run a single test phase in a container.

    Args:
        docker_client: Docker client
        image_name: Name of the image to run
        instance: Instance data dictionary
        apply_patch: Whether to apply the model patch
        container_name: Name for the container
        eval_script_content: eval.sh content from Agent (or fallback)

    Returns:
        Dictionary with 'output', 'exit_code', 'patch_applied'
    """
    container = None
    try:
        # CRITICAL: Use semaphore to limit concurrent container operations
        # Without this, 12 tasks × 4 phases × 10+ exec_run = 480+ concurrent ops → daemon crash
        container_ops_semaphore = _get_container_ops_semaphore()

        async with container_ops_semaphore:
            logger.debug(f'   [Container Ops] Acquired slot for {container_name}')
            assert docker_client is not None, 'docker client is not initialized'
            assert docker is not None, 'docker library is not initialized'
            # Remove any existing container with the same name
            try:
                old_container = docker_client.containers.get(container_name)
                logger.debug(f'   Removing existing container: {container_name}')
                old_container.remove(force=True)
            except docker.errors.NotFound:
                pass  # Container doesn't exist, which is good
            except Exception as e:
                logger.warning(f'   Failed to remove old container: {e}')

            # Create and start container
            # NOTE: Do not set working_dir here - if /testbed doesn't exist, container will fail to start
            container = docker_client.containers.create(
                image=image_name,
                name=container_name,
                detach=True,
                command='tail -f /dev/null',
                platform='linux/x86_64',
            )
            container.start()
            logger.debug(f'   Container started: {container_name}')

            # Verify repository exists in container (should be cloned by Dockerfile)
            # NOTE: eval.py runs on host, commands execute in evaluation container
            repo = instance.get('repo', '')
            base_commit = instance.get('base_commit', '')
            testbed_dir = instance.get('testbed_dir', '/testbed')  # Use from instance or default

            # Verify repository was properly set up in Dockerfile
            if repo and base_commit:
                # First, ensure testbed_dir exists (even if empty)
                # This is needed because old Dockerfiles might not have created it
                check_dir_cmd = f'test -d {testbed_dir}'
                exit_code, _ = container.exec_run(check_dir_cmd)
                if exit_code != 0:
                    logger.warning(f'   ⚠️  {testbed_dir} does not exist, creating it...')
                    mkdir_cmd = f'mkdir -p {testbed_dir}'
                    exit_code, output = container.exec_run(mkdir_cmd)
                    if exit_code != 0:
                        error_msg = output.decode('utf-8', errors='ignore') if output else 'Unknown error'
                        raise RuntimeError(f'Failed to create {testbed_dir}: {error_msg}')
                    logger.debug(f'   ✅ Created {testbed_dir}')

                # Check if testbed_dir is a git repository
                check_git_cmd = f'test -d {testbed_dir}/.git'
                exit_code, _ = container.exec_run(check_git_cmd)

                if exit_code != 0:
                    logger.error(f'   ❌ {testbed_dir} is not a git repository')
                    logger.error('   The Dockerfile should have cloned the repository')
                    logger.error('   This might be an old Dockerfile generated before the prompt update')
                    raise RuntimeError(
                        f'Repository not found at {testbed_dir}. '
                        f'The Dockerfile must clone the repository and checkout to {base_commit}. '
                        f'Please regenerate the Dockerfile with the updated Agent.'
                    )

                # Verify we're at the correct commit
                get_commit_cmd = f'cd {testbed_dir} && git rev-parse HEAD'
                exit_code, output = container.exec_run(get_commit_cmd)
                if exit_code == 0:
                    current_commit = output.decode('utf-8', errors='ignore').strip()
                    if current_commit == base_commit:
                        logger.debug(f'   ✅ Repository at correct commit: {base_commit[:12]}')
                    else:
                        # Check if base_commit is an ancestor of the current HEAD.
                        # This happens when the Dockerfile commits extra state on top of base_commit.
                        # eval.sh uses `git clean -fd` (not -fdx) so gitignored build artifacts
                        # (node_modules, build/, target/, etc.) survive; eval.sh handles the rest.
                        is_ancestor_cmd = f'cd {testbed_dir} && git merge-base --is-ancestor {base_commit} HEAD'
                        anc_exit, _ = container.exec_run(is_ancestor_cmd)
                        if anc_exit == 0:
                            logger.debug(
                                f'   ✅ base_commit {base_commit[:12]} is an ancestor of HEAD '
                                f'({current_commit[:12]}) — Dockerfile pre-committed build artifacts. '
                                f'eval.sh will handle git state reset.'
                            )
                        else:
                            logger.warning(
                                f'   ⚠️  Commit mismatch: expected {base_commit[:12]}, got {current_commit[:12]}'
                            )
                            logger.debug('   Attempting to fetch and reset to base_commit...')

                            # Try to fetch all branches and tags to ensure we have the commit
                            # Some commits might be on non-default branches or orphaned
                            fetch_cmd = f'cd {testbed_dir} && git fetch --all --tags --force'
                            exit_code, output = container.exec_run(fetch_cmd)
                            if exit_code != 0:
                                error_msg = output.decode('utf-8', errors='ignore') if output else 'Unknown error'
                                logger.warning(f'   ⚠️  Git fetch failed: {error_msg}')
                            else:
                                logger.debug('   ✅ Fetched all branches and tags')

                            # Now try to reset to the target commit with clean
                            # Note: -fd (NOT -fdx) preserves gitignored files (node_modules, build/, target/, etc.)
                            # so that Dockerfile-installed packages survive the reset.
                            reset_cmd = f'cd {testbed_dir} && git reset --hard {base_commit} && git clean -fd'
                            exit_code, output = container.exec_run(reset_cmd)
                            if exit_code != 0:
                                error_msg = output.decode('utf-8', errors='ignore') if output else 'Unknown error'
                                logger.error(f'   ❌ Failed to reset to base_commit {base_commit}: {error_msg}')

                                # Try one more time with fetch from origin
                                logger.debug('   Trying to fetch commit directly from origin...')
                                fetch_commit_cmd = f'cd {testbed_dir} && git fetch origin {base_commit}'
                                exit_code, output = container.exec_run(fetch_commit_cmd)
                                if exit_code == 0:
                                    logger.debug('   ✅ Fetched commit from origin')
                                    # Try reset again
                                    exit_code, output = container.exec_run(reset_cmd)
                                    if exit_code != 0:
                                        error_msg = (
                                            output.decode('utf-8', errors='ignore') if output else 'Unknown error'
                                        )
                                        raise RuntimeError(
                                            f'Failed to reset to base_commit {base_commit} after fetch: {error_msg}'
                                        )
                                else:
                                    error_msg = output.decode('utf-8', errors='ignore') if output else 'Unknown error'
                                    raise RuntimeError(
                                        f'Failed to reset to base_commit {base_commit} after fetch: {error_msg}\n'
                                        f'This commit might not exist in the repository or might be from a force-pushed branch.'
                                    )
                            logger.debug(
                                f'   ✅ Reset to base_commit and cleaned working directory: {base_commit[:12]}'
                            )
                else:
                    logger.debug(f'   ✅ {testbed_dir} is a git repository')

            # Use eval script passed as parameter (already prioritized in caller)
            eval_script = eval_script_content

            # Get patches
            # NOTE: 'patch' is the golden/correct patch for this instance
            # test_patch is the test assertions
            golden_patch = instance.get('patch', '')  # This is the correct fix
            test_patch = instance.get('test_patch', '')  # Test assertions

            # ====== NEW APPROACH: Inject patches into eval.sh ======
            # The eval.sh script contains a placeholder for patch content:
            #   git apply -v - <<'EOF_114329324912'
            #   EOF_114329324912
            # We need to inject the appropriate patch(es) between these markers:
            # - Phase A: Inject test_patch only (no golden_patch)
            # - Phase B: Inject golden_patch + test_patch (concatenated)

            patch_applied = None
            eval_script_injected = eval_script

            if eval_script:
                # Determine which patches to inject
                # Special handling for when test_patch is empty
                patch_to_inject = ''

                # Get test_patch_is_empty from outer scope (set earlier)
                test_patch_is_empty = not test_patch or not test_patch.strip()

                if test_patch_is_empty:
                    # When test_patch is empty, use alternative strategy:
                    # - Phase A (apply_patch=False): No patches (empty EOF block)
                    # - Phase B (apply_patch=True): Only golden_patch
                    if apply_patch and golden_patch:
                        # Phase B: Apply golden patch only
                        patch_to_inject = golden_patch
                        logger.debug('   Injecting: golden_patch only (test_patch is empty)')
                        patch_applied = True
                    else:
                        # Phase A: No patches
                        patch_to_inject = ''
                        logger.debug('   Injecting: no patches (base_commit only, test_patch is empty)')
                        patch_applied = None
                else:
                    # Normal case: test_patch is present
                    if apply_patch and golden_patch:
                        # Phase B: Apply golden patch first, then test patch
                        patch_to_inject = golden_patch
                        if test_patch:
                            patch_to_inject += '\n' + test_patch
                        logger.debug('   Injecting: golden_patch + test_patch')
                        patch_applied = True  # Assume success, eval.sh will handle errors
                    elif test_patch:
                        # Phase A: Apply test patch only
                        patch_to_inject = test_patch
                        logger.debug('   Injecting: test_patch only')
                        patch_applied = None  # No golden patch in Phase A

                # Inject patch into eval.sh by replacing content between EOF markers
                # Look for the pattern: <<'EOF_114329324912' ... EOF_114329324912
                # The EOF markers should be on their own lines, with the patch content between them
                import re

                # Match: <<'EOF_114329324912' followed by optional whitespace, content, then optional whitespace and EOF_114329324912
                # We'll preserve the structure and only replace the middle content
                eof_pattern = r"(<<'EOF_114329324912')\s*(.*?)\s*(EOF_114329324912)"

                match = re.search(eof_pattern, eval_script, re.DOTALL)
                if match:
                    # Replace the content between EOF markers with the patch
                    # Format: <<'EOF_114329324912'\n<patch>\nEOF_114329324912
                    eval_script_injected = (
                        eval_script[: match.end(1)]  # Include <<'EOF_114329324912'
                        + '\n'
                        + patch_to_inject
                        + '\n'  # Add patch with newlines
                        + match.group(3)  # Include EOF_114329324912
                        + eval_script[match.end(3) :]  # Rest of script
                    )
                    logger.debug(f'   ✅ Injected {len(patch_to_inject)} bytes into eval.sh EOF placeholder')
                else:
                    # No EOF placeholder found, eval.sh might not follow the expected format
                    logger.warning('   ⚠️  eval.sh does not contain EOF_114329324912 placeholder')
                    logger.warning('   ⚠️  Falling back to manual patch application before eval.sh')
                    eval_script_injected = eval_script

                    # Fallback: manually apply patches before running eval.sh (old behavior)
                    if apply_patch and golden_patch:
                        # Write golden patch to container using put_archive (safer than heredoc)
                        import io
                        import tarfile

                        tar_stream = io.BytesIO()
                        with tarfile.open(fileobj=tar_stream, mode='w') as tar:
                            # Create tarinfo for the patch file
                            tarinfo = tarfile.TarInfo(name='patch.diff')
                            patch_bytes = golden_patch.encode('utf-8')
                            tarinfo.size = len(patch_bytes)
                            tar.addfile(tarinfo, io.BytesIO(patch_bytes))

                        tar_stream.seek(0)
                        container.put_archive('/tmp', tar_stream)

                        # Try to apply golden patch
                        exit_code, output = container.exec_run(
                            'git apply -p1 -v /tmp/patch.diff',
                            workdir=testbed_dir,
                        )

                        patch_applied = exit_code == 0

                        if exit_code != 0:
                            logger.warning(
                                f'   Golden patch apply failed with exit code {exit_code}, trying with fuzz...'
                            )
                            # Try with fuzz
                            exit_code, output = container.exec_run(
                                'patch --batch --fuzz=5 -p1 -i /tmp/patch.diff',
                                workdir=testbed_dir,
                            )
                            patch_applied = exit_code == 0

                        if patch_applied:
                            logger.debug(f'   {APPLY_PATCH_PASS}')
                        else:
                            logger.warning(f'   {APPLY_PATCH_FAIL}')
                            # Log the error output for debugging
                            error_output = output.decode('utf-8', errors='ignore') if output else 'No output'
                            logger.warning(f'   Golden patch error output: {error_output[:500]}')

                    # Always write test_patch if available (applied after golden_patch in Phase B)
                    test_patch_applied = None
                    if test_patch:
                        # Write test_patch to container using put_archive (safer than heredoc)
                        import io
                        import tarfile

                        tar_stream = io.BytesIO()
                        with tarfile.open(fileobj=tar_stream, mode='w') as tar:
                            # Create tarinfo for the patch file
                            tarinfo = tarfile.TarInfo(name='test_patch.diff')
                            patch_bytes = test_patch.encode('utf-8')
                            tarinfo.size = len(patch_bytes)
                            tar.addfile(tarinfo, io.BytesIO(patch_bytes))

                        tar_stream.seek(0)
                        container.put_archive('/tmp', tar_stream)

                        # Apply test_patch
                        # Note: In Phase B, this applies on top of golden_patch
                        # If they modify the same file in non-overlapping hunks, git apply handles it correctly
                        # If they have overlapping hunks, git apply will fail and we catch it
                        exit_code, output = container.exec_run(
                            'git apply -p1 -v /tmp/test_patch.diff',
                            workdir=testbed_dir,
                        )

                        test_patch_applied = exit_code == 0

                        if exit_code != 0:
                            logger.warning(f'   Test patch apply failed with exit code {exit_code}')
                            # Log error output to help debug
                            error_output = output.decode('utf-8', errors='ignore') if output else 'No output'
                            logger.warning(f'   Test patch error output: {error_output[:500]}')

                            # Try with fuzz (more lenient)
                            exit_code, output = container.exec_run(
                                'patch --batch --fuzz=5 -p1 -i /tmp/test_patch.diff',
                                workdir=testbed_dir,
                            )
                            test_patch_applied = exit_code == 0

                        if test_patch_applied:
                            logger.debug('   Test patch applied successfully')
                        else:
                            logger.error('   Test patch failed to apply!')
                            # This is critical in Phase A (only test_patch should apply)
                            # In Phase B, if test_patch fails after golden_patch succeeds, it might indicate
                            # overlapping changes, but golden_patch effects are still present

            # Write and execute the injected eval script.
            # NOTE: eval.sh starts with `git reset --hard HEAD && git clean -fd`, so any
            # manual patch application here would be undone immediately.  Patches are
            # injected directly into the eval.sh EOF placeholder above (lines 790-799),
            # which is the correct and only place they are applied.
            if eval_script_injected:
                import io
                import tarfile

                tar_stream = io.BytesIO()
                with tarfile.open(fileobj=tar_stream, mode='w') as tar:
                    tarinfo = tarfile.TarInfo(name='eval.sh')
                    script_bytes = eval_script_injected.encode('utf-8')
                    tarinfo.size = len(script_bytes)
                    tarinfo.mode = 0o755  # Make executable
                    tar.addfile(tarinfo, io.BytesIO(script_bytes))

                tar_stream.seek(0)
                container.put_archive('/', tar_stream)

                # Run eval.sh with a hard timeout via asyncio so the event loop is not
                # blocked indefinitely if tests hang (e.g. dead-lock, infinite loop).
                logger.debug(f'   Running eval.sh (timeout: {eval_timeout_seconds}s)...')
                loop = asyncio.get_running_loop()

                def _exec_eval():
                    return container.exec_run(
                        '/bin/bash /eval.sh',
                        workdir=testbed_dir,
                    )

                try:
                    exec_result = await asyncio.wait_for(
                        loop.run_in_executor(None, _exec_eval),
                        timeout=eval_timeout_seconds,
                    )
                    exit_code, output = exec_result
                except asyncio.TimeoutError:
                    logger.error(f'   ❌ eval.sh timed out after {eval_timeout_seconds}s')
                    exit_code = 124  # standard timeout exit code
                    output = b''

                test_output = output.decode('utf-8', errors='ignore') if output else ''

                # Prefer the exit code embedded in the script output (OMNIGRIL_EXIT_CODE=N)
                # over the Docker exec exit code, which may be 0 even when the test fails
                # due to shell error handling.
                match = EXIT_CODE_RE.search(test_output)
                if match:
                    extracted_exit = int(match.group(1))
                    logger.info(f'   Test exit code (from output): {extracted_exit}')
                    exit_code = extracted_exit
            else:
                logger.warning('   No eval script provided')
                test_output = ''
                exit_code = 1
                patch_applied = False

            return {
                'output': test_output,
                'exit_code': exit_code,
                'patch_applied': patch_applied,
            }

    except Exception as e:
        logger.error(f'   ❌ Test phase failed: {e}')
        logger.error(traceback.format_exc())
        return {
            'output': '',
            'exit_code': -1,
            'patch_applied': False,
            'error': str(e),
        }
    finally:
        # Cleanup container with retry mechanism
        if container:
            max_retries = 3
            for attempt in range(max_retries):
                try:
                    container.stop(timeout=10)
                    container.remove(force=True)
                    logger.debug(f'   🧹 Cleaned up container: {container_name}')
                    break  # Success, exit retry loop
                except Exception as e:
                    if attempt < max_retries - 1:
                        logger.warning(f'   ⚠️  Failed to cleanup container (attempt {attempt + 1}/{max_retries}): {e}')
                        await asyncio.sleep(2)  # Wait before retry
                    else:
                        logger.error(f'   ❌ Failed to cleanup container after {max_retries} attempts: {e}')
                        # Try force remove as last resort
                        try:
                            container.remove(force=True)
                            logger.debug(f'   ✅ Force removed container: {container_name}')
                        except Exception as force_err:
                            logger.error(f'   ❌ Force remove also failed: {force_err}')


def _classify_by_exit_codes(prev_exit: int | None, after_exit: int | None) -> str:
    """Classify test result by exit codes."""
    if prev_exit is None or after_exit is None:
        return 'error'
    prev_fail = prev_exit != 0
    after_pass = after_exit == 0
    if prev_fail and after_pass:
        return 'fail2pass'
    if prev_fail and not after_pass:
        return 'fail2fail'
    if (not prev_fail) and after_pass:
        return 'pass2pass'
    if (not prev_fail) and (not after_pass):
        return 'pass2fail'
    return 'error'


def generate_test_eval_script(instance: dict[str, Any]) -> str:
    """Generate standardized test script for container execution.

    Args:
        instance: Instance data dictionary

    Returns:
        Test script content as string
    """
    # Get working directory from instance or use default
    # Reference: dataset_handler.py uses testbed_dir (default /testbed)
    # Most projects have code in testbed root, not in a 'src' subdirectory
    testbed_dir = instance.get('testbed_dir', '/testbed')
    working_dir = instance.get('working_dir', testbed_dir)  # Default to testbed_dir itself
    # If working_dir is relative, make it relative to testbed_dir
    workdir = f'{testbed_dir}/{working_dir}' if not working_dir.startswith('/') else working_dir

    base_commit = instance.get('base_commit', '')

    # Comprehensive test script generation supporting multiple languages
    # Priority order: more specific detection first
    # Supported languages: Rust, Go, C#, Java, TypeScript, JavaScript, Python, Ruby, PHP, C, C++
    return f"""#!/bin/bash
set -e

cd {testbed_dir}

# CRITICAL: Clean any uncommitted changes before checkout
# This prevents "error: Your local changes to the following files would be overwritten by checkout"
# Note: -fd (not -fdx) preserves gitignored files (node_modules, build/, target/, etc.)
# so that Dockerfile-installed packages survive across test phases.
git reset --hard HEAD
git clean -fd

# Checkout to base commit (if provided)
{f'git checkout {base_commit}' if base_commit else '# No base_commit specified, staying at current commit'}

# Apply patches using git apply (EOF block for patch injection)
git apply -v - <<'EOF_114329324912'
EOF_114329324912

# Change to working directory if different from testbed
cd {workdir}

# Detect project type and run appropriate tests
# Priority: Rust > Go > C# > Swift > Scala > Kotlin > Java > TypeScript > Node.js > Python > Ruby > PHP > C/C++

if [ -f "Cargo.toml" ] || [ -f "../Cargo.toml" ] || [ -f "../../Cargo.toml" ]; then
    # Rust project
    echo "Detected Rust project, running tests..."
    cargo test --verbose || cargo test || true
    rc=$?
elif [ -f "go.mod" ] || [ -f "../go.mod" ] || [ -f "../../go.mod" ]; then
    # Go project
    echo "Detected Go project, running tests..."
    go test ./... -v || go test -v || true
    rc=$?
elif find . -maxdepth 2 -name "*.csproj" -o -name "*.sln" 2>/dev/null | grep -q .; then
    # C# / .NET project
    echo "Detected C# / .NET project, running tests..."
    dotnet test --verbosity normal || dotnet test || true
    rc=$?
elif [ -f "Package.swift" ] || [ -f "../Package.swift" ]; then
    # Swift project
    echo "Detected Swift project, running tests..."
    swift test || true
    rc=$?
elif [ -f "build.sbt" ] || [ -f "../build.sbt" ]; then
    # Scala project
    echo "Detected Scala project, running tests..."
    sbt test || true
    rc=$?
elif [ -f "build.gradle.kts" ] || [ -f "../build.gradle.kts" ]; then
    # Kotlin project (Gradle with Kotlin DSL)
    echo "Detected Kotlin project, running tests..."
    ./gradlew test || ./gradlew test --info || true
    rc=$?
elif [ -f "pom.xml" ]; then
    # Java Maven project
    echo "Detected Java Maven project, running tests..."
    mvn test || true
    rc=$?
elif [ -f "build.gradle" ]; then
    # Java Gradle project
    echo "Detected Java Gradle project, running tests..."
    ./gradlew test || true
    rc=$?
elif [ -f "tsconfig.json" ] || [ -f "tsconfig.base.json" ] || find . -maxdepth 2 -name "tsconfig*.json" 2>/dev/null | grep -q .; then
    # TypeScript project (check before Node.js as TS projects also have package.json)
    echo "Detected TypeScript project, running tests..."
    # Try TypeScript-specific test commands first
    (npm run test 2>/dev/null || yarn test 2>/dev/null || npm test 2>/dev/null || yarn test 2>/dev/null || true)
    rc=$?
elif [ -f "package.json" ]; then
    # Node.js / JavaScript project
    echo "Detected Node.js / JavaScript project, running tests..."
    npm test || yarn test || npm run test || yarn test || true
    rc=$?
elif [ -f "Gemfile" ] || [ -f "../Gemfile" ]; then
    # Ruby project
    echo "Detected Ruby project, running tests..."
    bundle exec rspec || bundle exec rake test || bundle exec rake || rake test || rake || true
    rc=$?
elif [ -f "composer.json" ] || [ -f "../composer.json" ]; then
    # PHP project
    echo "Detected PHP project, running tests..."
    vendor/bin/phpunit || phpunit || ./vendor/bin/phpunit || php vendor/bin/phpunit || true
    rc=$?
elif [ -f "CMakeLists.txt" ] || [ -f "../CMakeLists.txt" ]; then
    # C/C++ project with CMake
    echo "Detected C/C++ CMake project, running tests..."
    # Try to build and run tests if CMake build directory exists
    if [ -d "build" ] || [ -d "../build" ]; then
        (cd build 2>/dev/null || cd ../build 2>/dev/null || true) && ctest --verbose || make test || true
    else
        echo "CMake build directory not found, attempting to build..."
        mkdir -p build && cd build && cmake .. && cmake --build . && ctest --verbose || make test || true
    fi
    rc=$?
elif [ -f "Makefile" ] || [ -f "../Makefile" ]; then
    # C/C++ project with Makefile
    echo "Detected C/C++ Makefile project, running tests..."
    make test || make check || make || true
    rc=$?
elif find . -maxdepth 3 -name "*.cpp" -o -name "*.cxx" -o -name "*.cc" -o -name "*.c" -o -name "*.h" -o -name "*.hpp" 2>/dev/null | head -1 | grep -q .; then
    # C/C++ project detected by source files (fallback for projects without CMakeLists.txt or Makefile)
    echo "Detected C/C++ project (by source files), running tests..."
    # Try common C/C++ test patterns
    if [ -f "configure" ]; then
        ./configure && make test || make check || make || true
    elif [ -f "autogen.sh" ]; then
        ./autogen.sh && ./configure && make test || make check || make || true
    else
        make test || make check || make || true
    fi
    rc=$?
elif [ -f "pytest.ini" ] || [ -f "setup.py" ] || [ -f "requirements.txt" ] || [ -f "pyproject.toml" ] || find . -maxdepth 2 -name "*.py" 2>/dev/null | head -1 | grep -q .; then
    # Python project
    echo "Detected Python project, running tests..."
    python -m pytest -v || python -m unittest discover || python setup.py test || true
    rc=$?
else
    echo "Unknown project type, attempting common test commands..."
    # Try common test commands as fallback (ordered by commonality)
    (cargo test 2>/dev/null && rc=0) || \\
    (go test ./... -v 2>/dev/null && rc=0) || \\
    (dotnet test 2>/dev/null && rc=0) || \\
    (npm test 2>/dev/null && rc=0) || \\
    (python -m pytest -v 2>/dev/null && rc=0) || \\
    (mvn test 2>/dev/null && rc=0) || \\
    (./gradlew test 2>/dev/null && rc=0) || \\
    (bundle exec rspec 2>/dev/null && rc=0) || \\
    (vendor/bin/phpunit 2>/dev/null && rc=0) || \\
    (make test 2>/dev/null && rc=0) || \\
    (rc=1 && exit 1)
fi

echo "OMNIGRIL_EXIT_CODE=$rc"

# Clean before final checkout to avoid dirty state
cd {testbed_dir}
git reset --hard HEAD
git clean -fd
{f'git checkout {base_commit}' if base_commit else '# No base_commit specified'}
"""


def analyze_test_results(output_a: str, output_b: str, instance: dict[str, Any]) -> dict[str, list[str]]:
    """Analyze test results and classify test cases.

    Args:
        output_a: Phase A test output (test_patch only)
        output_b: Phase B test output (test_patch + patch)
        instance: Instance data dictionary

    Returns:
        Dictionary with test case classifications
    """
    logger.debug('   Extracting test cases from outputs...')

    # Extract test cases from outputs using regex patterns
    tests_a = _extract_test_cases_by_status(output_a)
    tests_b = _extract_test_cases_by_status(output_b)

    failed_a = tests_a.get('failed', set()) | tests_a.get('error', set())
    passed_a = tests_a.get('passed', set())
    failed_b = tests_b.get('failed', set()) | tests_b.get('error', set())
    passed_b = tests_b.get('passed', set())

    logger.info(f'   Phase A: {len(passed_a)} passed, {len(failed_a)} failed')
    logger.info(f'   Phase B: {len(passed_b)} passed, {len(failed_b)} failed')

    # Classify test cases
    all_tests = failed_a | passed_a | failed_b | passed_b

    result = {
        TEST_CATEGORY_FAIL2FAIL: [],
        TEST_CATEGORY_FAIL2PASS: [],
        TEST_CATEGORY_PASS2PASS: [],
        TEST_CATEGORY_PASS2FAIL: [],
    }

    for test in all_tests:
        a_failed = test in failed_a
        b_failed = test in failed_b

        if a_failed and b_failed:
            result[TEST_CATEGORY_FAIL2FAIL].append(test)
        elif a_failed and not b_failed:
            result[TEST_CATEGORY_FAIL2PASS].append(test)
        elif not a_failed and not b_failed:
            result[TEST_CATEGORY_PASS2PASS].append(test)
        elif not a_failed and b_failed:
            result[TEST_CATEGORY_PASS2FAIL].append(test)

    # Sort for consistent output
    for key in result:
        result[key] = sorted(result[key])

    # Special case: compilation failure handling
    # If prev run fails to compile but after run compiles and tests pass, treat ALL passed tests as FAIL_TO_PASS
    if _looks_like_compile_failure(output_a, list(failed_a)) and passed_b:
        logger.info('   Detected compilation failure in Phase A -> all Phase B passed tests are FAIL_TO_PASS')
        result[TEST_CATEGORY_FAIL2PASS] = sorted(passed_b)
        result[TEST_CATEGORY_FAIL2FAIL] = []
        result[TEST_CATEGORY_PASS2FAIL] = []
        result[TEST_CATEGORY_PASS2PASS] = []

    return result


def _is_valid_test_case_name(name: str, framework: str = 'unknown') -> bool:
    """Validate if a string looks like a valid test case name.

    This filters out false positives from regex matching (e.g., "to", "all", "successfully").

    Args:
        name: Potential test case name
        framework: Test framework hint (pytest, junit, cargo, dotnet, etc.)

    Returns:
        True if valid test case name, False if likely a false positive
    """
    # Basic validation: minimum length
    if len(name) < 3:
        return False

    # Exclude common false positives (words that appear in error messages)
    COMMON_FALSE_POSITIVES = {
        'to',
        'all',
        'the',
        'from',
        'with',
        'and',
        'or',
        'not',
        'be',
        'is',
        'are',
        'successfully',
        'due',
        'because',
        'after',
        'before',
        'during',
        'while',
        'validation',
        'checks',
        'configuration',
        'connection',
        'error',
        'failed',
        'passed',
        'test',
        'run',
        'running',
        'tests',
        'suite',
        'case',
        'result',
    }
    if name.lower() in COMMON_FALSE_POSITIVES:
        return False

    # Framework-specific validation
    if framework == 'pytest':
        # Pytest tests must contain :: separator
        if '::' not in name:
            return False
        # Should have file path or module structure
        if not any(x in name.lower() for x in ['test_', 'test/', '.py::', '/test']):
            return False

    elif framework == 'junit' or framework == 'unittest':
        # JUnit/unittest tests should have package/module structure (at least one dot)
        if '.' not in name:
            return False
        # Class name should exist and start with uppercase
        parts = name.split('.')
        if len(parts) >= 2 and not any(p and p[0].isupper() for p in parts[:-1]):
            return False

    elif framework == 'cargo':
        # Cargo tests often have :: separator or are simple names
        # Should contain alphanumeric and underscores
        if not re.match(r'^[\w:]+$', name):
            return False

    elif framework == 'dotnet':
        # .NET tests should have namespace structure
        if '.' not in name:
            return False
        # Should contain alphanumeric, dots, underscores
        if not re.match(r'^[\w.]+$', name):
            return False

    elif framework == 'go':
        # Go tests should start with Test and use CamelCase
        if not name.startswith('Test'):
            return False
        # Should contain alphanumeric and underscores
        if not re.match(r'^Test[A-Za-z0-9_/]+$', name):
            return False

    elif framework == 'javascript':
        # Jest/Mocha tests can have various formats, basic length check
        if len(name) < 5:
            return False
        # Filter out obvious false positives
        if name.lower() in ['passed', 'failed', 'error', 'skipped']:
            return False

    elif framework == 'php':
        # PHPUnit tests should contain :: or Test keyword
        if '::' not in name and 'Test' not in name and 'test' not in name:
            return False

    elif framework == 'cpp':
        # C++ gtest/Catch2 tests should have proper naming
        if len(name) < 5:
            return False
        # Should contain alphanumeric, dots, underscores, slashes
        if not re.match(r'^[\w./]+$', name):
            return False

    elif framework == 'c':
        # CTest tests are usually simple names
        if not re.match(r'^[\w_-]+$', name):
            return False

    elif framework == 'ruby':
        # RSpec/Minitest tests
        if len(name) < 5:
            return False
        # Should contain alphanumeric, underscores, #
        if not re.match(r'^[\w#:_]+$', name):
            return False

    return True


def _extract_test_cases_by_status(content: str) -> dict[str, set[str]]:
    """Extract test cases by status from test output.

    Supports multiple test frameworks: pytest, unittest, Maven, Gradle, Cargo, .NET, etc.

    Now includes validation to filter out false positives like "to", "all", "successfully".

    Args:
        content: Test output content

    Returns:
        Dictionary with 'passed', 'failed', 'error', 'skipped' sets
    """
    out: dict[str, set[str]] = {'passed': set(), 'failed': set(), 'error': set(), 'skipped': set()}

    if not content:
        return out

    # Pytest patterns
    _RGX_PYTEST_NODEID_STATUS = re.compile(
        r'^([^\s]+::[^\s]+)\s+(PASSED|FAILED|SKIPPED|XFAIL|XPASS|ERROR)\b', re.MULTILINE
    )
    _RGX_PYTEST_NODEID = re.compile(r'^(?:FAILED|ERROR)\s+([^\s]+::[^\s]+)', re.MULTILINE)
    # Removed overly permissive patterns that match "FAILED to..." as test names:
    # _RGX_PYTEST_FAILED_LINE = re.compile(r"^FAILED\s+([^\s]+)", re.MULTILINE)
    # _RGX_PYTEST_PASSED_LINE = re.compile(r"^PASSED\s+([^\s]+)", re.MULTILINE)

    # Unittest patterns
    _RGX_UNITTEST_FAIL = re.compile(r'^(FAIL|ERROR):\s+([^\s(]+)\s*\(([^)]+)\)', re.MULTILINE)

    # Gradle/JUnit patterns
    _RGX_JUNIT_STYLE = re.compile(r'^(\S+)\s*>\s*(\S+)\s+FAILED\b', re.MULTILINE)
    _RGX_GRADLE_JUNIT_STATUS = re.compile(r'^(\S+)\s*>\s*(\S+)\s+(PASSED|FAILED|SKIPPED)\b', re.MULTILINE)

    # Maven patterns
    _RGX_MAVEN_SUREFIRE_FAIL = re.compile(r'^\[ERROR\]\s+([^\s]+)\(([^\)]+)\)\s+Time elapsed', re.MULTILINE)
    _RGX_MAVEN_SUREFIRE_RUNNING = re.compile(r'^\[INFO\]\s+Running\s+([A-Za-z0-9_.$]+)\s*$', re.MULTILINE)

    # Cargo patterns
    _RGX_CARGO_PASSED = re.compile(r'^test\s+([^\s]+)\s+\.\.\.\s+ok', re.MULTILINE)
    _RGX_CARGO_FAILED = re.compile(r'^test\s+([^\s]+)\s+\.\.\.\s+FAILED', re.MULTILINE)

    # .NET patterns
    _RGX_XUNIT_PASSED = re.compile(r'^\s*√\s+(.+?)\s*\[', re.MULTILINE)
    _RGX_XUNIT_FAILED = re.compile(r'^\s*X\s+(.+?)\s*\[', re.MULTILINE)
    _RGX_DOTNET_PASSED = re.compile(r'^Passed!\s+-\s+([^\s(]+)', re.MULTILINE)
    _RGX_DOTNET_FAILED = re.compile(r'^Failed!\s+-\s+([^\s(]+)', re.MULTILINE)
    # Improved NUnit patterns: require namespace structure (ClassName.MethodName)
    _RGX_NUNIT_PASSED = re.compile(
        r'^\s*Passed\s+([A-Z][a-zA-Z0-9_]*\.[a-zA-Z0-9_]+(?:\.[a-zA-Z0-9_]+)*)', re.MULTILINE
    )
    _RGX_NUNIT_FAILED = re.compile(
        r'^\s*Failed\s+([A-Z][a-zA-Z0-9_]*\.[a-zA-Z0-9_]+(?:\.[a-zA-Z0-9_]+)*)', re.MULTILINE
    )

    # Go test patterns
    _RGX_GO_PASSED = re.compile(r'^---\s+PASS:\s+(\S+)', re.MULTILINE)
    _RGX_GO_FAILED = re.compile(r'^---\s+FAIL:\s+(\S+)', re.MULTILINE)
    _RGX_GO_SKIP = re.compile(r'^---\s+SKIP:\s+(\S+)', re.MULTILINE)

    # TypeScript/JavaScript patterns (Jest, Mocha)
    _RGX_JEST_PASSED = re.compile(r'^\s*✓\s+(.+?)(?:\s+\(\d+\s*ms\))?$', re.MULTILINE)
    _RGX_JEST_FAILED = re.compile(r'^\s*✕\s+(.+?)(?:\s+\(\d+\s*ms\))?$', re.MULTILINE)
    _RGX_MOCHA_PASSED = re.compile(r'^\s*✓\s+(.+?)(?:\s+\(\d+ms\))?$', re.MULTILINE)
    _RGX_MOCHA_FAILED = re.compile(r'^\s*\d+\)\s+(.+)$', re.MULTILINE)

    # PHP (PHPUnit) patterns
    _RGX_PHPUNIT_OK = re.compile(r'^\s*OK\s+\(.+\)$', re.MULTILINE)
    _RGX_PHPUNIT_FAIL = re.compile(r'^\d+\)\s+(.+)$', re.MULTILINE)
    _RGX_PHPUNIT_TEST_PASS = re.compile(r'^\s*\.\s+(\S+)', re.MULTILINE)

    # C++ (gtest, Catch2) patterns
    _RGX_GTEST_PASSED = re.compile(r'^\[\s+OK\s+\]\s+(.+?)\s+\(\d+\s+ms\)', re.MULTILINE)
    _RGX_GTEST_FAILED = re.compile(r'^\[\s+FAILED\s+\]\s+(.+)', re.MULTILINE)
    _RGX_CATCH2_PASSED = re.compile(r'^passed:\s+(.+)$', re.MULTILINE)
    _RGX_CATCH2_FAILED = re.compile(r'^failed:\s+(.+)$', re.MULTILINE)

    # C (CTest) patterns
    _RGX_CTEST_PASSED = re.compile(r'^\d+/\d+\s+Test\s+#\d+:\s+(\S+)\s+\.+\s+Passed', re.MULTILINE)
    _RGX_CTEST_FAILED = re.compile(r'^\d+/\d+\s+Test\s+#\d+:\s+(\S+)\s+\.+\*+Failed', re.MULTILINE)

    # Ruby (RSpec, Minitest) patterns
    _RGX_RSPEC_PASSED = re.compile(r'^\s+(\S+)\s+\(PASSED\)', re.MULTILINE)
    _RGX_RSPEC_FAILED = re.compile(r'^\s+(\S+)\s+\(FAILED\)', re.MULTILINE)
    _RGX_MINITEST_OK = re.compile(r'^(\w+#test_\w+)\s+=\s+[\d.]+\s+s\s+=\s+\.$', re.MULTILINE)
    _RGX_MINITEST_FAIL = re.compile(r'^(\w+#test_\w+)\s+=\s+[\d.]+\s+s\s+=\s+F$', re.MULTILINE)

    # Extract pytest results (most common pattern)
    for m in _RGX_PYTEST_NODEID_STATUS.finditer(content):
        nodeid, status = m.group(1), m.group(2)
        if status == 'PASSED':
            out['passed'].add(nodeid)
        elif status == 'FAILED':
            out['failed'].add(nodeid)
        elif status == 'ERROR':
            out['error'].add(nodeid)
        elif status in ('SKIPPED', 'XFAIL', 'XPASS'):
            out['skipped'].add(nodeid)

    for m in _RGX_PYTEST_NODEID.finditer(content):
        out['failed'].add(m.group(1))
    # Removed overly permissive line-based matching (caused false positives)

    # Extract unittest results
    for m in _RGX_UNITTEST_FAIL.finditer(content):
        _, test_name, cls_mod = m.group(1), m.group(2), m.group(3)
        out['failed'].add(f'unittest::{cls_mod}.{test_name}')

    # Extract Gradle/JUnit results
    for m in _RGX_JUNIT_STYLE.finditer(content):
        cls, test = m.group(1), m.group(2)
        out['failed'].add(f'junit::{cls}.{test}')

    for m in _RGX_GRADLE_JUNIT_STATUS.finditer(content):
        cls, test, status = m.group(1), m.group(2), m.group(3)
        sig = f'junit::{cls}.{test}'
        if status == 'PASSED':
            out['passed'].add(sig)
        elif status == 'FAILED':
            out['failed'].add(sig)
        elif status == 'SKIPPED':
            out['skipped'].add(sig)

    # Extract Maven results
    for m in _RGX_MAVEN_SUREFIRE_FAIL.finditer(content):
        test, cls = m.group(1), m.group(2)
        out['failed'].add(f'junit::{cls}.{test}')

    for m in _RGX_MAVEN_SUREFIRE_RUNNING.finditer(content):
        out['passed'].add(f'junit::{m.group(1)}')

    # Extract Cargo results
    for m in _RGX_CARGO_PASSED.finditer(content):
        out['passed'].add(f'cargo::{m.group(1)}')
    for m in _RGX_CARGO_FAILED.finditer(content):
        out['failed'].add(f'cargo::{m.group(1)}')

    # Extract .NET results
    for m in _RGX_XUNIT_PASSED.finditer(content):
        name = m.group(1).strip()
        if '[xUnit.net' in name:
            name = name.split(']', 1)[-1].strip()
        out['passed'].add(f'dotnet::{name}')

    for m in _RGX_XUNIT_FAILED.finditer(content):
        name = m.group(1).strip()
        if '[xUnit.net' in name:
            name = name.split(']', 1)[-1].strip()
        out['failed'].add(f'dotnet::{name}')

    for m in _RGX_DOTNET_PASSED.finditer(content):
        out['passed'].add(f'mstest::{m.group(1)}')
    for m in _RGX_DOTNET_FAILED.finditer(content):
        out['failed'].add(f'mstest::{m.group(1)}')

    for m in _RGX_NUNIT_PASSED.finditer(content):
        out['passed'].add(f'nunit::{m.group(1)}')
    for m in _RGX_NUNIT_FAILED.finditer(content):
        out['failed'].add(f'nunit::{m.group(1)}')

    # Extract Go test results
    for m in _RGX_GO_PASSED.finditer(content):
        out['passed'].add(f'go::{m.group(1)}')
    for m in _RGX_GO_FAILED.finditer(content):
        out['failed'].add(f'go::{m.group(1)}')
    for m in _RGX_GO_SKIP.finditer(content):
        out['skipped'].add(f'go::{m.group(1)}')

    # Extract TypeScript/JavaScript test results (Jest, Mocha)
    for m in _RGX_JEST_PASSED.finditer(content):
        test_name = m.group(1).strip()
        if test_name and len(test_name) > 3:
            out['passed'].add(f'jest::{test_name}')
    for m in _RGX_JEST_FAILED.finditer(content):
        test_name = m.group(1).strip()
        if test_name and len(test_name) > 3:
            out['failed'].add(f'jest::{test_name}')
    for m in _RGX_MOCHA_PASSED.finditer(content):
        test_name = m.group(1).strip()
        if test_name and len(test_name) > 3:
            out['passed'].add(f'mocha::{test_name}')
    for m in _RGX_MOCHA_FAILED.finditer(content):
        test_name = m.group(1).strip()
        if test_name and len(test_name) > 3:
            out['failed'].add(f'mocha::{test_name}')

    # Extract PHP (PHPUnit) test results
    for m in _RGX_PHPUNIT_FAIL.finditer(content):
        test_name = m.group(1).strip()
        if '::' in test_name or 'Test' in test_name:
            out['failed'].add(f'phpunit::{test_name}')
    for m in _RGX_PHPUNIT_TEST_PASS.finditer(content):
        test_name = m.group(1).strip()
        if len(test_name) > 3:
            out['passed'].add(f'phpunit::{test_name}')

    # Extract C++ test results (gtest, Catch2)
    for m in _RGX_GTEST_PASSED.finditer(content):
        out['passed'].add(f'gtest::{m.group(1).strip()}')
    for m in _RGX_GTEST_FAILED.finditer(content):
        out['failed'].add(f'gtest::{m.group(1).strip()}')
    for m in _RGX_CATCH2_PASSED.finditer(content):
        out['passed'].add(f'catch2::{m.group(1).strip()}')
    for m in _RGX_CATCH2_FAILED.finditer(content):
        out['failed'].add(f'catch2::{m.group(1).strip()}')

    # Extract C test results (CTest)
    for m in _RGX_CTEST_PASSED.finditer(content):
        out['passed'].add(f'ctest::{m.group(1)}')
    for m in _RGX_CTEST_FAILED.finditer(content):
        out['failed'].add(f'ctest::{m.group(1)}')

    # Extract Ruby test results (RSpec, Minitest)
    for m in _RGX_RSPEC_PASSED.finditer(content):
        out['passed'].add(f'rspec::{m.group(1)}')
    for m in _RGX_RSPEC_FAILED.finditer(content):
        out['failed'].add(f'rspec::{m.group(1)}')
    for m in _RGX_MINITEST_OK.finditer(content):
        out['passed'].add(f'minitest::{m.group(1)}')
    for m in _RGX_MINITEST_FAIL.finditer(content):
        out['failed'].add(f'minitest::{m.group(1)}')

    # Post-processing: validate and filter out false positives
    validated_out: dict[str, set[str]] = {'passed': set(), 'failed': set(), 'error': set(), 'skipped': set()}
    total_before = sum(len(v) for v in out.values())
    filtered_count = 0

    for status_key, test_set in out.items():
        for test in test_set:
            # Infer framework from test name format
            if '::' in test and not test.startswith((
                'junit::',
                'cargo::',
                'dotnet::',
                'nunit::',
                'mstest::',
                'unittest::',
                'go::',
                'jest::',
                'mocha::',
                'phpunit::',
                'gtest::',
                'catch2::',
                'ctest::',
                'rspec::',
                'minitest::',
            )):
                # Raw pytest format (file.py::test_name)
                framework = 'pytest'
                clean_name = test
            elif test.startswith('junit::'):
                framework = 'junit'
                clean_name = test[7:]  # Remove 'junit::' prefix
            elif test.startswith('cargo::'):
                framework = 'cargo'
                clean_name = test[7:]  # Remove 'cargo::' prefix
            elif test.startswith('dotnet::') or test.startswith('nunit::') or test.startswith('mstest::'):
                framework = 'dotnet'
                clean_name = test.split('::', 1)[-1]
            elif test.startswith('unittest::'):
                framework = 'unittest'
                clean_name = test[10:]  # Remove 'unittest::' prefix
            elif test.startswith('go::'):
                framework = 'go'
                clean_name = test[4:]  # Remove 'go::' prefix
            elif test.startswith('jest::') or test.startswith('mocha::'):
                framework = 'javascript'
                clean_name = test.split('::', 1)[-1]
            elif test.startswith('phpunit::'):
                framework = 'php'
                clean_name = test[9:]  # Remove 'phpunit::' prefix
            elif test.startswith('gtest::') or test.startswith('catch2::'):
                framework = 'cpp'
                clean_name = test.split('::', 1)[-1]
            elif test.startswith('ctest::'):
                framework = 'c'
                clean_name = test[7:]  # Remove 'ctest::' prefix
            elif test.startswith('rspec::') or test.startswith('minitest::'):
                framework = 'ruby'
                clean_name = test.split('::', 1)[-1]
            else:
                framework = 'unknown'
                clean_name = test

            # Validate test name
            if _is_valid_test_case_name(clean_name, framework):
                validated_out[status_key].add(test)
            else:
                filtered_count += 1
                logger.debug(f'Filtered invalid test name: {test} (framework={framework})')

    total_after = sum(len(v) for v in validated_out.values())
    if filtered_count > 0:
        logger.debug(f'   Filtered {filtered_count} invalid test names ({total_before} -> {total_after} valid tests)')

    return validated_out


def _looks_like_compile_failure(content: str, failures: list[str]) -> bool:
    """Check if the output looks like a compilation failure."""
    # Check if any failure signatures indicate compilation errors
    if any(x.startswith(('javac::', 'csharp::')) for x in failures):
        return True

    # Check for common compilation error indicators
    indicators = [
        'COMPILATION ERROR',
        'Compilation failure',
        'BUILD FAILURE',
        'BUILD FAILED',
        'Build FAILED',
        'error: could not compile',  # Rust
        'error[E',  # Rust error code
        'FAILED (errors=',  # unittest compile/import error
    ]
    if any(ind in content for ind in indicators):
        return True

    # Maven-specific check
    return bool('maven-compiler-plugin' in content and 'BUILD FAILURE' in content)
