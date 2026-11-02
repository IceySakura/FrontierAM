"""Evaluation operations for SWE-bench tasks.

This module contains functions for evaluating patches in the runtime environment.
"""

import json
import logging
import re

from typing import Any

from nexus.extensions.tasks.swe.constants import SWEErrorCode, SWEbenchInstance
from nexus.extensions.tasks.swe.dataset_handler import get_handler
from nexus.runtime import Runtime


logger = logging.getLogger(__name__)


def _parse_patch_files(patch: str) -> list[str]:
    """Extract old and new file paths from diff headers.

    Header parsing also supports incomplete patches. Both sides of a rename are
    needed to restore existing test files and remove newly added test paths.
    """
    files: list[str] = []
    seen: set[str] = set()
    for m in re.finditer(r'^diff --git a/(.+?) b/(.+?)$', patch, re.M):
        for path in (m.group(1), m.group(2)):
            if path and path != '/dev/null' and path not in seen:
                seen.add(path)
                files.append(path)
    return files


def _is_safe_relpath(path: str) -> bool:
    """Reject absolute paths and `..` traversal so restore stays inside the repo."""
    if not path or path.startswith('/'):
        return False
    parts = path.split('/')
    return '..' not in parts


def _normalize_instance_for_runtime(instance: SWEbenchInstance) -> SWEbenchInstance:
    """Convert structured repo_language metadata to a string before runtime invocation.

    Preserve the original structure in repo_language_detail and leave other fields
    unchanged for dataset handlers that expect a string language name.
    """
    repo_language = instance.get('repo_language')
    if not isinstance(repo_language, dict):
        return instance
    flat = repo_language.get('language')
    if not isinstance(flat, str) or not flat:
        flat = instance.get('language') if isinstance(instance.get('language'), str) else ''
    normalized = dict(instance)
    normalized['repo_language'] = flat
    # Retain structured language metadata for consumers that need it.
    normalized['repo_language_detail'] = repo_language
    return normalized  # type: ignore[return-value]


async def evaluate_and_grade_in_runtime(
    runtime: Runtime,
    patch_text: str,
    instance: SWEbenchInstance,
    instance_id: str,
    timeout_seconds: int,
    *,
    include_tests_status: bool = True,
    reverse_patch: bool = False,
) -> dict[str, Any]:
    """Run evaluation + grading fully inside the runtime.

    Args:
        reverse_patch: If True, apply patch with --reverse (for SWE-Smith oracle mode)
    """
    return await runtime.invoke(
        _evaluate_and_grade,
        patch_text,
        _normalize_instance_for_runtime(instance),
        instance_id,
        timeout_seconds,
        include_tests_status,
        reverse_patch,
    )


async def evaluate_patch_in_runtime(
    runtime: Runtime,
    instance: SWEbenchInstance,
    patch_text: str,
    instance_id: str,
    timeout_seconds: int,
    eval_script: str,
) -> dict[str, Any]:
    return await runtime.invoke(
        _evaluate_patch,
        _normalize_instance_for_runtime(instance),
        patch_text,
        instance_id,
        timeout_seconds,
        eval_script,
    )


def _compute_partial_score(instance_report: dict[str, Any]) -> float:
    """Compute partial reward based on test pass rates.

    Score = f2p_rate * p2p_rate, where:
      - f2p_rate: fraction of fail-to-pass tests that now pass (resolution)
      - p2p_rate: fraction of pass-to-pass tests that still pass (maintenance)

    Returns 0.0 if tests_status is unavailable or patch was not applied.
    """
    tests_status = instance_report.get('tests_status')
    if not tests_status:
        return 0.0

    f2p = tests_status.get('FAIL_TO_PASS', {})
    p2p = tests_status.get('PASS_TO_PASS', {})

    f2p_success = len(f2p.get('success', []))
    f2p_failure = len(f2p.get('failure', []))
    f2p_total = f2p_success + f2p_failure
    f2p_rate = f2p_success / f2p_total if f2p_total > 0 else 0.0

    p2p_success = len(p2p.get('success', []))
    p2p_failure = len(p2p.get('failure', []))
    p2p_total = p2p_success + p2p_failure
    p2p_rate = p2p_success / p2p_total if p2p_total > 0 else 1.0

    return f2p_rate * p2p_rate


def _evaluate_and_grade(
    patch_text: str,
    instance: SWEbenchInstance,
    instance_id: str,
    timeout_seconds: int,
    include_tests_status: bool = True,
    reverse_patch: bool = False,
) -> dict[str, Any]:
    """Apply patch, run tests, and compute report inside runtime.

    Intended to be executed inside the runtime via `Runtime.invoke`.

    Args:
        patch_text: The model's prediction patch to apply
        instance: SWE-bench instance data
        instance_id: Instance identifier
        timeout_seconds: Timeout for operations
        include_tests_status: Whether to include test status in report
        reverse_patch: If True, apply patch with --reverse (for SWE-Smith oracle mode)
    """
    from nexus.extensions.tasks.swe.constants import KEY_PREDICTION
    from nexus.extensions.tasks.swe.dataset_handler import get_handler

    dataset_type = instance['dataset_type']
    dataset_handler = get_handler(dataset_type)

    eval_script = dataset_handler.get_eval_script(instance)
    logger.debug(f'eval_script {eval_script}')

    result = _evaluate_patch(
        instance=instance,
        patch_text=patch_text,
        instance_id=instance_id,
        timeout_seconds=timeout_seconds,
        eval_script=eval_script,
        reverse_patch=reverse_patch,
    )

    apply_log = result.get('apply_log', '') or ''
    test_output = result.get('test_output', '') or ''
    test_output_path = result.get('test_output_path', '') or ''
    completed = bool(result.get('completed', False))
    error = result.get('error')
    error_code = result.get('error_code')

    report_json = ''
    resolved = False
    score = 0.0

    if not error and completed:
        prediction = {
            'instance_id': instance_id,
            KEY_PREDICTION: patch_text or '',
        }
        try:
            report = dataset_handler.get_eval_report_map(
                instance=instance,
                prediction=prediction,
                test_log_path=test_output_path,
                include_tests_status=include_tests_status,
            )
            report_json = json.dumps(report, indent=2)
            resolved = bool(report.get(instance_id, {}).get('resolved', False))
            score = 1.0 if resolved else 0.0
        except Exception as e:
            error = f'Failed to finalize evaluation report: {type(e).__name__}: {e}'
            error_code = SWEErrorCode.EVAL_REPORT_FAILED
            logger.exception('Failed to finalize evaluation report for %s', instance_id)

    return {
        'completed': completed and not bool(error),
        'error': error,
        'error_code': error_code,
        'apply_log': apply_log,
        'test_output': test_output,
        'test_output_path': test_output_path,
        'report_json': report_json,
        'resolved': resolved,
        'score': score,
    }


def _evaluate_patch(
    instance: SWEbenchInstance,
    patch_text: str,
    instance_id: str,
    timeout_seconds: int,
    eval_script: str,
    reverse_patch: bool = False,
) -> dict[str, Any]:
    """Apply patch and execute eval script inside the runtime environment.

    Args:
        patch_text: The model's prediction patch to apply
        instance_id: Instance identifier
        timeout_seconds: Timeout for operations
        eval_script: Script to run tests
        reverse_patch: If True, apply patch with --reverse (for SWE-Smith oracle mode)
    """
    import os
    import stat
    import subprocess
    import traceback

    from pathlib import Path

    # Import dependencies inside the function because it executes in the runtime.
    from nexus.extensions.tasks.swe.env import strip_dangling_display
    from nexus.extensions.tasks.swe.infer import remove_binary_diffs
    from nexus.utils.shell import get_os_env_without_ld_library_path, resolve_first_existing_path

    log_root = Path('/tmp/swebench') / instance_id.lower()
    log_root.mkdir(parents=True, exist_ok=True)

    repo_dir = instance.get('workdir', '/testbed')
    apply_log_parts: list[str] = []

    # Set up environment, including user's shell environment and remove LD_LIBRARY_PATH inherited from nix env
    os_envs = get_os_env_without_ld_library_path()
    # Remove a DISPLAY that points at an X server which was never started; left in
    # place it breaks browser-driven eval scripts in ways that never name DISPLAY.
    os_envs = strip_dangling_display(os_envs)
    # Set BASH_ENV only when the file exists and its parent is traversable by unprivileged children.
    # Use mode bits because this process may run as root.
    bashrc = Path(os.path.expanduser('~/.bashrc'))
    if bashrc.is_file() and bashrc.parent.stat().st_mode & stat.S_IXOTH:
        os_envs['BASH_ENV'] = str(bashrc)

    # Ensure nix runtime bin is in PATH for tools like patch, git, etc.
    # Append (not prepend) so instance-image python/pytest takes priority over nix python.
    nix_bin = '/nix/runtime/nix-env/bin'
    current_path = os_envs.get('PATH', '')
    if nix_bin not in current_path:
        os_envs['PATH'] = f'{current_path}:{nix_bin}' if current_path else nix_bin

    # Resolve the bash executable path
    executable_bash = resolve_first_existing_path(
        ['/nix/runtime/nix-env/bin/bash', '/bin/bash', 'bash'],
        fallback_to_last=True,
    )

    # Remove non-applicable binary diff blocks before applying the patch; preserve text changes.
    patch_text_for_apply = remove_binary_diffs(patch_text or '')
    if patch_text_for_apply != (patch_text or ''):
        apply_log_parts.append(
            f'[SANITIZE] removed binary diff blocks before apply: '
            f'{len(patch_text or "")} -> {len(patch_text_for_apply)} chars'
        )

    patch_file = log_root / 'patch.diff'
    patch_file.write_text(patch_text_for_apply)

    if reverse_patch:
        apply_log_parts.append('[ORACLE] Applying reverse patch for oracle mode')
        # Reference: SWE-Smith/SWE-Cosmos uses `git apply -R` with fallbacks for oracle mode
        git_apply_cmds = [
            'git apply -R --verbose --ignore-whitespace',
            'patch -R --batch --fuzz=5 -p1 --no-backup-if-mismatch -r /dev/null -i',
        ]
    else:
        git_apply_cmds = [
            'git apply --verbose --ignore-whitespace',
            'patch --batch --fuzz=5 -p1 --no-backup-if-mismatch -r /dev/null -i',
        ]

    applied_patch = False
    last_error = ''

    for git_apply_cmd in git_apply_cmds:
        cmd = f'{git_apply_cmd} {patch_file}'
        try:
            proc = subprocess.run(
                cmd,
                shell=True,
                cwd=repo_dir,
                capture_output=True,
                text=True,
                encoding='utf-8',
                timeout=timeout_seconds,
                env=os_envs,
                executable=executable_bash,
            )
            apply_log_parts.append(f'[MODEL PATCH] $ {cmd}\nstdout:\n{proc.stdout}\nstderr:\n{proc.stderr}\n')
            if proc.returncode == 0:
                applied_patch = True
                break
            last_error = proc.stderr.strip()
        except subprocess.TimeoutExpired as e:
            last_error = f'Timeout while applying patch with `{git_apply_cmd}`: {e}'
            apply_log_parts.append(last_error)

    if not applied_patch:
        return {
            'completed': False,
            'resolved': False,
            'error': f'Failed to apply patch: {last_error}',
            'error_code': SWEErrorCode.PATCH_APPLY_FAILED,
            'apply_log': '\n'.join(apply_log_parts),
        }

    # Restore test paths after applying the model patch and before test preparation.
    # Use HEAD as the available base tree. Skip restoration when evaluating the reference patch.
    test_patch = instance.get('test_patch', '') or ''
    if test_patch and not reverse_patch:
        base_ref = 'HEAD'
        restored: list[str] = []
        removed: list[str] = []
        skipped: list[str] = []
        for rel in _parse_patch_files(test_patch):
            if not _is_safe_relpath(rel):
                skipped.append(rel)
                continue
            # Does this path exist in the base tree? Use `git cat-file -e HEAD:<path>`.
            exists = subprocess.run(
                f'git cat-file -e "{base_ref}:{rel}"',
                shell=True,
                cwd=repo_dir,
                capture_output=True,
                text=True,
                env=os_envs,
                executable=executable_bash,
            )
            if exists.returncode == 0:
                # File exists at base: restore it, discarding any model.patch tampering.
                subprocess.run(
                    f'git checkout {base_ref} -- "{rel}"',
                    shell=True,
                    cwd=repo_dir,
                    capture_output=True,
                    text=True,
                    env=os_envs,
                    executable=executable_bash,
                )
                restored.append(rel)
            else:
                # File absent at base (test_patch creates it): drop any version the agent
                # pre-created so the eval_script's test_patch produces the canonical one.
                subprocess.run(
                    f'rm -f -- "{rel}"',
                    shell=True,
                    cwd=repo_dir,
                    capture_output=True,
                    text=True,
                    env=os_envs,
                    executable=executable_bash,
                )
                removed.append(rel)
        apply_log_parts.append(
            f'[ANTI-TAMPER] base_ref={base_ref} '
            f'restored={len(restored)}{": " + ", ".join(restored) if restored else ""}'
            f'{" | removed(new): " + ", ".join(removed) if removed else ""}'
            f'{" | skipped(unsafe): " + ", ".join(skipped) if skipped else ""}'
        )

    apply_log = '\n'.join(apply_log_parts)

    eval_script_path = log_root / 'eval.sh'
    test_output_path = log_root / 'test_output.txt'

    # Replace hardcoded /home/{repo} paths in eval script with actual repo_dir (only for multi-swe-bench, do not apply to other datasets)
    if instance.get('dataset_type', '') == 'multi-swe-bench' and eval_script and instance.get('repo'):
        repo = instance.get('repo')
        eval_script = eval_script.replace(f'/home/{repo}', str(repo_dir))

    eval_script_path.write_text(eval_script or '')

    try:
        proc = subprocess.run(
            [executable_bash, str(eval_script_path)],
            cwd=repo_dir,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            encoding='utf-8',
            timeout=timeout_seconds,
            env=os_envs,
        )
        dataset_handler = get_handler(instance.get('dataset_type'))
        dataset_handler.eval_write_test_output(instance, test_output_path, proc.stdout)
    except subprocess.TimeoutExpired as e:
        partial_output = e.output or ''
        if isinstance(partial_output, bytes):
            partial_output = partial_output.decode('utf-8', errors='replace')
        output_path = ''
        try:
            test_output_path.write_text(partial_output, encoding='utf-8')
            output_path = str(test_output_path)
        except OSError:
            logger.warning('Failed to save timed-out test output for %s', instance_id, exc_info=True)
        return {
            'completed': False,
            'resolved': False,
            'error': f'Test execution timed out after {timeout_seconds}s: {e}',
            'error_code': SWEErrorCode.TEST_EXECUTION_TIMEOUT,
            'apply_log': apply_log,
            'test_output': partial_output,
            'test_output_path': output_path,
        }
    except Exception as e:
        return {
            'completed': False,
            'resolved': False,
            'error': f'Failed to run eval script: {e}',
            'error_code': SWEErrorCode.EVAL_SCRIPT_FAILED,
            'apply_log': apply_log,
            'traceback': traceback.format_exc(),
        }

    test_output = proc.stdout or ''
    try:
        test_output = test_output_path.read_text(encoding='utf-8')
    except (OSError, UnicodeError) as e:
        if isinstance(e, UnicodeDecodeError):
            # Recover readable evidence without treating a damaged log as a valid grading result.
            test_output = e.object.decode('utf-8', errors='replace')
        return {
            'completed': False,
            'resolved': False,
            'error': f'Failed to read test output: {type(e).__name__}: {e}',
            'error_code': SWEErrorCode.TEST_OUTPUT_READ_FAILED,
            'apply_log': apply_log,
            'test_output': test_output,
            'test_output_path': str(test_output_path),
            'return_code': proc.returncode,
            'traceback': traceback.format_exc(),
        }

    return {
        'completed': True,
        'apply_log': apply_log,
        'test_output': test_output,
        'test_output_path': str(test_output_path),
        'return_code': proc.returncode,
    }
