"""Host-side git operations in the sandbox repo, via ``runtime.invoke``.

The wrappers dispatch to the sandbox-side ``run_apply_sequence`` / ``run_apply`` / ``run_grade`` /
``reset_repo`` / ``run_reverse_patch`` in :mod:`nexus.extensions.tasks.swe_any.runtime_setup`
(referenced by import-path string, so importing this module never pulls in the sandbox code).
"""

from __future__ import annotations

import logging

from typing import Any

from nexus.runtime.models import InvokeRequest


logger = logging.getLogger(__name__)

APPLY_SEQUENCE_TARGET = 'nexus.extensions.tasks.swe_any.runtime_setup::run_apply_sequence'
APPLY_TARGET = 'nexus.extensions.tasks.swe_any.runtime_setup::run_apply'
GRADE_TARGET = 'nexus.extensions.tasks.swe_any.runtime_setup::run_grade'
RESET_TARGET = 'nexus.extensions.tasks.swe_any.runtime_setup::reset_repo'
REVERSE_PATCH_TARGET = 'nexus.extensions.tasks.swe_any.runtime_setup::run_reverse_patch'


async def grade_instance(
    runtime: Any,
    repo_dir: str,
    *,
    pregold_steps: list[tuple[str, str]],
    postgold_steps: list[tuple[str, str]],
    test_runner_script: str,
    test_files: list[str],
    request_timeout: float,
    base_commit: str,
) -> dict[str, Any]:
    """Execution-grade one instance in-sandbox: run the oracle tests pre-gold and post-gold.

    Returns ``{pregold, postgold}`` phase envelopes with validated ``test_statuses``.
    Infra failures propagate.

    *request_timeout* is the per-phase sandbox ceiling. The host aiohttp deadline is set to ~2× that
    (+margin) because ``run_grade`` runs pregold then postgold sequentially, each up to the ceiling:
    a single value for both would let the host give up at ~1× while the sandbox job runs on to ~2×,
    orphaning it against the shared ``repo_dir``.
    """
    host_deadline = 2.0 * request_timeout + 60.0
    request = InvokeRequest.build(
        GRADE_TARGET,
        venv_id=runtime.default_venv_id,
        request_timeout=host_deadline,  # host aiohttp deadline: covers BOTH sequential phases
        repo_dir=repo_dir,
        pregold_steps=[[name, patch] for name, patch in pregold_steps],
        postgold_steps=[[name, patch] for name, patch in postgold_steps],
        test_runner_script=test_runner_script,
        test_files=test_files,
        base_commit=base_commit,
        timeout=request_timeout,  # sandbox-side PER-PHASE subprocess ceiling
    )
    result: dict[str, Any] = await runtime.invoke(request)
    return result


async def apply_patches(
    runtime: Any,
    repo_dir: str,
    steps: list[tuple[str, str]],
    *,
    base_commit: str,
    request_timeout: float,
) -> dict[str, dict[str, Any]]:
    """Reset to pristine, apply ordered *steps*, and LEAVE them applied (persist; no trailing reset).

    The persist counterpart of :func:`apply_check_sequence`: used to set up state that a later step
    must observe — Issue's fully-masked working tree (reset to pristine, then apply feature_mask +
    test_mask, then let the agent browse it). *steps* is an ordered ``[(name, patch)]``; returns
    ``{name: {ok, empty, skipped, error}}``. Infra failures propagate to the caller.
    """
    request = InvokeRequest.build(
        APPLY_TARGET,
        venv_id=runtime.default_venv_id,
        request_timeout=request_timeout,
        repo_dir=repo_dir,
        steps=[[name, patch] for name, patch in steps],
        base_commit=base_commit,
        timeout=request_timeout,  # sandbox-side subprocess ceiling (not just the aiohttp wrap)
    )
    result: dict[str, Any] = await runtime.invoke(request)
    return result.get('results') or {}


async def reverse_patch(
    runtime: Any,
    repo_dir: str,
    patch: str,
    *,
    base_commit: str,
    request_timeout: float = 120.0,
) -> str:
    """git-canonical inverse of *patch*, computed in the sandbox (reset → ``git apply`` → ``git diff -R``).

    Used to derive the gold/restore patch from an AGENT-authored mask patch, where the host has only
    the forward text (the heuristic side already carries its inverse from ``build_edit_patch``). Since
    git generates it, the inverse is correct for cases a text reverser can't cover (whole-file
    create/delete, content starting with ``--`` / ``++``, ``\\ No newline``, multi-file). Returns ''
    when *patch* is empty OR does not apply — a soft outcome; the caller's apply-check then rejects
    that candidate and falls back. Infra failures propagate to the caller.
    """
    if not (patch or '').strip():
        return ''
    request = InvokeRequest.build(
        REVERSE_PATCH_TARGET,
        venv_id=runtime.default_venv_id,
        request_timeout=request_timeout,
        repo_dir=repo_dir,
        patch=patch,
        base_commit=base_commit,
        timeout=request_timeout,  # sandbox-side subprocess ceiling (not just the aiohttp wrap)
    )
    result: dict[str, Any] = await runtime.invoke(request)
    return result.get('reverse') or ''


async def apply_check_sequence(
    runtime: Any,
    repo_dir: str,
    steps: list[tuple[str, str]],
    *,
    base_commit: str,
    request_timeout: float,
) -> dict[str, dict[str, Any]]:
    """Validate that *steps* apply CUMULATIVELY in order, each on top of the previous (in-sandbox).

    *steps* is an ordered list of ``(name, patch)``; the sandbox resets to pristine, then git
    applies each patch on the accumulated tree (an empty patch is a no-op) and resets again at the
    end. Returns ``{name: {ok, empty, skipped, error}}`` in order — the first failing step stops
    the rest (``skipped=True``). Infra failures propagate to the caller, which records them as a
    soft per-feature error.
    """
    request = InvokeRequest.build(
        APPLY_SEQUENCE_TARGET,
        venv_id=runtime.default_venv_id,
        request_timeout=request_timeout,
        repo_dir=repo_dir,
        steps=[[name, patch] for name, patch in steps],
        base_commit=base_commit,
        timeout=request_timeout,  # sandbox-side subprocess ceiling (not just the aiohttp wrap)
    )
    result: dict[str, Any] = await runtime.invoke(request)
    return result.get('results') or {}


async def reset_repo(
    runtime: Any,
    repo_dir: str,
    *,
    base_commit: str,
    request_timeout: float = 300.0,
) -> bool:
    """Reset to *base_commit*, clean untracked files, and verify HEAD in the sandbox repo.

    This is intentionally strict: continuing after a reset failure would generate patches against
    an unknown tree. The sandbox target resolves and records the observed HEAD before success.
    """
    if not base_commit:
        raise ValueError('base_commit is required to reset the SWE-Any sandbox repository')
    try:
        request = InvokeRequest.build(
            RESET_TARGET,
            venv_id=runtime.default_venv_id,
            request_timeout=request_timeout,
            repo_dir=repo_dir,
            base_commit=base_commit,
            timeout=request_timeout,  # sandbox-side subprocess ceiling
        )
        result: dict[str, Any] = await runtime.invoke(request)
    except Exception as exc:
        raise RuntimeError(f'failed to reset {repo_dir} to base_commit {base_commit}: {exc}') from exc
    if not result.get('reset'):
        detail = result.get('error') or result.get('reason') or 'unknown reset failure'
        raise RuntimeError(f'failed to reset {repo_dir} to base_commit {base_commit}: {detail}')
    observed_head = str(result.get('head') or '')
    logger.debug('swe_any reset_repo %s: base_commit=%s head=%s', repo_dir, base_commit, observed_head)
    return True
