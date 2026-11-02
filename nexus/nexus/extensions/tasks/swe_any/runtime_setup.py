"""Sandbox-side execution for SWE-Any stages that need ``runtime.invoke``.

Implements **Parse** (``run_parse``), the deterministic **Mask** draft
(``run_mask``), and the deterministic **Eval** revert (``run_eval``). Select (host
heuristic + Claude Code agentic pass) does not use this module — the agent writes a
per-slice agentic JSON under ``/workspace/swe_any/``, and the host downloads /
archives to JSONL.

Invoked via ``runtime.invoke`` inside the repo sandbox (the module ships into
the worker venv via ``worker_venv``, so ``nexus.extensions.tasks.swe_any`` is
importable there). Parse:

1. (optionally) ``npm install``s ``scip-python`` when the baked image lacks it;
2. runs ``scip-python index`` over the repo source to emit ``parse_index.scip``;
3. builds the entity/relation graph and writes ``parse_graph.json`` inside the sandbox
   (so a later Select/Mask stage can consume it without a re-upload);
4. returns the sandbox artifact paths + node/edge counts for the host to
   download and score.

Coding rules (see docs/explanation/runtime-coding-guidelines.md): this target
uses the pure parse library via direct import and plain ``subprocess`` (wrapped
in ``asyncio.to_thread`` so the long index does not block the worker loop). It
needs no ``sio`` / ``require_runtime_control`` — there is no cross-process state.

``run_mask`` consumes a Select feature + ``parse_graph.json`` + the repo source to
emit a draft mask patch (host adds the agent refine on top). ``run_eval`` turns the
feature's candidate native test ids into a draft ``heuristic_mask_patch`` /
``heuristic_gold_patch`` pair (delete all candidates / restore them verbatim); the
agent test-oracle pass then authors the final test_mask + test_gold on top. Export
runs the execution-validated grade.
"""

from __future__ import annotations

import asyncio
import contextlib
import json
import logging
import os
import subprocess
import tempfile
import time

from pathlib import Path
from typing import Any


logger = logging.getLogger(__name__)


def _run_parse_blocking(
    *,
    repo_dir: str,
    graph_path: str,
    project_name: str | None,
    install_scip_python: bool,
    npm_registry: str | None,
    index_filename: str,
    index_timeout: float,
    npm_install_timeout: float,
) -> dict[str, Any]:
    """Do the blocking index + graph build. Runs in a worker thread.

    Always re-indexes and rebuilds — redo is governed by nexus' batch DB status
    (``run-batch --resume``), not by any output-file caching here.
    """
    from nexus.extensions.agents.installed_common import (
        DEFAULT_RUNTIME_BIN_DIR,
        build_npm_env,
        find_runtime_binary,
    )
    from nexus.extensions.tasks.swe_any.stage.parse import build_graph
    from nexus.extensions.tasks.swe_any.utils.classify_tests import annotate_tests
    from nexus.extensions.tasks.swe_any.utils.graph import filter_graph, iter_edge_facts, write_json
    from nexus.extensions.tasks.swe_any.utils.scip import ensure_scip_python, find_scip_python, run_scip_index

    repo_path = Path(repo_dir)
    if not repo_path.is_dir():
        raise FileNotFoundError(f'repo dir not found in sandbox: {repo_dir}')

    graph_file = Path(graph_path)
    graph_file.parent.mkdir(parents=True, exist_ok=True)

    # scip-python is a Node CLI and a baked user image usually lacks node/npm, so
    # run it through the nexus runtime's nix toolchain (like claude_code): resolve
    # npm there and prepend that bin dir to PATH. Install outside the repo so
    # `scip-python index .` never traverses its node_modules.
    npm_bin = find_runtime_binary('npm') or 'npm'
    env = {**os.environ, **build_npm_env(npm_registry=npm_registry)}
    # scip-python is a Node CLI; V8's default old-space (~2-4 GiB) OOMs indexing the largest repos
    # (cpython / spaCy / ... -> "JavaScript heap out of memory", abort -6). Raise it to 8 GiB. The
    # parse sandbox is provisioned with 16 GiB so node's non-old-space + external protobuf buffers
    # still fit under the container limit (heap ~= container/2). Preserve any inherited NODE_OPTIONS.
    _node_opts = env.get('NODE_OPTIONS', '').strip()
    env['NODE_OPTIONS'] = f'{_node_opts} --max-old-space-size=8192'.strip()
    nix_bin_dirs = [DEFAULT_RUNTIME_BIN_DIR]
    scip_install_dir = graph_file.parent

    scip_python = find_scip_python(scip_install_dir, extra_bin_dirs=nix_bin_dirs)
    if not scip_python and install_scip_python:
        scip_python = ensure_scip_python(
            install_dir=scip_install_dir,
            npm_bin=npm_bin,
            npm_registry=npm_registry,
            timeout=npm_install_timeout,
            env=env,
            extra_bin_dirs=nix_bin_dirs,
        )
    if not scip_python:
        raise FileNotFoundError(
            'scip-python not found in sandbox and install_scip_python=False. Bake it into the image '
            'or set install_scip_python=True.'
        )

    index_path = run_scip_index(
        repo_dir,
        scip_python=scip_python,
        project_name=project_name,
        output=index_filename,
        timeout=index_timeout,
        env=env,
    )
    # Build fresh from the index (no graph_path → no reuse). scip-python barely
    # tags tests, so enrich is_test with Python conventions before persisting.
    graph = build_graph(index_path)
    test_stats = annotate_tests(graph)
    write_json(graph, graph_file)
    internal = filter_graph(graph, include_external=False)
    return {
        'graph_path': str(graph_file),
        'index_path': str(index_path),
        'nodes_internal': internal.number_of_nodes(),
        'nodes_total': graph.number_of_nodes(),
        'nodes_test': test_stats['tests'],
        'nodes_test_added': test_stats['added'],
        'edges_internal': sum(1 for _ in iter_edge_facts(internal)),
        'edges_total': sum(1 for _ in iter_edge_facts(graph)),
        'scip_python': scip_python,
    }


async def run_parse(
    *,
    repo_dir: str,
    graph_path: str,
    project_name: str | None = None,
    install_scip_python: bool = True,
    npm_registry: str | None = 'http://mirrors.tencent.com/npm/',
    index_filename: str = 'parse_index.scip',
    index_timeout: float = 3600.0,
    npm_install_timeout: float = 600.0,
) -> dict[str, Any]:
    """Index *repo_dir* and build ``parse_graph.json`` at *graph_path* inside the sandbox.

    Returns a JSON-serialisable dict consumed by the host ``swe_any_parse`` task.
    Failures propagate to the host via ``runtime.invoke`` (no soft-error envelope).
    """
    start = time.time()
    result = await asyncio.to_thread(
        _run_parse_blocking,
        repo_dir=repo_dir,
        graph_path=graph_path,
        project_name=project_name,
        install_scip_python=install_scip_python,
        npm_registry=npm_registry,
        index_filename=index_filename,
        index_timeout=index_timeout,
        npm_install_timeout=npm_install_timeout,
    )
    result['execution_time'] = time.time() - start
    logger.debug(
        'swe_any run_parse done: %s nodes / %s edges total; %s nodes / %s edges internal',
        result['nodes_total'],
        result['edges_total'],
        result['nodes_internal'],
        result['edges_internal'],
    )
    return result


def _run_mask_blocking(
    *,
    graph_path: str,
    repo_dir: str,
    feature: list[str],
    heuristic_patch_path: str,
) -> dict[str, Any]:
    """Deterministic Mask draft for one Select feature (no LLM).

    Classifies the feature (``parse_graph.json``) into signature-kept entries vs deleted
    helpers and rewrites the repo source into a DRAFT mask/gold patch. Writes only the raw
    draft ``.patch`` (at *heuristic_patch_path*) so the refine agent reads it by path; the rest
    of the payload goes back to the host via the invoke result — nothing is downloaded.
    Candidate-test discovery is Eval's concern (a pure-graph step it runs itself), not Mask's.
    """
    from nexus.extensions.tasks.swe_any.stage.mask import apply_mask, classify_feature
    from nexus.extensions.tasks.swe_any.utils.graph import read_json

    graph = read_json(graph_path)
    plan = classify_feature(graph, feature)
    mask = apply_mask(repo_dir, graph, plan)

    # Only the raw draft patch touches the sandbox (the refine agent reads it by path, so the
    # host need not embed the whole diff into the prompt string); everything else is returned.
    patch_file = Path(heuristic_patch_path)
    patch_file.parent.mkdir(parents=True, exist_ok=True)
    patch_file.write_text(str(mask['mask_patch']), encoding='utf-8')

    # The plan carries the files it touches (``files``) alongside the entry/delete classification.
    heuristic_mask_plan = {**plan.to_dict(), 'files': mask['files']}
    return {
        'heuristic_mask_plan': heuristic_mask_plan,
        'heuristic_mask_patch': mask['mask_patch'],
        'heuristic_gold_patch': mask['gold_patch'],
        'warnings': [*plan.warnings, *mask['warnings']],
    }


async def run_mask(
    *,
    graph_path: str,
    repo_dir: str,
    feature: list[str],
    heuristic_patch_path: str,
) -> dict[str, Any]:
    """Sandbox-side Mask draft for one feature (deterministic; no LLM).

    *feature* is the Select agentic pass's ``feature_node_ids`` (function ids). The agent *refine*
    pass over the draft patch and the test-oracle validation run host-side on top
    of this (like Select's agentic pass), so they are not invoked here.
    """
    start = time.time()
    result = await asyncio.to_thread(
        _run_mask_blocking,
        graph_path=graph_path,
        repo_dir=repo_dir,
        feature=feature,
        heuristic_patch_path=heuristic_patch_path,
    )
    result['execution_time'] = time.time() - start
    plan = result.get('heuristic_mask_plan') or {}
    logger.debug(
        'swe_any run_mask done: %s entries / %s deletes; %s files',
        len(plan.get('entries') or []),
        len(plan.get('deletes') or []),
        len(plan.get('files') or []),
    )
    return result


def _run_eval_blocking(
    *,
    graph_path: str,
    repo_dir: str,
    tests: list[str],
    heuristic_mask_patch_path: str,
    heuristic_gold_patch_path: str,
) -> dict[str, Any]:
    """Build the DRAFT native-test deletion patch over *tests* (no LLM).

    Mirrors ``run_mask``: *tests* are the feature's candidate native test cases, and the draft
    ``heuristic_mask_patch`` deletes ALL of them from the source (its inverse ``heuristic_gold_patch``
    puts them back verbatim). Writes BOTH drafts to the sandbox — the mask at *heuristic_mask_patch_path*
    and the gold at *heuristic_gold_patch_path* — so the Eval agent can read the two test-side drafts by
    path (symmetric with the feature mask/gold); the bodies are also returned for diagnostics.
    Line-based off ``enclosing_range`` via the shared ``delete_functions_patch``;
    unresolvable ids / files land in ``warnings``.
    """
    from nexus.extensions.tasks.swe_any.utils.graph import read_json
    from nexus.extensions.tasks.swe_any.utils.source_edit import delete_functions_patch

    graph = read_json(graph_path)
    out = delete_functions_patch(repo_dir, graph, tests)

    # Both drafts land in the sandbox (the agent reads them by path, like run_mask); the bodies are
    # also returned via the invoke result (the host needs them for the choose/apply-check/unified row).
    for path, body in ((heuristic_mask_patch_path, out['patch']), (heuristic_gold_patch_path, out['reverse'])):
        patch_file = Path(path)
        patch_file.parent.mkdir(parents=True, exist_ok=True)
        patch_file.write_text(str(body), encoding='utf-8')

    return {
        'heuristic_tests': tests,
        'heuristic_mask_patch': out['patch'],
        'heuristic_gold_patch': out['reverse'],
        'heuristic_test_files': out['files'],
        'warnings': out['warnings'],
    }


async def run_eval(
    *,
    graph_path: str,
    repo_dir: str,
    tests: list[str],
    heuristic_mask_patch_path: str,
    heuristic_gold_patch_path: str,
) -> dict[str, Any]:
    """Sandbox-side DRAFT native-test deletion builder (no LLM), mirroring ``run_mask``.

    Runs BEFORE the Eval agent: builds a deletion patch over the feature's candidate *tests* and writes
    BOTH raw drafts — the mask at *heuristic_mask_patch_path*, its inverse gold at
    *heuristic_gold_patch_path* — for the agent to read by path. Returns the draft
    ``heuristic_mask_patch`` / ``heuristic_gold_patch`` for diagnostics and author/refine context.
    These drafts are never accepted directly as the Eval oracle.
    """
    start = time.time()
    result = await asyncio.to_thread(
        _run_eval_blocking,
        graph_path=graph_path,
        repo_dir=repo_dir,
        tests=tests,
        heuristic_mask_patch_path=heuristic_mask_patch_path,
        heuristic_gold_patch_path=heuristic_gold_patch_path,
    )
    result['execution_time'] = time.time() - start
    logger.debug(
        'swe_any run_eval draft done: deletes %s candidate tests across %s files',
        len(result.get('heuristic_tests') or []),
        len(result.get('heuristic_test_files') or []),
    )
    return result


def _git_env_and_bin() -> tuple[dict[str, str], str]:
    """Subprocess env + git executable for sandbox repo ops.

    Mirrors swe/eval.py: strip the nix ``LD_LIBRARY_PATH`` and make sure the runtime's
    git is reachable even when the baked image ships none on PATH.
    """
    from nexus.utils.shell import get_os_env_without_ld_library_path, resolve_first_existing_path

    env = get_os_env_without_ld_library_path()
    nix_bin = '/nix/runtime/nix-env/bin'
    current_path = env.get('PATH', '')
    if nix_bin not in current_path:
        env['PATH'] = f'{current_path}:{nix_bin}' if current_path else nix_bin
    git_bin = resolve_first_existing_path(
        ['/nix/runtime/nix-env/bin/git', '/usr/bin/git', 'git'], fallback_to_last=True
    )
    return env, git_bin


def _reset(
    repo_dir: str,
    *,
    env: dict[str, str],
    git_bin: str,
    timeout: float,
    base_commit: str = '',
) -> tuple[bool, str]:
    """Reset to *base_commit* (or HEAD) plus ``git clean -fd`` without raising.

    Returns ``(ok, error)`` so patch/grade operations can fail closed instead of continuing against
    an unknown tree. The status-reporting variant is :func:`_reset_repo_blocking`.
    """
    reset_cmd = [git_bin, 'reset', '--hard', base_commit] if base_commit else [git_bin, 'reset', '--hard']
    for cmd in (reset_cmd, [git_bin, 'clean', '-fd']):
        try:
            proc = subprocess.run(cmd, cwd=repo_dir, capture_output=True, text=True, env=env, timeout=timeout)
        except Exception as exc:  # noqa: BLE001 - callers receive the git-level failure as data
            return False, f'{type(exc).__name__}: {exc}'
        if proc.returncode != 0:
            return False, (proc.stderr or proc.stdout or '').strip()
    return True, ''


def _apply_patch(repo_dir: str, patch: str, *, env: dict[str, str], git_bin: str, timeout: float) -> tuple[bool, str]:
    """``git apply --ignore-whitespace`` one patch from a temp file → ``(ok, error)``.

    When a patch wants to create files that already exist, retry once after removing every such
    conflicting regular file. The originals are restored if the retry also fails.
    """
    with tempfile.NamedTemporaryFile('w', suffix='.diff', delete=False, encoding='utf-8') as fh:
        fh.write(patch if patch.endswith('\n') else patch + '\n')
        patch_file = fh.name
    cmd = [git_bin, 'apply', '--ignore-whitespace', patch_file]
    try:
        proc = subprocess.run(
            cmd, cwd=repo_dir, capture_output=True, text=True, encoding='utf-8', timeout=timeout, env=env
        )
        if proc.returncode == 0:
            return True, ''

        first_error = (proc.stderr or proc.stdout or '').strip()
        from nexus.extensions.tasks.swe_any.utils.source_edit import patch_files

        root = Path(repo_dir).resolve()
        backups: list[tuple[Path, bytes, int]] = []
        for relative in sorted(patch_files(patch)):
            if f'{relative}: already exists in working directory' not in first_error:
                continue
            path = root / relative
            try:
                path.resolve(strict=False).relative_to(root)
            except (OSError, ValueError):
                continue
            if path.is_symlink() or not path.is_file():
                continue
            backups.append((path, path.read_bytes(), path.stat().st_mode & 0o777))

        if not backups:
            return False, first_error

        retry_succeeded = False
        try:
            for path, _content, _mode in backups:
                path.unlink()
            retry = subprocess.run(
                cmd, cwd=repo_dir, capture_output=True, text=True, encoding='utf-8', timeout=timeout, env=env
            )
            if retry.returncode == 0:
                retry_succeeded = True
                logger.warning(
                    'swe_any git apply recovered by replacing conflicting files: %s',
                    [str(path.relative_to(root)) for path, _content, _mode in backups],
                )
                return True, ''
            retry_error = (retry.stderr or retry.stdout or '').strip()
            return False, f'{first_error}; retry after deleting conflicting files failed: {retry_error}'
        finally:
            if not retry_succeeded:
                for path, content, mode in backups:
                    with contextlib.suppress(OSError):
                        path.unlink()
                    path.parent.mkdir(parents=True, exist_ok=True)
                    path.write_bytes(content)
                    os.chmod(path, mode)
    except Exception as exc:  # noqa: BLE001 - a git-level failure is data, not a crash
        return False, f'{type(exc).__name__}: {exc}'
    finally:
        with contextlib.suppress(OSError):
            os.unlink(patch_file)


def _apply_steps_blocking(
    *,
    repo_dir: str,
    steps: list[list[str]],
    reset_before: bool,
    reset_after: bool,
    timeout: float,
    base_commit: str = '',
) -> dict[str, dict[str, Any]]:
    """Apply ordered ``[name, patch]`` *steps*, each on top of the previous, recording per-step status.

    The single core that both the validate-and-discard (:func:`run_apply_sequence`) and the
    apply-and-keep (:func:`run_apply`) targets compose from — they differ only in the reset bookends
    (``reset_before`` / ``reset_after``). An empty patch is a no-op (``ok=True, empty=True``); the
    first failing step stops the rest (``skipped=True``) since the accumulated state cannot proceed.
    Returns ``{name: {ok, empty, skipped, error}}`` in step order (failures recorded, not raised).
    """
    env, git_bin = _git_env_and_bin()
    results: dict[str, dict[str, Any]] = {}
    if reset_before:
        reset_ok, reset_error = _reset(
            repo_dir,
            env=env,
            git_bin=git_bin,
            timeout=timeout,
            base_commit=base_commit,
        )
        if not reset_ok:
            for index, (name, _patch) in enumerate(steps):
                results[name] = {
                    'ok': False,
                    'empty': False,
                    'skipped': index > 0,
                    'error': f'base reset failed: {reset_error}',
                }
            if not steps:
                results['_reset'] = {
                    'ok': False,
                    'empty': False,
                    'skipped': False,
                    'error': f'base reset failed: {reset_error}',
                }
            return results
    stopped = False
    try:
        for name, patch in steps:
            if stopped:
                results[name] = {'ok': False, 'empty': False, 'skipped': True, 'error': 'a prior step did not apply'}
            elif not (patch or '').strip():
                results[name] = {'ok': True, 'empty': True, 'skipped': False, 'error': ''}
            else:
                ok, error = _apply_patch(repo_dir, patch, env=env, git_bin=git_bin, timeout=timeout)
                results[name] = {'ok': ok, 'empty': False, 'skipped': False, 'error': error}
                stopped = not ok
    finally:
        if reset_after:
            _reset(repo_dir, env=env, git_bin=git_bin, timeout=timeout, base_commit=base_commit)
    return results


async def run_apply_sequence(
    *, repo_dir: str, steps: list[list[str]], timeout: float = 120.0, base_commit: str = ''
) -> dict[str, Any]:
    """VALIDATE that patches apply CUMULATIVELY (each on the previous), then reset — no state kept.

    Deterministic, no LLM. Used by host Mask (mask → +gold round-trip) and Eval (test_mask →
    +restore → +extra, the eval-time order) to reject a patch set that would not stack. *steps* is a
    list of ``[name, patch]``. Returns ``{results: {name: {ok, empty, skipped, error}}, all_ok}``.
    """
    results = await asyncio.to_thread(
        _apply_steps_blocking,
        repo_dir=repo_dir,
        steps=steps,
        reset_before=True,
        reset_after=True,
        timeout=timeout,
        base_commit=base_commit,
    )
    all_ok = all(r['ok'] for r in results.values())
    logger.info('swe_any run_apply_sequence done: %s', {name: r['ok'] for name, r in results.items()})
    return {'results': results, 'all_ok': all_ok}


async def run_apply(
    *, repo_dir: str, steps: list[list[str]], timeout: float = 120.0, base_commit: str = ''
) -> dict[str, Any]:
    """Reset to pristine, apply patches, and LEAVE them applied (no trailing reset).

    Deterministic, no LLM. The persist counterpart of :func:`run_apply_sequence`: reset to pristine,
    then git-apply each step in order and keep the result — Issue's fully-masked-state setup (apply
    feature_mask + test_mask, then let the agent browse that tree). *steps* is a list of
    ``[name, patch]``. Returns ``{results, all_ok}``.
    """
    results = await asyncio.to_thread(
        _apply_steps_blocking,
        repo_dir=repo_dir,
        steps=steps,
        reset_before=True,
        reset_after=False,
        timeout=timeout,
        base_commit=base_commit,
    )
    all_ok = all(r['ok'] for r in results.values())
    logger.info('swe_any run_apply done: %s', {name: r['ok'] for name, r in results.items()})
    return {'results': results, 'all_ok': all_ok}


def _reverse_patch_blocking(*, repo_dir: str, patch: str, timeout: float, base_commit: str = '') -> dict[str, Any]:
    """git-canonical inverse of *patch*: reset to pristine, ``git apply`` it, capture ``git diff -R``.

    Runs where the source lives (sandbox), so the reverse is produced by git itself and is correct
    for every case a text-only reverser struggles with (whole-file create/delete, ``\\ No newline``,
    content that starts with ``--`` / ``++``, multi-file with or without ``diff --git``). ``git add
    -A`` before the diff so created/deleted files show up in the inverse. Uses the same
    ``--ignore-whitespace`` as the apply-check so a patch that passes there also reverses here.

    Returns ``{ok, reverse, error}``: ``ok=False`` with an empty ``reverse`` when the patch does not
    apply (a soft outcome — the host maps it to "no usable agentic gold/restore" and the apply-check
    rejects that candidate). An empty patch reverses to empty. Always resets the tree at the end.
    """
    if not (patch or '').strip():
        return {'ok': True, 'reverse': '', 'error': ''}

    env, git_bin = _git_env_and_bin()
    reset_ok, reset_error = _reset(
        repo_dir,
        env=env,
        git_bin=git_bin,
        timeout=timeout,
        base_commit=base_commit,
    )
    if not reset_ok:
        return {'ok': False, 'reverse': '', 'error': f'base reset failed: {reset_error}'}
    ok, error = _apply_patch(repo_dir, patch, env=env, git_bin=git_bin, timeout=timeout)
    if not ok:
        _reset(repo_dir, env=env, git_bin=git_bin, timeout=timeout, base_commit=base_commit)
        return {'ok': False, 'reverse': '', 'error': error}
    try:
        # Stage everything (incl. created/deleted files) so the inverse captures whole-file changes,
        # then emit HEAD<-index reversed = the canonical inverse patch.
        subprocess.run([git_bin, 'add', '-A'], cwd=repo_dir, capture_output=True, text=True, env=env, timeout=timeout)
        diff = subprocess.run(
            [git_bin, 'diff', '--cached', '-R', '--no-color'],
            cwd=repo_dir,
            capture_output=True,
            text=True,
            encoding='utf-8',
            timeout=timeout,
            env=env,
        )
        if diff.returncode != 0:
            return {'ok': False, 'reverse': '', 'error': (diff.stderr or '').strip()}
        return {'ok': True, 'reverse': diff.stdout, 'error': ''}
    except Exception as exc:  # noqa: BLE001 - a git-level failure is data, not a crash
        return {'ok': False, 'reverse': '', 'error': f'{type(exc).__name__}: {exc}'}
    finally:
        _reset(repo_dir, env=env, git_bin=git_bin, timeout=timeout, base_commit=base_commit)


async def run_reverse_patch(
    *, repo_dir: str, patch: str, timeout: float = 120.0, base_commit: str = ''
) -> dict[str, Any]:
    """Sandbox-side git-canonical inverse of *patch* (deterministic, no LLM).

    Derives the gold/restore patch from an AGENT-authored mask patch (the heuristic side already
    gets its inverse from ``build_edit_patch``). See :func:`_reverse_patch_blocking`.
    """
    result = await asyncio.to_thread(
        _reverse_patch_blocking,
        repo_dir=repo_dir,
        patch=patch,
        timeout=timeout,
        base_commit=base_commit,
    )
    logger.info(
        'swe_any run_reverse_patch done: ok=%s reverse_len=%s', result.get('ok'), len(result.get('reverse') or '')
    )
    return result


_GRADE_LOG_CAP = 131072  # 128 KiB per phase log; keep head+tail, drop the middle when larger.
_GRADE_RESULTS_CAP = 8 * 1024 * 1024


def _cap_log(text: str, limit: int = _GRADE_LOG_CAP) -> str:
    """Head+tail truncate a grade log so a huge failing-test dump does not blow up invoke/disk."""
    if len(text) <= limit:
        return text
    half = limit // 2
    return f'{text[:half]}\n... [swe_any: {len(text) - 2 * half} chars omitted] ...\n{text[-half:]}'


def _run_structured_test_runner(
    *,
    repo_dir: str,
    runner_script: str,
    test_files: list[str],
    timeout: float,
    env: dict[str, str],
) -> dict[str, Any]:
    """Run an Eval-authored shell adapter and validate its JSONL result file."""
    from nexus.extensions.tasks.swe_any.utils.test_runner import RUNNER_PROTOCOL_VERSION, parse_runner_results

    runner_path = result_path = ''
    try:
        with tempfile.NamedTemporaryFile('w', suffix='.sh', delete=False, encoding='utf-8') as runner_file:
            runner_file.write(runner_script if runner_script.endswith('\n') else runner_script + '\n')
            runner_path = runner_file.name
        os.chmod(runner_path, 0o700)
        result_fd, result_path = tempfile.mkstemp(suffix='.jsonl')
        os.close(result_fd)
        os.unlink(result_path)  # The protocol requires the runner to create/overwrite it.

        syntax = subprocess.run(
            ['bash', '-n', runner_path],
            cwd=repo_dir,
            capture_output=True,
            text=True,
            encoding='utf-8',
            errors='replace',
            timeout=timeout,
            env=env,
        )
        if syntax.returncode != 0:
            return {
                'ok': False,
                'error': f'test_runner.sh syntax error: {(syntax.stderr or syntax.stdout).strip()}',
                'error_kind': 'runner_protocol_error',
                'log': _cap_log((syntax.stdout or '') + (syntax.stderr or '')),
            }

        runner_env = {
            **env,
            'SWE_ANY_REPO_DIR': repo_dir,
            'SWE_ANY_RESULTS_PATH': result_path,
            'SWE_ANY_RUNNER_PROTOCOL': str(RUNNER_PROTOCOL_VERSION),
            'SWE_ANY_TEST_FILES_JSON': json.dumps(test_files, separators=(',', ':')),
        }
        proc = subprocess.run(
            ['bash', runner_path],
            cwd=repo_dir,
            capture_output=True,
            text=True,
            encoding='utf-8',
            errors='replace',
            timeout=timeout,
            env=runner_env,
        )
        log = _cap_log((proc.stdout or '') + (proc.stderr or ''))
        try:
            if Path(result_path).stat().st_size > _GRADE_RESULTS_CAP:
                raise OSError(f'result file exceeds {_GRADE_RESULTS_CAP} bytes')
            result_text = Path(result_path).read_text(encoding='utf-8')
        except OSError as exc:
            return {
                'ok': False,
                'error': f'test_runner.sh did not write SWE_ANY_RESULTS_PATH: {exc}',
                'error_kind': 'runner_protocol_error',
                'log': log,
                'returncode': proc.returncode,
            }
        try:
            protocol = parse_runner_results(result_text)
        except ValueError as exc:
            return {
                'ok': False,
                'error': f'invalid test runner results: {exc}',
                'error_kind': 'runner_protocol_error',
                'log': log,
                'returncode': proc.returncode,
            }
        summary = protocol['summary']
        duplicate_test_ids = protocol.get('duplicate_test_ids') or []
        if duplicate_test_ids:
            summary = {
                **summary,
                'duplicate_test_ids': duplicate_test_ids,
                'unique_count': len(protocol['tests']),
            }
        if summary['status'] == 'runner_error':
            error_kind = str(summary.get('error_kind') or 'runner_error')
            message = str(summary.get('message') or 'test runner reported an execution error')
            return {
                'ok': False,
                'error': message,
                'error_kind': error_kind,
                'log': log,
                'returncode': proc.returncode,
                'test_statuses': protocol['tests'],
                'runner_summary': summary,
            }
        if proc.returncode != 0:
            return {
                'ok': False,
                'error': f'test_runner.sh exited {proc.returncode} after a completed summary',
                'error_kind': 'runner_exit_error',
                'log': log,
                'returncode': proc.returncode,
                'test_statuses': protocol['tests'],
                'runner_summary': summary,
            }
        return {
            'ok': True,
            'error': '',
            'log': log,
            'returncode': proc.returncode,
            'test_statuses': protocol['tests'],
            'runner_summary': summary,
        }
    finally:
        for path in (runner_path, result_path):
            if path:
                with contextlib.suppress(OSError):
                    os.unlink(path)


def _run_grade_phase(
    *,
    repo_dir: str,
    steps: list[list[str]],
    test_runner_script: str,
    test_files: list[str],
    env: dict[str, str],
    git_bin: str,
    timeout: float,
    base_commit: str = '',
) -> dict[str, Any]:
    """Apply *steps*, run the structured runner, then reset.

    Only targets that exist after every patch in *steps* are passed to the runner. This matters when
    ``test_mask_patch`` deletes an original test and ``test_gold_patch`` creates a generalized oracle
    under another path: asking pytest for both paths aborts collection before the real oracle runs.

    The required *test_runner_script* returns validated ``test_statuses``.
    """
    reset_ok, reset_error = _reset(
        repo_dir,
        env=env,
        git_bin=git_bin,
        timeout=timeout,
        base_commit=base_commit,
    )
    if not reset_ok:
        return {'ok': False, 'error': f'base reset failed: {reset_error}', 'log': ''}
    try:
        for name, patch in steps:
            if not (patch or '').strip():
                continue
            ok, error = _apply_patch(repo_dir, patch, env=env, git_bin=git_bin, timeout=timeout)
            if not ok:
                return {'ok': False, 'error': f'{name}: {error}', 'log': ''}
        if not test_files:
            # No target = pytest collects the whole repo; refuse rather than grade the wrong suite.
            return {'ok': False, 'error': 'no test_files to grade', 'log': ''}

        root = Path(repo_dir).resolve()
        existing: list[str] = []
        missing: list[str] = []
        for file in test_files:
            candidate = (root / file).resolve()
            try:
                candidate.relative_to(root)
            except ValueError:
                missing.append(file)
                continue
            (existing if candidate.is_file() else missing).append(file)
        if not existing:
            return {
                'ok': False,
                'error': f'no existing test_files to grade: {missing}',
                'log': '',
                'test_files': [],
                'missing_test_files': missing,
            }

        # Run the tests in the IMAGE's own environment, not the nix runtime's: strip the nix bin from
        # PATH (so the container's python3/pytest win) and use a NON-login shell so a profile can't
        # re-prepend the nix bin. Mirrors swe/infer.py::_run_init_cmds.
        from nexus.utils.shell import get_clean_sandbox_env

        test_env = get_clean_sandbox_env()
        test_env.update({'FORCE_COLOR': '0', 'NO_COLOR': '1', 'PY_COLORS': '0', 'TERM': 'dumb'})
        if not test_runner_script.strip():
            return {'ok': False, 'error': 'missing test_runner_script', 'log': ''}
        result = _run_structured_test_runner(
            repo_dir=repo_dir,
            runner_script=test_runner_script,
            test_files=existing,
            timeout=timeout,
            env=test_env,
        )
        return {**result, 'test_files': existing, 'missing_test_files': missing}
    except Exception as exc:  # noqa: BLE001 - a run failure is data, not a crash
        return {'ok': False, 'error': f'{type(exc).__name__}: {exc}', 'log': ''}
    finally:
        _reset(repo_dir, env=env, git_bin=git_bin, timeout=timeout, base_commit=base_commit)


def _run_grade_blocking(
    *,
    repo_dir: str,
    pregold_steps: list[list[str]],
    postgold_steps: list[list[str]],
    test_runner_script: str,
    test_files: list[str],
    timeout: float,
    base_commit: str = '',
) -> dict[str, Any]:
    """Execution-grade one instance: run the oracle tests pre-gold (buggy) and post-gold (fixed).

    *pregold_steps* stack the bug + revealed oracle tests (feature masked → tests should FAIL);
    *postgold_steps* additionally restore the implementation (tests should PASS). Both execute the
    same structured runner.
    """
    env, git_bin = _git_env_and_bin()
    pregold = _run_grade_phase(
        repo_dir=repo_dir,
        steps=pregold_steps,
        test_runner_script=test_runner_script,
        test_files=test_files,
        env=env,
        git_bin=git_bin,
        timeout=timeout,
        base_commit=base_commit,
    )
    postgold = _run_grade_phase(
        repo_dir=repo_dir,
        steps=postgold_steps,
        test_runner_script=test_runner_script,
        test_files=test_files,
        env=env,
        git_bin=git_bin,
        timeout=timeout,
        base_commit=base_commit,
    )
    return {'pregold': pregold, 'postgold': postgold}


async def run_grade(
    *,
    repo_dir: str,
    pregold_steps: list[list[str]],
    postgold_steps: list[list[str]],
    test_runner_script: str,
    test_files: list[str],
    timeout: float = 1800.0,
    base_commit: str = '',
) -> dict[str, Any]:
    """Sandbox-side execution grading for Export (no LLM). See :func:`_run_grade_blocking`.

    Runs the oracle tests twice (pre-gold buggy / post-gold fixed) so the host can derive and
    validate FAIL_TO_PASS / PASS_TO_PASS by execution. Rows return structured status maps.
    """
    result = await asyncio.to_thread(
        _run_grade_blocking,
        repo_dir=repo_dir,
        pregold_steps=pregold_steps,
        postgold_steps=postgold_steps,
        test_runner_script=test_runner_script,
        test_files=test_files,
        timeout=timeout,
        base_commit=base_commit,
    )
    logger.info(
        'swe_any run_grade done: pregold_ok=%s postgold_ok=%s',
        result.get('pregold', {}).get('ok'),
        result.get('postgold', {}).get('ok'),
    )
    return result


def _reset_repo_blocking(*, repo_dir: str, timeout: float, base_commit: str = '') -> dict[str, Any]:
    """Reset *repo_dir* to *base_commit* (or HEAD), clean it, and verify the observed HEAD.

    Ignored files (venvs, build output) are kept (``-fd``, not ``-fdx``), matching swe_cosmos.
    Passing an explicit commit also undoes agent commits instead of trusting mutable image HEAD.
    A missing commit or non-git tree is reported as ``reset=False``.
    """
    env, git_bin = _git_env_and_bin()

    inside = subprocess.run(
        [git_bin, 'rev-parse', '--is-inside-work-tree'],
        cwd=repo_dir,
        capture_output=True,
        text=True,
        env=env,
        timeout=timeout,
    )
    if inside.returncode != 0 or inside.stdout.strip() != 'true':
        return {'reset': False, 'reason': 'not a git work tree', 'error': (inside.stderr or '').strip()}

    reset_cmd = [git_bin, 'reset', '--hard', base_commit] if base_commit else [git_bin, 'reset', '--hard']
    for cmd in (reset_cmd, [git_bin, 'clean', '-fd']):
        proc = subprocess.run(cmd, cwd=repo_dir, capture_output=True, text=True, env=env, timeout=timeout)
        if proc.returncode != 0:
            return {
                'reset': False,
                'reason': f'{" ".join(cmd[1:])} failed',
                'error': (proc.stderr or proc.stdout or '').strip(),
                'base_commit': base_commit,
            }
    head = subprocess.run(
        [git_bin, 'rev-parse', 'HEAD'],
        cwd=repo_dir,
        capture_output=True,
        text=True,
        env=env,
        timeout=timeout,
    )
    if head.returncode != 0:
        return {
            'reset': False,
            'reason': 'could not observe HEAD after reset',
            'error': (head.stderr or head.stdout or '').strip(),
            'base_commit': base_commit,
        }
    observed_head = head.stdout.strip()
    if base_commit:
        expected = subprocess.run(
            [git_bin, 'rev-parse', f'{base_commit}^{{commit}}'],
            cwd=repo_dir,
            capture_output=True,
            text=True,
            env=env,
            timeout=timeout,
        )
        expected_head = expected.stdout.strip()
        if expected.returncode != 0 or observed_head != expected_head:
            return {
                'reset': False,
                'reason': 'observed HEAD does not match base_commit',
                'error': (expected.stderr or '').strip(),
                'base_commit': base_commit,
                'head': observed_head,
            }
    return {'reset': True, 'reason': '', 'error': '', 'base_commit': base_commit, 'head': observed_head}


async def reset_repo(*, repo_dir: str, timeout: float = 300.0, base_commit: str = '') -> dict[str, Any]:
    """Reset the sandbox repo to *base_commit* between features/slices (no LLM).

    ``git reset --hard <base_commit>`` + ``git clean -fd`` means image HEAD drift and agent commits
    cannot leak into a stage. Returns ``{reset, reason, error, base_commit, head}`` and never raises
    for a git-level failure; the host decides whether a given reset is strict or best-effort.
    """
    result = await asyncio.to_thread(
        _reset_repo_blocking,
        repo_dir=repo_dir,
        timeout=timeout,
        base_commit=base_commit,
    )
    logger.info('swe_any reset_repo %s: reset=%s %s', repo_dir, result.get('reset'), result.get('reason') or '')
    return result
