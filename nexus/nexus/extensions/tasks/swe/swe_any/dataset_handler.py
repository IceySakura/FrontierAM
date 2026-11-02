"""Dataset handler for swe_any (feature-implementation) instances.

Reuses the swe_task orchestration (sandbox lifecycle, model-patch apply, scoring) and ports only the
swe_any evaluation core. The instance image holds the CORRECT repo (the feature is implemented and
its native tests exist); the runtime first resets to the declared ``base_commit``, then reconstructs
the "re-implement it" hole by applying, each on top of the last:

    feature_mask_patch   (stub the feature's entries, delete its helpers)
    test_mask_patch      (delete the native oracle tests)

For inference we then drop git history via GIT_REINIT_SCRIPT (a fresh single-commit repo) so the
agent cannot recover the reference implementation by walking back revisions. For evaluation we
commit the hole and then ``test_gold_patch``; protected test paths are restored to the hole in the
work tree while HEAD retains the trusted gold state. After the framework applies the model patch,
``get_eval_script`` restores those paths from HEAD and invokes Eval's grade-only structured
``test_runner_script``.
Rows without a supported structured runner are rejected. ``get_eval_report_map`` marks the instance
resolved iff every FAIL_TO_PASS/PASS_TO_PASS passes.

The buggy state is applied at runtime rather than baked into a Docker image (mirrors SWE-Smith),
so the same published SWE-Env image serves every masked feature it produced.
"""

from __future__ import annotations

import json
import logging
import shlex

from pathlib import Path, PurePosixPath
from typing import Any

from nexus.extensions.tasks.swe.constants import SWEbenchInstance
from nexus.extensions.tasks.swe.infer import (
    apply_patch_and_commit,
    remove_git_dir,
    run_git_reinit,
    run_init_cmds_in_runtime,
)
from nexus.extensions.tasks.swe.swe_any.utils import (
    RESULTS_END,
    RESULTS_START,
    RUNNER_PROTOCOL_VERSION,
    RunnerProtocolError,
    extract_runner_results,
    patch_files,
)
from nexus.extensions.tasks.swe.swe_bench.constants import END_TEST_OUTPUT, START_TEST_OUTPUT
from nexus.runtime import Runtime


logger = logging.getLogger(__name__)

_RUNNER_PATH = '/tmp/swe_any_test_runner.sh'
_RUNNER_RESULTS_PATH = '/tmp/swe_any_test_results.jsonl'


def _workdir(instance: SWEbenchInstance) -> str:
    return instance.get('workdir') or '/testbed'


def _as_list(value: Any) -> list[str]:
    """FAIL_TO_PASS / PASS_TO_PASS may be a JSON string or a list; normalize to a list of ids."""
    if value is None:
        return []
    if isinstance(value, str):
        try:
            value = json.loads(value)
        except json.JSONDecodeError:
            return []
    return [str(v) for v in value] if isinstance(value, list) else []


def _base_commit(instance: SWEbenchInstance) -> str:
    """Return the immutable source revision that Export recorded for this instance."""
    base = instance.get('base') or {}
    base_commit = str(instance.get('base_commit') or base.get('sha') or '')
    if not base_commit:
        raise ValueError(f'SWE-Any instance {instance.get("instance_id")!r} is missing base_commit')
    return base_commit


async def _prepare_repo(instance: SWEbenchInstance, runtime: Runtime, workdir: str) -> None:
    """Reset to the declared source revision and configure commits used by inference reinit."""
    base_commit = _base_commit(instance)
    await run_init_cmds_in_runtime(
        runtime,
        [
            f'git config --global --add safe.directory {shlex.quote(workdir)}',
            "git config --local user.name 'swe-any'",
            "git config --local user.email 'swe-any@anonymous.com'",
            'git config --local commit.gpgsign false',
            f'git reset --hard {shlex.quote(base_commit)} && git clean -fd',
        ],
        working_dir=workdir,
    )


async def _apply_hole(
    instance: SWEbenchInstance,
    runtime: Runtime,
    *,
    commit: bool = False,
) -> None:
    """Apply feature_mask + test_mask, optionally committing each patch."""
    workdir = _workdir(instance)
    for name, key in (('feature_mask', 'feature_mask_patch'), ('test_mask', 'test_mask_patch')):
        patch = str(instance.get(key) or '')
        if patch.strip():
            # Exported JSONL rows may omit the patch's final newline. Unified-diff hunk counts
            # remain correct, but git apply treats EOF immediately after the last hunk line as
            # a corrupt patch. Normalize only the transport terminator; patch content is unchanged.
            if not patch.endswith('\n'):
                patch += '\n'
            message = f'Apply SWE-Any {name.replace("_", " ")}' if commit else None
            await apply_patch_and_commit(runtime, patch, f'/tmp/swe_any_{name}.diff', workdir, message)


def _protected_test_paths(instance: SWEbenchInstance) -> list[str]:
    """Return validated repo-relative paths whose solver edits must not reach grading."""
    paths: set[str] = set()
    for key in ('test_mask_patch', 'test_gold_patch'):
        paths.update(patch_files(str(instance.get(key) or '')))
    paths.update(str(path) for path in (instance.get('test_files') or []))

    safe: set[str] = set()
    for raw in paths:
        path = PurePosixPath(raw)
        normalized = path.as_posix()
        if (
            not raw
            or '\x00' in raw
            or '\n' in raw
            or '\r' in raw
            or path.is_absolute()
            or '..' in path.parts
            or normalized in ('', '.')
        ):
            raise ValueError(f'Unsafe SWE-Any protected test path: {raw!r}')
        safe.add(normalized)
    return sorted(safe)


def _restore_protected_test_commands(instance: SWEbenchInstance, source: str = 'HEAD') -> list[str]:
    """Restore oracle paths from one trusted commit without changing unrelated implementation files."""
    commands: list[str] = []
    quoted_source = shlex.quote(source)
    for path in _protected_test_paths(instance):
        quoted_path = shlex.quote(path)
        quoted_object = shlex.quote(f'{source}:{path}')
        # The source may intentionally omit a test file. Remove both its index entry and work-tree
        # copy first, then restore it only when the trusted source actually contains it.
        commands.extend([
            f'git rm -rf --cached --ignore-unmatch -- {quoted_path}',
            f'rm -rf -- {quoted_path}',
            (f'if git cat-file -e {quoted_object} 2>/dev/null; then git checkout {quoted_source} -- {quoted_path}; fi'),
        ])
    return commands


# =============================================================================
# DatasetHandler interface (module form; adapted via ModuleDatasetHandler)
# =============================================================================


def get_problem_statement(instance: SWEbenchInstance) -> str:
    return instance.get('problem_statement') or ''


def get_repo_language(instance: SWEbenchInstance) -> str | None:
    # swe_any is Python-only for now; the value drives patch extraction's file extension.
    return 'py'


async def infer_runtime_init(instance: SWEbenchInstance, runtime: Runtime, **kwargs: Any) -> None:
    """Reconstruct the masked hole, then drop git history so the fix can't be recovered from revisions."""
    workdir = _workdir(instance)
    await _prepare_repo(instance, runtime, workdir)
    await _apply_hole(instance, runtime)

    if kwargs.get('remove_git_dir', False):
        await remove_git_dir(runtime, workdir)
        return

    runtime_base_commit = await run_git_reinit(runtime, workdir)
    instance['runtime_base_commit'] = runtime_base_commit
    logger.debug('swe_any updated runtime_base_commit: %s', runtime_base_commit)


async def eval_runtime_init(instance: SWEbenchInstance, runtime: Runtime) -> None:
    """Commit gold tests in HEAD, then put their work-tree paths back into the masked hole state."""
    workdir = _workdir(instance)
    await _prepare_repo(instance, runtime, workdir)
    await _apply_hole(instance, runtime, commit=True)
    test_gold_patch = str(instance.get('test_gold_patch') or '')
    if test_gold_patch.strip():
        if not test_gold_patch.endswith('\n'):
            test_gold_patch += '\n'
        await apply_patch_and_commit(
            runtime,
            test_gold_patch,
            '/tmp/swe_any_test_gold.diff',
            workdir,
            'Apply SWE-Any test gold',
        )
        await run_init_cmds_in_runtime(
            runtime,
            [
                # The shared commit helper tolerates commit failures; SWE-Any relies on HEAD for
                # restoration, so reject a dirty index/work tree before using that invariant.
                'test -z "$(git status --porcelain)"',
                *_restore_protected_test_commands(instance, source='HEAD^'),
            ],
            working_dir=workdir,
        )


def _heredoc_write(path: str, text: str) -> str:
    """Write arbitrary trusted text without shell expansion, choosing a collision-free delimiter."""
    delimiter = 'SWE_ANY_RUNNER_EOF'
    while delimiter in text:
        delimiter += '_'
    body = text if text.endswith('\n') else text + '\n'
    return f"cat > {shlex.quote(path)} <<'{delimiter}'\n{body}{delimiter}"


def get_eval_script(instance: SWEbenchInstance) -> str:
    """Compose the script run AFTER the model patch is applied: restore oracle tests, run them.

    :func:`eval_runtime_init` leaves the protected work-tree paths in the masked hole while HEAD
    retains the committed ``test_gold_patch``. The framework applies the model patch to that hole;
    this script then discards model edits to protected paths, restores their trusted HEAD versions,
    and runs the test command over ``test_files`` between the standard output markers.

    Every accepted row materializes and invokes ``instance['test_runner_script']`` with the same
    protocol and targets Export graded.
    """
    workdir = _workdir(instance)
    test_gold_patch = str(instance.get('test_gold_patch') or '')
    runner_script = str(instance.get('test_runner_script') or '').strip()
    if not runner_script:
        raise ValueError('SWE-Any instance is missing test_runner_script')
    runner_protocol = int(instance.get('test_runner_protocol') or 0)
    if runner_protocol != RUNNER_PROTOCOL_VERSION:
        raise ValueError(f'Unsupported SWE-Any test runner protocol: {runner_protocol}')
    test_files = [str(path) for path in (instance.get('test_files') or [])]
    restore_tests = _restore_protected_test_commands(instance) if test_gold_patch.strip() else []
    runner_setup = [
        _heredoc_write(_RUNNER_PATH, runner_script),
        f'chmod 700 {shlex.quote(_RUNNER_PATH)}',
    ]
    run_tests = '\n'.join([
        f'rm -f {shlex.quote(_RUNNER_RESULTS_PATH)}',
        (
            f'SWE_ANY_REPO_DIR={shlex.quote(workdir)} '
            f'SWE_ANY_RESULTS_PATH={shlex.quote(_RUNNER_RESULTS_PATH)} '
            f'SWE_ANY_RUNNER_PROTOCOL={runner_protocol} '
            f'SWE_ANY_TEST_FILES_JSON={shlex.quote(json.dumps(test_files, separators=(",", ":")))} '
            f'bash {shlex.quote(_RUNNER_PATH)}'
        ),
        'runner_rc=$?',
        # Keep shell xtrace out of the machine-readable JSONL block.
        'set +x',
        f"printf '%s\\n' {shlex.quote(RESULTS_START)}",
        f'if [ -f {shlex.quote(_RUNNER_RESULTS_PATH)} ]; then cat {shlex.quote(_RUNNER_RESULTS_PATH)}; fi',
        f"printf '%s\\n' {shlex.quote(RESULTS_END)}",
        'set -x',
        'if [ "$runner_rc" -ne 0 ]; then echo "test_runner.sh exited $runner_rc" >&2; fi',
    ])
    lines = [
        '#!/bin/bash',
        # No `set -u`: the conda `testbed` activate dereferences vars unset in a non-interactive
        # shell and would abort before pytest (swe_env strips it via postprocess_eval_script).
        # Keep setup fail-closed so a failed test restore cannot fall through to solver-authored
        # tests. Disable errexit only after the trusted grading state is ready, because ordinary
        # test failures must still be emitted and parsed.
        'set -exo pipefail',
        f'cd {shlex.quote(workdir)}',
        *runner_setup,
        *restore_tests,
        'set +e',
        f": '{START_TEST_OUTPUT}'",
        run_tests,
        f": '{END_TEST_OUTPUT}'",
    ]
    return '\n'.join(lines) + '\n'


def get_eval_report_map(
    instance: SWEbenchInstance,
    prediction: dict[str, str],
    test_log_path: str,
    include_tests_status: bool,
) -> dict[str, Any]:
    """Grade the run: resolved iff every FAIL_TO_PASS and PASS_TO_PASS test PASSED in the log."""
    instance_id = prediction.get('instance_id', instance.get('instance_id', ''))
    report: dict[str, Any] = {
        instance_id: {
            'patch_is_None': prediction.get('model_patch') is None,
            'patch_exists': bool(prediction.get('model_patch')),
            'patch_successfully_applied': True,
            'resolved': False,
        }
    }

    try:
        log = Path(test_log_path).read_text(encoding='utf-8')
    except OSError:
        log = ''
    runner_script = str(instance.get('test_runner_script') or '').strip()
    protocol_ok = False
    status_map: dict[str, str] = {}
    try:
        if not runner_script:
            raise RunnerProtocolError('missing test_runner_script')
        if int(instance.get('test_runner_protocol') or 0) != RUNNER_PROTOCOL_VERSION:
            raise RunnerProtocolError('unsupported test runner protocol version')
        protocol = extract_runner_results(log)
        protocol_ok = protocol['summary']['status'] == 'completed'
        status_map = dict(protocol['tests']) if protocol_ok else {}
    except RunnerProtocolError:
        pass

    f2p = _as_list(instance.get('FAIL_TO_PASS'))
    p2p = _as_list(instance.get('PASS_TO_PASS'))
    f2p_ok = [t for t in f2p if status_map.get(t) == 'PASSED']
    p2p_ok = [t for t in p2p if status_map.get(t) == 'PASSED']
    resolved = protocol_ok and bool(f2p) and len(f2p_ok) == len(f2p) and len(p2p_ok) == len(p2p)

    report[instance_id]['resolved'] = resolved
    report[instance_id]['status'] = 'resolved' if resolved else ('unresolved' if protocol_ok else 'runner_error')
    if include_tests_status:
        report[instance_id]['tests_status'] = {
            'FAIL_TO_PASS': {'success': f2p_ok, 'failure': [t for t in f2p if t not in f2p_ok]},
            'PASS_TO_PASS': {'success': p2p_ok, 'failure': [t for t in p2p if t not in p2p_ok]},
        }
    return report
