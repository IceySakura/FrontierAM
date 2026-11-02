"""One-feature execution for the SWE-Any Export stage."""

from __future__ import annotations

import json

from pathlib import Path
from typing import Any

from nexus.extensions.tasks.swe_any.stage.export.harbor import write_harbor_instance
from nexus.extensions.tasks.swe_any.stage.export.schema import (
    ExportTaskInput,
    build_swebench_instance,
    derive_structured_reports,
    empty_export_row,
    export_instance_id,
    export_unified_row,
)
from nexus.extensions.tasks.swe_any.utils import constants
from nexus.extensions.tasks.swe_any.utils.repo_git import grade_instance, reset_repo


async def run_export_feature(
    task_input: ExportTaskInput,
    runtime: Any,
    feature: dict[str, Any],
    *,
    instance: dict[str, Any],
    harbor_root: Path,
) -> tuple[dict[str, Any], dict[str, Any] | None, str | None]:
    """Grade and optionally emit one feature, returning unified row, instance row, and error."""
    index = feature['index']
    row = empty_export_row(index)
    row['instance_id'] = export_instance_id(index, instance)
    # A pristine start prevents a timed-out prior grade from leaking its tree state.
    await reset_repo(
        runtime,
        task_input.repo_dir,
        base_commit=task_input.base_commit,
    )

    bug = [
        ('feature_mask_patch', feature['feature_mask_patch']),
        ('test_mask_patch', feature['test_mask_patch']),
    ]
    gold_tests = [('test_gold_patch', feature['test_gold_patch'])]
    grade = await grade_instance(
        runtime,
        task_input.repo_dir,
        pregold_steps=[*bug, *gold_tests],
        postgold_steps=[*bug, ('feature_gold_patch', feature['feature_gold_patch']), *gold_tests],
        test_runner_script=str(feature['test_runner_script']),
        test_files=feature['test_files'],
        base_commit=task_input.base_commit,
        request_timeout=task_input.grade_timeout,
    )
    pregold, postgold = grade.get('pregold', {}), grade.get('postgold', {})
    missing_test_files = sorted(
        set(pregold.get('missing_test_files') or []) | set(postgold.get('missing_test_files') or [])
    )
    if missing_test_files:
        row['warnings'].append(f'ignored nonexistent test_gold targets: {missing_test_files}')

    # Grade logs are feature-owned diagnostics.
    host_slice_dir = Path(task_input.output_dir) / task_input.repo_name / constants.slice_dirname(index)
    host_slice_dir.mkdir(parents=True, exist_ok=True)
    (host_slice_dir / 'grade_pregold.log').write_text(str(pregold.get('log') or ''), encoding='utf-8')
    (host_slice_dir / 'grade_postgold.log').write_text(str(postgold.get('log') or ''), encoding='utf-8')
    for phase_name, phase in (('pregold', pregold), ('postgold', postgold)):
        if 'test_statuses' in phase or 'runner_summary' in phase:
            (host_slice_dir / f'grade_{phase_name}_results.json').write_text(
                json.dumps(
                    {
                        'tests': phase.get('test_statuses') or {},
                        'summary': phase.get('runner_summary'),
                    },
                    indent=2,
                    ensure_ascii=False,
                    sort_keys=True,
                )
                + '\n',
                encoding='utf-8',
            )

    instance_row: dict[str, Any] | None = None
    if not (pregold.get('ok') and postgold.get('ok')):
        setup_error = f'pregold={pregold.get("error") or "ok"}; postgold={postgold.get("error") or "ok"}'
        phase_error_kind = postgold.get('error_kind') or pregold.get('error_kind')
        if phase_error_kind:
            row['error_kind'] = str(phase_error_kind)
        elif 'no existing test_files to grade' in setup_error or 'no test_files to grade' in setup_error:
            row['error_kind'] = 'target_missing'
        else:
            row['error_kind'] = 'grade_setup_failed'
        row['error'] = f'grade setup failed: {setup_error}'
    else:
        postgold_statuses = dict(postgold.get('test_statuses') or {})
        if not any(status == 'PASSED' for status in postgold_statuses.values()):
            row['error_kind'] = 'no_postgold_pass'
            row['error'] = f'{row["error_kind"]}: postgold produced no passing oracle tests'
        else:
            reports = derive_structured_reports(dict(pregold.get('test_statuses') or {}), postgold_statuses)
            fail_to_pass = reports['FAIL_TO_PASS']
            pass_to_pass = reports['PASS_TO_PASS']
            pass_to_fail = reports['PASS_TO_FAIL']
            row['fail_to_pass'], row['pass_to_pass'] = fail_to_pass, pass_to_pass
            if not fail_to_pass:
                row['error_kind'] = 'oracle_nongating'
                row['error'] = 'no FAIL_TO_PASS: the executed oracle does not gate the fix'
            elif pass_to_fail:
                row['error_kind'] = 'gold_regression'
                row['error'] = f'gold regresses tests (PASS_TO_FAIL): {pass_to_fail}'
            else:
                # Persist only targets that really existed after test_gold was applied. This keeps
                # rollout evaluation and Harbor on the same target set Export executed.
                graded_feature = {**feature, 'test_files': list(postgold.get('test_files') or [])}
                instance_row = build_swebench_instance(
                    graded_feature,
                    instance,
                    fail_to_pass=fail_to_pass,
                    pass_to_pass=pass_to_pass,
                )
                instance_dir = write_harbor_instance(harbor_root, instance_row)
                row['harbor_path'] = str(instance_dir)
                row['success'] = True

    row['completed'] = row['runtime_error'] is None
    return export_unified_row(row), instance_row, None


__all__ = ['run_export_feature']
