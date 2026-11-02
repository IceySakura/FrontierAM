"""One-feature execution for the SWE-Any Issue stage."""

from __future__ import annotations

import logging

from pathlib import Path
from typing import Any

from nexus.extensions.tasks.swe_any.stage.issue.prompts import build_issue_prompt
from nexus.extensions.tasks.swe_any.stage.issue.schema import (
    IssueTaskInput,
    empty_issue_row,
    issue_unified_row,
    normalize_issue_agentic,
)
from nexus.extensions.tasks.swe_any.utils import constants
from nexus.extensions.tasks.swe_any.utils.agent_output import parse_agent_json
from nexus.extensions.tasks.swe_any.utils.artifacts import write_json_file
from nexus.extensions.tasks.swe_any.utils.repo_git import apply_patches
from nexus.extensions.tasks.swe_any.utils.sandbox_files import clear_sandbox_file, download_text_artifact, upload_text


logger = logging.getLogger(__name__)


async def run_issue_feature(
    task_input: IssueTaskInput,
    runtime: Any,
    feature: dict[str, Any],
) -> tuple[dict[str, Any], dict[str, Any], str | None]:
    """Write one feature's problem statement and return agentic, unified, and optional error rows."""
    index = feature['index']
    host_slice_dir = Path(task_input.output_dir) / task_input.repo_name / constants.slice_dirname(index)
    row = empty_issue_row(index)
    agentic_row: dict[str, Any] = {'index': index, 'problem_statement': ''}

    # Build the fully masked working tree and leave it applied while the agent runs.
    setup = await apply_patches(
        runtime,
        task_input.repo_dir,
        [
            ('feature_mask_patch', feature['feature_mask_patch']),
            ('test_mask_patch', feature['test_mask_patch']),
        ],
        base_commit=task_input.base_commit,
        request_timeout=task_input.request_timeout,
    )
    # Fail closed: an empty response would otherwise run the agent on the pristine implementation.
    if not (setup and all(result.get('ok') for result in setup.values())):
        row['error'] = 'masked-state setup failed: feature_mask/test_mask patches do not apply together'
    else:
        sandbox_agentic_path = constants.sandbox_path(
            task_input.agentic_filename,
            swe_any_dir=task_input.swe_any_dir,
        )
        feature_gold_path = constants.sandbox_path(
            constants.ISSUE_FEATURE_GOLD_FILENAME,
            swe_any_dir=task_input.swe_any_dir,
        )
        test_gold_path = constants.sandbox_path(
            constants.ISSUE_TEST_GOLD_FILENAME,
            swe_any_dir=task_input.swe_any_dir,
        )
        try:
            await upload_text(runtime, feature_gold_path, feature['feature_gold_patch'] or '')
            await upload_text(runtime, test_gold_path, feature['test_gold_patch'] or '')
            prompt = build_issue_prompt(
                working_dir=task_input.repo_dir,
                feature_desc=feature['feature_desc'],
                feature_gold_patch_path=feature_gold_path,
                test_gold_patch_path=test_gold_path,
                output_path=sandbox_agentic_path,
            )
            agent_output = await task_input.run_agent(
                user_prompt=prompt,
                working_dir=task_input.repo_dir,
                timeout=task_input.agent_timeout,
            )
            if not bool(getattr(agent_output, 'success', False)):
                raise RuntimeError(f'agent success=false exit_code={getattr(agent_output, "exit_code", None)}')
            downloaded = await download_text_artifact(
                runtime,
                sandbox_agentic_path,
                host_slice_dir / task_input.agentic_filename,
            )
            normalized = normalize_issue_agentic(
                parse_agent_json(Path(downloaded).read_text(encoding='utf-8') if downloaded else '')
            )
            row.update(normalized)
            agentic_row['problem_statement'] = row['problem_statement']
        except Exception as exc:  # noqa: BLE001 - the issue agent is best-effort
            message = f'{type(exc).__name__}: {exc}'
            logger.exception('swe_any_issue feature %s agent failed for %s', index, task_input.repo_name)
            row['warnings'].append(f'issue agent failed: {message}')
        finally:
            await clear_sandbox_file(runtime, sandbox_agentic_path)
            await clear_sandbox_file(runtime, feature_gold_path)
            await clear_sandbox_file(runtime, test_gold_path)

    row['completed'] = row['runtime_error'] is None
    row['success'] = bool(row['source'])
    write_json_file(agentic_row, host_slice_dir / task_input.agentic_filename)
    return agentic_row, issue_unified_row(row), None


__all__ = ['run_issue_feature']
