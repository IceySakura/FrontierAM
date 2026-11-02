"""One-feature execution for the SWE-Any Mask stage."""

from __future__ import annotations

import logging

from pathlib import Path
from typing import Any

from nexus.extensions.tasks.swe_any.stage.mask.prompts import build_mask_refine_prompt, name_map
from nexus.extensions.tasks.swe_any.stage.mask.schema import MaskTaskInput, empty_mask_row, mask_unified_row
from nexus.extensions.tasks.swe_any.utils import constants
from nexus.extensions.tasks.swe_any.utils.agent_output import parse_agent_json
from nexus.extensions.tasks.swe_any.utils.artifacts import write_json_file
from nexus.extensions.tasks.swe_any.utils.repo_git import apply_check_sequence, reset_repo, reverse_patch
from nexus.extensions.tasks.swe_any.utils.sandbox_files import clear_sandbox_file, download_text_artifact
from nexus.extensions.tasks.swe_any.utils.source_edit import patch_files
from nexus.runtime.models import InvokeRequest


logger = logging.getLogger(__name__)

_MASK_TARGET = 'nexus.extensions.tasks.swe_any.runtime_setup::run_mask'


async def run_mask_feature(
    task_input: MaskTaskInput,
    runtime: Any,
    graph: Any,
    *,
    index: int,
    feature_desc: str,
    feature_node_ids: list[str],
) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any], str | None]:
    """Mask one selected feature and return heuristic, agentic, unified, and optional error rows."""
    host_slice_dir = Path(task_input.output_dir) / task_input.repo_name / constants.slice_dirname(index)
    sandbox_graph_path = constants.sandbox_path(task_input.graph_filename, swe_any_dir=task_input.swe_any_dir)
    # Fixed path for the raw draft read by the refine agent; serial execution makes reuse safe.
    sandbox_patch = constants.sandbox_path(
        Path(task_input.heuristic_filename).with_suffix('.patch').name,
        swe_any_dir=task_input.swe_any_dir,
    )
    row = empty_mask_row(index, feature_node_ids)
    agentic_row: dict[str, Any] = {
        'index': index,
        'cover_rate': 0.0,
        'agentic_mask_patch': '',
        'agentic_gold_patch': '',
    }
    heuristic_mask_patch = ''
    heuristic_gold_patch = ''
    agentic_mask_patch = ''
    agentic_gold_patch = ''

    # Start pristine so an earlier feature's refine cannot leak edits.
    await reset_repo(
        runtime,
        task_input.repo_dir,
        base_commit=task_input.base_commit,
    )
    try:
        request = InvokeRequest.build(
            _MASK_TARGET,
            venv_id=runtime.default_venv_id,
            request_timeout=task_input.request_timeout,
            graph_path=sandbox_graph_path,
            repo_dir=task_input.repo_dir,
            feature=feature_node_ids,
            heuristic_patch_path=sandbox_patch,
        )
        result: dict[str, Any] = await runtime.invoke(request)
    except Exception as exc:  # noqa: BLE001 - a per-feature invoke failure stays soft
        message = f'{type(exc).__name__}: {exc}'
        logger.exception('swe_any_mask feature %s run_mask failed for %s', index, task_input.repo_name)
        row['runtime_error'] = message
        heuristic_row = {'index': index, 'nodes': feature_node_ids}
        write_json_file(heuristic_row, host_slice_dir / task_input.heuristic_filename)
        write_json_file(agentic_row, host_slice_dir / task_input.agentic_filename)
        return (
            heuristic_row,
            agentic_row,
            mask_unified_row(row, mask_patch='', gold_patch=''),
            f'feature {index} mask failed: {message}',
        )

    heuristic_row = {'index': index, **result}
    write_json_file(heuristic_row, host_slice_dir / task_input.heuristic_filename)
    plan = result.get('heuristic_mask_plan') or {}
    entries = plan.get('entries') or []
    deletes = plan.get('deletes') or []
    heuristic_mask_patch = result.get('heuristic_mask_patch') or ''
    heuristic_gold_patch = result.get('heuristic_gold_patch') or ''
    row.update({
        'files': plan.get('files') or [],
        'warnings': result.get('warnings') or [],
    })

    if task_input.use_agent and heuristic_mask_patch:
        sandbox_agentic_path = constants.sandbox_path(
            task_input.agentic_filename,
            swe_any_dir=task_input.swe_any_dir,
        )
        prompt = build_mask_refine_prompt(
            working_dir=task_input.repo_dir,
            feature_desc=feature_desc,
            output_path=sandbox_agentic_path,
            heuristic_patch_path=sandbox_patch,
            entries=name_map(graph, entries),
            deletes=name_map(graph, deletes),
        )
        try:
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
            parsed = parse_agent_json(Path(downloaded).read_text(encoding='utf-8') if downloaded else '')
            agentic_mask_patch = str(parsed.get('mask_patch') or '').strip()
            # Scope is a monitoring signal, not a filter.
            touched = patch_files(agentic_mask_patch)
            in_scope = touched & set(row['files'])
            row['cover_rate'] = (len(in_scope) / len(touched)) if touched else 0.0
            out_of_scope = sorted(touched - set(row['files']))
            if out_of_scope:
                row['warnings'].append(f'agentic patch touches non-feature files: {out_of_scope}')
                logger.warning('swe_any_mask feature %s refine touched non-feature files: %s', index, out_of_scope)
            # Gold for a refined mask comes from git, not a textual patch reverser.
            if agentic_mask_patch:
                agentic_gold_patch = await reverse_patch(
                    runtime,
                    task_input.repo_dir,
                    agentic_mask_patch,
                    base_commit=task_input.base_commit,
                    request_timeout=task_input.request_timeout,
                )
        except Exception as exc:  # noqa: BLE001 - refine failure is soft; heuristic still stands
            message = f'{type(exc).__name__}: {exc}'
            logger.exception('swe_any_mask feature %s refine failed for %s', index, task_input.repo_name)
            row['warnings'].append(f'refine agent failed: {message}')
        finally:
            await clear_sandbox_file(runtime, sandbox_agentic_path)

    async def mask_applicable(mask_patch: str, gold_patch: str) -> bool:
        if not mask_patch:
            return False
        try:
            results = await apply_check_sequence(
                runtime,
                task_input.repo_dir,
                [('feature_mask_patch', mask_patch), ('feature_gold_patch', gold_patch)],
                base_commit=task_input.base_commit,
                request_timeout=task_input.request_timeout,
            )
            # Fail closed on empty/renamed/truncated responses.
            return bool(results) and all(result.get('ok') for result in results.values())
        except Exception as exc:  # noqa: BLE001 - only infrastructure failures reach here
            message = f'{type(exc).__name__}: {exc}'
            logger.warning('swe_any_mask feature %s apply-check failed: %s', index, message)
            row['warnings'].append(f'apply-check failed: {message}')
            if row['runtime_error'] is None:
                row['runtime_error'] = f'apply-check infra failure: {message}'
            return False

    mask_patch, gold_patch = '', ''
    if agentic_mask_patch and agentic_gold_patch and await mask_applicable(agentic_mask_patch, agentic_gold_patch):
        mask_patch, gold_patch, row['source'] = agentic_mask_patch, agentic_gold_patch, 'agentic'
    elif heuristic_mask_patch and await mask_applicable(heuristic_mask_patch, heuristic_gold_patch):
        if agentic_mask_patch:
            row['warnings'].append('agentic mask did not apply; falling back to the heuristic draft')
        mask_patch, gold_patch, row['source'] = heuristic_mask_patch, heuristic_gold_patch, 'heuristic'

    # Unexpected interruptions and business failure remain separate status channels.
    row['completed'] = row['runtime_error'] is None
    row['success'] = bool(row['source'])
    if row['completed'] and not row['success']:
        row['error'] = (
            'no maskable functions in feature'
            if not heuristic_mask_patch
            else 'no applicable mask patch (git apply --check failed)'
        )
    row['files'] = sorted(patch_files(mask_patch))

    agentic_row = {
        'index': index,
        'cover_rate': row['cover_rate'],
        'agentic_mask_patch': agentic_mask_patch,
        'agentic_gold_patch': agentic_gold_patch,
    }
    write_json_file(agentic_row, host_slice_dir / task_input.agentic_filename)
    return heuristic_row, agentic_row, mask_unified_row(row, mask_patch=mask_patch, gold_patch=gold_patch), None


__all__ = ['run_mask_feature']
