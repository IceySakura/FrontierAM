"""One-feature execution for the SWE-Any Eval stage."""

from __future__ import annotations

import logging

from pathlib import Path
from typing import Any

from nexus.extensions.tasks.swe_any.stage.eval.heuristics import find_candidate_tests
from nexus.extensions.tasks.swe_any.stage.eval.prompts import (
    build_test_oracle_prompt,
    build_test_oracle_verifier_prompt,
)
from nexus.extensions.tasks.swe_any.stage.eval.schema import (
    EvalTaskInput,
    EvalVerifierVerdictCode,
    empty_eval_row,
    eval_unified_row,
    parse_eval_verifier_verdict,
)
from nexus.extensions.tasks.swe_any.utils import constants
from nexus.extensions.tasks.swe_any.utils.artifacts import write_json_file
from nexus.extensions.tasks.swe_any.utils.repo_git import apply_check_sequence, reset_repo
from nexus.extensions.tasks.swe_any.utils.sandbox_files import clear_sandbox_file, download_text_artifact, upload_text
from nexus.extensions.tasks.swe_any.utils.source_edit import patch_test_targets
from nexus.runtime.models import InvokeRequest


logger = logging.getLogger(__name__)

_EVAL_TARGET = 'nexus.extensions.tasks.swe_any.runtime_setup::run_eval'


async def _run_oracle_verifier(
    task_input: EvalTaskInput,
    runtime: Any,
    host_slice_dir: Path,
    *,
    feature_mask_patch: str,
    feature_gold_patch: str,
    test_mask_patch: str,
    test_gold_patch: str,
    test_runner_script: str,
) -> dict[str, Any]:
    """Run the independent semantic verifier and return its strict verdict."""
    feature_mask_path = constants.sandbox_path(
        constants.EVAL_FEATURE_MASK_FILENAME,
        swe_any_dir=task_input.swe_any_dir,
    )
    feature_gold_path = constants.sandbox_path(
        constants.EVAL_FEATURE_GOLD_FILENAME,
        swe_any_dir=task_input.swe_any_dir,
    )
    test_mask_path = constants.sandbox_path(
        constants.EVAL_AGENTIC_MASK_FILENAME,
        swe_any_dir=task_input.swe_any_dir,
    )
    test_gold_path = constants.sandbox_path(
        constants.EVAL_AGENTIC_GOLD_FILENAME,
        swe_any_dir=task_input.swe_any_dir,
    )
    runner_path = constants.sandbox_path(
        constants.EVAL_AGENTIC_RUNNER_FILENAME,
        swe_any_dir=task_input.swe_any_dir,
    )
    verdict_path = constants.sandbox_path(
        constants.EVAL_AGENTIC_VERIFIER_FILENAME,
        swe_any_dir=task_input.swe_any_dir,
    )
    artifact_paths = [feature_mask_path, feature_gold_path, test_mask_path, test_gold_path, runner_path, verdict_path]
    try:
        for path in artifact_paths:
            await clear_sandbox_file(runtime, path)
        for path, content in (
            (feature_mask_path, feature_mask_patch),
            (feature_gold_path, feature_gold_patch),
            (test_mask_path, test_mask_patch),
            (test_gold_path, test_gold_patch),
            (runner_path, test_runner_script),
        ):
            await upload_text(runtime, path, content)
        prompt = build_test_oracle_verifier_prompt(
            working_dir=task_input.repo_dir,
            feature_mask_path=feature_mask_path,
            feature_gold_path=feature_gold_path,
            test_mask_path=test_mask_path,
            test_gold_path=test_gold_path,
            runner_path=runner_path,
            verdict_output_path=verdict_path,
        )
        agent_output = await task_input.run_agent(
            user_prompt=prompt,
            working_dir=task_input.repo_dir,
            timeout=task_input.agent_timeout,
        )
        if not bool(getattr(agent_output, 'success', False)):
            raise RuntimeError(f'agent success=false exit_code={getattr(agent_output, "exit_code", None)}')
        verdict_download = await download_text_artifact(
            runtime,
            verdict_path,
            host_slice_dir / constants.EVAL_AGENTIC_VERIFIER_FILENAME,
        )
        if not verdict_download:
            raise ValueError('verifier did not write its verdict')
        return parse_eval_verifier_verdict(Path(verdict_download).read_text(encoding='utf-8'))
    finally:
        for path in artifact_paths:
            await clear_sandbox_file(runtime, path)


async def run_eval_feature(
    task_input: EvalTaskInput,
    runtime: Any,
    graph: Any,
    *,
    index: int,
    feature: list[str],
    feature_mask_patch: str,
    feature_gold_patch: str,
) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any], str | None]:
    """Build one feature's test oracle and return its three rows plus an optional error."""
    host_slice_dir = Path(task_input.output_dir) / task_input.repo_name / constants.slice_dirname(index)
    sandbox_graph_path = constants.sandbox_path(task_input.graph_filename, swe_any_dir=task_input.swe_any_dir)
    row = empty_eval_row(index)
    heuristic_row: dict[str, Any] = {'index': index}
    agentic_row: dict[str, Any] = {
        'index': index,
        'agentic_mask_patch': '',
        'agentic_gold_patch': '',
        'agentic_runner_script': '',
        'agentic_test_files': [],
        'agentic_verifier': None,
    }

    # Start pristine so an earlier feature's oracle agent cannot leak edits.
    await reset_repo(
        runtime,
        task_input.repo_dir,
        base_commit=task_input.base_commit,
    )

    # Deterministic leaf tests and their mechanical patches are hints for the oracle agent.
    # They are never eligible as the Eval deliverable: only refine -> verify can commit an oracle.
    candidates = find_candidate_tests(graph, feature, include_feature_tests=task_input.include_feature_tests)
    candidate_ids = [candidate['id'] for candidate in candidates]

    heuristic_mask_patch_path = ''
    heuristic_gold_patch_path = ''
    if candidates:
        mask_draft_path = constants.sandbox_path(
            constants.EVAL_HEURISTIC_MASK_FILENAME,
            swe_any_dir=task_input.swe_any_dir,
        )
        gold_draft_path = constants.sandbox_path(
            constants.EVAL_HEURISTIC_GOLD_FILENAME,
            swe_any_dir=task_input.swe_any_dir,
        )
        try:
            request = InvokeRequest.build(
                _EVAL_TARGET,
                venv_id=runtime.default_venv_id,
                request_timeout=task_input.request_timeout,
                graph_path=sandbox_graph_path,
                repo_dir=task_input.repo_dir,
                tests=candidate_ids,
                heuristic_mask_patch_path=mask_draft_path,
                heuristic_gold_patch_path=gold_draft_path,
            )
            result: dict[str, Any] = await runtime.invoke(request)
            heuristic_row = {'index': index, **result}
            write_json_file(heuristic_row, host_slice_dir / task_input.heuristic_filename)
            heuristic_mask_patch_path = mask_draft_path
            heuristic_gold_patch_path = gold_draft_path if result.get('heuristic_gold_patch') else ''
            row['warnings'].extend(result.get('warnings') or [])
        except Exception as exc:  # noqa: BLE001 - the optional draft must not block the agents
            message = f'{type(exc).__name__}: {exc}'
            logger.exception('swe_any_eval feature %s run_eval draft failed for %s', index, task_input.repo_name)
            row['warnings'].append(f'heuristic draft failed: {message}')

    agentic_mask_patch = ''
    agentic_gold_patch = ''
    agentic_runner_script = ''
    # The author/refine pass is the only source of a candidate oracle. A missing mechanical
    # draft changes the prompt into an author-from-context task; it never promotes heuristics.
    feature_mask_patch_path = constants.sandbox_path(
        constants.EVAL_FEATURE_MASK_FILENAME,
        swe_any_dir=task_input.swe_any_dir,
    )
    feature_gold_patch_path = constants.sandbox_path(
        constants.EVAL_FEATURE_GOLD_FILENAME,
        swe_any_dir=task_input.swe_any_dir,
    )
    agentic_mask_out_path = constants.sandbox_path(
        constants.EVAL_AGENTIC_MASK_FILENAME,
        swe_any_dir=task_input.swe_any_dir,
    )
    agentic_gold_out_path = constants.sandbox_path(
        constants.EVAL_AGENTIC_GOLD_FILENAME,
        swe_any_dir=task_input.swe_any_dir,
    )
    agentic_runner_out_path = constants.sandbox_path(
        constants.EVAL_AGENTIC_RUNNER_FILENAME,
        swe_any_dir=task_input.swe_any_dir,
    )
    prompt = build_test_oracle_prompt(
        working_dir=task_input.repo_dir,
        candidates=candidates,
        mask_output_path=agentic_mask_out_path,
        gold_output_path=agentic_gold_out_path,
        runner_output_path=agentic_runner_out_path,
        feature_mask_path=feature_mask_patch_path,
        feature_gold_path=feature_gold_patch_path,
        heuristic_test_mask_path=heuristic_mask_patch_path,
        heuristic_test_gold_path=heuristic_gold_patch_path,
    )
    try:
        # Fixed output paths are reused across serial features. Clear first as well as in
        # ``finally`` so an interrupted prior call cannot be mistaken for this agent's output.
        await clear_sandbox_file(runtime, agentic_mask_out_path)
        await clear_sandbox_file(runtime, agentic_gold_out_path)
        await clear_sandbox_file(runtime, agentic_runner_out_path)
        await upload_text(runtime, feature_mask_patch_path, feature_mask_patch)
        await upload_text(runtime, feature_gold_patch_path, feature_gold_patch)
        agent_output = await task_input.run_agent(
            user_prompt=prompt,
            working_dir=task_input.repo_dir,
            timeout=task_input.agent_timeout,
        )
        if not bool(getattr(agent_output, 'success', False)):
            raise RuntimeError(f'agent success=false exit_code={getattr(agent_output, "exit_code", None)}')
        mask_download = await download_text_artifact(
            runtime,
            agentic_mask_out_path,
            host_slice_dir / constants.EVAL_AGENTIC_MASK_FILENAME,
        )
        gold_download = await download_text_artifact(
            runtime,
            agentic_gold_out_path,
            host_slice_dir / constants.EVAL_AGENTIC_GOLD_FILENAME,
        )
        runner_download = await download_text_artifact(
            runtime,
            agentic_runner_out_path,
            host_slice_dir / constants.EVAL_AGENTIC_RUNNER_FILENAME,
        )
        agentic_mask_patch = (Path(mask_download).read_text(encoding='utf-8') if mask_download else '').strip()
        agentic_gold_patch = (Path(gold_download).read_text(encoding='utf-8') if gold_download else '').strip()
        agentic_runner_script = (Path(runner_download).read_text(encoding='utf-8') if runner_download else '').strip()
        if not agentic_runner_script:
            raise ValueError('agent did not write test_runner.sh')
        agentic_row = {
            'index': index,
            'agentic_mask_patch': agentic_mask_patch,
            'agentic_gold_patch': agentic_gold_patch,
            'agentic_runner_script': agentic_runner_script,
            # Only the forward oracle defines grade targets. A mask-only path may be deleted
            # permanently and must never be handed to pytest.
            'agentic_test_files': patch_test_targets(agentic_gold_patch),
        }
    except Exception as exc:  # noqa: BLE001 - oracle agent failure is soft
        message = f'{type(exc).__name__}: {exc}'
        logger.exception('swe_any_eval feature %s test-oracle failed for %s', index, task_input.repo_name)
        row['warnings'].append(f'oracle agent failed: {message}')
    finally:
        # Every path is fixed and reused by the next serial feature.
        await clear_sandbox_file(runtime, agentic_mask_out_path)
        await clear_sandbox_file(runtime, agentic_gold_out_path)
        await clear_sandbox_file(runtime, agentic_runner_out_path)
        await clear_sandbox_file(runtime, feature_mask_patch_path)
        await clear_sandbox_file(runtime, feature_gold_patch_path)
        if heuristic_mask_patch_path:
            await clear_sandbox_file(runtime, heuristic_mask_patch_path)
        if heuristic_gold_patch_path:
            await clear_sandbox_file(runtime, heuristic_gold_patch_path)

    async def oracle_applicable(mask_patch: str, gold_patch: str) -> bool:
        steps = [('test_mask_patch', mask_patch), ('test_gold_patch', gold_patch)]
        if not any(patch for _, patch in steps):
            return False
        try:
            results = await apply_check_sequence(
                runtime,
                task_input.repo_dir,
                steps,
                base_commit=task_input.base_commit,
                request_timeout=task_input.request_timeout,
            )
            # Fail closed on empty/renamed/truncated responses.
            return bool(results) and all(result.get('ok') for result in results.values())
        except Exception as exc:  # noqa: BLE001 - only infrastructure failures reach here
            message = f'{type(exc).__name__}: {exc}'
            logger.warning('swe_any_eval feature %s apply-check failed: %s', index, message)
            row['warnings'].append(f'apply-check failed: {message}')
            if row['runtime_error'] is None:
                row['runtime_error'] = f'apply-check infra failure: {message}'
            return False

    verifier: dict[str, Any] | None = None
    oracle_is_applicable = bool(
        agentic_gold_patch and agentic_runner_script and await oracle_applicable(agentic_mask_patch, agentic_gold_patch)
    )
    if oracle_is_applicable:
        try:
            verifier = await _run_oracle_verifier(
                task_input,
                runtime,
                host_slice_dir,
                feature_mask_patch=feature_mask_patch,
                feature_gold_patch=feature_gold_patch,
                test_mask_patch=agentic_mask_patch,
                test_gold_patch=agentic_gold_patch,
                test_runner_script=agentic_runner_script,
            )
        except Exception as exc:  # noqa: BLE001 - verifier failure rejects only this oracle
            message = f'{type(exc).__name__}: {exc}'
            logger.exception('swe_any_eval feature %s oracle verifier failed for %s', index, task_input.repo_name)
            row['warnings'].append(f'oracle verifier failed: {message}')
    agentic_row['agentic_verifier'] = verifier

    # test_gold defines FAIL_TO_PASS and must be non-empty; test_mask may be empty for extra-only tests.
    test_mask_patch, test_gold_patch, test_runner_script = '', '', ''
    verifier_accepted = bool(verifier and verifier.get('verdict') == EvalVerifierVerdictCode.CLEAN)
    if oracle_is_applicable and verifier_accepted:
        test_mask_patch, test_gold_patch, test_runner_script, row['source'] = (
            agentic_mask_patch,
            agentic_gold_patch,
            agentic_runner_script,
            'agentic',
        )

    row['completed'] = row['runtime_error'] is None
    row['success'] = oracle_is_applicable and verifier_accepted
    if row['completed'] and not row['success']:
        if verifier and not verifier_accepted:
            row['error'] = f'oracle verifier rejected ({verifier["verdict"]}): {verifier["rationale"]}'
        else:
            row['error'] = 'no verified agentic test oracle with a structured runner'
    # test_mask only says what the solver may not see. Grade targets come exclusively from the
    # forward oracle; Export filters this list again after applying patches so deleted/renamed
    # gold headers cannot abort the runner on a nonexistent path.
    row['test_files'] = patch_test_targets(test_gold_patch)
    write_json_file(agentic_row, host_slice_dir / task_input.agentic_filename)
    unified_row = eval_unified_row(
        row,
        patches={
            'test_mask_patch': test_mask_patch,
            'test_gold_patch': test_gold_patch,
            'test_runner_script': test_runner_script,
            'verifier': verifier,
        },
    )
    return heuristic_row, agentic_row, unified_row, None


__all__ = ['run_eval_feature']
