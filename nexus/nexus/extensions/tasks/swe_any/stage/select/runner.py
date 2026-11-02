"""One-slice execution helpers for the SWE-Any Select stage."""

from __future__ import annotations

import logging
import random

from pathlib import Path
from typing import Any

from nexus.extensions.tasks.swe_any.stage.select.heuristics import (
    component_quotas,
    node_entry,
    sample_seeds,
    seed_components,
    select_subgraph,
    undirected_projection,
)
from nexus.extensions.tasks.swe_any.stage.select.prompts import build_select_prompt
from nexus.extensions.tasks.swe_any.stage.select.schema import (
    SelectAgenticRow,
    SelectStatusRow,
    SelectTaskInput,
    load_seed_policy,
    normalize_select_agentic_row,
)
from nexus.extensions.tasks.swe_any.utils import constants
from nexus.extensions.tasks.swe_any.utils.agent_output import parse_agent_json
from nexus.extensions.tasks.swe_any.utils.artifacts import write_json_file
from nexus.extensions.tasks.swe_any.utils.graph import filter_graph, node_info, read_json
from nexus.extensions.tasks.swe_any.utils.repo_git import reset_repo
from nexus.extensions.tasks.swe_any.utils.sandbox_files import clear_sandbox_file, download_text_artifact


logger = logging.getLogger(__name__)


def prepare_select_heuristic_context(task_input: SelectTaskInput) -> dict[str, Any]:
    """Load the working graph and prepare deterministic seed/slice context for one repo."""
    host_out_dir = Path(task_input.output_dir) / task_input.repo_name
    graph = filter_graph(
        read_json(task_input.graph_path),
        include_external=task_input.include_external,
        include_tests=task_input.include_tests,
    )
    allowed = set(graph.nodes)
    if not allowed:
        raise ValueError(
            'no candidate nodes under include_tests='
            f'{task_input.include_tests}, include_external={task_input.include_external}'
        )
    graph_function_ids = {node for node in allowed if node_info(graph, node).get('kind') == 'function'}
    undirected = undirected_projection(graph, allowed)

    policy_path = Path(task_input.seed_policy_path or (host_out_dir / constants.SEED_POLICY_FILENAME))
    exclude_globs = [*load_seed_policy(policy_path), *task_input.exclude_globs]
    components = seed_components(
        graph,
        allowed,
        exclude_globs=exclude_globs,
        min_component_nodes=task_input.min_component_nodes,
        max_component_nodes=task_input.max_component_nodes,
    )
    component_size_of = {node: len(component) for component in components for node in component}
    quotas = component_quotas(
        [len(component) for component in components],
        max_bfs_nodes=task_input.max_bfs_nodes,
        max_subgraphs=task_input.max_subgraphs,
    )
    logger.debug(
        'swe_any_select %s seed pool: %d/%d nodes in %d components (sizes=%s, quotas=%s, '
        'exclude_globs=%s, component window=[%d, %d])',
        task_input.repo_name,
        len(component_size_of),
        len(allowed),
        len(components),
        [len(component) for component in components[:8]],
        quotas[:8],
        exclude_globs or '-',
        task_input.min_component_nodes,
        task_input.max_component_nodes,
    )
    seeds = sample_seeds(
        graph,
        rng=random.Random(task_input.rng_seed),
        components=components,
        max_bfs_nodes=task_input.max_bfs_nodes,
        max_subgraphs=task_input.max_subgraphs,
    )
    return {
        'graph': graph,
        'allowed': allowed,
        'graph_function_ids': graph_function_ids,
        'undirected': undirected,
        'component_size_of': component_size_of,
        'seeds': seeds,
    }


def run_select_heuristic_slice(
    task_input: SelectTaskInput,
    context: dict[str, Any],
    *,
    index: int,
    seed: str,
) -> tuple[dict[str, Any], str | None]:
    """Build and archive one deterministic heuristic slice."""
    host_slice_dir = Path(task_input.output_dir) / task_input.repo_name / constants.slice_dirname(index)
    slice_rng_seed = None if task_input.rng_seed is None else task_input.rng_seed + index + 1
    try:
        heuristic_slice = select_subgraph(
            context['graph'],
            seed=seed,
            max_bfs_nodes=task_input.max_bfs_nodes,
            drop_bfs_prob=task_input.drop_bfs_prob,
            min_bfs_per_depth=task_input.min_bfs_per_depth,
            max_bfs_per_depth=task_input.max_bfs_per_depth,
            include_tests=task_input.include_tests,
            include_external=task_input.include_external,
            rng=random.Random(slice_rng_seed),
            rng_seed=slice_rng_seed,
            allowed=context['allowed'],
            undirected=context['undirected'],
            component_size=context['component_size_of'].get(seed, 0),
        )
    except Exception as exc:  # noqa: BLE001 - one heuristic slice is a soft failure
        error = f'slice {index} heuristic failed: {type(exc).__name__}: {exc}'
        logger.exception('swe_any_select %s', error)
        heuristic_row = {'index': index, 'seed': seed, 'meta': {}, 'heuristic_nodes': []}
        write_json_file(heuristic_row, host_slice_dir / task_input.heuristic_filename)
        return heuristic_row, error

    heuristic_row = {'index': index, **heuristic_slice}
    write_json_file(heuristic_row, host_slice_dir / task_input.heuristic_filename)
    return heuristic_row, None


async def run_select_agentic_slice(
    task_input: SelectTaskInput,
    runtime: Any,
    context: dict[str, Any],
    heuristic_row: dict[str, Any],
    *,
    reset_before: bool = True,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Run and archive one agentic pass from an existing heuristic row."""
    index = int(heuristic_row['index'])
    host_slice_dir = Path(task_input.output_dir) / task_input.repo_name / constants.slice_dirname(index)
    if reset_before:
        await reset_repo(runtime, task_input.repo_dir, base_commit=task_input.base_commit)

    sandbox_graph_path = constants.sandbox_path(constants.GRAPH_FILENAME, swe_any_dir=task_input.swe_any_dir)
    sandbox_agentic_path = constants.sandbox_path(task_input.agentic_filename, swe_any_dir=task_input.swe_any_dir)
    heuristic_nodes = heuristic_row['heuristic_nodes']
    prompt = build_select_prompt(
        working_dir=task_input.repo_dir,
        graph_path=sandbox_graph_path,
        output_path=sandbox_agentic_path,
        candidates=heuristic_nodes,
        include_external=task_input.include_external,
        min_final_nodes=task_input.min_final_nodes,
        max_final_nodes=task_input.max_final_nodes,
    )

    warnings: list[str] = []
    agentic_row = SelectAgenticRow(index=index)
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
        raw = parse_agent_json(Path(downloaded).read_text(encoding='utf-8') if downloaded else '')
        normalized = normalize_select_agentic_row(
            raw,
            graph_function_ids=context['graph_function_ids'],
            slice_ids={node['id'] for node in heuristic_nodes},
            node_builder=lambda entity_id: node_entry(context['graph'], entity_id),
            min_final_nodes=task_input.min_final_nodes,
            max_final_nodes=task_input.max_final_nodes,
        )
        agentic_row = SelectAgenticRow(index=index, **normalized)
    except Exception as exc:  # noqa: BLE001 - runtime / I/O / parse agent failure is soft
        message = f'{type(exc).__name__}: {exc}'
        logger.exception('swe_any_select slice %s agentic pass failed for %s', index, task_input.repo_name)
        warnings.append(f'agentic pass failed: {message}')
    finally:
        # The fixed sandbox path must never leak stale output into the next slice.
        await clear_sandbox_file(runtime, sandbox_agentic_path)

    agentic = agentic_row.model_dump()
    write_json_file(agentic, host_slice_dir / task_input.agentic_filename)
    status = SelectStatusRow(index=index, completed=True, runtime_error=None, warnings=warnings).model_dump()
    return agentic, status


async def run_select_feature(
    task_input: SelectTaskInput,
    runtime: Any,
    context: dict[str, Any],
    *,
    index: int,
    seed: str,
) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any], str | None]:
    """Run one legacy Select slice and return its heuristic, agentic, status, and error rows."""
    # Legacy Select reset before the heuristic so an earlier agent cannot bias this slice.
    await reset_repo(runtime, task_input.repo_dir, base_commit=task_input.base_commit)
    heuristic, error = run_select_heuristic_slice(task_input, context, index=index, seed=seed)
    if error:
        agentic = SelectAgenticRow(index=index).model_dump()
        write_json_file(
            agentic,
            Path(task_input.output_dir)
            / task_input.repo_name
            / constants.slice_dirname(index)
            / task_input.agentic_filename,
        )
        status = SelectStatusRow(index=index, completed=False, runtime_error=error).model_dump()
        return heuristic, agentic, status, error

    agentic, status = await run_select_agentic_slice(
        task_input,
        runtime,
        context,
        heuristic,
        reset_before=False,
    )
    return heuristic, agentic, status, None


__all__ = [
    'prepare_select_heuristic_context',
    'run_select_agentic_slice',
    'run_select_feature',
    'run_select_heuristic_slice',
]
