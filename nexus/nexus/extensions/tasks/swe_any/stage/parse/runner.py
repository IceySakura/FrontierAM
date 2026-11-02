"""Connected-runtime execution helpers for the SWE-Any Parse stage."""

from __future__ import annotations

import logging

from pathlib import Path, PurePosixPath
from typing import Any

from nexus.extensions.tasks.swe_any.stage.parse.prompts import build_seed_policy_prompt
from nexus.extensions.tasks.swe_any.stage.parse.schema import ParseTaskInput
from nexus.extensions.tasks.swe_any.utils import constants
from nexus.extensions.tasks.swe_any.utils.agent_output import parse_agent_json
from nexus.extensions.tasks.swe_any.utils.artifacts import write_json_file
from nexus.extensions.tasks.swe_any.utils.graph import node_info, read_json
from nexus.extensions.tasks.swe_any.utils.sandbox_files import download_text_artifact
from nexus.runtime.models import InvokeRequest


logger = logging.getLogger(__name__)

_PARSE_TARGET = 'nexus.extensions.tasks.swe_any.runtime_setup::run_parse'


def _graph_profile(graph: Any, limit: int) -> list[tuple[str, int, int]]:
    """Return the largest internal directories as ``(directory, entities, files)``."""
    entities: dict[str, int] = {}
    files: dict[str, set[str]] = {}
    for node in graph.nodes:
        info = node_info(graph, node)
        if info.get('external'):
            continue
        file = info.get('file') or ''
        if not file:
            continue
        directory = str(PurePosixPath(file).parent)
        entities[directory] = entities.get(directory, 0) + 1
        files.setdefault(directory, set()).add(file)
    ranked = sorted(entities.items(), key=lambda item: (-item[1], item[0]))
    return [(directory, count, len(files[directory])) for directory, count in ranked[:limit]]


def _policy_globs(raw: dict[str, Any]) -> list[str]:
    """Return usable ``exclude_globs`` without second-guessing their breadth."""
    return [glob for glob in (raw.get('exclude_globs') or []) if isinstance(glob, str) and glob.strip()]


async def run_seed_policy_pass(
    task_input: ParseTaskInput,
    runtime: Any,
    *,
    graph_path: str,
    host_out_dir: Path,
) -> dict[str, Any]:
    """Run the optional seed-policy agent; failures are returned instead of raised."""
    graph = read_json(graph_path)
    sandbox_policy_path = constants.sandbox_path(
        task_input.seed_policy_filename,
        swe_any_dir=task_input.swe_any_dir,
    )
    prompt = build_seed_policy_prompt(
        working_dir=task_input.repo_dir,
        graph_path=constants.sandbox_path(task_input.graph_filename, swe_any_dir=task_input.swe_any_dir),
        policy_path=sandbox_policy_path,
        profile=_graph_profile(graph, task_input.profile_dirs),
    )
    try:
        agent_output = await task_input.run_agent(
            user_prompt=prompt,
            working_dir=task_input.repo_dir,
            timeout=task_input.agent_timeout,
        )
        if not bool(getattr(agent_output, 'success', False)):
            raise RuntimeError(f'agent success=false exit_code={getattr(agent_output, "exit_code", None)}')
        host_policy_path = host_out_dir / task_input.seed_policy_filename
        downloaded = await download_text_artifact(runtime, sandbox_policy_path, host_policy_path)
        raw = parse_agent_json(Path(downloaded).read_text(encoding='utf-8') if downloaded else '')
        globs = _policy_globs(raw)
    except Exception as exc:  # noqa: BLE001 - agent / I/O / parse are all soft here
        message = f'{type(exc).__name__}: {exc}'
        logger.warning('swe_any_parse seed policy pass failed for %s: %s', task_input.repo_name, message)
        return {'seed_policy_error': message}

    write_json_file(
        {
            'exclude_globs': globs,
            'exclude_reasons': raw.get('exclude_reasons') if isinstance(raw.get('exclude_reasons'), dict) else {},
            'notes': str(raw.get('notes') or ''),
        },
        host_out_dir / task_input.seed_policy_filename,
    )
    logger.debug('swe_any_parse %s seed policy: %d exclude globs', task_input.repo_name, len(globs))
    return {
        'seed_policy_path': str(host_out_dir / task_input.seed_policy_filename),
        'exclude_globs': globs,
    }


async def run_parse_operation(task_input: ParseTaskInput, runtime: Any) -> dict[str, Any]:
    """Run Parse and download its graph, without running the optional policy agent."""
    sandbox_graph_path = constants.sandbox_path(task_input.graph_filename, swe_any_dir=task_input.swe_any_dir)
    request = InvokeRequest.build(
        _PARSE_TARGET,
        venv_id=runtime.default_venv_id,
        request_timeout=task_input.request_timeout,
        repo_dir=task_input.repo_dir,
        graph_path=sandbox_graph_path,
        project_name=task_input.project_name,
        install_scip_python=task_input.install_scip_python,
        npm_registry=task_input.npm_registry,
        index_filename=constants.INDEX_FILENAME,
        index_timeout=task_input.index_timeout,
        npm_install_timeout=task_input.npm_install_timeout,
    )
    result: dict[str, Any] = await runtime.invoke(request)

    host_out_dir = Path(task_input.output_dir) / task_input.repo_name
    host_graph_path: str | None = None
    if task_input.download and result.get('graph_path'):
        host_graph_path = await download_text_artifact(
            runtime,
            result['graph_path'],
            host_out_dir / task_input.graph_filename,
        )

    return {
        'nodes_internal': int(result.get('nodes_internal', 0)),
        'nodes_total': int(result.get('nodes_total', 0)),
        'edges_internal': int(result.get('edges_internal', 0)),
        'edges_total': int(result.get('edges_total', 0)),
        'sandbox_graph_path': result.get('graph_path'),
        'graph_path': host_graph_path,
        'index_path': result.get('index_path'),
    }


async def run_parse_repo(task_input: ParseTaskInput, runtime: Any) -> dict[str, Any]:
    """Run Parse plus its optional seed-policy pass in an already-connected runtime."""
    fields = await run_parse_operation(task_input, runtime)
    host_graph_path = fields['graph_path']
    if host_graph_path is None:
        policy = {'seed_policy_error': 'seed policy needs the downloaded graph (download=false)'}
        logger.warning('swe_any_parse skipping seed policy for %s: no host graph', task_input.repo_name)
    else:
        policy = await run_seed_policy_pass(
            task_input,
            runtime,
            graph_path=host_graph_path,
            host_out_dir=Path(task_input.output_dir) / task_input.repo_name,
        )
    return {**fields, **policy}


__all__ = ['run_parse_operation', 'run_parse_repo', 'run_seed_policy_pass']
