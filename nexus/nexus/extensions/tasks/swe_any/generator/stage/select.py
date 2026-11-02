#!/usr/bin/env python3
"""SWE-Any batch generator for ``nexus run-batch`` — Select at scale.

Consumes the same **SWE-Env instances** as Parse, but only yields a
``swe_any_select`` BatchItem when Parse has already written
``<output_dir>/<repo_name>/parse_graph.json`` on the host. Each item re-acquires the
baked per-repo image, uploads the graph into the sandbox once, and runs N
heuristic slices + Claude Code agentic passes under ``max_workers``.

Each item starts from ``config/stage/select.yaml`` — the same file
``run-single`` uses — with the ``SWE_ANY_CONFIG_PATCH`` patches deep-merged over
it (comma-separated, later wins) and the LLM-backend patch
(``config/client/<name>.yaml``) for the merged ``openai_clients.main.name``
applied after that (see ``batch_config``). Only per-instance fields and the
provider timeout floor are injected here, so exporting any ``${oc.env:...}`` the
YAML references still overrides the default.

Usage (after a Parse batch has produced graphs):
  SWE_ANY_INSTANCES=./debug/swe_env.jsonl  GONGFENG_TOKEN=... OPENAI_API_KEY=... \\
  uv run nexus run-batch -g nexus/extensions/tasks/swe_any/generator/stage/select.py \\
      -d ./outputs/swe_any_select_batch.db --max-concurrent 4

Environment:
  - SWE_ANY_INSTANCES: path to a ``.jsonl`` file containing one instance per line (required)
  - SWE_ANY_CONFIG_PATCH: optional comma-separated YAML patches merged over the
    default config in order, later wins (task knobs + the instances block)
  - SWE_ANY_INSTANCE_IDS / SWE_ANY_SKIP / SWE_ANY_LIMIT / SWE_ANY_LANGUAGE: subset + paging (same as Parse)
  - SWE_ANY_RNG_SEED: fix the seed draw (default: unseeded, i.e. a fresh draw per run)
  - SWE_ANY_OUTPUT_DIR: host artifact dir (default: ./outputs/swe_any)
  - OPENAI_API_KEY / GONGFENG_TOKEN: host LLM credential + sandbox token (secrets)
  - Anything the config references via ``${oc.env:...}`` (model, timeouts, …)
"""

from __future__ import annotations

import copy

from collections.abc import Iterator
from typing import Any

from nexus import BatchItem
from nexus.extensions.tasks.swe_any.utils.batch_config import (
    STAGE_CONFIG_DIR,
    agentic_config_and_spec,
    attach_cache_task_id,
    config_patch_paths,
    resolve_config,
)
from nexus.extensions.tasks.swe_any.utils.instance import (
    host_graph_path,
    output_dir,
    repo_name_of,
    selected_instances,
)
from nexus.runner.single import RunInput


_DEFAULT_YAML = STAGE_CONFIG_DIR / 'select.yaml'


def _selected_select_instances(spec: dict[str, Any]) -> Iterator[dict[str, Any]]:
    """Yield only instances with a host ``parse_graph.json`` — Select's hard precondition.

    The generator injects ``graph_path`` from ``host_graph_path(inst)``, so an
    instance without that file could not run anyway; skipping it here turns a
    guaranteed sandbox failure into a quiet no-op.
    """
    out_dir = output_dir()
    for inst in selected_instances(spec):
        if host_graph_path(inst, out_dir=out_dir).is_file():
            yield inst


def _build_config(base: dict[str, Any], inst: dict[str, Any]) -> RunInput:
    config = copy.deepcopy(base)
    resolved = resolve_config(config)

    repo_name = repo_name_of(inst)
    repo_dir = str(inst.get('workdir') or '/testbed')
    agent_timeout = float(resolved['agent_config']['timeout'])

    task_config = config['task_config']
    task_config['repo_name'] = repo_name
    task_config['graph_path'] = str(host_graph_path(inst))
    task_config['output_dir'] = output_dir()
    task_config['image'] = inst['image']
    task_config['repo_dir'] = repo_dir
    # Batch select releases each sandbox after writing archives.
    task_config['release_runtime'] = True

    # Provider timeout applies to each runtime invoke, not to all semaphore rounds.
    provider_config = config['runtime_provider_config']
    provider_config['request_timeout'] = max(
        float(resolved['runtime_provider_config']['request_timeout']),
        agent_timeout + 600.0,
    )

    attach_cache_task_id(config, inst['instance_id'])
    return RunInput.from_dict(config)


def generate_configs() -> Iterator[BatchItem]:
    base, spec = agentic_config_and_spec(_DEFAULT_YAML)
    for inst in _selected_select_instances(spec):
        instance_id = str(inst['instance_id'])
        yield BatchItem(
            config=_build_config(base, inst),
            run_id=instance_id,
            metadata={
                'instance_id': instance_id,
                'repo_name': inst.get('repo_name'),
                'language': inst.get('language'),
                'stage': 'select',
                'graph_path': str(host_graph_path(inst)),
            },
        )


def get_total_count() -> int:
    _, spec = agentic_config_and_spec(_DEFAULT_YAML)
    return sum(1 for _ in _selected_select_instances(spec))


def get_batch_metadata() -> dict[str, Any]:
    _, spec = agentic_config_and_spec(_DEFAULT_YAML)
    return {
        'task_name': 'swe_any_select',
        'stage': 'select',
        'instances': spec.get('path', ''),
        'language': spec.get('language', ''),
        'output_dir': output_dir(),
        'config_patch': ','.join(config_patch_paths()),
    }
