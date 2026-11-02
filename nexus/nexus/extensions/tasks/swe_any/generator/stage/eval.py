#!/usr/bin/env python3
"""SWE-Any batch generator for ``nexus run-batch`` — Eval at scale.

Consumes the same **SWE-Env instances** as the earlier stages, but only yields a
``swe_any_eval`` BatchItem when Mask has already written
``<output_dir>/<repo_name>/mask.jsonl`` on the host. Each item re-acquires
the baked per-repo image, uploads ``parse_graph.json`` into the sandbox once, and
for every masked feature builds an optional draft test-deletion hint (``run_eval``), then runs
the required Claude Code oracle author/refine and independent verifier passes. The draft is
never an accepted fallback.

Each item starts from ``config/stage/eval.yaml`` — the same file ``run-single``
uses — with the ``SWE_ANY_CONFIG_PATCH`` patches deep-merged over it and the
LLM-backend patch (``config/client/<name>.yaml``) applied after (see
``batch_config``). Only per-instance fields and the provider timeout floor are
injected here.

Usage (after Parse + Select + Mask batches have run):
  SWE_ANY_INSTANCES=./debug/swe_env.jsonl  GONGFENG_TOKEN=... OPENAI_API_KEY=... \\
  uv run nexus run-batch -g nexus/extensions/tasks/swe_any/generator/stage/eval.py \\
      -d ./outputs/swe_any_eval_batch.db --max-concurrent 4

Environment:
  - SWE_ANY_INSTANCES: path to a ``.jsonl`` of one instance per line (required)
  - SWE_ANY_CONFIG_PATCH: optional comma-separated YAML patches (later wins)
  - SWE_ANY_INSTANCE_IDS / SWE_ANY_SKIP / SWE_ANY_LIMIT / SWE_ANY_LANGUAGE: subset + paging
  - SWE_ANY_OUTPUT_DIR: host artifact dir (default: ./outputs/swe_any)
  - OPENAI_API_KEY / GONGFENG_TOKEN: host LLM credential + sandbox token (secrets)
  - Anything the config references via ``${oc.env:...}``
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
    host_mask_path,
    output_dir,
    repo_name_of,
    selected_instances,
)
from nexus.runner.single import RunInput


_DEFAULT_YAML = STAGE_CONFIG_DIR / 'eval.yaml'


def _selected_eval_instances(spec: dict[str, Any]) -> Iterator[dict[str, Any]]:
    """Yield only instances with a host ``mask.jsonl`` — Eval's hard precondition.

    Mask's masked features live in that unified file; without it there is nothing to
    build tests for, so skipping the instance turns a guaranteed no-op into a quiet skip.
    """
    out_dir = output_dir()
    for inst in selected_instances(spec):
        if host_mask_path(inst, out_dir=out_dir).is_file():
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
    task_config['mask_path'] = str(host_mask_path(inst))
    task_config['output_dir'] = output_dir()
    task_config['image'] = inst['image']
    task_config['repo_dir'] = repo_dir
    # Batch eval releases each sandbox after writing archives.
    task_config['release_runtime'] = True

    # Provider timeout applies to each runtime invoke (the test-design agent is the long one).
    provider_config = config['runtime_provider_config']
    provider_config['request_timeout'] = max(
        float(resolved['runtime_provider_config']['request_timeout']),
        agent_timeout + 600.0,
    )

    attach_cache_task_id(config, inst['instance_id'])
    return RunInput.from_dict(config)


def generate_configs() -> Iterator[BatchItem]:
    base, spec = agentic_config_and_spec(_DEFAULT_YAML)
    for inst in _selected_eval_instances(spec):
        instance_id = str(inst['instance_id'])
        yield BatchItem(
            config=_build_config(base, inst),
            run_id=instance_id,
            metadata={
                'instance_id': instance_id,
                'repo_name': inst.get('repo_name'),
                'language': inst.get('language'),
                'stage': 'eval',
                'mask_path': str(host_mask_path(inst)),
            },
        )


def get_total_count() -> int:
    _, spec = agentic_config_and_spec(_DEFAULT_YAML)
    return sum(1 for _ in _selected_eval_instances(spec))


def get_batch_metadata() -> dict[str, Any]:
    _, spec = agentic_config_and_spec(_DEFAULT_YAML)
    return {
        'task_name': 'swe_any_eval',
        'stage': 'eval',
        'instances': spec.get('path', ''),
        'language': spec.get('language', ''),
        'output_dir': output_dir(),
        'config_patch': ','.join(config_patch_paths()),
    }
