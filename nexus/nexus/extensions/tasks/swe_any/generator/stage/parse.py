#!/usr/bin/env python3
"""SWE-Any batch generator for ``nexus run-batch`` — Parse at scale.

Consumes a JSONL of **SWE-Env instances** and emits one ``swe_any_parse``
BatchItem per instance. The repo source comes from each instance's baked
``image`` at its ``workdir`` (no upload). Each item indexes the repo and then
runs the agentic seed-policy pass; Select chains off both artifacts.

Each item starts from ``config/stage/parse.yaml`` — the same file
``run-single`` uses — with the ``SWE_ANY_CONFIG_PATCH`` patches deep-merged over
it (comma-separated, later wins) and the LLM-backend patch
(``config/client/<name>.yaml``) for the merged ``openai_clients.main.name``
applied after that (see ``batch_config``). Only per-instance fields and computed
timeout floors are injected here, so exporting any ``${oc.env:...}`` the YAML
references still overrides the default.

Usage:
  SWE_ANY_INSTANCES=./debug/swe_env.jsonl  GONGFENG_TOKEN=... OPENAI_API_KEY=... \\
  uv run nexus run-batch -g nexus/extensions/tasks/swe_any/generator/stage/parse.py \\
      -d ./outputs/swe_any_parse_batch.db --max-concurrent 10

Environment:
  - SWE_ANY_INSTANCES: path to a ``.jsonl`` file containing one instance per line (required)
  - SWE_ANY_CONFIG_PATCH: optional comma-separated YAML patches merged over the
    default config in order, later wins (task knobs + the instances block)
  - SWE_ANY_INSTANCE_IDS: comma-separated instance_ids (or a JSON-list file) to include (default: all)
  - SWE_ANY_SKIP / SWE_ANY_LIMIT: paging over the (filtered) instances
  - SWE_ANY_LANGUAGE: instance filter (default: python; "all" disables it)
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
    output_dir,
    repo_name_of,
    selected_instances,
)
from nexus.runner.single import RunInput


_DEFAULT_YAML = STAGE_CONFIG_DIR / 'parse.yaml'


def _build_config(base: dict[str, Any], inst: dict[str, Any]) -> RunInput:
    config = copy.deepcopy(base)
    resolved = resolve_config(config)

    repo_name = repo_name_of(inst)
    repo_dir = str(inst.get('workdir') or '/testbed')

    # Mirror ParseTaskInput.ensure_request_timeout_covers_parse so the exported
    # config shows the floor the task would apply anyway; the +300s matches the
    # validator's invoke-overhead slack.
    resolved_task = resolved['task_config']
    request_timeout = max(
        float(resolved_task['request_timeout']),
        float(resolved_task['index_timeout']) + float(resolved_task['npm_install_timeout']) + 300.0,
    )
    agent_timeout = float(resolved['agent_config']['timeout'])

    task_config = config['task_config']
    task_config['repo_name'] = repo_name
    task_config['output_dir'] = output_dir()
    # Baked per-repo image from the instance; the task configures the runtime with it.
    task_config['image'] = inst['image']
    # Repo source is baked into that image at its workdir (SWE-Env `workdir`).
    task_config['repo_dir'] = repo_dir
    task_config['project_name'] = inst.get('repo') or repo_name
    task_config['request_timeout'] = request_timeout
    # Batch parse releases each sandbox after downloading its artifacts.
    task_config['release_runtime'] = True

    # The sandbox is held for index + npm install and then the seed-policy agent.
    provider_config = config['runtime_provider_config']
    provider_config['request_timeout'] = max(
        float(resolved['runtime_provider_config']['request_timeout']),
        request_timeout + agent_timeout + 600.0,
    )

    attach_cache_task_id(config, inst['instance_id'])
    return RunInput.from_dict(config)


def generate_configs() -> Iterator[BatchItem]:
    base, spec = agentic_config_and_spec(_DEFAULT_YAML)
    for inst in selected_instances(spec):
        instance_id = str(inst['instance_id'])
        yield BatchItem(
            config=_build_config(base, inst),
            run_id=instance_id,
            metadata={
                'instance_id': instance_id,
                'repo_name': inst.get('repo_name'),
                'language': inst.get('language'),
                'stage': 'parse',
            },
        )


def get_total_count() -> int:
    _, spec = agentic_config_and_spec(_DEFAULT_YAML)
    return sum(1 for _ in selected_instances(spec))


def get_batch_metadata() -> dict[str, Any]:
    _, spec = agentic_config_and_spec(_DEFAULT_YAML)
    return {
        'task_name': 'swe_any_parse',
        'stage': 'parse',
        'instances': spec.get('path', ''),
        'language': spec.get('language', ''),
        'output_dir': output_dir(),
        'config_patch': ','.join(config_patch_paths()),
    }
