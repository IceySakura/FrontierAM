"""Native OSWorld generator: independent model protocol and harness choices.

The batch API and the osworld_native run part call build_item without
mutating the process environment. The six modes share one
task/runtime configuration and one evaluator implementation.
"""

from __future__ import annotations

import hashlib
import os
import re

from collections.abc import Iterator
from dataclasses import dataclass
from typing import Any

from nexus import BatchItem, RunInput
from nexus.dataset.auth import WedataKey
from nexus.extensions.tasks.os_world.data import task_source
from nexus.extensions.tasks.os_world.data.dataset_files import apply_dataset_task_files, dataset_index
from nexus.extensions.tasks.os_world.execution.sandbox_policy import (
    apply_sandbox_policy,
    sandbox_metadata,
    uses_large_sandbox,
)
from nexus.extensions.tasks.os_world.generators import config as baseline
from nexus.extensions.tasks.os_world.generators.harness import cc_config, codex_config


SUPPORTED = frozenset({
    ('gpt', 'bare'),
    ('claude', 'bare'),
    ('gpt', 'codex'),
    ('claude', 'cc'),
    ('hunyuan', 'codex'),
    ('hunyuan', 'cc'),
})


@dataclass(frozen=True)
class Selection:
    model_family: str
    harness: str


def selection(*, model_family: str | None = None, harness: str | None = None) -> Selection:
    family = model_family or os.getenv('OSWORLD_MODEL_FAMILY', '').strip()
    chosen = harness or os.getenv('OSWORLD_HARNESS', '').strip()
    if (family, chosen) not in SUPPORTED:
        raise ValueError(
            'Set OSWORLD_MODEL_FAMILY and OSWORLD_HARNESS to a supported pair: '
            'gpt/bare, claude/bare, gpt/codex, claude/cc, hunyuan/codex, hunyuan/cc'
        )
    return Selection(family, chosen)


def _required(name: str) -> str:
    value = os.getenv(name, '').strip()
    if not value:
        raise ValueError(f'{name} is required by the native OSWorld generator')
    return value


def _credential(name: str) -> str:
    _required(name)
    return '${oc.env:' + name + '}'


def _model_client(task: task_source.OSWorldBatchTask, choice: Selection) -> dict[str, Any]:
    if choice.model_family == 'hunyuan':
        from nexus.extensions.tasks.os_world.generators.clients import _hunyuan_openai_clients

        _required('HUNYUAN_MODEL')
        _required('HUNYUAN_BASE_URL')
        return _hunyuan_openai_clients(task)['main']

    client_args: dict[str, Any] = {
        'base_url': _required('NEXUS_BASE_URL'),
        'api_key': _credential('NEXUS_API_KEY'),
    }
    request_args: dict[str, Any] = {'model': _required('NEXUS_MODEL')}
    if choice.model_family == 'claude':
        name = 'anthropic_messages'
        client_args.update(
            api_style=os.getenv('NEXUS_ANTHROPIC_API_STYLE', 'messages'),
            timeout=task_source.env_float('NEXUS_LLM_TIMEOUT', 900.0),
            max_retries=task_source.env_int('NEXUS_LLM_MAX_RETRIES', 3),
            cache_task_id=hashlib.md5(task.task_id.encode()).hexdigest(),
            cache_control_enabled=True,
            cache_control_strategy='claude_code',
        )
        provider = os.getenv('NEXUS_LLM_PROVIDER', '').strip()
        if provider:
            client_args['provider'] = provider
    elif choice.harness == 'bare':
        name = 'openai_responses'
        client_args.update(
            ensure_v1_path=True,
            capture_raw_io=True,
            max_retries=task_source.env_int('NEXUS_LLM_MAX_RETRIES', 5),
        )
    else:
        name = 'openai_responses'
        client_args.update(
            timeout=task_source.env_float('NEXUS_LLM_TIMEOUT', 3600.0),
            max_retries=task_source.env_int('NEXUS_LLM_MAX_RETRIES', 5),
            prompt_cache_task_id=os.getenv('OSWORLD_SESSION_NAMESPACE', '') + '/' + task.task_id,
        )
        request_args.update(
            reasoning={'effort': os.getenv('CODEX_REASONING_EFFORT', 'none')},
            max_output_tokens=task_source.env_int('CODEX_MAX_OUTPUT_TOKENS', 64000),
        )
        summary = os.getenv('CODEX_REASONING_SUMMARY', 'none')
        if summary != 'none':
            request_args['reasoning']['summary'] = summary
    return {'name': name, 'config': {'client_args': client_args, 'request_args': request_args}}


def _agent(task: task_source.OSWorldBatchTask, choice: Selection) -> tuple[str, dict[str, Any]]:
    if choice.harness == 'bare':
        if choice.model_family == 'gpt':
            name = 'cua_gpt_agent'
            return name, baseline.gpt_cua_config(task)
        return 'cua_anthropic_agent', baseline.claude_cua_config()
    if choice.harness == 'codex':
        return 'codex_native_bridge', codex_config()
    return 'claude_code_native_bridge', cc_config()


def _tasks(*, task_id: str | None = None) -> list[task_source.OSWorldBatchTask]:
    # A missing dataset must never quietly switch this entry to a personal
    # examples checkout.
    if dataset_index() is None:
        raise ValueError('Set OSWORLD_DATASET_URI for the native OSWorld entry')
    if task_id is not None:
        if not re.fullmatch(r'[A-Za-z0-9][A-Za-z0-9_-]*', task_id):
            raise ValueError('task_id must name exactly one task')
        tasks = list(task_source.iter_selected_tasks(task_ids={task_id}, skip=0, limit=1))
        if len(tasks) != 1:
            raise ValueError('Selected task_id is absent from the dataset or excluded by the domain/curated selection')
    else:
        tasks = list(task_source.iter_selected_tasks())
    if not tasks:
        raise ValueError('No OSWorld dataset tasks selected')
    return tasks


def build_item(task: task_source.OSWorldBatchTask, choice: Selection) -> BatchItem:
    # A shared task/runtime block makes proxy, credentials and scoring independent
    # of the model/harness; in particular bare Claude no longer loses proxy fields.
    task_config = baseline.build_task_config(task)
    task_config.update(
        client_password=os.getenv('OSWORLD_CLIENT_PASSWORD', 'password'),
        cache_dir=os.getenv('NEXUS_OSWORLD_CACHE_DIR', ''),
        unsupported_evaluator_policy=os.getenv('OSWORLD_UNSUPPORTED_EVALUATOR_POLICY', 'fail'),
    )
    runtime_config = baseline._runtime_configs()
    runtime_config['main']['provision_mode'] = baseline._ags_provision_mode()
    runtime_config['main']['timeout'] = task_source.env_int('OSWORLD_SANDBOX_TIMEOUT', 15300)
    if baseline._ags_provision_mode() == 'template':
        _required('AGS_OSWORLD_TEMPLATE')
        _credential('AGS_API_KEY')
    else:
        _required('AGS_OSWORLD_TOOL_ID')
        _credential('AGS_SECRET_ID')
        _credential('AGS_SECRET_KEY')
    name, agent_config = _agent(task, choice)
    block = _model_client(task, choice)
    _credential('WEDATA_KEY')
    config: dict[str, Any] = {
        'imports': [
            'nexus.extensions.tasks.os_world',
            (
                'nexus.extensions.agents.claude_code.agent'
                if name == 'claude_code_native_bridge'
                else 'nexus.extensions.agents.codex.agent'
                if name == 'codex_native_bridge'
                else f'nexus.extensions.agents.cua.{name}.agent'
            ),
            f'nexus.extensions.openai_clients.{block["name"]}',
        ],
        'task_name': 'os_world',
        'task_config': task_config,
        'runtime_provider_name': 'ags',
        'runtime_provider_config': baseline._runtime_provider_config(),
        'runtime_configs': runtime_config,
        'agent_name': name,
        'agent_config': agent_config,
        'openai_clients': {'main': block},
        'processors': {'chat': {'name': 'chat_tracker', 'args': {'tools_mode': 'branch_last'}}},
        'runner': {'hooks': {'output_contract': {'name': 'osworld_output_contract', 'args': {}}}},
    }
    namespace = os.getenv('OSWORLD_SESSION_NAMESPACE', '').strip()
    if namespace:
        config['runner']['hooks']['run_dir'] = {'name': 'osworld_run_dir', 'args': {'run_dir': namespace}}
    # Control-plane large-task policy needs a tool id, not a template. Never
    # silently replace a caller's template with a different sandbox.
    large_policy = task_source.env_bool('OSWORLD_LARGE_SANDBOX_POLICY', True)
    if baseline._ags_provision_mode() == 'template' and large_policy and uses_large_sandbox(task):
        raise ValueError(
            'Selected task requires the large-sandbox policy; use control_plane, '
            'or explicitly set OSWORLD_LARGE_SANDBOX_POLICY=false for your prepared template'
        )
    if baseline._ags_provision_mode() == 'control_plane' and large_policy:
        config = apply_sandbox_policy(task, config)
    apply_dataset_task_files(config, task_config, task.task_id)
    return BatchItem(
        config=RunInput.model_validate(config),
        run_id=f'{task.domain}__{task.task_id}',
        metadata={
            **task_source.batch_task_metadata(task),
            **(
                sandbox_metadata(task, config)
                if baseline._ags_provision_mode() == 'control_plane' and large_policy
                else {'sandbox_profile': 'caller-selected', 'sandbox_policy_version': 'explicit-override'}
            ),
            'model_family': choice.model_family,
            'harness': choice.harness,
            'routing_model': block['config']['request_args']['model'],
            'generator': 'osworld_native',
        },
    )


def _validate_data_credential() -> None:
    WedataKey.from_raw(_required('WEDATA_KEY'), source='WEDATA_KEY (export the raw JSON inside single quotes)')


def generate_configs(*, model_family: str | None = None, harness: str | None = None) -> Iterator[BatchItem]:
    choice = selection(model_family=model_family, harness=harness)
    _validate_data_credential()
    baseline._validate_evaluator_host_deps()
    for task in _tasks():
        yield build_item(task, choice)


def get_total_count(*, model_family: str | None = None, harness: str | None = None) -> int:
    selection(model_family=model_family, harness=harness)
    _validate_data_credential()
    return len(_tasks())


def get_batch_metadata(*, model_family: str | None = None, harness: str | None = None) -> dict[str, Any]:
    choice = selection(model_family=model_family, harness=harness)
    return {
        'task_name': 'os_world',
        'model_family': choice.model_family,
        'harness': choice.harness,
        'dataset_uri': task_source.dataset_source(),
        'generator': 'osworld_native',
    }


def single_config(*, task_id: str, model_family: str | None = None, harness: str | None = None) -> RunInput:
    choice = selection(model_family=model_family, harness=harness)
    _validate_data_credential()
    baseline._validate_evaluator_host_deps()
    item = build_item(_tasks(task_id=task_id)[0], choice)
    return item.config.model_copy(update={'run_id': item.run_id})
