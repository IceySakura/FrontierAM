#!/usr/bin/env python3
"""Split one V1 generator run_input into the platform's two halves.

The split tables describe V1 task configuration. The platform assembles each run from two
places, and every key has to live in exactly one of them:

  dataset row        per-task. Platform requires ``id`` + ``meta_data_``, with
                     ``meta_data_.nexus_run_input`` carrying only what differs
                     between tasks: ``domain`` and ``task_id``.

  eval-set config    common to the whole batch -- imports, task_name, runtime,
                     agent, and the ``task_config`` keys that are identical for
                     all 361 tasks. Pasted into 评估集自定义参数
                     ``nexus_run_input``; secrets stay as ``${oc.env:...}`` and
                     resolve from 七彩石.

The model layer belongs to NEITHER: the platform rebuilds
``openai_clients.main`` from three fields on the evaluation TASK, so
``--client-form-out`` emits those instead (see ``_client_form``).

Both halves are derived from the same ``generate_configs()`` output rather than
hand-written, so a platform run and a local ``nexus run-batch`` submit the same
run_input -- which is what makes the offline score a valid baseline to align
against. The script asserts the split is exhaustive and non-overlapping, so a
new generator field cannot silently go missing from both halves.

Usage::

    export OSWORLD_DATASET_URI=hy_gui_eval/osworld@v1.1.1
    export OSWORLD_DATASET_SPLIT=test
    # Configure model/harness and export WEDATA_KEY via the secret provider.
    python -m nexus.extensions.tasks.os_world.scripts.export_platform_dataset \\
        --generator nexus.extensions.tasks.os_world.generators.generator_native \\
        -o /data/workspace/tmp/osworld_v1_dataset.jsonl \\
        --eval-set-out /data/workspace/tmp/osworld_v1_eval_set.json \\
        --client-form-out /data/workspace/tmp/osworld_v1_client_form.json
"""

from __future__ import annotations

import argparse
import copy
import importlib
import inspect
import json
import sys

from pathlib import Path
from typing import Any, get_type_hints

from nexus.core.agent import agent_registry


# The two task_config keys that differ per task. V1's ``domain`` is a real axis
# of the run (it selects the application the task lives under and is part of the
# on-disk path), not display metadata -- which is why this table has one more
# entry than V2's.
PER_TASK_TASK_CONFIG_KEYS = ('domain', 'task_id')

# Paths outside task_config that a single task is allowed to differ on.
# ``sandbox_policy`` routes two of the 369 standard tasks (the multi_apps
# instructor task and the gimp upscale one) to an 8c/32g sandbox, which changes
# the tool, the runtime provider, and -- because that provider lives in its own
# module -- the import list.
#
# A plain dataset row cannot express this: the row is the platform's
# ``customize`` layer and loses to the eval set's ``dataset_version`` layer, so
# all 369 tasks would silently take the eval set's tool. The values are
# therefore emitted with the platform's ``__fix__`` suffix, which higher layers
# skip and the last merge round strips (param_utils.py:156-254).
#
# Leaf paths rather than whole blocks: ``runtime_provider_config`` also holds
# the ``${oc.env:AGS_SECRET_ID}`` references, and interpolation is not resolved
# at the customize layer -- copying that block into a row would hand the worker
# the literal string.
PER_TASK_OVERRIDE_PATHS = (
    'imports',
    'runtime_provider_name',
    'runtime_provider_config.cli_harness',
    'runtime_configs.main.tool_id',
    'runtime_configs.main.tool_name',
)

# param_utils.py:156-254. A row value without it is discarded in favour of the
# eval set's, and nothing says so.
_FIX_SUFFIX = '__fix__'

# Execution-local paths never identify platform data: task_files does.
DATASET_LOCAL_PATH_KEYS = ('task_root', 'task_file', 'cache_dir', 'setup_cache_dir')

# V2 needs a third table here for host paths whose field default is already right
# for the platform (its scratch cache, its mode-600 key file). V1 has neither:
# ``proxy_config_file`` is the only other path in its task_config and the
# generator already emits it as ``${oc.env:OSWORLD_PROXY_CONFIG_FILE}``, so it
# travels as a reference without help. The table stays, empty, because
# ``_assert_split_is_total`` subtracts it -- a V1 key that later wants dropping
# has a place to be declared instead of quietly vanishing from both halves.
LOCAL_ONLY_TASK_CONFIG_KEYS: tuple[str, ...] = ()

# Platform-managed, per Agent-Bench spec 2.3: the platform injects trajectory
# recording and token accounting itself. Emitting our own would either be
# ignored or fight with it, and `runner.timeout` is owned by the Server layer.
#
# `openai_clients` is here because the platform REBUILDS it: `build_params`
# overwrites ``openai_clients['main']`` with ``_build_main_model_config(...)``,
# assembled from the task's own 「外部服务URL」 form (url / model name /
# parameter_configuration). A block left here is therefore either overwritten or
# -- when the form is incomplete -- the reason the run dies with "Failed to build
# main model config". The values are not lost: they move to 输出二, where
# base_url becomes 推理服务URL, request_args.model becomes 模型名称, and
# client_args (with api_key lifted one level up, beside main_client_args rather
# than inside it) becomes parameter_configuration. Dropping it here is also what
# keeps the model credential out of the uploaded config: for cua_gpt_agent that
# is the ``${oc.env:GPT_CUA_API_KEY,...}`` chain, and no part of it survives into
# either half. See `_client_form`.
PLATFORM_MANAGED_KEYS = ('processors', 'runner', 'run_id', 'openai_clients')

MAX_UPLOAD_BYTES = 100 * 1024 * 1024

# Never written to disk. The real key lives in the platform's create-task form
# (and in 七彩石 for the local/offline path); a file under docs/ that is also a
# committed artifact is the last place it should appear.
API_KEY_PLACEHOLDER = '<REDACTED: 在平台表单填, 真值在七彩石 / 平台 UI, 勿落盘>'

# client_args keys that are not part of 推理超参 because the form has its own
# field for them: base_url is 推理服务URL, api_key is a SIBLING of
# main_client_args (see _client_form), and cache_task_id is per-task -- the
# platform has no per-row hook for it, and a batch-common value would make all
# 361 tasks share one prefix-cache bucket.
_FORM_OWNED_CLIENT_ARGS = ('base_url', 'api_key', 'api_key__env', 'cache_task_id', 'prompt_cache_task_id')

# nexus/runner/single.py:58. Kept as a literal rather than imported so this
# script stays runnable without the runner's import chain.
_ENV_SUFFIX = '__env'

# Omit runtime-derived values from the common evaluation configuration.
#
# Upstream routing is decided by the task's model configuration. An agent model
# can nevertheless select a CLI/catalog identity (Codex); _drop_derivable keeps
# that harness field even when it happens to equal today's default.
_DERIVABLE_AGENT_CONFIG_KEYS = ('model',)

# Mirrors AgsProvider._infer_tool_type for the only kind V1 produces
# (`providers/ags/provider.py:624`). `domain` is the other derived one:
# `domain or f'{region}.tencentags.com'` at `:316`.
_TOOL_TYPE_BY_RUNTIME_KIND = {'osworld': 'osworld'}

# `_infer_launch_mode` runs before `_infer_tool_type`; osworld derives native
# launch mode from its runtime kind without an explicit launch-mode field.
_LAUNCH_MODE_BY_RUNTIME_KIND = {'osworld': 'native'}

# Loopback endpoints identify the exporting host, not the platform worker.
# Reject them in exported model-service configuration.
_HOST_LOCAL_HOSTS = ('localhost', '127.0.0.1', '0.0.0.0', '::1')


def _extract_keys(task_config: dict[str, Any], keys: tuple[str, ...]) -> dict[str, Any]:
    return {key: task_config[key] for key in keys if key in task_config}


def _drop_keys(task_config: dict[str, Any], keys: tuple[str, ...]) -> None:
    for key in keys:
        task_config.pop(key, None)


_MISSING = object()


def _at_path(config: dict[str, Any], path: str) -> Any:
    """The value at a dotted path, or ``_MISSING`` -- distinct from a stored None."""
    node: Any = config
    for part in path.split('.'):
        if not isinstance(node, dict) or part not in node:
            return _MISSING
        node = node[part]
    return node


def _set_fixed(row: dict[str, Any], path: str, value: Any) -> None:
    """Write ``value`` at a dotted path with the leaf renamed ``<leaf>__fix__``."""
    *parents, leaf = path.split('.')
    node = row
    for part in parents:
        node = node.setdefault(part, {})
    node[f'{leaf}{_FIX_SUFFIX}'] = value


def _overrides(config: dict[str, Any], baseline: dict[str, Any]) -> dict[str, Any]:
    """The ``__fix__`` block a task needs to keep its own sandbox on the platform.

    Empty for the 367 tasks that match the eval-set half, which is the point:
    only the routed ones pay for the escape hatch.
    """
    row: dict[str, Any] = {}
    for path in PER_TASK_OVERRIDE_PATHS:
        value = _at_path(config, path)
        if value is not _MISSING and value != _at_path(baseline, path):
            _set_fixed(row, path, value)
    return row


def _agent_defaults(agent_name: str, imports: list[str]) -> dict[str, Any]:
    """Field defaults of the agent's own input model, keyed by field name.

    The generator's ``imports`` are what a run would load to populate the
    registry; the exporter never runs that step, so it does it here.
    """
    for module in imports:
        importlib.import_module(module)
    agent = agent_registry.get(agent_name)
    parameter = next(iter(inspect.signature(agent).parameters.values()))
    input_type = get_type_hints(agent).get(parameter.name, parameter.annotation)
    return {name: field.default for name, field in input_type.model_fields.items()}


def _drop_derivable(dumped: dict[str, Any]) -> list[str]:
    """Strip the values the platform's own code already derives. Returns what went.

    Only the eval-set half is trimmed. The offline ``run-batch`` config keeps all
    of it, because locally these are not restatements at all: ``domain`` can point
    at a non-default AGS region. On the platform the same keys are answered by
    输出二 and by the provider's own inference, and the eval set is the one copy
    nobody updates when the task's model changes.

    A model name is only dropped when it equals the agent's own field default --
    exporting an ``OSWORLD_CLAUDE_MODEL=something-else`` batch would otherwise
    hand the platform a config that quietly reverts.
    """
    dropped = []
    agent_config = dumped.get('agent_config')
    if isinstance(agent_config, dict):
        defaults = _agent_defaults(dumped['agent_name'], dumped.get('imports') or [])
        trimmed = dict(agent_config)
        for key in _DERIVABLE_AGENT_CONFIG_KEYS:
            # Codex's model selects the CLI/catalog identity, not upstream
            # routing. Preserve the entire harness contract on the platform.
            if key == 'model' and dumped['agent_name'] == 'codex_native_bridge':
                continue
            if key in trimmed and trimmed[key] == defaults.get(key):
                trimmed.pop(key)
                dropped.append(f'agent_config.{key}')
        dumped['agent_config'] = trimmed

    provider_config = dumped.get('runtime_provider_config')
    if isinstance(provider_config, dict) and 'domain' in provider_config:
        region = provider_config.get('region')
        if provider_config['domain'] == f'{region}.tencentags.com':
            dumped['runtime_provider_config'] = {k: v for k, v in provider_config.items() if k != 'domain'}
            dropped.append('runtime_provider_config.domain')

    runtime_configs = dumped.get('runtime_configs')
    if isinstance(runtime_configs, dict):
        trimmed_runtimes = {}
        for name, runtime in runtime_configs.items():
            kind = (runtime or {}).get('runtime_kind')
            for key, table in (
                ('tool_type', _TOOL_TYPE_BY_RUNTIME_KIND),
                ('launch_mode', _LAUNCH_MODE_BY_RUNTIME_KIND),
            ):
                if isinstance(runtime, dict) and runtime.get(key) == table.get(kind):
                    runtime = {k: v for k, v in runtime.items() if k != key}
                    dropped.append(f'runtime_configs.{name}.{key}')
            trimmed_runtimes[name] = runtime
        dumped['runtime_configs'] = trimmed_runtimes

    return dropped


def _interpolate_env_suffixes(value: Any) -> Any:
    """Rewrite ``foo__env: "VAR"`` as ``foo: "${oc.env:VAR}"`` for the eval set.

    The two spellings mean the same thing locally and NOT on the platform, which
    is the whole reason this exists:

      ``foo__env`` is resolved by the nexus worker (`single.py:413`,
      `_resolve_run_input_env_suffixes`) out of ``os.environ``.

      ``${oc.env:VAR}`` is resolved by the PLATFORM, before the run_input is ever
      submitted, through an overridden ``oc.env`` resolver that reads a
      ``ContextVar`` populated from 七彩石 (`eval_class/.../param_utils.py:132-152`).

    Rainbow's values go into that ContextVar and are deliberately never exported
    to ``os.environ`` ("避免全局 os.environ 污染"). So a ``__env`` key in a platform
    eval set reads an environment the worker does not have: the run dies on
    ``Environment variable 'AGS_SECRET_ID' ... is not set`` even though the key is
    sitting in 七彩石.

    The V1 generators emit ``${oc.env:}`` directly and never the ``__env``
    spelling, so this is a no-op safety net rather than a live rewrite -- kept
    because it is the reason V1 chose ``${oc.env:}`` in the first place.
    """
    if isinstance(value, dict):
        out: dict[Any, Any] = {}
        for key, child in value.items():
            text = str(key)
            if text.endswith(_ENV_SUFFIX) and isinstance(child, str) and child.strip():
                out[text.removesuffix(_ENV_SUFFIX)] = f'${{oc.env:{child.strip()}}}'
                continue
            out[key] = _interpolate_env_suffixes(child)
        return out
    if isinstance(value, list):
        return [_interpolate_env_suffixes(item) for item in value]
    return value


def _dataset_row(item: Any, baseline: dict[str, Any] | None = None) -> dict[str, Any]:
    config = item.config
    if config.task_files is None or config.task_files.remote is None:
        raise ValueError('Native OSWorld platform export requires remote COS task_files')
    task_config = config.task_config or {}
    metadata = item.metadata or {}

    run_input: dict[str, Any] = {
        'task_config': _extract_keys(task_config, PER_TASK_TASK_CONFIG_KEYS),
    }
    # ``baseline`` is the item the eval-set half was built from; without one
    # there is nothing to differ from and the row carries no overrides.
    if baseline is not None:
        run_input.update(_overrides(config.model_dump(exclude_none=True), baseline))
    if config.task_files is not None:
        # exclude_none keeps rows compact. `size` must survive: both remote
        # readers derive the zip's central-directory offset from it and raise
        # 'Remote task files zip requires size metadata' without it.
        run_input['task_files'] = config.task_files.model_dump(exclude_none=True)

    # Siblings of nexus_run_input, not inside it: these are for the platform's
    # detail view and must not be merged into run_input. V1 instructions are
    # public benchmark text (unlike V2's, which ship in a gated task class), so
    # the row carries them as the label a reviewer needs to recognise a task.
    meta: dict[str, Any] = {'nexus_run_input': run_input}
    if metadata.get('instruction'):
        meta['instruction'] = metadata['instruction']
    if metadata.get('domain'):
        meta['domain'] = metadata['domain']

    # `id` mirrors the offline run_id (<domain>__<task_id>) so a platform result
    # can be matched against the offline baseline row by row during alignment.
    return {'id': item.run_id, 'meta_data_': meta}


def _eval_set_config(item: Any) -> tuple[dict[str, Any], list[str]]:
    """Everything that is identical across the batch.

    Returns the config plus the keys that were pruned as empty, so the
    exhaustiveness check can tell "deliberately dropped" from "went missing".
    """
    if item.config.task_files is None or item.config.task_files.remote is None:
        raise ValueError('Native OSWorld platform export requires remote COS task_files')
    dumped = item.config.model_dump(exclude_none=True)
    task_config = dict(dumped.pop('task_config', None) or {})
    dumped.pop('task_files', None)  # per-task, lives in the dataset row
    for key in PLATFORM_MANAGED_KEYS:
        dumped.pop(key, None)
    # Empty containers the model defaults in; noise in a config you paste by hand.
    pruned = [k for k, v in dumped.items() if v in ({}, [])]
    for key in pruned:
        dumped.pop(key)

    _drop_keys(task_config, PER_TASK_TASK_CONFIG_KEYS)
    _drop_keys(task_config, LOCAL_ONLY_TASK_CONFIG_KEYS)
    # Host caches/trajectory roots are execution-local, not data sources.
    _drop_keys(task_config, DATASET_LOCAL_PATH_KEYS)

    dumped['task_config'] = task_config
    return _interpolate_env_suffixes(dumped), pruned


def _assert_split_is_total(
    item: Any,
    row: dict[str, Any],
    common: dict[str, Any],
    pruned: list[str],
) -> list[str]:
    """Every generator field must land in exactly one half (minus local-only).

    Guards the failure mode that has no visible symptom: a field the generator
    starts emitting that ends up in neither half, so the platform silently runs
    with a default while the offline baseline does not.
    """
    full = item.config.model_dump(exclude_none=True)
    row_run_input = row['meta_data_']['nexus_run_input']

    problems: list[str] = []

    top_level = set(full) - {'task_config', 'task_files'}
    missing_top = sorted(top_level - set(common) - set(PLATFORM_MANAGED_KEYS) - set(pruned))
    if missing_top:
        problems.append(f'top-level keys in neither half: {missing_top}')

    full_tc = set(full.get('task_config') or {})
    covered_tc = set(common.get('task_config') or {}) | set(row_run_input.get('task_config') or {})
    local_keys = set(LOCAL_ONLY_TASK_CONFIG_KEYS)
    if item.config.task_files is not None:
        local_keys.update(DATASET_LOCAL_PATH_KEYS)
    missing_tc = sorted(full_tc - covered_tc - local_keys)
    if missing_tc:
        problems.append(f'task_config keys in neither half: {missing_tc}')

    # Both halves is worse than neither for a per-task key: the dataset row is
    # the platform's `customize` layer, which loses to the eval set's
    # `dataset_version` layer, so a duplicated key means every task silently
    # runs with the value baked in from whichever task was exported first.
    overlap = sorted(set(common.get('task_config') or {}) & set(row_run_input.get('task_config') or {}))
    if overlap:
        problems.append(f'task_config keys in both halves: {overlap}')

    if ('task_files' in full) != ('task_files' in row_run_input):
        problems.append('task_files present in the generator output but not in the dataset row')
    if 'task_files' in common:
        problems.append('task_files must be per-task only; dataset_version would override the row')
    if full.get('task_files') != row_run_input.get('task_files'):
        problems.append('dataset row task_files differs from the local run')
    if '${' in json.dumps(row_run_input, ensure_ascii=False):
        problems.append('dataset row must not contain interpolation; only dataset_version resolves it')

    # An override written without the suffix reads as a normal customize-layer
    # key, which the eval set overwrites -- the task runs on the wrong sandbox
    # and the row that says otherwise is right there in the dataset.
    overrides = {k: v for k, v in row_run_input.items() if k not in {'task_config', 'task_files'}}
    unsuffixed = sorted(path for path in _leaf_paths(overrides) if not path.endswith(_FIX_SUFFIX))
    if unsuffixed:
        problems.append(f'per-task overrides the eval set would overwrite: {unsuffixed}')

    return problems


def _assert_batch_common(items: list[Any]) -> list[str]:
    """The eval-set half is derived from one item, so it must hold for all of them.

    ``_assert_split_is_total`` compares key spaces and would pass happily while
    every task ran with the first task's value -- exactly the failure the split
    exists to prevent, and one that produces no error, just 361 identical runs.
    Anything else that differs between items is a new per-task key that belongs
    in PER_TASK_TASK_CONFIG_KEYS or PER_TASK_OVERRIDE_PATHS, or an execution-local
    path that belongs in DATASET_LOCAL_PATH_KEYS.
    """
    first = items[0].config.model_dump(exclude_none=True)
    batch_common = set(first) - {'task_files', *PLATFORM_MANAGED_KEYS}
    # A top-level key whose every difference is already spelled out as a
    # ``__fix__`` override travels per task; only the rest is a real conflict.
    varying = {
        key for key in batch_common for other in items[1:] if _at_path(_with_overrides(other, first), key) != first[key]
    }
    task_config_varying = {
        key
        for key in (first.get('task_config') or {})
        for other in items[1:]
        if (other.config.task_config or {}).get(key) != first['task_config'][key]
    } - set(PER_TASK_TASK_CONFIG_KEYS)

    problems = []
    # ``task_config`` always differs -- that is what the dataset row is for; the
    # keys inside it are checked separately just below.
    if conflicting := varying - {'task_config'}:
        problems.append(f'top-level keys differ between tasks but the eval set carries only one: {sorted(conflicting)}')
    if task_config_varying:
        problems.append(f'task_config keys differ between tasks but are not per-task: {sorted(task_config_varying)}')
    return problems


def _leaf_paths(value: Any, path: str = '') -> list[str]:
    """Dotted paths to every non-dict value. A list is a leaf -- ``imports`` is one."""
    if isinstance(value, dict):
        return [leaf for key, child in value.items() for leaf in _leaf_paths(child, f'{path}.{key}'.lstrip('.'))]
    return [path]


def _with_overrides(item: Any, baseline: dict[str, Any]) -> dict[str, Any]:
    """``item``'s config with every declared override path rewound to the baseline.

    What is left differs for a reason the split has no answer for.
    """
    config = copy.deepcopy(item.config.model_dump(exclude_none=True))
    for path in PER_TASK_OVERRIDE_PATHS:
        if _at_path(config, path) is _MISSING:
            continue
        *parents, leaf = path.split('.')
        node = config
        for part in parents:
            node = node[part]
        expected = _at_path(baseline, path)
        # A path the baseline does not have at all (the codex harness flag the
        # large-sandbox route adds) rewinds by going away, not to None.
        if expected is _MISSING:
            del node[leaf]
        else:
            node[leaf] = expected
    return config


def _assert_no_host_local_urls(common: dict[str, Any]) -> list[str]:
    """No batch-common value may point at a loopback address.

    ``agent_config.upstream_base_url`` is the live case: the V1 claude harness
    talks to a gateway the developer runs beside the batch, and a loopback URL
    exported into an eval set reaches the platform worker as a connection refused
    from inside the agent, naming nothing that would lead back to this config.
    The URL has no host-path table of its own because the fix is not a 七彩石
    reference -- the gateway has to be reachable from the platform first.
    """
    found = sorted(
        f'{path}={value}' for path, value in _string_leaves(common) if any(host in value for host in _HOST_LOCAL_HOSTS)
    )
    return [f'batch-common values point at this machine only: {found}'] if found else []


def _string_leaves(value: Any, path: str = '') -> list[tuple[str, str]]:
    if isinstance(value, dict):
        return [leaf for key, child in value.items() for leaf in _string_leaves(child, f'{path}.{key}'.lstrip('.'))]
    if isinstance(value, list):
        return [leaf for i, child in enumerate(value) for leaf in _string_leaves(child, f'{path}[{i}]')]
    return [(path, value)] if isinstance(value, str) else []


def _client_form(item: Any) -> dict[str, Any]:
    """The 「外部服务URL」 form, derived from the generator's ``openai_clients.main``.

    This is 输出二. The platform does not read a model out of the eval-set
    config: ``build_params`` overwrites ``openai_clients['main']`` with
    ``_build_main_model_config(...)``, which it assembles from three form fields
    on the task. So the same client block that drives an offline ``run-batch``
    has to be re-expressed as that form, and deriving it here is what keeps the
    two from drifting when the model or gateway changes.

    The mapping (skill §4A.1)::

        client_args.base_url        -> 推理服务URL
        request_args.model          -> 模型名称
        openai_clients.main.name    -> parameter_configuration.nexus_main_client_name
        client_args (the rest)      -> parameter_configuration.main_client_args
        client_args.api_key         -> parameter_configuration.api_key   <- SIBLING

    ``api_key`` being a sibling of ``main_client_args`` rather than a child is
    the one part of this that looks wrong and isn't: nested, it still reaches
    the client (``_deep_update`` merges main_client_args into client_args), but
    the platform's own credential-injection path reads the sibling, so the
    nested spelling authenticates with whatever the platform defaulted to --
    typically a 401 with a key that is visibly present in the form.

    ``request_args`` beyond the model is NOT part of the form: the platform
    funnels those through ``decoding_params``, which is a different field with
    its own blacklist. Left out deliberately -- see the returned ``_comment``.
    """
    clients = (item.config.openai_clients or {}) if hasattr(item.config, 'openai_clients') else {}
    main = clients.get('main')
    if main is None:
        return {}
    dumped = main.model_dump(exclude_none=True) if hasattr(main, 'model_dump') else dict(main)
    config = dict(dumped.get('config') or {})
    client_args = dict(config.get('client_args') or {})
    request_args = dict(config.get('request_args') or {})
    if config.get('enforced_request_args'):
        raise ValueError(
            'The external-URL form cannot represent OpenAIConfig.enforced_request_args. '
            'Use --model-source platform-service and configure its model service explicitly; '
            'do not silently drop enforced request policy.'
        )

    carried = {k: v for k, v in client_args.items() if k not in _FORM_OWNED_CLIENT_ARGS}
    leftover_request_args = {k: v for k, v in request_args.items() if k != 'model'}

    return {
        '_comment': [
            '输出二 —— 创建评估任务时填的「外部服务URL」三个字段。不要粘进评估集自定义参数:',
            '平台的 _build_main_model_config 会用这三项重建 openai_clients.main,',
            '评估集里再写一份只会被覆盖(或在表单不全时让整个任务死在',
            '"Failed to build main model config")。',
            '',
            'api_key 与 main_client_args 同级, 不是它的子键 —— 嵌进去也能到客户端,',
            '但平台自己的凭证注入路径读不到, 症状是"key 明明填了却 401"。',
            '',
            f'request_args 里除 model 之外的项({sorted(leftover_request_args) or "无"})不在此表单:',
            '平台走 decoding_params 这条独立通道(且有黑名单)。需要它们时在任务的',
            '推理参数里单独填, 不要塞进 main_client_args。',
        ],
        '推理服务URL': client_args.get('base_url', ''),
        '模型名称': request_args.get('model', ''),
        'parameter_configuration': {
            'api_key': API_KEY_PLACEHOLDER,
            'nexus_main_client_name': dumped.get('name', ''),
            'main_client_args': carried,
        },
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('-o', '--out', required=True, help='Dataset .jsonl for platform upload')
    parser.add_argument(
        '--model-source',
        choices=('external-url', 'platform-service'),
        default='external-url',
        help='platform-service leaves main client creation to Taiji (ONE_STOP_SERVICE/MODEL_REPOSITORY)',
    )
    parser.add_argument(
        '--eval-set-out',
        help='Write the batch-common nexus_run_input here, for 评估集自定义参数',
    )
    parser.add_argument(
        '--client-form-out',
        help=(
            'Write the 「外部服务URL」 form here (推理服务URL / 模型名称 / 推理超参), '
            'i.e. the model layer the platform rebuilds from the task rather than the eval set. '
            'api_key is emitted as a placeholder.'
        ),
    )
    parser.add_argument(
        '--generator',
        default='nexus.extensions.tasks.os_world.generators.generator_native',
        help=(
            'Generator module to export. Defaults to the native entry; legacy exports '
            'must explicitly select their legacy generator.'
        ),
    )
    args = parser.parse_args(argv)
    if args.model_source == 'platform-service' and args.client_form_out:
        parser.error('--client-form-out is for external-url, not a platform-managed model service')

    generator = importlib.import_module(args.generator)

    items = list(generator.generate_configs())
    if not items:
        print('error: generator produced no tasks', file=sys.stderr)
        return 2

    # The eval-set half speaks for items[0], so that is what a row differs from.
    baseline = items[0].config.model_dump(exclude_none=True)
    rows = [_dataset_row(i, baseline) for i in items]
    common, pruned = _eval_set_config(items[0])
    derivable = _drop_derivable(common)
    # The local InferProxy SDK knobs are not platform bridge_litellm arguments.
    # Neither is emitted in common; only an external-URL export needs a form.
    # Keep agent imports/config intact so CC/Codex handles either injected Chat client.

    problems = (
        [
            problem
            for item, row in zip(items, rows, strict=True)
            for problem in _assert_split_is_total(item, row, common, pruned)
        ]
        + _assert_batch_common(items)
        + _assert_no_host_local_urls(common)
    )
    if problems:
        for p in problems:
            print(f'error: {p}', file=sys.stderr)
        return 2

    payload = '\n'.join(json.dumps(r, ensure_ascii=False) for r in rows) + '\n'
    out_path = Path(args.out).expanduser()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(payload, encoding='utf-8')
    size = len(payload.encode('utf-8'))
    print(f'dataset:  {len(rows)} rows -> {out_path} ({size / 1024:.0f} KiB)')
    routed = [r['id'] for r in rows if set(r['meta_data_']['nexus_run_input']) - {'task_config', 'task_files'}]
    if routed:
        print(f'          {len(routed)} rows override the batch sandbox: {", ".join(routed)}')
    if size > MAX_UPLOAD_BYTES:
        print(f'warning: exceeds the {MAX_UPLOAD_BYTES // 1024 // 1024} MB local-upload limit', file=sys.stderr)

    if args.eval_set_out:
        eval_path = Path(args.eval_set_out).expanduser()
        eval_path.parent.mkdir(parents=True, exist_ok=True)
        eval_path.write_text(json.dumps(common, indent=2, ensure_ascii=False) + '\n', encoding='utf-8')
        print(f'eval set: {len(common)} top-level keys -> {eval_path}')
        if derivable:
            print(f'          dropped as code-derivable: {", ".join(derivable)}')

    if args.client_form_out:
        form = _client_form(items[0])
        if not form:
            print('error: the generator produced no openai_clients.main to derive the form from', file=sys.stderr)
            return 2
        form_path = Path(args.client_form_out).expanduser()
        form_path.parent.mkdir(parents=True, exist_ok=True)
        form_path.write_text(json.dumps(form, indent=2, ensure_ascii=False) + '\n', encoding='utf-8')
        print(f'client form: {form["模型名称"] or "<no model>"} @ {form["推理服务URL"] or "<no url>"} -> {form_path}')

    return 0


if __name__ == '__main__':
    raise SystemExit(main())
