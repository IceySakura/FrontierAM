"""Config-merge pipeline for SWE-Any ``run-batch`` generators.

Produces the merged per-instance config (and the resolved ``instances`` spec) each generator
feeds to ``RunInput``; the instance schema / selection / host-path helpers that consume that
spec live in the sibling :mod:`.instance` module.

Configuration pipeline, applied per instance by each generator:

1. ``load_stage_config`` reads the stage's ``config/stage/*.yaml`` — the
   same file ``run-single`` uses, so single and batch share one set of defaults —
   and deep-merges each user patch in ``SWE_ANY_CONFIG_PATCH`` (comma-separated,
   left to right, later wins) over it. Merge flavor is RFC 7386: dicts recurse,
   ``None`` deletes the key, everything else (incl. lists) replaces. Patches are
   meant for task-level knobs only, so several ``custom/*.yaml`` compose by
   permutation (e.g. a cheap-budget patch × a backend patch).
2. ``pop_instance_spec`` lifts the top-level ``instances`` block (source jsonl +
   filters + paging) out of the config, since it is not a ``RunInput`` field.
   The stage config carries this block with ``${oc.env:...}`` defaults, so its
   env twins (``SWE_ANY_INSTANCES`` / ``…_INSTANCE_IDS`` / ``…_LANGUAGE`` /
   ``…_SKIP`` / ``…_LIMIT``) and any patch override resolve the same way as every
   other knob; ``instance.selected_instances`` then drives the batch from the result.
3. ``apply_client_patch`` resolves a probe copy, reads the merged
   ``openai_clients.main.name`` and merges ``config/client/<name>.yaml`` over
   that. The client patch owns the LLM transport mechanics (model naming,
   ``anthropic_version``, ``thinking`` …); a user patch should not have to know
   them, and it runs after the user patch so it always wins. A new backend is one
   more ``client/<name>.yaml`` file in :data:`CLIENT_CONFIG_DIR`.
4. The generator injects per-instance fields (image, repo_dir, timeouts floors)
   and hands the dict to ``RunInput.from_dict``, which resolves every
   ``${oc.env:...}`` at once — so an exported env var overrides the YAML default
   at every layer above. Nothing is resolved before that point except the
   read-only probe, so ``escape_non_oc_interpolations`` only ever runs inside
   ``from_dict`` / :func:`resolve_config`.

Secrets (``OPENAI_API_KEY`` / ``GONGFENG_TOKEN``) are never defaulted anywhere.
"""

from __future__ import annotations

import copy
import hashlib
import os
import time

from collections.abc import Sequence
from pathlib import Path
from typing import Any, cast

import yaml

from omegaconf import OmegaConf

from nexus.utils.common import escape_non_oc_interpolations


# Every YAML the pipeline consumes lives here, one subdir per category:
# ``stage/<stage>.yaml`` (stage defaults, shared with run-single),
# ``pipeline/<task>.yaml`` (feature-pipeline tasks), ``client/<name>.yaml`` (one per LLM backend),
# and ``custom/*.yaml`` (user patches).
# This module lives in ``utils/``; the config tree stays at the package root.
CONFIG_DIR = Path(__file__).parent.parent / 'config'
STAGE_CONFIG_DIR = CONFIG_DIR / 'stage'
PIPELINE_CONFIG_DIR = CONFIG_DIR / 'pipeline'
CLIENT_CONFIG_DIR = CONFIG_DIR / 'client'
CUSTOM_CONFIG_DIR = CONFIG_DIR / 'custom'


def _load_yaml_mapping(path: Path, *, role: str) -> dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(f'{role} not found: {path}')
    data = yaml.safe_load(path.read_text(encoding='utf-8'))
    if not isinstance(data, dict):
        raise TypeError(f'{role} must be a mapping: {path}')
    return data


def _deep_merge(base: Any, patch: Any) -> Any:
    """Merge *patch* over *base*, RFC 7386 flavor.

    Dicts merge key by key; a ``None`` patch value deletes the key (this is how
    a patch turns a section off, e.g. ``processors: {chat: null}``); anything
    else — lists included — replaces wholesale.
    """
    if isinstance(base, dict) and isinstance(patch, dict):
        merged = dict(base)
        for key, value in patch.items():
            if value is None:
                merged.pop(key, None)
            elif key in merged:
                merged[key] = _deep_merge(merged[key], value)
            else:
                merged[key] = value
        return merged
    return copy.deepcopy(patch)


def config_patch_paths() -> list[str]:
    """Parse ``SWE_ANY_CONFIG_PATCH`` into an ordered list of patch entries.

    Comma-separated (like ``SWE_ANY_INSTANCE_IDS``), so several patches compose by
    permutation: later entries win on any key they share. Each entry is resolved by
    :func:`resolve_patch_path`. A single entry (or an empty value) is the
    degenerate case.
    """
    raw = os.getenv('SWE_ANY_CONFIG_PATCH', '').strip()
    return [p.strip() for p in raw.split(',') if p.strip()]


def resolve_patch_path(entry: str) -> Path:
    """Resolve a ``SWE_ANY_CONFIG_PATCH`` entry to a file.

    A bare name is looked up under ``config/custom/`` first (with a ``.yaml``
    appended when the entry has no suffix), so ``smoke`` and ``smoke.yaml`` both
    reach ``config/custom/smoke.yaml``. Only if nothing matches there is the entry
    treated as a relative (to cwd) or absolute path.
    """
    named = CUSTOM_CONFIG_DIR / entry
    if named.is_file():
        return named
    if not Path(entry).suffix:
        named_yaml = CUSTOM_CONFIG_DIR / f'{entry}.yaml'
        if named_yaml.is_file():
            return named_yaml
    raw = Path(entry)
    if raw.is_file():
        return raw
    raise FileNotFoundError(
        f'SWE_ANY_CONFIG_PATCH entry not found: {entry!r} (looked under {CUSTOM_CONFIG_DIR}/ then as a path)'
    )


def load_stage_config(default_yaml: str | Path, patch_paths: Sequence[str]) -> dict[str, Any]:
    """Load the stage's default YAML and deep-merge each user patch over it, in order.

    Patches are applied left to right, so a later entry overrides an earlier one on
    any shared key (RFC 7386, via :func:`_deep_merge`); each is located by
    :func:`resolve_patch_path` (``config/custom/`` first, then a path). The returned
    dict still carries every ``${oc.env:...}`` interpolation verbatim — resolution
    happens once, later, inside ``RunInput.from_dict``.
    """
    config = _load_yaml_mapping(Path(default_yaml), role='default config')
    for entry in patch_paths:
        patch_file = resolve_patch_path(entry)
        patch = yaml.safe_load(patch_file.read_text(encoding='utf-8'))
        if patch is None:
            continue  # an empty patch file is a no-op
        if not isinstance(patch, dict):
            raise TypeError(f'config patch must be a mapping: {patch_file}')
        config = _deep_merge(config, patch)
    return config


def resolve_config(config: dict[str, Any]) -> dict[str, Any]:
    """Resolve a copy of *config* (env interpolations) for read-only probing."""
    escaped = escape_non_oc_interpolations(copy.deepcopy(config))
    resolved = OmegaConf.to_container(OmegaConf.create(escaped), resolve=True)
    if not isinstance(resolved, dict):
        raise TypeError(f'config must resolve to a dict, got: {type(resolved).__name__}')
    return cast(dict[str, Any], resolved)


def apply_client_patch(config: dict[str, Any]) -> dict[str, Any]:
    """Merge ``config/client/<name>.yaml`` over *config* for the merged client name.

    The name is read from a resolved probe so a user patch (or ``OPENAI_CLIENT_NAME``)
    switching the backend is honored without the patch carrying backend internals.
    Only ``openai_clients.main.name`` is resolved — not the whole config — so a
    dry run (``get_total_count`` / ``get_batch_metadata``) never forces the
    no-default secrets that live elsewhere in the config.
    """
    # ``main: null`` in a patch leaves the key present with a None value, so guard
    # with ``or {}`` at each level rather than relying on the ``.get`` default.
    raw_name = ((config.get('openai_clients') or {}).get('main') or {}).get('name')
    name = str(resolve_config({'name': raw_name})['name'] or '') if raw_name is not None else ''
    patch_file = CLIENT_CONFIG_DIR / f'{name}.yaml'
    if not patch_file.is_file():
        available = sorted(p.stem for p in CLIENT_CONFIG_DIR.glob('*.yaml'))
        raise ValueError(
            f'no client patch registered for openai_clients.main.name={name!r}: '
            f'expected {patch_file} (registered: {available})'
        )
    return _deep_merge(config, _load_yaml_mapping(patch_file, role='client patch'))


# Markers that the LLM is reached through a Nexus gateway which partitions its prompt cache by
# ``cache_task_id`` (a ``gateway_routing._ROUTING_KEYS`` param). Any one of these is enough:
#   - the model-eval host in the base_url,
#   - a Claude gateway key (``claude`` in the key; same gate as ``swe`` / ``swe_cosmos``),
#   - a trpc gateway credential, recognized by its ``?provider=...`` routing query (e.g.
#     ``provider=aws_third`` / ``provider=google``) — the litellm client forwards the key verbatim
#     as the Bearer, so the id rides the query the gateway already parses.
# A direct GCP/Bedrock key carries none of these markers and must NOT be mangled.
_CACHE_GATEWAY_HOST = 'llm-api.model-eval.woa.com'


def attach_cache_task_id(config: dict[str, Any], instance_id: str) -> None:
    """Bake a per-run ``cache_task_id`` onto the main client's api_key query, IN PLACE.

    The gateway partitions its prompt cache by this id, so two instances running concurrently through
    one key would otherwise cross-wire each other's prefix caches. The id is a 32-hex md5 of
    ``time:pid:instance_id`` — unique per instance AND per stage (each stage's generator is a separate
    process, so ``time``/``pid`` differ). Every agentic stage (Parse/Select/Mask/Eval/Issue) calls
    this per instance; Export has no client. Consumers: ``vertex``/``vertex_anthropic`` retain the
    ``cache_task_id`` query via ``merge_gateway_routing``, and the model-eval/Claude gateways read it.

    No-op when the api_key is unset (dry run / ``get_total_count``), an id is already present, or the
    LLM is a DIRECT provider (none of the gateway markers above: ``claude`` in the key, a
    ``?provider=...`` gateway query, or the model-eval host) — a raw provider key must not be
    mangled. Mirrors ``swe/swe_cosmos``'s per-run cache_task_id.
    """
    api_key = os.getenv('OPENAI_API_KEY', '')
    base_url = os.getenv('OPENAI_BASE_URL', '')
    if not api_key or 'cache_task_id=' in api_key:
        return
    if 'claude' not in api_key and 'provider=' not in api_key and _CACHE_GATEWAY_HOST not in base_url:
        return
    main = (config.get('openai_clients') or {}).get('main')
    client_config = main.get('config') if isinstance(main, dict) else None
    client_args = client_config.get('client_args') if isinstance(client_config, dict) else None
    if not isinstance(client_args, dict):
        return
    seed = f'{time.time()}:{os.getpid()}:{instance_id}'
    cache_task_id = hashlib.md5(seed.encode('utf-8')).hexdigest()
    separator = '&' if '?' in api_key else '?'
    # Bake the resolved key (+id) here, overriding the ``${oc.env:OPENAI_API_KEY}`` interpolation:
    # RunInput.from_dict resolves it to the same concrete value before persisting anyway.
    client_args['api_key'] = f'{api_key}{separator}cache_task_id={cache_task_id}'


# --- The ``instances:`` spec block --------------------------------------------
# Which instances a batch runs (source jsonl + filters + paging) lives in a top-level
# ``instances:`` block — the stage config ships it with ${oc.env:...} defaults (so the
# SWE_ANY_* env twins keep working), a patch may override it, and run-single ignores it
# (not a RunInput field). ``pop_instance_spec`` lifts it out before ``RunInput.from_dict``;
# its resolved value drives :func:`.instance.selected_instances`.
_INSTANCE_SPEC_KEYS = frozenset({'path', 'ids', 'language', 'skip', 'limit'})


def pop_instance_spec(config: dict[str, Any]) -> dict[str, Any]:
    """Remove and return the ``instances`` block from a merged config (or ``{}``).

    Popping keeps the block out of ``RunInput`` (which would silently ignore it)
    and out of ``apply_client_patch``'s probe; it is resolved on its own by
    :func:`resolve_instance_spec`.
    """
    raw = config.pop('instances', None)
    if raw is None:
        return {}
    if not isinstance(raw, dict):
        raise TypeError(f'`instances` must be a mapping, got: {type(raw).__name__}')
    unknown = set(raw) - _INSTANCE_SPEC_KEYS
    if unknown:
        raise ValueError(f'unknown instances keys: {sorted(unknown)} (allowed: {sorted(_INSTANCE_SPEC_KEYS)})')
    return raw


def resolve_instance_spec(raw: dict[str, Any]) -> dict[str, Any]:
    """Resolve any ``${oc.env:...}`` inside a popped ``instances`` block."""
    if not raw:
        return {}
    return resolve_config({'instances': raw})['instances']


def base_config_and_spec(default_yaml: str | Path) -> tuple[dict[str, Any], dict[str, Any]]:
    """Load a stage's shared batch config plus its resolved ``instances`` spec.

    This base helper makes no LLM assumption: it merges the stage YAML with
    ``SWE_ANY_CONFIG_PATCH``, then lifts out and resolves ``instances`` separately
    to drive the batch. Deterministic stages such as Export use it directly.
    """
    config = load_stage_config(default_yaml, config_patch_paths())
    spec = resolve_instance_spec(pop_instance_spec(config))
    return config, spec


def agentic_config_and_spec(default_yaml: str | Path) -> tuple[dict[str, Any], dict[str, Any]]:
    """Load an agentic stage's batch config and apply its LLM-backend patch."""
    config, spec = base_config_and_spec(default_yaml)
    return apply_client_patch(config), spec
