"""SWE-Env instance handling — schema readers, task hydration, batch selection, host paths.

One home for everything about an *instance* (the SWE-Env repo record):

- read it: ``load_single_instance`` (one-row, run-single) / ``load_instances`` (a JSONL batch);
- hydrate a task input from it: ``apply_instance`` (task side);
- select a batch subset honoring the resolved ``instances`` spec: ``selected_instances`` (+ the
  language / id / paging filters, generator side);
- derive per-instance host artifact paths: ``repo_name_of`` / ``output_dir`` / ``host_*_path``.

The complementary :mod:`.batch_config` owns the *config* merge pipeline that produces the
``instances`` spec this module consumes; the two are kept apart on purpose.
"""

from __future__ import annotations

import json
import os

from collections.abc import Iterator
from pathlib import Path
from typing import Any

from . import constants


_DEFAULT_OUTPUT_DIR = './outputs/swe_any'


def load_single_instance(path: str) -> dict[str, Any]:
    """Load exactly one instance from a JSONL file."""
    rows = [json.loads(line) for line in Path(path).read_text(encoding='utf-8').splitlines() if line.strip()]
    if len(rows) != 1:
        raise ValueError(f'expected exactly one instance in {path}, got {len(rows)}')
    instance = rows[0]
    if not isinstance(instance, dict):
        raise ValueError(f'instance in {path} must be a JSON object')
    return instance


def load_instances(path: str) -> list[dict[str, Any]]:
    """Load SWE-Env instances from a JSONL file (one object per line)."""
    p = Path(path)
    if p.suffix != '.jsonl':
        raise ValueError(f'instances path must be a .jsonl file, got: {path}')
    with open(p, encoding='utf-8') as f:
        return [json.loads(line) for line in f if line.strip()]


def apply_instance(task_input: Any) -> None:
    """Hydrate common repo fields from an inline instance or one-row JSONL."""
    instance = task_input.instance
    if instance is None and task_input.instance_path:
        instance = load_single_instance(task_input.instance_path)
        task_input.instance = instance
    if instance is None:
        return

    instance_id = str(instance.get('instance_id') or '')
    image = str(instance.get('image') or '')
    base_commit = str(instance.get('base_commit') or (instance.get('base') or {}).get('sha') or '')
    if not instance_id or not image or not base_commit:
        raise ValueError('instance requires non-empty instance_id, image, and base_commit')
    task_input.repo_name = str(instance.get('repo_name') or instance_id)
    task_input.image = image
    task_input.repo_dir = str(instance.get('workdir') or constants.SANDBOX_REPO_DIR)
    task_input.base_commit = base_commit


def _language_filter(value: Any) -> set[str] | None:
    """Set of allowed languages, or ``None`` for no filter (empty or ``all``)."""
    raw = str(value or '').strip().lower()
    if not raw or raw == 'all':
        return None
    return {lang.strip() for lang in raw.split(',') if lang.strip()}


def _id_filter(value: Any) -> set[str] | None:
    """Set of allowed instance_ids, or ``None`` for no filter (empty).

    Accepts a YAML list or a comma string; each entry is a literal id or a path
    to a JSON list of ids.
    """
    if not value:
        return None
    entries = value if isinstance(value, list) else [e.strip() for e in str(value).split(',')]
    ids: set[str] = set()
    for item in entries:
        if not item:
            continue
        if os.path.isfile(item):
            with open(item, encoding='utf-8') as f:
                ids.update(json.load(f))
        else:
            ids.add(str(item))
    return ids or None


def _as_int(value: Any) -> int:
    """Coerce a spec value to int; empty / missing is 0."""
    return int(value or 0)


def selected_instances(spec: dict[str, Any]) -> Iterator[dict[str, Any]]:
    """Yield SWE-Env instances honoring the resolved ``instances`` *spec*.

    *spec* comes from the stage config's ``instances`` block (see
    :func:`.batch_config.resolve_instance_spec`), whose ``${oc.env:...}`` defaults already
    fold in the ``SWE_ANY_*`` env twins — so this reads plain values, no env here.
    """
    path = str(spec.get('path') or '').strip()
    if not path:
        raise ValueError('no instances source: set instances.path (or its SWE_ANY_INSTANCES default).')

    langs = _language_filter(spec.get('language'))
    ids = _id_filter(spec.get('ids'))
    skip = _as_int(spec.get('skip'))
    limit = _as_int(spec.get('limit'))

    skipped = 0
    produced = 0
    for inst in load_instances(path):
        instance_id = str(inst.get('instance_id', '') or '')
        if not instance_id or not inst.get('image'):
            continue
        if langs is not None and str(inst.get('language', '')).lower() not in langs:
            continue
        if ids is not None and instance_id not in ids:
            continue
        if skipped < skip:
            skipped += 1
            continue
        yield inst
        produced += 1
        if limit and produced >= limit:
            break


def repo_name_of(inst: dict[str, Any]) -> str:
    return str(inst.get('repo_name') or inst['instance_id'])


def output_dir() -> str:
    """Host dir receiving per-repo artifacts (parse_graph.json, parse_seed_policy.json, archives)."""
    return os.getenv('SWE_ANY_OUTPUT_DIR', _DEFAULT_OUTPUT_DIR)


def host_graph_path(inst: dict[str, Any], *, out_dir: str | None = None) -> Path:
    """Host path where Parse downloads ``parse_graph.json`` for this instance."""
    return Path(out_dir or output_dir()) / repo_name_of(inst) / constants.GRAPH_FILENAME


def host_select_path(inst: dict[str, Any], *, out_dir: str | None = None) -> Path:
    """Host path where Select writes its unified ``select.jsonl`` for this instance (Mask's input)."""
    return Path(out_dir or output_dir()) / repo_name_of(inst) / constants.SELECT_JSONL_FILENAME


def host_mask_path(inst: dict[str, Any], *, out_dir: str | None = None) -> Path:
    """Host path where Mask writes its unified ``mask.jsonl`` for this instance (Eval's input)."""
    return Path(out_dir or output_dir()) / repo_name_of(inst) / constants.MASK_JSONL_FILENAME


def host_eval_path(inst: dict[str, Any], *, out_dir: str | None = None) -> Path:
    """Host path where Eval writes its unified ``eval.jsonl`` for this instance (Issue/Export input)."""
    return Path(out_dir or output_dir()) / repo_name_of(inst) / constants.EVAL_JSONL_FILENAME


def host_issue_path(inst: dict[str, Any], *, out_dir: str | None = None) -> Path:
    """Host path where Issue writes its unified ``issue.jsonl`` for this instance (Export's input)."""
    return Path(out_dir or output_dir()) / repo_name_of(inst) / constants.ISSUE_JSONL_FILENAME


def host_manifest_path(inst: dict[str, Any], *, out_dir: str | None = None) -> Path:
    """Host path where pre-feature writes this repo's feature manifest."""
    return Path(out_dir or output_dir()) / repo_name_of(inst) / constants.FEATURE_MANIFEST_FILENAME


def host_feature_result_path(inst: dict[str, Any], index: int, *, out_dir: str | None = None) -> Path:
    """Host path where one feature task writes its terminal result."""
    return (
        Path(out_dir or output_dir())
        / repo_name_of(inst)
        / constants.slice_dirname(index)
        / constants.FEATURE_RESULT_FILENAME
    )
