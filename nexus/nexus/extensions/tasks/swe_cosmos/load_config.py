"""Load and merge run-part YAML.

:func:`load_config` folds the files named by ``RUNTIME_CONFIG`` and ``LLM_CONFIG`` into one dict, in that order. Each value is a stem under ``configs/<kind>/`` (``.yaml`` is appended if missing) or a path anywhere on disk. An unset env var skips that axis; there is no default file. The bugmaker pipeline has no scaffold, so there is no agent axis — a run part is either the sandbox/pipeline profile or the LLM backend.

The files are the same envelope ``nexus run batch --use`` takes, so a generator can sit next to ``--use`` / ``--data`` / ``--set`` instead of mixing the two:

    body:     # partial RunInput this file contributes
    merge:    # per-dotted-path list strategy
    source:   # CLI log label; this loader discards it

Only lists need an explicit ``merge:`` op — ``append``, ``append_unique`` (what ``imports`` wants) or ``replace``. Undeclared paths use the defaults: two dicts recurse key by key, ``null`` deletes the key, anything else replaces.

This function only loads and merges. ``${oc.*}`` strings stay unresolved until ``RunInput.from_dict``. See ``nexus/extensions/tasks/swe/swe_cosmos/load_config.py`` for the full note on the interpolation forms (``oc.env`` / ``oc.decode`` / ``oc.select``, and quoting) and on the ``foo__env: VARNAME`` convention that keeps secrets out of the batch DB.
"""

from __future__ import annotations

import os

from pathlib import Path
from typing import Any

from nexus.runner.compose.merge import deep_merge
from nexus.runner.compose.model import RawRunPart
from nexus.runner.compose.resolver import resolve_non_data_part


CONFIG_DIR = Path(__file__).parent / 'configs'

# env var -> subdirectory under configs/ searched for a bare name
ENV_TO_FOLDER = {
    'LLM_CONFIG': 'llms',
    'RUNTIME_CONFIG': 'runtime',
}


def _merge_part(config: dict[str, Any], env_var: str) -> dict[str, Any]:
    """Merge the run part named by *env_var* into *config*, or return it unchanged when unset."""
    entry = os.getenv(env_var, '').strip()
    if not entry:
        return config
    subdir = ENV_TO_FOLDER[env_var]
    if not entry.endswith('.yaml'):
        entry = f'{entry}.yaml'
    candidates = [CONFIG_DIR / subdir / entry, Path(entry)]
    path = next((p for p in candidates if p.is_file()), None)
    if path is None:
        available = sorted(p.stem for p in (CONFIG_DIR / subdir).glob('*.yaml'))
        raise FileNotFoundError(
            f'{env_var}={entry!r} not found: looked under {CONFIG_DIR / subdir}/ then as a path (available: {available})'
        )
    part = resolve_non_data_part(RawRunPart(kind='use', value=str(path), source=str(path)))
    return deep_merge(config, part.body, part.merge)


def load_config() -> dict[str, Any]:
    """Merge runtime, then LLM. The LLM file wins on the keys it owns."""
    config: dict[str, Any] = {}
    for env_var in ('RUNTIME_CONFIG', 'LLM_CONFIG'):
        config = _merge_part(config, env_var)
    return config
