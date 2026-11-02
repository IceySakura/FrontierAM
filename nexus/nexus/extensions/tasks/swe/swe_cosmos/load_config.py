"""Load and merge run-part YAML.

:func:`load_config` folds the files named by ``RUNTIME_CONFIG``, ``AGENT_CONFIG`` and ``LLM_CONFIG`` into one dict, in that order. Each value is a stem under ``configs/<kind>/`` (``.yaml`` is appended if missing) or a path anywhere on disk. An unset env var skips that axis; there is no default file.

The files are the same envelope ``nexus run batch --use`` takes, so a generator can sit next to ``--use`` / ``--data`` / ``--set`` instead of mixing the two:

    body:     # partial RunInput this file contributes
    merge:    # per-dotted-path list strategy (see Merge)
    source:   # CLI log label; this loader discards it

This function only loads and merges. ``${oc.*}`` strings stay unresolved until ``RunInput.from_dict``.

Merge
-----
Declared on ``merge:`` per dotted path. Only lists need an explicit op:

* ``append`` — concatenate, keep duplicates.
* ``append_unique`` — append items not already present. What ``imports`` wants.
* ``replace`` — take the incoming value wholesale, even for a dict or list.

Undeclared paths use the defaults: two dicts recurse key by key; ``null`` deletes the key; any other value (lists included) replaces.

The three files share one tree, so they write into each other's keys. LLM merges last and owns backend-only ``agent_config`` fields (``thinking_preset``, ``no_thinking_request_args``). A scaffold that does not declare them ignores them (agent inputs are pydantic, ``extra='ignore'``). Oracle later deletes ``agent_name`` / ``agent_config`` / ``openai_clients`` with ``null``.

YAML interpolations (OmegaConf, resolved at ``RunInput.from_dict``)
-------------------------------------------------------------------
``oc.env`` — environment variable on the *generate* machine, at *generate* time. Always a string, even when the default looks numeric or boolean::

    working_dir: ${oc.env:AGENT_WORKING_DIR,/testbed}
    api_key: ${oc.env:OPENAI_API_KEY,''}

``oc.decode`` — cast that string back to int / float / bool. Required wherever the consumer is not a string (``client_args`` go into ``AsyncOpenAI(**client_args)``)::

    timeout: ${oc.decode:${oc.env:LLM_TIMEOUT,900}}
    run_oracle: ${oc.decode:${oc.env:RUN_ORACLE,false}}

``oc.select`` — copy a value from another path in the *merged* config. That is why it can see the LLM block even when written in the agent file. Second argument is the default if the path is missing::

    model: ${oc.select:openai_clients.main.config.request_args.model,null}

Quoting — a default that contains ``[``, ``]`` or ``,`` must be quoted, or OmegaConf parses it as a list::

    ANTHROPIC_MODEL: ${oc.env:ANTHROPIC_MODEL,'claude-opus-5[1m]'}

Non-``oc`` ``${...}`` (JS template strings in prompts, and so on) is escaped by ``from_dict`` and left literal.

Delayed env and secrets (not OmegaConf)
---------------------------------------
``foo__env: VARNAME`` is a runner convention. ``run()`` replaces it with ``foo: os.environ[VARNAME]`` on the *execute* machine, so the secret never lands in the batch DB::

    token__env: GONGFENG_RUNTIME_TOKEN

``OPENAI_API_KEY`` stays ``oc.env`` because the generator may append a per-instance ``cache_task_id`` after seeing the resolved string.

``runtime_image: current`` is a Gongfeng/Docker sentinel (resolve to the matching nexus-runtime image), not an interpolation.
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
    'AGENT_CONFIG': 'agents',
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
    """Merge runtime, then agent, then LLM. The LLM file wins on ``agent_config`` keys it owns."""
    config: dict[str, Any] = {}
    for env_var in ('RUNTIME_CONFIG', 'AGENT_CONFIG', 'LLM_CONFIG'):
        config = _merge_part(config, env_var)
    return config
