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

The three files share one tree, so they write into each other's keys. Oracle later deletes ``agent_name`` / ``agent_config`` / ``openai_clients`` with ``null``.

Scaffold parameters that depend on the model
--------------------------------------------
A few scaffold parameters differ by model family: ``thinking_preset`` for Claude and Hunyuan, ``wrap_plaintext_reasoning_content`` for GPT, the ``ANTHROPIC_MODEL`` env var for the CLIs that read it. They belong to the scaffold, so the scaffold file states them, once per family::

    # agents/claude_code_eval.yaml
    agent_config_by_llm:
      claude:  {thinking_preset: bedrock-claude, extra_env: {ANTHROPIC_MODEL: '${llm:anthropic_model}'}}
      hunyuan: {thinking_preset: hunyuan}

The llm file states only facts about the model, under ``llm:``::

    # llms/claude-opus-5.openai.yaml
    llm:
      family: claude
      anthropic_model: 'claude-opus-5[1m]'
      no_thinking_request_args: {...}

After folding, the section for ``llm.family`` is merged into ``agent_config``, ``${llm:<key>}`` in it is replaced by that fact (``${llm:<key>?}`` when the wire may omit it: the key is then left out, so the scaffold's own default stands), and both ``llm`` and ``agent_config_by_llm`` are removed - neither is a RunInput field. So an llm file never names a scaffold, and a scaffold with nothing family-specific writes nothing.

Why this is not just style: before it, every llm file wrote ``thinking_preset`` and friends straight into ``agent_config``, and 1920 of the 2160 shared combinations landed at least one key the selected scaffold does not declare. Pydantic inputs are ``extra='ignore'``, so all of them were dropped without a word.

Every key must have a consumer
------------------------------
``runner/single.py::_build_input`` keeps only the keys an agent's or task's Input model declares and drops the rest silently. A run then finishes on a config that is not the one written, and the cost is already paid. So :func:`load_config` checks ``agent_config`` and ``task_config`` against those Input models here, on the generate machine before any sandbox or model call, and raises listing every key nothing would read. To add a key, declare the field on the Input model first.

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

import importlib
import os
import re

from pathlib import Path
from typing import Any

from pydantic import BaseModel

from nexus.core import agent_registry, task_registry
from nexus.runner.compose.merge import deep_merge
from nexus.runner.compose.model import RawRunPart
from nexus.runner.compose.resolver import resolve_non_data_part
from nexus.runner.single import _get_input_type


SHARED_DIR = Path(__file__).parent  # agents/, llms/ and runtime/ sit next to this file

# env var -> subdirectory under configs/ searched for a bare name
ENV_TO_FOLDER = {
    'AGENT_CONFIG': 'agents',
    'LLM_CONFIG': 'llms',
    'RUNTIME_CONFIG': 'runtime',
}

#: What an llm file may put in ``llm.family``, and what a scaffold may key a section by.
#: Closed set, so a typo on either side raises instead of silently matching nothing.
LLM_FAMILIES = ('gpt', 'claude', 'hunyuan')

_LLM_REF = re.compile(r'\$\{llm:([A-Za-z_][A-Za-z0-9_]*)(\?)?\}')

#: Marker for an optional ``${llm:key?}`` the llm file did not declare: the key is left out.
_OMIT = object()


def _search_dirs(subdir: str, task_dir: Path | None) -> list[Path]:
    """Task's own configs/<subdir> first, then the shared one."""
    dirs: list[Path] = []
    if task_dir is not None:
        dirs.append(task_dir / 'configs' / subdir)
    dirs.append(SHARED_DIR / subdir)
    return dirs


def _merge_part(config: dict[str, Any], env_var: str, task_dir: Path | None) -> dict[str, Any]:
    """Merge the run part named by *env_var* into *config*, or return it unchanged when unset."""
    entry = os.getenv(env_var, '').strip()
    if not entry:
        return config
    subdir = ENV_TO_FOLDER[env_var]
    if not entry.endswith('.yaml'):
        entry = f'{entry}.yaml'
    dirs = _search_dirs(subdir, task_dir)
    candidates = [d / entry for d in dirs] + [Path(entry)]
    path = next((p for p in candidates if p.is_file()), None)
    if path is None:
        available = sorted({p.stem for d in dirs if d.is_dir() for p in d.glob('*.yaml')})
        looked = ' then '.join(f'{d}/' for d in dirs)
        raise FileNotFoundError(
            f'{env_var}={entry!r} not found: looked under {looked}, then as a path (available: {available})'
        )
    part = resolve_non_data_part(RawRunPart(kind='use', value=str(path), source=str(path)))
    return deep_merge(config, part.body, part.merge)


def load_config(task_dir: Path | str | None = None) -> dict[str, Any]:
    """Merge runtime, then agent, then LLM. The LLM file wins on ``agent_config`` keys it owns.

    *task_dir* is the directory holding the task's ``configs/``; pass ``Path(__file__).parent``.
    """
    resolved = Path(task_dir) if task_dir is not None else None
    config: dict[str, Any] = {}
    for env_var in ('RUNTIME_CONFIG', 'AGENT_CONFIG', 'LLM_CONFIG'):
        config = _merge_part(config, env_var, resolved)
    config = _apply_llm_family(config)
    _require_consumers(config)
    return config


def _apply_llm_family(config: dict[str, Any]) -> dict[str, Any]:
    """Fold the scaffold's section for the selected model family into ``agent_config``.

    Both helper keys are removed either way: neither is a ``RunInput`` field, so leaving
    them in would only get them dropped later.

    ``${llm:...}`` refs are filled in two places: the folded family section (refs that only
    apply to one family, like ``use_azure_provider``), and the flat top-level ``agent_config``
    (refs that apply to every family, like a scaffold reading the wire's effort level without
    caring whether it is spelled ``reasoning.effort`` or ``output_config.effort``). The flat
    pass runs over whatever ``agent_config`` holds after the fold, so a section entry can
    itself be a ref and still resolve.
    """
    config = dict(config)
    facts = config.pop('llm', None)
    sections = config.pop('agent_config_by_llm', None)

    if facts is not None and (not isinstance(facts, dict) or facts.get('family') not in LLM_FAMILIES):
        raise ValueError(f'llm.family must be one of {list(LLM_FAMILIES)}, got {facts!r}')
    if sections is not None:
        if not isinstance(sections, dict) or not all(isinstance(v, dict) for v in sections.values()):
            raise TypeError('agent_config_by_llm must map a model family to a mapping')
        if unknown := sorted(set(sections) - set(LLM_FAMILIES)):
            raise ValueError(f'agent_config_by_llm has unknown families {unknown}; known: {list(LLM_FAMILIES)}')
        # No llm axis selected (a test loading one scaffold alone): no family, so no section applies.
        if facts is not None and facts['family'] in sections:
            section = _fill_llm_refs(sections[facts['family']], facts, path=f'agent_config_by_llm.{facts["family"]}')
            config = deep_merge(config, {'agent_config': section})
    # Fill flat top-level refs too (see docstring). Only meaningful with an llm axis selected;
    # without one there are no facts, so leave the refs as literal text for the caller to see.
    if facts is not None and isinstance(config.get('agent_config'), dict):
        config['agent_config'] = _fill_llm_refs(config['agent_config'], facts, path='agent_config', drop_null=True)
    return config


def _omit_cause(element: Any, facts: dict[str, Any], drop_null: bool) -> str:
    """Why this list element resolved to ``_OMIT``, so the message names the fix that applies.

    A list has no "leave the key out" option, so either source is an error here -- but they are
    fixed differently: declare the fact, or stop declaring it as null.
    """
    match = _LLM_REF.fullmatch(element) if isinstance(element, str) else None
    key = match.group(1) if match else None
    if key is not None and drop_null and key in facts and facts[key] is None:
        return f'llm.{key} is declared null, which would drop this element; a list cannot hold a key that may vanish'
    return 'an optional ${llm:...?} that this llm file does not declare cannot sit in a list'


def _fill_llm_refs(value: Any, facts: dict[str, Any], *, path: str, drop_null: bool = False) -> Any:
    """Replace ``${llm:key}`` with the llm file's fact.

    ``${llm:key?}`` may be undeclared: the key holding it is then left out, so whatever the
    scaffold already set stays. A fact declared as ``null`` is different - it is a value. In a
    family section it is merged like any other and ``deep_merge`` turns ``null`` into "delete the
    key". The flat top-level pass has no merge step after it, so it passes ``drop_null=True`` to get
    the same result: the key is left out, not kept as ``None``.

    ``path`` is the full location used in error messages, e.g. ``agent_config_by_llm.gpt.x`` or
    ``agent_config.x``.
    """
    if isinstance(value, dict):
        filled = {k: _fill_llm_refs(v, facts, path=f'{path}.{k}', drop_null=drop_null) for k, v in value.items()}
        return {k: v for k, v in filled.items() if v is not _OMIT}
    if isinstance(value, list):
        filled = [_fill_llm_refs(v, facts, path=f'{path}[{i}]', drop_null=drop_null) for i, v in enumerate(value)]
        omitted = [i for i, v in enumerate(filled) if v is _OMIT]
        if omitted:
            raise ValueError(f'{path}[{omitted[0]}]: {_omit_cause(value[omitted[0]], facts, drop_null)}')
        return filled
    if not isinstance(value, str) or '${llm:' not in value:
        return value
    match = _LLM_REF.fullmatch(value)
    if match is None:
        raise ValueError(f'{path}: ${{llm:...}} must be the whole value, got {value!r}')
    key, optional = match.group(1), match.group(2)
    if key in facts:
        return _OMIT if (drop_null and facts[key] is None) else facts[key]
    if optional:
        return _OMIT
    raise ValueError(
        f'{path} needs llm.{key}, which this llm file does not declare '
        f'(it declares {sorted(k for k in facts if k != "family")}). Declare it, as null if this '
        f'wire deliberately has none.'
    )


def _require_consumers(config: dict[str, Any]) -> None:
    """Raise if ``agent_config`` or ``task_config`` holds a key its Input model would drop.

    ``runner/single.py::_build_input`` keeps only declared fields and drops the rest without a
    word, so a run finishes on a config that is not the one written and the cost is already paid.
    This runs on the generate machine, before any sandbox or model call.
    """
    for module in config.get('imports') or []:
        importlib.import_module(module)
    problems = []
    for kind, registry, name_key, config_key in (
        ('agent', agent_registry, 'agent_name', 'agent_config'),
        ('task', task_registry, 'task_name', 'task_config'),
    ):
        name = config.get(name_key)
        if not name:
            continue
        input_type = _get_input_type(registry.get(name))
        if not (isinstance(input_type, type) and issubclass(input_type, BaseModel)):
            raise TypeError(f'{kind} {name!r} takes {input_type!r}, not a pydantic Input model')
        if unread := sorted(set(config.get(config_key) or {}) - set(input_type.model_fields)):
            problems.append(f'{config_key} has {unread}, which {kind} {name!r} ({input_type.__name__}) ignores')
    if problems:
        raise ValueError(
            'these keys would be dropped silently at run time: '
            + '; '.join(problems)
            + '. Move them under agent_config_by_llm, or declare the field on the Input model first.'
        )
