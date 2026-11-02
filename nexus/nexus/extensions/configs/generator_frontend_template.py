"""Frontend rollout generator, on the shared scaffold layer.

The cell is three files, none of which this module chooses:

    RUNTIME_CONFIG=frontend_runtime                 configs/runtime/
    AGENT_CONFIG=kilo                               configs/agents/
    LLM_CONFIG=gpt-6-sol.openai_responses           configs/llms/

    DATASET_PATH=./cases.jsonl FRONTEND_LIMIT=5 \\
      nexus run-batch -g <this file> -d rollout.db --max-concurrent 2

What stays here is only what varies per instance: which cases to run, the
prompt text each one carries, and the per-instance cache ids. Everything that
varies per EXPERIMENT -- harness, version, model, gateway, effort, sandbox
size -- is in those three YAML files, and the first two are shared verbatim
with SWE.

That is the whole claim of the shared layer, and this file is the proof: a
frontend rollout needs no scaffold code of its own. The previous
`generator_rollout.py` carried 672 lines, a `_runtime_container_image()`
switch per scaffold and its own ScaffoldSpec / ModelWire tables.

Frontend-specific facts, and why each is here rather than in a shared file:

* The prompt is the instance. SWE's task reads a dataset row by id; a frontend
  case carries its text, so the generator passes `prompt` through.
* `working_dir` is `/workspace`, not SWE's `/testbed`. The task reads it from
  the runtime profile; the agent profiles default to `/testbed`, so the overlay
  copies the task's value onto `agent_config.working_dir`. The questions-mode
  pre-start scripts and prompt are built from that same value, so none of the
  four can drift -- set it with `AGENT_WORKING_DIR` (the runtime profile's knob;
  the older generators' `FRONTEND_WORKING_DIR` does not reach this layer).
* The webenv image is per runtime profile. Production picks a node22 image for
  codebuddy / dsh / handy / pi / workbuddy; here that is `FRONTEND_IMAGE`, not
  a per-scaffold switch.
* Task fields the YAML cannot carry on every checkout go through
  `FRONTEND_TASK_CONFIG_EXTRA` (JSON), validated against the running checkout.
* The image can differ per agent: `FRONTEND_IMAGE_BY_AGENT` (JSON `{agent_name: image}`), for
  production's node22 image on some harnesses. Unlisted agents keep the runtime's image.
* One `cache_task_id` per trajectory, on the wire and in `metadata.cache_task_id` (see `cache_id.py`).
  AIHub 2.0 schedules the provider account from it, which replaced the retired `account_id2` pinning;
  HyWay hands the recorded id to the CoT hack so rethink decodes on the account that produced the
  encrypted reasoning.
* No anti-hack. Frontend cases have no hidden oracle patch to leak and no
  grader score, so the switches SWE pins would be meaningless.
"""

from __future__ import annotations

import base64
import hashlib
import io
import json
import os
import shlex
import tarfile

from collections.abc import Iterator
from pathlib import Path
from typing import Any

from omegaconf import OmegaConf

from nexus.extensions.configs.cache_id import cache_task_overlay
from nexus.extensions.configs.load_config import _require_consumers, load_config
from nexus.runner.batch.batch_config import BatchItem
from nexus.runner.compose.merge import deep_merge
from nexus.runner.single import RunInput, _resolve_env_suffix_config


DEFAULT_WORKING_DIR = '/workspace'


def _env_int(name: str, default: int | None = None) -> int | None:
    """The integer ``name`` is set to, or ``default`` when it is unset.

    Raises ``ValueError`` naming the variable; a bare ``int(raw)`` says only that some
    string is not an int, which does not say which knob to fix.
    """
    raw = os.getenv(name, '').strip()
    if not raw:
        return default
    try:
        return int(raw)
    except ValueError:
        raise ValueError(f'{name}={raw!r} is not an integer') from None


def _resolve_scalars(values: dict[str, Any]) -> dict[str, Any]:
    """Resolve `oc.*` and `foo__env` in a lifted dict.

    Some per-instance overlays need a value from the merged config before the
    whole tree is resolved (is this an OpenAI wire? does the key already carry
    a query string?). Lifting the two or three scalars we need keeps the rest
    of the tree unresolved, which is what `RunInput.from_dict` expects.

    `oc.select` paths inside a lifted dict are relative to THAT dict, not to
    the full config -- so only lift self-contained values.
    """
    resolved = OmegaConf.to_container(OmegaConf.create(values), resolve=True)
    if not isinstance(resolved, dict):
        raise TypeError(f'lifted probe must resolve to a dict, got {type(resolved).__name__}')
    return _resolve_env_suffix_config({str(k): v for k, v in resolved.items()}, path='probe')


def _checked_working_dir(base: dict[str, Any]) -> str:
    """The task's resolved ``working_dir``, refusing a value the pre-start ``rm -rf`` must not touch.

    ``load_config`` leaves the tree unresolved, so the raw value is the ``${oc.env:...}``
    text; the scripts need the real path, hence the lift.
    """
    raw = (base.get('task_config') or {}).get('working_dir')
    value = (
        str(_resolve_scalars({'working_dir': raw})['working_dir'] or DEFAULT_WORKING_DIR)
        if raw
        else DEFAULT_WORKING_DIR
    )
    if not value.startswith('/') or len(Path(value).parts) < 2:
        raise ValueError(f'working_dir={value!r} is not an absolute path below the root')
    return value


def _probe_llm(config: dict[str, Any]) -> dict[str, str]:
    """The few LLM scalars the per-instance overlays need."""
    client = config.get('openai_clients', {}).get('main', {}).get('config', {})
    lifted = {
        'model': client.get('request_args', {}).get('model', ''),
        'api_key': client.get('client_args', {}).get('api_key', ''),
    }
    resolved = _resolve_scalars(lifted)
    return {key: str(value or '') for key, value in resolved.items()}


def _per_instance_cache_id(instance_id: str, attempt: int = 1) -> str:
    """Stable per-instance cache id.

    Derived from the instance rather than a uuid so a rerun of one case reuses
    its own prefix cache instead of a cold one, and two cases never share.
    ``attempt`` (1-based) is part of the seed, so repeated sampling
    (``EVAL_TIMES``) gives each try its own scope instead of every try of one
    case sharing the first try's cache.
    """
    seed = f'{os.getenv("FRONTEND_EXPERIMENT", "frontend")}:{instance_id}:{attempt}'
    return hashlib.md5(seed.encode()).hexdigest()


def _required_dataset_path() -> str:
    dataset_path = os.getenv('DATASET_PATH', '').strip()
    if not dataset_path:
        raise ValueError('DATASET_PATH is required and must point to a .jsonl of cases')
    return dataset_path


def _extract_instance_ids(path: Path) -> set[str]:
    """Ids from a selector file: `.jsonl` line objects or a `.json` list.

    An element may be a bare string or an object with `instance_id`. A line
    that yields neither names the file and the line, because a silently
    skipped selector is a run that scores over the wrong denominator.
    """
    ids: set[str] = set()
    if path.suffix == '.jsonl':
        for lineno, raw in enumerate(path.read_text(encoding='utf-8').splitlines(), 1):
            if not (line := raw.strip()):
                continue
            entry = json.loads(line)
            found = entry if isinstance(entry, str) else entry.get('instance_id')
            if not found:
                raise ValueError(f'{path}:{lineno} has no instance_id')
            ids.add(str(found))
        return ids
    data = json.loads(path.read_text(encoding='utf-8'))
    for index, entry in enumerate(data if isinstance(data, list) else [data]):
        found = entry if isinstance(entry, str) else entry.get('instance_id')
        if not found:
            raise ValueError(f'{path}[{index}] has no instance_id')
        ids.add(str(found))
    return ids


def _id_set(name: str) -> set[str]:
    """`<NAME>` as a comma-separated id list or a selector file path."""
    raw = os.getenv(name, '').strip()
    if not raw:
        return set()
    path = Path(raw)
    if path.suffix in {'.json', '.jsonl'}:
        if not path.is_file():
            raise FileNotFoundError(f'{name}={raw!r} is not a file')
        return _extract_instance_ids(path)
    return {part.strip() for part in raw.split(',') if part.strip()}


def _selected_instance_filters() -> tuple[set[str], set[str]]:
    include = _id_set('INCLUDE_INSTANCES')
    exclude = _id_set('EXCLUDE_INSTANCES')
    if overlap := include & exclude:
        raise ValueError(
            f'INCLUDE_INSTANCES and EXCLUDE_INSTANCES both name {sorted(overlap)[:3]}; '
            f'that is a configuration error, not an empty selection'
        )
    return include, exclude


def _question_rows(path: Path) -> list[dict[str, Any]]:
    """Read an eval question set (``turns`` / ``assets`` / ``repo``) into case rows.

    Same file the eval packs read; ``id``/``title`` replace ``instance_id``/``prompt``.
    """
    rows = [json.loads(line) for line in path.read_text(encoding='utf-8').splitlines() if line.strip()]
    for i, row in enumerate(rows, 1):
        if not row.get('id'):
            raise ValueError(f'{path}:{i} has no id')
        if not (row.get('turns') or []):
            raise ValueError(f'{path}:{i} ({row["id"]}) has no turns')
    return rows


def _iter_selected_instances(working_dir: str = DEFAULT_WORKING_DIR) -> Iterator[tuple[str, dict[str, Any]]]:
    """Yield (instance_id, metadata) for each case surviving the filters.

    Two modes, chosen by ``FRONTEND_QUESTIONS``:

    * unset (a rollout): rows of ``DATASET_PATH`` carry ``instance_id`` and one ``prompt``.
    * set (an eval): rows carry ``turns`` / ``assets`` / ``repo``; the prompt is assembled and the
      pre-start scripts clone the repo and unpack the assets, exactly as the eval packs do.

    Shared by `generate_configs`, `get_total_count` and `get_batch_metadata`,
    so the runner's progress denominator cannot disagree with what is produced.
    """
    questions = _questions_path()
    dataset_path = str(questions) if questions else _required_dataset_path()
    include_ids, exclude_ids = _selected_instance_filters()
    skip = _env_int('FRONTEND_SKIP', 0) or 0
    limit = _env_int('FRONTEND_LIMIT')

    path = Path(dataset_path)
    if not path.is_file():
        raise FileNotFoundError(f'Dataset file not found: {dataset_path}')
    if path.suffix != '.jsonl':
        raise ValueError(f'Dataset must be .jsonl for batch mode: {dataset_path}')

    cases = (
        _question_rows(path)
        if questions
        else [json.loads(raw) for raw in path.read_text(encoding='utf-8').splitlines() if raw.strip()]
    )

    missing = include_ids - {str(c.get('instance_id') or c.get('id') or '') for c in cases}
    if missing:
        raise ValueError(
            f'INCLUDE_INSTANCES names {len(missing)} case(s) absent from {dataset_path}: {sorted(missing)[:3]}'
        )

    skipped = 0
    produced = 0
    for case in cases:
        instance_id = str(case.get('instance_id') or case.get('id') or '')
        if not instance_id:
            continue
        if include_ids and instance_id not in include_ids:
            continue
        if instance_id in exclude_ids:
            continue
        if skipped < skip:
            skipped += 1
            continue
        if questions:
            meta: dict[str, Any] = {
                'dataset_path': dataset_path,
                'prompt': _question_prompt(case, working_dir),
                'pre_start_scripts': _question_pre_start(questions.parent, case, working_dir),
                'language': case.get('language', 'unknown'),
                'data_source': case.get('source') or 'frontend_eval_questions',
                'title': case.get('title') or '',
            }
        else:
            meta = {
                'dataset_path': dataset_path,
                # The prompt IS the instance here: a frontend case carries its own
                # text, where a SWE row is looked up by id inside the task.
                'prompt': str(case.get('prompt') or ''),
                'language': case.get('language', 'unknown'),
                'data_source': case.get('data_source', ''),
            }
        if limit is not None and produced >= limit:
            return
        yield (instance_id, meta)
        produced += 1


def _langfuse_overlay(base: dict[str, Any], instance_id: str, agent_name: Any, model: str) -> dict[str, Any]:
    """Name each Langfuse trace after its instance, as the production compiler does; the YAML cannot know the id."""
    if 'langfuse' not in (base.get('processors') or {}):
        return {}
    args: dict[str, Any] = {'trace_metadata': {'instance_id': instance_id, 'agent_name': agent_name, 'model': model}}
    if not os.getenv('LANGFUSE_TRACE_NAME'):
        args['trace_name'] = instance_id
    return {'processors': {'langfuse': {'args': args}}}


def _questions_path() -> Path | None:
    """`FRONTEND_QUESTIONS`: a jsonl of eval questions (turns / assets / repo / title / id).

    Off by default: without it the generator reads one prompt per row from DATASET_PATH, which is
    what a rollout uses. With it, each row is converted the same way the 21-question eval packs do
    (see `frontend/eval/runs/humaneval_*/generator.py`): turns become the prompt, `assets` are
    unpacked into the workspace by pre-start scripts, `repo` is cloned at its commit.
    """
    raw = os.getenv('FRONTEND_QUESTIONS', '').strip()
    if not raw:
        return None
    path = Path(raw).expanduser()
    if not path.is_file():
        raise FileNotFoundError(f'FRONTEND_QUESTIONS={raw!r} is not a file')
    return path


def _question_prompt(row: dict[str, Any], working_dir: str = DEFAULT_WORKING_DIR) -> str:
    """Verbatim from the eval packs, so a rollout and an eval read the same prompt."""
    turns = row.get('turns') or []
    body = []
    for i, turn in enumerate(turns, 1):
        text = turn if isinstance(turn, str) else json.dumps(turn, ensure_ascii=False)
        if len(turns) == 1:
            body.append(text.strip())
        else:
            body.append(f'## 第 {i} 轮\n{text.strip()}')
    notes = []
    assets = [str(p) for p in (row.get('assets') or [])]
    if assets:
        notes.append('本题附件已放在工作区，相对路径：' + '、'.join(assets) + '。请直接使用这些文件。')
    if row.get('repo'):
        notes.append(f'仓库已检出到 {working_dir}，在现有代码上改。')
    head = f'# {row.get("title") or row["id"]}\n'
    if notes:
        head += '\n'.join(notes) + '\n\n'
    return head + '\n\n'.join(body)


def _asset_scripts(asset_root: Path, instance_id: str, working_dir: str = DEFAULT_WORKING_DIR) -> list[dict[str, Any]]:
    """Write ``assets/<id>/`` into ``working_dir`` in chunks (verbatim from the eval packs).

    Pre-start runs as ``python -c <script>``; one argument over MAX_ARG_STRLEN (128KB) dies with
    E2BIG, so the base64 blob is appended in pieces.
    """
    src = asset_root / 'assets' / instance_id
    if not src.is_dir():
        return []
    buf = io.BytesIO()
    with tarfile.open(fileobj=buf, mode='w:gz') as tar:
        for path in sorted(src.rglob('*')):
            if path.is_file():
                tar.add(path, arcname=path.relative_to(src).as_posix())
    blob = base64.b64encode(buf.getvalue()).decode('ascii')
    out: list[dict[str, Any]] = [{'interpreter': 'bash', 'fail_on_error': True, 'code': 'rm -f /tmp/nexus_assets.b64'}]
    step = 60000
    for i in range(0, len(blob), step):
        part = blob[i : i + step]
        out.append({
            'interpreter': 'python',
            'fail_on_error': True,
            'timeout': 60,
            'code': f'open("/tmp/nexus_assets.b64","a").write("{part}")\n',
        })
    out.append({
        'interpreter': 'python',
        'fail_on_error': True,
        'timeout': 300,
        'code': (
            'import base64, io, tarfile\n'
            'from pathlib import Path\n'
            'raw = base64.b64decode(Path("/tmp/nexus_assets.b64").read_text())\n'
            f'dst = Path({working_dir!r})\n'
            'dst.mkdir(parents=True, exist_ok=True)\n'
            'n = 0\n'
            'with tarfile.open(fileobj=io.BytesIO(raw), mode="r:gz") as tar:\n'
            '    for member in tar.getmembers():\n'
            '        name = member.name\n'
            '        if not member.isfile() or name.startswith("/") or ".." in name.split("/"):\n'
            '            continue\n'
            '        out = dst / name\n'
            '        out.parent.mkdir(parents=True, exist_ok=True)\n'
            '        src_f = tar.extractfile(member)\n'
            '        if src_f is None:\n'
            '            raise SystemExit("missing " + name)\n'
            '        out.write_bytes(src_f.read())\n'
            '        n += 1\n'
            'print("ASSETS", n)\n'
            'if n == 0:\n'
            '    raise SystemExit("no assets unpacked")\n'
        ),
    })
    return out


def _question_pre_start(
    asset_root: Path, row: dict[str, Any], working_dir: str = DEFAULT_WORKING_DIR
) -> list[dict[str, Any]]:
    """Verbatim from the eval packs: mkdir the working dir, clone ``repo`` at its commit, unpack assets."""
    quoted = shlex.quote(working_dir)
    scripts: list[dict[str, Any]] = [{'interpreter': 'bash', 'code': f'mkdir -p {quoted}', 'fail_on_error': True}]
    repo = row.get('repo') or {}
    if repo.get('url') and repo.get('commit'):
        url = str(repo['url'])
        commit = str(repo['commit'])
        scripts.append({
            'interpreter': 'bash',
            'fail_on_error': True,
            'timeout': 600,
            'code': (
                f'rm -rf {quoted}/* {quoted}/.[!.]* 2>/dev/null || true\n'
                f'git clone --no-checkout {url!s} {quoted}\n'
                f'git -C {quoted} checkout {commit}\n'
            ),
        })
    scripts.extend(_asset_scripts(asset_root, str(row['id']), working_dir))
    return scripts


def _task_config_extra(base: dict[str, Any]) -> dict[str, Any]:
    """`FRONTEND_TASK_CONFIG_EXTRA`: a JSON object merged into every instance's `task_config`.

    For the task fields a runtime YAML cannot hold on every checkout -- e.g. `tar_exclude_paths`,
    `tar_exclude_extensions`, `task_subdir`, which only the frontend checkout's `FrontendTaskInput`
    declares. Writing them in the YAML would make `validate.py` reject the file on a checkout
    that lacks them (correctly: they would be dropped silently). Here they are checked against
    the checkout actually running, so the same command line works on both and fails loudly,
    before any sandbox, on the one that cannot honour it.

    Per-instance values (`cache_task_id`, `prompt`, ...) do not belong here: they are new for every
    instance, so they stay in `generate_configs`.
    """
    raw = os.getenv('FRONTEND_TASK_CONFIG_EXTRA', '').strip()
    if not raw:
        return {}
    extra = json.loads(raw)
    if not isinstance(extra, dict):
        raise ValueError(f'FRONTEND_TASK_CONFIG_EXTRA must be a JSON object, got {type(extra).__name__}')
    _require_consumers({**base, 'task_config': deep_merge(base.get('task_config') or {}, extra)})
    return extra


def _image_by_agent() -> dict[str, str]:
    """`FRONTEND_IMAGE_BY_AGENT`: a JSON object mapping `agent_name` to a webenv image.

    Production runs codebuddy / dsh / handy / pi / workbuddy on a node22 image and the rest on the
    default one. A runtime YAML belongs to one run and `FRONTEND_IMAGE` sets one image for all of
    it, so a mixed batch cannot be written there. An agent not listed keeps the runtime's image.
    """
    raw = os.getenv('FRONTEND_IMAGE_BY_AGENT', '').strip()
    if not raw:
        return {}
    mapping = json.loads(raw)
    if not isinstance(mapping, dict) or not all(
        isinstance(k, str) and isinstance(v, str) and v for k, v in mapping.items()
    ):
        raise ValueError(
            'FRONTEND_IMAGE_BY_AGENT must be a JSON object of non-empty strings: {"<agent_name>": "<image>"}'
        )
    return mapping


def _eval_times() -> int:
    """`EVAL_TIMES`: how many attempts per instance. Default 1, must be >= 1.

    Mirrors ``nexus/extensions/tasks/hitl/recipe/terminal/generator.py``: with more than one, each
    run gets ``run_id = <instance_id>__try<N>`` so the attempts are distinct rows in the DB.
    """
    times = _env_int('EVAL_TIMES', default=1)
    if times is None or times < 1:
        raise ValueError(f'EVAL_TIMES must be a positive integer, got {os.getenv("EVAL_TIMES")!r}')
    return times


def generate_configs() -> Iterator[BatchItem]:
    # Folded once: the three run parts do not vary by instance.
    base = load_config()
    probe = _probe_llm(base)
    task_extra = _task_config_extra(base)
    images = _image_by_agent()
    if image := images.get(str(base.get('agent_name'))):
        base = deep_merge(base, {'runtime_provider_config': {'image': image}})
    eval_times = _eval_times()
    main_client = (base.get('openai_clients') or {}).get('main') or {}
    client_name = str(main_client.get('name') or '')
    main_client_args = _resolve_scalars((main_client.get('config') or {}).get('client_args') or {})
    # One source for all four places it has to agree: task_config, agent_config, the
    # pre-start scripts and the prompt.
    working_dir = _checked_working_dir(base)

    for instance_id, meta in _iter_selected_instances(working_dir):
        for attempt in range(1, eval_times + 1):
            overlay: dict[str, Any] = {
                'task_config': {
                    **task_extra,
                    'instance_id': instance_id,
                    'dataset_path': meta['dataset_path'],
                    'prompt': meta['prompt'],
                }
            }
            cache_id = _per_instance_cache_id(instance_id, attempt)
            client_overlay, effective_cache_id = cache_task_overlay(
                client_name, main_client_args, probe['api_key'], cache_id
            )
            if client_overlay:
                overlay['openai_clients'] = {'main': {'config': client_overlay}}
            # The agent's working_dir (agent axis, default /testbed) must match the task's; follow the task.
            agent_overlay: dict[str, Any] = {}
            if 'working_dir' in (base.get('task_config') or {}):
                agent_overlay['working_dir'] = working_dir
            # `pre_start_scripts` lives on the AGENT input (codebuddy / opencode), not on
            # frontend_task -- putting it in task_config would be dropped silently on a checkout
            # whose task does not declare it. The eval packs put it on the agent for the same reason.
            if ps := meta.get('pre_start_scripts'):
                agent_overlay['pre_start_scripts'] = ps
            if agent_overlay:
                overlay['agent_config'] = agent_overlay
            overlay = deep_merge(overlay, _langfuse_overlay(base, instance_id, base.get('agent_name'), probe['model']))

            instance_config = deep_merge(base, overlay)
            metadata: dict[str, Any] = {
                'instance_id': instance_id,
                'agent_name': instance_config.get('agent_name'),
                'model': probe['model'],
                'language': meta['language'],
                'data_source': meta['data_source'],
            }
            if effective_cache_id is not None:
                # Only when the wire really carried it; HyWay passes this to the CoT hack.
                metadata['cache_task_id'] = effective_cache_id
            yield BatchItem(
                config=RunInput.from_dict(instance_config),
                run_id=instance_id if eval_times == 1 else f'{instance_id}__try{attempt}',
                metadata=metadata,
            )


def get_total_count() -> int:
    return sum(1 for _ in _iter_selected_instances()) * _eval_times()


def get_batch_metadata() -> dict[str, Any]:
    base = load_config()
    probe = _probe_llm(base)
    include_ids, exclude_ids = _selected_instance_filters()
    return {
        'task_name': base.get('task_name'),
        'agent_name': base.get('agent_name'),
        'model': probe['model'],
        'runtime_config': os.getenv('RUNTIME_CONFIG', ''),
        'agent_config': os.getenv('AGENT_CONFIG', ''),
        'llm_config': os.getenv('LLM_CONFIG', ''),
        'include_instances': sorted(include_ids),
        'exclude_instances': sorted(exclude_ids),
        'total': get_total_count(),
    }
