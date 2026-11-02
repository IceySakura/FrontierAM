"""Task-side spec loader — the SINGLE place code_mt reads dataset fields.

Design goal (mirrors terminal/task.py): the generator and the hytuner dataset
loader pass ONLY a path (`task_dir`) plus the few provisioning knobs the
framework needs before the task runs (docker image, resource spec, runtime
topology). EVERY business / grading field is read HERE, at task runtime, from
`{task_dir}/task.toml` — so both entry points stay byte-aligned and no field can
be silently dropped (the pre-refactor generator, for instance, never forwarded
`[other].setup_commands`, so G1/G2 git-truncation + gitignore hygiene never ran).

`load_task_spec` returns a plain dict keyed by CodeMTTaskInput field names (plus
`problem_statement`, which is stashed into ctx.task_config, not a model field).
`apply_task_spec` writes those onto the task_input, but ONLY for keys the caller
did NOT already provide — precedence is: index/generator override > task.toml.
"""

from __future__ import annotations

import re

from pathlib import Path
from typing import Any

from nexus.extensions.projects.code_mt.task.models import LLMJudgeConfig, ScoreDimension


def _load_toml(path: Path) -> dict[str, Any]:
    try:
        import tomllib
    except ImportError:
        import tomli as tomllib
    with open(path, 'rb') as f:
        return tomllib.load(f)


def load_task_spec(task_dir: str) -> dict[str, Any]:
    """Read {task_dir}/task.toml and return CodeMTTaskInput field values.

    Reads [other] (business/grading knobs), [verifier] (eval mode/timeout/env),
    [environment] (workdir), [agent] (timeout), [metadata] (language), and the
    baked problem_statement.md. Never touches provisioning-only fields (image /
    resources) — those belong to the generator / hytuner loader.
    """
    base = Path(task_dir)
    toml_path = base / 'task.toml'
    if not toml_path.is_file():
        return {}
    data = _load_toml(toml_path)

    other = data.get('other', {})
    env = data.get('environment', {})
    meta = data.get('metadata', {})
    agent_cfg = data.get('agent', {})
    verifier_cfg = data.get('verifier', {})
    judge_cfg = data.get('judge', {})

    # verifier_mode: harbor's native runtime topology, read straight from
    # [verifier].environment_mode ("separate" == SWE artifact-collect + fresh
    # verifier runtime; "shared"/absent == grade in the agent's runtime). This is
    # the SAME vocabulary harbor_bridge uses (resolve_task_verifier_mode); code_mt
    # no longer derives a proprietary eval_mode. Harbor's default is shared. The
    # pre-0720 [verifier].mode / [other].eval_mode are honored only as a fallback:
    # legacy "dual_runtime" -> separate, everything else -> shared.
    environment_mode = verifier_cfg.get('environment_mode')
    legacy_mode = verifier_cfg.get('mode') or other.get('eval_mode')
    if environment_mode in ('shared', 'separate'):
        verifier_mode = environment_mode
    elif legacy_mode == 'dual_runtime':
        verifier_mode = 'separate'
    else:
        verifier_mode = 'shared'
    language = other.get('programming_language') or meta.get('programming_language') or meta.get('language', '')

    # Image boot hook ([environment.boot] + [environment.healthcheck]). harbor's
    # Task model has NO boot field, so — exactly like harbor_bridge's generator
    # (_build_boot_config) — we read it from raw task.toml and hand it to the
    # shared _boot_image_services helper at runtime. boot_env is sourced from
    # [environment.env] because gongfeng bypasses the image ENTRYPOINT (its ENV
    # lines are lost). healthcheck fields tune the post-boot readiness poll.
    boot_section = env.get('boot') or {}
    boot_command = str(boot_section.get('command', '')).strip() if isinstance(boot_section, dict) else ''
    boot_env = {str(k): str(v) for k, v in (env.get('env') or {}).items()}
    healthcheck = env.get('healthcheck') or {}
    hc_command = healthcheck.get('command', '') if isinstance(healthcheck, dict) else ''
    hc_command = hc_command.strip() if isinstance(hc_command, str) else ''
    hc_url_match = re.search(r'(https?://\S+)', hc_command) if hc_command else None

    # score_kind is AUTHORITATIVELY decided by [judge].reward_kind: the judge owns
    # all reward fields, and reward_kind fixes the final numeric RANGE — binary
    # snaps the merged score to 0/1, continuous keeps the float. [other] fields
    # (score_dimensions / score_merge) only decide HOW dimensions combine (all in
    # float space); they must NOT override the final 0/1 decision. [other].score_kind
    # is only a fallback for pre-0720 dirs that carry no [judge].reward_kind.
    reward_kind = judge_cfg.get('reward_kind')
    if reward_kind == 'binary':
        score_kind = 'binary'
    elif reward_kind == 'continuous':
        score_kind = 'partial'
    else:
        score_kind = other.get('score_kind', 'binary')

    # Business / grading fields the task + autosetup read from task_config at
    # runtime (via the task.py flatten of task_input.extra). Builder tasks carry
    # these ONLY in [other] (no problem_statement.md, and a task_dir-only launch
    # has no index override), so they MUST be sunk here or the builder prompt
    # renders hollow: empty test_patch / golden_patch / base_image => the agent
    # flies blind => systematic full_state_fails (this was the 92%->74% regression).
    task_type = other.get('task_type', 'SWE')
    extra: dict[str, Any] = {
        'programming_language': language,
        'language': language,
        'tools': other.get('tools', []),
    }
    for _k in (
        'base_image',
        'base_commit',
        'repo',
        'test_patch',
        'golden_patch',
        'level',
        'call_mode',
        'two_stage_submit',
        'instance_id',
    ):
        if _k in other:
            extra[_k] = other[_k]

    # Builder prompt inputs from task-dir files. The 0720 generator does NOT inline
    # test_patch / golden_patch into [other]; it writes them as task-dir files
    # (tests/{test,golden}.patch, mirrored under .source/). The builder prompt
    # (instruction.md {test_patch_display}/{golden_patch_display}) and the agent
    # autosetup both read these from task_config.extra, so sink them from disk when
    # [other] omitted them — otherwise the builder prompt renders hollow (empty
    # patches => systematic full_state_fails, the 92%->74% regression). Gated to
    # Builder so a golden_patch never leaks into a SWE / RepoQA task_config. [other]
    # still wins (checked via extra.get) when a generator did inline them.
    if task_type == 'Builder':
        for _pk, _rels in (
            ('test_patch', ('tests/test.patch', '.source/test.patch')),
            ('golden_patch', ('tests/golden.patch', '.source/golden.patch')),
        ):
            if extra.get(_pk):
                continue
            for _rel in _rels:
                _p = base / _rel
                if _p.is_file():
                    extra[_pk] = _p.read_text(encoding='utf-8')
                    break

    spec: dict[str, Any] = {
        'task_type': other.get('task_type', 'SWE'),
        'setup_commands': list(other.get('setup_commands', [])),
        # Harbor-native verifier topology + boot (all consumed by task.py exactly
        # as harbor_bridge does; collect hooks / artifacts / tests are read by the
        # harbor Verifier straight from the harbor Task, so they are NOT carried here).
        'verifier_mode': verifier_mode,
        'boot_command': boot_command,
        'boot_env': boot_env,
        'healthcheck_command': hc_command,
        'healthcheck_url': hc_url_match.group(1) if hc_url_match else '',
        'healthcheck_retries': int(healthcheck.get('retries') or 180) if isinstance(healthcheck, dict) else 180,
        'healthcheck_interval_sec': float(healthcheck.get('interval_sec') or 3.0)
        if isinstance(healthcheck, dict)
        else 3.0,
        'healthcheck_timeout_sec': float(healthcheck.get('timeout_sec') or 3.0)
        if isinstance(healthcheck, dict)
        else 3.0,
        'healthcheck_start_period_sec': float(healthcheck.get('start_period_sec') or 0.0)
        if isinstance(healthcheck, dict)
        else 0.0,
        'skip_tests_upload': bool(verifier_cfg.get('skip_tests_upload', False)),
        'test_timeout': float(verifier_cfg.get('timeout_sec', 1800.0)),
        'mt_max_turns': int(other.get('mt_max_turns', 1)),
        'mt_feedback_max_length': int(other.get('mt_feedback_max_length', 3000)),
        'agent_timeout': float(agent_cfg.get('timeout_sec', 7200.0)),
        'score_merge': other.get('score_merge', 'sequential'),
        'score_kind': score_kind,
        'eval_cleanup_strategy': other.get('eval_cleanup_strategy'),
        'verifier_env': {str(k): str(v) for k, v in (verifier_cfg.get('env') or {}).items()},
        'extra': extra,
    }

    if 'workdir' in env:
        spec['workdir'] = env['workdir']

    # Reward-hacking / baseline monitor policy. These are CodeMTTaskInput fields;
    # the parquet path sets them directly, but a task-dir launch (harbor generator)
    # needs them sunk from [other] so the 0814 convertor forms can declare
    # anti-hack / baseline policy per dataset. Only sink when present so the
    # CodeMTTaskInput defaults (detection on, anti_hack off) otherwise stand.
    for _mk in (
        'enable_anti_hack',
        'enable_anti_hack_prompt',
        'is_validation',
        'enable_reward_hacking_detection',
        'enable_output_monitoring',
        'monitors',
        'baseline_penalties',
        'codeswitch_penalty',
        'output_format_bad_pattern_penalty',
        'str_replace_old_eq_new_penalty',
        'reward_lower_bound',
    ):
        if _mk in other:
            spec[_mk] = other[_mk]

    dims = other.get('score_dimensions') or []
    if dims:
        spec['score_dimensions'] = [ScoreDimension(**d) for d in dims]

    judge = other.get('llm_judge')
    if isinstance(judge, dict) and judge:
        spec['llm_judge'] = LLMJudgeConfig(**judge)

    ps = base / 'problem_statement.md'
    if ps.is_file():
        spec['problem_statement'] = ps.read_text(encoding='utf-8')
    elif other.get('problem_statement'):
        # Builder tasks ship the statement inline in [other], not as a .md file.
        spec['problem_statement'] = other['problem_statement']

    return spec


def apply_task_spec(task_input: Any, provided_task_config: dict[str, Any], spec: dict[str, Any]) -> None:
    """Write spec onto task_input for keys the caller did NOT already provide.

    `provided_task_config` is the raw dict the generator / index put in
    RunInput.task_config (before the task merged model defaults into it). Any key
    present there is an explicit index/generator override and wins over task.toml.
    `problem_statement` is not a model field, so it is stashed into the shared
    task_config for task.py's turn-0 prompt.
    """
    provided = set(provided_task_config or {})
    for key, val in spec.items():
        if key == 'problem_statement':
            if not provided_task_config.get('problem_statement'):
                provided_task_config['problem_statement'] = val
            continue
        if key == 'extra':
            # task.py flattens task_input.extra LAST into task_config, so a naive
            # overwrite here would let task.toml beat an index/generator override.
            # Honor precedence: drop any extra key already provided at the
            # task_config top level, and keep anything already on task_input.extra.
            existing = dict(getattr(task_input, 'extra', {}) or {})
            loaded = {k: v for k, v in val.items() if k not in provided}
            task_input.extra = {**loaded, **existing}
            continue
        if key in provided:
            continue
        if hasattr(task_input, key):
            setattr(task_input, key, val)
