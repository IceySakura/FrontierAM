#!/usr/bin/env python3
"""aidata —— aidata 基准测试的 case-runner CLI 。

用法::

    ./aidata.py run <case_id> [-- extra args passed to nexus run-single]
    ./aidata.py list
    ./aidata.py validate <case_id>
    ./aidata.py package <case_id> [out.tgz]
    ./aidata.py --help

向后兼容： ``aidata.py <case_id>`` 若没有显式的子命令，会被视为
``aidata.py run <case_id>`` ，因此老的肌肉记忆（ ``run-case.sh <id>`` ）
仍然可用。

设计说明 —— 为什么使用 Python 而不是 bash：

- ``validate`` 可以复用引擎的 case.yaml / rubric.yaml 加载器
  （ ``engine.case_spec`` ），因此 CLI 与运行时对什么形状可接受
  达成一致，无需在 shell 中重复 schema 规则。
- 当一个 case 半损坏时提供更好的错误信息（回溯）。
- ``package`` 使用标准库 ``tarfile`` —— 不依赖各发行版之间宿主
  ``tar`` 标志的兼容性。
- 仍然可以轻易地重入：每次子进程调用都通过 ``subprocess.run``
  并显式传递 env 。
"""

from __future__ import annotations

import argparse
import contextlib
import os
import subprocess
import sys
import tarfile
import time

from pathlib import Path
from typing import Any

# Schema 常量位于 case.schema —— 这里以前是重复的一份，且已发生漂移。
# 保留下划线前缀的拼写作为本地名称，以避免改动本文件内的每一处调用点
# （代码量不小）。
from nexus.extensions.tasks.aidata.case.schema import (
    _ALLOWED_AGENT_TOOLS_BY_CATEGORY,
    _ALLOWED_DIMENSIONS,
    _LEGACY_VERIFY_KINDS,
    _METADATA_DATE_RE,
    _V42_ASSERT_TYPES,
)
from nexus.extensions.tasks.aidata.rubric.reward import validate_reward_config


SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parents[3]
# CASES_DIR 可能位于此仓库之外 —— 当上游 nexus aidata
# 扩展被消费而不打包 case 数据时，可通过 ``AIDATA_CASES_ROOT``
# 指向下游的 ``cases/`` 树（引擎的 ``paths.py`` 支持相同的 env ）。
# 若在仓内 / 单树开发中，则回退到 ``./cases`` 。
CASES_DIR = Path(os.getenv('AIDATA_CASES_ROOT') or (SCRIPT_DIR / 'cases'))
TEMPLATE = SCRIPT_DIR / 'runtime.template.yaml'  # litellm （默认 / 非 Claude ）
TEMPLATE_ANTHROPIC = SCRIPT_DIR / 'runtime.anthropic.template.yaml'  # vertex_anthropic passthrough （Claude ）
RESULTS_DIR = REPO_ROOT / 'workspace' / 'mcp' / 'tmp'

# --template 接受下列别名或一个路径。 编排器会按模型自动挑选；
# 手工的 `aidata.py run` 则显式选择（默认为 litellm ）。
_TEMPLATE_ALIASES = {
    'litellm': TEMPLATE,
    'default': TEMPLATE,
    'anthropic': TEMPLATE_ANTHROPIC,
    'vertex': TEMPLATE_ANTHROPIC,
    'passthrough': TEMPLATE_ANTHROPIC,
}


def _resolve_template(name: str | None) -> Path:
    """解析 --template ：接受别名、路径、或者默认的 litellm 模板。"""
    if not name:
        return TEMPLATE
    if name in _TEMPLATE_ALIASES:
        return _TEMPLATE_ALIASES[name]
    p = Path(name).expanduser()
    if not p.exists() and not p.is_absolute():
        p = SCRIPT_DIR / name  # 允许相对 aidata 目录的裸文件名
    return p


# ---------------------------------------------------------------------------
# 辅助函数
# ---------------------------------------------------------------------------


def _err(msg: str, *, exit_code: int = 2) -> None:
    print(f'error: {msg}', file=sys.stderr)
    sys.exit(exit_code)


def _manifest_file(case_dir: Path) -> Path | None:
    """v4.2 的 ``task.yaml`` 优先于遗留的 ``case.yaml`` ；返回第一个存在的
    文件，若二者都没有则返回 None 。
    """
    for name in ('task.yaml', 'case.yaml'):
        p = case_dir / name
        if p.exists():
            return p
    return None


def _rubric_file(case_dir: Path) -> Path | None:
    """规范位置是 ``<case_dir>/rubric.yaml`` （扁平结构）。 若存在则返回
    该路径，否则返回 None 。
    """
    p = case_dir / 'rubric.yaml'
    return p if p.exists() else None


def _available_cases() -> list[str]:
    if not CASES_DIR.is_dir():
        return []
    found: list[str] = []
    for root, dirs, _files in os.walk(CASES_DIR):
        # 跳过隐藏 / 下划线前缀的目录。
        dirs[:] = [d for d in dirs if not d.startswith(('.', '_'))]
        root_path = Path(root)
        if _manifest_file(root_path) is not None:
            found.append(root_path.relative_to(CASES_DIR).as_posix())
    return sorted(found)


def _require_case(case_id: str) -> Path:
    if not case_id:
        _err('<case_id> is required')
    case_dir = CASES_DIR / case_id
    if not case_dir.is_dir():
        print(f'error: no such case: {case_dir}', file=sys.stderr)
        print('available cases:', file=sys.stderr)
        for name in _available_cases():
            print(f'  {name}', file=sys.stderr)
        sys.exit(2)
    if _manifest_file(case_dir) is None:
        _err(f'missing manifest: expected {case_dir / "task.yaml"} (preferred) or {case_dir / "case.yaml"} (legacy)')
    return case_dir


def _candidate_env_files() -> list[Path]:
    """返回 CLI 使用的 .env 文件，按优先级顺序排列。

    与 ``generator.py`` 和 ``tests/conftest.py`` 保持一致：下游 cases
    树可以在其 fixture 旁携带凭证，而上游仓库仍可通过
    ``<repo>/.env`` 直接进行开发。
    """
    candidates: list[Path] = []
    cases_env = CASES_DIR.parent / '.env'
    if cases_env.is_file():
        candidates.append(cases_env)
    repo_env = REPO_ROOT / '.env'
    if repo_env.is_file() and repo_env not in candidates:
        candidates.append(repo_env)
    return candidates


def _load_env_into(env: dict[str, str]) -> None:
    """将 ``.env`` 文件合并到 ``env`` 中。

    值按原样使用 —— 不做 shell 展开，除了外层一对匹配的引号会被去除
    以外不做其他去引号处理。 已存在的环境变量值优先于文件；
    cases 树的 ``.env`` 优先于仓库的 ``.env`` 。
    """
    for env_file in _candidate_env_files():
        for raw_line in env_file.read_text(encoding='utf-8').splitlines():
            line = raw_line.strip()
            if not line or line.startswith('#'):
                continue
            if line.startswith('export '):
                line = line[len('export ') :].lstrip()
            if '=' not in line:
                continue
            key, _, value = line.partition('=')
            key = key.strip()
            value = value.strip()
            if len(value) >= 2 and value[0] == value[-1] and value[0] in ('"', "'"):
                value = value[1:-1]
            env.setdefault(key, value)
        print(f'[aidata] loaded {env_file}', file=sys.stderr)


# ---------------------------------------------------------------------------
# 子命令： run
# ---------------------------------------------------------------------------


def cmd_run(args: argparse.Namespace) -> int:
    case_dir = _require_case(args.case_id)
    if _rubric_file(case_dir) is None:
        _err(
            f'{args.case_id}: no rubric.yaml — a committed cases/<id>/rubric.yaml is required to score '
            'this case (there is no auto-generated fallback). Hand-write one first.'
        )

    env = os.environ.copy()
    _load_env_into(env)

    # 将模板渲染到仓库树之外的一个短命文件中。
    run_dir = Path(os.environ.get('TMPDIR', '/tmp')) / 'nexus_aidata_runs'
    run_dir.mkdir(parents=True, exist_ok=True)
    # best-effort ：某些文件系统会拒绝 chmod （NFS 、 fuse 等）
    with contextlib.suppress(OSError):
        run_dir.chmod(0o700)

    ts = str(int(time.time()))
    # case_id 可能包含 '/' （例如 ``online_shopping/scenario_add_cart`` ） ——
    # 确保 run yaml 与输出 json 的每个 case 父目录都存在，
    # 否则 ``write_text`` / ``nexus run-single --output`` 会因嵌套目录
    # 抛出 ``FileNotFoundError`` 。
    template = _resolve_template(getattr(args, 'template', None))
    if not template.exists():
        _err(f'template not found: {template}')

    run_yaml = run_dir / f'{args.case_id}.{ts}.yaml'
    run_yaml.parent.mkdir(parents=True, exist_ok=True)
    rendered = template.read_text(encoding='utf-8').replace('{{CASE_ID}}', args.case_id)
    run_yaml.write_text(rendered, encoding='utf-8')

    out = RESULTS_DIR / f'aidata_{args.case_id}_{ts}.result.json'
    out.parent.mkdir(parents=True, exist_ok=True)

    print(f'[aidata] template: {template.name}', file=sys.stderr)
    print(f'[aidata] yaml:   {run_yaml}', file=sys.stderr)
    print(f'[aidata] output: {out}', file=sys.stderr)

    cmd = ['uv', 'run', 'nexus', 'run-single', str(run_yaml), '--output', str(out)]
    cmd.extend(args.extra)
    return subprocess.call(cmd, cwd=str(REPO_ROOT), env=env)


# ---------------------------------------------------------------------------
# 子命令： list
# ---------------------------------------------------------------------------


def _case_summary(case_dir: Path) -> str:
    """优先取 README.md 中第一行非空、非标题的行；否则回退到
    task.yaml / case.yaml 中的 ``task_id`` 。"""
    readme = case_dir / 'README.md'
    if readme.is_file():
        for line in readme.read_text(encoding='utf-8').splitlines():
            stripped = line.strip()
            if stripped and not stripped.startswith('#'):
                return stripped
    manifest = _manifest_file(case_dir)
    if manifest is not None:
        for line in manifest.read_text(encoding='utf-8').splitlines():
            if line.startswith('task_id:'):
                return line.split(':', 1)[1].strip()
    return ''


def cmd_list(_args: argparse.Namespace) -> int:
    cases = _available_cases()
    if not cases:
        print(f'(no cases found in {CASES_DIR})', file=sys.stderr)
        return 0
    for case_id in cases:
        case_dir = CASES_DIR / case_id
        summary = _case_summary(case_dir)
        print(f'  {case_id:<32s}  {summary}')
    return 0


# ---------------------------------------------------------------------------
# 子命令： validate
# ---------------------------------------------------------------------------


def _validate_case(case_id: str) -> list[str]:
    """返回一个人类可读的问题列表（空 = OK ）。

    接受两种布局：

    - v4.2 （首选）： ``task.yaml`` + ``instruction.md`` +
      ``rubric.yaml`` ，其 ``checks[].assert.type`` 为
      {state_equals, text_contains, tool_called, judge_claim, test_result} 之一。
    - v2 （遗留，仍然会被警告但视为有效）： ``case.yaml`` + ``rubric.yaml`` ，
      其 ``checks[].verify`` 为 {llm_judge, eq, regex, shell} 之一。

    纯 Python + 标准库 + pyyaml （已经是 aidata 的依赖）；
    这就是 CI 运行的内容，因此新的 DSL 规则应在 case 落地前
    在这里被强制执行。
    """
    case_dir = CASES_DIR / case_id
    problems: list[str] = []

    try:
        import yaml  # type: ignore[import-not-found]
    except ImportError:
        problems.append('PyYAML not installed — `uv run nexus modules install aidata`')
        return problems

    # ---- manifest （task.yaml 首选， case.yaml 遗留） -------------------
    manifest = _manifest_file(case_dir)
    if manifest is None:
        problems.append('missing manifest: expected task.yaml (preferred) or case.yaml (legacy)')
        manifest_name = 'task.yaml'
        data: Any = {}
    else:
        manifest_name = manifest.name
        try:
            data = yaml.safe_load(manifest.read_text(encoding='utf-8')) or {}
        except Exception as exc:
            problems.append(f'{manifest_name} parse error: {exc}')
            data = {}

    if manifest is not None and not isinstance(data, dict):
        problems.append(f'{manifest_name}: must be a mapping')
        data = {}

    # prompt 来源： task.yaml 内联 user_prompt 或 instruction.md
    if isinstance(data, dict):
        problems.extend(_validate_metadata(data.get('metadata'), manifest_name))
        has_inline_prompt = bool((data.get('user_prompt') or '').strip())
        ps = case_dir / 'instruction.md'
        has_ps = ps.exists() and ps.read_text(encoding='utf-8').strip()
        if not has_inline_prompt and not has_ps:
            problems.append(f'{manifest_name}: user_prompt is empty and instruction.md is missing/empty')
        for ctx in data.get('context_files') or []:
            candidates = [
                case_dir / 'data' / str(ctx),
                case_dir / 'files' / str(ctx),
                case_dir / str(ctx),
            ]
            if not any(c.exists() for c in candidates):
                problems.append(
                    f'{manifest_name}: context_files entry not found (looked in data/, files/, root): {ctx}'
                )
        boot = data.get('bootstrap_script')
        if boot and not (case_dir / str(boot)).exists():
            problems.append(f'{manifest_name}: bootstrap_script not found: {boot}')
        for spec in data.get('extra_mcp_servers') or []:
            if not isinstance(spec, dict) or not spec.get('name'):
                problems.append(f'{manifest_name}: extra_mcp_servers entry missing `name`: {spec!r}')
            elif 'client' not in spec and 'command' not in spec:
                problems.append(
                    f'{manifest_name}: extra_mcp_servers[{spec.get("name")!r}] '
                    f'missing `command` (flat form) or `client` (nested form)'
                )
        problems.extend(_validate_agent_tools(data.get('agent_tools'), manifest_name))

    # ---- rubric （cases/<id>/rubric.yaml ） --------------------------------
    rubric_file = _rubric_file(case_dir)
    if rubric_file is None:
        # 需要一个已提交的 rubric.yaml —— 任务运行器在没有它时会硬报错
        # （不存在自动生成的回退）。
        problems.append(f'{manifest_name}: no rubric.yaml (required for scoring)')
    else:
        rubric_rel = rubric_file.relative_to(case_dir).as_posix()
        try:
            r = yaml.safe_load(rubric_file.read_text(encoding='utf-8')) or {}
        except Exception as exc:
            problems.append(f'{rubric_rel} parse error: {exc}')
            r = {}
        if not isinstance(r, dict) or not r.get('checks'):
            problems.append(f'{rubric_rel}: must be a dict with non-empty `checks`')
        else:
            # v2 rubric 已冻结：现有的 59 个生产 case 在运行时仍会加载并给出
            # DeprecationWarning ，但新 case 必须使用 v4.2 。
            _version = r.get('version') if isinstance(r, dict) else None
            from nexus.extensions.tasks.aidata.case.schema import V2_VERSION_LITERALS

            if _version in V2_VERSION_LITERALS or (isinstance(_version, str) and _version.strip() in {'2', '2.0'}):
                problems.append(
                    f'{rubric_rel}: v2 rubric (version={_version!r}) is no longer accepted for new cases. '
                    f'Migrate to v4.2 by rewriting verify: blocks as assert: blocks. '
                    f'See RUBRIC.md §7 for the migration guide. '
                    f'(Existing v2 cases in production continue to run with a deprecation warning.)'
                )
                return problems
            is_v42 = str(r.get('version') or '').startswith('4.') or any(
                isinstance(c, dict) and 'assert' in c for c in r['checks']
            )
            seen_ids: set[str] = set()
            for i, c in enumerate(r['checks']):
                problems.extend(_validate_check(c, i, is_v42, rubric_rel, seen_ids))
            problems.extend(validate_reward_config(r, case_dir=case_dir, rubric_rel=rubric_rel))

    return problems


def _validate_agent_tools(raw: Any, manifest_name: str) -> list[str]:
    """对 ``task.yaml::agent_tools`` 进行静态校验。

    运行时解析位于 ``engine.case_spec._parse_agent_tools_spec`` ；
    保持此形状检查同步，让作者能从 ``aidata validate``
    获得快速反馈。
    """
    if raw is None:
        return []
    problems: list[str] = []

    if not isinstance(raw, dict):
        return [
            f'{manifest_name}: agent_tools must be a mapping of tool-directory → list, '
            f'e.g. {{web_search: [google_serper_search, visit]}}; got {type(raw).__name__}'
        ]

    unknown_categories = set(raw) - _ALLOWED_AGENT_TOOLS_BY_CATEGORY.keys()
    if unknown_categories:
        problems.append(
            f'{manifest_name}: agent_tools has unknown category keys: '
            f'{sorted(unknown_categories)}; expected one of '
            f'{sorted(_ALLOWED_AGENT_TOOLS_BY_CATEGORY)}'
        )

    for category, items in raw.items():
        if category not in _ALLOWED_AGENT_TOOLS_BY_CATEGORY:
            continue  # 已经在上面标记过
        if not isinstance(items, list):
            problems.append(f'{manifest_name}: agent_tools.{category} must be a list of tool names')
            continue
        allow = _ALLOWED_AGENT_TOOLS_BY_CATEGORY[category]
        for i, name in enumerate(items):
            if isinstance(name, bool):
                problems.append(
                    f'{manifest_name}: agent_tools.{category}[{i}] must be a tool name '
                    f'(file stem under nexus/extensions/tools/{category}/), not a boolean'
                )
                continue
            if not isinstance(name, str) or not name.strip():
                problems.append(f'{manifest_name}: agent_tools.{category}[{i}] must be a non-empty string')
                continue
            tool = name.strip()
            if tool not in allow:
                problems.append(
                    f'{manifest_name}: agent_tools.{category}[{i}]={tool!r} is not allowed; '
                    f'expected one of {sorted(allow)}'
                )

    return problems


def _validate_metadata(raw: Any, manifest_name: str) -> list[str]:
    """对 ``task.yaml::metadata`` 进行静态检查。 OK 时返回 [] ，否则返回
    一个人类可读的问题列表。 允许 ``None`` （早于 metadata 块的 case ），
    并会给出一条 WARN 风格的条目提醒作者去添加 —— 这不是硬失败，
    因此遗留 case 不必在同一 PR 里既做 metadata 迁移又做 rubric 调整。

    必须与 ``case/spec.py::_parse_case_metadata`` 保持一致，
    因为后者在任务时执行 —— 那个函数强制的规则，
    这里也应静态标记，这样 CI 能在运行时崩溃前捕获它们。
    """
    problems: list[str] = []
    if raw is None:
        problems.append(
            f'{manifest_name}: metadata block missing (add `metadata:` with at '
            f'least `created_at`; see SCHEMA.md §metadata)'
        )
        return problems
    if not isinstance(raw, dict):
        problems.append(f'{manifest_name}: metadata must be a mapping')
        return problems
    # pyyaml 会在我们看到之前把裸 ``2026-04-21`` 自动强制转换成 ``datetime.date`` 。
    # 与其强迫作者给每个日期加引号，不如接受两种形式并统一到字符串形式。
    created_at_raw = raw.get('created_at')
    import datetime as _dt

    if isinstance(created_at_raw, _dt.date) and not isinstance(created_at_raw, _dt.datetime):
        created_at = created_at_raw.isoformat()
    else:
        created_at = created_at_raw
    if not isinstance(created_at, str) or not _METADATA_DATE_RE.match(created_at):
        problems.append(f'{manifest_name}: metadata.created_at must be YYYY-MM-DD (got {created_at_raw!r})')
    authors = raw.get('authors')
    if authors is not None and (not isinstance(authors, list) or not all(isinstance(x, str) for x in authors)):
        problems.append(f'{manifest_name}: metadata.authors must be a list of strings')
    tags = raw.get('tags')
    if tags is not None and (not isinstance(tags, list) or not all(isinstance(x, str) for x in tags)):
        problems.append(f'{manifest_name}: metadata.tags must be a list of strings')
    return problems


def _validate_check(
    c: Any,
    i: int,
    is_v42: bool,
    rubric_rel: str,
    seen_ids: set[str],
) -> list[str]:
    """校验一个 rubric check 。 清晰地区分 v4.2 与遗留的 v2 规则。"""
    problems: list[str] = []
    if not isinstance(c, dict):
        problems.append(f'{rubric_rel}: checks[{i}] not a mapping')
        return problems
    cid = c.get('id')
    if not cid:
        problems.append(f'{rubric_rel}: checks[{i}].id is empty')
    else:
        if cid in seen_ids:
            problems.append(f'{rubric_rel}: duplicate check id {cid!r}')
        seen_ids.add(cid)
    if not c.get('description'):
        problems.append(f'{rubric_rel}: checks[{i}].description is empty')

    if is_v42:
        # v4.2 schema 分支
        dim = c.get('dimension')
        if not dim:
            problems.append(f'{rubric_rel}: checks[{i}].dimension is empty')
        elif dim not in _ALLOWED_DIMENSIONS:
            problems.append(f'{rubric_rel}: checks[{i}].dimension={dim!r} not in {sorted(_ALLOWED_DIMENSIONS)}')
        w = c.get('weight')
        if w is None:
            problems.append(f'{rubric_rel}: checks[{i}].weight is missing')
        else:
            try:
                w_f = float(w)
                if w_f <= 0:
                    problems.append(f'{rubric_rel}: checks[{i}].weight must be > 0 (got {w!r})')
            except (TypeError, ValueError):
                problems.append(f'{rubric_rel}: checks[{i}].weight must be numeric (got {w!r})')
        a = c.get('assert')
        if not isinstance(a, dict):
            problems.append(f'{rubric_rel}: checks[{i}].assert must be a mapping')
            return problems
        atype = a.get('type')
        if not atype:
            problems.append(f'{rubric_rel}: checks[{i}].assert.type is empty')
        elif atype not in _V42_ASSERT_TYPES:
            problems.append(f'{rubric_rel}: checks[{i}].assert.type={atype!r} not in {sorted(_V42_ASSERT_TYPES)}')
        # 特定原语的最小字段
        if atype == 'state_equals':
            jmespath_predicate_keys = (
                'exists',
                'equals',
                'one_of',
                'count_op',
                'contains',
                'contains_all',
                'contains_any',
                'regex_match',
                'gt',
                'gte',
                'lt',
                'lte',
            )
            if 'jmespath' in a:
                # jmespath 分支： query 必需，另加恰好一个谓词。
                if not str(a.get('jmespath') or '').strip():
                    problems.append(f'{rubric_rel}: checks[{i}].assert[state_equals] jmespath query is empty')
                present_preds = [k for k in jmespath_predicate_keys if k in a]
                # count_op + count_value 组合作为一个谓词槽位。
                if 'count_value' in a and 'count_op' not in a:
                    problems.append(f'{rubric_rel}: checks[{i}].assert[state_equals] count_value requires count_op')
                if 'count_op' in a and 'count_value' not in a:
                    problems.append(f'{rubric_rel}: checks[{i}].assert[state_equals] count_op requires count_value')
                if not present_preds:
                    problems.append(
                        f'{rubric_rel}: checks[{i}].assert[state_equals] jmespath shape needs '
                        f'one predicate: {list(jmespath_predicate_keys)!r}'
                    )
                elif len(present_preds) > 1:
                    problems.append(
                        f'{rubric_rel}: checks[{i}].assert[state_equals] jmespath shape accepts '
                        f'exactly one predicate (got {present_preds!r})'
                    )
            elif 'equals' not in a and 'one_of' not in a and 'expected' not in a:
                problems.append(
                    f'{rubric_rel}: checks[{i}].assert[state_equals] needs one of: '
                    f'equals / one_of / expected (or use `jmespath` with a predicate)'
                )
            # 数据源对遗留形状同样必需 —— 如果两者都不给，运行时求值器会
            # 短路到 INVALID/empty_path ，但我们更愿在此静态标记。
            # jmespath 形状在省略 `from` 时会默认为整个 state ，
            # 因此那里不需要数据源。
            if 'jmespath' not in a and not a.get('from') and not a.get('source'):
                problems.append(
                    f'{rubric_rel}: checks[{i}].assert[state_equals] needs '
                    f'a data source: `from` (slice name) or `source` '
                    f'(flat dotted path)'
                )
        elif atype == 'text_contains':
            if not any(k in a for k in ('contains', 'contains_all', 'contains_any', 'regex')):
                problems.append(
                    f'{rubric_rel}: checks[{i}].assert[text_contains] needs '
                    f'one of: contains / contains_all / contains_any / regex'
                )
        elif atype == 'tool_called':
            if not a.get('tool') and not a.get('tool_prefix') and not a.get('acceptable_tools'):
                problems.append(
                    f'{rubric_rel}: checks[{i}].assert[tool_called] needs tool, tool_prefix, or acceptable_tools'
                )
            problems.extend(_validate_acceptable_tools(a, i, rubric_rel, 'tool_called'))
            problems.extend(_validate_count_bounds(a, i, rubric_rel, 'tool_called'))
        elif atype == 'tool_called_sequence':
            steps = a.get('steps')
            if not isinstance(steps, list) or not steps:
                problems.append(
                    f'{rubric_rel}: checks[{i}].assert[tool_called_sequence] needs a non-empty `steps` list'
                )
            else:
                for j, step in enumerate(steps):
                    if not isinstance(step, dict):
                        problems.append(
                            f'{rubric_rel}: checks[{i}].assert[tool_called_sequence].steps[{j}] must be a mapping'
                        )
                        continue
                    if not step.get('tool') and not step.get('tool_prefix') and not step.get('acceptable_tools'):
                        problems.append(
                            f'{rubric_rel}: checks[{i}].assert[tool_called_sequence].steps[{j}] '
                            f'needs tool, tool_prefix, or acceptable_tools'
                        )
                    problems.extend(
                        _validate_acceptable_tools(
                            step,
                            i,
                            rubric_rel,
                            f'tool_called_sequence].steps[{j}',
                        )
                    )
        elif atype == 'judge_claim':
            if not (a.get('claim') or '').strip():
                problems.append(f'{rubric_rel}: checks[{i}].assert[judge_claim].claim is empty')
            if 'jmespath' in a and not isinstance(a.get('jmespath'), str):
                problems.append(f'{rubric_rel}: checks[{i}].assert[judge_claim] jmespath query must be a string')
            elif 'jmespath' in a and not a.get('jmespath', '').strip():
                problems.append(f'{rubric_rel}: checks[{i}].assert[judge_claim] jmespath query is empty')
        elif atype == 'test_result':
            # 显式指定 test_name ，或者依赖 check.id 通过命名约定进行查找
            pass
    else:
        # v2 遗留 schema —— 保留 v4.2 之前的规则，让老的 case 在迁移期间
        # 仍能正常校验。 必需字段与具体 verifier 相关，与
        # engine/rubric.py 的运行时默认值一致：
        #
        # - llm_judge 默认 source 为 "all" ； claim 必需。
        # - regex 默认 source 为 "all" ； pattern 必需。
        # - shell 只需要 command 。
        # - eq 需要 source path + expected 值。
        verify = c.get('verify') or 'llm_judge'
        if verify not in _LEGACY_VERIFY_KINDS:
            problems.append(f'{rubric_rel}: checks[{i}].verify={verify!r} not in {sorted(_LEGACY_VERIFY_KINDS)}')
        elif verify == 'llm_judge':
            if not (c.get('claim') or '').strip():
                problems.append(f'{rubric_rel}: checks[{i}].claim is empty')
        elif verify == 'regex':
            if not (c.get('pattern') or '').strip():
                problems.append(f'{rubric_rel}: checks[{i}].pattern is empty')
        elif verify == 'shell':
            if not (c.get('command') or '').strip():
                problems.append(f'{rubric_rel}: checks[{i}].command is empty')
        elif verify == 'eq':
            if not (c.get('source') or '').strip():
                problems.append(f'{rubric_rel}: checks[{i}].source is empty')
            if 'expected' not in c:
                problems.append(f'{rubric_rel}: checks[{i}].expected is missing')
    return problems


def _validate_acceptable_tools(a: dict[str, Any], i: int, rubric_rel: str, assert_type: str) -> list[str]:
    if 'acceptable_tools' not in a:
        return []
    raw = a.get('acceptable_tools')
    if isinstance(raw, str):
        if raw.strip():
            return []
        return [f'{rubric_rel}: checks[{i}].assert[{assert_type}] acceptable_tools must not be empty']
    if not isinstance(raw, list):
        return [f'{rubric_rel}: checks[{i}].assert[{assert_type}] acceptable_tools must be a list of strings']
    if not raw or not all(isinstance(x, str) and x.strip() for x in raw):
        return [f'{rubric_rel}: checks[{i}].assert[{assert_type}] acceptable_tools must be a non-empty list of strings']
    return []


def _validate_count_bounds(a: dict[str, Any], i: int, rubric_rel: str, assert_type: str) -> list[str]:
    problems: list[str] = []
    parsed: dict[str, int] = {}
    for key in ('min_count', 'max_count'):
        if key not in a:
            continue
        raw = a[key]
        if isinstance(raw, bool):
            problems.append(f'{rubric_rel}: checks[{i}].assert[{assert_type}] {key} must be an integer')
            continue
        if isinstance(raw, int):
            value = raw
        elif isinstance(raw, str):
            try:
                value = int(raw)
            except ValueError:
                problems.append(f'{rubric_rel}: checks[{i}].assert[{assert_type}] {key} must be an integer')
                continue
        else:
            problems.append(f'{rubric_rel}: checks[{i}].assert[{assert_type}] {key} must be an integer')
            continue
        if value < 0:
            problems.append(f'{rubric_rel}: checks[{i}].assert[{assert_type}] {key} must be >= 0')
            continue
        parsed[key] = value
    if parsed.get('min_count', 0) > parsed.get('max_count', parsed.get('min_count', 0)):
        problems.append(
            f'{rubric_rel}: checks[{i}].assert[{assert_type}] '
            f'min_count={parsed["min_count"]} exceeds max_count={parsed["max_count"]}'
        )
    return problems


def cmd_validate(args: argparse.Namespace) -> int:
    _require_case(args.case_id)
    problems = _validate_case(args.case_id)
    if problems:
        print(f'case_id={args.case_id}: {len(problems)} problem(s)')
        for p in problems:
            print(f'  - {p}')
        return 1
    print(f'case_id={args.case_id}: OK')
    return 0


# ---------------------------------------------------------------------------
# 子命令： package
# ---------------------------------------------------------------------------


def cmd_package(args: argparse.Namespace) -> int:
    case_dir = _require_case(args.case_id)
    out = Path(args.out) if args.out else Path(f'{args.case_id.replace("/", "__")}.tgz')
    with tarfile.open(out, 'w:gz') as tar:
        tar.add(case_dir, arcname=args.case_id)
    size = out.stat().st_size
    human = (
        f'{size / 1024 / 1024:.1f} MB'
        if size >= 1024 * 1024
        else f'{size / 1024:.1f} KB'
        if size >= 1024
        else f'{size} B'
    )
    print(f'[aidata] packaged {args.case_id} → {out} ({human})', file=sys.stderr)
    print(out)
    return 0


# ---------------------------------------------------------------------------
# 分发
# ---------------------------------------------------------------------------


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog='aidata',
        description='Case-runner CLI for the aidata benchmark.',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            'Examples:\n'
            '  aidata run channel_inventory_ppt -v\n'
            '  aidata list\n'
            '  aidata validate channel_inventory_ppt\n'
            '  aidata package channel_inventory_ppt /tmp/channel.tgz\n\n'
            'Back-compat: `aidata <case_id>` is treated as `aidata run <case_id>`.'
        ),
    )
    sub = parser.add_subparsers(dest='subcommand')

    p_run = sub.add_parser('run', help='Run a case through nexus run-single.')
    p_run.add_argument(
        '--template',
        '-t',
        default=None,
        help='Runtime template (place BEFORE case_id): alias "litellm" (default) / "anthropic" '
        '(vertex_anthropic Bedrock passthrough) / "vertex" / "passthrough", or a path.',
    )
    p_run.add_argument('case_id')
    p_run.add_argument(
        'extra',
        nargs=argparse.REMAINDER,
        help='Extra args forwarded to `nexus run-single` (e.g. -v).',
    )
    p_run.set_defaults(func=cmd_run)

    p_list = sub.add_parser('list', help='List all cases with a short summary.')
    p_list.set_defaults(func=cmd_list)

    p_val = sub.add_parser('validate', help='Check case.yaml + rubric.yaml syntactically.')
    p_val.add_argument('case_id')
    p_val.set_defaults(func=cmd_validate)

    p_pack = sub.add_parser('package', help='Tar up cases/<id>/ as a distributable bundle.')
    p_pack.add_argument('case_id')
    p_pack.add_argument('out', nargs='?', default=None, help='Output path (default: ./<case_id>.tgz).')
    p_pack.set_defaults(func=cmd_package)

    return parser


def main(argv: list[str] | None = None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)

    # 向后兼容 shim ：如果第一个参数是裸的 case_id ，就把它当作
    # `run <case_id> <...rest>` 。 在 argparse 解析之前就要做这件事，
    # 否则 argparse 会拒绝这个未知的子命令。
    if (
        argv
        and argv[0] not in ('run', 'list', 'validate', 'package', 'help', '-h', '--help')
        and (CASES_DIR / argv[0]).is_dir()
    ):
        argv = ['run', *argv]

    if argv and argv[0] == 'help':
        _build_parser().print_help()
        return 0

    parser = _build_parser()
    ns = parser.parse_args(argv)
    if not hasattr(ns, 'func'):
        parser.print_help()
        return 2
    return int(ns.func(ns) or 0)


if __name__ == '__main__':
    sys.exit(main())
