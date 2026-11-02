#!/usr/bin/env python3
"""供 ``nexus run-batch`` 使用的 aidata 批处理生成器。

对 ``cases/`` 下每个可发现的 case 生成一个 :class:`BatchItem` ，
以便现有的 ``@task_registry.register('aidata')`` 任务（见 ``engine/task.py`` ）
无需任务侧改动即可在标准 nexus 批处理运行器中运行——
``nexus.runner.batch.runner.run_batch`` 最终会为每个 item 调用 ``run_single`` ，
所以 ``aidata.py run`` 目前所支持的一切（ ``services`` /
``extra_mcp_servers`` / 每个 case 的 ``runtime.main.image`` 覆盖 / case
rubric / 基于 junit 的检查）在
``nexus run-batch -g this/file -d <db>`` 下继续可用，并额外获得
并发、恢复、重试、SQLite 轨迹+日志捕获、 ``inspect-batch``
TUI 和结构化导出。

用法::

    cd <repo-root>           # /data/workspace/eval/nexus
    uv run nexus run-batch \\
        -g nexus/extensions/tasks/aidata/generator.py \\
        -d /tmp/aidata_batch.sqlite \\
        --max-concurrent 4 \\
        --retry-failed 1 \\
        --export-format jsonl

选择项（全部可选，通过环境变量驱动，同一生成器脚本可
沿用标准 ``-g <file>`` 调用而无需每次运行时的额外参数）::

    AIDATA_CASE_IDS=diabetes_followup,online_shopping/scenario_add_cart \\
        uv run nexus run-batch ...

    AIDATA_SKIP_PENDING_UPSTREAM=0   # 默认 1 ——丢弃声明了
                                     # ``pending_upstream:`` 的 case （对齐
                                     # 动态测试层的 ``pytest.skip`` ）。
                                     # 设为 0 强制包含它们，例如
                                     # 用于验证新烘焙的组合镜像
                                     # 是否真的解锁了这些 case 。

    AIDATA_CONFLICT_GROUPS=gongfeng  # 逗号分隔的互斥分组名，
                                     # 会盖到每个 BatchItem 的
                                     # metadata.conflict_groups 上。 仅在
                                     # 运行以 ``BatchConfig.enable_conflict_groups
                                     # = True`` 启动时生效（目前是
                                     # Python API ； CLI 暴露待完成）。

设计说明——为何不为每个 case 子进程调用 ``aidata.py run`` ：

- 批处理的价值（ db 支撑的恢复 / 重试 / 轨迹双写 / 共享
  日志捕获）只有当 items 与批处理运行器在同一进程内经过 ``run_single``
  时才会启用。 每个 case fork ``aidata.py`` 会让每次运行都重新支付
  gongfeng 认证 + 模块导入成本，同时丢失
  所有批处理级别的埋点钩子。
- ``RunInput.from_dict`` 与 ``nexus run-single`` 从磁盘读取渲染后
  yaml 时执行的 OmegaConf 插值行为一致
  （见 ``nexus.runner.single.RunInput.from_dict`` ），因此在
  相同环境变量下，每个 case 的行为与 ``aidata.py run <case_id>``
  逐字节一致。
"""

from __future__ import annotations

import logging
import os

from collections.abc import Iterator
from pathlib import Path
from typing import Any

import yaml

from nexus import BatchItem, RunInput

# CASES_DIR 来自 ``engine.paths`` ，让生成器与
# ``run_aidata`` 任务在遍历哪棵树上取得一致； ``AIDATA_CASES_ROOT``
# 会同时覆盖两者。 回退到 ``<package>/cases`` 用于
# 仓库内开发（该 PR 有意不携带 case 数据，因此
# 回退目录为空，直到运维将该环境变量指向下游
# Tasks 仓库）。
from nexus.extensions.tasks.aidata.engine.paths import CASES_DIR


logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# 布局
# ---------------------------------------------------------------------------


SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parents[3]
TEMPLATE = SCRIPT_DIR / 'runtime.template.yaml'


# ---------------------------------------------------------------------------
# .env 传播
#
# ``nexus run-batch`` 是普通进程，不是 ``aidata.py`` 包装器，所以
# 它不会像 ``aidata.py::_load_env_into`` 为 ``run-single`` 所做的那样
# 自动把 ``.env`` 合并进 ``os.environ`` 。 若无此逻辑，每个 case 的
# ``${oc.env:GONGFENG_TOKEN}`` 插值都会解析为 '' ，导致
# 第一次 ``runtime.acquire`` 就会因 gongfeng 抛出令人困惑的
# 'token missing' 错误。
#
# 查找顺序对齐 ``tests/conftest.py`` ：位于（下游） cases 目录旁边的
# ``.env`` 优先于上游 ``<repo>/.env`` ，这在同一 nexus 检出与多个下游
# Tasks 树复用时非常有用。 使用 ``setdefault`` 语义，让在 CLI 上
# 显式导出的变量优先于两个文件。
# ---------------------------------------------------------------------------


def _candidate_env_files() -> list[Path]:
    candidates: list[Path] = []
    cases_env = CASES_DIR.parent / '.env'
    if cases_env.is_file():
        candidates.append(cases_env)
    repo_env = REPO_ROOT / '.env'
    if repo_env.is_file() and repo_env not in candidates:
        candidates.append(repo_env)
    return candidates


def _load_repo_env_once() -> None:
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
            os.environ.setdefault(key, value)
        logger.debug('aidata generator: merged %s into os.environ', env_file)


_load_repo_env_once()


# ---------------------------------------------------------------------------
# case 发现 + 过滤
# ---------------------------------------------------------------------------


def _manifest_file(case_dir: Path) -> Path | None:
    """v4.2 的 ``task.yaml`` 优先于遗留的 ``case.yaml`` ；当目录不是 case
    （无 manifest）时返回 None 。"""
    for name in ('task.yaml', 'case.yaml'):
        p = case_dir / name
        if p.exists():
            return p
    return None


def _discover_cases() -> list[str]:
    """遍历 ``cases/`` ，返回每个含 manifest 的 case_id 。

    对齐 ``aidata.py::_available_cases`` ，让静态列表与 CLI
    暴露的一致。 case_id 可能包含 ``/`` 用于
    表示嵌套（例如 ``online_shopping/scenario_add_cart`` ——
    批处理运行器的 ``generate_run_id`` 仅在 SQLite ``run_id``
    上把斜杠净化为 ``_`` ；喂给 ``task_config`` 的 case_id
    保留斜杠，因为 :mod:`engine.case_spec` 会把它当作
    ``CASES_DIR`` 下的路径使用）。
    """
    if not CASES_DIR.is_dir():
        return []
    found: list[str] = []
    for root, dirs, _files in os.walk(CASES_DIR):
        # 与 aidata.py 保持一致：跳过隐藏 / 下划线前缀的分组。
        dirs[:] = [d for d in dirs if not d.startswith(('.', '_'))]
        rp = Path(root)
        if _manifest_file(rp) is not None:
            found.append(rp.relative_to(CASES_DIR).as_posix())
    return sorted(found)


def _env_bool(name: str, default: bool) -> bool:
    raw = os.getenv(name)
    if raw is None:
        return default
    return raw.strip().lower() in ('1', 'true', 'yes', 'on')


def _env_list(name: str) -> list[str]:
    raw = os.getenv(name) or ''
    return [s.strip() for s in raw.split(',') if s.strip()]


def _is_pending_upstream(manifest: dict[str, Any]) -> bool:
    """case 是否声明了非空的 ``pending_upstream:`` 标记。

    当 case 因结构性原因在外部依赖落地前无法通过（组合
    docker 镜像烘焙、未发布的 MockServer tag ……）时，会设置
    此字段。 动态测试层
    （ ``tests/test_cases_startup.py`` ）会对它们执行
    ``pytest.skip`` ；此生成器默认镜像同样的跳过，
    以免在完整基准的 fan-out 中把沙箱配额耗在
    保证会失败的 case 上。 若确实想在批处理中包含它们
    （例如上游刚发货、你想确认 case 通过），
    可翻转 ``AIDATA_SKIP_PENDING_UPSTREAM=0`` 。
    """
    raw = manifest.get('pending_upstream')
    return isinstance(raw, str) and bool(raw.strip())


def _load_manifest(case_id: str) -> dict[str, Any] | None:
    """将 ``cases/<id>/(task.yaml|case.yaml)`` 读为普通 dict 。

    读取 / 解析失败返回 None （已记录日志）。 我们在此容忍失败，
    避免单个畸形 case 污染整套 suite 上 ``--max-concurrent``
    的运行—— ``aidata.py validate`` 才是真正强制 manifest 卫生
    的地方；此生成器只需窥视 metadata + ``pending_upstream`` 。
    """
    case_dir = CASES_DIR / case_id
    f = _manifest_file(case_dir)
    if f is None:
        return None
    try:
        data = yaml.safe_load(f.read_text(encoding='utf-8')) or {}
    except Exception as exc:
        logger.error('aidata generator: %s parse failed: %s', f, exc)
        return None
    if not isinstance(data, dict):
        logger.error('aidata generator: %s is not a YAML mapping', f)
        return None
    return data


def _selected_cases() -> list[tuple[str, dict[str, Any]]]:
    """应用 ``AIDATA_CASE_IDS`` 允许列表 + ``AIDATA_SKIP_PENDING_UPSTREAM``
    过滤（默认：跳过）。

    返回 ``(case_id, parsed_manifest)`` 对，让调用方无需
    重复读取每个 manifest 两次（此处选择、
    :func:`generate_configs` 中打 metadata ）。
    """
    all_ids = _discover_cases()
    explicit = set(_env_list('AIDATA_CASE_IDS'))
    if explicit:
        unknown = sorted(explicit - set(all_ids))
        if unknown:
            logger.warning(
                'aidata generator: %d AIDATA_CASE_IDS entries not found under %s: %s',
                len(unknown),
                CASES_DIR,
                ', '.join(unknown),
            )
        ordered = [cid for cid in all_ids if cid in explicit]
    else:
        ordered = list(all_ids)

    skip_pending = _env_bool('AIDATA_SKIP_PENDING_UPSTREAM', default=True)
    out: list[tuple[str, dict[str, Any]]] = []
    skipped_pending: list[str] = []
    for case_id in ordered:
        manifest = _load_manifest(case_id)
        if manifest is None:
            continue
        if skip_pending and _is_pending_upstream(manifest):
            skipped_pending.append(case_id)
            continue
        out.append((case_id, manifest))

    if skipped_pending:
        logger.debug(
            'aidata generator: skipped %d pending_upstream case(s): %s '
            '(set AIDATA_SKIP_PENDING_UPSTREAM=0 to include them)',
            len(skipped_pending),
            ', '.join(skipped_pending),
        )
    return out


# ---------------------------------------------------------------------------
# RunInput 构造
# ---------------------------------------------------------------------------


_TEMPLATE_CACHE: dict[str, Any] | None = None


def _load_template() -> dict[str, Any]:
    """仅解析 ``runtime.template.yaml`` 一次。

    模板携带 ``task_config.case_id: "{{CASE_ID}}"`` 作为文本
    占位符，供遗留的 ``aidata.py run`` 代码路径消费。 我们
    在下面为每个 case 整体覆盖 ``task_config`` ，因此
    ``{{CASE_ID}}`` 字面量永远不会传到 ``RunInput.from_dict`` （虽然它
    能容忍，但会让审计日志更嘈杂）。
    """
    global _TEMPLATE_CACHE
    if _TEMPLATE_CACHE is None:
        loaded = yaml.safe_load(TEMPLATE.read_text(encoding='utf-8')) or {}
        if not isinstance(loaded, dict):
            raise ValueError(f'{TEMPLATE}: must be a YAML mapping')
        _TEMPLATE_CACHE = loaded
    return _TEMPLATE_CACHE


def _build_run_input(case_id: str) -> RunInput:
    """组装每个 case 的 :class:`RunInput` 。

    策略：浅克隆缓存的模板并覆盖唯一的
    与 case 相关的顶层键（ ``task_config`` ）。 其他所有
    段（ imports 、 runtime_provider_* 、 runtime_configs 、 agent_* 、
    openai_clients ）均逐字来自 ``runtime.template.yaml`` ，因此
    在那里的编辑会以与 ``aidata.py run`` 相同的方式流入
    批处理运行。

    ``RunInput.from_dict`` 在构造时以 ``os.environ`` 解析
    ``${oc.env:…}`` 插值，与 ``run_single`` 中的
    ``RunInput.from_file`` 完全一致。 此处急切执行，以便
    任何缺失的环境变量在阶段 1 （配置生成）
    浮现，而非在运行中途。
    """
    config = dict(_load_template())  # 顶层浅拷贝
    config['task_config'] = {'case_id': case_id}
    return RunInput.from_dict(config)


def _build_metadata(case_id: str, manifest: dict[str, Any]) -> dict[str, Any]:
    """打到 BatchItem 上、方便审计的字段。

    在 ``inspect-batch`` 与下游导出中显现，让 reviewer
    可按 tag / pending 状态透视，而无需从磁盘
    重新加载 case manifest 。
    """
    md_raw = manifest.get('metadata')
    md = md_raw if isinstance(md_raw, dict) else {}
    metadata: dict[str, Any] = {
        'case_id': case_id,
        'task_id': manifest.get('task_id'),
        'tags': list(md.get('tags') or []),
        'pending_upstream': (manifest.get('pending_upstream') or '').strip() or None,
    }
    groups = _env_list('AIDATA_CONFLICT_GROUPS')
    if groups:
        metadata['conflict_groups'] = groups
    return metadata


# ---------------------------------------------------------------------------
# 供 ``GeneratorLoader`` 消费的公共生成器 API 。
# ---------------------------------------------------------------------------


def generate_configs() -> Iterator[BatchItem]:
    pairs = _selected_cases()
    logger.debug('aidata generator: yielding %d case(s)', len(pairs))
    for case_id, manifest in pairs:
        try:
            run_input = _build_run_input(case_id)
        except Exception as exc:
            # 模板损坏 / 环境变量缺失应视为硬失败
            # （避免批处理静默产出不完整的配置集），
            # 但单个 case 后段抛异常不应杀掉整个生成过程。
            # 记录日志并跳过。
            logger.error(
                'aidata generator: skipping %s — RunInput build failed: %s',
                case_id,
                exc,
            )
            continue
        yield BatchItem(
            config=run_input,
            run_id=case_id,
            metadata=_build_metadata(case_id, manifest),
        )


def get_total_count() -> int:
    return len(_selected_cases())


def get_batch_metadata() -> dict[str, Any]:
    pairs = _selected_cases()
    return {
        'task_type': 'aidata',
        'cases_dir': str(CASES_DIR),
        'selected_count': len(pairs),
        'selected_case_ids': [cid for cid, _ in pairs],
        'skip_pending_upstream': _env_bool('AIDATA_SKIP_PENDING_UPSTREAM', default=True),
        'conflict_groups': _env_list('AIDATA_CONFLICT_GROUPS'),
    }
