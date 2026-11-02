"""稳定的 ``check.id`` → junit ``testcase`` 名称映射。

aidata 的 v4.2 rubric 层始终只处理稳定的 ``check.id`` 值（例如
``pr_merge_order``）。 而底层的 junit 报告使用的却是各案例自身测试框架
所产生的 testcase 名称（``test_pr_merge_order``、 ``TestPRMergeOrder``
或自由格式的字符串）。

本模块负责在两者之间搭桥。 标注人员永远看不到原始名称；只有案例作者
（开发者）会看到，并且只在其框架挑选了非常规名称时才会看到。

解析顺序（每个 check）：

1. 若 ``tests/check_map.yaml`` 存在且包含 ``check.id`` 的条目，则使用其
   ``junit_name``。
2. 否则从 ``check.id`` 合成候选名称，让 junit 报告选取首个匹配：
     - ``test_<check_id>``      （pytest 约定）
     - ``Test<PascalCase(check_id)>``  （Go 约定）
     - ``<check_id>``           （字面回退）

若两者都无法匹配，会记录一条醒目的警告日志，让案例作者从一行日志中
就能看到哪个 id 无法匹配以及有哪些可用名称。
"""

from __future__ import annotations

import logging
import re

from pathlib import Path
from typing import Any

from .junit import JunitReport, TestCaseResult


logger = logging.getLogger(__name__)


CHECK_MAP_FILENAME = 'check_map.yaml'


def load_check_map(tests_dir: Path) -> dict[str, str]:
    """将 ``tests/check_map.yaml`` 加载为 ``{check_id: junit_name}``。

    文件缺失或为空时容忍：返回空字典。 YAML 格式错误则以 ``ValueError``
    抛出——案例作者必须在评分前修复。
    """
    path = tests_dir / CHECK_MAP_FILENAME
    if not path.exists():
        return {}
    import yaml  # 局部 import；PyYAML 已是 case 级依赖

    try:
        data = yaml.safe_load(path.read_text(encoding='utf-8')) or {}
    except Exception as exc:
        raise ValueError(f'cannot parse {path}: {exc}') from exc
    if not isinstance(data, dict):
        raise ValueError(f'{path}: must be a YAML mapping (got {type(data).__name__})')

    mapping: dict[str, str] = {}
    for check_id, entry in data.items():
        if isinstance(entry, str):
            mapping[str(check_id)] = entry
            continue
        if not isinstance(entry, dict):
            raise ValueError(
                f'{path}: entry for check_id={check_id!r} must be a string (junit_name) '
                f'or a mapping with key ``junit_name``; got {type(entry).__name__}'
            )
        junit_name = entry.get('junit_name')
        if not isinstance(junit_name, str) or not junit_name:
            raise ValueError(
                f'{path}: entry for check_id={check_id!r} must provide a non-empty ``junit_name``; got {entry!r}'
            )
        mapping[str(check_id)] = junit_name
    return mapping


def _pascal_case(check_id: str) -> str:
    """``pr_merge_order`` → ``PrMergeOrder``（Go 风格测试函数名）。

    非字母数字连续段视为词边界；空段丢弃。
    """
    words = re.split(r'[^A-Za-z0-9]+', check_id)
    return ''.join(w[:1].upper() + w[1:] for w in words if w)


def derive_candidates(check_id: str) -> list[str]:
    """按顺序返回给定 ``check.id`` 需要探测的 junit testcase 名称。"""
    return [
        f'test_{check_id}',
        f'Test{_pascal_case(check_id)}',
        check_id,
    ]


def resolve_testcase(
    check_id: str,
    report: JunitReport,
    explicit_map: dict[str, str] | None = None,
) -> TestCaseResult | None:
    """返回与 ``check_id`` 匹配的 :class:`TestCaseResult`，无匹配则为 None。

    ``explicit_map``（来自 ``check_map.yaml``）优先于启发式候选，但仍通过
    :meth:`JunitReport.find` 查找，因此映射文件中的拼写错误会体现为一条
    干净的 MISSING 记录，而不是崩溃。
    """
    explicit = (explicit_map or {}).get(check_id)
    candidates: list[str] = []
    if explicit:
        candidates.append(explicit)
    candidates.extend(derive_candidates(check_id))
    return report.find(*candidates)


def summarise_mapping(
    check_ids: list[str],
    report: JunitReport,
    explicit_map: dict[str, str] | None = None,
) -> dict[str, dict[str, Any]]:
    """一次性对整份 rubric 生成便于调试的摘要。

    返回 ``{check_id: {"matched_name": str | None, "passed": bool | None,
    "candidates": [...]}}``。 供 rubric 日志与 ``aidata.py validate``
    使用，指出首个未能解析的 check 供审阅者定位。
    """
    out: dict[str, dict[str, Any]] = {}
    for cid in check_ids:
        cands = derive_candidates(cid)
        explicit = (explicit_map or {}).get(cid)
        tc = resolve_testcase(cid, report, explicit_map=explicit_map)
        out[cid] = {
            'matched_name': tc.name if tc else None,
            'passed': tc.passed if tc else None,
            'explicit': explicit,
            'candidates': cands,
        }
    return out


__all__ = [
    'CHECK_MAP_FILENAME',
    'derive_candidates',
    'load_check_map',
    'resolve_testcase',
    'summarise_mapping',
]
