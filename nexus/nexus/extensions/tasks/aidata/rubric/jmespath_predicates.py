"""JMESPath 谓词分发 —— 用于 ``state_equals(jmespath:...)`` 以及被
``sql:`` / ``state_sql:`` 分支共享的标量比较辅助函数。
"""

import functools
import operator
import re

from collections.abc import Callable
from typing import Any

import jmespath

from .config import VerifierResult


# ---------------------------------------------------------------------------
# JMESPath 分发辅助（供 state_equals + jmespath 分支使用）
# ---------------------------------------------------------------------------
#
# state_equals 的 jmespath 分支允许 rubric 作者在 state 切片（或整个 state
# 字典）上运行任意 JMESPath 查询，然后通过九种可插拔谓词之一决定通过/失败。
# 这是"看看 agent 实际改动了什么"的路径，供不希望依赖轨迹正则匹配
# （"工具名是否出现在 thought 文本中"）或 judge_claim （"让 LLM 阅读 state JSON"）
# 的验证器使用。
#
# 为什么用 JMESPath ： BSD 许可、单一作者维护、内置聚合函数
# （``length`` / ``sum`` / ``min`` / ``max`` / ``avg`` ），因此一行查询就能
# 表达"已支付订单数量"或"今日已退款总额"，无需谓词方面的额外杂技。
# AWS CLI 使用同一语言，所以团队一半人已经掌握了语法。
#
# 谓词键名（恰好指定一个；若为零或 >1 则为 ``INVALID`` ）：
#
#   exists         bool         真值测试（None / [] / {} / "" / 0 → False）
#   equals         any          标量 == 值（带 int/float 强制转换）
#   one_of         list         标量 ∈ 列表（带 int/float 强制转换）
#   count_op +     str + int    op(len(result), count_value)； op ∈
#   count_value                 {==, !=, >, >=, <, <=}
#   contains       any          元素 ∈ 列表  /  子串 ∈ 字符串
#   contains_all   list         list 中每个元素 ∈ result
#   contains_any   list         list 中至少一个元素 ∈ result
#   regex_match    str          re.search(pattern, str(result)) 命中
#   gt / gte /     number       result <op> value （数值比较）
#   lt / lte


_COUNT_OPS: dict[str, Callable[[Any, Any], bool]] = {
    '==': operator.eq,
    '!=': operator.ne,
    '>': operator.gt,
    '>=': operator.ge,
    '<': operator.lt,
    '<=': operator.le,
}


# 消费查询结果的谓词名称。 由 ``_validate_check`` 与 ``_eval_state_equals_jmespath``
# 内部的多谓词守卫共享，以便新增谓词时两处保持同步。
_JMESPATH_PREDICATE_KEYS: tuple[str, ...] = (
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


@functools.lru_cache(maxsize=256)
def _jmespath_compile_cached(query: str) -> Any:
    """编译并缓存一个 JMESPath 表达式。 ``query`` 通常很小，且在同一 rubric
    对多个 rollout 求值时会反复复用；缓存让每次检查的开销 ≈ 免费。
    """
    return jmespath.compile(query)


def _coerce_numeric_equal(a: Any, b: Any) -> bool:
    """带 int/float 容差的相等比较 —— 与 ``_eval_state_equals`` 的 sql: 分支
    保持镜像一致，让用户在两处得到相同的、无意外的数值比较语义。
    """
    try:
        return float(a) == float(b)
    except (TypeError, ValueError):
        return str(a) == str(b)


def _is_truthy_jmespath_result(val: Any) -> bool:
    """JMESPath 的 ``exists`` 语义 —— 显式实现，因为 Python 的 ``bool()`` 会
    把 ``0`` 和 ``False`` 视为假值，而这里正是我们想要的（返回 ``0`` "肯定
    没有匹配"的聚合查询应当让 ``exists: true`` 断言失败），但值得明确写下
    以便文档字符串与代码保持一致。
    """
    if val is None:
        return False
    if isinstance(val, (list, dict, str)):
        return len(val) > 0
    return bool(val)


def _check_jmespath_predicate(cfg: dict, result: Any) -> VerifierResult:
    """将已配置的谓词分发到 JMESPath 查询结果上。

    调用方已完成查询编译 + 执行；此函数是纯数据操作 →
    ``(passed, verdict, detail, raw_head)`` 。
    """
    present = [k for k in _JMESPATH_PREDICATE_KEYS if k in cfg]
    if 'count_op' in cfg or 'count_value' in cfg:
        # 两个 count_* 键是配对的；把它们合并成一个"谓词槽位"以便通过
        # 多谓词守卫，但要求两者都存在。
        present = [k for k in present if k != 'count_value']
        if 'count_op' in cfg and 'count_value' not in cfg:
            return False, 'INVALID', 'count_op requires count_value', ''
        if 'count_value' in cfg and 'count_op' not in cfg:
            return False, 'INVALID', 'count_value requires count_op', ''
    if len(present) > 1:
        return (
            False,
            'INVALID',
            f'multiple predicates specified: {present!r} (specify exactly one)',
            '',
        )
    if not present:
        return (
            False,
            'INVALID',
            f'missing predicate: specify one of {list(_JMESPATH_PREDICATE_KEYS)!r}',
            '',
        )

    head = repr(result)[:200]

    if 'exists' in cfg:
        want = bool(cfg['exists'])
        got = _is_truthy_jmespath_result(result)
        passed = got == want
        return passed, 'Y' if passed else 'N', f'exists={got} want={want}', head

    if 'equals' in cfg:
        expected = cfg['equals']
        passed = result == expected or _coerce_numeric_equal(result, expected)
        return passed, 'Y' if passed else 'N', f'got={result!r} equals={expected!r}', head

    if 'one_of' in cfg:
        allowed = cfg.get('one_of')
        if not isinstance(allowed, list):
            return False, 'INVALID', 'one_of must be a list', ''
        passed = any(result == c or _coerce_numeric_equal(result, c) for c in allowed)
        return passed, 'Y' if passed else 'N', f'got={result!r} one_of={allowed!r}', head

    if 'count_op' in cfg:
        op_name = str(cfg['count_op'])
        op = _COUNT_OPS.get(op_name)
        if op is None:
            return False, 'INVALID', f'unknown count_op {op_name!r} (one of {list(_COUNT_OPS)!r})', ''
        try:
            count = len(result) if result is not None else 0
        except TypeError:
            return False, 'INVALID', f'count_op needs a sized result; got {type(result).__name__}', ''
        try:
            target = int(cfg['count_value'])
        except (TypeError, ValueError):
            return False, 'INVALID', f'count_value must be an int; got {cfg["count_value"]!r}', ''
        passed = op(count, target)
        return passed, 'Y' if passed else 'N', f'count={count} {op_name} {target}', head

    if 'contains' in cfg:
        needle = cfg['contains']
        if isinstance(result, str):
            passed = str(needle) in result
        elif isinstance(result, (list, tuple, dict)):
            passed = needle in result
        else:
            return False, 'INVALID', f'contains needs list/dict/str; got {type(result).__name__}', ''
        return passed, 'Y' if passed else 'N', f'needle={needle!r} in result? {passed}', head

    if 'contains_all' in cfg:
        needles = cfg['contains_all']
        if not isinstance(needles, list):
            return False, 'INVALID', 'contains_all must be a list', ''
        if isinstance(result, str):
            missing = [n for n in needles if str(n) not in result]
        elif isinstance(result, (list, tuple, dict)):
            missing = [n for n in needles if n not in result]
        else:
            return False, 'INVALID', f'contains_all needs list/dict/str; got {type(result).__name__}', ''
        passed = not missing
        return passed, 'Y' if passed else 'N', f'missing={missing!r}' if missing else f'all_found={needles!r}', head

    if 'contains_any' in cfg:
        needles = cfg['contains_any']
        if not isinstance(needles, list):
            return False, 'INVALID', 'contains_any must be a list', ''
        if isinstance(result, str):
            found = [n for n in needles if str(n) in result]
        elif isinstance(result, (list, tuple, dict)):
            found = [n for n in needles if n in result]
        else:
            return False, 'INVALID', f'contains_any needs list/dict/str; got {type(result).__name__}', ''
        passed = bool(found)
        return passed, 'Y' if passed else 'N', f'found={found!r} any_of={needles!r}', head

    if 'regex_match' in cfg:
        pattern = str(cfg['regex_match'])
        try:
            matched = re.search(pattern, str(result)) is not None
        except re.error as exc:
            return False, 'INVALID', f'bad_regex: {exc}', ''
        return matched, 'Y' if matched else 'N', f'regex={pattern!r} matched={matched}', head

    # 数值比较符（gt / gte / lt / lte ）。 放在集合形状谓词之后，
    # 因为它们是唯一需要查询结果本身可以强制转换成数字的谓词。
    for key, op_name in (('gt', '>'), ('gte', '>='), ('lt', '<'), ('lte', '<=')):
        if key not in cfg:
            continue
        try:
            got_num = float(result)
            want_num = float(cfg[key])
        except (TypeError, ValueError):
            return False, 'INVALID', f'{key}: both result and value must be numeric (got {result!r}, {cfg[key]!r})', ''
        op = _COUNT_OPS[op_name]
        passed = op(got_num, want_num)
        return passed, 'Y' if passed else 'N', f'got={got_num} {op_name} {want_num}', head

    # 不可达 —— 多谓词守卫已捕获了空的情况。
    return False, 'INVALID', 'unhandled predicate', ''
