"""``state_equals`` 原语 —— 分发 jmespath / state_sql: / sql: /
结构化切片 / 扁平点分路径分支。 ``_dotted_get`` 在函数体内延迟导入，
以打破 dispatch↔selectors 循环。
"""

import logging

from typing import Any

from ..config import JunitLookupContext, RubricLimits, VerifierResult
from ..jmespath_predicates import _check_jmespath_predicate, _jmespath_compile_cached
from ..sql import (
    _compare_sql_scalar,
    _resolve_sql_runtime,
    _rubric_sqlite_path,
    _snapshot_sql_scalar,
    _snapshot_state_sql_scalar,
)


_logger = logging.getLogger(__name__)


async def _eval_state_equals_jmespath(cfg: dict, state: dict) -> VerifierResult:
    """state_equals 的 JMESPath 分支 —— 查询切片 + 运行谓词。"""
    query = str(cfg.get('jmespath') or '').strip()
    if not query:
        return False, 'INVALID', 'state_equals(jmespath): empty query', ''
    slice_name = cfg.get('from')
    if slice_name:
        base = state.get(slice_name)
        if base is None:
            return False, 'MISSING', f'missing_slice: {slice_name!r}', ''
    else:
        base = state
    try:
        expr = _jmespath_compile_cached(query)
    except Exception as exc:  # noqa: BLE001 — 将编译错误转为 INVALID
        return False, 'INVALID', f'bad_jmespath: {exc}', ''
    try:
        result = expr.search(base)
    except Exception as exc:  # noqa: BLE001 — 将求值错误转为 INVALID
        return False, 'INVALID', f'jmespath_eval_failed: {exc}', ''
    return _check_jmespath_predicate(cfg, result)


async def _eval_state_equals(
    cfg: dict, state: dict, runtime: Any, junit_ctx: JunitLookupContext | None, *, limits: RubricLimits | None = None
) -> VerifierResult:
    """结构化相等比较。

    支持四种形态：

    1. **JMESPath 查询** —— ``jmespath: "orders[?status=='paid']"`` 加上九种
       谓词之一（``exists`` / ``equals`` / ``one_of`` / ``count_op``
       + ``count_value`` / ``contains`` / ``contains_all`` / ``contains_any``
       / ``regex_match`` / ``gt`` / ``gte`` / ``lt`` / ``lte`` ）。 ``from`` 可选 ——
       默认为整个 ``state`` 字典。 推荐用于验证一个写类工具确实变更了 mock 状态
       （相较于仅在轨迹上正则匹配工具名）。
    2. **切片 + 点分路径** （默认 aidata 风格）—— ``from: emails`` +
       ``path: 0.subject`` + ``equals`` / ``one_of`` 。
    3. **扁平点分路径** （v2 ``eq`` 迁移）—— 省略 ``from`` 并设置
       ``source: trajectory.0.role`` 以遍历整个 ``state`` 字典。
    4. **SQLite 标量探针** —— ``from: "sql:SELECT ..."`` （或在旧版 ``source:``
       键上使用同样格式）。 只读地针对引擎在 ``state['_rubric_sqlite_path']``
       中缓存的 sqlite 文件路径执行（见 ``engine.task.run_aidata`` ）。 SQL 必须
       返回单个标量单元（``0`` / ``1`` / 文本 / …），随后使用与 (2) 相同的
       强制转换规则通过 ``equals`` / ``one_of`` 进行比较。
    5. **State SQL 标量探针** —— ``from: "state_sql:<namespace>"`` 加上
       ``sql: "SELECT ..."`` 。 将 dump_state 的 ``state[namespace]['tables']``
       物化到内存 SQLite 数据库中，然后执行标量 SQL 探针。 当用例已从
       ``task.yaml::rubric_sqlite_path`` 迁移出去时，这可以让 rubric 保持易读。
       允许 ``state_sql:taobao,pdd`` 形式的多命名空间。 该形式下表名会加前缀，
       例如 ``taobao_orders`` 与 ``pdd_orders`` 。

    分发顺序： ``jmespath`` 键 → ``state_sql:`` 前缀 → ``sql:`` 前缀
    → 结构化切片 → 扁平点分路径。 ``jmespath`` 与 SQL 形状的探针会忽略 ``path`` 。
    """
    from ..dispatch import _dotted_get

    # --- jmespath 分支（最高优先级 —— 通过键显式选择进入）----
    if 'jmespath' in cfg:
        return await _eval_state_equals_jmespath(cfg, state)

    slice_name = cfg.get('from')
    path = cfg.get('path', '')
    src = str(slice_name if slice_name is not None else '').strip()
    if not src:
        src = str(cfg.get('source') or '').strip()

    # --- state_sql: 分支 -------------------------------------------------
    if src.startswith('state_sql:'):
        rest = src[len('state_sql:') :].strip()
        if not rest:
            return False, 'INVALID', 'state_equals(state_sql:): empty namespace', ''
        if ':' in rest and not str(cfg.get('sql') or '').strip():
            namespace, sql = rest.split(':', 1)
            namespace = namespace.strip()
            sql = sql.strip()
        else:
            namespace = rest
            sql = str(cfg.get('sql') or '').strip()
        if not namespace:
            return False, 'INVALID', 'state_equals(state_sql:): empty namespace', ''
        if not sql:
            return False, 'INVALID', 'state_equals(state_sql:): empty sql', ''
        if not sql.upper().startswith('SELECT'):
            return False, 'INVALID', 'state_equals(state_sql:): only SELECT is allowed', ''
        if path:
            _logger.warning(
                '[rubric.state_sql] ignoring non-empty path=%r for state_sql probe (from=%r)',
                path,
                src[:80],
            )
        val, err = _snapshot_state_sql_scalar(state, namespace, sql)
        if err is not None:
            return False, 'MISSING', f'{err} (namespace={namespace!r})', ''
        return _compare_sql_scalar(cfg, val, label=f'state_sql:{namespace} sql={sql[:120]!r}')

    # --- sql: 分支 -------------------------------------------------------
    if src.startswith('sql:'):
        sql = src[len('sql:') :].strip()
        if not sql:
            return False, 'INVALID', 'state_equals(sql:): empty query', ''
        if path:
            _logger.warning(
                '[rubric.sql] ignoring non-empty path=%r for sql: probe (from=%r)',
                path,
                src[:80],
            )

        db_path = _rubric_sqlite_path(state)
        if db_path is None:
            return (
                False,
                'MISSING',
                'state_equals(sql:) requires task.yaml::rubric_sqlite_path '
                '(engine seeds state[_rubric_sqlite_path]); none set for this case',
                '',
            )
        sql_runtime, route_err = await _resolve_sql_runtime(state, runtime)
        if route_err is not None:
            return False, 'MISSING', f'{route_err} (db_path={db_path!r})', ''
        val, err = await _snapshot_sql_scalar(db_path, sql, runtime=sql_runtime)
        if err is not None:
            return False, 'MISSING', f'{err} (db_path={db_path!r})', ''

        return _compare_sql_scalar(cfg, val, label=f'sql={sql[:120]!r}')

    # --- 结构化切片分支 -----------------------------------------
    if slice_name:
        base = state.get(slice_name)
        if base is None:
            return False, 'MISSING', f'missing_slice: {slice_name}', ''
        full_path = path
    else:
        full_path = cfg.get('source', '') or path
        base = state
    if not full_path:
        if slice_name is None:
            return False, 'INVALID', 'empty_path', ''
        got = base
        found = True
    else:
        found, got = _dotted_get(base, full_path)
    if not found:
        return False, 'MISSING', f'path_not_found: from={slice_name!r} path={full_path!r}', ''
    if 'one_of' in cfg:
        allowed = cfg.get('one_of')
        passed = got in allowed if isinstance(allowed, list) else False
        detail = f'got={got!r} one_of={allowed!r}'
    elif 'equals' in cfg:
        expected = cfg['equals']
        passed = got == expected
        detail = f'got={got!r} equals={expected!r}'
    elif 'expected' in cfg:  # 向后兼容别名
        expected = cfg['expected']
        passed = got == expected
        detail = f'got={got!r} expected={expected!r}'
    else:
        return False, 'INVALID', 'missing_equals_or_one_of', ''
    return passed, 'Y' if passed else 'N', detail, ''
