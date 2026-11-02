"""SQL 探针层，用于 ``state_equals(sql:...)`` 与
``state_equals(state_sql:...)`` 分支。

实现了两条执行路径：

- 通过沙箱运行时（或宿主回退）操作实时 sqlite DB —— ``_snapshot_sql_scalar``。
- 在内存中物化 dump_state 命名空间 —— ``_snapshot_state_sql_scalar``。

两者均返回 ``(value, error_detail)``，并共享 ``_compare_sql_scalar`` 用于
one_of / equals / expected 谓词分发。
"""

import json
import logging
import shlex
import sqlite3

from typing import Any

from .config import VerifierResult
from .jmespath_predicates import _coerce_numeric_equal


_logger = logging.getLogger(__name__)


def _rubric_sqlite_path(state: dict) -> str | None:
    """返回 ``state_equals(sql:...)`` 声明的 sqlite 路径，否则返回 None。

    引擎契约：``engine.task.run_aidata`` 会在替换 ``__CASE_UUID__`` 与服务
    占位符后，从 ``task.yaml::rubric_sqlite_path`` 中植入
    ``state['_rubric_sqlite_path']``。 rubric 层与业务无关 —— 它不会嗅探业务
    专属的环境变量键（如 ``ONLINE_SHOPPING_DB`` 等）；需要 sqlite 探针的用例
    必须在其清单中声明该路径。
    """
    v = state.get('_rubric_sqlite_path')
    if isinstance(v, str) and v.strip():
        return v.strip()
    return None


async def _resolve_sql_runtime(
    state: dict,
    default_runtime: Any | None,
) -> tuple[Any, str | None]:
    """决定由哪个运行时来执行 ``state_equals(sql:...)`` 探针。

    引擎契约：当 sqlite 预言机位于 sidecar 服务容器中时，
    ``engine.task.run_aidata`` 会（从
    ``task.yaml::rubric_sqlite_path: service:<name>:<path>``）植入
    ``state['_rubric_sqlite_service']``。 此时探针必须通过
    ``require_context().get_runtime(<name>)`` 在 **该 sidecar 内部** 执行 ——
    main 无法访问 sidecar 的文件系统。

    返回 ``(runtime, error_detail)``：

    * ``error_detail is None`` → 调用方使用 ``runtime`` 继续执行（对遗留的
      main 运行时路径来说可能是 ``default_runtime``，对跨容器路径来说则是
      sidecar 运行时）。
    * ``error_detail is not None`` → 调用方将其作为 ``MISSING`` 判定上抛；
      用例作者配置了错误的路由，或者 ExecutionContext 不可用（例如未进入
      ``async with ExecutionContext`` 的宿主端单元测试）。
    """
    service_name = state.get('_rubric_sqlite_service')
    if not isinstance(service_name, str) or not service_name.strip():
        return default_runtime, None
    service_name = service_name.strip()
    try:
        from nexus.core import require_context

        ctx = require_context()
    except Exception as exc:  # noqa: BLE001 —— 下方作为 MISSING 上抛
        return None, f'no_execution_context: {exc}'
    acquired = getattr(ctx, '_runtimes', {}) or {}
    if service_name not in acquired:
        return None, (f'service_runtime_not_acquired: {service_name!r}; available={sorted(acquired)!r}')
    return acquired[service_name], None


_SQL_PROBE_SNIPPET = r"""
import base64, json, os, sqlite3, sys
db = os.environ["NEXUS_RUBRIC_DB"]
sql = base64.b64decode(os.environ["NEXUS_RUBRIC_SQL_B64"]).decode("utf-8")
try:
    with sqlite3.connect(f"file:{db}?mode=ro", uri=True) as c:
        row = c.execute(sql).fetchone()
except sqlite3.Error as exc:
    sys.stdout.write(json.dumps({"error": f"sqlite_error: {exc}"}))
    sys.exit(0)
if row is None:
    sys.stdout.write(json.dumps({"row": None}))
else:
    # row[0] may be int/float/str/bytes/None; JSON-encode via default=str for
    # bytes/Decimal-like oddities so the host side always gets parseable text.
    sys.stdout.write(json.dumps({"row": [row[0]]}, default=str, ensure_ascii=False))
""".strip()


async def _snapshot_sql_scalar(
    db_path: str,
    sql: str,
    *,
    runtime: Any | None = None,
) -> tuple[Any, str | None]:
    """针对 ``db_path`` 只读运行 ``sql``；返回 (value, error_detail)。

    仅返回结果集第一行第一列的值。

    两条执行路径：

    * 提供了 ``runtime`` → 通过 ``runtime.run_command`` 在沙箱 **内部**
      运行一段极小的 ``python3 -c`` 片段，并解析其 stdout。 这是生产路径：
      rubric 声明的 ``rubric_sqlite_path`` 是一个 **沙箱内** 路径（例如
      ``/tmp/mock-run/<uuid>/online_shopping.db``），只有沙箱文件系统能打开
      它；同时避免了每次探针都把整个数据库文件（通常若干 MB）传回宿主。

      运行时既可以是 agent 的 ``main``（遗留：将其数据库写在 agent 容器内
      的 stdio mocks），也可以是 sidecar ``service_*`` 运行时（sidecar
      mock-server 容器 —— 上游的路由由 :func:`_resolve_sql_runtime` 通过读取
      ``state['_rubric_sqlite_service']`` 决定）。
    * ``runtime`` 为 None → 回退到宿主上的 ``sqlite3.connect``。 用于处理
      普通文件的单元测试和本地开发循环。

    任何 sqlite 错误（或运行时失败）都会被吞入 ``error_detail``，从而让
    rubric 行呈现可检查的 ``detail``，而不是让整个 rollout 抛异常。
    """
    if runtime is not None:
        return await _snapshot_sql_scalar_via_runtime(runtime, db_path, sql)

    uri = f'file:{db_path}?mode=ro'
    try:
        with sqlite3.connect(uri, uri=True) as conn:
            row = conn.execute(sql).fetchone()
    except sqlite3.Error as exc:
        _logger.warning('[rubric.sql] sqlite error on %r: %s', db_path, exc)
        return None, f'sqlite_error: {exc}'
    if row is None:
        return None, 'sql_returned_no_rows'
    return row[0], None


async def _snapshot_sql_scalar_via_runtime(
    runtime: Any,
    db_path: str,
    sql: str,
) -> tuple[Any, str | None]:
    """通过 ``runtime.run_command`` 在沙箱内部执行 SQL 探针。

    我们通过 ``extra_env`` 传入 ``db_path`` 与 base64 包装的 ``sql``，让调用方
    的 SQL 无需承受两层 shell 转义（online_shopping_mcp 的框架在处理含引号、
    换行和中文字面量的 SQL 时曾吃过大亏）。 该片段会打印一行 JSON，描述结果
    的标量行或错误；python3 的任何非零退出码都会被包装成 ``error_detail``。
    """
    import base64

    sql_b64 = base64.b64encode(sql.encode('utf-8')).decode('ascii')
    cmd = f'python3 -c {shlex.quote(_SQL_PROBE_SNIPPET)}'
    try:
        res = await runtime.run_command(
            command=cmd,
            timeout=30,
            extra_env={
                'NEXUS_RUBRIC_DB': db_path,
                'NEXUS_RUBRIC_SQL_B64': sql_b64,
            },
        )
    except Exception as exc:
        _logger.warning('[rubric.sql] run_command failed on %r: %s', db_path, exc)
        return None, f'run_command_failed: {exc}'

    exit_code = getattr(res, 'return_code', None)
    if exit_code is None:
        exit_code = getattr(res, 'exit_code', None)
    stdout = (getattr(res, 'stdout', '') or '').strip()
    stderr = (getattr(res, 'stderr', '') or '').strip()
    if exit_code not in (0, None):
        return None, (f'sql_probe_exit={exit_code} stdout_head={stdout[:120]!r} stderr_head={stderr[:120]!r}')
    if not stdout:
        return None, f'sql_probe_empty_stdout (stderr_head={stderr[:120]!r})'
    try:
        payload = json.loads(stdout)
    except json.JSONDecodeError as exc:
        return None, f'sql_probe_bad_json: {exc} stdout_head={stdout[:120]!r}'
    if 'error' in payload:
        _logger.warning('[rubric.sql] sqlite error on %r: %s', db_path, payload['error'])
        return None, str(payload['error'])
    row = payload.get('row')
    if row is None:
        return None, 'sql_returned_no_rows'
    return row[0], None


def _quote_sqlite_identifier(name: str) -> str:
    """为 SQLite DDL/DML 引用来自 state 的表名或列名。"""
    if not name:
        raise ValueError('empty identifier')
    return '"' + name.replace('"', '""') + '"'


def _state_sql_rows(payload: Any) -> list[dict[str, Any]]:
    """将常见的 dump_state 表形状归一化为行字典。"""
    if isinstance(payload, dict) and isinstance(payload.get('rows'), list):
        raw_rows = payload.get('rows') or []
    elif isinstance(payload, list):
        raw_rows = payload
    else:
        return []
    return [row for row in raw_rows if isinstance(row, dict)]


def _state_sql_columns_from_schema(schema: Any) -> list[str]:
    """将 dump_state schema 元数据归一化为 SQLite 列名。"""
    if not isinstance(schema, list):
        return []
    columns: list[str] = []
    seen: set[str] = set()
    for item in schema:
        if isinstance(item, str):
            col = item
        elif isinstance(item, dict):
            raw_name = item.get('name') or item.get('column') or item.get('column_name')
            col = raw_name if isinstance(raw_name, str) else ''
        else:
            col = ''
        col = col.strip()
        if col and col not in seen:
            seen.add(col)
            columns.append(col)
    return columns


def _state_sql_cell(value: Any) -> Any:
    """将类 JSON 的 state 单元格转换为 SQLite 可绑定的值。"""
    if isinstance(value, (str, int, float, type(None))):
        return value
    if isinstance(value, bool):
        return int(value)
    return json.dumps(value, ensure_ascii=False, sort_keys=True)


def _load_state_sql_tables(
    conn: sqlite3.Connection,
    *,
    namespace: str,
    state_slice: Any,
    prefix_tables: bool,
) -> int:
    """将一个 dump_state 命名空间加载到已存在的 SQLite 连接中。"""
    if not namespace.replace('_', '').isalnum():
        raise ValueError(f'invalid namespace: {namespace!r}')
    if not isinstance(state_slice, dict):
        raise ValueError(f'state namespace {namespace!r} is not an object')
    tables_obj = state_slice.get('tables')
    if not isinstance(tables_obj, dict):
        raise ValueError(f'state namespace {namespace!r} has no object-shaped tables')
    table_schemas = state_slice.get('tableSchemas')
    if not isinstance(table_schemas, dict):
        table_schemas = {}

    loaded = 0
    for table_name, table_payload in tables_obj.items():
        if not isinstance(table_name, str) or not table_name.replace('_', '').isalnum():
            continue
        rows = _state_sql_rows(table_payload)
        columns = sorted({str(col) for row in rows for col in row if str(col)})
        if not columns and isinstance(table_payload, dict):
            columns = _state_sql_columns_from_schema(table_payload.get('columns'))
        if not columns:
            columns = _state_sql_columns_from_schema(table_schemas.get(table_name))
        if not columns:
            continue

        sqlite_table_name = f'{namespace}_{table_name}' if prefix_tables else table_name
        quoted_table = _quote_sqlite_identifier(sqlite_table_name)
        quoted_columns = [_quote_sqlite_identifier(col) for col in columns]
        conn.execute(f'CREATE TABLE {quoted_table} ({", ".join(quoted_columns)})')

        if rows:
            placeholders = ', '.join('?' for _ in columns)
            insert_sql = f'INSERT INTO {quoted_table} ({", ".join(quoted_columns)}) VALUES ({placeholders})'
            conn.executemany(
                insert_sql,
                [[_state_sql_cell(row.get(col)) for col in columns] for row in rows],
            )
        loaded += 1
    return loaded


def _state_sqlite_connection(state: dict, namespaces: list[str]) -> sqlite3.Connection:
    """将一个或多个 dump_state 命名空间物化到内存 SQLite 数据库中。

    支持当前 mocks 使用的两种表布局：
    - taobao：``tables.orders`` 直接是行字典的列表。
    - pdd：``tables.orders`` 是 ``{"count": n, "rows": [...]}``。

    对于单个命名空间，表名保持不变（``orders``）。
    对于多个命名空间，表名会加上前缀（``taobao_orders``、
    ``pdd_orders``），从而让单条 SQL 探针能够对比两个平台。
    """
    conn = sqlite3.connect(':memory:')
    conn.row_factory = sqlite3.Row
    loaded = 0
    try:
        prefix_tables = len(namespaces) > 1
        for namespace in namespaces:
            state_slice = state.get(namespace)
            if state_slice is None:
                raise ValueError(f'missing_slice: {namespace!r}')
            loaded += _load_state_sql_tables(
                conn,
                namespace=namespace,
                state_slice=state_slice,
                prefix_tables=prefix_tables,
            )
        if loaded == 0:
            raise ValueError(f'namespaces {namespaces!r} have no loadable tables')
        return conn
    except Exception:
        conn.close()
        raise


_STATE_SQLITE_CACHE: dict[tuple[int, str], sqlite3.Connection] = {}


def _parse_state_sql_namespaces(raw: str) -> list[str]:
    namespaces = [part.strip() for part in raw.split(',') if part.strip()]
    if not namespaces:
        raise ValueError('empty namespace')
    for namespace in namespaces:
        if not namespace.replace('_', '').isalnum():
            raise ValueError(f'invalid namespace: {namespace!r}')
    return namespaces


def _state_sql_cache(state: dict, namespace_spec: str) -> tuple[sqlite3.Connection | None, str | None]:
    """为一个或多个 dump_state 命名空间返回缓存的内存 SQLite DB。"""
    try:
        namespaces = _parse_state_sql_namespaces(namespace_spec)
    except ValueError as exc:
        return None, str(exc)
    cache_key = (id(state), ','.join(namespaces))
    cached = _STATE_SQLITE_CACHE.get(cache_key)
    if cached is not None:
        return cached, None
    try:
        conn = _state_sqlite_connection(state, namespaces)
    except Exception as exc:  # noqa: BLE001 - 无效 state 形状会成为 rubric detail
        return None, f'state_sql_materialize_failed: {exc}'
    _STATE_SQLITE_CACHE[cache_key] = conn
    return conn, None


def _snapshot_state_sql_scalar(state: dict, namespace_spec: str, sql: str) -> tuple[Any, str | None]:
    """针对某个 dump_state 命名空间运行一个标量 SQL 探针。"""
    conn, err = _state_sql_cache(state, namespace_spec)
    if err is not None:
        return None, err
    assert conn is not None
    try:
        row = conn.execute(sql).fetchone()
    except sqlite3.Error as exc:
        return None, f'sqlite_error: {exc}'
    if row is None:
        return None, 'sql_returned_no_rows'
    return row[0], None


def _compare_sql_scalar(cfg: dict, val: Any, *, label: str) -> VerifierResult:
    """应用被 sql 与 state_sql 共享的 state_equals 标量谓词。"""
    if 'one_of' in cfg:
        allowed = cfg.get('one_of')
        if not isinstance(allowed, list):
            return False, 'INVALID', f'{label}: one_of must be a list', ''
        passed = False
        for c in allowed:
            try:
                if val is not None and float(val) == float(c):
                    passed = True
                    break
            except (TypeError, ValueError):
                pass
            if str(val) == str(c):
                passed = True
                break
        return passed, 'Y' if passed else 'N', f'{label} got={val!r} one_of={allowed!r}', ''

    if 'equals' in cfg:
        expected = cfg['equals']
        passed = _coerce_numeric_equal(val, expected) if val is not None else False
        return passed, 'Y' if passed else 'N', f'{label} got={val!r} equals={expected!r}', ''

    if 'expected' in cfg:  # 向后兼容别名
        expected = cfg['expected']
        passed = _coerce_numeric_equal(val, expected) if val is not None else False
        return passed, 'Y' if passed else 'N', f'{label} got={val!r} expected={expected!r}', ''

    return False, 'INVALID', f'{label}: must set either equals or one_of', ''
