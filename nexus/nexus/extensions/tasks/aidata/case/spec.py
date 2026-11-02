"""加载并验证 ``cases/<id>/`` 清单和 rubric。

一个 "case" 是自包含的基准场景。目前接受两种布局
（首选 v4.2，v2 仍可用；每个 case 恰好存在一种）：

v4.2 布局（对齐 APEX-SWE）::

    cases/<id>/
    ├── task.yaml               ← 清单（v4.2 名称）
    ├── instruction.md          ← 面向 agent 的提示词（当 task.yaml
    │                             未内联 ``user_prompt`` 时使用）
    ├── rubric.yaml             ← v4.2 checks[] + assert DSL
    ├── data/                   ← 上下文 fixture （agent 可见）
    ├── tests/                  ← 可选，产出 junit 的测试
    ├── run-test.sh             ← 可选：由引擎调用以生成 test_result
    ├── solution/               ← 面向审阅者的参考（agent 不可见）
    └── assets/                 ← 引导脚本 / 种子数据

v2 布局（遗留；所有迁移前的 case ）::

    cases/<id>/
    ├── case.yaml
    ├── rubric.yaml
    ├── files/
    ├── example_rollout.json
    └── assets/

task.yaml / case.yaml 的 schema （两者字段一致）::

    task_id:             str         # 人类标签，会出现在输出中
    user_prompt:         str         # 若存在 instruction.md 则可选
    context_files:       list[str]   # 相对于 cases/<id>/{data,files}/
    extra_mcp_servers:   list[spec]  # 可选；合并进默认 MCP 目录
    agent_tools:         list|dict   # 可选；供 main_agent 使用的静态 Nexus 工具
    services:            list[spec]  # 可选；辅助的 docker 容器
    bootstrap_script:    str         # 可选；相对于 case 目录
    ask_user:            mapping     # 可选；确定性的人类响应
    initial_trajectory:  str|list    # 可选；预置的 OpenAI-chat 消息
    agent_timeout:       int         # 可选，默认 1800
    observation_time_timeout: float  # 可选； main_agent 工具 / 观察超时
    max_iterations:      int         # 可选，默认 200
"""

from __future__ import annotations

import json
import logging

from pathlib import Path
from typing import Any

from .models import (
    AidataCaseMetadata,
    AidataService,
    AidataServiceCommand,
    AidataServiceConnection,
    AidataServiceFile,
    AidataTaskInput,
)
from .paths import CASES_DIR
from .schema import (
    _ALLOWED_AGENT_TOOLS_BY_CATEGORY,
    _METADATA_DATE_RE,
)


logger = logging.getLogger(__name__)

# case 作者可以通过 ``task.yaml::agent_tools`` 暴露给 ``main_agent`` 的工具。
# 该映射与磁盘布局对应 —— key 为 ``nexus/extensions/tools/`` 下的目录名，
# value 为该目录下的文件名（同时也作为 ``tool_registry`` 的 ID ）。
# 若要开放新目录，请在此显式添加，避免无意间暴露不该做静态工具的代码。


_ALLOWED_AGENT_TOOLS = {tool for tools in _ALLOWED_AGENT_TOOLS_BY_CATEGORY.values() for tool in tools}


# ---------------------------------------------------------------------------
# case.yaml
# ---------------------------------------------------------------------------


def case_manifest_path(case_id: str) -> Path:
    """返回 ``case_id`` 对应最合适清单文件的绝对路径。

    v4.2 ``task.yaml`` 优先于遗留的 ``case.yaml`` （用意是让迁移中的 case 可以并排评审）。
    调用方需自行做存在性检查。
    """
    case_dir = CASES_DIR / case_id
    task_file = case_dir / 'task.yaml'
    if task_file.exists():
        return task_file
    return case_dir / 'case.yaml'


def resolve_case(task_input: AidataTaskInput) -> None:
    """就地修改 ``task_input`` ：从 case 清单
    （优先 ``cases/<case_id>/task.yaml`` ， ``case.yaml`` 作为回退）以及
    ``instruction.md`` 中填充所有未显式设置的字段。

    若 ``case_id`` 为空则不做任何操作。若设置了 case_id 但两种清单文件均缺失 / 格式错误
    则抛出 ``ValueError`` 。
    """
    if not task_input.case_id:
        return
    case_dir = CASES_DIR / task_input.case_id
    case_file = case_manifest_path(task_input.case_id)
    if not case_file.exists():
        raise ValueError(
            f'case manifest not found for case_id={task_input.case_id!r}; '
            f'expected {case_dir / "task.yaml"} or {case_dir / "case.yaml"}'
        )

    import yaml  # 局部 import —— 避免 PyYAML 出现在 import 期依赖图中

    data = yaml.safe_load(case_file.read_text(encoding='utf-8')) or {}
    if not isinstance(data, dict):
        raise ValueError(f'case file must be a YAML mapping: {case_file}')

    if not task_input.user_prompt.strip() and data.get('user_prompt'):
        task_input.user_prompt = str(data['user_prompt'])
    # v4.2： instruction.md 是 task.yaml 未内联提示词时提示词的规范放置位置。
    # 保留 ``case.yaml`` 类型 case 不变（它们仍使用内联的 ``user_prompt:`` ）。
    if not task_input.user_prompt.strip():
        ps_file = case_dir / 'instruction.md'
        if ps_file.exists():
            try:
                task_input.user_prompt = ps_file.read_text(encoding='utf-8').strip()
            except Exception as exc:
                raise ValueError(f'cannot read instruction.md: {exc}') from exc
    if not task_input.context_files and data.get('context_files'):
        task_input.context_files = [_resolve_case_path(case_dir, str(p)) for p in (data.get('context_files') or [])]
    if data.get('task_id') and task_input.task_id == 'aidata_default':
        task_input.task_id = str(data['task_id'])
    # 数值覆写：仅当 YAML key 显式存在 且 调用方保留了模型默认值时应用。
    # 使用 ``in data`` 而非布尔判断，以便若模型级验证允许显式的 0 值时也能干净地往返传输。
    # 由于 pydantic v2 无法优雅区分 “调用方传了默认值” 与 “未提供” ，
    # 我们把与模型默认相同的值视为 “未指定” 。
    if 'agent_timeout' in data and task_input.agent_timeout == 1800:
        task_input.agent_timeout = int(data['agent_timeout'])
    if 'observation_time_timeout' in data and task_input.observation_time_timeout is None:
        task_input.observation_time_timeout = float(data['observation_time_timeout'])
    if 'max_iterations' in data and task_input.max_iterations == 200:
        task_input.max_iterations = int(data['max_iterations'])
    if not task_input.initial_trajectory and data.get('initial_trajectory') is not None:
        task_input.initial_trajectory = _parse_initial_trajectory(data['initial_trajectory'], case_dir, case_file)
    if not task_input.dump_state and data.get('dump_state') is not None:
        task_input.dump_state = _parse_dump_state_spec(data['dump_state'], case_file)
    if not task_input.extra_mcp_servers and data.get('extra_mcp_servers'):
        extras = data['extra_mcp_servers']
        if not isinstance(extras, list):
            raise ValueError(f'{case_file}: extra_mcp_servers must be a list')
        task_input.extra_mcp_servers = [normalize_extra_spec(s) for s in extras]
    if not task_input.agent_tools and data.get('agent_tools') is not None:
        task_input.agent_tools = _parse_agent_tools_spec(data['agent_tools'], case_file)
    if task_input.wx_agent_hub is None and data.get('wx_agent_hub') is not None:
        task_input.wx_agent_hub = _parse_wx_agent_hub_spec(data['wx_agent_hub'], case_file)
    if not task_input.services and data.get('services'):
        svc_raw = data['services']
        if not isinstance(svc_raw, list):
            raise ValueError(f'{case_file}: services must be a list')
        task_input.services = [_parse_service_spec(s, case_file) for s in svc_raw]
    if not task_input.bootstrap_script and data.get('bootstrap_script'):
        task_input.bootstrap_script = str(data['bootstrap_script'])
    if task_input.ask_user is None and data.get('ask_user') is not None:
        task_input.ask_user = _parse_ask_user_config(data['ask_user'], case_dir, case_file)
    if not task_input.rubric_sqlite_path and data.get('rubric_sqlite_path'):
        raw_sqlite = data['rubric_sqlite_path']
        if not isinstance(raw_sqlite, str) or not raw_sqlite.strip():
            raise ValueError(f'{case_file}: rubric_sqlite_path must be a non-empty string')
        raw_sqlite = raw_sqlite.strip()
        # `service:<name>:<path>` 前缀会将探针路由到 sidecar 运行时
        # （在 services[].name 中声明），而不是 main。裸路径保留
        # 遗留的 “探针在 main 上运行” 行为，完全向后兼容。
        if raw_sqlite.startswith('service:'):
            rest = raw_sqlite[len('service:') :]
            if ':' not in rest:
                raise ValueError(
                    f'{case_file}: rubric_sqlite_path with `service:` prefix must be '
                    f'`service:<name>:<path>` (got {raw_sqlite!r})'
                )
            svc_name, svc_path = rest.split(':', 1)
            svc_name = svc_name.strip()
            svc_path = svc_path.strip()
            if not svc_name or not svc_path:
                raise ValueError(
                    f'{case_file}: rubric_sqlite_path service-prefix form requires '
                    f'a non-empty service name and path (got {raw_sqlite!r})'
                )
            # 将面向人类的 service.name 翻译为 ExecutionContext 用来索引运行时的 runtime_name
            # —— ``engine/rubric.py::_resolve_sql_runtime`` 通过 runtime_name
            # （即 ``engine/services.py::setup_services`` 注册所用的名字）查找 ctx._runtimes。
            # case 作者写更友好的 `name` （例如 `lifegym-mocks` ），
            # 我们在此解析对应的 runtime_name，因此拼写错误会在静态验证阶段直接爆掉。
            name_to_runtime = {s.name: s.runtime_name for s in task_input.services}
            if svc_name not in name_to_runtime:
                raise ValueError(
                    f'{case_file}: rubric_sqlite_path references service {svc_name!r} '
                    f'but services[].name only declares {sorted(name_to_runtime)!r}'
                )
            task_input.rubric_sqlite_service = name_to_runtime[svc_name]
            task_input.rubric_sqlite_path = svc_path
        else:
            task_input.rubric_sqlite_path = raw_sqlite
    if task_input.case_metadata is None and data.get('metadata') is not None:
        task_input.case_metadata = _parse_case_metadata(data['metadata'], case_file)
    if task_input.runtime_image is None and data.get('runtime') is not None:
        task_input.runtime_image = _parse_runtime_main_image(data['runtime'], case_file)
    if not task_input.pending_upstream and data.get('pending_upstream'):
        raw_pending = data['pending_upstream']
        if not isinstance(raw_pending, str) or not raw_pending.strip():
            raise ValueError(f'{case_file}: pending_upstream must be a non-empty string (got {raw_pending!r})')
        task_input.pending_upstream = raw_pending.strip()


def _parse_initial_trajectory(raw: Any, case_dir: Path, case_file: Path) -> list[dict[str, Any]]:
    """从内联 list 或 JSON 文件解析 ``task.yaml::initial_trajectory`` 。"""
    if isinstance(raw, str):
        rel = raw.strip()
        if not rel:
            raise ValueError(f'{case_file}: initial_trajectory path must be non-empty')
        path = _resolve_case_path(case_dir, rel)
        try:
            payload = json.loads(Path(path).read_text(encoding='utf-8'))
        except Exception as exc:
            raise ValueError(f'{case_file}: cannot read initial_trajectory {rel!r}: {exc}') from exc
    else:
        payload = raw
    if not isinstance(payload, list) or not all(isinstance(msg, dict) for msg in payload):
        raise ValueError(f'{case_file}: initial_trajectory must be a list of message objects')
    return [dict(msg) for msg in payload]


def _parse_dump_state_spec(raw: Any, case_file: Path) -> dict[str, dict[str, Any]]:
    """解析 ``task.yaml::dump_state`` 中每个 mock 工具的参数。"""
    if not isinstance(raw, dict):
        raise ValueError(f'{case_file}: dump_state must be a mapping of namespace -> argument mapping')
    parsed: dict[str, dict[str, Any]] = {}
    for raw_name, raw_args in raw.items():
        name = str(raw_name).strip()
        if not name:
            raise ValueError(f'{case_file}: dump_state contains an empty namespace')
        if not name.replace('_', '').replace('-', '').isalnum():
            raise ValueError(f'{case_file}: dump_state namespace {name!r} must be a plain identifier')
        if raw_args is None:
            parsed[name] = {}
            continue
        if not isinstance(raw_args, dict):
            raise ValueError(f'{case_file}: dump_state.{name} must be a mapping of tool arguments')
        parsed[name] = dict(raw_args)
    return parsed


def _parse_ask_user_config(raw: Any, case_dir: Path, case_file: Path) -> dict[str, Any]:
    """解析 ``task.yaml::ask_user`` 并解析其脚本路径。"""
    if not isinstance(raw, dict):
        raise ValueError(f'{case_file}: ask_user must be a mapping')
    cfg = dict(raw)
    enabled = bool(cfg.get('enabled', True))
    cfg['enabled'] = enabled
    script = str(cfg.get('script') or 'assets/ask_user.py')
    if enabled:
        script_path = _resolve_case_path(case_dir, script)
        if not Path(script_path).is_file():
            raise ValueError(f'{case_file}: ask_user script not found: {script!r}')
        cfg['script_path'] = script_path
    return cfg


def _parse_case_metadata(raw: Any, case_file: Path) -> AidataCaseMetadata:
    """将 ``task.yaml::metadata`` 提升为类型化的 :class:`AidataCaseMetadata` 。

    语义细节（而非纯 schema ）之所以放在这里，是因为它们也是 ``aidata.py validate`` 在
    CI 中静态复检的内容 —— 两条代码路径共享 :data:`_METADATA_DATE_RE` ，
    因此运行时永远看不到已被验证器拒绝的 case 。

    规则：
    - `created_at` 必填，需形似 ``YYYY-MM-DD`` 。
    - `authors` 可选 list[str] ；默认为空列表。
    - `tags` 可选 list[str] 。
    """
    if not isinstance(raw, dict):
        raise ValueError(f'{case_file}: metadata must be a mapping')
    from .models import AidataCaseMetadata

    # pyyaml 会把裸的 ``2026-04-21`` 自动转换为 ``datetime.date`` —— 同时接受
    # 字符串或 date 对象，并规范化为字符串，
    # 让下游消费者（ pydantic 模型， JSON 序列化）看到统一的类型，
    # 与作者书写 yaml 的方式无关。
    created_at_raw = raw.get('created_at')
    import datetime as _dt

    if isinstance(created_at_raw, _dt.date) and not isinstance(created_at_raw, _dt.datetime):
        created_at = created_at_raw.isoformat()
    else:
        created_at = created_at_raw
    if not isinstance(created_at, str) or not _METADATA_DATE_RE.match(created_at):
        raise ValueError(f'{case_file}: metadata.created_at must be YYYY-MM-DD (got {created_at_raw!r})')
    authors_raw = raw.get('authors') or []
    if not isinstance(authors_raw, list) or not all(isinstance(x, str) for x in authors_raw):
        raise ValueError(f'{case_file}: metadata.authors must be a list of strings')
    authors = [str(x) for x in authors_raw]
    tags_raw = raw.get('tags') or []
    if not isinstance(tags_raw, list) or not all(isinstance(x, str) for x in tags_raw):
        raise ValueError(f'{case_file}: metadata.tags must be a list of strings')
    return AidataCaseMetadata(
        authors=authors,
        created_at=created_at,
        tags=[str(x) for x in tags_raw],
    )


# 与 ``aidata.py::_validate_case`` 共享，让 CI 与运行时对
# “有效的 email / date” 保持一致定义。
# （在本模块顶部从 .schema 导入）


def _parse_runtime_main_image(raw: Any, case_file: Path) -> str:
    """从 case 清单中提取 ``runtime.main.image`` 。

    可接受形态::

        runtime:
          main:
            image: mirrors.tencent.com/.../image:tag

    目前仅读取 ``main.image`` —— 其它按运行时的配置
    （ exposed_ports、env、... ）属于 :class:`AidataService` 中的 sidecar 配置，
    对 main agent 容器没有实际意义。
    """
    if not isinstance(raw, dict):
        raise ValueError(f'{case_file}: runtime must be a mapping, got {type(raw).__name__}')
    main = raw.get('main') or {}
    if not isinstance(main, dict):
        raise ValueError(f'{case_file}: runtime.main must be a mapping, got {type(main).__name__}')
    image = main.get('image')
    if image is None:
        raise ValueError(
            f'{case_file}: runtime.main present but missing `image` — '
            f'drop the block entirely or set runtime.main.image explicitly'
        )
    if not isinstance(image, str) or not image.strip():
        raise ValueError(f'{case_file}: runtime.main.image must be a non-empty string (got {image!r})')
    return image.strip()


def _parse_wx_agent_hub_spec(raw: Any, case_file: Path) -> dict[str, list[str]]:
    """解析并验证 ``case.yaml::wx_agent_hub`` 块。

    可接受形态（两个子列表都可选，默认为 [] ）：

        wx_agent_hub:
          pre_installed:    [food_delivery]
          market_available: [ride_app, weather_app]

    返回规范化后的 dict ``{'pre_installed': [...], 'market_available': [...]}`` ，
    两个 key 始终存在（简化 :mod:`engine.mcp_catalog` 中下游消费者的处理）。
    未知 key / 类型错误会抛出 ``ValueError`` 并指向 case 文件，
    让作者立刻看到问题，而不是静默丢失某个声明。
    """
    if not isinstance(raw, dict):
        raise ValueError(f'{case_file}: wx_agent_hub must be a mapping, got {type(raw).__name__}')
    known = {'pre_installed', 'market_available'}
    unknown = set(raw) - known
    if unknown:
        raise ValueError(
            f'{case_file}: wx_agent_hub has unknown keys: {sorted(unknown)} (expected any of {sorted(known)})'
        )
    out: dict[str, list[str]] = {}
    for key in ('pre_installed', 'market_available'):
        value = raw.get(key) or []
        if not isinstance(value, list):
            raise ValueError(f'{case_file}: wx_agent_hub.{key} must be a list, got {type(value).__name__}')
        if not all(isinstance(v, str) and v for v in value):
            raise ValueError(f'{case_file}: wx_agent_hub.{key} must be a list of non-empty str')
        out[key] = list(value)
    return out


def _parse_agent_tools_spec(raw: Any, case_file: Path) -> list[str]:
    """将 ``task.yaml::agent_tools`` 解析为供 main_agent 使用的静态工具名列表。

    形态是从工具目录名（ ``nexus/extensions/tools/`` 下）到工具文件名列表的映射::

        agent_tools:
          web_search:
            - google_serper_search
            - visit

    评审者可以直接 grep ``nexus/extensions/tools/web_search/google_serper_search.py`` ，
    无需经过任何别名层。
    """
    if not isinstance(raw, dict):
        raise ValueError(
            f'{case_file}: agent_tools must be a mapping of '
            'tool-directory → list, e.g. {web_search: [google_serper_search, visit]}; '
            f'got {type(raw).__name__}'
        )

    unknown_categories = set(raw) - _ALLOWED_AGENT_TOOLS_BY_CATEGORY.keys()
    if unknown_categories:
        raise ValueError(
            f'{case_file}: agent_tools has unknown category keys: '
            f'{sorted(unknown_categories)}; expected one of '
            f'{sorted(_ALLOWED_AGENT_TOOLS_BY_CATEGORY)}'
        )

    tools: list[str] = []
    for category, items in raw.items():
        if not isinstance(items, list):
            raise ValueError(
                f'{case_file}: agent_tools.{category} must be a list of tool '
                f'names from nexus/extensions/tools/{category}/'
            )
        allow = _ALLOWED_AGENT_TOOLS_BY_CATEGORY[category]
        for i, name in enumerate(items):
            if isinstance(name, bool):
                raise ValueError(
                    f'{case_file}: agent_tools.{category}[{i}] must be a tool name '
                    f'(file stem under nexus/extensions/tools/{category}/), '
                    'not a boolean'
                )
            if not isinstance(name, str) or not name.strip():
                raise ValueError(f'{case_file}: agent_tools.{category}[{i}] must be a non-empty string')
            tool = name.strip()
            if tool not in allow:
                raise ValueError(
                    f'{case_file}: agent_tools.{category}[{i}]={tool!r} is not allowed; '
                    f'expected a file stem from nexus/extensions/tools/{category}/, '
                    f'one of {sorted(allow)}'
                )
            if tool not in tools:
                tools.append(tool)
    return tools


def normalize_extra_spec(raw: dict[str, Any]) -> dict[str, Any]:
    """将 case.yaml 的一个 ``extra_mcp_servers`` 条目规范化为
    ``ctx.load_mcp_tools`` 期望的形态。 接受两种输入形态：

      扁平（在 YAML 中书写更简短，推荐）::

          {name, command, args?, env?, runtime_name?}

      嵌套（等价于内部 ``_mcp_specs`` 输出）::

          {name, runtime_name?, client: {transport, command, args, env}}

    此处会把扁平形式展开为嵌套形式，让上线中的 case 只需在 YAML 中写最少内容。

    ``runtime_name`` 对每种形态都默认为 ``'main'`` ： MCP 连接
    从沙箱内部（通过代理）发起，因此任何 ``text/event-stream`` 响应
    都由运行时解析，而不是 host 上的编排器。 若确实需要 host 直连
    （例如仅从 host 可达的 mock ），作者必须显式设置 ``runtime_name: null`` 来退出该行为。
    """
    if not isinstance(raw, dict) or not raw.get('name'):
        raise ValueError(f'extra_mcp_servers entry must be a mapping with `name`: {raw!r}')
    if 'client' in raw:
        return {
            'name': raw['name'],
            'runtime_name': raw.get('runtime_name', 'main'),
            'client': raw['client'],
        }
    if 'url' in raw:
        client: dict[str, Any] = {
            'transport': raw.get('transport', 'aiohttp_streamable_http'),
            'url': raw['url'],
            'headers': dict(raw.get('headers') or {}),
        }
        if raw.get('timeout_s') is not None:
            client['timeout_s'] = float(raw['timeout_s'])
        return {
            'name': raw['name'],
            'runtime_name': raw.get('runtime_name', 'main'),
            'client': client,
        }
    if 'command' not in raw:
        raise ValueError(f'extra_mcp_servers entry {raw["name"]!r} missing `command` or `url` (flat form)')
    return {
        'name': raw['name'],
        'runtime_name': raw.get('runtime_name', 'main'),
        'client': {
            'transport': 'stdio',
            'command': raw['command'],
            'args': list(raw.get('args') or []),
            'env': dict(raw.get('env') or {}),
        },
    }


def _parse_service_spec(raw: Any, case_file: Path) -> AidataService:
    """将 ``case.yaml::services`` 中一条条目解析为 ``AidataService`` 。

    支持模型上文档化的完整扁平 YAML 形式。 未知 key 会抛错 —— 否则
    像 ``ready_check`` 与 ``ready-check`` 这样的拼写错误会静默丢失
    健康探针，让 task 越过尚未就绪的后端而继续运行。
    """
    if not isinstance(raw, dict) or not raw.get('name'):
        raise ValueError(f'{case_file}: services entry must be a mapping with `name`: {raw!r}')
    if not raw.get('runtime_name'):
        raise ValueError(f'{case_file}: service {raw.get("name")!r} missing `runtime_name`')

    def _parse_cmd(field: str) -> AidataServiceCommand | None:
        val = raw.get(field)
        if val is None:
            return None
        if isinstance(val, str):
            return AidataServiceCommand(command=val)
        if isinstance(val, dict):
            has_cmd = bool(val.get('command'))
            has_script = bool(val.get('script'))
            if has_cmd == has_script:
                raise ValueError(
                    f'{case_file}: service {raw["name"]!r}.{field} must have exactly '
                    f'one of `command` or `script` (got command={val.get("command")!r}, '
                    f'script={val.get("script")!r})'
                )
            known = {'command', 'script', 'cwd', 'env', 'timeout_s'}
            unknown = set(val) - known
            if unknown:
                raise ValueError(f'{case_file}: service {raw["name"]!r}.{field} has unknown keys: {sorted(unknown)}')
            return AidataServiceCommand(
                command=str(val['command']) if has_cmd else None,
                script=str(val['script']) if has_script else None,
                cwd=val.get('cwd'),
                env={str(k): str(v) for k, v in (val.get('env') or {}).items()},
                timeout_s=float(val.get('timeout_s', 120.0)),
            )
        raise ValueError(f'{case_file}: service {raw["name"]!r}.{field} must be str|mapping')

    connections: list[AidataServiceConnection] = []
    for conn_raw in raw.get('connections') or []:
        if not isinstance(conn_raw, dict):
            raise ValueError(f'{case_file}: service {raw["name"]!r} connection must be a mapping: {conn_raw!r}')
        connections.append(
            AidataServiceConnection(
                placeholder=str(conn_raw['placeholder']),
                internal_port=int(conn_raw['internal_port']),
                value_template=str(conn_raw.get('value_template', 'http://{host}:{port}')),
                env_name=conn_raw.get('env_name'),
            )
        )

    state_mounts_raw = raw.get('state_mounts') or {}
    if not isinstance(state_mounts_raw, dict):
        raise ValueError(
            f'{case_file}: service {raw["name"]!r} state_mounts must be a mapping, '
            f'got {type(state_mounts_raw).__name__}'
        )
    state_mounts = {str(k): str(v) for k, v in state_mounts_raw.items()}

    files: list[AidataServiceFile] = []
    files_raw = raw.get('files') or []
    if not isinstance(files_raw, list):
        raise ValueError(f'{case_file}: service {raw["name"]!r} files must be a list')
    case_dir = case_file.parent.resolve()
    for idx, file_raw in enumerate(files_raw):
        if not isinstance(file_raw, dict):
            raise ValueError(f'{case_file}: service {raw["name"]!r} files[{idx}] must be a mapping')
        known_file_keys = {'source', 'target'}
        unknown_file_keys = set(file_raw) - known_file_keys
        if unknown_file_keys:
            raise ValueError(
                f'{case_file}: service {raw["name"]!r} files[{idx}] has unknown keys: {sorted(unknown_file_keys)}'
            )
        source = file_raw.get('source')
        target = file_raw.get('target')
        if not isinstance(source, str) or not source.strip():
            raise ValueError(f'{case_file}: service {raw["name"]!r} files[{idx}].source must be a non-empty string')
        if not isinstance(target, str) or not target.strip():
            raise ValueError(f'{case_file}: service {raw["name"]!r} files[{idx}].target must be a non-empty string')
        source_path = Path(source)
        if source_path.is_absolute():
            raise ValueError(f'{case_file}: service {raw["name"]!r} files[{idx}].source must be relative to case dir')
        if not (case_dir / source_path).resolve().is_relative_to(case_dir):
            raise ValueError(f'{case_file}: service {raw["name"]!r} files[{idx}].source must stay inside case dir')
        if not Path(target).is_absolute():
            raise ValueError(f'{case_file}: service {raw["name"]!r} files[{idx}].target must be an absolute path')
        files.append(AidataServiceFile(source=source.strip(), target=target.strip()))

    known = {
        'name',
        'runtime_name',
        'image',
        'exposed_ports',
        'working_dir',
        'files',
        'startup',
        'ready_check',
        'ready_timeout_s',
        'ready_poll_interval_s',
        'connections',
        'state_mounts',
    }
    unknown = set(raw) - known
    if unknown:
        raise ValueError(f'{case_file}: service {raw["name"]!r} has unknown keys: {sorted(unknown)}')

    return AidataService(
        name=str(raw['name']),
        runtime_name=str(raw['runtime_name']),
        image=raw.get('image'),
        exposed_ports=[int(p) for p in (raw.get('exposed_ports') or [])],
        working_dir=raw.get('working_dir'),
        files=files,
        startup=_parse_cmd('startup'),
        ready_check=_parse_cmd('ready_check'),
        ready_timeout_s=float(raw.get('ready_timeout_s', 120.0)),
        ready_poll_interval_s=float(raw.get('ready_poll_interval_s', 2.0)),
        connections=connections,
        state_mounts=state_mounts,
    )


def _resolve_case_path(case_dir: Path, raw: str) -> str:
    """解析 task.yaml / case.yaml 中声明的上下文文件路径。

    尝试顺序：

      1. 若为绝对路径则原样返回
      2. ``case_dir / 'data' / raw``   ← v4.2 规范位置
      3. ``case_dir / 'files' / raw``  ← 遗留 v2 位置
      4. ``case_dir / raw``            ← 松散布局
      5. ``raw`` 本身（相对于调用方 cwd ，通常是 repo 根）
    """
    p = Path(raw).expanduser()
    if p.is_absolute():
        return str(p)
    for candidate in (case_dir / 'data' / raw, case_dir / 'files' / raw, case_dir / raw):
        if candidate.exists():
            return str(candidate)
    return raw


# ---------------------------------------------------------------------------
# rubric.yaml
# ---------------------------------------------------------------------------


def case_rubric_path(case_id: str) -> Path:
    """返回 ``case_id`` 对应的 rubric 文件绝对路径。

    规范位置为 ``cases/<case_id>/rubric.yaml`` （扁平结构）。 调用方仍需对
    返回路径做 ``exists()`` 检查 —— 未提交 rubric 的 case 在运行时是硬错误
    （见 engine/task.py ）。
    """
    return CASES_DIR / case_id / 'rubric.yaml'


def load_case_rubric(case_id: str) -> dict[str, Any]:
    """从 ``cases/<id>/rubric.yaml`` 加载 ``case_id`` 的 rubric 。

    调用方应先自行检查 ``case_rubric_path().exists()`` —— 若文件缺失我们仍会抛错
    （保持函数职责单一）；task 运行器会将缺失 rubric 视为硬错误。
    """
    rubric_file = case_rubric_path(case_id)
    if not rubric_file.exists():
        raise ValueError(f'rubric file not found for case_id={case_id!r}: {rubric_file}')
    import yaml

    data = yaml.safe_load(rubric_file.read_text(encoding='utf-8')) or {}
    if not isinstance(data, dict) or not data.get('checks'):
        raise ValueError(f'rubric {rubric_file} must be a dict with non-empty `checks`')
    # 默认版本推断：若任一 check 存在 ``assert`` 块，
    # 未显式版本化时视为 v4.2 。 这样即便作者忘写 ``version:`` 头，
    # 引擎的 dispatch 也能与作者意图保持一致。
    if not data.get('version'):
        if any(isinstance(c, dict) and 'assert' in c for c in data.get('checks', [])):
            data['version'] = '4.2'
        else:
            data['version'] = '2.0'
    return data


__all__ = [
    'case_manifest_path',
    'case_rubric_path',
    'load_case_rubric',
    'normalize_extra_spec',
    'resolve_case',
]
