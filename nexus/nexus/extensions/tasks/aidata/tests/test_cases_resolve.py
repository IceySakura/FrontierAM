"""静态层：每个 case 的 manifest 必须能被干净解析。

对每个发现的 case id 在新的 ``AidataTaskInput`` 上运行 ``resolve_case`` ，
断言 YAML 契约成立：

* ``user_prompt`` 最终非空（来自内联 ``user_prompt`` 或
  ``instruction.md`` ）。
* 若 case 声明了 ``extra_mcp_servers`` ，则每一项均归一化为
  ``{name, runtime_name?, client: {transport, command|url, ...}}`` 。
* 若声明了 ``services`` ，则每个 ``connections[].placeholder`` 都是
  ``__UPPER_SNAKE__`` 且唯一（多个 connection 可以共享同一
  ``internal_port`` ——例如在同一端口上暴露多个 URL ）。
* 若声明了 ``wx_agent_hub`` ，则两个列表字段为 ``list[str]`` 。
* 若声明了 ``metadata`` ，则 ``created_at`` 是 YYYY-MM-DD 格式。
* 存在时， ``rubric_sqlite_path`` 是非空字符串。
* 存在 ``rubric.yaml`` 时是合法 YAML （此处不强制 v2 vs
  v4.2 结构——那是 rubric.py 的职责）。

无网络、无 runtime 、无 gongfeng 。 一秒内运行完毕。
"""

from __future__ import annotations

import re

from typing import Any

import pytest
import yaml

from nexus.extensions.tasks.aidata.engine.case_spec import _parse_agent_tools_spec, resolve_case
from nexus.extensions.tasks.aidata.engine.models import AidataTaskInput

from .conftest import ALL_CASE_IDS, CASES_ROOT


DATE_RE = re.compile(r'^\d{4}-\d{2}-\d{2}$')
PLACEHOLDER_RE = re.compile(r'^__[A-Z][A-Z0-9_]*__$')


# 一个最小桩—— WithAgentRunner 要求 ``run_agent`` 可调用。 我们
# 在本文件中从不调用它，但 pydantic 会在构造时校验类型。
async def _never_called_agent(**_: Any) -> Any:  # pragma: no cover
    raise AssertionError('run_agent must not be invoked in the static layer')


def _build_input(case_id: str) -> AidataTaskInput:
    return AidataTaskInput(case_id=case_id, run_agent=_never_called_agent)


# ---------------------------------------------------------------------------
# 每个 case 参数化的测试
# ---------------------------------------------------------------------------


@pytest.mark.parametrize('case_id', ALL_CASE_IDS)
def test_case_manifest_resolves(case_id: str) -> None:
    """``resolve_case`` 成功并产生非空 prompt 。"""
    task_input = _build_input(case_id)
    resolve_case(task_input)

    assert task_input.user_prompt.strip(), (
        f'{case_id}: user_prompt empty after resolve_case — '
        f'neither task.yaml::user_prompt nor instruction.md provided content.'
    )


@pytest.mark.parametrize('case_id', ALL_CASE_IDS)
def test_extra_mcp_servers_normalised(case_id: str) -> None:
    """每个 ``extra_mcp_servers`` 条目都会经过 ``normalize_extra_spec`` 。"""
    task_input = _build_input(case_id)
    resolve_case(task_input)

    for spec in task_input.extra_mcp_servers:
        assert isinstance(spec, dict), f'{case_id}: extra_mcp_servers entry must be a dict: {spec!r}'
        assert spec.get('name'), f'{case_id}: extra_mcp_servers entry missing `name`: {spec!r}'
        client = spec.get('client') or {}
        assert isinstance(client, dict), f'{case_id}: extra_mcp_servers[{spec["name"]}].client must be mapping'
        transport = client.get('transport')
        assert transport in {'stdio', 'aiohttp_streamable_http', 'streamable_http', None}, (
            f'{case_id}: unknown transport {transport!r} for {spec["name"]}'
        )
        if transport == 'stdio' or (transport is None and 'command' in client):
            assert client.get('command'), f'{case_id}: stdio entry {spec["name"]} must set `command`'
        elif transport in ('aiohttp_streamable_http', 'streamable_http'):
            assert client.get('url'), f'{case_id}: HTTP entry {spec["name"]} must set `url`'
            # ``runtime_name: None`` （引擎从主机连接）与
            # ``runtime_name: main`` （连接从沙箱内部经由 runtime 代理发起）
            # 均合法。 ``interview_hire`` 使用后者，让打包好的 mock HTTP
            # sidecar 仍经过 main runtime 路由； ``email_http_poc`` 使用前者
            # 直接从主机连接。 两者在发行 case 中都有出现。


@pytest.mark.parametrize('case_id', ALL_CASE_IDS)
def test_agent_tools_shape(case_id: str) -> None:
    """若声明， ``agent_tools`` 会解析为允许的静态工具名。"""
    task_input = _build_input(case_id)
    resolve_case(task_input)

    allowed = {'google_serper_search', 'google_search', 'hunyuan_search', 'visit', 'visit_image'}
    assert len(task_input.agent_tools) == len(set(task_input.agent_tools)), f'{case_id}: duplicate agent_tools'
    for tool_name in task_input.agent_tools:
        assert tool_name in allowed, f'{case_id}: unsupported agent_tool {tool_name!r}'


def test_agent_tools_parser_category_keyed() -> None:
    """``agent_tools`` 镜像磁盘布局：键是 nexus/extensions/tools/
    下的工具目录，值是该目录内的文件
    stem 列表。"""
    parsed = _parse_agent_tools_spec(
        {'web_search': ['google_serper_search', 'visit']},
        CASES_ROOT / '_dummy' / 'task.yaml',
    )
    assert parsed == ['google_serper_search', 'visit']

    # web_search/ 下所有允许列表内的工具都能解析。
    assert _parse_agent_tools_spec(
        {'web_search': ['google_search', 'hunyuan_search', 'visit_image']},
        CASES_ROOT / '_dummy' / 'task.yaml',
    ) == ['google_search', 'hunyuan_search', 'visit_image']


def test_agent_tools_rejects_boolean_shortcut() -> None:
    """布尔值条目会隐藏实际调用的工具——拒绝这些写法，
    强制 case 作者写出真实的文件 stem 。"""
    with pytest.raises(ValueError, match='boolean'):
        _parse_agent_tools_spec({'web_search': [True]}, CASES_ROOT / '_dummy' / 'task.yaml')


def test_agent_tools_rejects_unknown_tool() -> None:
    """工具名必须匹配指定工具目录下的文件。"""
    with pytest.raises(ValueError, match='not allowed'):
        _parse_agent_tools_spec({'web_search': ['bing']}, CASES_ROOT / '_dummy' / 'task.yaml')


def test_agent_tools_rejects_unknown_category() -> None:
    """分类键必须匹配一个已被显式开放给静态 agent
    使用的目录。"""
    with pytest.raises(ValueError, match='unknown category'):
        _parse_agent_tools_spec({'browser': ['playwright']}, CASES_ROOT / '_dummy' / 'task.yaml')


@pytest.mark.parametrize('case_id', ALL_CASE_IDS)
def test_services_shape(case_id: str) -> None:
    """若 case 声明了 ``services`` ，每一项都具备最低可行的形状。"""
    task_input = _build_input(case_id)
    resolve_case(task_input)

    seen_runtime_names: set[str] = set()
    for svc in task_input.services:
        assert svc.runtime_name.startswith('service_'), (
            f"{case_id}: services[{svc.name}].runtime_name must start with 'service_' (got {svc.runtime_name!r})"
        )
        assert svc.runtime_name != 'main', f'{case_id}: services[{svc.name}].runtime_name must not be `main`'
        assert svc.runtime_name not in seen_runtime_names, f'{case_id}: duplicate runtime_name {svc.runtime_name!r}'
        seen_runtime_names.add(svc.runtime_name)

        # 唯一性约束在 placeholder（注入变量名）上，而不是 internal_port：
        # 引擎对每个 connection 各自查 runtime.port_mappings[internal_port]，
        # 允许多个连接共享同一端口（同一服务暴露多个 URL，如 .../mcp 与 base_url），
        # 只要 placeholder 互不相同即可。
        seen_placeholders: set[str] = set()
        for conn in svc.connections:
            assert PLACEHOLDER_RE.match(conn.placeholder), (
                f'{case_id}: placeholder {conn.placeholder!r} must match __UPPER_SNAKE__'
            )
            assert conn.placeholder not in seen_placeholders, (
                f'{case_id}: duplicate placeholder {conn.placeholder!r} in services[{svc.name}].connections'
            )
            seen_placeholders.add(conn.placeholder)


@pytest.mark.parametrize('case_id', ALL_CASE_IDS)
def test_wx_agent_hub_shape(case_id: str) -> None:
    """若声明， ``wx_agent_hub`` 有两个 ``list[str]`` 字段。"""
    task_input = _build_input(case_id)
    resolve_case(task_input)
    hub = task_input.wx_agent_hub
    if hub is None:
        pytest.skip(f'{case_id}: no wx_agent_hub block')
    assert hub is not None  # 供 ty 收窄

    for key in ('pre_installed', 'market_available'):
        value = hub.get(key, [])
        assert isinstance(value, list), f'{case_id}: wx_agent_hub.{key} must be a list'
        for item in value:
            assert isinstance(item, str), f'{case_id}: wx_agent_hub.{key} entry must be a str: {item!r}'
            assert item, f'{case_id}: wx_agent_hub.{key} entry must be non-empty'


@pytest.mark.parametrize('case_id', ALL_CASE_IDS)
def test_case_metadata_when_present(case_id: str) -> None:
    """若 case 携带 ``metadata`` ，须格式良好；否则跳过。

    此处刻意不将 metadata 设为强制项——若干遗留
    ``case.yaml`` 仍缺少该 block 。 CLI ``aidata.py validate``
    会执行更严格的规则；本测试仅断言 *如果* 存在
    该 block ，则能解析为一个类型化对象。
    """
    task_input = _build_input(case_id)
    resolve_case(task_input)
    meta = task_input.case_metadata
    if meta is None:
        pytest.skip(f'{case_id}: no metadata block')
    assert meta is not None  # 供 ty 收窄

    created_at = meta.created_at or ''
    assert DATE_RE.match(created_at), f'{case_id}: metadata.created_at must be YYYY-MM-DD (got {meta.created_at!r})'


@pytest.mark.parametrize('case_id', ALL_CASE_IDS)
def test_rubric_sqlite_path_nonempty_when_declared(case_id: str) -> None:
    """若声明了 ``rubric_sqlite_path`` ，它是非空字符串。

    缺失是合法的（ rubric.py 中的 fail-closed 行为）；空值
    则是 case 作者 bug 。
    """
    task_input = _build_input(case_id)
    resolve_case(task_input)

    case_dir = CASES_ROOT / case_id
    manifest_file = case_dir / 'task.yaml'
    if not manifest_file.exists():
        manifest_file = case_dir / 'case.yaml'
    raw = yaml.safe_load(manifest_file.read_text(encoding='utf-8')) or {}

    if 'rubric_sqlite_path' not in raw:
        pytest.skip(f'{case_id}: no rubric_sqlite_path declared')

    assert task_input.rubric_sqlite_path.strip(), f'{case_id}: rubric_sqlite_path declared but resolves to empty string'


@pytest.mark.parametrize('case_id', ALL_CASE_IDS)
def test_rubric_yaml_valid(case_id: str) -> None:
    """存在时， ``rubric.yaml`` 是语法上合法的 YAML 。

    形状 / DSL 版本刻意 **不** 强制——那是 rubric.py 的
    职责。 我们此处只想按 case 抓出普通拼写错误
    （引号不匹配、重复键等）。
    """
    case_dir = CASES_ROOT / case_id
    rubric_file = case_dir / 'rubric.yaml'
    if not rubric_file.exists():
        pytest.skip(f'{case_id}: no rubric.yaml to validate')

    raw = yaml.safe_load(rubric_file.read_text(encoding='utf-8'))
    assert isinstance(raw, dict), f'{case_id}: rubric.yaml must be a mapping'
    checks = raw.get('checks')
    assert isinstance(checks, list), f'{case_id}: rubric.yaml.checks must be a list: got {type(checks).__name__}'
    assert checks, f'{case_id}: rubric.yaml.checks must be non-empty'


# ---------------------------------------------------------------------------
# 元测试：确保发现过程真的找到了内容
# ---------------------------------------------------------------------------


def test_discovery_finds_cases(all_case_ids: list[str]) -> None:
    """对 case 发现的软性检查。

    上游 nexus aidata 扩展有意不携带 case 数据；下游
    消费方将 ``AIDATA_CASES_ROOT`` 指向自己的 ``cases/`` 树。
    我们希望在配置根目录下发现意外返回 ``[]`` 时得到清晰
    信号，但当环境未配置时不希望硬失败（例如
    有人在刚克隆的上游检出中运行 ``pytest`` ）。
    """
    if not CASES_ROOT.is_dir():
        pytest.skip(
            f'No case tree at {CASES_ROOT} — set AIDATA_CASES_ROOT to point '
            f'at a downstream cases/ directory to run static / dynamic tests '
            f'against actual cases.'
        )
    assert all_case_ids, (
        f'CASES_ROOT={CASES_ROOT} exists but has no case dirs '
        '(no `task.yaml` / `case.yaml` found). If this is a downstream tree, '
        'check the path; if you only want to test the engine wiring, '
        'unset AIDATA_CASES_ROOT to fall back to the empty in-repo default.'
    )
