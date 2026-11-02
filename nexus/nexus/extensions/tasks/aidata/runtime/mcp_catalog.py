"""aidata 的 MCP catalog 构建器。

``build_specs`` 组装 ``ctx.load_mcp_tools`` 消费的列表。
catalog 有三个固定条目：

1. **pdf-tools** —— 内置镜像的 PDF 阅读器 （无状态， 可选）。
2. **shell** —— 本地 ``shell_server.py``， 是 ``bash -c`` 的薄封装。
3. **wx_agent_hub 展开** —— 当用例声明 ``wx_agent_hub`` 时。

所有办公 mock （wechat、 email、 schedule 等） 由 mock-server sidecar 容器
提供， 并按用例通过 HTTP MCP 传输经 ``extra_mcp_servers`` 注入。 引擎不再
打包或上传 mock 脚本。

``mock_env`` 生成 sidecar mock-server 容器所需的环境变量 （LLM 路由、 用户
prompt、 上下文文件路径）。 由 ``services.py::setup_services`` 注入到
服务启动过程。
"""

from __future__ import annotations

import os

from typing import Any

from .paths import (
    AGENT_DESKTOP,
    CONFINE_DENY_DEFAULT,
    CONFINE_NET_DENY_DEFAULT,
    CONFINE_SO_IN,
    MCP_PY,
    MOCK_SERVER_ROOT_IN,
    PDF_TOOLS_BIN,
    SERVERS_DIR_IN,
    SHELL_PATH_IN,
    resolve_image_paths,
)


# ---------------------------------------------------------------------------
# LLM 路由 （由 mock 与主机侧 rubric judge 共享）
# ---------------------------------------------------------------------------


def resolve_mock_llm_creds() -> tuple[str, str, str]:
    """解析 mock 服务器和 rubric judge 使用的 (base_url, api_key, model)。"""
    base_url = (
        os.getenv('NEXUS_MOCK_LLM_BASE_URL')
        or os.getenv('SEARCH_OPENAI_BASE_URL')
        or 'http://llm-api.model-eval.woa.com'
    )
    api_key = (
        os.getenv('NEXUS_MOCK_LLM_API_KEY') or os.getenv('LLM_API_KEY') or os.getenv('SEARCH_OPENAI_API_KEY') or ''
    )
    model = (
        os.getenv('NEXUS_MOCK_LLM_MODEL')
        or os.getenv('SEARCH_OPENAI_MODEL')
        or 'api_naci_default_gemini-3.1-pro-preview'
    )
    return base_url, api_key, model


def resolve_judge_model() -> str:
    """Judge / rubric-builder 模型： 优先显式覆盖， 然后回退。"""
    return (
        os.getenv('NEXUS_JUDGE_MODEL')
        or os.getenv('JUDGE_OPENAI_MODEL')
        or os.getenv('NEXUS_MOCK_LLM_MODEL')
        or os.getenv('SEARCH_OPENAI_MODEL')
        or 'api_naci_default_gemini-3.1-pro-preview'
    )


def resolve_judge_creds() -> tuple[str, str, str]:
    """解析 rubric LLM judge 使用的 (base_url, api_key, model)。

    独立于 ``SEARCH_OPENAI_*`` / ``NEXUS_MOCK_LLM_*`` 尊重
    ``JUDGE_OPENAI_*``， 让运维可以把 agent 与 judge 路由到不同 LLM
    端点 （例如 agent 用便宜快速的模型， judge 用更严格更聪明的模型）。
    未设置 judge 专属覆盖时回退到 mock-llm 凭据 —— 单 LLM 布置无需
    额外配置。

    历史 bug 防范： ``rubric._judge_once`` 曾经调用
    ``setdefault('NEXUS_MOCK_LLM_BASE_URL', JUDGE_OPENAI_BASE_URL)``，
    但 ``normalize_host_env`` 早已从 ``SEARCH_OPENAI_BASE_URL`` 填充了
    ``NEXUS_MOCK_LLM_BASE_URL``， 因此 setdefault 变成 no-op， 而
    ``JUDGE_OPENAI_BASE_URL`` 悄悄失效。 将 judge 路由拆到独立的
    ``NEXUS_JUDGE_BASE_URL`` / ``NEXUS_JUDGE_API_KEY`` 命名空间， 从此
    永久跳出这种冲突。
    """
    mock_base, mock_key, _mock_model = resolve_mock_llm_creds()
    base_url = os.getenv('JUDGE_OPENAI_BASE_URL') or mock_base
    api_key = os.getenv('JUDGE_OPENAI_API_KEY') or mock_key
    model = resolve_judge_model()
    return base_url, api_key, model


def normalize_host_env() -> None:
    """把解析出的 LLM 凭据推到 ``os.environ``， 使主机侧模块 （rubric judge）
    看到与 mock 子进程相同的路由。

    注入两个独立的命名空间：

    - ``NEXUS_MOCK_LLM_*`` —— agent 侧 / mock-server 路由 （已有）。
    - ``NEXUS_JUDGE_*`` —— rubric-judge 路由。 独立命名空间使得
      ``JUDGE_OPENAI_BASE_URL`` ≠ ``SEARCH_OPENAI_BASE_URL`` 在 rubric
      层实际生效 （``llm_chat`` 在做 judge 调用时读取
      ``NEXUS_JUDGE_BASE_URL`` / ``NEXUS_JUDGE_API_KEY`` /
      ``NEXUS_JUDGE_MODEL``； 详见
      :func:`engine.servers.llm_client.llm_chat`）。
    """
    base_url, api_key, model = resolve_mock_llm_creds()
    judge_base, judge_key, judge_model = resolve_judge_creds()
    os.environ.setdefault('NEXUS_MOCK_LLM_BASE_URL', base_url)
    if api_key:
        os.environ.setdefault('NEXUS_MOCK_LLM_API_KEY', api_key)
    os.environ.setdefault('NEXUS_MOCK_LLM_MODEL', model)
    os.environ.setdefault('NEXUS_JUDGE_BASE_URL', judge_base)
    if judge_key:
        os.environ.setdefault('NEXUS_JUDGE_API_KEY', judge_key)
    os.environ.setdefault('NEXUS_JUDGE_MODEL', judge_model)


def fake_now_epoch(mock_now_iso: str | None) -> str | None:
    """将 ``MOCK_SERVER_NOW`` 的 ISO-8601 字符串转换为 unix 秒级时间戳文本。

    与 MockServer ``_common/clock.py`` 的解析行为镜像 （``Z`` → ``+00:00``
    然后 ``datetime.fromisoformat``）。 未设 / 空 / 无法解析时返回 ``None``，
    调用方可以干净地作门控 —— 为 None 时不注入 ``AIDATA_FAKE_NOW``，
    confine ``.so`` 也不动系统时钟。

    naive 值 （无 tzinfo） 被解释为 UTC； 每个上线用例都会锚定一个带
    ``+08:00`` 的值， 因此该分支只用于防御输入畸形的情形。
    """
    raw = (mock_now_iso or '').strip()
    if not raw:
        return None
    from datetime import datetime, timezone

    try:
        dt = datetime.fromisoformat(raw.replace('Z', '+00:00'))
    except ValueError:
        return None
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return str(int(dt.timestamp()))


def mock_env(user_prompt: str) -> dict[str, str]:
    """注入到 mock-server sidecar 启动的环境变量。

    提供 LLM 路由、 用户 prompt 上下文和 proxy 关闭配置。
    由 ``services.py::setup_services`` 注入到每个服务的
    startup.env， 使 sidecar mock 能生成基于 LLM 的内容。
    """
    base_url, api_key, model = resolve_mock_llm_creds()
    judge_model = resolve_judge_model()
    return {
        'PYTHONUNBUFFERED': '1',
        'NEXUS_MOCK_USER_PROMPT': user_prompt,
        'NEXUS_MOCK_LLM_BASE_URL': base_url,
        'NEXUS_MOCK_LLM_API_KEY': api_key,
        'NEXUS_MOCK_LLM_MODEL': model,
        'NEXUS_JUDGE_MODEL': judge_model,
        'NO_PROXY': '.woa.com,localhost,127.0.0.1',
        'SERPER_APP_ID': os.getenv('SEARCH_STATEFUL_USER', ''),
        'SERPER_APP_KEY': os.getenv('SEARCH_STATEFUL_APIKEY', ''),
        'JINA_APP_ID': os.getenv('DEFAULT_JINA_APP_ID', ''),
        'JINA_APP_KEY': os.getenv('DEFAULT_JINA_APP_KEY', ''),
    }


# ---------------------------------------------------------------------------
# WeChat Agent Hub 展开
# ---------------------------------------------------------------------------


_WX_AGENT_HUB_BASE_MOCKS: tuple[str, ...] = ('wx_agent_hub', 'wx_pay')


def _expand_wx_agent_hub_specs(
    wx_agent_hub: dict[str, Any],
    common_env: dict[str, str],
    mcp_py: str,
) -> list[dict[str, Any]]:
    """把 ``case.yaml::wx_agent_hub`` 展开为具体 MCP 规格。"""
    pre_installed = list(wx_agent_hub.get('pre_installed') or [])
    market_available = list(wx_agent_hub.get('market_available') or [])

    providers: list[str] = []
    seen: set[str] = set(_WX_AGENT_HUB_BASE_MOCKS)
    for name in (*pre_installed, *market_available):
        if name in seen:
            continue
        seen.add(name)
        providers.append(name)

    specs: list[dict[str, Any]] = []
    for mock_name in (*_WX_AGENT_HUB_BASE_MOCKS, *providers):
        specs.append({
            'name': mock_name,
            'runtime_name': 'main',
            'client': {
                'transport': 'stdio',
                'command': mcp_py,
                'args': [f'{MOCK_SERVER_ROOT_IN}/life/{mock_name}/server.py'],
                'env': dict(common_env),
            },
        })
    return specs


# ---------------------------------------------------------------------------
# MCP 规格构建器
# ---------------------------------------------------------------------------


def build_specs(
    user_prompt: str,
    extra_servers: list[dict[str, Any]] | None = None,
    service_env: dict[str, str] | None = None,
    wx_agent_hub: dict[str, Any] | None = None,
    main_image: str | None = None,
    mock_now: str | None = None,
) -> list[dict[str, Any]]:
    """构建 ``ctx.load_mcp_tools`` 消费的 MCP 规格列表。

    顺序： pdf-tools -> shell -> wx_agent_hub 展开 -> 每用例的 extras。
    办公 mock 不再在此打包 —— 它们通过每用例的 ``extra_mcp_servers``
    条目提供， 指向 mock-server sidecar。

    ``main_image`` 决定使用哪种镜像布局 （见 ``paths._IMAGE_LAYOUTS``）
    来提供 stdio-spawned 的 MCP 二进制 （shell / pdf-tools /
    wx_agent_hub）。 默认使用 ``paths`` 中跟随
    ``NEXUS_AIDATA_MAIN_IMAGE`` / ``runtime.template.yaml`` 的模块级
    值。 每用例的覆盖通过 ``resolve_case`` 之后的 ``task.py`` 传到这里。
    """
    if main_image:
        paths = resolve_image_paths(main_image)
        mcp_py = paths['mcp_py']
        pdf_tools_bin = paths['pdf_tools_bin']
        shell_path = paths['shell_path']
    else:
        mcp_py, pdf_tools_bin, shell_path = MCP_PY, PDF_TOOLS_BIN, SHELL_PATH_IN

    common_env = dict(mock_env(user_prompt))
    if service_env:
        common_env.update(service_env)
    specs: list[dict[str, Any]] = []

    if pdf_tools_bin:
        specs.append({
            'name': 'pdf-tools',
            'runtime_name': 'main',
            'client': {'transport': 'stdio', 'command': pdf_tools_bin, 'args': [], 'env': {}},
        })

    # Confine env hooks —— 由 engine/servers/shell_server.py 读取。 .so
    # 路径 + deny 列表在引擎导入时通过 paths.py 烘焙进来； mode 支持
    # 环境覆盖， 让 dev / CI 无需重建镜像即可在 enforce <-> warn 之间切换
    # 。 见 engine/confine/README.md §Modes。
    confine_mode = os.getenv('AIDATA_CONFINE_MODE', 'enforce')
    # 网络 deny 主机列表 （逗号分隔）。 同时驱动 libconfine 的 getaddrinfo()
    # 阻断和 shell_server 的 no_proxy force-direct 合并 —— 见
    # paths.CONFINE_NET_DENY_DEFAULT。 有意不在此处注入到 no_proxy 里：
    # spec env 会覆盖运行时容器继承的 no_proxy （.woa.com 等）， 因此
    # shell_server 在运行时进行合并。
    confine_net_deny = os.getenv('AIDATA_CONFINE_NET_DENY', CONFINE_NET_DENY_DEFAULT)

    shell_env = {
        'ALLOWED_DIR': AGENT_DESKTOP,
        'COMMAND_TIMEOUT': '180',
        'PYTHONUNBUFFERED': '1',
        'PATH': shell_path,
        'AIDATA_CONFINE_SO': CONFINE_SO_IN,
        'AIDATA_CONFINE_DENY': CONFINE_DENY_DEFAULT,
        'AIDATA_CONFINE_NET_DENY': confine_net_deny,
        'AIDATA_CONFINE_MODE': confine_mode,
    }

    # Faketime 锚点 —— 由 libconfine.c 读取， 让 agent 的 `bash -c date`
    # （以及每一个动态链接工具的时钟） 报告 mock 的 MOCK_SERVER_NOW，
    # 而不是容器的系统时钟。 门控前提是用例确实锚定了冻结时钟：
    # 当 ``mock_now`` 为 None/无法解析时不添加该键， .so 也不动时间，
    # 因此没有冻结时钟的用例逐字节保持不变。 可通过每次运行的
    # AIDATA_FAKE_NOW 环境变量 （epoch 秒） 覆盖以做临时测试。
    resolved_fake_now = os.getenv('AIDATA_FAKE_NOW') or fake_now_epoch(mock_now)
    if resolved_fake_now:
        shell_env['AIDATA_FAKE_NOW'] = resolved_fake_now

    specs.append({
        'name': 'shell',
        'runtime_name': 'main',
        'client': {
            'transport': 'stdio',
            'command': mcp_py,
            'args': [f'{SERVERS_DIR_IN}/shell_server.py'],
            'env': shell_env,
        },
    })

    if wx_agent_hub is not None:
        specs.extend(_expand_wx_agent_hub_specs(wx_agent_hub, common_env, mcp_py))

    if extra_servers:
        override_names = {s['name'] for s in extra_servers if 'name' in s}
        specs = [s for s in specs if s.get('name') not in override_names]
        specs.extend(extra_servers)

    return specs


__all__ = [
    'build_specs',
    'fake_now_epoch',
    'mock_env',
    'normalize_host_env',
    'resolve_judge_creds',
    'resolve_judge_model',
    'resolve_mock_llm_creds',
]
