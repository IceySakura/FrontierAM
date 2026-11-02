"""沙箱侧路径常量 + 镜像内置二进制位置。

此处所有路径都是沙箱运行时（gongfeng 容器）内的路径，
而非宿主机上的路径。 两层布局：

- ``AGENT_DESKTOP``  是 agent 的 shell MCP 唯一可以到达的区域。
  它是任务上传原始 ``context_files`` 以及 agent 被
  期望投放自身交付物（ ``output/`` ）的地方。 skills 符号链接
  也在此处物化，因此 ``/workspace/skills/...`` 从 desktop
  可见。

- ``INTERNAL_ROOT`` 是框架的私有 scratch 区域。 Mock MCP
  服务器脚本、持久化的 mock 状态和每个 case 的 bootstrap 资产都
  存放于此。 shell MCP 的 ``ALLOWED_DIR``
  指向 desktop，因此 agent 无法 cat 内部 JSON 文件并
  绕过预期的 MCP 抽象（WooCommerce REST API、tencent-docs、
  weiyun 等）。

镜像内置的 bin（ ``MCP_PY`` / ``PDF_TOOLS_BIN`` / ``SHELL_PATH_IN`` ）
描述了镜像放置其 Python 解释器、pdf-tools-mcp
辅助工具的位置，以及 shell MCP 看到的 PATH。 完整映射位于
下方的 ``_IMAGE_LAYOUTS`` 中，由 ``resolve_image_paths`` 查询；
模块级默认值面向 ``aio-agent-rc`` ，harness 可以在 Python
导入此文件之前通过设置 ``NEXUS_AIDATA_MAIN_IMAGE`` 按运行覆盖。
每个 case 的覆盖通过
``mcp_catalog.build_specs(image=...)`` 进行，无需环境变量。

``PDF_TOOLS_BIN = ''`` （空字符串）表示镜像没有 pdf-tools-mcp
二进制—— ``mcp_catalog.build_specs`` 随后会跳过 pdf-tools 规格，使 MCP
加载在没有它的情况下也能成功。 Agent 仍然可以通过 shell MCP 使用
``pdftotext`` / ``pypdf`` /
``pdfplumber`` / markitdown / tesseract。
"""

from __future__ import annotations

import os

from pathlib import Path


WORKSPACE_ROOT = '/workspace'

# --- agent 可见表面 --------------------------------------------------

AGENT_DESKTOP = f'{WORKSPACE_ROOT}/desktop'
AGENT_INPUTS_DIR = f'{AGENT_DESKTOP}/inputs'
AGENT_OUTPUT_DIR = f'{AGENT_DESKTOP}/output'

# --- 框架私有（对 agent 隐藏） ----------------------------------

INTERNAL_ROOT = f'{WORKSPACE_ROOT}/.aidata'
SERVERS_DIR_IN = f'{INTERNAL_ROOT}/servers'
STATE_DIR_IN = f'{INTERNAL_ROOT}/state'
CASE_DIR_IN = f'{INTERNAL_ROOT}/case'

# v4.2 测试协议路径。 `TESTS_DIR_IN` 有意对 agent 可见，以便
# shell MCP （ALLOWED_DIR=AGENT_DESKTOP）可以调用 `bash /workspace/tests/...` 。
# junit 报告路径是 *run-test.sh 被要求存放其
# 输出的位置* ，无论 case 作者内部使用哪个框架。
TESTS_DIR_IN = f'{WORKSPACE_ROOT}/tests'
JUNIT_REPORT_PATH_IN = f'{TESTS_DIR_IN}/report.junit.xml'
# `SOLUTION_DIR_IN` 是 agent 不可见的、面向审阅者的参考
# 工件（golden_rollout.json / solve.py / golden.patch / NOTES.md）之家。 位于
# INTERNAL_ROOT 之下，因此 agent 的 shell MCP 无法 `cat` 到 oracle。
SOLUTION_DIR_IN = f'{INTERNAL_ROOT}/solution'

# 用于 MockServer 风格状态的每个 mock 根目录（MockDB 布局）。 位于
# 框架私有的 STATE_DIR_IN 之下，因此每个在沙箱内运行的 mock
# （无论是 bundled-local 还是来自 MockServer 镜像）都获得一个
# 隔离的子目录，agent 的 shell MCP 无法到达。 与
# MockServer 的 ``MOCK_SERVER_DATA_DIR`` env 契约在线对齐：每个
# 使用 ``mock_data_dir(<name>)`` 的 mock 最终都位于
# ``{MOCK_SERVER_DATA_DIR_IN}/<name>/`` 之下。
MOCK_SERVER_DATA_DIR_IN = f'{STATE_DIR_IN}/mock_server'

# --- LD_PRELOAD confine kit -------------------------------------------------
#
# 通过 LD_PRELOAD 强制执行的用户空间黑名单文件系统沙箱。 该 kit
# （ ``libconfine.c`` + ``compile.sh`` + README）随引擎发布，
# 并在任务开始时上传到 ``CONFINE_DIR_IN`` ； ``compile.sh`` 随后
# 在原地针对运行时镜像的确切
# glibc 构建 ``libconfine.so`` ，因此任何基于 glibc 的镜像都无需修改 Dockerfile 即可工作。
#
# 完整设计 + 绕过矩阵参见 ``engine/confine/README.md`` 。
CONFINE_DIR_IN = f'{INTERNAL_ROOT}/confine'
CONFINE_SO_IN = f'{CONFINE_DIR_IN}/libconfine.so'
CONFINE_SH_IN = f'{CONFINE_DIR_IN}/compile.sh'

# shell MCP 的 bash 子进程的默认 deny 前缀。 顺序无关；
# 匹配是冒号分隔的字符串前缀，在 ``libconfine.c::is_denied_raw`` 内部
# 进行 ``/`` 边界检查。
#
# 该列表通过遍历 ``cases/`` 下每个已发布的 case
# 并将其引用的绝对路径分区为 " agent 必须
# 读取 " 与 " agent 必须不读取 " 得出。 每当新 case 添加一个私有
# 状态目录时，将其追加到此处并在基线和 enforce 模式下运行
# ``tests/test_sandbox_confinement.py`` 以保持契约通过。
CONFINE_DENY_DEFAULT = ':'.join([
    INTERNAL_ROOT,  # /workspace/.aidata —— 引擎私有
    '/var/lib/mock-server',  # MockServer 状态 + sqlite oracle
    '/var/lib/jd_backend',  # channel_inventory_ppt 的 woocommerce 商店
    '/opt/mock-server/src',  # MockServer 源码树（oracle 答案）
    '/opt/mock-server/vendor',  # MockServer vendored 依赖
    # Nexus 运行时 / 控制平面源码树。 运行时服务器（在
    # 运行时端口的 uvicorn）从此处运行并暴露 invoke/call_tool/
    # call_executor RPC；让源码可读会让 agent 得以 grep
    # 工具注册表 + RPC 契约，并尝试通过直接调用 executor 绕过 MCP。
    # 对 agent 的 bash 而言可以安全 deny：PATH 使用 /nix/runtime/
    # nix-env（一个兄弟目录，未匹配），且 agent 自己的 python 位于
    # /opt/aio-deps 之下——两者都不位于 /nix/runtime/nexus 之下。 正在运行的
    # 服务器进程不受影响（它不在 shell MCP 的 LD_PRELOAD 之下）。
    '/nix/runtime/nexus',  # Nexus 运行时源码 + 控制平面 RPC
])

# shell MCP 的 bash 子进程的网络 deny 主机。 逗号分隔的主机
# 后缀；驱动两个独立的层（都以此列表为键）：
#
#   1. libconfine.c getaddrinfo() hook——为这些主机返回 EAI_NONAME，因此
#      任何动态链接工具（curl / wget / python / node）都无法
#      解析它们。 只有当 confine .so 被 LD_PRELOAD 时才生效。
#   2. shell_server.py no_proxy 合并——将这些主机追加到继承的
#      no_proxy，因此它们被强制 DIRECT（不经沙箱代理），这
#      正是使 libconfine getaddrinfo() 阻塞真正生效的原因。 若无此项，
#      裸 ``curl skillhub.cn`` 会通过代理隧道，代理会解析
#      代理主机（而非 skillhub.cn），因此 getaddrinfo hook 永远看不到它。
#
# 二者共同阻止公共 SkillHub 商店：任务必须使用沙箱内的
# mcp-skillhub 镜像。 匹配基于后缀，因此 ``skillhub.cn`` 也覆盖
# ``www.`` / ``api.`` 子域名。 可通过 AIDATA_CONFINE_NET_DENY 按运行覆盖。
#
# 并非对抗性硬墙：直接 IP 连接、手动重设的
# http_proxy、静态链接的二进制或裸系统调用可绕过两个层
# （沙箱内没有 seccomp / NET_ADMIN）。 真正的墙需要在代理本身上做
# egress 策略。 参见 engine/confine/README.md §Network deny。
CONFINE_NET_DENY_DEFAULT = ','.join([
    'skillhub.cn',  # 公共 SkillHub 商店（通过后缀匹配覆盖 www./api.）
    'skillhub-1388575217.cos.ap-guangzhou.myqcloud.com',  # 其 COS zip 桶
])

# 向后兼容别名：某些地方仍称 ``TASK_ROOT`` 。 保持它指向
# 内部根，以便引擎或 case 中的任何内容都无需更改。
TASK_ROOT = INTERNAL_ROOT

# --- 镜像内置二进制 ------------------------------------------------
#
# 目前发布两种镜像布局，case 可以通过在其 task.yaml / case.yaml 中
# 设置 ``runtime.main.image`` 来选择使用哪一种：
#
#   aio-agent-rc （默认）        —— nexus-base + /opt/aio-deps/.venv
#   aidata-aio-agent:2.x        —— 遗留，全部位于 /opt/mcp-benchmarks/ 下
#
# 下方的映射是每种布局将 Python / pdf-tools / shell 的 PATH
# 放置位置的唯一真相来源。 ``resolve_image_paths``
# 采用前缀匹配，因此版本标签（ ``:v0.1.1-dev`` 、 ``:2.3`` …）都会落在
# 正确的布局上，无需触碰代码。 需要此处未列出布局的 case
# 必须显式添加，或继续使用默认。
#
# 未声明的 case 会继承 ``runtime.template.yaml`` →
# ``runtime_configs.main.image`` ；该镜像名会被 harness 的 run-case.sh
# 作为 ``NEXUS_AIDATA_MAIN_IMAGE`` 转发到本模块，以便
# 解析出的 MCP_PY 与 runner 实际启动的镜像相匹配。

_LEGACY_MCP_BENCH_PATHS: dict[str, str] = {
    'mcp_py': '/opt/mcp-benchmarks/venv/bin/python',
    'pdf_tools_bin': '/opt/mcp-benchmarks/py312/bin/pdf-tools-mcp',
    'shell_path': (
        '/opt/mcp-benchmarks/venv/bin:'
        '/opt/mcp-benchmarks/py311/bin:'
        '/opt/mcp-benchmarks/py312/bin:'
        '/opt/mcp-benchmarks/py313/bin:'
        '/opt/mcp-benchmarks/npm/node_modules/.bin:'
        '/nix/runtime/nix-env/bin:'
        '/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin'
    ),
}

_IMAGE_LAYOUTS: dict[str, dict[str, str]] = {
    'aio-agent-rc': {
        'mcp_py': '/opt/aio-deps/.venv/bin/python',
        'pdf_tools_bin': '/opt/aio-deps/.venv/bin/pdf-tools-mcp',
        'shell_path': (
            '/opt/aio-deps/.venv/bin:'
            '/nix/runtime/nix-env/bin:'
            '/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin'
        ),
    },
    # mcp-clean（deploy-aio-work/Dockerfile.mcp-clean）—— 8 个 MCP（无 k8s），
    # 每个位于自己的 /opt/venv/<svc>/ uv venv 中。 /opt/venv/fs 装有 `mcp`
    # SDK（FastMCP + TransportSecuritySettings），因此 shell_server
    # 从中启动没问题。 该镜像不发布 pdf-tools-mcp；此处用空
    # 字符串使 build_specs 完全跳过 pdf-tools 规格。
    # 列在 work_mcp_mirror 之前，以便对
    # ``work_mcp_mirror:mcp-clean-v*`` 的子串匹配解析到此布局，而非
    # 遗留的 /opt/mcp-benchmarks（mcp-clean 中并不存在）。
    'mcp-clean': {
        'mcp_py': '/opt/venv/fs/bin/python',
        'pdf_tools_bin': '',
        'shell_path': (
            '/opt/venv/fs/bin:'
            '/opt/venv/db/bin:'
            '/opt/venv/git/bin:'
            '/opt/venv/wecom/bin:'
            '/opt/venv/wiki/bin:'
            '/opt/venv/monitor/bin:'
            '/opt/venv/loki/bin:'
            '/opt/venv/datainfra/bin:'
            '/nix/runtime/nix-env/bin:'
            '/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin'
        ),
    },
    # mcp-aio 标签发布在同一个 work_mcp_mirror 仓库之下，
    # 但采用较新的 /opt/venv/<svc> 布局。 保持此条目位于
    # 下方通用 work_mcp_mirror 回退之前。
    'mcp-aio': {
        'mcp_py': '/opt/venv/fs/bin/python',
        'pdf_tools_bin': '',
        'shell_path': (
            '/opt/venv/fs/bin:'
            '/opt/venv/db/bin:'
            '/opt/venv/git/bin:'
            '/opt/venv/wecom/bin:'
            '/opt/venv/wiki/bin:'
            '/opt/venv/monitor/bin:'
            '/opt/venv/loki/bin:'
            '/opt/venv/datainfra/bin:'
            '/nix/runtime/nix-env/bin:'
            '/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin'
        ),
    },
    # 遗留的 aidata-aio-agent 与 wecom-forked work_mcp_mirror 镜像
    # 共享同一 /opt/mcp-benchmarks/* 布局（后者构建自
    # aidata-aio-agent:2.x 并在其上添加 /opt/wecom-env）。 保持
    # 一份路径以避免漂移。
    'aidata-aio-agent:2': dict(_LEGACY_MCP_BENCH_PATHS),
    'work_mcp_mirror': dict(_LEGACY_MCP_BENCH_PATHS),
}


def resolve_image_paths(image: str) -> dict[str, str]:
    """查找与 ``image`` 关联的路径集合。

    匹配基于 ``<repo>/<name>[:tag]`` 的子串前缀，因此
    从 ``:v0.1-dev`` → ``:v0.1.1-dev`` 或 ``:2.1`` → ``:2.3`` 的升级
    无需更新此表。 若为未知布局则抛出 ``ValueError`` ，
    这样 case 作者能得到清晰信号，而非静默地
    回退到指向错误二进制的路径。
    """
    if not image:
        raise ValueError('resolve_image_paths(image=""): empty image name')
    for key, paths in _IMAGE_LAYOUTS.items():
        if key in image:
            return paths
    known = ', '.join(sorted(_IMAGE_LAYOUTS))
    raise ValueError(
        f'unknown image layout {image!r}; expected one of: {known}. '
        f'Add a new entry to engine.paths._IMAGE_LAYOUTS if you introduced a new image.'
    )


# 模块级默认值。 ``NEXUS_AIDATA_MAIN_IMAGE`` 让 harness /
# tests / CLI 预先声明默认 ``main`` 运行时将
# 使用哪个镜像，因此 ``MCP_PY`` 在导入时已经指向正确的布局。
# 每个 case 的覆盖仍在
# ``mcp_catalog.build_specs`` 中的规格构建时生效——此顶级值仅是
# 尚未改造为接受显式 image 的调用点的默认值。

_DEFAULT_IMAGE = os.getenv('NEXUS_AIDATA_MAIN_IMAGE', 'aio-agent-rc')
_DEFAULT_PATHS = resolve_image_paths(_DEFAULT_IMAGE)

MCP_PY = os.getenv('NEXUS_AIDATA_MCP_PY') or _DEFAULT_PATHS['mcp_py']
PDF_TOOLS_BIN = os.getenv('NEXUS_AIDATA_PDF_TOOLS_BIN', _DEFAULT_PATHS['pdf_tools_bin'])
SHELL_PATH_IN = os.getenv('NEXUS_AIDATA_SHELL_PATH') or _DEFAULT_PATHS['shell_path']

# MockServer 源码树，通过
# ``COPY --from=mocks /opt/mock-server/src/mock_server /opt/mock-server`` 注入到 aidata-aio-agent 镜像中（参见
# https://git.woa.com/HunYuanRLEnv/MockServer docs/integration.md）。 想要生成 MockServer 打包 mock 的
# 引擎规格构建器使用路径
# ``{MOCK_SERVER_ROOT_IN}/<group>/<name>/server.py`` 。
MOCK_SERVER_ROOT_IN = '/opt/mock-server'

# 宿主侧的 mock-server 源码目录 —— 现在位于兄弟包 ``mock_servers/``
# （Commit 1 重构时从 ``engine/servers/`` 迁出； ``engine/servers/`` 现在只
# 是一个 3 文件的兼容 shim ，实体脚本住在 ``mock_servers/`` 下）。
# 引擎在任务开始时把此目录上传到 SERVERS_DIR_IN ，以便沙箱可以将它们
# 作为 stdio MCP 服务器启动。 此路径是遗留的 " local " 打包 mock 之家；
# 新 mock 应添加到 MockServer 仓库（HunYuanRLEnv/MockServer）并通过
# ``MOCK_SERVER_ROOT_IN`` 引用 —— 参见 ``mcp_catalog._BUNDLED_MOCKS`` 。
SERVERS_DIR = Path(__file__).resolve().parents[1] / 'mock_servers'

# 宿主侧的 LD_PRELOAD confine kit —— Commit 1 重构后位于 ``mock_servers/confine/`` 。
# 在任务开始时上传到 CONFINE_DIR_IN（参见 engine/task.py 步骤 2.5）；
# 运行时随后运行 CONFINE_SH_IN 以针对镜像的 glibc 生产 CONFINE_SO_IN。
CONFINE_DIR = Path(__file__).resolve().parents[1] / 'mock_servers' / 'confine'

# 宿主侧的 ``cases/`` 树根。 默认为 ``<package>/cases``
# （遗留的仓内位置）。 环境变量覆盖使得 aidata 的
# 上游合并可以在不打包 case 数据的情况下工作：开发者
# 将 ``AIDATA_CASES_ROOT`` 指向其下游 Tasks 仓库 / 子模块，
# 从而让 :func:`engine.task.run_aidata` 与测试套件都看到相同的
# case bundle。
CASES_DIR = Path(os.getenv('AIDATA_CASES_ROOT') or (Path(__file__).resolve().parents[1] / 'cases'))


__all__ = [
    'AGENT_DESKTOP',
    'AGENT_INPUTS_DIR',
    'AGENT_OUTPUT_DIR',
    'CASE_DIR_IN',
    'CASES_DIR',
    'CONFINE_DENY_DEFAULT',
    'CONFINE_NET_DENY_DEFAULT',
    'CONFINE_DIR',
    'CONFINE_DIR_IN',
    'CONFINE_SH_IN',
    'CONFINE_SO_IN',
    'INTERNAL_ROOT',
    'JUNIT_REPORT_PATH_IN',
    'MCP_PY',
    'MOCK_SERVER_DATA_DIR_IN',
    'MOCK_SERVER_ROOT_IN',
    'PDF_TOOLS_BIN',
    'SERVERS_DIR',
    'SERVERS_DIR_IN',
    'SHELL_PATH_IN',
    'SOLUTION_DIR_IN',
    'STATE_DIR_IN',
    'TASK_ROOT',
    'TESTS_DIR_IN',
    'WORKSPACE_ROOT',
    'resolve_image_paths',
]
