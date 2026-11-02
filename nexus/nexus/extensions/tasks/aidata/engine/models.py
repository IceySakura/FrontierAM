"""aidata task 的 Pydantic I/O 模型。

从编排器中拆分出来，让其它调用方（CLI 命令、可视化、集成测试）无需
拉入整个 engine 就能导入这些形状。
"""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field

from nexus.core import TaskRunInput, TaskRunOutput, WithAgentRunner, WithScore


class AidataCaseMetadata(BaseModel):
    """case 的维护元数据——上线时间、作者链和主题内容。 从
    ``cases/<id>/task.yaml::metadata`` 填充；由 ``engine.case_spec`` 解析
    并回显到 ``AidataTaskOutput``，供审计与分类使用。

    ``created_at`` 是仅有的必填字段——其余都是便利字段
    （``aidata.py list`` 列、CI 标签过滤）。 起源 / 迁移历史应写入 case
    自己的 ``README.md``（人类可读），不应放这里；本结构服务于 Harness
    的审计轨迹。
    """

    authors: list[str] = Field(default_factory=list)
    """历史作者列表（追加式）；未显式声明时默认为空列表。 """

    created_at: str = ''
    """case 首次提交日期，格式 ``YYYY-MM-DD``。 稳定值——不会随每次编辑
    变动。 格式由 ``aidata.py validate`` 强制。 """

    tags: list[str] = Field(default_factory=list)
    """自由格式分类标签（``wecom``、 ``pytest``、 ``demo`` 等）。 供 CI
    分组、搜索以及 ``list`` 输出列使用。 """


class AidataServiceCommand(BaseModel):
    """在 service 容器内运行的命令（startup / ready_check）。

    ``command`` / ``script`` 有且只有一个必须设置。

    - ``command``：内联 shell 片段，直接传给 ``runtime.run_command``。
      适合单行命令与短小段落。
    - ``script``：宿主机上相对于 case 目录的路径，指向 ``.sh`` /
      ``.py`` / 其它可执行文件。 引擎会将其上传到 service 容器的临时
      路径并执行——``.sh`` 走 ``bash``、 ``.py`` 走 ``python3``、其它扩展
      名直接执行（依赖其 shebang）。 适合任何写在 YAML 里会很难看的
      非平凡逻辑。
    """

    command: str | None = None
    script: str | None = None
    cwd: str | None = None
    env: dict[str, str] = Field(default_factory=dict)
    timeout_s: float = 120.0

    def __init__(self, **data):
        super().__init__(**data)
        if bool(self.command) == bool(self.script):
            raise ValueError(
                'AidataServiceCommand requires exactly one of `command` or `script` '
                f'(got command={self.command!r}, script={self.script!r})'
            )


class AidataServiceConnection(BaseModel):
    """描述该 service 暴露的一个具名 ``host:port`` 句柄。

    在 task 启动时（service 容器就绪后）解析，并作为占位符 / 环境
    变量注入下列位置：
      - ``user_prompt``（文本替换）
      - ``extra_mcp_servers[].client.{args,env}``（文本替换）
      - ``bootstrap_script``（通过 ``env_name`` 暴露为环境变量）
    """

    placeholder: str
    """在 user_prompt / MCP 规格里搜索并替换的字面串。
    约定：``__AIDATA_<NAME>_URL__``——足够独特，不会误匹配到 agent
    生成的文本。 """

    internal_port: int
    """容器侧端口；必须同时出现在 ``AidataService.exposed_ports`` 中。 """

    value_template: str = 'http://{host}:{port}'
    """格式串，可选包含 {host}、 {port}、 {base_url} 三种占位。 """

    env_name: str | None = None
    """若设置，此值也会作为 ``${env_name}`` 导出到 ``bootstrap_script``
    的环境变量中。 例如：env_name=JD_BACKEND_BASE_URL。 """


class AidataServiceFile(BaseModel):
    """启动前上传到 service 容器的 host 侧 case 资产。"""

    source: str
    """相对于宿主机上 case 目录的路径。 """

    target: str
    """service 容器内的绝对路径。 """


class AidataService(BaseModel):
    """与主运行时并行拉起的一个辅助 service 容器。

    每个 service 都是一个独立的 nexus 运行时（在 docker provider 上
    是独立 docker 容器 / 在 gongfeng 上是独立沙箱），通过
    ``ctx.get_runtime(runtime_name)`` 获取。 connections 声明主运行时
    访问该 service 时使用的公开 host:port 值。
    """

    name: str
    """人类可读的标签，例如 ``jd-backend``。 """

    runtime_name: str
    """必须区别于 ``'main'``、也区别于同一 case 中的其它任何 service；
    作为 ``ExecutionContext._runtimes`` 的 key 使用。 """

    image: str | None = None
    """容器镜像。 若为 None，gongfeng provider 会回退到 provider 级
    默认（即与 main 相同的 aidata-aio-agent 镜像），这对仅需运行该
    镜像里已有的 Python helper 的 service 而言是可用的。 """

    exposed_ports: list[int] = Field(default_factory=list)
    """要发布的内部端口；每个都会自动分配一个宿主端口。 """

    working_dir: str | None = None

    files: list[AidataServiceFile] = Field(default_factory=list)
    """启动前上传到该 service 容器的 host 侧 case 资产。 """

    startup: AidataServiceCommand | None = None
    """service 容器就绪后在其内部运行的命令（通常是
    ``nohup <server> & wait-for-healthz``）。 ``startup.command`` /
    ``startup.env`` 中出现的连接占位符 **不会** 被替换（service
    启动通常绑定到 0.0.0.0:<internal>，占位符在此毫无意义）。 """

    ready_check: AidataServiceCommand | None = None
    """在 service 容器内部轮询的命令，直到其退出码为 0 为止。 """

    ready_timeout_s: float = 120.0
    """内部 ready_check 与延迟端口映射各自使用的最长等待时间。 """

    ready_poll_interval_s: float = 2.0
    """内部 ready_check 与延迟端口映射的轮询间隔。 """

    connections: list[AidataServiceConnection] = Field(default_factory=list)

    state_mounts: dict[str, str] = Field(default_factory=dict)
    """映射 ``<mock_name> → <容器内目录>``。 agent 结束后，
    ``engine.state.download_state`` 会遍历每一项，从 **该 service** 容器
    （而非主运行时）拉取该目录，并把解析后的 state 合并到与 stdio
    版 mock 会填充的同一顶层 result key 下。

    Key 必须与 ``download_state`` 产出的切片 key 相符——目前只有
    ``'email'`` 已接线（sidecar 模式的 email mock 会在其镜像 VOLUME
    默认 ``/var/lib/mock-server/email/`` 下每次发送写入一份
    ``email_<ts>.json``）。 更多 mock 从 stdio
    （``engine/servers/<name>_mock.py``）迁移到 sidecar mock-server 镜像
    时，可按需在此新增条目。

    为空时是 no-op——把 state 存放在主容器 ``STATE_DIR_IN`` 里的 stdio
    case 会照旧工作。 """


class AidataTaskInput(TaskRunInput, WithAgentRunner):
    task_id: str = 'aidata_default'

    case_id: str = ''
    """指向 ``cases/<case_id>/case.yaml`` 的可选引用。 设置后，凡未在
    ``task_config`` 里显式提供的字段（user_prompt / context_files /
    timeouts）都会从 case 文件读取。 显式的 task_config 值总是胜出——
    对开发期一次性覆写 prompt 很有用。 """

    traj_id: str = ''
    """由 ``nexus.runner.hytuner`` 注入的 HyTuner 轨迹 id。

    形状通常为 ``{global_step}_{uid}_{sample_index}_{rollout_n}``。
    保留此字段是为了让审计 / 调试面板能把 task 输出与 HyTuner 轨迹行
    关联起来。
    """

    user_prompt: str = ''
    """agent 应满足的 task 描述。 """

    context_files: list[str] = Field(default_factory=list)
    """支撑文件的宿主机路径（PDF / HTML / MD / TXT / XLSX）。 会以原样
    上传到沙箱的 ``AGENT_INPUTS_DIR``，由 agent 通过 pdf-tools / xlsx
    skill / shell 读取（内容永远不会内联——只展示路径清单加建议工具）。 """

    extra_mcp_servers: list[dict[str, Any]] = Field(default_factory=list)
    """要追加到默认目录的额外 MCP 规格。 每一项形状与
    ``engine.mcp_catalog.build_specs`` 产出的条目一致——即

        {name, runtime_name?, client: {transport, command, args, env}}

    可用于借用 aidata-aio-agent 镜像里已经打包好的 MCP-Bench server
    （excel、 filesystem、 playwright、 woocommerce 等）。 缺省时为
    no-op，不影响不需要额外 MCP 的既有 case。 """

    agent_tools: list[str] = Field(default_factory=list)
    """除 ``base/finish`` 与上下文加载的 MCP 工具外，暴露给
    ``main_agent`` 的额外静态 Nexus 工具。

    面向宿主侧工具注册表中的工具，例如 ``google_serper_search`` 和
    ``visit``——它们并不是 MCP server。 case manifest 通过工具目录
    声明，由 ``case_spec`` 解析，例如::

        agent_tools:
          web_search:
            - google_serper_search
            - visit

    缺省时为 no-op，保持既有的“仅 MCP”aidata 工具面。 """

    wx_agent_hub: dict[str, Any] | None = None
    """选择性声明本 case 运行在 WeChat Agent Hub 运行时中。 设置后，
    引擎会自动挂载来自 MockServer 镜像的 ``wx_agent_hub`` +
    ``wx_pay`` MCP server，以及 ``pre_installed`` /
    ``market_available`` 里列出的每一个 provider。

    Case 作者写::

        wx_agent_hub:
          pre_installed:    [food_delivery]
          market_available: [ride_app, weather_app]

    而不是手工列出与之对应的 ``extra_mcp_servers`` 条目。

    形状（在 :func:`case_spec.resolve_case` 中校验）::

        {
            'pre_installed':    list[str],  # 预先视为已安装的 provider
            'market_available': list[str],  # 可通过应用市场发现的 provider
        }

    Provider 名称必须对应 aidata-aio-agent 镜像里已存在的 MockServer
    ``life/<name>/`` mock。 bootstrap 脚本（通常会 source
    ``/opt/mock-server/life/wx_agent_hub/bootstrap_helper.sh``）会用
    相同的列表填充
    ``$MOCK_SERVER_DATA_DIR/wx_agent_hub/wx_agent_hub__installation.json``，
    使 provider 侧的安装门与之匹配。

    缺省时不做任何 wx_agent_hub 处理；case 使用标准的 aidata 默认
    catalog 运行。 该字段是对 MockServer “life mock 必须走 case 级
    extra_mcp_servers 显式声明”这条红线的例外——``wx_agent_hub`` 被
    视作运行时标记，而非某个业务场景 mock，因此享有专用顶层字段。 """

    services: list[AidataService] = Field(default_factory=list)
    """可选的辅助 service 容器（每个是一个 docker runtime）。

    用于声明那些不应与 agent shell 共享进程空间的后端——例如一个
    只能通过 REST 访问的 WooCommerce mock。 task 启动时，引擎会：

      1. ``runtime_provider.configure(s.runtime_name, image=s.image, exposed_ports=s.exposed_ports, ...)``
      2. ``await ctx.get_runtime(s.runtime_name)``  ← 实际的 ``docker run``
      3. 在 service 容器内部运行 ``s.startup.command``
      4. 轮询 ``s.ready_check`` 直到退出码为 0（或触发 ``ready_timeout_s``）
      5. 使用 ``runtime.port_mappings`` 把每个
         ``s.connections[].placeholder`` 解析为 ``host:port``，并替换到
         ``user_prompt`` / ``extra_mcp_servers`` 中，或（若设了
         ``env_name``）作为环境变量暴露给 ``bootstrap_script``。 """

    bootstrap_script: str = ''
    """可选的 shell 脚本路径（相对于 case 目录），task 会在上传 case
    资产后、加载 MCP 工具之前，在沙箱内部运行它。 主要用于拉起后台
    service（例如在 :10003 上启动 toolathlon_woocommerce_api_server 并
    curl 播种数据）。 脚本的 cwd 是 ``CASE_DIR_IN``，因此可以按相对
    路径引用同级资产（``./assets/jd_products.json`` 等）。 """

    ask_user: dict[str, Any] | None = None
    """可选的确定性人类回复配置。 启用后，aidata 会暴露一个由 case
    的 ask_user 脚本支撑的本地 ``ask_user`` 工具。 """

    agent_timeout: int = 1800
    observation_time_timeout: float | None = None
    max_iterations: int = 200

    initial_trajectory: list[dict[str, Any]] = Field(default_factory=list)
    """从 ``task.yaml::initial_trajectory`` 加载的可选种子 OpenAI 聊天
    轨迹。 aidata runner 会把它传给支持种子历史的 agent
    （``main_agent`` 支持），这样先前的工具调用对模型是可见的，同时
    在实际运行中不会重放这些副作用。 """

    dump_state: dict[str, dict[str, Any]] = Field(default_factory=dict)
    """按 MCP 粒度提供的 ``_dump_state`` 调用参数（可选）。

    默认情况下，引擎对每个发现的 ``*_dump_state`` 工具都以 ``{}``
    调用，让每个 mock 应用其会话感知的默认值。 需要为 SQL rubric join
    额外抓取表的 case 可这样启用::

        dump_state:
          pdd:
            tables: orders,order_items,products,shops
          taobao:
            includeCatalog: true

    Key 是 MCP server 命名空间（``pdd``、 ``taobao``）。 Value 会作为
    JSON 风格的工具参数直接透传，字符串值中的 service/case 占位符会
    先被解析。 """

    rubric_sqlite_path: str = ''
    """一个可选的运行时 sqlite 文件路径，供 rubric 的
    ``state_equals(sql:...)`` 探针打开。 支持 ``__CASE_UUID__`` /
    ``__AIDATA_CASE_UUID__`` 占位符——引擎会在把解析后的路径写入
    ``state['_rubric_sqlite_path']``（供 :mod:`engine.rubric` 使用）
    之前，把它们替换为本次 episode 的 UUID。

    缺省 / 为空时，``state_equals(sql:...)`` 检查会以 ``MISSING``
    判定失败。 这是有意为之的 fail-closed：case 作者忘了声明路径时，
    不会静默地让所有 sql check 得 0 分而无明显信号。 case 契约：该
    路径必须能被生成该 DB 的同一个 MCP server 写入，因此典型用法是
    与 mock 自身的环境变量对齐（例如把
    ``extra_mcp_servers[*].env.ONLINE_SHOPPING_DB`` 与本字段保持
    一致）。 """

    rubric_sqlite_service: str = ''
    """可选的 sidecar service ``runtime_name``，SQL 探针应在其
    runtime 上执行，而非 ``main``。 当 sqlite oracle 位于 sidecar
    mock-server 容器内时必需——例如 ``life/online_shopping`` 会把
    ``rollout.db`` 写到 ``service_lifegym_mocks`` runtime 内的
    ``/var/lib/mock-server/online_shopping/``，agent 的 ``main``
    runtime 无法打开该文件。

    默认空串 → 探针在 ``main`` 上运行（stdio mock 的 legacy 行为，
    其 db 位于主容器内 ``/tmp/mock-run/<uuid>/``）。

    编写快捷方式：case 作者一般不直接设置本字段——而是在
    ``task.yaml::rubric_sqlite_path`` 中使用 ``service:<name>:<path>``
    前缀，:func:`engine.case_spec.resolve_case` 会自动把它拆分成
    ``rubric_sqlite_service`` + ``rubric_sqlite_path``。 解析器写入的
    是 ``services[].runtime_name``（不是 ``name``），因为
    :class:`nexus.core.context.ExecutionContext` 用 ``runtime_name``
    作为 runtime 的 key——前缀里那个面向人类的 ``name`` 只是解析器为
    case 作者提供的书写便利。 """

    case_metadata: AidataCaseMetadata | None = None
    """当 ``cases/<case_id>/task.yaml::metadata`` 存在时解析得到。
    对于尚未升级的 legacy case 为 ``None``（此时 validator 会告警）。 """

    runtime_image: str | None = None
    """``main`` runtime 的显式镜像，从 ``task.yaml::runtime.main.image``
    读取。 设置后，引擎会在获取 ``main`` 之前调用
    ``ctx.runtime_provider.configure('main', image=...)``，覆盖
    ``runtime.template.yaml`` 的声明。 该值也会转发给
    ``mcp_catalog.build_specs``，让 pdf-tools / shell 的 MCP 规格
    对准该镜像布局中的正确二进制（见 :mod:`engine.paths` 中的
    ``_IMAGE_LAYOUTS``）。

    缺省时 → main runtime 使用 harness 默认（即当前
    ``runtime.template.yaml`` 里的 ``aio-agent-rc``）。 case 作者只在
    case 依赖了仅 legacy ``aidata-aio-agent:2.x`` 镜像才附带的二进制
    （例如 wecom stdio mock、 ``/opt/mcp-benchmarks`` 常驻工具）时才
    需要设置本字段。 """

    pending_upstream: str = ''
    """设置后，dynamic 测试层会以给定原因跳过该 case。 用于 case 正在
    等待某个上游发布（例如 MockServer MR 尚未打 tag、复合 docker
    镜像未构建），从结构上无法通过直到该依赖到位。 编写风格：一句短
    描述，例如
    ``"waiting on MockServer feat/add-online-shopping tag"``。 static
    层仍会检查这些 case 的 manifest 形状。 """


class AidataTaskOutput(TaskRunOutput, WithScore):
    task_id: str = ''
    score: float = 0.0
    """最终 task 分数。 默认由 ScorePolicy 折叠 hard-gate 后的连续分；
    配置 ``config.reward.script`` 时则由脚本输出，并在失败的 hard gate 上
    继续封顶。 """

    continuous_score: float = 0.0
    """ScorePolicy 模式下是在最终坍缩之前的 hard-gate 后连续分；reward
    脚本模式下等于脚本输出经 hard gate 封顶后的最终分。 """

    score_policy: dict[str, Any] = Field(default_factory=dict)
    """默认回显已解析的 ``{threshold, continuous, use_threshold}``；
    reward 脚本模式回显 ``{mode: script, script, timeout_sec}``。 """

    rubric: dict[str, Any] = Field(default_factory=dict)
    rubric_results: list[dict[str, Any]] = Field(default_factory=list)
    rubric_version: str = '2.0'
    """产生 ``rubric_results`` 的 rubric schema 版本：legacy ``verify:``
    case 为 ``2.0``，使用面向标注人员的 assert DSL 的 v4.2 case 为
    ``4.2``。 """

    dimension_scores: dict[str, float] = Field(default_factory=dict)
    """v4.2 rubric 的按维度加权分数（correctness / functional /
    robustness——取决于 rubric 给 check 打的维度标签）。 legacy v2 rubric
    为空字典。 """

    junit_summary: dict[str, Any] = Field(default_factory=dict)
    """当 ``run-test.sh`` 生成了 ``tests/report.junit.xml`` 时的摘要。
    形状：``{"total": int, "passed": int, "failed": int,
    "testcase_names": [...]}``。 未调用 ``run-test.sh`` 时为空。 """

    case_metadata: dict[str, Any] = Field(default_factory=dict)
    """``cases/<case_id>/task.yaml::metadata`` 的回显——``{authors,
    created_at, tags}``。 早于 metadata 段的 case 为空字典。
    在结果 JSON 中出现，供审计和面板使用。 """

    state_summary: dict[str, Any] = Field(default_factory=dict)
    state_dump_path: str = ''
    """完整 ``state`` 字典以 JSON 写出的宿主机路径。 当
    ``AIDATA_STATE_DUMP_DIR`` 已设置（否则默认 ``/tmp/aidata-state/``）
    时填充。 通过 ``AIDATA_STATE_DUMP=0`` 禁用 state dump 时为空。 """

    state_snapshots: dict[str, str] = Field(default_factory=dict)
    """按阶段索引的宿主机侧原始 ``download_state`` JSON 路径。

    当前 key 为 ``before_agent``（在 MCP 工具加载之后、agent 执行之前
    捕获）与 ``after_agent``（在 agent 执行之后、MCP 关闭之前捕获）。
    通过 ``AIDATA_STATE_DUMP=0`` 禁用或相应捕获失败时为空。
    """

    error_message: str | None = None
    extra_fields: dict[str, Any] = Field(default_factory=dict)
    """通过 Nexus 的 HyTuner runner 透传的辅助字段。

    AIDATA 用它把奖励有效性和终止元数据与 span 级
    ``hytuner.termination_reason`` 信号一并暴露。
    """

    wall_time_s: float = 0.0


__all__ = [
    'AidataCaseMetadata',
    'AidataService',
    'AidataServiceCommand',
    'AidataServiceConnection',
    'AidataServiceFile',
    'AidataTaskInput',
    'AidataTaskOutput',
]
