# OTel Unified Trace MVP

## 1. 结论

官方 OpenTelemetry Distro 负责 Provider、自动 instrumentation 和 OTLP 导出；`nexus.utils.otel` 只负责 Nexus
Span mirror、非 HTTP 传播和子进程 carrier；unified sink 继续作为 Nexus trace 的唯一事实源。

每个进程独立上报到同一个 Collector，不通过 Socket.IO 或 Host 回传 OTel Span 数据。

```text
Nexus START / UPDATE / END
            │
            ▼
  nexus.utils.otel.nexus_bridge
            │
            ▼
      OTel mirror Span ───── current context ───── native OTel Span
            │                                       │
            └──────────────── OTLP ─────────────────┘
                                    │
                                    ▼
                                Collector
```

- **Nexus Span**：Nexus 原生 Span，进入 unified sink。
- **mirror Span**：由 Nexus 生命周期驱动的真实 OTel SDK Span。
- **native Span**：FastAPI、aiohttp、httpx、SQLAlchemy 或显式 transport instrumentation 产生的 OTel Span。

mirror 与 native Span 共用全局 Provider 和 current context，因此组成同一棵 OTel Span 树。

## 2. 目标与边界

### 2.1 目标

1. Host、Runtime Base、Worker/venv 和 Nexus Python MCP 进入同一条 W3C trace。
2. Nexus Span 镜像为真实 SDK Span，native Span 能挂在 mirror 下。
3. HTTP 复用官方 instrumentation；Socket.IO 与 MCP 只补薄的 CLIENT/SERVER Span 和 carrier。
4. 随机 Python 子进程只获得调用时的 `TRACEPARENT` / `TRACESTATE`，不继承 Collector 配置。
5. OTel 关闭、缺依赖或导出失败时，不改变 unified sink 和业务结果。

### 2.2 非目标

- 不把 OTel Span 写入 unified sink，也不做 Host relay、Collector 代理或离线重放。
- 不强制 SDK 使用 Nexus Span ID。
- 不增加 `sitecustomize.py`、修改 `PYTHONPATH`，也不在随机子进程中自动提取环境 carrier。
- 不覆盖 `os.exec*`、multiprocessing、原生扩展或非 Python 进程启动方式。
- 不保证未 import Nexus 的第三方 MCP 自动初始化 OTel。
- 不增加 LLM 自动 instrumentation，避免与 Nexus 原生 LLM Span 重复。

## 3. 模块与初始化

所有直接调用 OpenTelemetry Python API/SDK 的 Nexus 代码放在 `nexus.utils.otel`：

```text
nexus/utils/otel/
├── auto.py             # 开关、官方 initialize()、初始化幂等
├── environment.py      # 受控进程环境 overlay
├── redaction.py        # exporter / tenant / control-plane token 日志脱敏
├── lifecycle.py        # force_flush / best-effort flush / shutdown
├── propagation.py      # W3C inject / extract
├── transport.py        # 非 HTTP CLIENT/SERVER Span
├── subprocess.py       # 子进程环境 carrier instrumentor
├── mcp.py              # MCP _meta carrier 与 transport Span
├── nexus_bridge.py     # Nexus START / UPDATE / END mirror
├── nexus_semconv.py    # 全量内容编码与查询字段
└── ids.py              # Nexus ID / OTel hex 编码转换
```

边界：

- `nexus.trace`、Runtime、MCP 和 Runner 只调用上述 helper，不直接 import `opentelemetry`。
- bridge 读取 Nexus Span，但不参与 unified sink 调度。
- OTel 关闭时不加载 OTel 包、不 patch subprocess。

### 3.1 依赖

`[otel]` extra 使用 `opentelemetry-distro` 与 `opentelemetry-exporter-otlp-proto-http`，并安装 FastAPI、ASGI、
aiohttp client/server、httpx、requests、logging、
SQLAlchemy、Redis 和 MySQL instrumentation。Runtime 镜像安装 `[runtime,otel]`；启用 OTel 的 fresh Worker venv
安装当前 wheel 的 `[otel]`。`[dev]` 包含 Distro 与 HTTP exporter。当前 Nuitka binary wheel 不打包 `[otel]`，
OTel 属于普通 Python wheel / Runtime 镜像能力。

### 3.2 唯一初始化入口

`nexus/__init__.py` 在 module discovery 前调用 `initialize_if_enabled()`：

1. `NEXUS_OTEL_ENABLED != true` 时返回。
2. 用 `setdefault` 补 Distro、Configurator、trace exporter、host role 和 service name；传播器固定为 `tracecontext`。
3. 若全局 TracerProvider 已配置，不重复配置 Provider；非官方 sitecustomize 场景只补 Nexus subprocess hook。
4. 否则调用官方
   `opentelemetry.instrumentation.auto_instrumentation.initialize(swallow_exceptions=False)`。
5. 进程锁保证只初始化一次；缺 `[otel]` 时只告警一次并让本进程 helper 全部 no-op；配置错误 fail fast。

Nexus 不实现 Configurator、Provider、Sampler、SpanProcessor 或 Exporter。官方 Distro 按标准环境变量创建这些对象，并加载
所有已安装且依赖兼容的 `opentelemetry_instrumentor` entry point。禁用单个 instrumentation 使用：

```bash
OTEL_PYTHON_DISABLED_INSTRUMENTATIONS=logging,mysql
```

## 4. Nexus mirror

mirror 使用全局 Provider 创建 SDK Span，不构造 `ReadableSpan`，也不直接调用 Exporter：

```text
Nexus START  -> 创建 mirror，保存 nexus span_id -> OTel Span
Nexus UPDATE -> 同步最终属性和本批新增事件
Nexus END    -> 同步属性、状态并结束 mirror，删除映射
```

`SpanController.add_event(..., origin='local' | 'replay')` 的规则：

- `local`：正常 dispatch，并驱动 mirror。
- `replay`：正常进入 unified sink，但不驱动 mirror。

因此 Runtime 原生 Nexus Span 在源进程导出一次，Host replay 不重复导出。

### 4.1 父节点与 current

创建 mirror 的父节点优先级：

1. `parent_span_id` 对应的本地 mirror。
2. 当前已提取或 native OTel context。
3. root。

进入 Nexus span context 时把 mirror 设为 current，退出时恢复原 context；activation 按 async task/thread 隔离。

```text
[A] HTTP SERVER
└── [M] runtime.invoke
    ├── [A] HTTPX CLIENT
    └── [A] runtime.worker.invoke       # Socket.IO CLIENT
        └── [A] runtime.worker.invoke   # Socket.IO SERVER
            └── [M] agent.main
                └── [A] mcp.call
```

`[M]` 只由 Nexus 生命周期驱动；`[A]` 只由官方或显式 instrumentation 驱动。

### 4.2 ID

保留 Nexus base64url ID 与 OTel hex ID 的双向编码 helper，供兼容、查询和展示；它不参与 SDK ID 生成。

mirror 通过以下属性关联原 Span：

```text
nexus.trace_id
nexus.span_id
nexus.parent_span_id
```

## 5. 跨边界传播

通用规则：只传播 W3C `traceparent` / `tracestate`，不传播 baggage；非法 carrier fail open。普通 HTTP 边界以
HTTP/native current context 为准；Nexus 自己发起并携带 `RemoteTraceContext` 的 managed invoke 边界，以及没有自动提取
上下文的消息边界，以 payload 中有效的 carrier 为准，缺失或非法时才回退 current。这样既避免代理或网关重建 HTTP
SERVER trace 截断 Host -> Runtime 语义链，也避免异步服务器复用任务时把前一次请求的 current 误当成新消息父节点。
在 invoke 中启动的长期 Uvicorn 服务使用空 `contextvars.Context` 创建后台 task，不继承启动请求的 current；后续每个
HTTP 请求仍由官方 instrumentation 从请求 header 独立提取父上下文。

### 5.1 Host -> Runtime Base

aiohttp/httpx instrumentation 仍注入 HTTP header，FastAPI/ASGI instrumentation 仍提取并创建 SERVER Span。进入 Nexus
managed invoke 后，Runtime Base 将 `RemoteTraceContext.otel_carrier` 中有效的 carrier 作为业务执行链的权威父上下文；
只有 payload carrier 缺失或非法时才沿用 HTTP/native current。这样即使工蜂网关重建了 HTTP trace，Base direct execution、
异步 job 和 Base -> Worker dispatch 仍接回 Host 的 `runtime.invoke`。已经启动的 ASGI SERVER Span 不做事后 reparent，
它仍表示实际 HTTP transport 链。

### 5.2 Runtime Base -> Worker

1. Base 创建 `runtime.worker.invoke` CLIENT Span，并把当前 carrier 写入 Socket.IO payload。
2. Worker 将 payload 中有效的 carrier 作为权威父上下文，创建同名 SERVER Span并设为 current；没有有效 carrier 时回退
   Worker 的 current。
3. Worker 内的 mirror 和 native Span 成为该 SERVER Span 的后代。

Socket.IO 只发送 carrier，不发送 OTel Span 数据。

### 5.3 MCP

1. caller 创建 `mcp.call` CLIENT Span，在其内写入 `_meta.traceparent` / `_meta.tracestate`。
2. Proxy/Server 将 `_meta` 中有效的 carrier 作为权威父上下文，创建 `mcp.call` SERVER Span；没有有效 carrier 时回退
   transport current。即使异步 HTTP handler 带有残留或不相关的 current，也不能覆盖 MCP 消息声明的父节点。
3. Proxy 向下游转发时重新创建 CLIENT Span并注入 fresh carrier。

`mcp.call` transport Span 只记录协议元数据；调用链已有的 Nexus tool Span 若含 arguments、result 或认证字段，
mirror 会原样同步。

Nexus-managed Python MCP 获得受控 OTel 环境并通过 import Nexus 初始化。第三方 STDIO MCP 会剥离 Host 的 exporter
环境，只接收消息 carrier；其内部是否初始化 OTel 不属于本 MVP。

### 5.4 Distributed control plane -> Runtime

Distributed runner 的 HTTP 服务不维护自定义 middleware。Control Gateway 与 Application Gateway 安装并由 Distro
自动加载官方 `opentelemetry-instrumentation-aiohttp-server`；它负责提取 W3C header、创建 SERVER Span 和记录 HTTP
属性。默认排除 `/v1/health`，避免健康检查污染 Trace。

Application Gateway 的顶层 `web.Application` 使用该官方 instrumentation，所有标准 execution 路由共享同一套
HTTP SERVER Span 语义。

`POST /v1/executions` 后的 Scheduler 与 InstanceManager 通过数据库解耦，HTTP current 无法自然跨越轮询边界，因此只在
两个内部 JSON 字段中保存 W3C carrier：

```text
Control Gateway aiohttp SERVER
└── Execution.trace_carrier_json
    └── distributed.scheduler.assign PRODUCER
        └── Assignment.trace_carrier_json
            └── distributed.assignment.start CONSUMER
                └── Runtime Base runtime.invoke SERVER
                    └── Runtime Worker / MCP / model / tool
```

- execution carrier 在创建 execution 时写入，作为每次调度尝试的权威父上下文。
- Scheduler 在 `distributed.scheduler.assign` 内生成 fresh assignment carrier；重试会生成新的 assignment carrier。
- InstanceManager 仅在启动该 assignment 的 task/context 中激活 carrier。Asyncio 与 Thread executor 都会在创建执行单元时
  复制当前 `contextvars`，后续独立请求不会复用该上下文。
- Runtime Base 的 `runtime.invoke` SERVER Span 以 payload carrier 为准，避免 Application Gateway、Ingress 或其他 HTTP
  中间层重建 trace 后截断业务链。
- chart 首次安装使用普通 `db-init` Job；覆盖升级使用 `pre-upgrade` hook，先重建同名 Job 并完成
  carrier 字段迁移，再滚动 distributed workload。
- 两个字段只属于内部 ORM/migration，不暴露在 execution 或 assignment 公共 API。

#### 5.4.1 Carrier capture boundary

`CreateExecution.trace_carrier` explicitly captures the current W3C carrier by default for database persistence.
Explicit carriers, including explicit `None`, take precedence. This infrastructure behavior stays in the execution
repository and does not depend on monkey patching the dataclass constructor.

`nexus-hyway` is registered under `project.entry-points.opentelemetry_instrumentor`, alongside `nexus-subprocess`.
`NexusHywayInstrumentor` installs a reversible patch on `ExecutionRepository._new_execution_row` after the repository
module is imported. This common boundary covers both single and bulk creation without eagerly importing database
dependencies during OTel startup. Both official auto-instrumentation and programmatically configured providers load
the entry point, honoring `OTEL_PYTHON_DISABLED_INSTRUMENTATIONS` and `NEXUS_OTEL_ENABLED`.

The cloud row wrapper adds `hyway_slice_id` from a top-level execution's submission idempotency key. For batch children,
it adds `hyway_instance_id` from `spec['batch']['item_id']`, retaining the parent's slice ID; internal child retry keys
never become slice IDs. Existing tracestate members are preserved. Only the new ORM row's carrier is enriched, leaving
the caller's request and current context unchanged. Normal persistence writes the result to
`Execution.trace_carrier_json`; public API payloads do not expose it.

When `NEXUS_OTEL_ENABLED=true`, the submit CLI adds the `otel` extra to source-package submissions so dynamically
built environments install the OTel dependencies. This follows the requested switch even if the local OTel runtime
is unavailable. The CLI has no custom OTel span or key injection; cloud HTTP instrumentation provides the request
span, and the two correlation keys are added entirely on the cloud before persistence.

#### 5.4.2 End-to-end validation

Validate propagation with two cases:

1. Run an ordinary instrumented `nexus submit batch` without an upstream `tracestate`. The stored execution carrier
   must contain `traceparent` and `tracestate` with `hyway_slice_id`; materialized children also contain
   `hyway_instance_id`. The gateway, scheduler, manager, runtime and model-request spans must share the same trace ID.
2. In a test process, create a valid OTel parent `SpanContext` with `TraceState([('nexus', 'validation')])`, activate it,
   and invoke the real batch submit client in that same process. Do not modify HTTP headers or database rows directly.
   Verify that tracestate preserves `nexus=validation` alongside the correlation members in `executions.trace_carrier_json`, the resulting
   `assignments.trace_carrier_json`, and the headers received by a controlled local model HTTP endpoint.

The final assertion belongs at the model HTTP boundary. A tracing backend may preserve `tracestate` as propagation
state without rendering it as a searchable span attribute, so the Jaeger or Tempo UI alone is not sufficient evidence
that the outgoing model request carried the header.

## 6. 进程环境与子进程

`environment.py` 为工蜂 Runtime Base、Worker/venv 和 Nexus-managed MCP 生成可信 overlay：复制 Host 的全部 `OTEL_*`
和 `NEXUS_OTEL_*`，再覆盖 process role、service name 与 `OTEL_PROPAGATORS=tracecontext`。通用 `RuntimeProvider`、
AGS 与本地 Docker provider 不隐式注入 OTel；这些 provider 后续若要支持，应各自定义受信启动边界。
signal-specific exporter、
TLS、compression、timeout、BSP、span limit 和 instrumentation 配置无需维护白名单。

关键字段示例：

```text
NEXUS_OTEL_ENABLED
NEXUS_OTEL_PROCESS_ROLE
OTEL_SERVICE_NAME
OTEL_RESOURCE_ATTRIBUTES
OTEL_EXPORTER_OTLP_{ENDPOINT,TRACES_ENDPOINT,PROTOCOL,HEADERS,CERTIFICATE,CLIENT_CERTIFICATE,CLIENT_KEY,...}
OTEL_{TRACES_EXPORTER,METRICS_EXPORTER,LOGS_EXPORTER,TRACES_SAMPLER,TRACES_SAMPLER_ARG,PROPAGATORS}
OTEL_PYTHON_{DISTRO,CONFIGURATOR,DISABLED_INSTRUMENTATIONS}
```

默认角色：

```text
host            -> nexus-host
runtime-base    -> nexus-runtime-base
runtime-worker  -> nexus-runtime-worker
mcp             -> nexus-mcp-<server>
distributed-control-gateway     -> nexus-distributed-control-gateway
distributed-application-gateway -> nexus-distributed-application-gateway
distributed-scheduler           -> nexus-distributed-scheduler
distributed-reconciler          -> nexus-distributed-reconciler
distributed-instance-manager    -> nexus-distributed-instance-manager
```

overlay 在用户环境之后合并，用户不能覆盖受控 Collector、认证和传播配置。

Host 与 Runtime 所在网络无法共用 Collector 入口时，在受信 Host 部署环境中额外设置：

```bash
NEXUS_OTEL_RUNTIME_EXPORTER_OTLP_ENDPOINT=http://runtime-collector:4318
```

Host 自身仍使用标准 `OTEL_EXPORTER_OTLP_ENDPOINT`。生成 `runtime-base` / `runtime-worker` overlay 时，上述受控值会
成为子进程的标准 `OTEL_EXPORTER_OTLP_ENDPOINT`；Runtime 内启动的 Nexus-managed MCP 随后继承 Runtime endpoint。
未设置该变量时保持兼容，Runtime 继续继承 Host endpoint。任务或 runtime config 中的 `OTEL_*` / `NEXUS_OTEL_*`
仍会被过滤，不能覆盖受控 endpoint、认证或传播配置。

### 6.1 随机 Python 子进程

`nexus-subprocess` entry point 由官方 auto-instrumentation 加载，统一 hook：

```text
subprocess.Popen
asyncio.create_subprocess_exec
asyncio.create_subprocess_shell
```

每次调用复制显式 env 或当前 `os.environ`，删除全部 `OTEL_*` / `NEXUS_OTEL_*` 和旧 carrier，再从调用时的 current
context 注入大写 `TRACEPARENT` / `TRACESTATE`；不修改全局环境，hook 失败时退回原参数。并发 task 各自获得自己的
carrier。Worker 与 Nexus-managed MCP 的启动边界显式进入 trusted scope，保留完整 OTel 配置；普通 subprocess 和用户
shell 不会获得 Collector header，也不会仅因 hook 而自动产生 span。受管子进程只有在显式保留 OTel 环境并初始化 Nexus
后，才能继续创建并上报自己的子 span。

## 7. 配置、内容与生命周期

只使用标准环境变量：

```bash
NEXUS_OTEL_ENABLED=true
OTEL_TRACES_EXPORTER=otlp_proto_http
OTEL_EXPORTER_OTLP_ENDPOINT=http://collector:4318
# 可选：Host 与 Runtime 需要不同 Collector 入口时设置
NEXUS_OTEL_RUNTIME_EXPORTER_OTLP_ENDPOINT=http://runtime-collector:4318
OTEL_EXPORTER_OTLP_PROTOCOL=http/protobuf
OTEL_SERVICE_NAME=nexus-host
OTEL_RESOURCE_ATTRIBUTES="tps.tenant.id=<tenant-token>,deployment.environment=prod"
OTEL_TRACES_SAMPLER=parentbased_traceidratio
OTEL_TRACES_SAMPLER_ARG=1.0
OTEL_PROPAGATORS=tracecontext
```

`NEXUS_OTEL_ENABLED=true` 是唯一启用开关；endpoint、protocol、headers、sampler 和 exporter 均由官方实现解析；
传播固定使用 W3C tracecontext。
智研 Collector 用 Resource 字段 `tps.tenant.id` 路由租户，不放在 OTLP authorization header。

智研内网两个 OTLP/HTTP 入口的 scheme 不同：Host 直连 `https://trace.zhiyan.tencent-cloud.net:4318`，IDC 直连
`http://otel-collect-proxy.zhiyan.tencent-cloud.net:4318`。Host 入口发送明文 HTTP 会返回 400；IDC 入口使用 HTTPS
会触发 TLS `wrong version number`。必须按进程所在网络选择入口并保留对应 scheme，不能把网络层接收失败当作链路上报成功。

### 7.1 全量内容

不使用 `SanitizingSpanExporter`。mirror 默认同步 Nexus Span 已有的全部 attributes、event attributes、status code 和
status message，包括 prompt、response、inputs、outputs、tool/MCP arguments、result 及认证字段。标量和同类型数组直接
写入；dict、嵌套和混合结构编码为 JSON 字符串，并补充 Nexus ID 与常用查询字段。

`openai_completion` 保持传统 Nexus span：名称和 `SpanKind.CLIENT` 不变，不生成或透传任何 `gen_ai.*` 属性，也不从
`inputs` / `outputs.usage` 派生 GenAI model/token 字段。请求模型、响应模型和 token 用量仍完整保存在原始
`inputs` / `outputs` JSON 中。它因此不会进入依赖 GenAI 分类字段的 LLM 专用视图，应从普通 APM trace 查询。

关键调用边界补充不含请求正文和结果的传统 transport span：

```text
agent.observation
├── runtime.invoke -> HTTP / Socket.IO -> runtime.worker.invoke
└── mcp.call CLIENT -> mcp.call SERVER
```

不额外创建 `ToolExecutor.tool.call` span。`runtime.invoke` 只记录协议、方法和 import target；`mcp.call` 只记录服务名、
工具名与 RPC 方法，不记录 arguments/result。MCP `_meta` 继续携带 W3C carrier，使这些 span 与下游 Runtime、Worker 和
托管 MCP 保持同一个 trace ID。

`OTEL_EXPORTER_OTLP_*_HEADERS` 及 endpoint URL 中的 userinfo 只交给官方 Exporter；Nexus 不把它们额外写入 Span、
resource、artifact 或日志。这不影响 Nexus Span 自身已有的认证字段，它们仍按“全量内容”同步。native Span 保持官方
默认语义；需要清洗 native URL、异常或数据库字段时在 Collector processor 处理。
智研的 `tps.tenant.id` 按协议必须随 Resource 上报，但 Nexus 在启动命令和运行日志中对对应
`OTEL_RESOURCE_ATTRIBUTES` 做隐藏。

### 7.2 错误与 flush

- mirror、carrier、transport、subprocess hook 和导出失败均 fail open，不进入 `RunHooks.hook_failures`。
- 官方初始化配置错误 fail fast；Collector 错误由官方 SDK 记录，不改变业务或 unified sink。
- Runtime/Worker 在 trace barrier 通过共享异步 helper offload best-effort `force_flush`；Worker stop 与单次/批量 CLI
  也显式 flush。Live service 不注入命令级 `finally`，由官方 SDK 的进程退出生命周期收尾。

### 7.3 Distributed Helm 配置

Distributed chart 通过一个全局 `otel` 配置给五类常驻进程注入相同 exporter 配置，并分别设置稳定的 service name 与
process role。control/workload 镜像都安装 `[otel]`，无需 sidecar 或 relay：

```yaml
otel:
  enabled: true
  endpoint: http://otel-collector.observability.svc:4318
  protocol: http/protobuf
  tracesExporter: otlp_proto_http
  tracesSampler: always_on
```

`OTEL_RESOURCE_ATTRIBUTES` 与 `OTEL_EXPORTER_OTLP_HEADERS` 从 Kubernetes Secret 读取，不进入 values 或 Pod template。
修改 `otel` 或 Secret 后需要滚动重启各 Deployment/StatefulSet；已经初始化的 Provider 不会运行时重载配置。数据库升级到
`0003_add_trace_carriers` 后才可启动新版本 Scheduler、Gateway 与 InstanceManager。

## 8. 改动清单

| 模块 | 改动 |
| --- | --- |
| `nexus/utils/otel/` | 新增全部 OTel 集成 |
| `nexus/__init__.py` | module discovery 前官方 auto-initialization |
| `nexus/trace/controller.py` | local 驱动 mirror，replay 跳过 |
| `nexus/trace/tracer.py` | mirror current 与远端 fallback context |
| Runtime Base / Worker | Socket.IO transport Span、carrier、trace barrier flush |
| Distributed Gateway / Scheduler / Manager | 官方 aiohttp SERVER instrumentation、两级 DB carrier、PRODUCER/CONSUMER Span |
| Runtime client | `runtime.invoke` 传统 transport Span |
| 工蜂 provider / Worker venv | 受控环境与 `[otel]` 依赖；AGS / Docker 暂不支持注入 |
| MCP client / proxy / factory | `_meta` 传播、transport Span、managed env |
| subprocess | 调用时 carrier 注入；不改业务进程启动方式 |
| CLI | 删除分散 bootstrap，单次/批量命令退出点 flush |
| packaging | HTTP exporter、aiohttp server、subprocess entry point、Runtime/distributed 镜像 `[otel]`；Nuitka 暂不支持 |

删除 `nexus/trace/otel/`、旧 MCP telemetry、`SanitizingSpanExporter`、分散 `bootstrap_otel()`、自定义
instrumentation allowlist 以及手工 FastAPI/ASGI instrumentation。

## 9. 验证与部署

| 层级 | 必须验证 |
| --- | --- |
| disabled | 不加载 OTel、不 patch subprocess，Nexus 行为不变 |
| mirror | START/UPDATE/END、replay、父节点优先级、full-content、native child |
| boundary | HTTP、Socket.IO、MCP、distributed DB polling、fresh venv、并发 subprocess 仅 carrier、第三方 MCP 无 exporter env；消息 carrier 覆盖残留 current；长期 HTTP 服务不继承启动请求 context |
| failure | 缺依赖只告警一次；初始化失败 fail fast；Collector 不可用不影响业务 |
| real Collector | mirror、native child、跨边界父子关系、`service.name` 与 `nexus.*` 可查询 |

本地确定性 E2E 必须使用官方 stdio MCP client 和真实受管子进程，并断言以下父子链，而不只是检查 trace ID 相同：

```text
Nexus mirror
└── native model.request
    └── Host mcp.call CLIENT
        └── managed MCP mcp.call SERVER
            └── managed subprocess span
```

工蜂分布式回归还需检查每次 `Base runtime.worker.invoke CLIENT` 都是对应
`Worker runtime.worker.invoke SERVER` 的直接父节点。

Kubernetes distributed 回归还需断言 execution 创建请求、Scheduler PRODUCER、Manager CONSUMER、Runtime Base SERVER
使用同一个 Trace ID，并逐级校验 parent span ID；只检查 service name 或 Trace ID 相同不足以证明传播正确。

真实 endpoint 和凭据只通过临时进程环境传入；智研回归测试使用
`NEXUS_OTEL_E2E_TENANT_ID`，输出只记录 trace ID、Span 名称、父子关系和结果，不打印 token。

部署约束：

- 旧 Host 若已有 bridge 注入 W3C HTTP header，可不升级；新 Runtime 仍能继续后续链路。
- 只有安装新 Nexus 的进程具有 subprocess hook。
- 所有直接上报进程必须能访问同一 Collector 并获得相应凭据。
- 本方案不使用 relay；Runtime 无法直连 Collector 时视为部署网络条件不满足。
