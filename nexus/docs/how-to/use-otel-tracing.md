# 使用 OpenTelemetry 查看 Nexus 调用链

Nexus 可以把任务执行过程作为 OpenTelemetry Trace 上报到智研 APM。启用后，可以在一条调用链中查看 Task、Agent、
模型调用、工具调用、Runtime、Worker 和 MCP 的执行顺序、耗时与状态。

本文只介绍如何启用、查看和排查上报。OTel 通过进程环境变量启用，Nexus 原有任务 YAML 不需要增加 Trace 专用字段。

## 使用前准备

1. 打开[智研项目工作台](https://zhiyan.woa.com/worktable/#/)，选择目标项目并进入对应的监控功能。
2. 从该项目的监控接入信息中获取 OTel 租户 Token、环境值和项目 ID。OTel 租户 Token 用于上报，不是登录智研页面
   使用的访问 Token。
3. 使用普通 Python wheel 或 Nexus Runtime 镜像。`tencent-nexus-binary` 当前不支持 OTel。

!!! warning "保护租户 Token"

    不要把真实 Token 写进任务 YAML、脚本、文档、日志或 Git 仓库。建议通过本地环境变量、部署系统 Secret
    或密钥管理服务注入。

## 安装 OTel 依赖

使用内部 PyPI 安装 Nexus 时，增加 `otel` extra：

```bash
pip install "tencent-nexus[otel]" \
  --extra-index-url https://mirrors.tencent.com/repository/pypi/tencent_pypi/simple
```

在 Nexus 源码仓库中开发时执行：

```bash
uv sync --extra otel
```

当前 Nexus Runtime 镜像已安装 `[runtime,otel]`。使用 `worker_venv: true` 时，runner 也会把当前 wheel 的 OTel
依赖安装到 fresh Worker venv，不需要用户进入 Runtime 手工安装。

## 启用并上报

Nexus 当前使用 OTLP/HTTP，端口为 `4318`。根据 **Nexus 进程所在网络** 选择上报地址：

| 运行位置 | 上报地址 |
| --- | --- |
| DevCloud、办公网 | `https://trace.zhiyan.tencent-cloud.net:4318` |
| IDC、工蜂环境 | `http://otel-collect-proxy.zhiyan.tencent-cloud.net:4318` |

两个入口的 HTTP scheme 不同：DevCloud/办公网入口使用 HTTPS，IDC/工蜂入口使用 HTTP。

!!! warning "Host 与 Runtime 跨网络运行"

    Collector 地址按**产生 Span 的进程所在网络**选择，而不是按任务发起位置统一选择。例如 DevCloud Host 发起工蜂 Runtime
    时，Host 使用 DevCloud/办公网入口，Runtime Base、Worker 和受管 MCP 使用 IDC/工蜂入口。两侧不能共用同一个 endpoint；
    部署或 Runtime 启动配置必须分别注入对应地址。

### DevCloud、办公网

```bash
export ZHIYAN_OTEL_TENANT_ID='<从目标项目监控功能获取>'
export ZHIYAN_OTEL_ENVIRONMENT='<从目标项目监控功能获取>'
export ZHIYAN_PROJECT_ID='<从目标项目监控功能获取>'
export NEXUS_OTEL_ENABLED=true
export OTEL_TRACES_EXPORTER=otlp_proto_http
export OTEL_EXPORTER_OTLP_PROTOCOL=http/protobuf
export OTEL_EXPORTER_OTLP_ENDPOINT=https://trace.zhiyan.tencent-cloud.net:4318
export OTEL_SERVICE_NAME=nexus-host
export OTEL_RESOURCE_ATTRIBUTES="tps.tenant.id=${ZHIYAN_OTEL_TENANT_ID},deployment.environment.name=${ZHIYAN_OTEL_ENVIRONMENT},nexus.project.id=${ZHIYAN_PROJECT_ID}"
export OTEL_TRACES_SAMPLER=always_on
export OTEL_PROPAGATORS=tracecontext
export OTEL_METRICS_EXPORTER=none
export OTEL_LOGS_EXPORTER=none
```

### IDC、工蜂环境

```bash
export ZHIYAN_OTEL_TENANT_ID='<从目标项目监控功能获取>'
export ZHIYAN_OTEL_ENVIRONMENT='<从目标项目监控功能获取>'
export ZHIYAN_PROJECT_ID='<从目标项目监控功能获取>'
export NEXUS_OTEL_ENABLED=true
export OTEL_TRACES_EXPORTER=otlp_proto_http
export OTEL_EXPORTER_OTLP_PROTOCOL=http/protobuf
export OTEL_EXPORTER_OTLP_ENDPOINT=http://otel-collect-proxy.zhiyan.tencent-cloud.net:4318
export OTEL_SERVICE_NAME=nexus-host
export OTEL_RESOURCE_ATTRIBUTES="tps.tenant.id=${ZHIYAN_OTEL_TENANT_ID},deployment.environment.name=${ZHIYAN_OTEL_ENVIRONMENT},nexus.project.id=${ZHIYAN_PROJECT_ID}"
export OTEL_TRACES_SAMPLER=always_on
export OTEL_PROPAGATORS=tracecontext
export OTEL_METRICS_EXPORTER=none
export OTEL_LOGS_EXPORTER=none
```

三个 `ZHIYAN_*` 变量必须来自同一个目标项目的监控接入信息。环境变量必须在启动 `nexus` 命令或首次 `import nexus`
前设置；OTel Provider 初始化后再修改 Host 配置不会重建已经创建的 exporter。

!!! tip "存在 HTTP(S) 代理时"

    如果环境设置了 `HTTP_PROXY` 或 `HTTPS_PROXY`，应把所选 Collector 域名加入 `NO_PROXY` 和 `no_proxy`，避免内网上报
    被代理转发。

## 在智研 APM 查看

1. 打开[智研项目工作台](https://zhiyan.woa.com/worktable/#/)，选择上报配置对应的项目。
2. 从该项目的监控功能进入 **应用性能监控（APM）** 调用链查询页面。
3. 选择与 `ZHIYAN_OTEL_ENVIRONMENT` 相同的环境。
4. 把时间范围调整到 Nexus 任务的实际运行时间。
5. 有 OTel Trace ID 时直接查询；没有时，可以先按时间和服务名筛选。

常见服务名如下：

| 服务名 | 表示的执行位置 |
| --- | --- |
| `nexus-host` | 发起任务的 Nexus 进程 |
| `nexus-runtime-base` | Nexus Runtime Base |
| `nexus-runtime-worker` | fresh Worker venv |
| `nexus-mcp-<server>` | Nexus 管理的 MCP Server |
| `nexus-distributed-control-gateway` | Distributed Control Gateway |
| `nexus-distributed-application-gateway` | Distributed Application Gateway |
| `nexus-distributed-scheduler` | Distributed Scheduler |
| `nexus-distributed-reconciler` | Distributed Reconciler |
| `nexus-distributed-instance-manager` | Distributed InstanceManager |

!!! note "Trace ID"

    智研 APM 的 Trace ID 是 32 位十六进制 OTel Trace ID。Span 属性中的 `nexus.trace_id` 用于关联 Nexus
    原生 Trace，不要把它误填到 APM 的 Trace ID 查询框。

## 可以看到什么

一条完整链路通常包含以下节点：

```text
task
└── agent
    ├── openai_completion
    └── agent.observation
        └── runtime.invoke
            └── runtime.worker.invoke
                └── mcp.call
                    └── subprocess span（可选）
```

具体节点取决于任务实际使用的 Agent、Runtime 和工具。用户可以查看：

- 一次任务和 Agent 每轮执行的先后顺序。
- 模型、工具、Runtime 和 MCP 调用的耗时及成功/失败状态。
- DevCloud/办公网进程与 Runtime Base、Worker、受管 MCP 之间的完整调用关系。
- HTTP 请求的路径、状态码和耗时。
- Task、Runtime、Worker、MCP 等用于定位本次运行的关联信息。
- Nexus Span 已记录的输入、输出和事件内容。

普通第三方 MCP 或普通子进程不一定会单独上报 Span；只有它们自身启用了 OTel 并产生 Span 时，才会在 APM 中显示其
内部执行节点。

!!! note "HTTP Span 与业务链"

    工蜂网关或反向代理可能重建纯 HTTP transport 的父上下文，因此 HTTP SERVER Span 不一定直接挂在 Host HTTP CLIENT
    Span 下。Nexus managed Runtime、Worker 和 MCP 会通过 payload carrier 继续业务语义链。判断链路完整性时，应同时检查
    `runtime.invoke`、`runtime.worker.invoke` 和 `mcp.call` 的父子关系，不要只看网关 HTTP Span。

Distributed runner 还会用 execution/assignment 内部 carrier 跨过 Scheduler 与 InstanceManager 的数据库轮询边界。正常
链路包含 `distributed.scheduler.assign`、`distributed.assignment.start` 和 Runtime Base 的 `runtime.invoke`，三者应逐级
保持父子关系；Application Gateway 的 HTTP Span 表示实际 transport，不取代 payload carrier 的业务父上下文。

## 如何确认链路正常

完成一次包含模型调用和工具调用的任务后，检查：

- 智研 APM 能在对应项目、环境和时间范围内找到本次运行。
- `nexus-host`、Runtime、Worker 和受管 MCP 节点使用同一个 OTel Trace ID。
- 模型和工具调用位于对应的 Agent 节点下面。
- 使用 Runtime 时能看到 `runtime.invoke`；使用 fresh Worker 时能看到 `runtime.worker.invoke`。
- 使用 Nexus 管理的 MCP 时能看到 `mcp.call`。
- 使用 distributed runner 时能看到 `distributed.scheduler.assign`、`distributed.assignment.start`，并继续进入
  Runtime Base 的 `runtime.invoke`。
- 每个节点都有合理的开始时间、耗时和状态。
- Runtime 或 Worker 位于 IDC/工蜂时，其 exporter 使用 IDC/工蜂 endpoint，而不是 Host endpoint。

没有使用某项能力时，对应 Span 不会出现。例如任务没有使用 MCP，就不应期待 `mcp.call`。

## 常见问题

| 现象 | 检查方法 |
| --- | --- |
| APM 完全没有数据 | 检查是否安装 `[otel]`、`NEXUS_OTEL_ENABLED` 是否为 `true`、租户 Token 是否正确，以及项目、环境和时间范围是否选对。 |
| DevCloud/办公网上报返回 HTTP 400 | 检查 endpoint 是否误写成 `http://`；该入口必须使用 `https://trace.zhiyan.tencent-cloud.net:4318`。 |
| IDC/工蜂上报出现 TLS `wrong version number` | 检查 endpoint 是否误写成 `https://`；该入口必须使用 `http://otel-collect-proxy.zhiyan.tencent-cloud.net:4318`。 |
| 只有 `nexus-host`，没有 Runtime 或 Worker | 检查 Runtime 镜像是否包含 OTel、Runtime/Worker 是否使用 IDC/工蜂 endpoint、所在网络能否访问 Collector，并确认任务确实使用了 Runtime/Worker。 |
| 查不到刚结束的任务 | 等待数秒后重试，并扩大查询时间范围。 |
| 第三方 MCP 或普通子进程没有独立 Span | 这是允许的；第三方进程只有自行启用 OTel 后才会上报内部 Span。 |
| 业务成功，但日志中有 Collector 错误 | OTel 导出失败不会改变任务结果；根据错误信息检查 endpoint、网络、Token 和 Collector 状态。 |

## 数据安全

智研 APM 中可能展示 Nexus Span 已记录的模型输入输出、工具参数和执行结果。Nexus 不会统一清洗这些业务内容；启用前应
确认任务数据允许进入对应的观测项目，并遵守项目的数据权限和保密要求。

`tps.tenant.id` 是 Collector 路由所需的 Resource 属性。不要把租户 Token 复制到任务 YAML、prompt、工具参数或其他 Span
属性，也不要打印包含完整 OTel 环境的日志。模型密钥和其他认证信息同样不得主动写入 Trace。
