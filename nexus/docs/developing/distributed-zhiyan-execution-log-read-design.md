# Execution Manager Logs 智研双源读取设计

## 文档状态

- 状态：Draft，待评审
- 范围：Execution 详情页的 Manager Logs 页签
- 目标版本：第一期 MVP
- 本文只描述设计；评审通过后再拆实现计划

## 背景

当前 Console 的 Execution -> Manager Logs 链路为：

    Console
      -> GET /v1/executions/{execution_id}/manager-logs
      -> Control Gateway 查询 current_assignment 和 InstanceManager
      -> Instance Manager GET /v1/internal/logs
      -> 返回 Manager 本地日志尾部

这条链路适合运行态，但不适合终态。Execution 进入终态后，Manager 可能已经释放，
current_assignment、Manager Pod 或 Manager 本地日志都可能不可用。

TKEx dynamic Manager 已经通过共享卷和 zhiyan-agent 将 JSONL 日志上报到智研，日志中包含
execution_id、assignment_id、attempt_id 和 manager_id 等结构化字段。

智研 OpenAPI 已通过实际查询验证可用：

- 项目：Nexus-Agent
- 接入：dev_nexus_manager_log
- 查询接口：POST /log/v2/search/query
- execution_id 等值过滤可用
- 日志字段已展开为结构化字段

设计依据：

- [iWiki 4007943077：智研 OpenAPI 查询、Header、返回格式和限流](https://iwiki.woa.com/p/4007943077)
- [iWiki 4007804560：日志接入、采集 Agent 和配置下发](https://iwiki.woa.com/p/4007804560)
- [iWiki 268627573：Token、projectname、staffname 鉴权规则和错误码](https://iwiki.woa.com/p/268627573)

## 目标

1. Console 只依赖一个 Execution 日志接口，不感知日志后端。
2. starting、running、waiting_input、stopping 等运行态直接读取 Instance Manager。
3. succeeded、failed、killed、lost、expired 等终态读取智研日志。
4. 智研 Token 只存在 Control Gateway 服务端 Secret，不进入浏览器、URL、日志或响应。
5. 智研终态日志延迟入库时，页面显示 ingest_pending，并进行有界重试。
6. 对日志条数、字节数、查询时间窗口和智研 QPS 做后端约束。
7. 保留旧 manager-logs 接口，降低滚动发布期间的兼容风险。

## 非目标

第一期不包含：

- 日志下载、导出或离线任务；
- 任意全文搜索、关键词搜索或跨 Execution 聚合；
- 日志流式订阅；
- 新增日志存储或 Nexus 侧持久化日志缓存；
- 修改 Instance Manager 的日志写入格式；
- 让 Nexus 通过智研 OpenAPI 创建接入点或执行智研侧配置下发；
- Kubernetes/预定义 Manager 的新日志采集方案。

如果某个 Allocation 没有 zhiyan_log 配置，本期不假设它已经具备智研日志，必须返回明确的未配置状态。

## 方案选择

### 方案 A：Console 双源读取

前端根据 Execution 状态直接请求 Instance Manager 或智研。

缺点：

- 前端需要保存智研项目、接入和鉴权协议；
- 状态判断、重试和时间窗口逻辑分散到浏览器；
- 容易把 Token 放入前端或错误响应；
- 两种日志格式难以统一。

不采用。

### 方案 B：Control Gateway 统一选择日志源

Console 调用统一的 /logs 接口，Control Gateway 完成鉴权、状态判断、日志源选择、
智研查询和格式归一化。

优点：

- 延续现有 Execution 访问控制；
- Token 和平台配置留在服务端；
- 终态与运行态的切换只在一个后端边界完成；
- 可统一处理智研最终一致性、限流和错误语义。

采用此方案。

### 方案 C：先将智研日志同步回 Nexus

增加异步同步任务和 Nexus 侧日志存储，页面只读 Nexus。

优点是终态读取稳定，但需要引入新的存储生命周期、清理策略和同步故障处理，超出一期需求。
暂不采用；如果后续需要下载、长期归档或复杂检索，再单独设计。

## 总体架构

    Console ExecutionManagerLogsTab
      |
      | GET /v1/executions/{execution_id}/logs?lines=200
      v
    Control Gateway
      |
      +-- execution:read + Execution ownership check
      |
      +-- load Execution and immutable Allocation provider snapshot
      |
      +-- status router
      |     |
      |     +-- active -> Instance Manager /v1/internal/logs
      |     |
      |     +-- terminal -> Zhiyan OpenAPI /log/v2/search/query
      |
      +-- normalize text/records to ExecutionLogResponse
      v
    Console

日志源选择必须在 Gateway 完成，前端不根据 status 自己分支。

## 状态路由

使用 nexus.distributed.execution.models.TERMINAL_EXECUTION_STATUSES 作为终态唯一来源。

| Execution 状态 | source | availability | 行为 |
|---|---|---|---|
| queued | none | not_started | 尚未分配 Manager，不发起外部查询 |
| assigned | instance_manager | not_started 或 ready | 有 Manager 且可访问时读取，否则返回未就绪 |
| starting | instance_manager | ready 或 unavailable | 读取当前 Manager 尾部 |
| running | instance_manager | ready 或 unavailable | 读取当前 Manager 尾部 |
| waiting_input | instance_manager | ready 或 unavailable | 仍属于运行态，读取当前 Manager |
| stopping | instance_manager | ready 或 unavailable | 停止期间仍优先读取当前 Manager |
| succeeded | zhiyan | ready、ingest_pending 或 unavailable | 按 execution_id 查询智研 |
| failed | zhiyan | ready、ingest_pending 或 unavailable | 按 execution_id 查询智研 |
| killed | zhiyan | ready、ingest_pending 或 unavailable | 按 execution_id 查询智研 |
| lost | zhiyan | ready、ingest_pending 或 unavailable | 按 execution_id 查询智研 |
| expired | zhiyan | ready、ingest_pending 或 unavailable | 按 execution_id 查询智研 |

运行态不能因为 Instance Manager 暂时不可用就静默切换到智研：智研存在上报延迟，可能还没有最新日志。
终态一律查询智研；智研不可用时返回明确的 unavailable/error_code，不回退到 Instance Manager，避免同一个状态在不同刷新时返回两个语义不同的日志源。

## Gateway API

### 请求

    GET /v1/executions/{execution_id}/logs?lines=200

约束：

- 沿用现有 execution:read scope 和 Execution owner/environment 可见性校验；
- lines 默认 200，允许范围 1..1000；
- 未知 query 参数忽略，第一期不支持 query、cursor、start_time、end_time；
- 未授权访问继续使用现有资源隐藏语义，不泄露 Execution 是否存在。

### 响应

    {
      "execution_id": "exe_xxx",
      "execution_status": "running",
      "source": "instance_manager",
      "availability": "ready",
      "partial": false,
      "lines": [
        {
          "timestamp": "2026-08-25T21:52:56.367Z",
          "level": "INFO",
          "message": "manager message",
          "fields": {
            "component": "instance_manager",
            "manager_id": "im_xxx",
            "assignment_id": "assignment_xxx",
            "attempt_id": "attempt_xxx"
          }
        }
      ],
      "line_count": 1,
      "fetched_at": "2026-08-25T22:00:00Z",
      "retry_after_ms": null,
      "error_code": null
    }

字段语义：

- source：none、instance_manager 或 zhiyan；
- availability：not_started、ready、ingest_pending、unavailable；
- partial：由于 lines 限制或智研时间窗口限制，只返回部分日志；
- lines：按时间升序返回，便于页面阅读；
- fields：仅返回受控的结构化字段，不把智研原始响应整体透传到浏览器；
- retry_after_ms：仅在 ingest_pending 时返回；
- error_code：稳定的 Nexus 错误码，不携带 Token、请求头或原始异常堆栈。

建议的稳定错误码：

| error_code | 含义 |
|---|---|
| manager_log_not_started | 尚未有可用 Manager |
| manager_log_unavailable | Manager 查询失败 |
| zhiyan_log_not_configured | Allocation 没有智研接入绑定 |
| zhiyan_log_ingest_pending | 终态已产生，但智研尚未查到日志 |
| zhiyan_log_query_failed | 智研查询失败或超时 |
| zhiyan_log_query_limited | 触发智研限流 |

正常的 not_started、ingest_pending 和空日志使用 HTTP 200；Execution 权限失败沿用现有隐藏语义；
内部协议错误使用 502；输入参数错误使用 400。

## Instance Manager 读取

复用现有接口：

    GET {manager.endpoint_url}/v1/internal/logs?lines={lines}

实现边界：

1. Gateway 从 Execution 的当前 assignment 取得 Manager endpoint；
2. 继续使用内部鉴权请求头；
3. 读取纯文本后按换行拆分为 LogLine；
4. 解析不到时间、级别或结构化字段时保留原始内容到 message；
5. 单次响应最大 1 MiB，超过时截断并设置 partial=true；
6. Manager 不存在或已释放时返回 unavailable，不泄露底层 URL 或异常堆栈。

旧 /v1/executions/{execution_id}/manager-logs 和 /stream 保持不变，供现有调用方兼容使用。

## 智研绑定和配置

### Allocation 绑定

不新增独立日志表。动态 Allocation 已保存不可变的 ProviderConfig snapshot，其中包含 `zhiyan_log.log_name`。
新接口读取 Execution 的 allocation_id，再从 Allocation snapshot 提取：

    {
      "allocation_spec_json": {
        "provider_type": "tkex",
        "provider_config_ref": "tkex/nanjing-prod",
        "provider_config": {
          "zhiyan_log": {
            "log_name": "dev_nexus_manager_log"
          }
        }
      }
    }

这样终态查询不会依赖当前 ProviderConfig，避免后续修改接入名导致历史 Execution 查询错误。当前持久化字段是
`EnvironmentAllocationRow.allocation_spec_json`，`EnvironmentAllocationRecord` 尚未暴露这个 JSON；一期采用
Allocation repository 的专用只读 `zhiyan_log_binding(allocation_id)` 查询并只返回 `log_name`、provider type 和
snapshot 是否完整，不扩大全量 provider snapshot 的公共序列化边界。

该查询不得返回 Provider credential。snapshot 中的 `credential_ref` 只用于判断配置身份，实际 TKEx/API credential
仍由服务端 Secret 解析。

对于已有但没有 snapshot 的 Allocation：

- 如果仍有 Manager，运行态继续读 Instance Manager；
- 进入终态后返回 zhiyan_log_not_configured；
- 不根据当前全局 ProviderConfig 猜测历史 dataflow。

### 普通配置

新增普通配置对象，用于描述智研查询协议，不保存 Token：

    {
      "zhiyan_log_query": {
        "enabled": true,
        "endpoint": "http://openapi.zhiyan.woa.com",
        "project_name": "Nexus-Agent",
        "staff_name": "<approved-service-staff>",
        "credential_ref": "nexus-zhiyan-openapi",
        "storage_type": "elasticsearch",
        "query_window_seconds": 21600,
        "ingest_grace_seconds": 300,
        "max_lines": 1000
      }
    }

`staff_name` 仅为当前联调示例；生产必须使用平台审批通过的服务调用身份，不能把个人企业微信名硬编码为默认值。

`dataflow_name` 不放在全局配置中，优先从 Allocation snapshot 的
`allocation_spec_json.provider_config.zhiyan_log.log_name` 获取。

### Secret 配置

在 DistributedSecrets 中新增独立的智研凭据命名空间：

    {
      "zhiyan_log_credentials": {
        "nexus-zhiyan-openapi": "<token>"
      }
    }

原因：智研 Token 不是 TKEx credential，也不应复用 provider_credentials 或 run_trace_credentials 的语义。
Secret 继续走现有数据库 Secret override、版本 CAS、脱敏查询和滚动重启流程。

启动或配置校验必须 fail close：enabled=true 时，endpoint、project、staff、credential 和 API 版本均需有效。

## 智研查询算法

### 请求构造

    POST {endpoint}/log/v2/search/query

请求头由 Gateway 注入：

    Content-Type: application/json
    x-api-version: v2
    token: <server-side secret>
    projectname: Nexus-Agent
    staffname: <approved-service-staff>

请求体：

    {
      "startTime": 0,
      "endTime": 0,
      "dataflowName": "dev_nexus_manager_log",
      "storageType": "elasticsearch",
      "filter": [
        {
          "key": "execution_id",
          "operator": "eq",
          "value": "exe_xxx"
        }
      ],
      "orderBy": "desc",
      "page": 1,
      "limit": 200
    }

智研按倒序取最新的 lines 条，Gateway 再反转为时间升序；这样和当前 Manager 日志尾部语义一致。

### 时间窗口

`ExecutionRow` 已有 `created_at`、`started_at` 和 `finished_at` 列，但当前 `ExecutionRecord` 和既有 Execution
详情序列化没有暴露后两个字段。新日志路由通过 repository 的专用 log context 读取这些现有列（不新增数据库列，
也不改变既有 Execution 详情响应），再应用以下查询窗口规则：

    terminal_at = finished_at || updated_at
    end_time    = min(now, terminal_at + ingest_grace_seconds)
    start_time  = max(created_at - 60s, end_time - query_window_seconds)

默认查询窗口为 6 小时，避免进入智研较低的长时间窗口 QPS 档位。超过窗口的 Execution 只展示尾部，
并将 partial=true。第一期不为长任务自动拆分多个时间窗口。

### 空结果与最终一致性

终态后首次查询没有记录时：

1. 若距离 terminal_at 小于 ingest_grace_seconds，返回 ingest_pending；
2. Console 按 retry_after_ms 进行最多 3 次刷新，间隔 2s、5s、10s；
3. 超过重试次数仍无数据，返回 unavailable，error_code=zhiyan_log_ingest_pending；
4. 不把“无数据”解释为“Execution 没有日志”。

Gateway 不在单次 HTTP 请求内长时间阻塞等待，避免占用 worker 和放大智研限流。

## Gateway 客户端和限流保护

新增独立的 ZhiyanLogClient，由 Control Gateway 创建并复用 aiohttp session：

- 使用应用生命周期管理 session，禁止每次请求新建连接池；
- 连接和读取超时分别配置，单次总耗时有上限；
- 只允许固定的智研 query path，不接受前端传入任意 URL；
- 只允许 execution_id 等值过滤，不把任意 Lucene/DSL 暴露给前端；
- 将智研 request_id 记录到 Gateway 诊断日志，但不返回 Token 或完整上游响应；
- 对同一 Execution 的前端自动重试做最小间隔保护；
- 收到 429 或智研限流错误时返回 zhiyan_log_query_limited，不立即盲目重试。

第一期不做持久化缓存。前端刷新按钮在请求未完成时禁用，自动重试严格有界；后续若 QPS 仍高，
再增加按 execution_id + window + lines 的短 TTL 进程内缓存和 in-flight 合并。

## Console 行为

ExecutionManagerLogsTab 改为请求 /logs，不再从 current_assignment.manager_id 决定是否发请求。

页面显示：

- 标题下显示 Instance Manager 或 Zhiyan；
- 运行态显示实时日志和手动刷新；
- 终态显示智研日志；
- ingest_pending 显示“Execution 已结束，智研日志入库中”，自动重试；
- not_started 显示“Manager 尚未启动”；
- unavailable 显示稳定错误码和手动重试按钮；
- 日志正文继续使用受控的 pre 展示，第一期不引入搜索框和下载按钮。

前端不读取、存储或传递任何智研 Token、project name 或 dataflow name。

## 可观测性

Control Gateway 增加以下指标：

- nexus_execution_log_query_total{source,result}；
- nexus_zhiyan_log_query_latency_seconds；
- nexus_zhiyan_log_query_records；
- nexus_zhiyan_log_ingest_pending_total；
- nexus_zhiyan_log_query_limited_total；
- nexus_execution_log_response_bytes。

结构化日志只记录 execution_id、assignment_id、attempt_id、source、availability、HTTP 状态、
智研业务 code、上游 request ID、耗时和返回条数。

禁止记录：

- Token；
- 完整请求头；
- 智研原始响应中的全部 message；
- provider credential；
- 未脱敏的 URL 查询串。

## 测试设计

### Control Gateway 单元测试

1. 所有运行态选择 Instance Manager。
2. 所有终态选择 Zhiyan。
3. queued 无 assignment 时不调用任何外部日志服务。
4. Manager 返回纯文本时正确转换为 LogLine。
5. Zhiyan 返回记录时按时间升序返回。
6. Zhiyan 请求包含正确的 dataflow、时间窗口和 execution_id filter。
7. Allocation snapshot 缺失时返回 zhiyan_log_not_configured。
8. 终态空结果返回 ingest_pending，重试次数受限。
9. 429、超时、非零业务 code 映射到稳定错误码。
10. 未授权 principal 无法通过新接口读取 Execution 日志。
11. Token 不出现在响应、异常文本、审计和结构化日志中。

### Console 测试

1. 新页面请求 /v1/executions/{id}/logs。
2. source badge 在 Instance Manager 和 Zhiyan 间切换。
3. ingest_pending 按服务端建议重试，最多三次。
4. not_started、unavailable 和空日志分别展示。
5. 行数选择仍限制在 100、200、500、1000。
6. 旧静态 Console 兼容测试同步更新，不再断言页面直接依赖 manager-logs。

### 集成验证

使用已部署的 Nexus-Agent / dev_nexus_manager_log 接入完成一次真实 Execution：

1. running 阶段页面能读取 Instance Manager；
2. Execution 进入终态后，页面 source 切换为 Zhiyan；
3. Zhiyan 中按 execution_id 能查到结构化日志；
4. Manager 释放后仍可读取终态日志；
5. 终态立即查询时能观察到 ingest_pending 和有界重试；
6. 智研 Token 不出现在浏览器 Network、Console、Gateway 日志或错误响应。

## 发布和回滚

### 发布顺序

1. 发布包含新配置模型和 ZhiyanLogClient 的 Control Gateway 镜像；
2. 写入普通配置和 Secret 配置，校验配置版本及 Secret 引用；
3. 先启用 Gateway 新接口，保持旧 /manager-logs；
4. 在测试环境验证真实智研查询和 Execution owner 隔离；
5. 发布 Console，切换到 /logs；
6. 观察智研 QPS、延迟、ingest_pending 比例和错误码；
7. 稳定后再评估旧接口下线。

### 回滚

- Console 回滚到旧版本即可恢复 Instance Manager 读取；
- Gateway 新接口可以保留，不影响旧 /manager-logs；
- 删除或禁用智研配置后，终态页面明确显示 zhiyan_log_not_configured，不影响 Execution 生命周期；
- 不回滚或删除已上报到智研的日志，也不执行数据清理。

## 验收标准

本设计进入开发后的 MVP 必须满足：

1. 页面只有一个日志读取入口；
2. 运行态直接读 Instance Manager；
3. 终态按不可变 Allocation snapshot 的 dataflow 查询智研；
4. Manager 释放后终态日志仍可读；
5. 智研未及时入库时页面不会误报“无日志”；
6. 查询接口和 Secret 均由 Gateway 控制，浏览器不接触智研凭证；
7. 请求、响应、日志和指标均有大小、时间窗口和 QPS 边界；
8. 旧 Manager logs API 在滚动发布期间保持兼容；
9. 真实智研接入和按 execution_id 过滤通过集成验证。

## 待评审决策

请重点确认以下 4 项：

1. 是否接受保留旧 /manager-logs，新增 /logs 作为 Execution 页签唯一入口；
2. 是否接受只支持 Allocation snapshot 中已配置 zhiyan_log 的动态 TKEx Manager，其他 Manager 明确显示未配置；
3. 是否接受第一期只展示最近 6 小时尾部日志，不做长任务跨窗口分页；
4. 是否接受智研 Token 使用独立的 DistributedSecrets.zhiyan_log_credentials 命名空间。

评审通过这 4 项后，再进入实现计划和代码开发。
