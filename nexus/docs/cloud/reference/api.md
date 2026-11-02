# API 参考

Nexus Cloud 对普通用户开放 submission、execution、Batch、结果、Live 交互、Environment 和 Shared Access API。Single、Batch
和 Live 均支持通过 `nexus submit` 创建。应用直接调用 HTTP API 的接入示例见 [Live 指南](../live.md)；
Live 创建后使用事件、命令和结果 API 交互。CLI 已封装代码和依赖上传，需要上传自定义代码时优先使用 CLI。

## 入口与认证

先按[认证与运行环境](../access-and-environments.md)选择个人 PAT 或应用身份，设置 `CONTROL_BASE_URL`、
`APPLICATION_BASE_URL` 及 `nexus_request` / `nexus_json`。基址可以包含发布前缀，但不含末尾 `/v1`；下方路径均包含 `/v1`。

```bash
set -euo pipefail

nexus_json GET "$CONTROL_BASE_URL/v1/me" | jq .
```

接口成功以 HTTP 状态和响应体共同判断。SGW 的 `errcode` 错误可能通过 HTTP `200` 返回；个人 API 与应用 API 的入口不能混用。

## 提交 API

| 方法 | 路径 | 用途 |
| --- | --- | --- |
| `GET` | `/v1/capabilities` | 查询 submission、Environment provider 和 workload image 能力 |
| `POST` | `/v1/submissions/single` | 提交 Single |
| `POST` | `/v1/submissions/live` | 提交 Live |
| `POST` | `/v1/submissions/batch` | 提交 Batch |

三类 submission 都使用 `submission_version=1`、`idempotency_key`、`environment_id` 和对应 task/code 配置。服务端返回 HTTP
`202`，常见字段包括：

```json
{
  "execution_id": "exe_xxx",
  "status": "queued",
  "phase": "queued",
  "reused": false,
  "control_url": "https://.../v1/executions/exe_xxx",
  "result_url": "https://.../v1/executions/exe_xxx/result",
  "task_web_ui": "https://.../v1/console/executions/exe_xxx"
}
```

Batch 还返回 `batch_id`、`batch_url`；Live 还返回 `events_url`、`command_url`。相同 idempotency key 与相同内容会复用原任务；
相同 key 对应不同内容返回 `submission_idempotency_conflict`。

source-package 请求涉及本地打包、Nexus wheel、`uv.lock` 和多项 Artifact 引用，不建议手写 JSON；使用 `nexus submit` 生成。
`POST /v1/executions` 是管理员/内部底层接口，不是普通用户提交入口。

## Execution API

### Control Gateway

| 方法 | 路径 | 用途 |
| --- | --- | --- |
| `GET` | `/v1/executions` | 分页查询可访问 execution |
| `GET` | `/v1/executions/{execution_id}` | 查询详情 |
| `GET` | `/v1/executions/{execution_id}/control-events` | 查询控制面事件 |
| `GET` | `/v1/executions/{execution_id}/logs?lines=200` | 查询任务最近日志及日志可用性 |
| `GET` | `/v1/executions/{execution_id}/build-logs?lines=200` | 查询任务关联的依赖构建日志 |
| `GET` | `/v1/builds/{build_id}/logs?lines=200` | 查询指定构建的日志 |
| `POST` | `/v1/executions/{execution_id}/stop` | 停止 execution 或 Batch parent |

日志的页面入口、请求示例、响应字段和重试方式见[查看任务日志](../runs-and-results.md#查看任务日志)。

列表默认使用 cursor 分页：

```bash
nexus_json GET "$CONTROL_BASE_URL/v1/executions?top_level=1&summary=1&limit=100" | jq .
```

常用参数：

| 参数 | 说明 |
| --- | --- |
| `limit` | 每页数量，默认 100，最大 500 |
| `cursor` | 上一页返回的 `next_cursor` |
| `status` | execution 状态 |
| `mode` | `job` 或 `live` |
| `spec_type` | `nexus_single`、`nexus_live` 或 `nexus_batch` |
| `q` | 按 execution id、manager id、spec type 等搜索 |
| `top_level=1` | 只列顶层任务，不列 Batch child |
| `summary=1` | 返回列表摘要 |
| `include_total=1` | 额外计算总数；大列表不建议每页使用 |
| `owner` | 只看指定 owner；只能选择自己或已直接授权给自己的 owner |

`offset` 仅为兼容旧客户端保留，不能与 `cursor` 同时使用。新代码应持续读取 `next_cursor` 直到它为空。

停止任务：

```bash
nexus_json POST "$CONTROL_BASE_URL/v1/executions/exe_xxx/stop" | jq .
```

中断本地轮询不会停止远端任务；只有调用 Stop 或在控制台点击 Stop 才会发起终止。

### Application Gateway

| 方法 | 路径 | 用途 |
| --- | --- | --- |
| `GET` | `/v1/executions/{execution_id}/result` | 读取 Single/Live 完整 `RunOutput` |
| `GET` | `/v1/executions/{execution_id}/info` | 读取 active Live 的 driver/runtime 信息 |
| `GET` | `/v1/executions/{execution_id}/live/events?cursor=0` | 读取 Live SSE |
| `POST` | `/v1/executions/{execution_id}/commands` | 向 Live 提交命令 |
| `ANY` | `/v1/executions/{execution_id}/runtimes/{runtime_name}/proxy/{path}` | Runtime proxy |

Control Gateway 详情中的 `result_json` 通常只是 external pointer，不应作为业务结果解析。按当前已授权的 Application 入口调用：

```bash
nexus_json GET "$APPLICATION_BASE_URL/v1/executions/exe_xxx/result" | jq .
```

如果 execution 在形成 `RunOutput` 前失败，可能没有 result pointer，应读取 execution 详情和 control events 中的错误。

## Batch API

Batch 以 parent `execution_id` 作为路径身份：

| 方法 | 路径 | 用途 |
| --- | --- | --- |
| `GET` | `/v1/executions/{execution_id}/batch` | Batch 汇总 |
| `GET` | `/v1/executions/{execution_id}/batch/items` | 分页列 child items |
| `GET` | `/v1/executions/{execution_id}/batch/items/{item_id}` | 单个 item |
| `GET` | `/v1/executions/{execution_id}/batch/items/{item_id}/attempts/{attempt_no}` | 某次 attempt |

`batch/items` 支持 `limit`（最大 500）、`cursor`、`status`、`item_id_prefix`、`child_execution_id` 和 `error_code`。响应使用
`next_cursor` 分页。Batch parent 的 `result_url` 不是所有 child 结果的聚合 JSON；批量导出见
[任务与结果](../runs-and-results.md#导出标准-jsonl)。

## Environment 与 Trace API

| 方法 | 路径 | 用途 |
| --- | --- | --- |
| `GET` | `/v1/me` | 当前 principal |
| `GET` | `/v1/environment-catalog` | 可申请配额的 Environment 列表 |
| `GET` | `/v1/my-environments` | 当前用户 Environment 与额度 |
| `PUT` | `/v1/my-environments/{environment_id}/quota` | 申请默认配额 |
| `GET` | `/v1/access/run-trace-backends` | 当前用户可选 backend 与默认顺序 |
| `POST` | `/v1/my-environments/{environment_id}/trace-database-credentials` | Create/Reset 只读账号 |

创建或重置 Trace Database 凭据的 body：

```json
{
  "backend_name": "tdsql",
  "operation": "create"
}
```

`operation` 只能是 `create` 或 `reset`。响应包含一次性密码并设置 `Cache-Control: no-store`；保存后不要写入文档或日志。

## Shared Access API

Shared Access 是 owner 到 grantee 的直接授权：

| 方法 | 路径 | 用途 |
| --- | --- | --- |
| `GET` | `/v1/access/users` | 获取可选用户及 `principal_id` |
| `GET` | `/v1/access/principal-grants` | 普通用户列出自己发出的授权 |
| `PUT` | `/v1/access/principal-grants/{owner_principal_id}/{grantee_principal_id}` | 创建/确保授权存在 |
| `DELETE` | `/v1/access/principal-grants/{owner_principal_id}/{grantee_principal_id}` | 撤销授权 |

普通用户只能以自己的 `principal_id` 作为 owner；越权 owner 会返回 `resource_not_found`。授权不传递，也不共享 Environment、Quota、
Token、Run Trace Backend Grant 或 Trace Database 凭据。

```bash
OWNER_PRINCIPAL_ID='<from-v1-me>'
GRANTEE_PRINCIPAL_ID='<from-v1-access-users>'

nexus_json PUT "$CONTROL_BASE_URL/v1/access/principal-grants/$OWNER_PRINCIPAL_ID/$GRANTEE_PRINCIPAL_ID" | jq .
```

被授权者查询时无需额外 header；`GET /v1/executions` 自动包含已授权 owner 的任务，也可以传 `owner=<principal_id>` 过滤。

## 状态与常见错误

Execution 状态取值：

- 非终态：`queued`、`assigned`、`starting`、`running`、`waiting_input`、`stopping`；其中 `waiting_input` 用于 Live。
- 终态：`succeeded`、`failed`、`killed`、`lost`、`expired`。

CLI `--wait` 的返回结构、Control 状态和导出 JSONL 状态的区别见[任务与结果](../runs-and-results.md)。

常见错误：

| 错误 | 含义 |
| --- | --- |
| `401 Unauthorized` | 当前调用身份无效，按个人 PAT 或应用身份分别检查 |
| `resource_not_found` | 资源不存在，或当前 principal 无权知道它存在 |
| `route_not_active` | 路由当前不可用，检查 Execution 状态 |
| `endpoint_not_ready` | assignment 已创建，但运行端点未 ready |
| `execution_closed` | execution 已终态或 Live route 已关闭 |
| `submission_idempotency_conflict` | 同一 idempotency key 对应不同请求内容 |
| `run_trace_backend_not_available` | backend 不存在、隐藏或未授权 |

Live 的 CLI/HTTP 提交、就绪判断、SSE 续传、输入和停止流程统一见 [Live 指南](../live.md)。
