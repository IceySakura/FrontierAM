# Live 指南

Live 用于需要持续观察执行过程、接收用户输入的云端任务。`nexus submit live` 与直接调用 HTTP submission API 创建的是
同一种 Cloud Live Execution；创建后都通过 Application API 读取事件、提交命令和取得结果。

```text
选择 Environment 与认证方式 → 提交 → 等待就绪
  → 读取 SSE → 收到 input.required → 提交 input.submit → 继续读取 SSE
  → 任务完成并读取结果，或主动 Stop 并等待终态
```

本页按这一流程组织。API 路径总表见 [API 参考](reference/api.md)，控制台操作及 JSONL 导出见
[任务与结果](runs-and-results.md)。

## 接入准备

在 [Nexus Cloud](https://cloud.nexus-agent.woa.com) 的 **My Environments** 中确认目标 Environment 已授权、Grant 已启用，
并具备 `submission:live`、`execution:read`、`execution:command`、`execution:stop` 及足够的资源额度。

| 运行环境 | 提交方式 |
| --- | --- |
| Dynamic | CLI 默认上传当前 Nexus 项目并准备依赖，适合自定义 task/agent |
| Predefined | 指定已有 pool，使用 `--config-only`；模块和依赖须已部署到 Manager |

Environment ID、pool 和可用代码以目标环境为准。代码、依赖及镜像的详细约束见
[Batch 指南](submit-batch.md#代码与运行环境)。下面分别给出 Dynamic CLI 示例和 Predefined HTTP 示例。

先按[认证与运行环境](access-and-environments.md)选择个人 PAT 或应用身份，设置 `CONTROL_BASE_URL`、
`APPLICATION_BASE_URL`，并定义 `nexus_request`、`nexus_json` 请求函数。后续步骤在同一个 Bash 终端执行，
需要 `curl`、`jq`、`openssl`；直接调用 HTTP API 不需要安装 Nexus。

每个逻辑任务只生成一次幂等键，并保存提交响应：

```bash
set -euo pipefail

umask 077
OUTPUT_DIR="$(mktemp -d "${TMPDIR:-/tmp}/nexus-cloud-live.XXXXXX")"
IDEMPOTENCY_KEY="live-$(openssl rand -hex 16)"
printf '%s\n' "$IDEMPOTENCY_KEY" > "$OUTPUT_DIR/idempotency-key"
```

网络超时后复用原幂等键和相同配置，不要重新执行生成 key 的命令。相同 key 对应不同请求内容会返回
`submission_idempotency_conflict`。日志和结果也可能含业务数据，输出目录应按私有文件管理。

**以下两种创建方式任选一种**，完成后进入共同的「等待就绪」章节。

<a id="方式一cli-与个人-pat"></a>

## 方式一：CLI 提交

CLI 使用个人 PAT。先按[快速开始](quickstart.md)准备 Nexus 项目与凭据，并在项目根目录执行：

```bash
uv sync --extra cloud
```

CLI 会处理代码和依赖上传；本页的交互示例在云端执行。

### 准备可交互的 RunInput

示例 [cloud-live.yaml](../examples/configs/cloud-live.yaml) 使用
[cloud_live.py](../examples/configs/cloud_live.py) 中的 Agent：依次等待两次用户输入，将收到的文本放入最终结果的
`agent_output.messages`。它不调用模型、不创建 Runtime，无需模型或 Runtime 凭据。

```yaml
imports:
  - nexus.extensions.tasks.identity
  - docs.examples.configs.cloud_live

task_name: identity
agent_name: cloud_live_collect_inputs
agent_config: {}

runner:
  timeout: 600
```

示例通过 Dynamic Environment 上传当前项目执行。使用 Predefined 时，必须先由管理员部署相应模块。
实际业务使用与 Single 相同的 RunInput；只有会调用 `get_user_input` 的 task/agent 才会等待用户输入。将普通单轮任务改为
Live 提交不会自动使它变成多轮任务。

### 提交任务

使用接入准备中选定的 API 地址，把 `NEXUS_ENVIRONMENT` 设置为同一部署下已授权的 Dynamic Environment ID：

```bash
export NEXUS_ENVIRONMENT='<dynamic-environment-id>'

uv run nexus submit live \
  docs/examples/configs/cloud-live.yaml \
  --environment "$NEXUS_ENVIRONMENT" \
  --control-url "$CONTROL_BASE_URL" \
  --taihu-token-file "$TAIHU_TOKEN_FILE" \
  --idempotency-key "$IDEMPOTENCY_KEY" \
  --execution-ttl-seconds 600 \
  --json > "$OUTPUT_DIR/submission.json"

jq . "$OUTPUT_DIR/submission.json"
```

CLI 在任务被接受后返回，不会留在终端中等待用户输入；Live CLI 不支持 `--wait`。保存响应后，继续[等待就绪](#等待就绪)。

<a id="方式二http-api-与应用身份"></a>

## 方式二：HTTP API 提交

个人 HTTP 脚本可以使用 PAT；线上应用使用已绑定用户的应用身份。两者提交相同的 JSON，请求头和授权基址由
[认证与运行环境](access-and-environments.md#选择认证方式)中的接入方式决定。

### 提交 config-only 任务

下例使用 `predefined-local` 的 `interactive` pool，只引用环境中已部署的内置模块。它使用 Mock client 返回
`CLOUD_LIVE_ECHO_OK`，用于验证 HTTP 提交与结果链路；它会直接完成，不产生 `input.required`。

```bash
cat > "$OUTPUT_DIR/request.json" <<JSON
{
  "submission_version": 1,
  "idempotency_key": "$IDEMPOTENCY_KEY",
  "environment_id": "predefined-local",
  "pool": "interactive",
  "code": {"mode": "config_only"},
  "run_input": {
    "imports": ["nexus.extensions"],
    "task_name": "identity",
    "agent_name": "single_turn",
    "agent_config": {
      "prompt": [{"role": "user", "content": "Return exactly CLOUD_LIVE_ECHO_OK"}]
    },
    "openai_clients": {
      "main": {
        "name": "mock",
        "config": {
          "client_args": {"responses": ["CLOUD_LIVE_ECHO_OK"]},
          "request_args": {"model": "mock-model"}
        }
      }
    }
  },
  "live": {"working_dir": "/workspace"},
  "queue_ttl_seconds": 300,
  "execution_ttl_seconds": 600,
  "stop_timeout_seconds": 60
}
JSON

nexus_json POST "$CONTROL_BASE_URL/v1/submissions/live" \
  -H 'Content-Type: application/json' \
  --data-binary @"$OUTPUT_DIR/request.json" > "$OUTPUT_DIR/submission.json"

jq . "$OUTPUT_DIR/submission.json"
jq -e 'has("errcode") | not' "$OUTPUT_DIR/submission.json"
```

`nexus_json` 同时检查 HTTP 和 SGW 错误，成功后才继续。业务交互任务可将 `run_input` 换成完整的 task、agent、runtime、model 和
processor 配置；`live` 是独立外层字段。config-only 不允许上传源码或申请新的 Manager 资源，所有引用模块和依赖都必须已在
目标 pool 中安装。需要源码上传时使用 CLI，避免手工组装 Artifact 和依赖引用。

## 等待就绪

以下步骤适用于上述两种创建方式，沿用接入准备中选定的请求函数和 API 基址。基址不含末尾 `/v1`：

```bash
EXECUTION_ID="$(jq -er '.execution_id' "$OUTPUT_DIR/submission.json")"
CONTROL_URL="$CONTROL_BASE_URL/v1/executions/$EXECUTION_ID"
EVENTS_URL="$APPLICATION_BASE_URL/v1/executions/$EXECUTION_ID/live/events"
COMMAND_URL="$APPLICATION_BASE_URL/v1/executions/$EXECUTION_ID/commands"
RESULT_URL="$APPLICATION_BASE_URL/v1/executions/$EXECUTION_ID/result"
```

统一使用 `execution_id` 管理任务，并保存业务 ID 与它的对应关系。上面的地址使用当前身份已获授权的 API 基址；
返回 URL 的使用规则见[程序入口与地址](access-and-environments.md#程序入口与地址)。

提交返回 `202` 只表示已受理，不表示 Manager 已 ready。先轮询 Control 状态：

```bash
wait_live_ready() {
  local execution status attempt
  for ((attempt=0; attempt<300; attempt++)); do
    execution="$(nexus_json GET "$CONTROL_URL")" || return
    status="$(printf '%s' "$execution" | jq -r .status)"
    case "$status" in
      running|waiting_input|succeeded)
        printf '%s\n' "$execution" | jq '{execution_id, status, phase}'
        return 0 ;;
      failed|killed|lost|expired)
        printf '%s\n' "$execution" | jq '{execution_id, status, phase, error_json}'
        return 1 ;;
    esac
    sleep 2
  done
  printf '等待就绪超时，请查询 Control 状态；任务未被自动停止。\n' >&2
  return 1
}
wait_live_ready
```

只在等待成功后继续。任务已 `succeeded` 时直接进入「读取结果」，跳过 SSE 和 command；HTTP Echo 示例通常走这条路径。
活动任务进入 `running` 或 `waiting_input` 后可以连接 SSE。Route active 和 endpoint ready 不是同一状态，首次 Application
请求若返回 `503 endpoint_not_ready`，短暂退避后重试。

## 读取事件与多轮输入

### SSE 与断线续传

从 `cursor=0` 开始读取。下面把 SSE 放到后台并保存事件，便于在同一个终端发送输入：

```bash
nexus_request GET "${EVENTS_URL}?cursor=0" -N > "$OUTPUT_DIR/events.sse" &
SSE_PID=$!
tail -n 30 "$OUTPUT_DIR/events.sse"
```

事件陆续到达后可以再次 `tail` 查看。每条 SSE 包含 `id`、`event` 和 JSON `data`。业务常关注 `input.required`、
`agent.event`、`run.output`、`run.error`；以 Control 的 Execution 状态判断任务是否结束，避免混用 SSE 内层状态与控制面状态。

断线时保存**最后成功处理**的 SSE event id，以它作为新的 `cursor` 重连；不要每次都从 0 开始。新的连接继续使用同一种认证方式：

```bash
LAST_EVENT_ID='<last-successfully-processed-sse-id>'
nexus_request GET "${EVENTS_URL}?cursor=${LAST_EVENT_ID}" -N
```

终态后的事件回放取决于部署的 Run Trace backend 和保留情况。收到 `410 trace_events_not_retained` 时读取 Control 状态和
最终结果，不要无限重连；断开 SSE 连接不会停止任务。

### 提交用户输入

当 CLI 示例的事件中出现 `input.required`、`request.hint` 为 `Input 1/2` 时，提交第一条输入：

```bash
nexus_json POST "$COMMAND_URL" \
  -H 'Content-Type: application/json' \
  -d '{"type":"input.submit","payload":{"text":"Alpha"},"idempotency_key":"turn-alpha-001"}' | jq .
```

命令返回 `202` 只表示进入队列，不代表本轮已被消费。继续查看事件文件，**收到新一轮 `Input 2/2` 请求后**再提交：

```bash
nexus_json POST "$COMMAND_URL" \
  -H 'Content-Type: application/json' \
  -d '{"type":"input.submit","payload":{"text":"Beta"},"idempotency_key":"turn-beta-002"}' | jq .
```

每次逻辑输入使用独立、稳定的 command `idempotency_key`，网络超时后用原 key 和原 payload 重试。不要因暂时仍看到
`waiting_input` 就重复投递或抢先提交下一轮，应等待对应的新输入请求。

向等待输入的 Agent 提交空文本会使 `get_user_input` 返回 `None`；是否正常结束由 task/agent 决定。需要强制停止任务时使用
Control Stop，而非依赖空输入。本页示例收到两条文本后正常结束，也支持空输入提前结束。

## Runtime 信息与代理

活动任务的 Runtime 信息通过 Application API 读取：

```bash
nexus_json GET "$APPLICATION_BASE_URL/v1/executions/$EXECUTION_ID/info" | jq '.runtime_info.runtimes'
```

本页两个示例都不创建 Runtime，因此没有可代理的 Runtime。业务任务包含 Runtime 时，用返回的 runtime name 访问：

```text
/v1/executions/{execution_id}/runtimes/{runtime_name}/proxy/{path}
```

代理目标路径由 Runtime 提供，HTTP/WebSocket 的支持还取决于入口网关。Runtime 仅在任务活动期间可用；终态后的业务结果读取
不应依赖 Runtime proxy。

## 读取结果与停止任务

任务可能经历 `queued → assigned → starting → running ↔ waiting_input`，最终进入
`succeeded`、`failed`、`killed`、`lost` 或 `expired`。查询状态：

```bash
nexus_json GET "$CONTROL_URL" | jq '{execution_id, status, phase, error_json}'
```

进入 `succeeded` 后读取完整 `RunOutput`，不要把 Control 详情中的 external result pointer 当作业务结果：

```bash
nexus_json GET "$RESULT_URL" > "$OUTPUT_DIR/result.json"
jq '.agent_output' "$OUTPUT_DIR/result.json"
```

CLI 两轮示例的 `agent_output.messages` 应为 `["Alpha", "Beta"]`；HTTP Echo 示例的 `agent_output.response` 应为
`CLOUD_LIVE_ECHO_OK`。若在形成 `RunOutput` 前失败，可能没有最终结果，应读取 `error_json` 和 Control events。

停止仍在运行的任务：

```bash
nexus_json POST "${CONTROL_URL}/stop" | jq '{execution_id, status, stop_reason}'
```

Stop 是异步操作，响应为 `stopping` 时继续查询，直到进入 `killed` 或其他终态。结束示例后关闭仍在运行的 SSE 后台读取进程：

```bash
if [ -n "${SSE_PID:-}" ]; then
  kill "$SSE_PID" 2>/dev/null || true
  wait "$SSE_PID" 2>/dev/null || true
fi
```

停止本地请求、关闭终端或中断状态轮询都不会自动停止云端任务。

## 常见错误

| HTTP 状态或错误 | 处理方式 |
| --- | --- |
| HTTP `200` 且响应含 `errcode` | SGW 拒绝了请求；检查发布前缀、应用签名、时间与 API 订阅 |
| `401 authentication_required` | 检查入口及所选认证方式的请求头 |
| `401 invalid_tai_app_identity` | 检查应用签名、时间、订阅及 `app_id` 绑定；所有应用均失败时由 API 负责人检查服务端配置 |
| `401 ambiguous_authentication` | 不要混用太湖身份与 Nexus 内部 Token、委托头 |
| `403 scope_required` | 补齐 Live 提交、查询、命令或停止所需权限 |
| `404 environment_grant_not_found` | 为调用用户或应用绑定用户配置目标 Environment Grant |
| `409 environment_grant_disabled` | 启用目标 Environment Grant |
| `409 environment_scope_required` | 在 Grant 中增加 `submission:live` |
| `409 submission_idempotency_conflict` | 恢复原 key 对应的请求；新逻辑任务使用新 key |
| `422` | 根据响应修正 submission、`run_input` 或 `live` 配置 |
| `503 endpoint_not_ready` | 端点尚未就绪，短暂退避后重试，并检查 Control 状态 |
| `execution_closed` | 任务已结束，读取终态和结果，停止发送命令或访问 Runtime |
| `410 trace_events_not_retained` | 事件无法继续回放，读取 Control 状态和最终结果 |

任务长时间停在 `queued`、依赖构建失败等通用问题见[常见问题](troubleshooting.md)。
