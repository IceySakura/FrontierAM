# 任务与结果

提交成功后先保存 `execution_id`，再查询状态、查看日志、读取结果。Batch 还返回 `batch_id`，但 Batch API 的路径使用 **parent 的
`execution_id`**；child 有各自的 `execution_id` 和结果。业务侧 `run_id` 与这些平台标识应分别保存。

## 提交响应与幂等重试

<a id="cli-返回"></a>

### 提交返回值

| 提交方式 | 何时返回 | 最终状态从哪里读取 |
| --- | --- | --- |
| Single/Batch CLI，不带 `--wait` | 请求被接受后 | 后续 Control 查询的 `status` |
| Single/Batch CLI，带 `--wait --json` | 先在 stderr 打印 accepted 通知，终态后向 stdout 输出 JSON | 最终 JSON 的 `terminal.status` |
| Live CLI | 请求被接受后；不支持 `--wait` | Control 查询和 Live 交互流程 |
| HTTP submission | `202` 表示请求被接受 | 后续 Control 查询的 `status` |

带 `--wait --json` 的 Batch 响应示例：

```json
{
  "execution_id": "exe_xxx",
  "batch_id": "batch_xxx",
  "status": "queued",
  "batch_url": "https://...",
  "result_url": "https://...",
  "terminal": {"status": "succeeded"}
}
```

这里顶层 `status` 是初次受理时的状态，`terminal.status` 才是等待得到的最终状态。单独查询 Control 返回的 `status` 则是当前状态。

### 幂等重试

每个逻辑任务先生成并保存一次 `IDEMPOTENCY_KEY`。相同 key 和相同内容会复用原任务，内容不同会被拒绝。
提交响应丢失时用原 key、原输入重试；已经取得 `execution_id` 后通常直接查原任务。修改配置或开始新任务时使用新 key。
中断等待只影响本地进程，不会停止远端任务。

## 查询任务状态

<a id="控制台"></a>

### 从控制台查询状态

打开 [Web 控制台](https://cloud.nexus-agent.woa.com/v1/console)：

1. 在 **Runs** 的搜索框输入 `execution_id`，找到顶层任务。
2. 查看 **Status**，点击任务行进入详情。
3. Batch 使用 **Batches** 查看汇总与 child 列表；选择 child 后查看它自己的状态和结果。

![在 dev 的 Runs 页面按 execution_id 查找任务](../assets/images/nexus-cloud/runs-list.png)

图为 dev 控制台中的实际测试任务。不同角色可见的导航菜单可能不同，查找和进入任务详情的方式相同。

### 通过 API 查询状态

程序查询先按[认证与运行环境](access-and-environments.md)设置 API 基址与 `nexus_json`：

```bash
set -euo pipefail

EXECUTION_ID='<execution-id>'
nexus_json GET "$CONTROL_BASE_URL/v1/executions/$EXECUTION_ID" \
  | jq '{execution_id, status, phase, error_json}'
```

### 理解状态与阶段

| Execution 状态 | 含义与下一步 |
| --- | --- |
| `queued`、`assigned`、`starting` | 尚未就绪，继续查看 `phase` 和 Control Events；长时间无进展时按[常见问题](troubleshooting.md)排查 |
| `running` | 正在执行 |
| `waiting_input` | Live 等待用户输入，按 [Live 指南](live.md#读取事件与多轮输入)处理 |
| `stopping` | 停止请求正在处理，继续等终态 |
| `succeeded` | 正常完成，读取结果并核对业务输出 |
| `failed`、`killed`、`lost`、`expired` | 已终态，查看原因；在形成 RunOutput 前失败时可能没有最终结果 |

`phase` 用于解释任务当前所处的细分阶段，不能用“已受理”或“Manager Ready”代替任务成功判据：

| `phase` | 含义与下一步 |
| --- | --- |
| `building_dependencies` | 依赖构建排队、进行中或失败后等待自动重试；[查看构建日志](#查看构建日志) |
| `dependency_build_failed` | 依赖构建已失败且不再重试；查看构建日志与[依赖构建失败](troubleshooting.md#依赖构建失败) |
| `waiting_resources` | 正在申请或拉起运行资源（Environment Allocation / Manager Pod） |
| `waiting_policy` | 依赖与运行资源已就绪，被调度策略门禁暂缓（`status` 仍为 `queued`）；查询 `/scheduler-policy` 查看 `reason_code` 与 `next_schedule_at`，详见[状态与原因查询](../reference/cloud/scheduler-policies.md#状态与原因查询) |
| `queued` | 依赖与运行资源已就绪且未被调度策略拦截，等待调度（空闲 Manager 执行槽位或用户并发额度） |
| `generating` | Batch 正在执行 Generator 生成或发布任务列表 |
| `materializing` | Batch 正在将任务列表物化为 items |
| `running_children` | Batch 子任务正在派发或执行中；可查 `/batch?include_status_counts=true` 查看 `child_execution_status_counts` 与 `child_waiting_policy_count`，单条子任务阶段需查具体 child 的 `execution_id` |
| `finalizing` | Batch 停止中或已关闭提交，正在汇总收尾尚未结束的 child |
| `uploading_result` | 执行体已完成，正在上传结果 Artifact |
| `releasing` | 正在排空或释放关联的运行资源（Environment Allocation） |
| `assigned`、`starting`、`running`、`waiting_input`、`stopping`、`succeeded`、`failed`、`killed`、`lost`、`expired` | 当前无上述特殊子阶段，直接透传 `status` 同名值；含义与上方 `Execution 状态` 表一致 |
| `unknown` | 状态投影异常，结合 `status`、`error_json` 与 Control Events 排查 |

当任务处于 `phase = waiting_policy`（或 Batch 汇总 `child_waiting_policy_count > 0`）时，可查询 `/scheduler-policy` 查看排队期门禁原因；当任务已进入 `running`（或 `running_children`）需确认运行期模型请求是否受共享请求预算限速或暂停（运行期限流不会改变 `phase`）时，可查询 `/request-policy`。两个接口均支持传入 Single、Batch child 或 Batch parent 的 `execution_id`，字段与 `reason_code` 说明见 [Scheduler 策略与潮汐容量准入：状态与原因查询](../reference/cloud/scheduler-policies.md#状态与原因查询)。

### 查询控制面事件

需要确认何时调度、启动、重试或停止时，在任务详情打开 **Control Events**，或通过 API 查询：

```bash
nexus_json GET "$CONTROL_BASE_URL/v1/executions/$EXECUTION_ID/control-events?cursor=0&limit=100" \
  | jq '.items[] | {seq, created_at, type, assignment_id, attempt_id, payload}'

# 查看策略暂缓、实际分配与任务过期事件；这些事件不能单独确定过期原因
nexus_json GET "$CONTROL_BASE_URL/v1/executions/$EXECUTION_ID/control-events?cursor=0&limit=100" \
  | jq '.items[] | select(.type == "scheduler.policy_deferred" or .type == "scheduler.policy_applied" or .type == "execution.status.changed") | {seq, created_at, type, payload}'
```

接口返回 `items`；继续读取时，把已处理的最后一条事件的 `seq` 作为下一次 `cursor`。

`scheduler.policy_deferred` 记录某次调度被策略暂缓；`scheduler.policy_applied` 只在成功创建 Assignment 时产生，不是策略恢复放行事件。
策略放行后，任务仍可能因没有可用 Manager 槽位而保持 `queued`，随后达到 queue TTL 并过期。
因此，`policy_deferred` 之后没有 `policy_applied`，只能证明任务曾被策略暂缓，不能据此认定它一直受策略阻塞直到过期。
确定过期原因还需结合排队期间留存的 `phase`、策略查询结果或调度日志；仅有上述事件时，应保留原因未知的结论。

## 查看任务日志

排查时先区分需要读取的信息：

| 信息 | 回答什么问题 | 入口 |
| --- | --- | --- |
| 任务日志 | 执行进程最近在做什么、出现了什么异常 | 详情页 **Logs** 或 Control `/logs` |
| 控制面事件 | 何时调度、启动、重试、停止或进入终态 | **Control Events** 或 Control `/control-events` |
| 构建日志 | 依赖安装或 wheel 构建卡在哪里 | **Builds** 详情或 Control `/build-logs` |
| 执行轨迹与业务结果 | Agent 的交互过程、任务输出是否满足预期 | Live 的 **Live Events**、[明细轨迹](#查询明细轨迹)和 [Spec / Result](#从控制台读取结果) |

任务日志是有限行数的进程日志，不保证包含完整的模型对话、工具调用或最终 `RunOutput`。

### 直接在页面查看

1. 打开 [Web 控制台](https://cloud.nexus-agent.woa.com/v1/console)，在 **Runs** 中搜索 `execution_id`，进入详情。
2. 点击 **Logs** 页签，默认加载最近 **200 lines**。
3. 按需选择 **100 / 200 / 500 / 1000 lines**，点击日志区域右上角的 **Refresh** 获取最新内容。
4. 查看 **Source**：运行期间通常是 **Instance Manager**；终态后从 **Zhiyan**（智研）读取已采集日志，需部署已配置采集与查询。

Batch 排查单条任务时，从 **Batches → 目标 Batch** 的列表点击 **Child execution**，再进入该 child 的 **Logs**。
点击 **Item** 可以查看重试历史，再选择对应 attempt 的 **Execution**。排查 Batch 生成或调度过程时，在 **Runs** 中打开
parent 的 `execution_id` 查看日志；parent 日志不汇总所有 child 的日志。

Logs 在打开页签、切换行数时加载，正常显示后可手动 **Refresh**；它不是持续推送的日志流。
终态后显示 **Execution finished; Zhiyan logs are still arriving** 时，页面在可见状态下按 2、5、10 秒间隔最多自动重试三次，
之后可手动刷新。Batch 汇总页约每 10 秒更新 parent 状态与汇总，不代表 child 列表、child 详情或 Logs 同步刷新。

### 通过 API 查询日志

先按[认证与运行环境](access-and-environments.md)设置 `CONTROL_BASE_URL`、认证方式和 `nexus_json`，在同一 Bash 终端执行：

```bash
set -euo pipefail

EXECUTION_ID='<execution-id>'
umask 077
LOG_OUTPUT_DIR="$(mktemp -d "${TMPDIR:-/tmp}/nexus-logs.XXXXXX")"

nexus_json GET "$CONTROL_BASE_URL/v1/executions/$EXECUTION_ID/logs?lines=200" \
  > "$LOG_OUTPUT_DIR/logs.json"

# 先确认日志来源和可用性，再读取内容。
jq '{execution_id, execution_status, source, availability, line_count,
     partial, fetched_at, retry_after_ms, error_code}' "$LOG_OUTPUT_DIR/logs.json"
jq -r '.lines[]? | [(.timestamp // ""), .level, (.logger // ""), .message] | @tsv' \
  "$LOG_OUTPUT_DIR/logs.json"

# 异常堆栈单独保留，避免只看 message 漏掉原因。
jq '.lines[]? | select(.exception != null) | {timestamp, fields, exception}' \
  "$LOG_OUTPUT_DIR/logs.json"
```

`GET /v1/executions/{execution_id}/logs` 与页面 Logs 使用相同接口，需要目标任务的 `execution:read` 权限。
`lines` 默认 200，最少 1，最多 1000；部署可以设置更低上限。返回的是最近日志的快照，不支持通过 `cursor` 翻查完整历史。

### 理解日志响应与可用性

| 响应字段 | 如何使用 |
| --- | --- |
| `execution_status` | 本次查询时的任务生命周期状态；日志是否可读取看 `availability` |
| `source` | `instance_manager` 为运行中的 Manager 文件，`zhiyan` 为终态日志查询，`none` 表示尚无来源 |
| `availability` | 按下表判断是否已有日志、需要等待还是来源不可用 |
| `lines`、`line_count` | 日志数组与本次返回行数；每行包含 `timestamp`、`level`、`message`，还可能有 `logger`、`fields`、`exception` |
| `partial` | `true` 表示存在截断或查询窗口限制；即使为 `false`，也不代表覆盖任务从开始到结束的全部日志 |
| `fetched_at` | 本次拉取时间；判断是否有新进展应比较日志行的时间与内容 |
| `retry_after_ms`、`error_code` | 建议重试间隔（毫秒）与不可用原因；允许为空 |

| `availability` | 含义与处理 |
| --- | --- |
| `ready` | 本次读取成功；`lines` 仍可能为空，不能据此判断任务成功或失败 |
| `not_started` | 尚未分配 Manager；查看任务 `phase` 和 Control Events |
| `ingest_pending` | 任务已终态，智研日志仍在入库；按 `retry_after_ms` 等待后有限重试 |
| `unavailable` | 查看 `error_code`：例如 `manager_log_unavailable` 表示 Manager 日志不可用，`zhiyan_log_not_configured` 表示终态日志查询条件未就绪 |

API 的 HTTP `200` 不保证日志可用。HTTP `429` 表示查询受限，应降低频率；HTTP `502` 表示日志上游查询失败，记录错误并退避重试。
日志未启用、已轮转、超出查询窗口或终态采集未配置时，可能取不到内容；日志为空或不可用本身不能证明任务卡住。
运行中读取的是 Manager 文件，其中可能包含同一 Manager 上其他任务的记录；关联异常时结合日志 `fields` 中的
`execution_id`、`assignment_id`、`attempt_id` 或 `run_id` 核对归属。

### 查看构建日志

<a id="补充查询控制事件与构建日志"></a>

任务还在构建依赖时，打开 **Builds → 对应构建 → Build Logs**；也可以从 Run 的 **Overview → Allocations** 中点击
Dependency 对应的构建。查询任务关联的构建日志：

```bash
nexus_json GET "$CONTROL_BASE_URL/v1/executions/$EXECUTION_ID/build-logs?lines=200" | jq .
```

构建日志返回 `items` 数组，每项对应一个关联构建，带有 build ID、`attempt`、`status`、`availability` 和 `lines`。
已知构建 ID 时，可直接查询 `GET /v1/builds/{build_id}/logs?lines=200`。更多说明见
[Building 阶段日志](troubleshooting.md#building-阶段日志)。

## 读取任务结果

### 从控制台读取结果

<a id="读取-url"></a>

1. 打开目标 Run；Batch 则先在 child 列表中选中具体子任务。
2. 切换到 **Spec / Result** 页签。
3. 在右侧 **Result** 区域点击 **Load result**，读取完整 `RunOutput`。
4. 核对 `success`，再读取任务需要的 `task_output`、`agent_output` 或 `processor_results`。

![Run 详情中选择 Spec / Result 并点击 Load result](../assets/images/nexus-cloud/run-result.png)

图为 dev 的实际测试任务，结果位于 `task_output`。快速开始示例应检查 `agent_output.response` 是否为
`NEXUS_ONLINE_SERVICE_OK`；不同任务的业务输出字段不同。

### 通过 API 读取

`result_url` 应保存，但使用时需要匹配当前身份可访问的入口。地址规则见
[程序入口与地址](access-and-environments.md#程序入口与地址)。通过选定的 Application API 读取：

```bash
nexus_json GET "$APPLICATION_BASE_URL/v1/executions/$EXECUTION_ID/result" | jq .
```

Control 详情中的 `result_json` 通常只是 external pointer，不能直接当作完整业务结果。

### 读取 Batch 子任务结果

Batch 的 parent 结果不是所有 child 输出的聚合 JSON：先查 `batch_url` 或 `/v1/executions/{execution_id}/batch` 的汇总，
再分页读取 child 列表，用每个 child 的 `execution_id` 读取结果。大量结果可用下方的[标准 JSONL 导出](#导出标准-jsonl)。

## 导出结果 JSONL

<a id="导出标准-jsonl"></a>

`nexus export` 可以导出本地 SQLite，也可以通过 `--execution-id` 导出已经终态的 Cloud Single、Live 或 Batch。
Single/Live/Batch 的 API-only 结果导出使用 Nexus 基础依赖；带 DB 参数的 results 和原始事件读取
需要 distributed extra 中的异步 MySQL 驱动：

```bash
uv sync --extra distributed
```

原有本地 SQLite 导出用法不变：

```bash
uv run nexus export ./batch.db -o ./results.jsonl
```

下面的 Cloud 命令使用个人 PAT。先按[认证与运行环境](access-and-environments.md)设置对应的个人 API 基址和
`TAIHU_TOKEN_FILE`，再设置本次输出目录和目标 Execution：

```bash
set -euo pipefail

umask 077
OUTPUT_DIR="$(mktemp -d "${TMPDIR:-/tmp}/nexus-export.XXXXXX")"
EXECUTION_ID='<execution-id>'
```

不提供 DB 参数时，Single/Live/Batch 的终态结果从 Control/Application API 读取：

```bash
uv run nexus export \
  --control-url "$CONTROL_BASE_URL" \
  --application-url "$APPLICATION_BASE_URL" \
  --execution-id "$EXECUTION_ID" \
  --taihu-token-file "$TAIHU_TOKEN_FILE" \
  --output "$OUTPUT_DIR/execution.jsonl"
```

可重复传入 `--execution-id`，合并多个 Batch、Single 或 Live 的结果：

```bash
uv run nexus export \
  --control-url "$CONTROL_BASE_URL" \
  --application-url "$APPLICATION_BASE_URL" \
  --execution-id exe_example_batch_a \
  --execution-id exe_example_batch_b \
  --execution-id exe_example_single \
  --taihu-token-file "$TAIHU_TOKEN_FILE" \
  --output "$OUTPUT_DIR/combined.jsonl"
```

先展开所有 Batch，再补入显式 Single/Live，去重后使用同一读取流程并发导出。
各 Batch 按它们在参数中出现的顺序排列，批次内保持 item 游标顺序；尚未入选的显式 Single/Live
随后按其参数顺序排列。每个 Execution 只安排一次结果读取流程，不按原始 run ID 或 item ID 去重。
同一个 Batch 无论单独导出还是与其他目标合并，均保留原始 run ID、item metadata、重试次数和状态。
没有 child 的失败/取消占位记录在 `failed` 和 `all` 模式下移除，在完整导出中保留。
重复指定父任务及其当前子执行时，保留 Batch 投影，
即使子 ID 写在父 ID 前面，结果也不改变。显式选择历史执行仍读取该 ID，不切换到当前重试。

至少一个 Batch 有非空 metadata 时，写出 `<output>.batch_metadata.json`。
只有一个 Batch 时沿用原 metadata 对象，即使同时选择了 Single/Live；多个 Batch 时按
`{"batches": {"父ExecutionID": {...}}}` 聚合所有所选批次的 metadata，空 metadata 保留为 `{}`。
全部 Batch metadata 为空时不生成旁挂文件；结果行全部被过滤时，仍按上述规则保留旁挂文件。
未被 Batch 覆盖的 Single/Live 保留单任务格式（`run_id=execution_id`、`metadata=null`）。

单 ID 和多 ID 都支持同一套 DB 参数：有 DB 时优先读取，仅缺少 RunOutput 才回退 result API。
DB 读取失败直接报错，不作 backend 预校验，也不自动换库。Boundless 查询按各目标的父任务创建时间
填写独立分区值，独立 Single/Live 使用自身创建时间；跨 Batch 时应省略固定的
`--parent-created-at`，让程序分别获取。显式提供该值时，它会覆盖所有目标的自动分区选择。
分区准备发生在筛选、去重后的逐条读取阶段；不保证整次导出只探测一次分区列或缓存父执行查询。
多 ID 仅支持 results，不能与 traces/follow 组合。

### 部分 Batch 结果与状态选择

已知父 Batch 和部分 item ID 时，可以通过 `--item-ids` 导出对应结果，无需先查询子 execution ID：

```bash
uv run nexus export \
  --control-url "$CONTROL_BASE_URL" \
  --application-url "$APPLICATION_BASE_URL" \
  --execution-id "$EXECUTION_ID" \
  --item-ids 'item-001,item-003' \
  --taihu-token-file "$TAIHU_TOKEN_FILE" \
  --content results --allow-partial --result-status all \
  --output "$OUTPUT_DIR/selected-items.jsonl"
```

`--item-ids` 仅支持一个 Batch 的 results；值为逗号分隔的 item ID，也可以重复传参。
匹配 Batch API 的 `item_id`，不是原始 `run_id`；去除参数首尾空格、合并重复 ID 后，
仍按 Batch item 顺序输出。每个入选 item 使用当前 `current_child_execution_id`，保留原始 run ID、
重试次数和 Batch metadata，读取仍为 DB 优先、缺少 RunOutput 时回退 result API。

`--item-ids` 只用于选择条目，不增加输出字段。导出的 `run_id` 与
`metadata._nexus.original_run_id` 使用原始运行 ID；通过父 Batch 和 item_id 查询该 item 的
`metadata._nexus.original_run_id` 即可对应，未保存原始运行 ID 时使用 item_id。
输出继续保留 `parent_execution_id` 和 `child_execution_id`，用于追溯批次及执行。

该参数与下表的状态规则取交集：不开 partial 时仍要求整个 Batch 满足完整导出条件；
默认 partial 只选成功 item，failed/all 继续执行产物过滤。存在但尚未终态或不符合状态选择的 item
按原规则跳过；不存在的 ID 会在读取结果或创建输出文件前报错，不会静默漏导。
筛选仍分页检查 Batch 元数据，不读取未入选 item 的结果正文；不支持与多个不同 execution ID、
纯 Single/Live、本地 SQLite 或 traces 组合。

以下为基于 `090aadbc01` 统一导出流程的 `result-status` 最终用法契约。
`--allow-partial` 放宽每个 Batch 的完成条件，并跳过分页读取时未终态的 item；
`--result-status` 进一步选择 Batch 的终态条目；`failed` 和 `all` 模式在读取和校验后过滤无产物的非成功行。
完整导出仍包含所有终态条目。

| 参数 | Batch 条目范围 |
| --- | --- |
| 不带 `--allow-partial`、不带 `--result-status` | 全部终态；要求父执行终态、生成完成、提交关闭、全部 item 终态且摘要一致 |
| `--allow-partial` | 仅成功条目，等同于再加 `--result-status success` |
| `--allow-partial --result-status success` | `succeeded` |
| `--allow-partial --result-status failed` | 选择所有非成功终态：`failed`、`lost`、`expired`、`killed`、`cancelled`；读取校验后仅写出有产物的非成功行 |
| `--allow-partial --result-status all` | 选择所有终态，写完文件后过滤无产物非成功行 |
| 显式 `--result-status`，但无 `--allow-partial` | 参数错误，包括显式 `all` |

**迁移已有脚本：需要继续包含失败条目时，将裸 `--allow-partial` 改为
`--allow-partial --result-status all`，但 all 也会移除无产物非成功行；需要保留这些空记录时使用完整导出。** `failed` 不只匹配字面值 `failed`，也不根据
RunOutput 内部的 `success` 字段筛选或覆盖执行状态。

`failed` 和 `all` 模式仅在一行的 `status != success`，且 `task_output`、`agent_output`、`processor_results`
三个字段各自均为 `None`（JSON `null`）、空字典 `{}`、空列表 `[]` 或空字符串 `""` 时不写出该行。
`0`、`False` 均算产物并保留；非空对象不递归检查是否含有真正的 trajectory，只有 score 也算产物。
只有状态、错误或 metadata 而三个产物字段全空的行会被过滤，RunOutput 对象存在本身不足以保留该行。
完整导出及默认或显式 `success` 模式不执行此产物过滤。

例如只导出 Batch 中有产物的失败类终态条目：

```bash
uv run nexus export \
  --control-url "$CONTROL_BASE_URL" \
  --application-url "$APPLICATION_BASE_URL" \
  --execution-id "$EXECUTION_ID" \
  --taihu-token-file "$TAIHU_TOKEN_FILE" \
  --content results --allow-partial --result-status failed \
  --output "$OUTPUT_DIR/failed-results.jsonl"
```

两个选择参数只适用于含至少一个 Batch 的 Cloud results；纯 Single/Live、本地 SQLite、traces
均不接受，results 也不支持 follow/resume。混合输入中的显式 Single/Live 仍必须终态，且不受
Batch 状态筛选影响。`failed` 和 `all` 模式下，显式失败执行同样接受无产物过滤，显式成功执行不受此过滤影响。
`success` 模式的混合文件仍可能包含显式指定的失败执行，即使该执行没有产物也保留。

先按 Batch 状态筛选、去重，再读取 DB/API 并将所有候选结果写完临时 JSONL。failed/all 模式随后逐行过滤临时文件中的无产物非成功行，再发布正式文件。
被 Batch 状态筛掉的 child 可以由显式 ID 补入；
已经入选的 Batch child 与显式 ID 重复时始终保留 Batch 记录，包括显式 ID 写在 Batch 前的情况。
写出时过滤不撤销去重，也不会再补入重复显式 ID。无 child 的失败/取消占位在 `failed` 和 `all` 中移除，
在完整导出中保留，单 Batch 和多 Batch 行为一致。
partial 不要求生成完成或提交关闭；读取过的条目随后完成不会补读，无入选条目且没有显式目标、
或所有候选行均被产物过滤时生成空 JSONL。成功日志只报告最终 `written_records`，不在读取或写入阶段维护逐条产物统计。

`success/failed` 会校验并保存入选 child 的 Control 终态快照，确认它与 Batch item 的选择类别一致，
并在发布前再次核对；即使全部结果行被过滤，仍复核已保存的快照与 Batch metadata。
身份、状态、attempt、结果引用或错误变化会使导出失败。
`all` 沿用基线的选择范围与校验，不增加这项针对所选类别的 child 快照要求。
所有模式仍校验分页、item 顺序及 Batch metadata；partial 不要求全批次处于同一时刻的一致快照。

### 结果来源与输出文件

如果需要优先从 Trace Database 读取结果，可额外提供
**My Environments → Trace Database Access** 中的只读账号及表名，获取步骤见
[申请数据库只读账号](#申请数据库只读账号)。密码保存到本地私有文件，不放进命令参数或 Git：

```bash
mkdir -p ~/.nexus/auth
chmod 700 ~/.nexus/auth
read -rsp 'Trace Database password: ' TRACE_DB_PASSWORD && echo
printf '%s' "$TRACE_DB_PASSWORD" > ~/.nexus/auth/trace-db-password
unset TRACE_DB_PASSWORD
chmod 600 ~/.nexus/auth/trace-db-password

uv run nexus export \
  --control-url "$CONTROL_BASE_URL" \
  --application-url "$APPLICATION_BASE_URL" \
  --execution-id "$EXECUTION_ID" \
  --taihu-token-file "$TAIHU_TOKEN_FILE" \
  --db-host '<host>' \
  --db-port 3306 \
  --db-name '<database>' \
  --db-user '<read-only-username>' \
  --db-password-file ~/.nexus/auth/trace-db-password \
  --trace-table 'trajectory_span_events_pr_<32-hex>' \
  --output "$OUTPUT_DIR/batch.jsonl"
```

每行结果字段保持为：

```text
run_id, status, start_time, end_time, retry_count, error,
task_output, agent_output, processor_results, metadata
```

RunOutput 的 `extra` 非空时沿用 main 的行为追加顶层 `extra`；该字段不参与产物筛选。

`status` 沿用原映射，保留 `success`、`failed`、`timeout` 和 `cancelled` 的区别；
`--result-status failed` 不会把所有非成功状态改写为 `failed`，RunOutput.success 也不覆盖状态。
Batch 按 API 的 item 游标读取，
若 metadata 保存了原始 `run_id` 则恢复该值，因而不保证文件按 `run_id` 字典序排列。
有 Batch metadata 时另写 `<output>.batch_metadata.json`。

Single/Live/Batch 提供完整数据库参数时，优先逐条读取 TDSQL 根 run span 更新记录中的输出，
不会全文搜索所有轨迹正文；仅当数据库中没有输出时，才从 Application 读取该子任务的已保存结果。
不提供数据库参数时直接从 Application 读取。单 ID 和多 ID 使用相同的读取规则。
起止时间优先来自 Control 生命周期事件，缺失时使用已查到的轨迹时间。
API 或数据库读取失败、结果结构无效均会中止导出并报告对应条目/子任务，不会当成空结果跳过。
成功条目缺少输出或输出的 `success=false` 时导出失败；失败类终态没有输出时保留状态与错误信息，
三个产物字段为 `null`，该行在 `failed` 和 `all` 模式下被过滤。失败条目有输出时保留已有产物，
即使输出的 `success=true` 也不改写执行终态；优先使用 RunOutput 的 `error`，缺失时从 Control 补齐。
没有 child 的失败/取消占位行在完整导出中保留 `BatchItemNoChildExecution` 错误。
输出存在但轨迹为空时，答案、评分等非空产物仍保留；`failed` 按上述三个产物字段判断。
结果按条目顺序、有界并发读取并逐条写盘，避免整页大结果在内存中聚合。

结果导出默认拒绝覆盖已有文件，显式 `--force` 可以替换；导出期间先写 `.partial`，校验通过后原子发布。
完整 Batch 导出要求 generation/submission 完成且 items 全部终态，导出前后快照不能变化；
partial 使用上文的范围与复核规则。读取或复核失败时不发布本次正式 JSONL。

## 运行中的原始轨迹事件导出

使用 `--content traces --follow`，可在 Single、Live 或 Batch 运行中持续导出已写入 Trace DB 的原始事件。
启动时先补读历史，然后持续发现新 child 和新 run span；失败重试产生的历史轨迹也会保留。
客户端直连 TDSQL，读取原始事件并写入本地 JSONL 文件。
本期支持 principal TDSQL 表 `trajectory_span_events_pr_<32-hex>`，每次指定一个 Execution 和一张表。
Control URL 与 DB 连接参数必须属于要导出的同一环境。

| 环境 | Control URL | Application URL（仅 results 需要） |
| --- | --- | --- |
| 正式 | `https://control-runner-nexus-agent.mcp.woa.com` | `https://application-runner-nexus-agent.mcp.woa.com` |
| runner-test | `https://test-control-runner-nexus-agent.mcp.woa.com` | `https://test-application-runner-nexus-agent.mcp.woa.com` |

下面的 Bash 函数复用连接参数，替换占位值后即可依次执行跟随、恢复或快照：

```bash
EXPORT_EXECUTION_ID="$EXECUTION_ID"
EXPORT_CONTROL_URL="$CONTROL_BASE_URL"
EXPORT_DB_HOST='<host>'
EXPORT_DB_NAME='<database>'
EXPORT_DB_USER='<read-only-username>'
EXPORT_TRACE_TABLE='trajectory_span_events_pr_<32-hex>'

export_cloud_traces() {
  uv run nexus export \
    --execution-id "$EXPORT_EXECUTION_ID" \
    --control-url "$EXPORT_CONTROL_URL" \
    --taihu-token-file "$TAIHU_TOKEN_FILE" \
    --content traces \
    --db-host "$EXPORT_DB_HOST" --db-port 3306 \
    --db-name "$EXPORT_DB_NAME" --db-user "$EXPORT_DB_USER" \
    --db-password-file ~/.nexus/auth/trace-db-password \
    --trace-table "$EXPORT_TRACE_TABLE" "$@"
}

# 持续导出，默认轮询间隔为 1 秒；Ctrl-C 停止本地导出，不停止云端任务。
export_cloud_traces --follow --poll-interval 1 -o "$OUTPUT_DIR/events.jsonl"

# 中断或进程崩溃后，使用相同来源和相同输出路径恢复。
export_cloud_traces --follow --resume -o "$OUTPUT_DIR/events.jsonl"

# 一次性导出当前可见事件，使用新文件；此模式不接受 --poll-interval。
export_cloud_traces -o "$OUTPUT_DIR/events-snapshot.jsonl"
```

traces 只需要 Control 和 Trace DB，不需要 Application URL。两个凭据文件必须是权限 `0600` 的普通文件。
结果导出和事件导出均使用 JSONL，但一行的含义不同：results 是一条最终任务结果，traces 是一条原始事件。

### 事件格式与排序

每行保留 principal 表的 12 列：

```text
id, run_id, run_span_id, parent_execution_id, record_id, event_type,
time_unix_nano, trace_id, span_id, parent_span_id, event_data, created_at
```

`event_type` 是 `start`、`update` 或 `end`。`event_data` 保持数据库中的原始 JSON 字符串，
纳秒时间和整数 ID 保持整数精度；`created_at` 编码为日期时间字符串，不给数据库的无时区值补上 UTC 标记。
事件里的 `run_id` 保留云端 Execution ID，便于和 API 关联。

同一 `(run_id, run_span_id)` 内按 `record_id` 递增；不同 child 的行交错追加，不作全局时间排序。
`record_id` 允许有间隙：TDSQL 会过滤纯 token、thinking、tool-call delta 等瞬时更新。
导出器不能补回已被过滤、上传失败或清理掉的数据，也不能用行数等于最大游标来验证完整性。

### 大事件

导出完整保留 event_data，不使用轨迹 API 的 8 MiB 正文省略规则。普通批次按行数和字节共同限流；
单条超过 8 MiB 的事件等待其他正文读取/写入批次排空后独占处理，完整写入后才提交 checkpoint。
文件采用分块编码/写入，避免拼接整批正文；数据库驱动仍可能持有完整单条记录。

DB/driver 包限制、内存不足、磁盘满或编码失败会停止导出并保留已提交位置，不截断或跳过正文。
--resume 继续使用原有 version=1 checkpoint；大行中途失败的未提交尾部会被回退并重新读取。

### 文件、恢复与结束

跟随模式生成：

```text
events.jsonl
events.jsonl.checkpoint.json
events.jsonl.manifest.json
events.jsonl.lock
```

每批数据持久化后才提交 checkpoint，manifest 记录已提交行数、字节数和观测状态。
同一输出只允许一个导出进程持有锁。锁文件可以留在原处，进程退出后 OS 锁会释放；不要删除锁文件来绕过锁。
输出和状态文件使用 `0600` 权限。

`--resume` 会校验来源、输出路径及已提交前缀的 hash、行数和游标，再截断未提交尾部并继续。
不要单独修改、移动或拼接数据与 checkpoint。凭据可以轮换，但数据库、表和 Execution 必须不变。
大文件恢复时需要流式扫描已提交前缀，校验期间不会重复追加旧事件。

普通 `tail -f events.jsonl` 可作实时预览；可靠消费程序应只读取 checkpoint 的 `committed_bytes` 之前的数据，
因为恢复可能截断尚未提交的尾部。traces 不支持 `--force`、stdout 或运行中压缩；需要新导出时选择新文件。

`--follow` 在云端任务终态后仍继续运行，方便补读晚到数据，直到 Ctrl-C 或 SIGTERM。
空读和终态都不会被解释为完整结束。一次性 traces 固定各流当时可见的上界，输出标记为 `snapshot`，
它不是整个 Batch 同一时刻的全局快照，也不提供断点恢复。

### 错误与排查

| 现象 | 处理 |
| --- | --- |
| 退出 2 | 检查参数组合、正数轮询间隔、表名及恢复来源；`--resume` 必须与 traces/follow 一起使用 |
| 退出 1 | 查看 stderr；网络问题有界重试，权限、数据、文件错误需要处理后再恢复 |
| 退出 130 / 143 | 分别是 Ctrl-C / SIGTERM，已提交数据保留，可使用 `--resume` |
| 目标终态但一直 0 行 | 核对 Control/DB 环境、表名和该 Execution 是否启用 TDSQL；不能把 0 行当作完整成功 |
| 提示输出已锁定 | 检查是否仍有同一输出的导出进程，等其退出，不通过删除锁文件抢占 |
| 恢复校验失败 | 保留原文件，检查是否缺少 checkpoint、被截短或修改；不要盲目重置游标 |

当前恢复协议面向 Linux/macOS 本地文件系统；网络盘的锁和 fsync 行为需另行验证。
设计依据见[Cloud 实时导出设计](../developing/cloud-export-events-design.md)。


## Run Trace Backend 与明细轨迹

<a id="查询明细轨迹"></a>

需要批量导出或查询明细轨迹时，再配置这部分能力。**My Environments → Run Trace Backends** 显示可用 backend 和默认顺序；
提交时可用 `--run-trace-sink <backend-name>` 指定已授权的 backend。省略时使用账号默认配置。

### 申请数据库只读账号

前提是管理员已授予 Environment 与 Run Trace Backend，并为该 backend 启用 Trace Database Access。
未出现 **Trace Database Access** 区域时，先联系管理员确认这些条件；仅有 Environment 配额不代表已获数据库访问能力。

1. 打开控制台的 **My Environments**。
2. 找到 **Trace Database Access**，选择需要查询的 backend。
3. 首次创建时，状态为 **not created**，点击右侧 **Create credentials**。

![My Environments 中的 Trace Database Access 和 Create credentials](../assets/images/nexus-cloud/trace-db-access.png)

数据库账号的两张图使用 dev 控制台界面与示例数据演示操作。dev 当时未启用该能力；图中的连接地址、账号和表名不可用于真实连接，
请以自己所在部署的页面信息为准。

4. 创建成功后出现 **Trace database credentials** 窗口，标注 **Shown once**。
5. 在关闭窗口前，使用各行右侧复制按钮保存 Host、Port、Database、Username、Password 和 Trace table。
   Password 默认隐藏，可通过眼睛图标显示或直接复制；请保存到私有凭据文件或凭据管理工具。

![一次性数据库凭据窗口及复制按钮，Password 默认隐藏](../assets/images/nexus-cloud/trace-db-credentials.png)

### 已有账号或忘记密码

创建后列表显示 Username 和 **ready** 状态，操作变为 **Reset password**。已有账号不需要每次查询都重新创建。
忘记密码时再点击 **Reset password**；新密码同样只在弹窗中展示一次，重置后应更新查询工具中保存的密码。

账号只有读取权限，不能修改或删除轨迹数据。页面 Scope 为 **All trace tables** 时，表示可读取这个独立 Trace Database
中的全部轨迹表，并非按用户表隔离；请以实际授权范围为准。

### 使用连接信息

把弹窗中的参数填入数据库查询工具，或[导出命令](#导出标准-jsonl)的 `--db-host`、`--db-port`、`--db-name`、`--db-user`、
`--db-password-file`、`--trace-table`。密码只写入权限 `0600` 的私有文件，不放入命令历史、文档、截图或 Git。
网络无法连接时先检查访问路径与数据库网络权限；不要反复重置密码来排查网络问题。

## 管理任务

### 停止任务

在任务详情点击 **Stop Execution**，或在列表中点击 **Stop**。程序使用同一个 Control Stop API：

```bash
nexus_json POST "$CONTROL_BASE_URL/v1/executions/$EXECUTION_ID/stop" \
  | jq '{execution_id, status, stop_reason}'
```

Stop 是异步操作；返回 `stopping` 时继续查询，直到进入 `killed` 或其他终态。停止 Batch parent 后，系统会停止尚未完成的
child，再结束 parent。终态任务不能重新启动，需要使用新的幂等键重新提交。

关闭终端、断开 SSE 或中断 `--wait` 都不会自动停止云端任务。Live 的输入和停止区别见
[Live 指南](live.md#读取结果与停止任务)。

### 共享 Runs 和 Batches

在 **Shared Access** 中选择用户并保存授权。被授权者可以查看该 owner 名下的 Runs/Batches、结果、事件和 child，并停止仍在运行的任务。
owner 可在同页撤销授权。普通用户默认看到自己的任务及其他 owner 直接授权的任务，管理员可以查看更多资源。

该授权不传递，也不共享 Environment、Quota、Token、Run Trace Backend Grant 或数据库凭据。API 说明见
[Shared Access API](reference/api.md#shared-access-api)。

### Task extra

在 `RunInput` 上填写独立的 JSON 对象，不放入 Batch metadata：

```python
RunInput(task_name='my_task', extra={'source_text': '...', 'reference_answer': '...'})
```

`extra` 按紧凑 UTF-8 JSON 计算，最大 **2 MiB**。`${...}` 和 `__env` 后缀按原文保存，
不参与配置替换。它随现有任务配置保存，运行后原样进入 `RunOutput.extra`，本地和 Cloud
结果 JSONL 导出均带顶层 `extra`。空 extra 不参与序列化，旧任务哈希不变；metadata、
数据库结构、提交和查询接口均不变。

extra 会增加输入、结果及 Trace 的存储和传输量，仍受既有总大小限制约束。特别是直接
提交 Single 的 JSON 请求体默认只有 1 MiB，接近 2 MiB 的 extra 可能被入口拒绝；Batch
通过 task-list 文件提交时受文件及单行限制约束。本功能不调整这些平台限额。

Cloud 导出通过已有 RunOutput 获取 extra，包含产出了失败结果的任务；未生成 RunOutput
的 lost/未启动任务不补读输入。使用兼容的提交、执行和导出版本，旧版 RunInput 会忽略新字段。
