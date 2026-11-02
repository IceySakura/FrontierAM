# Scheduler 策略与潮汐容量准入

Scheduler 根据后台 PolicyExecutor 发布的快照决定是否分配 Execution，分配事务不会调用外部接口或插件。
潮汐策略控制新分配；已经运行的任务继续执行，模型请求的 QPM 可由下文的共享请求预算控制。

## 部署配置

先完成数据库 migration，再把同一份 `policy_executor` 配置发布到 Control 和 Scheduler 使用的全局配置中，重启受影响的消费者。
配置沿用既有 `distributed config set` 的完整值写入和 expected-version 校验；不要用下面的片段覆盖其他全局配置。
未配置 binding 的部署保持原调度行为。

```yaml
policy_executor:
  workers: 2
  worker_startup_seconds: 30
  bindings:
    tidal:
      kind: scheduler
      policy: infer_proxy_tide
      source: infer_proxy_capacity
      revision: 1
      mode: shadow
      on_unavailable: defer
      environments: [my-environment]
      principals: [my-principal]
      refresh_seconds: 60
      timeout_seconds: 10
      max_age_seconds: 180
      max_defer_seconds: 30
      scopes:
        team-hy3:
          source_config:
            model: hy3-offline
            api_key: "<your-api-key>"
      params: {}
```

推理 API Key 直接填写在 scope 的 `source_config.api_key` 中，保存在受限的部署配置中；不要把真实 Key 提交到仓库。容量观测、决策快照和日志不包含 Key。
`environments` 是绑定允许的环境，`principals` 为空表示这些环境中已获任务提交授权的用户均可引用；非空时进一步限定 owner。
同一个额度组应复用相同 scope，不要给每个任务建立重复 scope。

先用 shadow 记录策略建议，再使用更高的 revision 切换到 enforce。显式 disabled 会恢复默认调度。
任何配置、插件实现或 imports 变更都必须增加受影响 binding 的 revision；旧进程不能覆盖新版本状态。

只有当前信号有效、`key_remaining_raw_qpm > 0` 且 `hard_brake == false` 才允许新分配。
Scheduler 不按 factor 设置额外的暂停或恢复阈值；缺失、非法或过期信号默认暂缓。`on_unavailable: default` 仅适合优化型策略，已有明确拒绝不会被它清除。

## 提交与查询

```bash
uv run nexus submit single run.yaml \
  --environment my-environment --source ./project \
  --scheduler-policy tidal --policy-scope team-hy3
```

`--policy-scope` 可重复指定。Batch 使用相同参数，只把选择传给业务子 Execution，编排父任务不受模型门禁影响。
HTTP 提交的对应字段为：

```json
{"scheduler_policy": {"binding": "tidal", "scopes": ["team-hy3"]}}
```

需要环境内强制应用时，在配置中设置：

```yaml
policy_executor:
  defaults:
    my-environment:
      binding: tidal
      scopes: [team-hy3]
```

defaults 与 bindings 应合并保存。提交者不能移除默认 scope 或选择另一 binding 来绕过它。
环境授权仍由原有提交与访问控制负责，策略绑定不会授予额外资源权限。

读取 `GET /v1/executions/{execution_id}/scheduler-policy`（支持 Single、Batch child 或 Batch parent）查看当前放行建议、原因、scope 快照版本、有效期及下一次评估时间。
该接口沿用 Execution 的访问权限。`scheduler.policy_deferred` / `scheduler.policy_applied` 控制事件记录实际使用的决策版本。
暂缓期间 Execution 持久化状态保持 `queued`，仍受 queue TTL 限制，没有新 Assignment 也没有新 attempt；依赖构建与运行资源就绪后，对外查询的只读阶段投影 `phase` 返回 `waiting_policy`。`phase` 依据上一轮调度的持久化结论推导，最多滞后一个 `max_defer_seconds`（默认 30 秒），实时评估状态与原因以 `/scheduler-policy` 为准。完整查询方式与 `reason_code` 对照表见下文[状态与原因查询](#状态与原因查询)。

## 扩展插件

部署端通过 `policy_executor.imports` 加载注册模块。模块使用
`nexus.distributed.policies.registry.policy_registry` / `signal_source_registry` 注册类。

- Source 实现 `async read(config, *, max_age_seconds) -> SignalObservation`。
- Policy 声明 `kind`、Pydantic `decision_model`，实现 `evaluate(inputs, previous_state) -> PolicyResult`。
- 可声明 `config_model` / `params_model`，首次评估前在隔离 worker 中验证配置。
- Policy 返回 decision、next_state、valid_until、reason_code；框架原子发布，不允许扩大来源有效期。

Scheduler 的 decision schema 是 `SchedulerDecision`，包含 allow 和可选 preferred_managers。
其他 kind 可使用自己的 schema 和相同执行器，业务效果由对应消费端接入后提供，不能通过改 kind 让 Scheduler 执行任意动作。
框架只加载受信部署代码，进程隔离用于超时和故障控制。

## 共享请求预算（`request_rate`）

`request_rate` 类型的 binding 为一个额度组配置总 QPM（`RequestRateDecision(allow, rate_per_minute)`），绑定同一策略的多个 InstanceManager 通过控制库短租约共享这份总额度，本机所有接入 `RequestAdmission` 的 Client 共用唯一的 `RequestBudgetControl` 平滑限流。设计见[共享请求预算与 Client Policy 接入](../../developing/shared-request-budget-policy-design.md)。

```yaml
policy_executor:
  imports: [nexus.distributed.policies.collections.infer_proxy_request_rate]
  bindings:
    team-a-hy3-rate:
      kind: request_rate
      policy: infer_proxy_request_rate
      source: infer_proxy_request_capacity
      revision: 1
      mode: enforce
      on_unavailable: defer
      environments: [my-environment]
      principals: [team-a-member]
      scopes:
        capacity:
          source_config:
            model: hy3-offline
            api_key: "<your-api-key>"
            quota_qpm: 600
      params:
        min_quota_ratio: 0.01
  request_defaults:
    my-environment:
      binding: team-a-hy3-rate
      scope: capacity

request_policy:
  binding: team-a-hy3-rate
  scope: capacity
request_budget:
  lease_seconds: 10
  renew_before_seconds: 4
```

- 一个 `request_rate` binding 可配置一个或多个 scope（各 scope 以 `(binding, scope)` 独立核算预算账本）：`mode: enforce` 时只支持 `on_unavailable: defer`，不提供「异常时各 client 恢复满速」的退路。
- `quota_qpm` 是该额度组申请到的总 QPM，由部署方在 `scopes.<scope>.source_config.quota_qpm`（或 `binding.params.quota_qpm`）中填写。每次有效观测将上次有效比例乘以 factor，并封顶到 1；候选比例严格低于 `min_quota_ratio` 时暂停并保留上次有效比例。发布的 `rate_per_minute = quota_qpm × 比例` 是当前可分配的总速率，暂停时为 0。
- **绑定与解析**：`request_defaults` 按环境指定默认引用，提交时解析并冻结到 Execution，Batch 子任务继承，编排父任务不带。环境配置了默认引用后，提交方不能绕过或切换为其他 binding；未设置环境强制默认时可通过 `--request-policy <binding> --request-scope <scope>`（HTTP `request_policy: {binding, scope}`）指定。
- **任务与 Manager 硬条件匹配**：每个启用请求预算的 InstanceManager 启动时固定绑定唯一 `binding/scope` 并登记能力，Scheduler 在筛选 Manager 时要求两者完全一致：带预算的任务不会落到无预算的 Manager，无预算的任务也不会落到受控 Manager。运行中不支持切换绑定，需要更换时先 drain 再替换。
- **租约化分配**：每个 Manager 持有唯一的 `RequestBudgetControl` 和至多一份有效租约，按等权、受各自上限约束的方式分摊总额。发放新份额时扣除所有尚未到期的租约。降低总额时，旧份额在到期前可能暂时高于新目标，最多经过一个租约周期收敛；低于下限或额度耗尽导致的暂停也受这一传播时间约束。降低总额或有新 Manager 加入时，超出新目标的旧租约不再续期，等自然到期后重新分配；低于新目标且无剩余可分配预算时，允许按原份额续租以避免空窗。
- **配额耗尽暂停与快照不可用**：scope 配置 `source_config.api_key` 时带该 Key 查询剩余额度。Request 层忽略 `hard_brake`，按 factor 调整上次有效速率并封顶到配置上限；候选速率严格低于下限时发布 `allow: false, rate_per_minute: 0`，保留暂停前比例（恰好等于下限仍允许）。低于下限暂停后，factor ≤ 1 持续暂停且不再衰减保存的比例；只有当前信号有效、额度为正且 factor > 1 才从保存的比例恢复。额度为 0 也发布零速率，并保留比例和下限暂停状态；已有份额在一个租约周期内失效。快照不可用或数据库断连时，enforce 下的 Manager 在旧租约到期后等待，不会自动旁路。
- **客户端接入与查询**：云端受控任务中的 `infer_proxy` 客户端自动通过 `RequestAdmission` 共享总额度，不再配置本地静态分摊 `tide_qpm`（同时配置会报错）。读取 `GET /v1/executions/{execution_id}/request-policy` 可查看绑定的 `binding/scope`、模式、配置与当前有效总 QPM、快照状态以及各 Manager 获配份额、租约到期时间与本地放行/利用率指标（`reason_code` 含义见下文[状态与原因查询](#状态与原因查询)）。
- **回退**：binding 改为 `disabled`（并递增 `revision`）。各 Manager 观察到后停止使用和续租，数据库中的份额按原期限自然回收，无需重新提交任务。

## 状态与原因查询

排队期调度准入策略（Scheduler Policy）与运行期共享请求预算（Request Policy）的可观测入口如下（当前内置策略实现为潮汐容量准入 `infer_proxy_tide` 与平滑限流 `infer_proxy_request_rate`）：

| 阶段 | 查询接口 | 策略字段 / 状态码 | 作用 |
| --- | --- | --- | --- |
| **排队期（Single / Batch 子任务）** | `GET /v1/executions/{execution_id}` | `phase = "waiting_policy"` | 标识依赖与运行资源已就绪，当前正在**调度策略门禁排队**（`status` 仍为 `"queued"`） |
| **排队期（Batch 父任务汇总）** | `GET /v1/executions/{parent_id}/batch?include_status_counts=true` | `child_waiting_policy_count` | 直接返回当前因调度策略暂缓的子任务总数，无需遍历子任务 |
| **排队期（查准入策略原因）** | `GET /v1/executions/{execution_id}/scheduler-policy` | `allow`、`reason_code`、`next_schedule_at` | 查看准入策略实时决策与受阻原因（支持 Single、Batch child 或 Batch parent；父任务看共享 `reason_code`，`next_schedule_at` 查具体 child） |
| **运行期（查请求限流/暂停）** | `GET /v1/executions/{execution_id}/request-policy` | `allow`、`effective_rate_per_minute`、`reason_code` | 查看 `running` 期间的实时有效 QPM 及暂停原因（支持传 Single、Batch child 或 **Batch parent**） |

### 1. 排队期准入策略（`/scheduler-policy`）

当 Single 的 `phase` 为 `waiting_policy`、或 Batch 汇总中 `child_waiting_policy_count > 0` 时，可直接查询 `/scheduler-policy`（传 Batch parent 时返回子任务共享的策略状态与 `reason_code`，其 `next_schedule_at` 固定为 `null`，具体下次准入时间在各自 child 上）：

```bash
# Single、Batch child 或 Batch parent 均可直接查询
nexus_json GET "$CONTROL_BASE_URL/v1/executions/$EXECUTION_ID/scheduler-policy" \
  | jq '{enabled, allow, reason_code, binding, mode, next_schedule_at}'

# Batch 还可结合汇总接口查看当前处于策略排队的子任务数量
nexus_json GET "$CONTROL_BASE_URL/v1/executions/$EXECUTION_ID/batch?include_status_counts=true" \
  | jq '{submitted_items, active_items, terminal_items, child_waiting_policy_count, child_execution_status_counts}'
```

`/scheduler-policy` 与 `scheduler.policy_deferred` 事件的 `reason_code`（`policy_*` 为框架通用码，其余为当前内置潮汐准入策略 `infer_proxy_tide` 返回码）：

| `reason_code` | 含义与下一步 |
| --- | --- |
| `capacity_available` / `policy_available` | 当前允许分配（若 `phase` 仍短暂显示 `waiting_policy`，等待下一轮调度认领即可，最多滞后一个 `max_defer_seconds`） |
| **`quota_exhausted`** | **额度耗尽**（潮汐）：绑定的模型或 API Key 剩余额度为 `0`，等待额度恢复 |
| **`hard_brake`** | **负载急停**（潮汐）：上游容量接口返回 `hard_brake=true` 暂停分配新任务，等待上游恢复 |
| `policy_unavailable` | 策略快照尚未生成、已过期或配置刚变更，按 `on_unavailable=defer` 保守暂缓；可查看 `scopes[].last_error` 与 `next_evaluate_at` |
| `policy_invalid` | 任务绑定的 `binding` / `scopes` 或环境默认策略在提交后被移除、修改或不再允许当前用户/环境引用，调度器写入 `policy_deferred:policy_invalid` 暂缓；联系管理员核对 `policy_executor` 配置 |
| `signal_expired` / `signal_incomplete` | 上游信号过期或关键字段缺失，暂停放行；等待后续周期恢复或联系管理员检查信号源 |

### 2. 运行期共享请求预算（`/request-policy`）

任务进入 `status = "running"`（或 Batch 处于 `phase = "running_children"`）后，共享请求预算限速或暂停**不会改变 `phase`**。如需确认运行期实时有效 QPM 或是否暂停发请求，直接查询 `/request-policy`（支持传 Single、Batch child 或 **Batch parent**）：

```bash
nexus_json GET "$CONTROL_BASE_URL/v1/executions/$EXECUTION_ID/request-policy" \
  | jq '{enabled, allow, reason_code, configured_quota_qpm, effective_rate_per_minute}'
```

`/request-policy` 的 `reason_code`（`policy_*` 为框架通用码，其余为当前内置潮汐限速策略 `infer_proxy_request_rate` 返回码）：

| `reason_code` | 含义与下一步 |
| --- | --- |
| `adopted` / `unchanged` / `resumed` | 正常放行，按 `effective_rate_per_minute`（QPM）速率平滑限流 |
| **`below_minimum`** | **触底暂停**（潮汐）：速率比例低于 `min_quota_ratio`（或暂停期间 `factor <= 1`），请求预算已暂停（`effective_rate_per_minute = 0`），等待上游 `factor > 1` 恢复 |
| **`quota_exhausted`** | **额度耗尽暂停**（潮汐）：绑定的模型或 API Key 剩余额度为 `0`，请求预算已暂停（`effective_rate_per_minute = 0`），等待额度恢复 |
| `policy_unavailable` / `signal_expired` / `signal_incomplete` | 策略快照未就绪或上游信号过期/缺失，`enforce` 下暂停发放请求预算 |
| `policy_shadow` / `policy_disabled` | 策略处于 `shadow` 或 `disabled` 且当前无有效快照，客户端不阻塞请求 |
