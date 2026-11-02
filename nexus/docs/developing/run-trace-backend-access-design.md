# Run Trace Backend 授权与选择设计

## 背景

Nexus distributed 已支持在 InstanceManager 中注册多个具名 Run Trace Backend，并通过 execution
`driver_config.run_trace_backends.sinks` 为一次运行选择一个或多个 sink。一个 backend 由同名的
`RunTraceSink` 和 `RunTraceSource` 组成，负责写入 raw trace、保存最终 `RunOutput`，以及后续回放。

现有实现仍缺少面向云端用户的授权边界：所有 Manager 都能看到全局 backend 配置，普通 cloud submission
又没有公开的 backend 选择字段。为了让不同 principal 使用不同的 TDSQL、对象存储或自定义 backend，需要把
“管理员注册 backend”“管理员授权 principal”“用户提交时选择 backend”分成三个独立层次。

本文描述第一版可上线的通用机制。TDSQL 只是 backend 的一种实现，不在提交和授权模型中硬编码。

## 目标

1. 运维通过可信配置注册任意数量、任意实现类型的 Run Trace Backend。
2. 管理员可以控制每个 principal 可见、可选择及默认使用的 backend。
3. 用户可以在 single、live 和 batch submission 中选择一个或多个已授权 backend。
4. backend 的 class、连接地址、账号和 Secret 只存在于服务端配置，永不进入用户提交或公开 API。
5. 选择在提交事务中解析并固化；retry 和 batch child 不因后续授权变化而改写目标。
6. 保持现有多 sink 严格写入语义、source fallback 顺序和 TDSQL writer 复用机制。
7. 为已有客户端提供可控的兼容迁移路径。

## 非目标

MVP 不包含以下能力：

- 通过 Console 创建或编辑 backend 连接配置；
- backend 配置热更新和多 revision 并存；
- environment 维度的 backend grant；
- best-effort sink 或按 sink 单独配置失败策略；
- Web submission 表单；
- 自动跨系统补偿 TDSQL 只读账号撤销；
- backend 用量、成本和 grant 数量统计。

## 现有能力与复用边界

- `DistributedConfig.run_trace_sinks` 已提供具名 backend 的 class、普通配置和 Secret 引用。
- `RunTraceBackendRegistry` 已能按有序名称列表构造多个 sink。
- `RunTraceSession` 已向所有 sink 投递相同的有序 raw records，并把每个 sink 的 source metadata 写入 envelope。
- 任一 sink 的 start、write、close 或 metadata 失败都会使 execution 失败；MVP 保持该语义。
- source 按 envelope 中的 backend 顺序读取，遇到明确的 source unavailable 时尝试下一个。
- TDSQL writer 已按数据库配置维护 writer pool，多套 TDSQL 配置不需要新增 writer 实现。

因此 MVP 只新增控制面 catalog metadata、principal grant、submission 解析和 Console/CLI 入口，不重建 trace
执行层。

## 术语

- **Backend**：一个具名且可信的 sink/source 配置。
- **Grant**：允许某个 principal 查看并选择某个 backend 的控制库记录。
- **Requested sinks**：用户在一次 submission 中显式提交的有序名称列表。
- **Principal defaults**：管理员为 principal 配置的有序默认 backend 列表。
- **Effective sinks**：submission 最终固化并交给 Manager 的 backend 列表。
- **Platform sink**：平台内部管理、用户不能 grant 或删除的 sink；MVP 中主要是 `artifact`。

## Backend Catalog

连接配置继续由 `run_trace_sinks` 管理。`RunTraceSinkConfig` 增加不敏感的展示与授权字段：

```python
kind: str = 'custom'
display_name: str | None = None
description: str | None = None
user_selectable: bool = False
system_managed: bool = False
required: bool = False
```

示例：

```json
{
  "run_trace_sinks": {
    "tdsql-team-a": {
      "kind": "tdsql",
      "displayName": "Team A Trace Database",
      "description": "Team A production traces",
      "userSelectable": true,
      "sinkClass": "nexus.runner.trace.tdsql::TDSQLTraceSink",
      "sourceClass": "nexus.runner.trace.tdsql::TDSQLTraceSource",
      "config": {
        "host": "write-host",
        "port": 3306,
        "user": "nexus_writer",
        "db": "runner_team_a"
      },
      "secretConfig": {
        "password": "tdsql-team-a-writer-password"
      }
    }
  }
}
```

`user_selectable` 默认关闭，避免已有内部 backend 在升级后自动暴露。管理员 catalog 和 `required_sinks` 会返回上述访问属性；
principal 的普通授权条目只返回名称、kind、展示信息和默认顺序。所有公开 API 都不序列化 `config`、`sinkConfig`、
`sourceConfig`、`secretConfig` 或 class URI。

`system_managed=true` 的 backend 不能通过 principal grant 暴露；`required=true` 只允许与
`system_managed=true` 组合，并会由平台追加到用户选择之后。`local` 声明为 system-managed、非 required，`artifact` 声明为
system-managed、required。Catalog、授权过滤和 required sink 返回值都消费这份 metadata，不按 backend 名称判断。

## 持久化模型

新增表 `distributed_run_trace_backend_grants`：

```text
principal_id       varchar(64)  primary key part, FK distributed_principals
backend_name       varchar(64)  primary key part
enabled            boolean      not null
default_order      integer      nullable
version            bigint       not null
created_by         varchar(128)  not null
updated_by         varchar(128)  not null
created_at         datetime      not null
updated_at         datetime      not null
```

约束和语义：

- 主键为 `(principal_id, backend_name)`；
- 同一 principal 的非空 `default_order` 必须唯一；
- 默认顺序从 `0` 开始并连续；
- `enabled=true` 表示可见且可选择；
- `default_order=null` 表示可选择但不在省略配置时自动使用；
- backend catalog 来自部署配置，数据库不保存连接配置，也不对配置项建立外键；
- grant 变更只影响新 submission。

repository 暴露以下边界：

```python
class RunTraceBackendGrantRepository:
    async def list_for_principal(self, principal_id: str) -> list[RunTraceBackendGrantRecord]: ...

    async def replace_for_principal(
        self,
        principal_id: str,
        grants: Sequence[RunTraceBackendGrantInput],
        *,
        actor: str,
        now: datetime,
    ) -> list[RunTraceBackendGrantRecord]: ...
```

授权解析由 policy 层完成：

```python
class RunTraceBackendAccessPolicy:
    async def resolve_for_submission(
        self,
        principal_id: str,
        requested_sinks: Sequence[str] | None,
    ) -> ResolvedRunTraceBackendSelection: ...
```

`replace_for_principal` 锁定该 principal 的现有 rows，在一个事务内计算新增、修改和删除，防止默认顺序出现中间态。

## Submission 模型和解析

公开 submission 顶层增加：

```json
{
  "run_trace_backends": {
    "sinks": ["tdsql-team-a", "cos-archive"]
  }
}
```

对应严格模型：

```python
class RunTraceBackendSelection(StrictModel):
    sinks: list[str]

class SubmissionBase(StrictModel):
    run_trace_backends: RunTraceBackendSelection | None = None
```

校验规则：

- `sinks` 至少一个、最多八个；
- principal defaults 同样最多八个；
- 名称符合现有 backend name 规则且不超过 64 字符；
- 不允许重复，保留用户给出的顺序；
- 显式选择的每个 backend 必须已配置、`user_selectable=true` 且存在 enabled grant；
- 普通用户面对不存在、未暴露或未授权名称时统一得到 `run_trace_backend_not_available`。

Control Gateway 必须使用认证上下文中的 `principal_id` 解析，不接受请求体伪造 owner。解析在创建 execution 的同一事务中
完成，结果写入已有 `executions.driver_config_json`：

```json
{
  "run_trace_backends": {
    "sinks": ["tdsql-team-a", "cos-archive"],
    "append_artifact_fallback": true
  }
}
```

`append_artifact_fallback` 是 Control Gateway 固化的内部 wire 标记，不属于公开 submission schema。沿用这个旧名称是为了兼容
滚动升级期间的旧 Manager；新 Manager 将其解释为追加所有已注册且对当前 execution 可用的 required backend，而不只追加
artifact。用户不能直接提交 `driver_config`。管理员专用的 raw execution API 保持现有精确选择能力和权限边界，不会被自动
追加 required backend。

## 默认与兼容策略

新增配置：

```python
require_principal_run_trace_backend_grant: bool = False
```

解析顺序：

1. 用户显式传 `sinks`：始终逐个验证 grant；
2. 用户未传且 principal 存在 enabled 的 `default_order`：使用 principal defaults；
3. 用户未传、没有 principal defaults、开关关闭：不写 execution 选择，继续使用当前
   `default_run_trace_sinks`；
4. 用户未传、没有 principal defaults、开关打开：返回
   `run_trace_backend_selection_required`。

该行为支持先部署能力、配置 grants，再开启强制模式。显式选择从第一天起就不能绕过 grant。

用户选择只替换可配置 backend。注册并且当前 execution 具备 artifact scope 时，`artifact` 仍由平台追加为最后一个
fallback；用户不能在请求中添加、删除或重排它。

## 幂等、重试和 Batch

- `run_trace_backends` 进入 normalized submission 和 `submission_sha256`；同一 idempotency key 改变选择会冲突。
- 命中已有 execution 时直接返回已持久化选择，不按当前 grant 重新解析。
- execution retry 复用原 `driver_config_json`。
- batch parent 在提交时只解析一次。
- `child_template` 保存 parent 的完整 `driver_config`，所有 child 和 child retry 复制该快照。
- batch 运行中撤销 grant 不影响已存在的 parent 或后续由该 parent 创建的 child。
- backend 从 Manager 配置中消失时应明确失败，不能静默改写到另一 backend。

## API

用户侧：

```text
GET /v1/access/run-trace-backends
```

```json
{
  "items": [
    {
      "name": "tdsql-team-a",
      "kind": "tdsql",
      "display_name": "Team A Trace Database",
      "description": "Team A production traces",
      "default_order": 0
    }
  ],
  "required_sinks": [
    {
      "name": "artifact",
      "kind": "artifact",
      "display_name": "Artifact Storage",
      "description": null,
      "system_managed": true,
      "required": true
    }
  ]
}
```

管理员侧：

```text
GET /v1/admin/run-trace-backends
GET /v1/admin/principals/{principal_id}/run-trace-backends
PUT /v1/admin/principals/{principal_id}/run-trace-backends
```

批量 PUT 请求：

```json
{
  "grants": [
    {"backend_name": "tdsql-team-a", "enabled": true, "default_order": 0},
    {"backend_name": "cos-archive", "enabled": true, "default_order": 1}
  ]
}
```

所有管理写操作需要 admin 角色，并写入一次聚合 audit event，其中分别列出新增、修改和删除的 backend。用户查询只返回当前
principal 的 enabled grant，不接受 principal 查询参数。

## Console

管理员在现有 `Access` 模块增加 `Trace backends` 视图：

- 选择 principal；
- 勾选可授权 backend；
- 配置默认 sink 顺序；
- 一次性保存整个 grant 集合；
- 下方展示 principal、backend、kind、默认顺序、状态和更新时间。

普通用户在 `My Environments` 增加只读的 `Run Trace Backends` 区域，展示已授权 backend、kind、描述和默认顺序。
平台 `artifact` 以 required 状态单独展示。Console 不展示 host、database、username 或 Secret 引用。

MVP 不提供 Web submission 表单。CLI 和 API 是选择入口。

## CLI

single、live 和 batch 共用可重复参数：

```bash
nexus submit single config.yaml \
  --environment production \
  --run-trace-sink tdsql-team-a \
  --run-trace-sink cos-archive
```

CLI 按命令行顺序构造 submission 顶层 `run_trace_backends.sinks`。未提供参数时省略该字段。

## TDSQL Trace Database Access

通用 backend grant 控制任务是否可以向 backend 写 trace。TDSQL 只读数据库账号是额外能力，但必须受同一 backend grant
约束：

- `Trace Database Access` 只列出当前 principal 已 grant 且在 `trace_database_access` 中启用的 TDSQL backend；
- create、reset、restore 和 reconcile 操作都在执行前重新验证 grant；disable 始终保留，供管理员安全收口账号；
- generic、local、artifact 等 backend 不进入数据库账号页面；
- 撤销 backend grant 时，只有账号明确处于 `disabled` 且没有进行中的 operation 才允许继续；`ready`、`error` 和操作中状态均返回
  `trace_database_account_must_be_disabled`；管理员先在现有 Database Accounts 页面 disable，再撤销 grant；
- 禁用只读账号不影响平台 writer，也不影响通过 execution API 读取已有结果。

撤销 grant 后，用户不能创建使用该 backend 的新 execution。已经持久化的 execution 和历史 result API 仍按 owner 权限
读取，不重新检查当前 grant。

迁移 `0029` 会为升级前已存在且仍活跃、状态异常或处于操作中的 TDSQL 只读账号回填 enabled、非默认 grant，避免外部账号仍
可读取而控制库没有对应授权的孤儿状态。只有已明确 disabled 且没有进行中 operation 的账号不回填；需要恢复时由管理员先显式
授权。

## Backend 生命周期约束

MVP 不持久化 backend revision。运维必须遵守：

- Secret 实际值可以在同一引用下轮转；
- 修改 host、database、sink/source class 或存储语义时创建新 backend 名称；
- 旧 backend 先设为 `user_selectable=false` 并撤销新授权，但继续注册 source；
- 等所有历史数据超过保留期后再删除旧 backend。

后续版本可在 execution 和 envelope 中固化不含 Secret 值的 config digest，并在 Manager 启动 execution 时校验。

## 失败语义与安全

- 多 sink 仍为 all-required：任一 sink 失败，execution 失败；
- 用户指定顺序同时决定写入 metadata 顺序和 source fallback 顺序；
- grant 检查发生在 Control Gateway，Manager 不访问授权数据库；
- grant 撤销只影响新 submission；
- backend 连接配置不能进入 submission、execution、audit、Console payload 或日志；
- 用户查询未知与未授权 backend 使用相同错误码；
- 管理 API 验证 backend catalog，不能为未配置或不可选择 backend 建立 grant。

## 实施顺序

1. 扩展 backend 安全元数据，增加 grant ORM、migration、record 和 repository，并以 repository 单测覆盖替换、默认顺序和
   授权解析。
2. 增加管理员/用户 API 和 audit，覆盖角色校验、字段脱敏、批量更新及错误码。
3. 扩展 submission 模型和 repository，固化 driver config，并覆盖显式选择、principal default、兼容 fallback、强制模式和
   幂等冲突。
4. 让 batch child template 继承 parent driver config，覆盖 parent、child 和 retry 一致性。
5. 增加 CLI `--run-trace-sink`，覆盖 single、live、batch payload 和顺序。
6. 增加 Console 管理与只读视图，并运行前端构建测试。
7. 将 TDSQL account list/create/reset/restore/reconcile 与 grant 联动，覆盖 grant 撤销前的账号禁用约束。
8. 更新 cloud API、管理员 backend 文档和 Helm 生产配置示例，完成定向测试、distributed 单测、Ruff、ty 和 Console build。

## 验收标准

- 管理员能给一个 principal 授权至少两个不同类型的 backend，并配置默认顺序；
- 普通用户只看到自己的 enabled grants 和平台 required sink；
- single、live、batch 能显式选择一个或多个 backend；
- 未授权、未配置、不可选择和重复 backend 在提交阶段被拒绝；
- 省略选择时依次支持 principal defaults、兼容 global defaults 和强制报错；
- execution 持久化有序选择，batch child 与 parent 完全一致；
- artifact 平台 fallback 不被普通用户选择覆盖；
- 多 sink envelope 顺序与 effective selection 一致；
- CLI 参数顺序原样进入 submission；
- TDSQL 账号 list 和凭证操作不能绕过 backend grant；
- 撤销仍有 active TDSQL 账号的 grant 会被安全拒绝；
- 现有未使用新字段的客户端在兼容模式下行为不变；
- 所有公开响应、日志和 audit 都不包含 backend Secret 或连接密码。
