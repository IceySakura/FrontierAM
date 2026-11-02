# 常见问题

先确认失败发生在哪一步，再检查对应配置。提交已被接受时保存 `execution_id`，不要因查询失败反复创建新任务。

| 失败阶段 | 先看什么 |
| --- | --- |
| 接入或认证 | API 入口、认证方式、HTTP 状态与错误体 |
| 任务编译 | 本地命令 stderr、Generator 或 Dataset＋Profile 输入 |
| 代码上传或依赖构建 | Control Events、本地构建报告、source 与 wheel 版本 |
| 排队或启动 | Execution 的 `phase`、Environment Grant 和容量 |
| 任务执行 | Control 错误、任务/Runtime 日志、模型与 Runtime 凭据 |
| 结果读取或导出 | 任务是否终态、Application 入口、结果/轨迹授权 |

## `401 Unauthorized`

按[认证方式](access-and-environments.md#选择认证方式)分流：

- **个人 PAT**：检查文件非空、权限为 `0600`，PAT 未过期且已授权 `nexus-agent`。
- **应用身份**：检查 Control/Application API 订阅、应用签名与时间、`app_id` 的 Nexus 用户绑定。
- **HTTP `200` 但响应含 `errcode`**：仍可能是 SGW 拒绝，使用接入页的 `nexus_json` 检查错误体。

个人 PAT 文件检查不需要打印其内容：

```bash
wc -c < "$TAIHU_TOKEN_FILE"
stat -c '%a %n' "$TAIHU_TOKEN_FILE"
```

用选定的认证方式请求 `/v1/me`，先确认身份，再检查 Environment 授权。不要复制浏览器 Cookie 或 `x-tai-identity`。

## `403 Forbidden` 或 Environment 不可用

先看错误来源：

- `scope_required`、`environment_grant_not_found`、`environment_scope_required` 等 Nexus 错误：打开 **My Environments**，
  确认目标 Environment 的 Access 为 `granted`，Grant 包含对应操作范围且额度足够。
- `acl.14001` / `acl_denied` 等入口网关错误：先检查访问路径及 iGate 授权；修改 Environment 配额不能解决网络入口拒绝。
- 应用网关订阅拒绝：检查请求是否使用当前应用获批的 API 发布前缀。

Environment 不可见、无法申请配额或额度不足时联系管理员；两套部署的 Environment ID 和权限不能混用。

<a id="物化失败"></a>

## 任务编译失败

先定位[第 3 步：任务编译](local-validation.md#3-任务编译)。`materialize` 的诊断信息写入 stderr，
stdout 是 JSON 摘要。核对 Generator、位置参数配置文件、`--data/--use` 三种入口没有混用；
单独的 Profile YAML 应通过 `--use` 引用。

Harbor 数据要求 Python >= 3.12、`harbor==0.20.0` 和包含有效 `task.toml` 的任务目录。
`local_task_files_*` 错误需检查目录/ZIP 是否存在、类型及声明的 size/hash 是否匹配；
`local_generator_harbor_task_files_unsupported` 表示使用了旧 Harbor Generator 的本地路径，应改用
`t3-harbor:///` Dataset + Profile。显式 v1 不支持本地任务文件，使用默认 v2 自动生成 v3 + Data List。

## Nexus wheel 构建失败

对应[第 4 步：构建 Nexus wheel](local-validation.md#4-构建-nexus-wheel)。确认在 Nexus 项目根目录执行 `uv build --wheel`，
并检查构建日志中的项目元数据、文件或构建后端错误。先让这一步成功，再进入 `modules wheel`。

## 依赖构建失败

从[提交前本地验证](local-validation.md)重现同一任务输入，依次完成任务编译、Nexus wheel、依赖 wheelhouse。
根据报错发生的阶段继续看下面的依赖冲突或离线安装说明。

确认 `--project-root` 与 `--source` 都指向完整 Nexus 项目：根目录 `pyproject.toml` 的 distribution 名必须是
`nexus-agent`，版本必须与本次构建的 wheel 一致，并存在通过 `uv lock --check` 的 `uv.lock`。`--source` 指向 Nexus checkout
父目录会报 `dependency_source_project_invalid`；source 与 wheel 的版本不同会报 `dependency_input_mismatch`。

### 依赖冲突

Cloud 联合解析基础声明、extras、项目显式约束和模块 Host 声明，优先保留 `uv.lock` 中兼容的版本。
声明允许升级时，旧锁版本不足会触发重新选版；声明之间存在真实冲突时会报 `dependency_module_lock_failed`。
根据解析器给出的包名和约束链，核对源项目与模块元数据，并同步重建 Nexus wheel。
旧 Builder 的 `dependency_module_lock_conflict` 表示基础锁版本被当作硬约束；使用联合解析需要更新 Builder 镜像。
`module-wheel-report.json` 只在构建成功后生成。同一 Batch 的 child 应使用相同的模块依赖，否则应拆分 Batch，
或先在项目中声明依赖并集。

### 离线安装失败

`dependency_wheelhouse_inconsistent` 表示 wheel 集合与最终锁不一致，例如缺少依赖、包含未锁定的包或同一包的多个版本。

`dependency_offline_install_failed` 或 `dependency_offline_validation_failed` 表示 wheel 构建后的
无索引分阶段安装、完整依赖检查、最终锁版本比对或 Nexus 导入/版本校验未通过。检查报错中的缺失 wheel、Python/平台兼容性及包版本；
完整复现命令见[第 5 步：构建依赖 wheelhouse](local-validation.md#5-构建依赖-wheelhouse)。

本地可导入不代表 Builder 一定具备未声明的系统依赖；`dependency_wheel_build_failed` 需先解决具体包的构建错误。
云端构建错误可在 Run 详情的控制事件中查看。校验产物应满足[第 6 步的成功判据](local-validation.md#6-检查验证结果)。


## 任务长时间停在 `queued`

先查看 Control 详情的 `phase` 和 **Control Events**，按 `phase` 分流定位是构建、资源申请、调度策略门禁还是槽位排队，状态名 `queued` 本身不足以判断卡在哪一步：

- `building_dependencies` / `dependency_build_failed`：依赖镜像或 wheelhouse 仍在构建或已失败，按[查看构建日志](runs-and-results.md#查看构建日志)和上文[依赖构建失败](#依赖构建失败)排查。
- `waiting_resources`：Environment Allocation 仍在申请或拉起 Manager，请求 `GET /v1/executions/{execution_id}/allocations` 查看 `status`、`desired_manager_count` 与 `ready_manager_count`（底层 `last_error_code` / `last_error_message` 仅管理员可见）。
- `waiting_policy`：被调度策略门禁暂缓。请求 `GET /v1/executions/{execution_id}/scheduler-policy` 查看当前策略决策，按[状态与原因查询](../reference/cloud/scheduler-policies.md#状态与原因查询)中的 `reason_code` 对照表区分策略门禁（如潮汐策略下的 `quota_exhausted`、`hard_brake`）与信号或配置异常（`policy_unavailable`、`policy_invalid`、`signal_expired`、`signal_incomplete`）。Batch 父任务可查 `GET /v1/executions/{parent_id}/batch?include_status_counts=true` 的 `child_waiting_policy_count` 及父任务的 `/scheduler-policy` 查看共享 `reason_code`（父任务 `next_schedule_at` 固定为 `null`，单任务下次准入时间需查询 Single 或具体 child）。
- `queued`：构建与资源已就绪、调度策略未拦截，正在等待调度（空闲 Manager 槽位或用户并发额度）；检查个人 Grant、Environment 容量和当前并发占用。

按实际任务规模设置 `--provisioning-timeout-seconds`、`--queue-ttl-seconds`；Batch child 的排队上限使用
`--child-queue-ttl-seconds`。延长超时不会修复缺少权限、容量或依赖构建失败。

若任务已进入 `running`（或 Batch 处于 `phase = running_children`）但长期无业务进展，除了查看[任务日志](runs-and-results.md#查看任务日志)，还可请求 `GET /v1/executions/{execution_id}/request-policy` 确认是否因共享请求预算（`below_minimum`、`quota_exhausted` 等）导致模型请求暂停（`effective_rate_per_minute = 0`），说明见 [Scheduler 策略与潮汐容量准入：状态与原因查询](../reference/cloud/scheduler-policies.md#状态与原因查询)。

## 模型配置加载失败

快速开始和 Batch 示例使用 `MODEL_SERVICE_API_APP_ID`、`MODEL_SERVICE_API_APP_KEY`、`MODEL_SERVICE_API_BASE_URL`。
`${oc.env:...}` 在本地解析，缺少变量会在提交前失败；凭据无效通常在模型调用阶段失败。

`api_key__env` 等引用在云端执行时才读取，本地 `export` 不会自动转发。按[任务凭据](access-and-environments.md#任务凭据)
检查注入位置。Nexus API 的个人 PAT 不能代替模型服务凭据。

## SWE 示例无法启动 Runtime

确认示例需要的 `GONGFENG_RUNTIME_TOKEN` 已配置，并且任务所在执行环境能访问 Runtime API。
本地插值的示例与运行时 `token__env` 引用的配置，读取凭据的时机不同。

沙盒资源不足时可能在 Runtime 创建阶段等待或失败；这与 Execution Manager 的 CPU、内存配额是两套资源。

## Dataset 报 `missing body`

Batch 的 Dataset＋Profile（Compose）输入需要 Nexus `DatasetItem`，每条 item 至少有 `id` 和 `body`。
原始评测数据不能直接当作 RunInput 片段；应像示例一样由 `body` 指定任务字段，再与 Profile 合并。

## `batch task list exceeds ... limit`

本地生成默认上限为 4 GiB，可通过 `NEXUS_SUBMIT_MAX_TASK_LIST_BYTES` 调整；但 Cloud 默认服务端仍限制 10,000 items、1 GiB
总大小和 16 MiB 单行，Environment/Grant 还可以覆盖这些值。优先减少每条 item 中重复的大字段、改用 Dataset/远程引用，或拆分
Batch。只调大本地环境变量不能绕过服务端额度。

## `--wait` 中断或查询失败

Single/Batch 打印 accepted 的 `execution_id` 后，远端任务已被接受；本地网络中断不会停止它。
重新打开控制台或查询原 Execution 即可。Live CLI 不支持 `--wait`，按 [Live 指南](live.md)读取状态与事件。

只有提交响应丢失、无法取得 `execution_id` 时才需要重试提交；复用保存的幂等键和原输入，不要生成新 key。

## 无法读取结果或导出 JSONL

先检查任务是否已终态，并确认当前身份有读取权限。Control 的 `result_json` 可能只是 external pointer，完整结果应从
Application API 或控制台 **Spec / Result → Load result** 获取。

- 在形成 `RunOutput` 前失败的任务，可能没有最终结果；先读取 `error_json` 和 Control Events。
- Batch parent 结果不是所有 child 输出的集合；用 child ID 读取，或使用[导出脚本](runs-and-results.md#导出标准-jsonl)。
- Live 返回 `trace_events_not_retained` 时，事件已无法回放，改读最终结果和状态，不要无限重连。
- Batch 导出还需要数据库只读账号；未看到申请入口时，按[账号申请说明](runs-and-results.md#申请数据库只读账号)检查授权。
- 数据库连接超时先排查网络；认证失败再核对用户名与密码，避免为网络问题反复重置密码。

## 大结果、TDSQL output 省略与压缩 Artifact

以下参数属于服务端 `DistributedConfig`，不是 Task 的 `RunInput.runner` 配置：

```yaml
artifact_run_result_compression: zstd             # 默认 none；zstd 使用 level 3
artifact_run_result_max_size_bytes: 1073741824    # 上传文件上限，压缩开启后按压缩字节计算
artifact_run_result_max_decoded_size_bytes: 2147483648  # JSON 解压后的上限，默认 2 GiB
artifact_run_result_decode_concurrency: 1         # 每个 Application Gateway 的结果解码并发
run_trace_sinks:
  hyway-tdsql-3:
    # 合并到该 backend 的完整原配置，保留已有连接参数及 credential 引用。
    sink_config:
      max_run_output_bytes: 8388608               # 显式启用 8 MiB 阈值；默认 0，关闭省略
```

只有同一个 run 启用了 `artifact` 结果后端时，TDSQL 才允许省略超限的根 `run.output.output`。
判断按整条事件序列化后的 UTF-8 大小进行；省略后保留事件身份、时间和业务成功标记，并增加
`output_omitted=true`、`output_omitted_reason=size_limit`、`original_event_bytes`。
其余 task、agent 和 LLM span 不适用此规则。Artifact 中的完整 `RunOutput` 不受 TDSQL 投影影响。

压缩只改变 Artifact 的文件编码，kind、对象命名、引用协议以及内部 JSON 的
`version/status/output/error` 不变。新读取端同时接受旧 JSON 和带内容长度、checksum 的 zstd frame，
验证压缩文件的 SHA256 后再解码，拒绝超出解压上限、损坏、截断或带额外尾部数据的内容。
若手动下载压缩文件，可保存为 `result.json.zst`；不能直接对它调用 `json.load()`。

运行器在 drain 前捕获业务输出，并在关闭原始 Trace sink 前发布结果。Trace 收尾失败仍会将 execution
标为失败，但已发布的 Artifact 可通过 `/result` 读取；不会因为 `result_json` 尚未写入就返回空对象。
错误路径的结果指针可带 `trace_complete=false`，不表示原始轨迹完整。捕获到输出后，Artifact 发布失败会明确
报 `ResultPersistenceFailed`；Manager 在完成上传前退出的输出仍无法保证恢复。

结果导出保持原 JSONL 字段。Batch 优先读 TDSQL 时，最新 attempt 的 output 被省略就回源 `/result`，
不取较旧 attempt 的 output。原始 Trace 导出保留数据库中的 null 和省略标记，不自动补写数据。
直接读取 TDSQL 的业务脚本需要实现回源；只通过结果 API 的调用方无需处理 zstd。

部署先更新 Application Gateway、结果读取 SDK/导出代码，再更新 Manager；先保持压缩为 `none`，
省略默认关闭，不必向旧 wheel 的 sink 配置加入新参数。确认执行端已升级后，再启用 zstd 和省略阈值。
回滚关闭新写入策略即可，保留 zstd 读取能力。存储压缩不减少恢复后的对象大小；解码并发和解压上限应结合
实例内存设置，Python 对象本身还有额外内存开销。
结果读取被取消或超时时，会先等待正在执行的读文件、解码或响应编码线程结束，再释放资源和并发名额；
因此返回取消或超时错误可能晚于触发时刻。

## Building 阶段日志

Console 的 **Builds** 入口按构建列出记录，点击 `bld_*` 打开构建详情。
列表读取 `GET /v1/builds`，详情直接读取 `GET /v1/builds/{build_id}/logs?lines=200`，使用日志响应中的构建状态和 attempt。
接口沿用 execution 的 owner/environment 读取权限；列表不依赖 Allocation，因此共享构建只显示一次。
Run 和 Infrastructure 的 Allocation 表中，Dependency 列链接到该构建。
任务级 `/v1/executions/{execution_id}/build-logs` 仍可查询 Batch 的关联构建。
每个 item 复用现有 Logs 的 `source`、`availability`、`lines`、`partial` 等字段，另带 build ID、attempt 和 status。
页面和 MCP 均可调用这个接口；构建期间读取 Builder 的本地 JSONL。
通过同一次 Pod 查询返回 `phase`，在构建状态徽标中展示等待创建、等待调度、容器准备和镜像拉取失败。
Builder 已退出但控制面尚未确认终态时，徽标显示 `Finalizing`，数据库状态仍由 Reconciler 确认；日志入库等待单独展示。
这些状态不会写入日志，也不增加单独的等待提示文案。
这不是事件历史；`ContainerCreating` 不等同于正在拉取镜像。
未配置智研时，构建结束后接口返回 `unavailable` / `build_log_expired`。
配置采集和查询后，终态通过智研回捞；入库等待沿用有界重试。

Builder 只写一个滚动 JSONL 文件，最多保留 8 MiB；查询最多返回 1000 行，各 build 共用 256 KiB 日志预算。
Builder 的子进程输出经脱敏后，通过标准库 `QueueHandler` / `QueueListener` 非阻塞入队并后台写入；文件轮转复用 `RotatingFileHandler`。
队列最多保留 4096 条记录，每条最多 8192 字符；满时淘汰最早的待写记录并计数，正常排空后写入丢弃提示。
退出最多等待 2 秒排空队列；写入持续阻塞时放弃剩余日志，不让日志 I/O 阻塞构建退出。这不是日志完整性保证。
命令标准输出超过 2 MiB 后仍继续写文件，超长单行省略；URL 凭据和常见 token 字段会脱敏。

发布需更新 Control Gateway（含控制台）、Environment Controller 和 Builder 镜像，无 Artifact 或数据库变更。
Gateway 需能访问 Builder TCP 8093；集群内 Kubernetes 的新权限仅为列举 Pod。
Builder 的新代码需通过镜像发布，更新任务 wheel 不会更新 Builder；既有 build 的镜像和 provider 快照不会自动切换。
通过数据库修改 builder 镜像配置时，需要重启 Control Gateway（提交时生成构建规格）及 Reconciler 加载配置。
已有构建记录不会自动替换镜像；本次代码修改不包含生产发布。

### Manager 文件日志配置

Manager 文件日志由部署或显式配置启用，普通动态 Kubernetes 容器不会由代码默认注入日志目录。
可通过 `environment_extra_env` 设置 `NEXUS_DIST_LOG_DIR`，目录需挂载且可写；TKEx 接智研时必须与采集挂载目录一致。
此配置在 manager 启动时生效，已启动且未配置文件日志的 manager 需要由新 Allocation 替代。
Run / Batch 的 Logs 复用这些文件；Builds 仅提供构建日志，查看进程日志请进入关联任务或管理员 Infrastructure 页面。


### Builder 智研接入（TKEx，可选）

使用 ProviderConfig 独立的 `builder_zhiyan_log`：`log_name`、`agent_image`、`volume_size_bytes`。
只有显式配置后，Builder 才添加与 Manager 相同的 agent 模板。
该配置只影响 Builder；Manager 继续使用原有 `zhiyan_log`，两者互不启用。
底层复用相同的配置类型与 sidecar 组装函数，接入名分别显式指定。

Builder 写 `/work/collected-logs/build.jsonl`，agent 通过共享卷读取 `/data/logs/build.jsonl`。
智研接入规则需要覆盖该文件，并解析 `dependency_build_id`、`build_attempt` 查询字段。
查询使用 Control Gateway 现有 `zhiyan_log_query` 和凭据配置，按 build ID 与 attempt 隔离。
成功的 Builder 退出且产物校验通过后，立即标记 `ready`，后续调度不等待日志上传。
Reconciler 在 `ready` 记录的 `next_retry_at` 保存 30 秒后的后台清理时间；重启后继续领取并清理 Job、Secret。
清理失败只重试清理，保持构建结果为 `ready`；30 秒是上传缓冲，不是已入库确认。
失败构建仍沿用原有等待、诊断收集和重试流程。
Agent 退出不会覆盖 Builder 的成功/失败结果，未上传的记录在 Pod 回收后无法恢复。

发布前确认接入的文件规则和字段解析，先更新 Gateway/Reconciler 代码，再更新 provider 配置；
使用新构建验证成功、失败两种情况的终态回捞，已有构建的 provider 快照不会自动更新。
智研采集和成功构建历史回捞已在 test 验证；本次后台清理变更仍需云端验证。


### Builds 列表索引

迁移 `0031_build_list_index` 为 `distributed_dependency_builds` 增加非唯一联合索引
`idx_dependency_builds_created_id(created_at, dependency_build_id)`，支持全局列表排序和游标分页。
列表仅读取展示字段，不加载构建 spec、provider 快照和产物 JSON。
该迁移支持重复执行和回滚；回滚只删除索引，不删除构建记录。云端执行迁移需单独确认。
owner/environment 过滤查询仍需根据实际执行计划评估，不由这条索引保证。
