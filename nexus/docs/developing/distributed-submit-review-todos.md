# Distributed Submit Review 待改动列表

本文件只记录 review 后已经明确确认需要修改的事项。讨论中的问题、解释和尚未确认的建议不进入列表；事项完成并验证后移入
“已完成”。

## 待处理

### 1. 太湖鉴权、默认系统 Token 与自助额度

- [x] 状态：已完成

- 公网请求只接受一种太湖身份：个人 PAT 由太湖入口消费并注入 `x-tai-identity`；应用使用 `x-rio-*` 调用 SGW，SGW 校验调用方
  签名和订阅后向后端增加被调用方签名。Nexus 不把 Bearer 解释为 submit token，也不直接校验调用方应用 Token。
- Gateway 使用同一个太湖应用 Token 解密个人 JWE，或校验 SGW 被调用方签名。个人身份严格校验 `alg=dir`、`enc=A256GCM`、
  `Expiration` 并以 `StaffId` 生成 principal；应用身份按 `x-rio-paasid` 查找管理员配置的 token binding。两种身份混用时 fail close。
- 首次合法访问在同一事务内幂等登记普通 principal 和唯一 `system_default` token row；没有明文、hash、prefix、Environment 或
  scopes，Console 只显示“默认”。保留 `SubmitTokenRow.api` 数据模型，但不开放 token API 或 Tokens 页面。
- Environment 可配置完整 `default_quota`；用户用“默认”身份整份认领，幂等创建 Grant 和 usage row。认领后的 limits 为快照，
  disabled Grant 不能由用户恢复，认领不预留计算资源。
- CLI 参数改为 `--taihu-token` / `--taihu-token-file`，token file 不进入 source package。
- 验证首访并发幂等、无效 Bearer 不影响 JWE、跨 owner 404、default quota 认领/快照/disabled Grant、旧 token 路由 404、
  DevCloud 无太湖链路和 runner-test 双太湖域名。

DevCloud 已通过 CLI single、live 和 batch 全链路；runner-test 已验证太湖身份解密、管理员登记、普通用户隔离、跨 owner 404、
默认额度幂等认领、single/live 最终结果及 Artifact 读取。Control/Application Gateway 均以版本 5 配置完成滚动，公网地址保留
普通浏览器太湖域名，内网调用继续使用 CLB 与 internal credential；本次未发布或修改 runner-prod。

### 2. InstanceManager 卷与持久化

- [x] 状态：已完成

统一 Manager 容器内目录，按 Environment 类型选择底层存储：

- predefined Manager 每个 Pod 使用独立 RWO PVC，并将整块 PVC 挂载到 `/data`；Local Run Trace、日志、venv、cache 和
  BatchControl 目录均随 PVC 持久化。
- dynamic Manager 不创建 PVC，将受 `max_workspace_bytes` 限制的 `emptyDir` 挂载到 `/data`；Pod 删除后回收全部内容。
- `/tmp` 使用独立、受限的 `emptyDir`，不再与 `/data` 或 cache 目录复用同一个卷。
- 移除将同一个 `manager-cache` 分别挂载到 `/data/nexus-environment`、`/data/nexus-distributed` 和 `/tmp` 的路径别名；
  `cacheRoot`、Run Trace 和 BatchControl 的现有容器内路径保持不变。
- Secret 按第 3 项统一为只读文件挂载，不并入数据卷。TKEx 日志 sidecar 仍由平台注入，不由 Helm chart 创建；runner-prod
  现有 `/logs/nexus-distributed` 日志卷保持不变。
- dependency builder 的 `/work` 和 Application Gateway 的 `/data/nexus-upload-spool` 继续使用受限 `emptyDir`。

落地项：

- 为 `instanceManager` 增加 persistence 和独立 tmp 配置；生产启用 PVC，隔离测试环境可以显式回退到受限 `emptyDir`。
- StatefulSet 使用名为 `data` 的 `volumeClaimTemplates`，缩容或删除 StatefulSet 时保留 PVC。
- Dynamic Environment renderer 将临时数据卷统一挂载到 `/data`，并为 `/tmp` 创建独立临时卷。
- 补充 Helm render、Kubernetes controller 和目录持久化语义测试。
- 从现有 `emptyDir` 切换到 `volumeClaimTemplates` 时先 drain Manager，再重建 StatefulSet；不能依赖原地修改不可变字段。

### 3. 平台配置与 Secret 收口

- [x] 状态：已完成

- Kubernetes 应用 Secret 只保存 database URL；内置 MySQL root password 使用独立基础设施 Secret。
- 普通平台配置存入 `distributed_config_overrides`；平台长期密钥存入 `distributed_config_secret_overrides`，统一映射为
  `DistributedSecrets`，不参与 repr、dump 或日志。
- ProviderConfig 业务表只保存非敏感参数与 `credential_ref`；远端 provider 凭据由 Reconciler 从
  `DistributedSecrets.provider_credentials` 解析。in-cluster ServiceAccount 和 imagePullSecret 继续由 Kubernetes 管理。
- allocation/build scoped Artifact ticket 继续使用短期 Secret，随资源生命周期创建和清理。
- 工蜂、模型和 Wedata 凭据属于 submission-scoped 用户输入，不进入 Helm 或平台配置；Wedata 单独存入
  `distributed_execution_credentials`，公开 API 不返回 payload。
- 所有角色启动时按 `*/* -> role/* -> role/identity` 加载一次配置；更新后滚动重启，不做热加载。
- 配置 CLI 支持普通配置的有效值展示、乐观版本控制和脱敏 secret 管理；密钥写入只接受文件或 stdin。
- DevCloud 验证后发布 runner-test，验证双太湖域名、用户隔离和内网 internal credential；不发布 runner-prod。

runner-test 已完成太湖首次访问、显式管理员提升、普通用户 403、Control/Application 跨用户 404 与临时用户清理验证；脱敏证据位于
`outputs/distributed-submit-e2e/runner-test-tai-auth-isolation-20260729.json`。本次未发布或修改 runner-prod。

### 4. Source-package 执行隔离

状态：后续处理，不在本期 MVP 实现

当前 source package 只面向可信内部用户。Manager 和 remote generator 处于同一个 Pod 与文件系统信任域；即使平台长期密钥改为
启动快照，用户代码仍可读取 Manager 进程持有的 database URL 和 internal token。第 3 项只收口部署配置，不提供用户代码隔离。

后续支持非可信 source package 时：

- Manager 使用服务端固定镜像和代码，用户提交的 wheel、generator 和 task 代码在独立 Pod、容器或 sandbox 中执行。
- 用户代码不挂载 database URL、internal token 或其他控制面凭据，子进程环境使用显式白名单，不继承 Manager 的完整环境。
- 用户代码只通过受限 API 访问控制面，并使用短期、allocation-scoped 凭据完成 Artifact 读取、结果上传和 runtime 调用。
- 限制用户执行单元的网络访问、资源和生命周期；停止或回收 allocation 时同步终止并清理对应执行单元。

### 5. Bootstrap Secret 示例与安装文档同步

- [x] 状态：已完成

Helm 的长期应用 Secret 只保留 `database-url`；内置 MySQL 的 root password 使用独立 Secret。其他普通配置和平台长期密钥在
db-init 后通过配置 CLI 导入数据库。

落地项：

- Secret 示例只展示 database URL 和可选内置 MySQL root password，不保留已迁移到数据库的旧 key。
- README 说明两阶段启动：先运行 db-init，再导入配置和密钥，最后启动长期进程。
- 生产 Helm 默认要求 `*/*` 普通/密钥配置基线且禁止 `auth_mode=disabled`；本地示例显式关闭该启动闸门。
- 安装示例测试验证应用 Secret 不含其他长期密钥，动态 workload 只引用 database Secret 和短期 ticket。

### 6. 镜像构建源与可复现性

- [x] 状态：已完成

当前 distributed 镜像构建没有显式统一 pip/uv 源，workload 镜像还会在构建时读取 GitHub `main` 分支上的 LiteLLM model cost
map。相同代码在不同时间可能产生不同镜像，受限网络下也可能无法构建。

落地项：

- universal、workload 和 builder 镜像统一使用部署侧指定的 pip/uv mirror，默认使用腾讯镜像；用户提交不能覆盖该配置。
- LiteLLM model cost map 使用固定版本或 commit，并校验 sha256；不能直接依赖可变的 `main` 内容。
- release 构建固定基础镜像 digest，记录镜像构建输入，并增加测试确保不存在未固定的远端构建依赖。

### 7. Submit E2E 覆盖补齐

- [x] 状态：已完成

所有原定在 Kind 执行的容器集群 E2E 均改在 DevCloud 测试容器集群执行，原有场景和验收指标不删减。此前 Kind 结果只作历史
参考，不能替代本分支的 DevCloud 验收证据。

当前 DevCloud dynamic 已跑通 source-package/wheelhouse smoke、25 Manager 的 2000-task batch、execution Manager 恢复和
batch Manager 恢复；predefined 已跑通兼容 live、SSE/input、runtime proxy、容量与重调度。补充验收已覆盖：

- [x] 从真实 shell 进程执行 `nexus submit single`、`nexus submit live` 和 `nexus submit batch`，核对输入上传、提交响应、状态推进、
  最终状态及结果产物；不能只使用 typed HTTP API 或上传 helper 单测作为入口验收。
- [x] 在 TKEx dynamic Environment 分别完成一次 typed single 和 live；live 覆盖 SSE、`input.required`、输入提交和正常结束。
- [x] 在 predefined pool 分别完成 typed single、live 和 batch，验证 config-only 限制、Manager 选择及终态产物。
- [x] 完成 dataset reference batch：从云端 dataset 引用生成 task list，执行所有 child，并核对 batch 汇总结果。
- [x] 完成真实 Taiji Dataset E2E：固定真实 release 版本而非 `@latest`，
   通过 submission-scoped `data_credentials.wedata` 提供 Wedata key，覆盖 UC/IO 元数据与版本读取、分页 materialize、task files、全部 child
   结果及资源回收。最小 smoke 可使用 `nexus-bench/cobol_demo_0702@v1`；分页场景可使用
   `nexus-test/terminal-bench-2@v4` 并设置安全的 `limit`。
- [x] 完成 wheelhouse fanout profile，验证 dependency build 结果向多 Manager 分发；基础 wheelhouse smoke 不重复计为该项完成。
- [x] 在 runner-dev 单独完成 `manager-2000` 容量验收：1 个 batch Manager、1999 个 execution Managers、16000 个 item、
   每个 execution Manager 4 slots，`max_inflight=7996`，`running` child 峰值至少 7200。该 stage 使用 16 个 Scheduler，
   必须先通过容量、DB 和镜像预检，最终验证 item/child/result 一一对应、allocation/resource 全量回收；25-Manager
   `config-baseline` 和 `source-heavy` 只作为常规基线，不计为 2000-Manager 大规模验收。

已完成项的脱敏证据位于 `outputs/distributed-submit-e2e/`：

- CLI、DevCloud Dataset 和 predefined typed：`devcloud-cli-smoke-20260724.json`、
  `devcloud-dataset-20260724.json`、`devcloud-predefined-typed-20260724.json`。
- TKEx typed single/live：`runner-dev-final-20260725.json` 的 `typed_smoke`。
- Taiji Dataset：`runner-dev-taiji-dataset-v2-20260725.json`。
- wheelhouse fanout：`runner-dev-wheelhouse-fanout-v3-20260725.json`。
- runner-dev 容量验收：`runner-dev-manager-2000-v6-20260725.json`。`config-baseline` 完成 2000/2000 item；
  `manager-2000` 拉起 2000 个 Manager，完成 16000/16000 item，`running` 峰值 7996，结果 Artifact 一一对应，
  restart/OOM 和上传失败增量均为 0，DB 观测峰值为 2336/32500 connections、`Threads_running=850`，allocation、
  StatefulSet、Service 和 Pod 全量回收。测试期间 runner-dev Application Gateway 使用 256 路结果上传、2048 waiters、
  300 秒 admission timeout 和 256 DB pool。

每项必须保存脱敏的提交配置、execution ID、关键状态、结果计数和清理结果；只有完整 E2E 成功后才从待处理移入“已完成”。

### 8. Artifact 下载背压与资源上限

- [x] 状态：已完成

Application Gateway 的 Artifact 下载会先完整写入 spool，当前没有独立的下载并发限制。Dynamic Manager 并行
启动时，大量 source package、wheelhouse 和 task list 读取可能同时打满 Artifact backend、Gateway 连接和
spool 空间；终态 `run_result` 读取还会将正文整体读入内存。

落地项：

- 为 Artifact GET 增加独立的 total/input/result 下载 semaphore，不与上传 semaphore 混用。
- 在接收请求前完成并发和 spool 容量准入；超限返回明确的可重试错误。
- 为 `run_result` 设置独立且明显小于通用 Artifact 的 size limit，并限制结果读取并发。
- 增加数百 Manager 并行 bootstrap 和并发 result 读取的容量与过载测试。

### 9. Artifact 上传 admission、deadline 与 lease 续租

- [x] 状态：已完成

上传在读取请求体前先等待并占用 input/result 与 total semaphore，但当前没有有界等待队列、正文 idle timeout 或总上传
deadline。持有有效 ticket 的慢客户端可以长期占用全部上传 lane。进入处理后又只申请一次 `receiving` lease，读取正文、
写 spool 和写 ArtifactStore 期间不续租；慢上传或 backend 抖动超过 lease TTL 时，maintenance 可能接管仍在运行的上传并
删除或重置对象。

落地项：

- 在读取正文前完成有界 admission；队列或 lane 已满时返回带明确重试语义的 `429/503`，不能无限保留等待中的连接。
- 为外部和内部 Artifact 上传增加正文 idle timeout 与总 deadline；超时后释放 admission 和 receiving claim，不依赖 Ingress
  的实现差异兜底。
- 增加带 `operation_id` fencing 的 lease 续租接口，在读取请求体和 Store PUT 期间定期续租。
- 进入耗时 Store 操作前再续租一次；续租失败后立即停止发布 ready。
- maintenance 处理批量任务时也要续租或缩小领取批次，避免后半批 claim 尚未开始就过期。
- 补充慢速/停滞上传、等待队列过载、长上传、Gateway 重启、lease 丢失和 maintenance 并发接管测试。

### 10. Reconciler 流程拆分为独立 loop

- [x] 状态：已完成

优先级：P1

当前 system loop 依次等待 core reconcile、EnvironmentController 和 ArtifactMaintenance，DependencyBuild 又串在
EnvironmentController 最前面。任意一个 provider、builder 或 ArtifactStore 调用变慢都会拖延其他控制流程。各类 reconcile
必须有独立的调度、超时和退避边界。

落地项：

- 拆分为 `dist-reconciler:core`、`dist-reconciler:environment`、`dist-reconciler:dependency-build`、
  `dist-reconciler:artifact-maintenance` 和 `dist-reconciler:batch` 五个独立 asyncio loop/task。
- core loop 只执行短 DB 事务；environment、dependency-build 和 maintenance 分别设置单次超时、batch size 和失败退避。
- 一个 loop 的可预期暂时故障不得阻塞或重启其他 loop；不可恢复的编程错误仍由进程级监督收敛。
- `tick_once()` 保留串行语义，仅供单测和手动调试；增加单个 loop 卡住时其他 loop 继续推进的测试。

### 11. ApplicationGatewayArtifactClient 生命周期

- [x] 状态：已完成

`ApplicationGatewayArtifactClient` 当前在未注入 session 时为每次请求创建 `aiohttp.ClientSession`，并在进程内
`_artifact_scopes` 中永久保留每个 Artifact 的 execution/assignment/attempt。这会在万级 child 下产生额外连接
开销和无界内存增长，且 Manager 重启后 `ArtifactTraceSource` 无法从该 map 恢复 scope。

落地项：

- Client 拥有或注入服务级共享 `ClientSession`，并在 InstanceManager、BatchControl 和 Reconciler 关闭时显式 close。
- 将 source 读取需要的 scope 放入可持久化的 trace metadata，或由调用方显式传入，删除无界
  `_artifact_scopes` map。
- 增加长时间复用、Manager 重启后 result 读取、session 关闭和连接池复用测试。

### 12. ArtifactStore 网络错误统一重试

- [x] 状态：已完成

ArtifactUploadCoordinator 只重试 `ArtifactStoreTransientError`，Generic Repository backend 目前只将部分 HTTP 状态映射为
该异常。DNS、建连超时、连接重置和请求超时不会进入配置的 backend 重试，一次短暂网络故障可能直接导致
task-list 生成或 batch parent 失败。

落地项：

- Store 边界将可重试的 `aiohttp.ClientError`、connect timeout 和 request timeout 统一映射为
  `ArtifactStoreTransientError`。
- 保持同一 object key、producer key 和 upload lease 重试，非幂等或身份冲突错误不得重试。
- PUT、GET、HEAD 和 DELETE 分别覆盖 HTTP 节流、连接中断、超时、永久 4xx 和重试耗尽测试。

### 13. Dataset 大数据集分页与读取效率

- [x] 状态：已完成

`iter_resolved_items()` 已支持 cursor 分页，但 `list_item_ids()` 仍只请求一页，大数据集可能静默截断。如果 Dataset
Service 列表接口只返回 ID，当前还会串行请求每个 item 正文，不适合万级数据集。

落地项：

- `list_item_ids()` 复用同一 cursor 分页实现，`limit` 对全量流生效，不得静默只返回首页。
- 明确 Dataset Service 的分页 response contract，使用真实接口覆盖 next cursor、空页、重复 cursor 和最后一页。
- ID-only response 使用有界并发拉取正文且保持输出顺序，或要求服务端分页直接返回完整 record。
- 增加跨多页的 dataset-reference batch 物化测试，验证 item 数、顺序、去重和 task-list sha256 稳定。

### 14. DependencyBuild attempt fencing

- [x] 状态：已完成

当前 claim 阶段提前递增 attempt，Controller 又先创建 Job、再把 DB 状态写为 `building`。Provider 已创建 Job 但请求超时、
或 Job 创建成功后 DB 提交失败时，会遗留旧 Job 并进入新 attempt；现有 ticket、Secret 和 output stage 都没有 attempt fencing，
旧 Job 可能向新 attempt 回写结果。

落地项：

- claim lease 与 build attempt 分离；先以 CAS 持久化 `building + attempt + provider_job_name`，再幂等创建对应 Job。
- Job、Secret、resource annotation、build-scoped ticket 和 wheelhouse producer identity 全部绑定 attempt。
- Application Gateway 写 wheelhouse 时校验当前 attempt；`stage_output()` 使用 build id + attempt CAS，旧 attempt 必须 fail close。
- Provider create 超时、DB 提交失败、后台删除中的旧 Pod 和旧 attempt 延迟上传都要有并发测试。

### 15. DependencyBuild claim lease 与异常收敛

- [x] 状态：已完成

Controller 当前先批量领取 claim、再串行处理，后半批 claim 可能在开始执行前就超过 lease TTL；已有 `renew()` 未被调用。
lease 丢失后异常分支再次执行 `_release()` 仍会抛错，并可能终止整个 Reconciler。

落地项：

- 改为领取后立即处理，或使用有界并发；耗时 provider 调用前后及执行期间按需续租。
- lease 丢失视为该 claim 已被其他 owner 接管，停止当前写入并幂等结束，不再使用失效 operation 重复 release/fail。
- attempt 只在真正开始新 builder Job 时递增，provider 查询、创建超时等控制面重试不得消耗 Job attempt。
- 覆盖批量 claim 排队过期、续租失败、provider 超时和 stale owner 收敛测试。

### 16. Wheelhouse source build 可复现性

- [x] 状态：已完成

当前确定性只覆盖 wheelhouse archive 外层。依赖为 sdist 时，pip 会在隔离环境中解析并安装额外 build requirements，相关版本和
产物字节不受现有 runtime lock/build key 完整约束；重新构建可能得到不同 wheel hash，并与仅使用 build key 的 producer identity
冲突。

落地项：

- 明确 MVP 是否支持 sdist；若只支持预编译 wheel，则使用 `--only-binary=:all:` 并对缺 wheel 明确失败。
- 若支持 sdist，锁定并验证 build requirements，明确可接受的构建网络来源和可复现边界。
- wheelhouse producer identity 和发布 CAS 按 attempt 隔离；最终 ready build 仍按 build key 复用，但不能要求不同 attempt
  必然生成相同字节。
- 增加 sdist、动态 build requirement、重复构建 hash 和上传结果不确定场景测试。

### 17. Wheelhouse 离线可安装性验证

- [x] 状态：已完成

Builder 当前生成依赖 wheel、复制用户上传的 Nexus wheel 后直接发布，未验证三份输入彼此一致，也未验证最终 wheelhouse 能按
Manager 的安装方式完整解析。构建可能先变为 ready，随后所有 Manager 在 bootstrap 阶段一起失败。

落地项：

- Builder 上传前创建临时 venv，使用与 Manager 相同的 `--no-index --find-links` 和 Nexus wheel extras 完成一次离线安装。
- 校验 Nexus wheel、source project、lockfile 和所选 extras 的基本一致性；缺失依赖或 extras 必须在 build 阶段失败。
- 安装完成后执行最小 import/metadata 校验，并将有界错误摘要持久化为 DependencyBuild 错误，供 API/UI 展示。
- 增加 wheel/source/lock 不一致、缺 wheel、无效 extra 和离线安装成功测试。

### 18. Legacy Environment Pool 与 Manager 兼容字段清理

- [x] 状态：已完成

`EnvironmentPoolRow`、`EnvironmentDemandRow`、`environment_key` 和 `workload_class` 均由本分支的 `0004` migration 引入，
`origin/main` 不包含这些结构。新版 single、live 和 batch 对 predefined/dynamic Environment 都创建 allocation；predefined
Manager 重新注册时也能使用 `environment_id`、`pool_id` 和 `workload`。本次采用离线升级且没有在途任务，不再保留这套中间版本
兼容路径。

统一后的 Manager 身份字段为：

```text
environment_id
pool_id
allocation_id          # predefined Manager 为 NULL
workload
allocation_spec_hash
owner_principal_id
```

落地项：

- 删除 `environment/pool.py`、`environment/specs.py`、`EnvironmentPoolRow`、`EnvironmentDemandRow`，以及
  `EnvironmentRepository` 中旧 pool/demand 的创建、claim、状态和统计接口。
- 删除 execution/manager record、ORM、API 和 scheduler 中的 `environment_key`；删除 Manager 的 `workload_class`，统一只使用
  `workload`。
- 删除 InstanceManager 的 `environment_type`、`environment_name` 和 `pool_name` 兼容字段；provider type 通过 Environment/allocation
  获取，展示名称通过 `EnvironmentRow.name` 获取。
- Helm 和 CLI 将 `--environment-name`、`--pool`、`--workload-class` 改为 `--environment-id`、`--pool-id`、`--workload`；
  predefined Manager 启动后使用新字段重新注册。
- 删除 bootstrap 的 `--environment-key`、`prepare_environment()` 和旧 pool 直读 ArtifactStore 路径；dynamic Manager 只保留
  allocation bootstrap。
- 删除 Scheduler/Reconciler 的 `EnvironmentDemandRow` fallback；原始管理员 `POST /v1/executions` 仍可创建无 allocation execution，
  因此 Execution 的 `allocation_id` 暂时保持 nullable，并继续使用 placement 路径。
- Environment inventory、Manager API 和 Console 改为读取 `environment_id`、`pool_id`、`workload`；动态资源状态来自
  `EnvironmentAllocationRow`，不再返回旧 `dynamic_pools`。
- 删除 `QuotaUsageRow.active_environment_pools`、`desired_manager_replicas` 及未生效的 `dynamic_environments`、
  `manager_replicas` 默认 quota；compute quota 继续以 allocation/resource usage 为准。
- 删除 Kubernetes/TKEx client 中只供旧 pool controller 使用的通用 `get/create/patch/delete` 别名。
- 直接精简尚未进入 `origin/main` 的 `0004` migration，不创建再删除 branch-only 表和字段。已执行过旧 `0004` 的非生产数据库
  重建；如必须保留，则使用一次性 cleanup migration，不把该兼容迁移带入正式链路。
- 正式升级前确认没有非终态 execution/assignment，停止所有 distributed 服务并清空 `instance_managers`；schema 升级后启动
  predefined Manager，等待全部按新字段健康上报后再开放 submit。
- 覆盖 fresh DB、从 `0003` 离线升级、predefined Manager 重报、single/live/batch 调度、inventory/API 和 schema 不含旧字段的测试。

### 19. 跨集群 Application Gateway 内网入口配置

- [x] 状态：已完成

Application Gateway 的公网和内网入口必须由部署配置明确区分。跨集群 TKEx Manager 和 dependency builder 使用内网 CLB；
Console、CLI 和上传使用公网入口。

落地项：

- 普通平台配置使用 `application_public_base_url` 和 `application_internal_base_url`；不再作为 Helm 业务 values/env 注入。
- 公网地址只用于浏览器、CLI 和客户端上传；内网地址只供 Reconciler 创建的 Manager 和 dependency builder 访问内部 Artifact
  API，两者不得混用。
- Reconciler 通过 `DistributedConfig.application_internal_base_url` 读取启动快照，并由 Manager renderer 和
  dependency builder renderer 注入 `NEXUS_DIST_APPLICATION_INTERNAL_BASE_URL`。MVP 不增加 CLB 自动发现或上报流程。
- Application Gateway Service 支持部署侧配置 TKEx 内网 LoadBalancer annotations、流量策略和已有 CLB；CLB 生命周期仍由部署
  管理，不由 EnvironmentController 动态创建。
- 当前不增加 `controlGateway.internalBaseUrl`：远端 Manager 和 builder 不访问 Control Gateway；Application Gateway 的旧 live
  compatibility 仅通过同集群 Service DNS 访问 Control Gateway，并将在兼容入口删除后消失。若未来拆分控制面集群，再独立设计。
- 若以后不同目标集群必须走不同网络入口，可在 ProviderConfig 增加可选 override；MVP 只保留一个全局稳定入口。
- 补充配置加载与 renderer 测试，验证 dynamic Manager、dependency builder 均收到相同地址，predefined 和控制面本地调用不受影响。
- 增加跨集群 TKEx E2E，从远端 Manager/Builder 校验 Application Gateway health、Artifact 下载和结果上传，保存脱敏后的
  workload 配置、目标集群和清理结果。

### 20. Environment Allocation lease 续租与事务 fencing

- [x] 状态：已完成

EnvironmentController 会一次领取多个 allocation 后串行处理，默认 lease 只有两分钟，而一个 allocation 可能串行执行多次最长
30 秒的 provider 请求。当前处理期间没有续租；lease 过期后，旧 owner 的永久错误路径还会先把 execution 写成 failed，再检查
allocation transition 是否仍持有 claim，可能覆盖新 owner 已经推进成功的结果。

落地项：

- 改为逐条领取后立即处理或使用有界并发，耗时 provider 调用前后及执行期间续租；续租失败后停止当前 owner 的所有写入。
- allocation observation、状态迁移、provider resource ref 和错误记录全部使用 `operation_id`/lease CAS。
- execution 失败与 allocation 的 fenced 状态迁移放入同一事务；只有仍持有有效 claim 的 provisioning owner 才能写
  `EnvironmentProvisioningFailed`。
- 覆盖批量 claim 排队过期、慢 provider、续租丢失、双 Reconciler 接管和旧 owner 延迟失败测试。

### 21. Environment 异常清理按状态阶段收敛

- [x] 状态：已完成

当前 `AllocationSpecConflictError` 和 `WorkloadPermanentError` 都进入 `_fail_provisioning()`，统一尝试把 allocation 转为
`DRAINING`。这对已经处于 `DELETING` 的 allocation 是非法转换；provider ref 或配置缺失时，进入 `DRAINING` 后又可能无法构造
allocator，导致资源、allocation 和 quota 长期无法释放。

落地项：

- 按 `REQUESTED/PROVISIONING/READY/DRAINING/DELETING` 分别处理永久错误，不再使用统一的 provisioning 失败出口。
- 尚未创建 provider 资源时可 fenced 地直接进入 `RELEASED` 并释放 quota；已经创建资源时必须保存足够的 provider snapshot，
  再进入 drain/delete。
- `DELETING` 阶段的错误保留在 `DELETING` 重试，不回退到 `DRAINING`；配置缺失和资源已不存在都要有明确、幂等的收敛路径。
- cleanup 错误不得错误改写已经终态的 execution；增加各状态永久错误、provider ref 丢失、资源不存在和强制清理测试。

### 22. TKEx API 响应 envelope 兼容

- [x] 状态：已完成

TKEx client 当前在 `code=0` 后直接返回完整 JSON payload，而 TKEx 文档中的成功响应可能是
`{"code": 0, "data": <Kubernetes resource>}`。Allocator 按顶层 `metadata/status` 读取，因此真实接口返回 `data` 包装时，
创建资源可以成功，但后续 observe 可能把有效资源当成无状态对象。现有 fake client 测试只覆盖裸 Kubernetes 对象。

落地项：

- 在 TKEx transport 边界统一规范化成功响应，同时兼容裸 Kubernetes 对象和 `code/data` envelope；业务层只接收规范化对象。
- 对 `data` 缺失、类型错误、业务错误码和 HTTP 错误分别建模，不能把异常响应降级为空资源。
- 使用真实文档响应 fixture 覆盖 StatefulSetPlus、Job 和 Secret 的 get/create/delete；保留裸对象测试用于兼容已有代理。
- 增加跨集群 TKEx smoke，核对目标 cluster header、namespace、资源 UID 和 ready/status 读取结果。

### 23. Environment provider 配置准入与 PATCH 语义

- [x] 状态：已完成

Environment 创建和更新目前只检查 dynamic 的 `provider_config_ref` 非空，不确认 ref 是否存在、provider type 是否匹配；
拼写错误会到 allocation 执行阶段才失败。`UpdateEnvironmentPayload` 与 repository 又用 `None` 同时表示“未提交字段”和
“显式清空”，导致已有 `provider_config_ref` 无法通过 PATCH 清除。

落地项：

- 在管理 API 创建、更新和启用 Environment 时，对 DB ProviderConfig registry 做 fail-close 校验，包括 ref 存在、provider type
  匹配和 workload namespace 一致。
- predefined Environment 必须没有 `provider_config_ref`；dynamic Environment 必须引用与自身 provider type 一致的 entry。
- PATCH 使用 `model_fields_set` 或显式 unset sentinel 区分未提交与 `null`，repository 同样保留该语义，允许合法字段被清空。
- ProviderConfig 变更后，已有 Environment 再次启用或创建 allocation 时重新校验；错误在提交前返回明确的管理员可读信息。
- 增加 ref 拼错、类型不匹配、namespace 不匹配、显式清空、部分更新和并发 config version 更新测试。

### 24. Environment limits 与 timeout 强类型配置

- [x] 状态：已完成

`resource_limits` 和 `default_timeouts` 当前是任意字典，只校验部分值是否为正数，不限制 key。管理员写错字段名时配置会被静默
保存，但 submission 和 controller 不会读取它，容易产生“限制已经生效”的错误判断。

落地项：

- 为 Environment resource limits 和 default timeouts 定义明确的 Pydantic schema，使用 `extra='forbid'`，统一字段名、单位、
  数值范围和可空语义。
- 创建、PATCH、token grant 和 submission 合并配置时使用同一套模型与交叉约束；数据库仍可保存 JSON，但只能写入规范化结果。
- Console 使用结构化表单或经 schema 校验的 JSON 编辑器，API/OpenAPI 返回具体未知字段和范围错误。
- 对升级前已有配置执行校验或一次性规范化；增加未知 key、错误类型、上下限冲突、默认值覆盖和向后升级测试。

### 25. Gateway HTTP 路由按领域拆分

- [x] 状态：已完成

Control Gateway 与 Application Gateway 已在进程和职责上分开，但各自的 HTTP payload、handler、路由注册和辅助逻辑仍集中在
`gateways/control.py` 与 `gateways/application.py`。Control 同时包含 submission、Artifact metadata、Access、Environment、
Execution、Batch 和运维接口；Application 同时包含 Artifact 正文、result、SSE、command 与 runtime proxy。HTTP/WebSocket
proxy、异常映射和 AppKey 又存在重复或分散实现，使权限、请求上限和错误语义容易在不同入口产生偏差。

落地项：

- 保留 `gateways/control.py` 和 `gateways/application.py` 作为稳定公开入口，继续导出 `create_control_app()` 和
  `create_application_app()`；CLI、Helm、URL 和外部 import 不变。
- 新增 `control_routes`，按 `submissions`、`artifacts`、`access`、`environments`、`executions` 和 `operations` 组织；
  新增 `application_routes`，按 `artifacts`、`executions` 和 `proxy` 组织。每个模块只通过 `register_*_routes(app)` 注册原有路由。
- 将共享 AppKey、typed API error/middleware 和完整的 HTTP/stream/WebSocket proxy helper 分别收敛到 `_keys.py`、
  `_errors.py` 和 `_proxy.py`；Environment-aware execution authorization 只保留一个实现，供两个 Gateway 复用。
- 不使用 aiohttp sub-application 做领域拆分，不增加 controller/service locator/DI 框架或新的部署服务；handler 继续通过
  `request.app[AppKey]` 获取既有依赖。
- 先完成 Gateway 权限、请求资源上限等行为修复，再用一次纯机械改动移动代码；拆分提交不得同时改变 API、middleware 顺序、
  状态码、响应结构或 cleanup context 生命周期。
- 更新当前直接 import Gateway 私有函数的单测，使其从真实所属模块导入；运行 Gateway/Auth 全量单测、Ruff 和现有集成测试，
  并核对所有注册路由在拆分前后完全一致。

### 26. Environment Grant 边界与实时生效

- [x] 状态：已完成

用户 execution 查询和操作必须同时按 principal owner、Environment 和当前 enabled Grant 过滤，不能只依赖请求 scope。

落地项：

- 提供唯一的 Environment-aware execution authorization，统一校验 scope、owner、execution Environment 和当前 Grant；
  Control/Application Gateway 全部复用。
- execution list、batch、allocation、control event、stop、command、result、SSE 和 runtime proxy 都必须应用相同边界；
  list 查询直接增加 Environment filter，不能先返回 owner 下的全部 execution 再由调用方筛选。
- 鉴权时 fail-close 校验 Environment enabled 和 Grant enabled；停用只拒绝后续 API 请求，不自动停止已有 execution。
- 覆盖跨 Environment 的 list/get/stop/command/result/SSE/proxy、Grant disabled 和 Environment disabled
  测试，并验证越权资源继续返回不可枚举的错误。

### 27. Application Gateway 路由级请求体与结果内存上限

- [x] 状态：已完成

Application Gateway 当前把通用 Artifact 最大值作为 aiohttp 全局 `client_max_size`，默认可达 1 GiB。Artifact 上传本身使用
流式读取，但 runtime HTTP proxy 会调用 `request.read()`，使普通已鉴权请求也能将大正文整体载入内存。
终态 `run_result` 读取还会将完整文件读入内存，其下载并发与独立 size limit 已在第 8 项处理。

落地项：

- 为普通 JSON、command 和 proxy 路由设置较小的默认请求体上限；Artifact 上传保留按 ticket、kind 和 Environment limit
  校验的流式路径，不能依赖全局 1 GiB 上限。
- runtime HTTP proxy 改为流式转发请求体，或设置独立且明确的小上限。
- `run_result` 使用独立且明显更小的 size limit，并与第 8 项的下载 admission、读取并发和 spool 上限保持一致。
- 增加超限 JSON、超限 proxy、chunked upload、并发大请求和 result 读取过载测试，确认 Gateway 不出现与请求体线性增长的
  内存峰值。

### 28. Artifact 归档校验边界

- [x] 状态：已完成

Application Gateway 不解析 source、wheel 或 wheelhouse，只做认证、限流以及正文 size/sha256 校验。归档语义校验放在已有资源
边界内执行，避免 Gateway event loop 承担大文件 CPU 和解压工作。

落地项：

- Builder 唯一一次有界检查 Nexus wheel，并复用解析出的版本、tag 和 extras；无效 wheel 为永久 build 失败。
- Bootstrap 流式安全解包 `source.tar.zst` 和 wheelhouse，限制路径、成员类型、数量、大小和 zstd window，不计算逐文件 hash。
- 删除独立 archive validation executor、队列、deadline 和配置；上传 claim 失败时仍安全释放 spool 与 receiving lease。
- 覆盖压缩炸弹、危险路径、symlink/hardlink、正文身份不一致和并发 Gateway 请求测试。

### 29. Submission 返回 URL 契约

- [x] 状态：已完成

typed submission 已删除抽象的 `application_url`，改为返回实际存在的 `control_url`、`result_url`、Web UI URL，live
额外返回 `events_url`、`command_url`，batch 额外返回 `batch_url`。公网 URL 使用部署配置，不依赖代理请求 Host。

落地项：

- 明确 `application_url` 的单一语义：要么实现稳定的 execution Application 入口，要么改为返回实际存在的
  `result_url`、`events_url`、`command_url` 等具体链接，不保留不可跟随的概念 URL。
- single、live 和 batch 分别定义哪些链接可用及其终态语义；未就绪时返回稳定状态，不通过 404 混淆“路由不存在”和
  “结果尚未产生”。
- CLI、Console、设计文档和 API 响应模型使用同一字段定义；增加提交后逐个访问返回 URL 的 HTTP E2E，不再只断言字符串后缀。

### 30. Gateway typed error contract

- [x] 状态：已完成

Control Gateway 当前把任意 `KeyError` 映射为 404、任意 `ValueError` 映射为 409；Application Gateway 也按
`ValueError` 文本选择状态码。Repository、序列化或 handler 中的编程错误因此可能被伪装成业务冲突且不记录异常，错误码和
响应结构也容易在两个 Gateway 间漂移。

落地项：

- 定义共享的 typed Gateway/domain exception，明确稳定 `code`、HTTP status、可公开 detail 和是否可重试；middleware 只映射
  已知异常。
- Repository 的 not-found、conflict、fencing、validation 和 temporary-unavailable 使用明确异常类型，不再依赖异常文本匹配。
- 未知 `KeyError`、`ValueError` 和其他异常统一记录上下文后返回脱敏 500；不能把编程错误降级为 404/409。
- Control/Application Gateway 使用一致的 JSON error envelope；补充已知错误状态码、未知异常 500、日志记录和敏感信息不外泄测试。

### 31. Legacy Live Compatibility 统一鉴权

- [x] 状态：已完成

`/compat/live/*` 作为显式开启的旧协议适配层保留，typed `nexus submit live` 仍是新调用主入口。

- 除 `/compat/live/health` 外，父 Application Gateway 统一校验个人 `x-tai-identity` 或应用 x-rio 身份；匿名、无效或混合身份不能
  进入兼容业务路由。
- Compatibility 过滤所有公网和调用方提供的内部身份头，再使用自身 internal credential 将已验证的 principal 或 `app_id` 委托给
  内部 Control/Application Gateway；不转发原始 JWE、x-rio 签名、`Authorization`、cookie 或调用方 internal token。
- 不恢复专用 compatibility submit token。execution 按真实用户应用 owner、Environment Grant、Quota、可见性和审计。
- 默认 placement 由部署参数固定。Helm 默认关闭，通过 `liveCompatibility.enabled=true` 显式开启；readiness 继续使用
  `/v1/health`。
- 单测覆盖身份转发、敏感头过滤、SSE/HTTP/WebSocket、旧请求映射和用户隔离；predefined 集成脚本覆盖完整旧 live 流程。

### 32. Submission 输入准入与 Driver 异常收敛

- [x] 状态：已完成

优先级：P1

typed single/live submission 目前只要求 `run_input` 和 `live` 为字典，batch task-list 也只校验 JSON 外形。非法配置可能在
Driver 启动后、异常保护前解析失败；Executor 会清理本地 task，但 InstanceManager 仍将 assignment 作为 running 上报，
使 execution 长期不收敛。

落地项：

- 抽出纯输入规范化逻辑，在 single/live submission admission 和 batch task-list materialize 阶段校验并保存规范化配置；
  不能在 Gateway 解析工作节点环境变量。
- Driver 的请求构造纳入统一异常保护；Executor 未捕获异常必须通知 InstanceManager，将 assignment/execution 收敛为
  明确失败并释放本地运行记录。
- 覆盖非法 single、live、task-list item，以及无 execution TTL 时的失败收敛测试。

### 33. Submission 资源数值边界与统一换算

- [x] 状态：已完成

优先级：P1

CPU、内存、Manager 数、slot、timeout 和 batch 数量字段缺少 finite、粒度及数据库范围约束。极小正数可能在 quota 中换算为
零、在 provider 侧又被抬高；无穷大或超大整数可能产生 500 或数据库溢出。

落地项：

- 抽出纯资源换算逻辑，拒绝 NaN/Infinity，明确 CPU millicore 和 memory byte 的最小粒度，并限制 Integer/BigInteger
  及 provider 可表达范围。
- 在乘法前校验 `manager_count * per_manager_resource/slots`，禁止依赖数据库或 provider 截断、抬高和溢出处理。
- allocation、quota、provider spec 使用同一份规范化资源结果；增加极小值、边界值、无穷大和乘法溢出测试。

### 34. Queued execution quota 接入 submission admission

- [x] 状态：已完成

优先级：P1

`queued_executions` 配置和 `QuotaRepository.admit_submission()` 已存在，但 typed submission 没有调用，当前限制不生效。

落地项：

- 明确 queued quota 统计 top-level 与 batch 活跃 child 的口径；batch 仍由 `max_inflight` 控制同时存在的 child execution。
- 非幂等复用的 top-level submission 在同一数据库事务内执行 admission；冲突、校验失败和事务回滚不得消耗额度。
- 保持 owner quota、Environment 和 Grant 的固定加锁顺序；覆盖并发提交、幂等重放、终态/过期任务及 batch child 测试。

### 35. Submission aggregate 与 DependencyBuild 并发幂等

- [x] 状态：已完成

优先级：P2

Execution 的内层幂等查询可能返回并发 winner，但 SubmissionRepository 仍会使用本地新 ID 创建后续聚合记录。
DependencyBuild 捕获唯一键冲突后使用普通 SELECT 读取 winner，在 MySQL Repeatable Read 下也可能不可见。

落地项：

- SubmissionRepository 作为 execution/allocation/batch 聚合的唯一幂等边界；并发唯一键竞争统一回滚并重跑完整事务，不能继续
  使用 loser 的本地 ID。
- DependencyBuild 使用可靠的 upsert/current read，或将其唯一键冲突纳入同一整事务重试；禁止提交半套聚合记录。
- 增加真实 MySQL 并发测试，核对唯一 execution/build、allocation 数、Artifact ref、quota 和 audit 均无重复或泄漏。

### 36. Submission phase 统一投影

- [x] 状态：已完成

优先级：P2

当前 submission 重放默认返回 `waiting_resources`，只覆盖部分 batch 状态；terminal single/live、dependency build 和 result
upload 等状态会得到错误 phase。

落地项：

- 抽出唯一纯 phase projector，从 execution、allocation、dependency build、batch control/generation 和 result Artifact
  的持久化状态推导设计文档中的 phase，不增加第二套状态机。
- 新建响应、幂等重放、execution 查询和管理页面复用同一投影，未知或矛盾状态 fail close 并记录诊断信息。
- 覆盖 predefined/dynamic、build pending/failed、batch generation/finalization、result upload 和全部终态。

### 37. Batch generation 提前准入与完整 deadline

- [x] 状态：已完成

优先级：P2

local task-list 已知的 item/byte 数和 remote compose 显式 limit 未在 submission 时与 Environment 限制比较，可能在资源申请后
才失败。`generation.timeout_seconds` 也只约束 remote 生成，不覆盖 local task-list 下载和分块入库。

落地项：

- admission 阶段拒绝 manifest `item_count/size_bytes` 或 compose limit 已明确超限的请求；运行时仍校验实际内容，不能信任
  manifest。
- generation deadline 覆盖下载、生成、Artifact 发布和分块 materialize 全阶段，并在每次持久化前执行 parent attempt
  fencing；超时后可安全重试或收敛失败。
- 覆盖已知超限、伪造 manifest、local/remote 超时、分块中断和 parent retry 恢复。

### 38. CLI `--wait` 保留 accepted 结果

- [x] 状态：已完成

优先级：P2

CLI 当前等到终态后才输出 submission 响应；提交已成功后若轮询瞬时失败或用户 Ctrl-C，execution ID 不可见，容易重复提交。

落地项：

- `202 Accepted` 后立即保留并向用户展示 execution/batch ID；等待失败或中断时仍返回可继续查询的标识。
- GET 轮询对网络错误和可重试状态码使用有界退避；Ctrl-C 只停止本地等待，不向远端发送 stop。
- 覆盖提交成功后断网、Gateway 5xx、Ctrl-C、终态返回和 JSON 输出契约。

### 39. Distributed `run_id` 契约统一

- [x] 状态：已完成

优先级：P2

设计要求同一 execution 的 `run_id=execution_id` 并在 infra retry 间保持稳定，但 single/live 会保留用户值，batch child 又使用
`item_id`，会改变外部副作用的幂等语义。

落地项：

- 按现有设计由 distributed Driver 强制使用 `execution_id` 作为 `run_id`；batch `item_id` 和 logical retry identity 使用独立
  字段保存，不能复用 `run_id`。
- single、live、batch parent/child、infra retry 和 batch logical retry 使用同一契约，并同步文档、ExecutionContext 和结果
  元数据测试。

### 40. Submission 与 Artifact 字段长度契约对齐

- [x] 状态：已完成

优先级：P3

合法的 128 字符 submission idempotency key 追加 Artifact kind 后会超过 upload API 的 128 字符限制；submission `pool`
允许 128 字符，而 execution/allocation 数据库列只允许 64 字符。

落地项：

- Artifact upload idempotency key 使用由 submission key、kind 和内容身份生成的有界稳定 key，不直接追加可变长后缀。
- API schema、repository、数据库和 provider 对 Environment/pool/identity 字段使用一致长度；在进入数据库前返回 typed
  validation error。
- 增加最大合法长度、超长拒绝、不同 Artifact kind 不碰撞和重试复用测试。

### 41. Reconciler 状态转换 locking 与 fencing

- [x] 状态：已完成

优先级：P1

Manager 使用 `Execution -> Assignment` 的固定顺序锁定并重新读取当前状态，Reconciler 的失联、session mismatch、ack/start
timeout、TTL stop 和 terminal cleanup 当前主要通过普通查询取得 ORM 对象后直接修改。Manager 完成任务或更新 assignment
期间，Reconciler 可能基于旧快照覆盖已经提交的新状态；已有 `active execution + non-active current assignment` 也没有统一
修复路径。

落地项：

- 候选扫描只读取 ID；实际处理按 `Execution -> Assignment` 的统一顺序执行 `FOR UPDATE SKIP LOCKED`，使用
  `populate_existing` 重新加载并在锁内复核 execution status、current assignment、assignment status 和 attempt。
- manager dead 与 session mismatch 在状态转换前重新校验 Manager session 和 heartbeat；使用行锁或 CAS fencing，不能依据
  候选扫描阶段的旧心跳直接判定失联。
- 增加 active execution 与缺失或非 active current assignment 的收敛流程；未实际启动且仍有 retry budget 时重新排队，
  已启动或不可重试时收敛为 `lost`，同时关闭 live route、取消旧 command 并释放 demand/quota。
- 所有状态转换和 control event 在同一事务内完成，冲突或前置条件变化时跳过，不写入基于旧状态的事件。
- 增加 Manager success/failure/heartbeat/session replace 与各 Reconciler repair 并发交错、双 Reconciler 抢占和重复执行测试。

### 42. Reconciler absolute deadline 与到期扫描

- [x] 状态：已完成

优先级：P1

queued expiry、execution TTL 和 stop timeout 当前先按 created/started/updated 时间取固定数量，再在 Python 中计算每条记录的
deadline。较老但长 TTL 或尚不可调度的记录可以长期占满扫描窗口，阻塞后面已经到期的任务；使用 `updated_at` 计算 stop
timeout 还会让无关更新意外延长停止期限。

落地项：

- 持久化 `queue_expires_at`、`execution_expires_at` 和 `stop_deadline_at`；predefined queue deadline 在提交时确定，
  dynamic execution 在首次 schedulable 时确定，execution/start 和 stop deadline 分别在对应状态转换时一次性确定。
- 使用 `(status, deadline)` 索引直接查询 `deadline <= now`，按 deadline 排序并通过 `FOR UPDATE SKIP LOCKED` 有界领取。
- deadline 一旦建立，不得被 heartbeat、command、event 或其他 `updated_at` 更新隐式延长；用户显式修改 timeout 不影响已经
  启动的 attempt，除非后续定义单独的管理操作。
- 为升级前记录提供明确的空值和惰性回填语义，避免部署升级后立即错误过期。
- 增加长短 TTL 混排、超过 batch size、dynamic allocation 长期 pending、stop 期间持续更新和多 Reconciler 并发测试。

### 43. Scheduler shard lease 提交边界与硬 fencing

- [x] 状态：已完成

优先级：P2

Scheduler 当前在 shard lease 事务提交前更新 `_owned_shards`、renew 时间和 discovery 状态。提交失败后，本地缓存可能继续把
未成功取得或续租的 shard 当作 owned；assignment 事务又只接收 shard IDs，没有重新验证 lease owner 和 expiry。

落地项：

- lease repository 只返回本次事务的候选结果；`session.commit()` 成功后再原子发布 `_owned_shards`、renew 时间、discovery
  cursor 和下一次 discovery 时间。
- commit 失败或结果不确定时不得推进本地时间和 cursor；清空相关缓存并在下一 tick 强制从数据库重新发现。
- 将 shard lease 作为 Scheduler 的硬 fencing：claim transaction 按 `shard_group + owner_id + lease_expires_at` 重新验证
  ownership，失去租约的 Scheduler 不得继续领取新 execution。
- 保留 execution 与 Manager 行锁作为最终并发保护，lease fencing 只约束新领取，不中断已经提交的 assignment。
- 增加 lease insert/renew commit deadlock、连接中断、unknown commit outcome、租约到期转移和两个 Scheduler 并发领取测试。

### 44. Batch lifecycle parent claim 与持久化游标

- [x] 状态：已完成

优先级：P2

每个 Reconciler 副本当前都会选择同一批 active/stopping batch parent，随后逐个处理；child sync cursor 仅保存在进程内。
单副本能够收敛，但多副本会重复扫描和争锁，进程重启后也会从头扫描。万级 child 下，完整扫描周期还会直接决定终态发现延迟。

落地项：

- 为 batch parent 增加带 owner、operation id 和 expiry 的 reconcile claim lease；每个 parent 同时只由一个 Reconciler 推进，
  长分页处理期间续租并使用 fencing 防止旧 owner 回写。
- 将 child sync cursor 持久化到 `BatchRunRow`，与 parent claim 在同一事务中更新；完成一轮后显式回到起点，重启和 owner
  转移继续沿用已提交进度。
- 每个 tick 对 parent 数、单 parent page 数和总扫描 item 数设置独立预算，避免大量 active batch 让一次 loop 长时间不返回。
- MVP 保留基于 status/version 的分页扫描；记录 full-scan duration、terminal discovery lag 和 scanned/changed ratio，
  达不到万级 batch 延迟目标时再引入 child-state dirty queue，暂不建立第二套状态机。
- 增加双 Reconciler、claim 过期接管、处理中崩溃、游标恢复、万级 active child 和 mass stop/fail-fast 测试。

### 45. Scheduler fair claim 查询收敛

- [x] 状态：已完成

优先级：P2

公平领取当前先查询 owner，再为每个 owner 单独执行一条带锁查询。默认 `scan_limit=250` 时，一个 tick 最坏产生超过 250 条候选
查询，并在后续 owner 查询期间持续持有已经领取的 execution 锁，增加事务时间和死锁概率。

落地项：

- 使用一次 ranked candidate 查询计算 owner 内顺序和跨 owner round-robin 次序，再按候选 ID 执行一次
  `FOR UPDATE SKIP LOCKED` 并重新校验 status、schedule time、allocation readiness 和 shard ownership。
- 保持当前公平语义：先覆盖更多 owner，再领取同一 owner 的后续任务；不能因批量化退化为单个大用户占满一个 tick。
- 对候选查询数、锁等待、tick transaction duration、claimed/assigned 数和 no-capacity 数增加指标。
- 使用真实 MySQL 执行计划和多 Scheduler 并发测试验证索引命中、锁顺序、公平性及无重复 assignment。

### 46. Scheduler/Reconciler 配置约束与运行可观测性

- [x] 状态：已完成

优先级：P3

部分控制面 timeout、TTL、poll interval 和 batch size 仍是无范围约束的基础类型；错误配置可能导致停止收敛、忙循环或 shard
lease 在续租前过期。Scheduler/Reconciler 也缺少足够指标定位 loop 卡顿、扫描饿死和 lease 抖动。

落地项：

- 为 poll interval、TTL、timeout、batch size、scan/lock limit 和 jitter 增加明确的 Pydantic 正值及范围约束。
- 增加交叉校验：lease renew interval 小于 lease TTL，heartbeat interval 小于 healthy TTL，healthy TTL 小于 dead TTL，
  jitter 位于允许范围，所有 reconcile batch size 至少为 1；非法配置在服务启动时 fail close。
- 分 loop 采集 tick duration、loop lag、success/error/backoff、scanned/claimed/changed 数；Scheduler 额外采集 owned shard、
  assignment 和 lock contention，Reconciler 额外采集 overdue lag、repair reason 和 batch full-scan duration。
- 日志携带 loop、owner、shard/parent、operation id 和 retry reason，避免高频逐任务日志。
- 增加配置边界、交叉约束、指标标签稳定性和故障注入测试。

### 47. BatchControl task-list 热路径与发布契约

- [x] 状态：已完成

优先级：P1

Batch Driver 启动时已经执行一次 `prepare()`，但后续每次填窗仍再次调用 `prepare()`；缓存命中时会重新读取并计算完整
task-list SHA256。接近 1 GiB 上限的 task-list 会在高频轮询期间持续占用磁盘带宽和 Manager event loop。Application Gateway
已经在 task-list 上传完成的事务中原子写入 Artifact ready、ArtifactRef 和 BatchRun 链接，BatchControl 随后再次调用
`set_generated_task_list()` 也是冗余事务。

落地项：

- 将 BatchControl 明确拆成一次性 `prepare/ensure_prepared` 和高频 `fill_window`；后者只读取已验证的本地 path/ref，不下载、
  materialize 或重新计算完整文件 hash。
- 按 `parent_execution_id + generation_epoch + ArtifactRef identity` 缓存 prepared state；文件缺失、identity 改变或 parent
  release 时失效。完整 SHA256 只在生成或下载完成时校验一次，不能依赖未经校验的 manifest。
- 将 `publish_task_list()` 契约固定为“返回前已在 Gateway 事务中完成 ready 和 BatchRun 链接”，删除 BatchControl 的重复链接
  事务；上传响应丢失时仍以同一 producer key 幂等重试。
- 增加大文件多轮填窗、缓存失效、Manager 重启恢复、上传成功但响应丢失和旧 parent attempt fencing 测试，并记录轮询期间
  task-list 读取字节数。

### 48. Batch ledger 热查询与 counter 收敛

- [x] 状态：已完成

优先级：P1

`finalize_parent_if_ready()` 当前在每个 lifecycle tick 对完整 ledger 执行非终态 `COUNT`，Batch 查询接口也在每次请求时 JOIN
全部 child 并 `GROUP BY` 状态；这与设计中的“运行期间使用 counter，仅在最终收敛时精确汇总”不一致。填窗查询使用
`status='pending' ORDER BY task_ordinal`，现有索引也没有同时覆盖 status 和 task ordinal，万级任务持续补窗时可能反复排序或
扫描已处理前缀。

落地项：

- active batch 先用 `terminal_items/materialized_count/total_items` 判断是否可能完成；只有 counter 到达终态边界时执行一次精确
  `COUNT/GROUP BY`，校验并修复 counter 漂移。
- 默认 Batch summary API 只返回持久化 counter 和最终 summary，不在 UI 轮询路径持续聚合全部 child；详细状态分布如确有需要，
  使用显式诊断参数或独立低频接口。
- `has_exhausted_failure()` 改为带索引的 `EXISTS/LIMIT 1`；为填窗增加
  `(parent_execution_id, status, task_ordinal)` 复合索引，并核对 stop、active sync 和 item cursor 查询的执行计划。
- 使用真实 MySQL 对 2 万和 10 万 item 跑完整 materialize、持续填窗、child 收敛、fail-fast、stop 和 finalize；断言运行期间
  不出现随 total item 数增长的全量扫描。

### 49. Cloud Batch generator metadata 契约

- [x] 状态：已完成

优先级：P1

现有 `BatchItem` 包含 item metadata，老版 Batch 会持久化该字段并可用于 grouping、tagging 和 conflict groups；
`GeneratorLoader.get_batch_metadata()` 也提供 batch 级 metadata。Cloud Batch 必须显式区分不携带 metadata 的 v1 和携带
item metadata 的 v2，不能静默丢弃。

落地项：

- task-list v1 只支持 `item_id + run_input`，遇到非空 item metadata 时 fail close。
- task-list v2 增加最大 64 KiB 的 JSON object item metadata，并纳入 canonical record、task-list hash、非确定性重试检查和
  ledger；generator、compose 和 Dataset 路径使用同一规则。
- v1/v2 均不支持 batch metadata；遇到非空值时 fail close。Conflict groups 仍未定义跨 Manager 调度语义，不对外宣称支持。
- 覆盖 item/batch/Dataset metadata、本地与云端一致性、manifest version fencing 和 generation retry 测试。

### 50. Batch item ID 与数据库排序语义统一

- [x] 状态：已完成

优先级：P2

`is_valid_batch_item_id()` 当前允许大小写、尾部空格和任意可打印 Unicode，而生产 MySQL 8 默认 VARCHAR collation 通常不区分
大小写、重音并忽略尾部空格。Python duplicate set、MySQL 主键/唯一键以及基于 `item_id > cursor` 的分页可能因此使用不同的
相等和排序语义，表现为物化中途唯一键冲突或游标异常。

落地项：

- 定义唯一 canonical item ID 契约；MVP 优先限制为有界的小写 ASCII 安全字符集并禁止首尾空白，不做隐式大小写或 Unicode
  归一化。
- generator writer、local task-list parser、ledger 和 API cursor 复用同一 validator；数据库相关 ID 列使用与契约一致的 binary
  collation 或等价字节序语义。
- 使用真实 MySQL 覆盖 `A/a`、尾部空格、重音字符、Unicode normalization、最大长度和分页边界，并验证 SQLite 单测不会掩盖
  生产 collation 差异。

### 51. Generator subprocess 有界协议与进程树回收

- [x] 状态：已完成

优先级：P2

云端 generator 目前通过 `communicate()` 将 stdout/stderr 全量缓存在 Batch Manager 内存中，只在 timeout 或 cancellation 时
终止进程组。高日志量 generator 可使 Manager OOM；generator 主进程正常退出但留下已关闭标准流的后台进程时，残留进程仍会
继续占用 allocation 资源。依赖 stdout 最后一行传递结果也容易被用户日志或退出钩子破坏。

落地项：

- 将 materialize 结果写入独立的小型 result file，并校验固定 schema；stdout/stderr 流式写入有上限的 spool 文件或截断日志，
  不在内存中无界聚合。
- 在 `finally` 中处理成功、失败、timeout 和 cancellation 的完整进程组回收，随后清理临时输出；保留 source-package 执行隔离
  作为第 4 项的独立后续工作。
- 增加超量 stdout/stderr、退出钩子输出、后台 child、非零退出、timeout、cancellation、无效 result file 和重复执行测试。

### 52. Batch lifecycle 模块归位与命名

- [x] 状态：已完成

优先级：P3

`RepositoryBatchLifecycleHook` 当前位于通用 `reconciler.py`，但它实际拥有 Batch parent 扫描、child 状态同步、logical retry、
fail-fast、stop fan-out、parent finalization、result Artifact 发布、分页 cursor 和事务重试，是完整的 Batch lifecycle
协调器，不是简单 hook，也不只是 Repository 适配层。通用 Reconciler 因此同时承载服务组装和 Batch 领域实现，职责不清。

落地项：

- 新建 `nexus/distributed/batch/lifecycle.py`，将 Batch lifecycle protocol、具体实现、Batch 专用 retry 常量及辅助函数移入该模块。
- 将 protocol 命名为 `BatchLifecycle`，具体实现命名为 `BatchLifecycleReconciler`；保留 protocol 供依赖注入和单测替身使用。
- `nexus/distributed/reconciler.py` 只负责构造组件、启动独立 loop/task 和通用 execution repair；继续在同一进程运行
  `dist-reconciler:batch`，本期不增加独立服务或部署单元。
- `repositories/batch.py` 保持纯 DB 查询和状态转换；`instance_manager/batch_control.py` 保持 task-list 生成、ledger materialize
  和 child 填窗，不能把跨事务/Artifact 编排下沉到 Repository。
- 为 `BatchLifecycle` 增加统一 `close()` 生命周期并由 `ReconcilerService` 调用；具体 Artifact client 的复用和关闭按第 11 项处理。
- 更新导入和测试位置，覆盖 Reconciler 组装、batch loop 独立运行、异常传播、stop/close，确保此次调整不改变状态机行为。

### 53. Record 映射统一为单一完整来源

- [x] 状态：已完成

优先级：P2

`repositories/_records.py` 当前只映射部分 ORM 字段，`ExecutionRepository`、`ManagerRepository` 和
`SchedulerRepository` 再分别修改返回的可变 Record 补齐字段。Record 新增可选字段后，遗漏某条补齐路径不会触发类型错误，
可能在不同查询接口中返回语义不一致的对象。

落地项：

- 由 `repositories/_records.py` 提供唯一、完整的 ORM 到 Record 映射，覆盖 ownership、Environment、allocation、workload、
  result Artifact 和 capability snapshot 等字段；Repository 不再修改 mapper 返回值。
- 对 JSON、list 和 dict 字段进行明确复制，避免 ORM mutable value 与跨事务 Record 共享引用。
- 能作为查询快照的 Record 优先改为不可变对象；确实需要分阶段构造时使用独立 builder 或显式 `replace()`，不依赖任意赋值。
- 为每个 mapper 增加字段完整性测试，并覆盖同一 ORM row 经不同 Repository 路径读取时结果一致；新增 ORM/Record 字段时测试必须
  显式失败。

### 54. Environment provider registry 与 client factory 统一

- [x] 状态：已完成

优先级：P2

`EnvironmentController` 和 `DependencyBuildController` 当前分别解析 provider config、创建并缓存 Kubernetes/TKEx client，
两边的 fallback、namespace 和 provider type 校验并不完全一致。同时 Environment 层直接持有 DependencyBuildController，
DependencyBuild 又依赖 Environment provider client，形成包级职责互穿。

落地项：

- 在 Reconciler composition root 构造共享的 `WorkloadClientResolver`，统一负责 allocation 配置快照解析、
  `credential_ref` 解析、namespace/provider type 校验和 client 生命周期。
- `EnvironmentController` 与独立的 DependencyBuild reconciler 仅依赖窄的 provider client Protocol，不互相构造或持有对方；
  loop 拆分继续按第 10 项处理。
- 将共享 provider port、registry 和命名规则放在中立模块；Kubernetes/TKEx adapter 分开实现，禁止把某个调用方的业务状态机放入
  transport client。
- provider ref 准入继续按第 23 项 fail close；两个流程必须复用相同校验结果，不保留不同的 legacy fallback。
- 在完成第 18 项旧链清理后，将 `environment/controller.py` 按 ports、rendering、allocation reconcile 和 providers 拆分，
  但不增加无实际多态价值的 Controller 基类。

### 55. Application Gateway Artifact transport 收口

- [x] 状态：已完成

优先级：P2

通用 Artifact 上传下载能力目前分别出现在 `artifacts/run_result.py`、InstanceManager bootstrap 和 DependencyBuild worker，
重复实现 content ticket、session、scope header、临时文件、流式 I/O、checksum 与错误映射。`run_result.py` 因此同时承担结果
格式、Trace Sink/Source 和通用 HTTP transport，模块边界不清。

落地项：

- 新建 `artifacts/gateway.py`，提供一个服务级共享的底层 `ApplicationGatewayArtifactTransport`，统一连接池、credential
  读取、scope header、流式传输、原子落盘、内容身份校验和 typed HTTP error。
- Run result、task list、source package 和 wheelhouse 各自保留窄的 publisher/reader Protocol 或 facade；不要公开一个包含
  所有 Artifact 操作的巨型 client。
- `artifacts/run_result.py` 只保留 immutable run-result codec、`ArtifactTraceSink` 和 `ArtifactTraceSource`；调用方不得自行
  复制 transport 细节。
- transport 的所有权和 `close()` 按第 11 项统一，Store 网络异常按第 12 项处理，上传下载 admission 分别按第 8、9 项处理。
- 将重复的文件 size/SHA256 计算归入 Artifact 领域辅助模块，并覆盖连接复用、credential 轮转、取消、校验失败及临时文件清理。

### 56. Batch Driver 隔离数据库事务重试

- [x] 状态：已完成

优先级：P2

`NexusBatchDriver` 通过 `BatchControl` 抽象调用 Batch 协调能力，却直接识别 SQLAlchemy/MySQL `OperationalError` 并执行事务重试。
这使执行 Driver 知道当前 adapter 的数据库实现，未来改成本地、HTTP 或其他 BatchControl 实现时仍被迫模拟数据库异常。

落地项：

- 将事务冲突识别和重试移动到 `BatchControlService`、Repository transaction boundary 或对应 manager-loop adapter；
  Driver 只处理稳定的领域级 transient/terminal error。
- 在 `distributed/db.py` 统一数据库可重试错误分类；各操作仍可保留自己的重试次数、deadline 和 backoff，不建立笼统的全局重试器。
- `BatchControl` port 移到中立的 `batch/ports.py`，其方法和异常契约不得引用 SQLAlchemy、MySQL errno 或具体 Repository。
- 删除 InstanceManager、Reconciler 和 Batch Driver 中重复的数据库错误判断，并覆盖 deadlock、lock timeout、连接中断、
  重试耗尽及非数据库 adapter 测试。

### 57. DistributedConfig 按组件暴露窄配置视图

- [x] 状态：已完成

优先级：P3

`DistributedConfig` 同时承载 Gateway、Scheduler、Reconciler、Manager、Batch、Artifact、Environment、DependencyBuild 和
数据库配置，大多数服务构造函数因此接收完整配置，实际依赖被隐藏，单测也需要准备大量无关字段。

落地项：

- 保留 `DistributedConfig` 作为代码默认值、数据库 overlay 和显式开发覆盖的唯一解析结果；生产 Helm 只提供 database URL bootstrap。
- 为 Scheduler、Reconciler、Artifact、Environment、DependencyBuild、InstanceManager 等组件定义只读 dataclass 或宽松
  Protocol view；构造函数只接收实际使用的配置切片。
- 每个 view 只快照其 Protocol 声明的密钥读取方法及对应密钥子集，不保留完整 `DistributedSecrets`，未声明方法访问时抛出
  `AttributeError`。
- 在各进程 composition root 从完整配置生成并注入组件视图；Repository 只接收事务逻辑确实需要的设置，禁止重新读取环境变量。
- 配置视图保持数据对象，不承载业务流程或 provider client 创建；增加构造、默认值和跨字段约束测试。

### 58. 核心 Port/Protocol 与对象命名契约收口

- [x] 状态：已完成

优先级：P3

Executor/Driver capability Protocol 已较清晰，但部分真实约定仍通过 `getattr` 或结构偶合表达；部分 `Client`、`Service`、
`Controller` 名称也没有反映对象的所有权和生命周期。需要做一次小范围契约收口，而不是建立统一 Repository/Controller
继承树。

落地项：

- 将所有 Driver 已实际提供的 `stop_disposition` 加入 `ExecutionDriver` Protocol，并使用 `Literal` 或 enum 表达合法值，
  删除 InstanceManager 的默认 `getattr` 约定。
- 将 `BatchControl` 移入中立 port 后，让具体实现显式满足该 Protocol；`ArtifactStore`、`EnvironmentAllocator` 和
  `ExecutionReporter` 等同领域实现可显式继承稳定 Protocol。
- 调用方专用的小型 facade 继续使用结构化 Protocol；如果显式继承会迫使 adapter 反向依赖调用方模块，则不增加继承。
- 命名统一为：`Client` 表示出站 transport，`Service` 表示长生命周期业务对象，`Reconciler` 表示幂等状态推进器，
  `Repository` 表示数据库事务边界，`Renderer` 表示纯 manifest 生成，`Driver` 表示执行行为。
- 将 `BatchControlClient` 改为 `BatchControlService`；`ArtifactMaintenanceService` 改为
  `ArtifactMaintenanceReconciler`；Batch lifecycle 命名按第 52 项处理。暂不改动已成为外部配置契约的 executor mode 名称。
- 不新增通用 BaseRepository、BaseController 或全局 lifecycle 基类；仅在存在真实替换实现、依赖注入或运行时 capability
  判断时引入 Protocol，并补充契约测试。

### 59. Console 信息架构与大规模列表收敛

- [x] 状态：已完成

优先级：P1

当前 Console 将 Runs、Environments、Grants、Allocations、Managers 和 System 全部作为同级页面，并在首屏或选中对象后预取
精确总数、详情、事件、allocation 和 batch items。Execution 使用 offset 分页、无条件 Assignment join 和逐行读取 Assignment；
Managers、Allocations 和 System 也会读取或聚合大量历史对象。数据规模增长后，普通查询、管理诊断和浏览器渲染会互相拖慢。

落地项：

- 页面按普通工作区和管理员区组织；Runs 默认排除 batch child，Infrastructure 收纳 Allocations 与 Managers，普通用户不渲染
  管理员入口。
- 为 Batch parent、item 和 child Execution 增加可刷新、可分享的独立 Console 路由；Batch Detail 使用持久化 summary 和稳定
  item cursor，支持状态、item ID、child execution ID 和 error code 查询。item/attempt 使用详情抽屉，child 复用统一 Execution
  Detail 并保留返回 parent 的上下文。
- Execution、Allocation、Manager 和 batch item 列表统一使用服务端 cursor；热路径不计算精确总数。Execution 列表去掉无搜索时
  的 Assignment join 和逐行查询，使用一次投影查询或批量加载，并补充匹配 owner/status/top-level 排序的索引。
- 页面只在进入对应视图时读取详情、events、result、allocation、batch items、System diagnostics 和 Manager inventory；切换筛选或
  选择对象时取消旧请求，防止重复请求和过期响应覆盖。
- Managers 默认只查询 healthy/current allocation，stale 数据按 allocation/session 汇总后显式展开；System 首屏使用短 TTL 的
  低基数健康摘要，诊断明细拆分为按需接口。
- Environment 和 Grant 默认表单只展示常用资源项，其他 limits/timeouts 收入 Advanced overrides；Environment 支持完整
  default quota，My Environments 支持“默认”身份认领。原始 submission JSON 只作为高级调试入口。
- 前端单测覆盖路由恢复、懒加载、请求取消、preset 和权限导航；API/数据库 fixture 覆盖十万级 batch ledger、万级 Manager 历史
  数据、cursor 稳定性、top-level 过滤和 owner 隔离，验证查询不会读取全量数据。DevCloud 使用既定 25 Manager/2000 task 常规
  batch 验证页面链路；2000 Manager 页面容量验证只随既定 runner-dev 专项执行，不扩大测试环境范围。

验证：分布式单测 1247 条、Console 单测 19 条、10 万 item ledger 与 1 万 Manager cursor 压测通过。DevCloud 以 25 Manager、
每个 20 slots 完成 2000 task batch，2000 个 child 和结果 Artifact 均唯一且成功；Playwright 验证 cursor 翻页、筛选、item drawer、
child 返回上下文、System 懒加载以及桌面和移动端布局。本次隔离测试 release 已清理。

## 已完成

暂无。
