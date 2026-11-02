# artifact service 作为第三种 ArtifactStore 后端 · 方案与验证

> 跨仓库：`artifact_service`（`/data/workspace/artifact_service`，代码在子包 `artifact_service/artifact_service/`）与 `nexus`（`nexus/distributed/artifacts` + `nexus/utils/artifact`）。
>
> **当前进度**：`artifact_service` 后端 `/system` 免登录子集已在分支 `feat/system-identity` 实现并通过测试（unit + integration）；nexus 侧 `ArtifactServiceStore` + `MultiProviderArtifactStore` 已实现并通过单测（见 §8、§12）。剩余待办为部署侧灰度（P0 太湖报备 / P2 后端灰度 / P4–P6 联调上线，见 §10）。
>
> **姊妹文档**：test 环境**首次部署 / 上线步骤**（镜像滚动 + DB override + `kill 1` 生效）见 [`artifact-service-test-deployment-runbook.md`](./artifact-service-test-deployment-runbook.md)（含 dev 差异附录 C）。本文 = **设计** + **验证精华**；详细上线操作以部署 runbook 为准。

## 1. 目标

分布式制品层 `nexus/distributed/artifacts/store.py` 的 `ArtifactStore` 抽象目前有两种实现：`local`（本地目录）与 `generic_repository`（HTTP + BasicAuth，生产强制）。本方案接入**第三种** `artifact_service`：把系统管道制品（`source_package` / `wheel` / `nexus_wheel` / `uv_lock` / `wheelhouse` / `task_list` / `run_result`）落到一个受控的系统 namespace（owner = `svc-nexus-dist`）。

难点在于：store 层的三处构造点（application writer / maintenance reconciler / control metadata）都在**非沙箱系统进程**，`put/get/stat/delete` 时**没有任何员工身份可带**——员工 PAT 只在沙箱内经 Host↔沙箱 Socket.IO 桥可取，网关 / reconciler 没有这座桥；GC（零 execution 上下文）尤其无 PAT 可用。

## 2. 核心设计：免登录路径 → 系统账号

太湖网关支持把**报备的路径前缀**配置为**免登录**：对这些路径不做鉴权、也不注入 `x-tai-identity`。据此在 artifact service 上为 store 所需的接口**另挂一套 `/system` 前缀路由**——这些路径经太湖配置为免登录，请求到达后端时**不带任何身份头**，后端据此判定为**系统账号 `svc-nexus-dist`**。

| 维度 | 结果 |
|---|---|
| 常驻凭据 | **零凭据**——无需任何常驻 PAT，无过期/轮转/泄露面 |
| 后端改动 | 一套 `/system` 前缀路由 + 一个身份分支（`get_system_identity`），无授权表 |
| 贴合太湖 | 复用太湖官方免登录能力，非应用层伪造委托 |
| GC / reconciler | 零 execution 上下文场景**天然可用**（无需 PAT） |
| 主要代价 | 免登录 = 无鉴权入口，**强依赖网络隔离**（见 §7） |

## 3. 已定决策

1. **前缀 = `/system`**：系统路由为 `/system/v1/...`，与员工路由 `/v1/...` 物理隔离。
2. **只镜像 store 需要的子集**：不镜像 grant 管理、owned/shared 列表、admin 等员工/管理语义接口，最小化免登录攻击面（清单见 §5.2）。唯一例外是一个只读的 `whoami` 探针，用于端到端验证「免登录路径 → 系统账号」链路是否打通（只回显固定系统主体，无副作用、无数据库访问）。
3. **`/system/*` 若携带身份头 → 直接 `400 PayloadInvalid`**：系统路由不接受 `x-tai-identity`（或 dev 头），避免「免登录路径又混入真人身份」的语义歧义。
4. **太湖侧用前缀匹配报备 `/system`**：太湖「免登录路径」配置为**前缀匹配**（弹窗实测：路径填 `/system`）。因后端只在 `/system` 前缀下暴露 §5.2 的最小子集（决策 2），前缀放行 `/system` 与逐条白名单**效果等价**（其它 `/system/*` 一律 404）。**代码评审红线**：禁止在 `/system` 前缀下挂载任何超出 §5.2 子集的路由——否则新增路由会自动免登录、不再经报备。`app.py` 的路由注册块即为权威白名单，须与太湖报备保持同步。
5. **保留 `generic_repository` 作为一键回退**：`artifact_store_provider` 一行切回即可，后端 `/system` 路由保留不影响其他请求。

## 4. 端到端架构

```
┌─ nexus/distributed 系统进程 + reconciler ───────────────────────────────────┐
│   ArtifactServiceStore (第三 provider)                                      │
│     不带任何身份头（无 Authorization / 无 x-tai-identity）                  │
│     base_url + '/system' 前缀                                               │
│     object_uri(object_key) = 纯函数 → artifactsvc:///_nexus_dist/{key}      │
│     put/get/stat/delete 复用 ArtifactClient                                 │
└──────────────────────────────────┬──────────────────────────────────────────┘
                                    │ HTTPS（仅经太湖网关，禁直连 pod）
                                    ▼
┌─ 太湖网关 ──────────────────────────────────────────────────────────────────┐
│   /system 前缀在免登录白名单内 → 不鉴权、不注入 x-tai-identity              │
│   /v1/*（员工路由）→ 照常鉴权 + 注入 x-tai-identity                          │
└──────────────────────────────────┬──────────────────────────────────────────┘
                                    ▼
┌─ artifact_service 后端 ─────────────────────────────────────────────────────┐
│   /v1/*         → get_identity        → 强制 x-tai-identity（零改动）        │
│   /system/v1/*  → get_system_identity:                                      │
│       system_identity.enabled 为 false → 401（默认关闭，防误开）            │
│       带 x-tai-identity / dev 头       → 400 PayloadInvalid（决策 3）        │
│       否则 → RequestIdentity(employee_id='svc-nexus-dist',                  │
│                              subject_kind='system')                         │
│   owner_id = svc-nexus-dist → 系统 namespace                                │
│   发放 COS 预签名 URL → 客户端直传腾讯云 COS                                │
└─────────────────────────────────────────────────────────────────────────────┘
```

- 员工路由 `/v1/*` **一行不改**，零 regression。
- 「免登录 = 系统身份」被锁死在 `/system/*` 专用路径上，不会因网关注入偶发失败把员工请求静默降级为系统身份。

## 5. HTTP 契约

### 5.1 系统路由前缀

所有系统路由 = 员工路由路径前加 `/system`。后端用配置化前缀（默认 `/system`）注册薄封装 handler，复用同一批 `Service` 方法，仅把身份依赖从 `get_identity` 换成 `get_system_identity`。

### 5.2 store 能力 → 所需接口映射

| store 操作 | 依赖的接口 |
|---|---|
| `put` | `POST /system/v1/folders`（沿路径建树）、`POST /system/v1/artifact-uploads`、`POST /system/v1/artifact-uploads/{upload_id}/complete` |
| `get` | `GET /system/v1/resources`（按路径逐层 resolve）、`POST /system/v1/artifacts/{artifact_id}/download-url` |
| `stat` | `GET /system/v1/resources`、`GET /system/v1/artifacts/{artifact_id}` |
| `delete` | `GET /system/v1/resources`、`DELETE /system/v1/artifacts/{artifact_id}`、（GC 需要时）`DELETE /system/v1/folders/{folder_id}` |
| 链路探针（决策 2 例外） | `GET /system/v1/whoami`（只读，回显系统主体，无副作用） |

**不镜像**（决策 2）：`/v1/artifacts`（owned/shared 列表）、`/v1/resources/{id}/grants*`（grant 增删查）、全部 `/v1/admin/*`。

### 5.3 太湖免登录白名单（P0 报备对象）

太湖侧只需报备**一条前缀规则**：路径 `/system`、匹配逻辑=前缀匹配（决策 4）。下表是该前缀下**允许存在**的接口清单——既是后端 `app.py` 挂载的白名单，也是代码评审红线的对照表（前缀下不得出现表外路由）：

```
GET    /system/v1/whoami                                # 只读链路探针
POST   /system/v1/folders
POST   /system/v1/artifact-uploads
POST   /system/v1/artifact-uploads/{upload_id}/complete
GET    /system/v1/resources
GET    /system/v1/artifacts/{artifact_id}
POST   /system/v1/artifacts/{artifact_id}/download-url
DELETE /system/v1/artifacts/{artifact_id}
DELETE /system/v1/folders/{folder_id}                   # 供 GC 删空文件夹
```

### 5.4 行为矩阵

| 路由 | 身份头 | `system_identity.enabled` | 结果 |
|---|---|---|---|
| `/v1/*` | 有 | — | 员工行为，owner = 真人（不变） |
| `/v1/*` | 无 | — | 401（不变） |
| `/system/v1/*` | 无 | true | 系统身份，owner = `svc-nexus-dist` |
| `/system/v1/*` | 有 | true | **400 PayloadInvalid**（决策 3） |
| `/system/v1/*` | 任意 | false | 401（默认关闭） |
| `/system/v1/<表外路径>` | 任意 | true | 404（未注册即不存在） |

## 6. artifact_service 后端改动（已实现，摘要）

分支 `feat/system-identity`。核心三处：

- **`domain.py`**：`RequestIdentity` 加 `subject_kind: Literal['employee','system'] = 'employee'`（默认兼容既有构造点）。`create_upload` / `create_folder` / `_require_owner` 只用 `employee_id`，自然拿到 `svc-nexus-dist`。`subject_kind` 为纯内存字段，仅供分支/审计，**从不参与鉴权**——故**无需新增数据库列/迁移**（避免对 test/prod 存量库重迁移）。
- **`config.py`**：`SystemIdentityConfig{ enabled=False（默认关闭，防误开）, subject='svc-nexus-dist', path_prefix='/system' }`；`Config.system_identity` 默认工厂。`create_app` 在 `enabled=true` 时打显著启动 `warning` 提醒运维确认网络隔离。
- **`app.py`**：`get_system_identity` 依赖——`enabled=false → 401`；带 `x-tai-identity` / dev 头 → `400 PayloadInvalid`（决策 3）；否则返回 `RequestIdentity('svc-nexus-dist', subject_kind='system')`。`_register_routes` 只在 `{prefix}` 下挂 §5.3 白名单（whoami + 8 store 接口），带醒目注释声明「本块即免登录白名单，禁止新增表外路由」；表外 `/system/v1/*` 未注册 → 404。

审计：owner/actor 恒为 `svc-nexus-dist` 即可归因系统动作；如需区分「哪个系统组件触发」可另带**非鉴权** `X-System-Component` 头仅记审计（绝不参与鉴权，可选增强）。

## 7. 安全命门 ⚠️（部署层保证，后端无法强制）

免登录路径 = **无鉴权入口**，与 `docs/design/artifact-service-design.md:120`（**artifact_service repo**，即 `/data/workspace/artifact_service`）「生产必须置于网关之后，不得暴露可绕过网关的直连入口」直接相关。硬前提：

> **`/system/*` 必须严格只能经太湖网关到达；k8s Service / Ingress / Pod 端口层面不能让内网或外部直连命中它。** 否则任何能连到 pod 的人都能以系统身份读写系统 namespace。

后端补偿：`system_identity.enabled` **默认 false**；开启时 `create_app` 打显著告警；`/system` 与 `/v1` 物理隔离，员工路由永不降级。

| 攻击面 | 结论 |
|---|---|
| 直连 pod 命中 `/system/*` | **本方案核心风险**，靠网络隔离消除；后端默认关闭 + 告警兜底 |
| 员工路由被降级为系统身份 | 不可能——两套路由物理隔离，`/v1/*` 强制鉴权 |
| `/system/*` 误带真人身份 | 400 拒绝（决策 3），不静默 |
| 前缀下误挂表外路由 | 前缀匹配会让其自动免登录 → 靠决策 4 代码评审红线拦截；表外路径未注册 → 404 |
| 中间人 / header 篡改 | 全链路 HTTPS + 太湖网关 |

## 8. nexus 侧改动（已实现）

### 8.1 `ArtifactServiceStore`（新文件 `nexus/distributed/artifacts/service_store.py`）

- 内部持一个 `ArtifactClient`（`nexus/utils/artifact/client.py`），**base_url 指向 `<host>/system`**，**不注入 Authorization、不注入 `x-tai-identity`**（`_SystemArtifactClient` 强制抹掉，即使进程环境设了 PAT / dev 头）。
- **可验证服务身份（`X-System-Secret`）**：`_SystemArtifactClient` 在 `_apply_auth_headers` 里，当配置了 `shared_secret` 时注入 `X-System-Secret: <secret>` 头（backend 常量时间比对）。这样「没有身份头」不再等于系统身份——调用方必须持有正确密钥。未配 secret 时（本地 dev 打不强制的 backend）发裸请求，向后兼容。`shared_secret` 由 `ArtifactServiceStore(shared_secret=...)` 透传，来源见 §8.3。
- `put/get/stat/delete` 复用 client 的 upload / download / resolve 能力；`get` 时本地重算 sha256（对齐 `GenericRepositoryArtifactStore.get`），补足 COS 只校 content-md5 的缺口。
- 员工 CLI 路径（普通 `ArtifactClient`）完全不受影响：base_url 仍是 `/v1`，照常带 PAT。

### 8.2 `object_uri` 纯函数（关键硬约束）

`repositories/artifact.py:154-176` 建库与 `gateways/application/routes/artifact_content.py:140-144` 上传两次算 `object_uri` 并断言相等（否则 `artifact_object_uri_mismatch`）；而 SDK 的 `artifact_id` 上传后才由服务端分配，**绝不能进 URI**。

```
object_key = artifact_object_key(owner_principal_id, artifact_id_nexus, sha256, kind)
           = f'{owner_hash(24hex)}/{artifact_id_nexus}/{sha256}/{canonical_filename}'
object_uri(object_key) = f'artifactsvc:///_nexus_dist/{object_key}'   # 纯函数
```

- `put`：沿路径 get-or-create folder（`idempotency_key=f'nexus-dist-dir:{path}'`）→ `create_upload(idempotency_key=f'nexus-dist-obj:{object_key}')` → COS 直传 → `complete_upload`。
- `get/stat/delete`：uri 反算路径 → `resolve_resource_id(path, expect='artifact')`（走 `GET /system/v1/resources` 逐层）→ 后端 `artifact_id_service` → `download_to_file` / `get_artifact` / `delete_artifact`。
- 幂等：object_key 含 `sha256`，重复 `put` 命中同路径视为已存在。
- **folder 路径 LRU 缓存**：`artifact_service_folder_cache_size` 默认 4096，只缓存中间 folder id。
- **核心不变量**：nexus 里 `art_*` id 在 artifact_service 中**只是一层 folder name**，服务端为真正的 leaf artifact 另分配 `art_01M…` ULID 形态 resource_id。两个 id 完全独立。

### 8.3 配置（`nexus/distributed/config/model.py`，**用共享密钥而非员工 PAT**）

```python
artifact_service_base_url: str = 'https://artifact-nexus-agent.mcp.woa.com'  # 默认 prod；test/dev 必须显式覆盖
artifact_service_system_prefix: str = '/system'
artifact_service_namespace_prefix: str = '/_nexus_dist'
artifact_service_folder_cache_size: int = 4096
artifact_store_read_fallback_providers: list[str] = []   # 切换期读兼容，见 §8.6
```

- `config/views.py` 的 `ArtifactConfigView` 加 `artifact_service_shared_secret()` 访问器（走 secret_reader，与 `artifact_credential` 同模式）。
- `config/secrets.py` 的 `DistributedSecrets` 新增 `artifact_service_shared_secret: SecretStr | None`（+ `read_artifact_service_shared_secret()`）。这是**服务级密钥**、不是员工 PAT；与 `artifact_writer`/`artifact_maintenance` 一样由部署侧下发（k8s Secret），需与 artifact_service 后端挂载的 `system_identity.shared_secret_file` 用同一值。

### 8.4 store 工厂追加分支（`store.py:create_artifact_store` / `_create_single_store`）

`artifact_store_provider == 'artifact_service'` 时构造 `ArtifactServiceStore(shared_secret=config.artifact_service_shared_secret())`。三处构造点（application writer / maintenance reconciler / control metadata）+ reconciler 全部通过工厂拿实例，零改动；不再有 `access != 'metadata' and pat is None` 的 PAT 分支。

### 8.5 生产门禁放开（`config/model.py::validate_for_service`）

- provider 白名单加入 `artifact_service`（`{local, generic_repository, artifact_service}`）。
- 生产（用 store 的服务）要求 provider ∈ shared `{generic_repository, artifact_service}`。
- provider=`artifact_service` 时要求 `artifact_service_base_url` 以 `https://` 开头，**且要求 `artifact_service_shared_secret()` 非空**（缺失即启动失败——「没有身份头」不能当凭据）。
- **无 PAT 校验项**。

### 8.6 `MultiProviderArtifactStore` 读兼容（切换期，`store.py`）

`object_uri` 带 provider scheme（`artifactsvc://` vs 旧 `generic://`）。直接把 provider 从 `generic_repository` 切到 `artifact_service`，历史 `generic://` 对象将无法被新后端解析（抛 `stored_object_provider_mismatch`）。为覆盖切换期新增 `artifact_store_read_fallback_providers`：**写只走 primary，读/删按 URI scheme 分发**——新对象走 `artifact_service`，历史 `generic://` 自动路由回 `generic_repository`。

- 每个后端类上声明 `uri_scheme: ClassVar[str]`：`LocalArtifactStore=file` / `GenericRepositoryArtifactStore=generic` / `ArtifactServiceStore=artifactsvc`。
- `put` / `object_uri` / `uri_scheme` **仅** primary（绝不双写）；`get/stat/delete` 按 `urlparse(uri).scheme` 查 `_by_scheme` 表转发；未知 scheme 抛 `stored_object_provider_mismatch`；`close()` 依次 close 所有后端并重抛第一个异常。
- 工厂：先剔除 `provider == artifact_store_provider` 的 fallback 项并去重；为空则返回裸 primary（不套 wrapper）；否则包 `MultiProviderArtifactStore(primary, fallbacks)`。`access` 逐个透传给 fallback（`generic_repository` 作 fallback 时 reconciler 仍取到 `maintenance` 凭据删历史对象）。
- 门禁：fallback 每项须在 `{local, generic_repository, artifact_service}`；生产 fallback 须 shared（禁 `local`）。
- 待历史 `generic://` 全部过期后移除该配置回到纯 `artifact_service`。

### 8.7 错误归一

`ArtifactClientError.http_status ∈ {408,425,429,5xx}` + aiohttp/COS 瞬时错误 → `ArtifactStoreTransientError`（触发 `store.py` 重试）；`404 → FileNotFoundError`；`409/412 → FileExistsError`；`400/401/403 → RuntimeError('artifact_store_forbidden')`（免登录未生效 / 误带身份头 / 系统身份被关闭）。

## 9. 关键流程

**上传 run_result**（application-gateway `/v1/internal/artifacts/run-result`）：`create_upload` → `ArtifactRow(status=uploading, uri=object_uri(object_key))` → 收字节 → 断言 `store.object_uri(key) == ArtifactRow.uri`（纯函数一致）→ `ArtifactServiceStore.put`（uri 反算路径 → get-or-create folder，打 `/system/v1/folders` 无身份头 → 免登录 → `svc-nexus-dist` → COS 直传 → complete）→ `publish_run_result_ready` → `status=ready`。

**读回 / 删除**：`get` = uri 反算路径 → `GET /system/v1/resources` 逐层 resolve → `POST /system/v1/artifacts/{id}/download-url` → COS GET → 本地重算 sha256；`delete` = 同上 resolve → `DELETE /system/v1/artifacts/{id}`。

## 10. 分阶段落地

| 阶段 | 内容 | 状态 / 验收 |
|---|---|---|
| P0 太湖报备 | 按 §5.3 报备一条前缀规则：路径 `/system`、前缀匹配 | ✅ 已完成（test 上 `curl http://test.artifact.nexus-agent.woa.com/system/v1/whoami` 三态齐全，见 §12.1；生效后 `/system/*` 到后端不带身份头） |
| P1 后端 | `subject_kind` + `SystemIdentityConfig` + `get_system_identity` + `/system` 子集路由 + 单测/集成测试 | ✅ 已完成（`feat/system-identity`）：带身份头 400、未带→system、enabled=false→401；表外 404；`/v1/*` 无 regression |
| P2 后端灰度 | test 部署 `system_identity.enabled=true`，确认网络隔离 | ✅ 已完成（test artifact-service host `http://test.artifact.nexus-agent.woa.com`；whoami 三态验证通过；直连 pod 无法命中 `/system/*`） |
| P3 nexus store | `ArtifactServiceStore` + object_uri 纯函数 + folder LRU + 反查 + 门禁放开 + 无 PAT 配置 | ✅ 已完成；单测 `test_service_store.py` 覆盖一致性/往返/sha256/幂等/LRU/无身份头/错误归一/门禁 |
| P3.5 切换读兼容 | `MultiProviderArtifactStore` + `artifact_store_read_fallback_providers` | ✅ 已完成；单测 `test_multi_provider_store.py` |
| P4 dev 联调 | dev 切 provider，跑 source_package + run_result 闭环（三层证据链 §12.3） | ✅ 已完成（附录 C 完整走通：INSERT 全局 override + 三个组件 kill 1 + 三层证据链齐过；文档见 runbook §附录 C） |
| P5 test 联调 | test k8s 部署，24h 观察 | 待办；无 `artifact_object_uri_mismatch`；错误率 < 1.5× generic |
| P6 prod 上线 | 与 SRE 确认网络隔离后切换；保留 `generic_repository` 回退 | 待办；上线 7 天无 P1 |

## 11. 风险与回退

| 风险 | 缓解 |
|---|---|
| `/system/*` 可绕过网关直连 | **网络隔离（命门 §7）** + 后端默认关闭 + 启动告警 + 写入速率告警 |
| 前缀下误挂表外路由 | 前缀匹配自动免登录 → 靠决策 4 代码评审红线拦截；`/v1/*` 不在 `/system` 前缀下永不受影响 |
| 后端故障 | `artifact_store_provider` 切 `generic_repository`，nexus 侧一行生效（决策 5） |
| `/system/*` 被中间件补注身份头 | 后端 400 拒绝，暴露而非静默 |

**回退路径**：切 `artifact_store_provider: generic_repository`（并恢复 `artifact_generic_*`），滚动重启；后端 `/system` 路由保留不影响其他请求。

## 12. 验证（test 联调 + dev e2e 精华）

> 详细上线操作（镜像滚动 / DB override / `kill 1`）见 [`artifact-service-test-deployment-runbook.md`](./artifact-service-test-deployment-runbook.md)。本节只给**验证方法与判据**。
>
> 端点：控制面 test `https://test-control-runner-nexus-agent.mcp.woa.com` / prod `https://control-runner-nexus-agent.mcp.woa.com` / dev `http://21.160.200.50`（HTTP，auth_mode=disabled 免 token）。`/system` 免登录路由 host 由太湖 P0 报备决定（dev pod 出网口示例 `http://test.artifact.nexus-agent.woa.com/system`）。dev 上 namespace 用 `/_nexus_dist_dev`、业务 owner=`local-dev`（`owner_hash=sha256("local-dev")[:24]=08846e81b3a87fd9cc39d402`）。

### 12.1 连通性探针 · whoami 三态（最轻量，不碰 DB/COS）

```bash
BASE='<太湖 /system 免登录 host>'   # 例如 http://test.artifact.nexus-agent.woa.com/system
curl -sS -o /dev/null -w 'noauth=%{http_code}\n'   $BASE/v1/whoami          # 期望 200
curl -sS                                           $BASE/v1/whoami; echo    # 期望 {"employee_id":"svc-nexus-dist","subject_kind":"system"}
curl -sS -o /dev/null -w 'with_hdr=%{http_code}\n' -H 'x-tai-identity: dummy' $BASE/v1/whoami   # 期望 400
```

- `noauth=200` + `svc-nexus-dist` → `/system` 免登录 + `system_identity.enabled` 都对。
- `401` → P0 未就绪 / `system_identity.enabled` 未生效 / 走到了会鉴权的路由。
- 无头也 `400` → 网关给无头请求注入了身份头，allowlist 路由配置不对。

### 12.2 切 nexus store（配置）+ 存量读兼容

在 nexus distributed helm 的 `distributed_config_overrides`（渲染进 global.json 的 `DistributedConfig` 字段）里：

```json
{
  "artifact_store_provider": "artifact_service",
  "artifact_service_base_url": "<太湖 /system passwordless 网关 host，仅到 host 不含 /system>",
  "artifact_service_system_prefix": "/system",
  "artifact_service_namespace_prefix": "/_nexus_dist",
  "artifact_service_folder_cache_size": 4096
}
```

- `artifact_service` 后端**无凭据**，`distributed_config_secret_overrides` 不需加 artifact backend credential。
- 影响三个服务：`application-gateway`（writer）、`reconciler`（maintenance）、control metadata；三者共享同一份全局 override，**不要只给个别服务单独切**，否则未切的服务解析不了新 scheme。滚动重启（`kill 1`）后生效。
- **存量数据切换期**：加 `"artifact_store_read_fallback_providers": ["generic_repository"]` + 恢复 `artifact_generic_*`（见 §8.6）；fallback 仅读/删，绝不双写；历史 `generic://` 清空后移除该配置。

### 12.3 三层证据链验证（一次真实 `submit single`，**不带 `--config-only`**）

| 层 | 观察点 | 意味着 |
|---|---|---|
| A | 控制面 DB `distributed_artifacts.uri` 前缀 = `artifactsvc://` | `primary=artifact_service` snapshot 生效 |
| B | artifact_service `/system` 侧 folder 树 + leaf `owner_id=svc-nexus-dist` | 系统身份写入闭环、idempotency fix 生效 |
| C | 从 artifact_service 下载 → sha256 复算 == 上传前 sha256 | 字节零污染 |

**Step 0 · 提交并拆 baseline**：`submit single`（最小 echo RunInput）拿到 `EXE=exe_...`，再拆三个系统制品 baseline：

```bash
curl -sS "$CONTROL_URL/v1/executions/$EXE" | python3 -c 'import json,sys; c=json.load(sys.stdin)["spec"]["code"]; [print(k.ljust(15), c[k]["artifact_id"], "sha256="+c[k]["sha256"], "size="+str(c[k]["size_bytes"])) for k in ("source_package","nexus_wheel","uv_lock")]'
```

**Layer A · DB 品牌位**（进任一 dev/test pod 连 MySQL，替换三个 `art_*`）：

```sql
SELECT artifact_id, kind, uri, sha256, size_bytes, status, owner_principal_id
FROM distributed_artifacts WHERE artifact_id IN ('art_...','art_...','art_...');
```

期望每行：`uri` = `artifactsvc:///_nexus_dist[_dev]/{owner_hash}/{artifact_id}/{sha256}/{canonical_name}`（前缀 `artifactsvc://` 是 primary 生效的硬证据）；`status=ready`；`sha256`/`size_bytes` 对齐 baseline。
**常见误期望**：`owner_principal_id` 是**业务 owner**（dev=`local-dev`），**不是** `svc-nexus-dist`——系统账号只在 artifact_service 服务端 leaf `owner_id` 出现，不落 nexus 这张表。（`created_at` 存 UTC，做时间过滤要转 UTC 或省略。）

**Layer B · `/system` 侧 folder 树 + owner**：whoami 三态（§12.1）通过后，沿路径逐层 list `GET $BASE/v1/resources?parent_folder_id=X` resolve 五段 `_nexus_dist[_dev] / owner_hash / nexus_art_id / sha256 / canonical_name`，命中 leaf 断言 `owner_id == svc-nexus-dist`。坑：`/system` 的 artifact 项常不返 `resource_type`，用 `folder_id` vs `artifact_id` 兜底判类型（对齐 `nexus/utils/artifact/client.py` 的 `_item_type`）；`/system` 无 `/v1/resolve`，只能逐层 list；`download-url` 响应 URL 字段跨方言可能叫 `url`/`download_url`/`signed_url`/`cos_url`/`presigned_url`。

**Layer C · 字节零污染**：`POST $BASE/v1/artifacts/{resource_id}/download-url` → `curl -sSL` 下载 → 本地 sha256 == baseline sha256、size == baseline size。若 mismatch：`curl -sSLI $URL` 看响应头是否含 `Content-Encoding`（有则 store 侧转码配置有问题）。

**三层齐过 = 切换 e2e 满分收尾。**

### 12.4 排障速查

| 现象 | 可能原因 | 处理 |
|---|---|---|
| `/system/v1/whoami` 返 401 | P0 未就绪 / `system_identity.enabled` 未生效 / 走到会鉴权的路由 | 核对太湖 allowlist + test/dev 配置已开 `system_identity` |
| 无头请求也返 400 | 网关给无头请求注入了 `x-tai-identity` | allowlist 路由配置错误，需让路由不注入任何身份头 |
| `submit` 报 `500 internal_error` / `PUT .../content 500` | 镜像里 folder-conflict recovery 或 object-idem hash 没生效 | 回部署 runbook §5 逐 pod 复核 image digest，再 `kill 1` |
| store 报 `artifact_store_forbidden`（RuntimeError） | 后端返回 400/401/403：免登录未生效 / 身份头被注入 / system_identity 关闭 | 部署故障非瞬时，回 §12.1 探针定位 |
| store 报 `stored_object_provider_mismatch`（ValueError） | 拿到的 `uri` scheme 不在当前 store 表——切换期忘配 `artifact_store_read_fallback_providers` | 按 §8.6 补 fallback（如 `["generic_repository"]`）；历史对象已清空可忽略 |
| store 报 `ArtifactStoreTransientError` | 408/425/429/5xx 或网络/COS 抖动 | 自动重试；持续则查后端与 COS |
| Layer A `uri` 仍 `file://` / `generic://` | DB 没改 或 `kill 1` 没跑 → snapshot 还是旧的 | 部署 runbook §5 重跑 `kill 1`；确认 pod ready 时间在 DB UPDATE 之后 |
| Layer B resolve 停在 owner_hash 段 | `owner_hash` 算错 | `python3 -c 'import hashlib;print(hashlib.sha256(b"<owner>").hexdigest()[:24])'` 重算 |
| dev/办公 curl `21.149.1.80:8080` 被 403 | igate 跨层直连 test k8s Service | 改用 `*.mcp.woa.com` 网关入口，不要直连 Service |

## 13. 关键代码位置

**后端（分支 `feat/system-identity`）** · 以下路径全部位于 **artifact_service 独立仓库**（`/data/workspace/artifact_service`），不在 nexus 仓库
- 系统身份依赖 / 路由注册（免登录白名单）/ 启动告警：`artifact_service/app.py`（`get_system_identity`、`_register_routes` 内 `{prefix}/v1/*` 块、`create_app`）
- 配置：`artifact_service/config.py::SystemIdentityConfig`；身份模型：`domain.py::RequestIdentity.subject_kind`
- owner 落地：`service.py::create_upload` / `create_folder` / `_require_owner`
- 设计约束：`docs/design/artifact-service-design.md:118,120,130`；后端 test 配置：`configs/artifact-service.test.yaml`

**nexus**（本仓库 `/data/workspace/nexus`）
- store 接口 / 工厂 / MultiProvider：`nexus/distributed/artifacts/store.py`（`ArtifactStore` Protocol 的 `uri_scheme`；`create_artifact_store` / `_create_single_store`；`MultiProviderArtifactStore`）
- 第三 provider 实现：`nexus/distributed/artifacts/service_store.py`（`ArtifactServiceStore` + `_SystemArtifactClient` 免身份头、`_ensure_folder_path`）
- object_uri 断言：`repositories/artifact.py:154-176`；上传断言 `gateways/application/routes/artifact_content.py:140-144`；`artifact_object_key` `repositories/artifact.py:835`；`_CANONICAL_FILENAMES` `repositories/artifact.py:46`
- 配置字段 / 门禁 / 视图：`config/model.py`（`artifact_service_*` + `artifact_store_read_fallback_providers`；`validate_for_service` provider/fallback 白名单 + 生产 HTTPS/shared）、`config/views.py::ArtifactConfigView`；snapshot 加载 `config/store.py::initialize_distributed_config`
- 单测：`tests/unit/distributed/artifacts/test_service_store.py`、`tests/unit/distributed/artifacts/test_multi_provider_store.py`
- SDK 复用：`nexus/utils/artifact/client.py`（`ArtifactClient`、`create_upload`、`upload_file`、`resolve_path`/`resolve_resource_id`、`_apply_auth_headers`）、`nexus/utils/artifact/uploader.py`（幂等建树样板）
