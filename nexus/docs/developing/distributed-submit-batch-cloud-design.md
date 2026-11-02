# Distributed Submit 和 Batch 上云设计

## 目标

本设计在现有 distributed `Execution -> Assignment -> InstanceManager` 模型上增加：

- `nexus submit single|live|batch` 三类受控提交接口。
- `predefined` 和 `dynamic` 两类 Environment 资源分配方式。
- 本地或云端生成 task list，再把 batch child 作为普通 `nexus_single` 调度。
- source package、Nexus wheel、lockfile、wheelhouse、task list 和最终结果的远端 Artifact 存储。
- 基于太湖个人 `x-tai-identity`、应用 x-rio 身份和 Environment Grant 的用户隔离。
- 可查看 batch 子任务、资源分配和系统资源的用户/管理员页面。
- 单 batch 1 万到 10 万 item，活跃 child 数始终受 `max_inflight` 限制。

本期不做：

- `DatabaseTraceSink` / `DatabaseTraceSource` 和完整 span/event/log 远端持久化。
- 终态 live run 的事件回放；动态 Manager 释放后只保留最终结果。
- `--retry-from`、跨 batch 失败项重放和普通用户 execution 删除。
- 累计 artifact bytes、batch item 数量或历史存储配额。
- 动态资源复用、共享 warm pool 或按 workload 自动合并 allocation。
- 用户自定义 Python 版本、pip/uv 参数、index URL、额外 requirements 或本地 modules。

## 核心模型

### Workload

所有 InstanceManager 继续使用同一个实现，通过保留字段 `workload` 区分职责：

| Run | Manager workload | Driver |
| --- | --- | --- |
| single | `execution` | `nexus_single` |
| live | `execution` | `nexus_live` |
| batch parent | `orchestration` | `nexus_batch` |
| batch child | `execution` | `nexus_single` |

不存在单独的 OrchestrationManager 类。`orchestration` Manager 只是只接受 `nexus_batch` 的普通
InstanceManager，不能承载 single/live/child。

### Environment

Environment 由管理员预先创建，再通过 Grant 授权给用户。字段至少包括：

```text
environment_id
name
allocation_type          predefined | dynamic
provider_type            predefined | kubernetes | tkex
provider_config_ref
resource_limits
default_timeouts
config_version
status                   enabled | disabled
```

MVP 只允许以下组合：

| allocation_type | provider_type | 行为 |
| --- | --- | --- |
| `predefined` | `predefined` | 使用已经存在的 pool 和 Manager，不申请资源 |
| `dynamic` | `kubernetes` | 每次 submit 创建独占 Kubernetes 资源 |
| `dynamic` | `tkex` | 每次 submit 创建独占 TKEx 资源 |

`provider_config_ref` 引用管理员通过 Control API/Console 管理的 `distributed_provider_configs` 记录。记录保存 provider type、
API endpoint、namespace、逻辑 `credential_ref` 和非敏感参数；未知、disabled、provider type 或 namespace 不匹配时，Environment
管理 API 和 submission 在创建 allocation 前 fail close。ProviderConfig 只允许原地修改非敏感配置，provider type 不可变。
MVP 的一个 Reconciler 部署只管理 `environment_namespace` 指定的一个 workload namespace；所有记录的 namespace 必须与它一致。
Environment 不物理删除，只能 disabled；allocation 保存创建时的 `config_version` 快照。
Disabled Environment 只禁止新 submission，不停止已经创建的 execution。

```json
{
  "provider_config_ref": "kubernetes/prod-v1",
  "status": "enabled",
  "config": {
    "provider_type": "kubernetes",
    "namespace": "nexus-prod",
    "auth_type": "secret",
    "api_url": "https://kubernetes.example",
    "credential_ref": "kube-prod-v1"
  }
}
```

远端 TKEx/Kubernetes 凭据保存在平台密钥配置中，ProviderConfig 只保存逻辑 `credential_ref`。Reconciler 从启动时加载的
`DistributedSecrets.provider_credentials` 解析强类型凭据；`in_cluster` Kubernetes 继续使用 Pod 的 ServiceAccount token/CA。
`credential_ref` 不能包含路径；远端 Kubernetes endpoint 必须为 HTTPS，TKEx endpoint 允许部署管理员配置 HTTP(S)，以兼容
官方内网 API。allocation 和 DependencyBuild 保存不含凭证明文的 ProviderConfig 快照，确保配置更新不改变在途资源。

### Allocator

为了避免与 InstanceManager 混淆，资源实现统一叫 allocator：

```text
PredefinedEnvironmentAllocator
KubernetesEnvironmentAllocator
TkexEnvironmentAllocator
```

它们实现最小接口：

```python
class EnvironmentAllocator(Protocol):
    async def observe(self, allocation): ...
    async def create(self, allocation): ...
    async def delete(self, allocation): ...
    async def observe_deletion(self, allocation): ...
```

`EnvironmentController` 是 `ReconcilerService` 内的普通类，不新增服务。Allocator 不实现通用 scale/patch；
本期 dynamic allocation 创建后规格不可变。

## 用户、Grant 和默认额度

用户入口接受两种互斥的太湖身份。个人或 CLI 的 `Authorization: Bearer` 是太湖 PAT，仅由太湖入口消费并转换为
`x-tai-identity` compact JWE；Nexus 不把 Bearer 解释为 submit token。应用调用方使用自身 App ID 和 App Token 生成
`x-rio-*` 请求头，SGW 校验调用方签名和 API 订阅后，使用 Nexus 的被调用方 Token 对后端请求签名。Gateway 使用平台密钥中的
同一个太湖应用 Token 解密个人 JWE，或校验 SGW 后端签名后读取 `x-rio-paasid`。个人 JWE 严格校验 `alg=dir`、
`enc=A256GCM` 和 `Expiration`，以 `StaffId` 生成稳定 principal，`LoginName` 只用于展示；应用 ID 必须命中管理员配置的 token
binding。个人、应用和内部委托身份混用时统一拒绝。

合法太湖身份首次访问时，在一个事务内幂等创建普通 principal 和唯一 `system_default` token row。该 token 没有明文、hash、
prefix、Environment 或 scopes，用户界面只显示“默认”，仅作为系统内部的额度认领身份，不作为 HTTP credential。数据库继续保留
`SubmitTokenRow` 和 `api` kind 以便未来扩展，但 MVP 不开放 token 创建、查询、吊销 API，也没有 Tokens 页面。

principal role 只有 `user` 和 `admin`。普通用户只能访问自己的 execution、Artifact 和 enabled Environment Grant；跨 owner
访问统一返回不泄露内部 detail 的 404。管理员可管理用户角色、Environment、Grant、allocation、Manager 和系统状态。

管理员可以给 Environment 配置完整的 `default_quota`：

```text
manager_count, cpu_millicores, memory_bytes, slots, active_batches
```

五项必须同时存在、为正整数且不超过 Environment 对应上限。普通用户通过 Environment catalog 查看可用环境，并使用“默认”
身份认领整份额度。认领幂等创建 `EnvironmentGrant` 和 resource usage row；Grant limits 是认领时的快照，后续修改 default quota
只影响新认领。已有 enabled Grant 直接返回，disabled Grant 必须由管理员处理，用户不能自行恢复。认领不预留资源，实际 submission
仍在 allocation admission 中同时受 Environment 上限、Grant 上限和本次申请值约束。

Grant canonical scopes 为 `submission:single`、`submission:live`、`submission:batch`、`artifact:write`、`execution:read`、
`execution:command` 和 `execution:stop`。自助认领授予完整 canonical scopes；管理员页面只提供 `All submissions` 和 `Custom`
快捷方式，单类型权限通过 Custom 配置。Grant disabled 只拒绝后续请求，不自动停止已有 execution。

MVP 只累计当前 compute allocation，不累计历史 item、bytes 或 storage。Environment 仍以逐请求硬限制控制 task 数、task-list、
source package、generation timeout、workspace 和单 Artifact 大小。Environment 与 Grant 的同名限制逐字段取更严格值；未知字段、
非法整数、默认 timeout 超限和 default quota 超限均拒绝入库。

CLI 使用 `--taihu-token` 或私有权限的 `--taihu-token-file` 将个人太湖 PAT 放入 Bearer，不把凭据写入 submission payload 或
source package。浏览器个人身份的 mutating API 使用由太湖身份派生的 CSRF token；应用身份不使用浏览器 CSRF。内部服务调用继续
使用 internal credential、受控身份委托或短期 allocation/Artifact ticket，与公网身份链路分离。

太湖协议实现统一放在 `nexus.distributed.access.taihu`：包含个人 JWE 解密、SGW 被调用方签名校验、太湖请求头常量和身份解析。
`access.auth` 只负责认证方式选择、`AuthContext`、权限策略和 repository 编排；不保留独立 `access.sgw` 模块。Gateway 兼容层
只从 `taihu` 引用需要过滤的身份头，不能重复实现签名或解密逻辑。

## Code 和依赖

### Code Mode

Submission 明确选择：

```json
{"code": {"mode": "config_only"}}
```

或：

```json
{
  "code": {
    "mode": "source_package",
    "source_package": {"artifact_id": "art_source", "kind": "source_package", "sha256": "...", "size_bytes": 123},
    "nexus_wheel": {"artifact_id": "art_wheel", "kind": "nexus_wheel", "sha256": "...", "size_bytes": 456},
    "uv_lock": {"artifact_id": "art_lock", "kind": "uv_lock", "sha256": "...", "size_bytes": 789},
    "extras": ["extensions"]
  }
}
```

或由用户选择已经包含项目代码和依赖的 workload 版本：

```json
{"code": {"mode": "image", "image_version": "user-build-20260802-001"}}
```

规则：

- `predefined` 只接受 `config_only`；CLI 必须显式传 `--config-only`。
- `dynamic` 默认打包 source package，也可显式使用 `config_only` 或 `image`。
- `config_only` 不上传 source、wheel 或 lockfile，不允许本地 imports、额外 requirements、modules、pip 参数或 index URL。
- `image` 只接受 Docker tag 语法的 `image_version`，拒绝 `latest`、repository、registry、digest、斜杠、冒号和 `@`；CLI
  使用 `--workload-image-version`，并与 `--config-only`、`--source` 和 `--wheel-extra` 互斥。
- `image` 不上传 source、wheel 或 lockfile，不创建 DependencyBuild 或 wheelhouse。Batch parent 和 execution Managers 使用同一版本。
- task list 和 run result 仍使用 Artifact，dataset reference 仍指向 Dataset Service；`config_only` 和 `image` 只改变代码交付方式。

Source package 的唯一格式是确定性 `source.tar.zst`；不保留 ZIP 兼容路径，也不在 archive 内嵌 manifest 或逐文件 hash。
CLI 使用 source root 的 `.gitignore` 语义，并独立强制排除 `.git`、venv、cache、输出目录、常见密钥模式，以及本次提交实际使用的
Taihu/WeData 凭据文件；强制规则和凭据路径不能被 `.gitignore` negation 重新包含。归档按路径排序写入，归一化 mtime、uid/gid、
owner 名、目录 mode 和普通文件 mode：目录固定为 `0755`，普通文件只按原 executable bit 取 `0644` 或 `0755`；zstd 参数固定并
启用 frame checksum。archive 的 `size_bytes` 和 sha256 在写入时一次计算完成。CLI 使用当前 checkout 构建
`nexus-agent` wheel，并要求项目根目录存在与 `pyproject.toml` 同步的 `uv.lock`；三个 Artifact 分别上传，用户不能提交任意
wheel 或 lockfile 替换后两个输入。

Source package 保留相对符号链接但不解引用。打包时用 `lstat/readlink` 校验链接最终目标存在、位于 source root 内且未被
`.gitignore`、强制规则或显式凭据路径排除；绝对、越界、悬空、真实解析成环的 symlink 以及 hardlink、FIFO、socket、device
一律拒绝。
解包器流式写入临时目录，只接受目录、普通文件和 symlink，校验成员路径、重复或类型冲突、成员数、单文件大小、总解压大小、
archive 大小和 zstd window；目录和普通文件先落盘，symlink 最后创建并再次严格解析。成功后原子替换 workspace，失败不留下
半成品。绝对资源上限和受限 Builder 已限定最坏开销，因此不再设置压缩比限制；配置统一使用 `max_archive_bytes`，不保留
`max_zip_bytes` 或 `max_source_compression_ratio` 兼容项。该格式尚未上线，因此不提供 ZIP 双读或迁移逻辑；Artifact kind、
submission JSON 和 `ArtifactRef` 不变，canonical 文件名改为 `source.tar.zst`，build key 随新的 Artifact hash 自然变化。

### 镜像和 DependencyBuild

部署只支持 Python 3.13，并使用三个不可变镜像角色：

| 镜像 | 内容 | 用途 |
| --- | --- | --- |
| `universal` | Tlinux 4、Python 3.13、`nexus-agent[distributed]` | Control/Application Gateway、Scheduler、Reconciler、db-init |
| `workload` | `universal` + `runtime`、boto3、litellm | config-only、source-package 与 image Manager |
| `builder` | `workload` + C/C++/Rust 工具链、build/wheel/auditwheel | wheelhouse builder Job |

Helm 仍保留 `controlImage`、`workloadImage`、`builderImage` 三个变量；默认 control 指向 `universal`。同一 release 的三个 tag
相同，生产环境分别使用 digest 固定内容。workload 不包含编译器，builder 不承载用户 execution。

Image mode 不接受完整镜像地址。Control Gateway 从部署配置 `environment_base_image` 提取固定 workload repository，将
`image_version` 解析为同一 repository 下的 `<repository>:<image_version>`，并把解析后的引用写入 immutable allocation snapshot；
用户不能切换 registry、namespace 或 image name。该受控镜像空间必须启用不可覆盖的版本 tag，版本一旦用于 submission 就不能
重新发布；节点继续使用 `IfNotPresent` 复用镜像缓存，避免大规模 Manager 启动时重复拉取。

`GET /v1/capabilities` 返回 image mode 是否启用、固定 workload repository 和当前 base image digest，供 CLI、页面及用户构建
镜像；不返回 pull credential。Image mode 仅在 production base image 已 digest-pinned 且 repository 可解析时启用。

用户版本必须以 capabilities 返回的 workload digest 为基础构建，保留 `/opt/nexus-venv/bin/python` 和 distributed bootstrap，
并提供只读 `/workspace`；项目源码可以放在该目录，也可以安装进 `/opt/nexus-venv`。平台忽略用户镜像的 ENTRYPOINT/CMD，仍以
固定 bootstrap 命令启动；可写目录只有受限的 `/data` 和 `/tmp`。MVP 不访问 registry 做 ancestry 或 attestation 检查，兼容性
由受控发布空间、固定构建规范以及 Manager 启动后的 capability 注册共同保证。

每个 dynamic source-package submission 在 DB 创建或复用一个 `DependencyBuild`。build key 是 owner、Environment、三个输入的
kind/sha256/size、有效 source-package limits、规范化 extras、Python ABI、平台、workload image 和 builder image 的 canonical
hash，不包含 `artifact_id`；只在 owner、Environment 和完整构建配方相同时复用内容相同的输入。状态机为：

```text
requested -> building -> ready
requested <- failed
building -> failed | cancelled
```

`DependencyBuildController` 是 Reconciler 进程内的普通类，使用 DB lease/fencing，并通过对应 Environment provider 创建一次性
Kubernetes Job 或 TKEx Job。Job 使用 build-scoped content ticket 下载 `source.tar.zst`、`nexus-agent.whl` 和 `uv.lock`，执行
frozen lock export 与 `pip wheel --require-hashes`，再生成确定性 `wheelhouse.tar.zst`。Builder 和后续 Manager 从同一个 immutable
build/allocation snapshot 读取有效 source-package limits，并用同一组成员数、单文件、总解压和 archive 上限解包源码。

Application Gateway 对公开上传的 `nexus_wheel` 只做认证、准入限流和整个正文的 size/sha256 校验，不打开 wheel。受 CPU、内存、
workspace 和 deadline 限制的 Builder 对 wheel 做唯一一次有界语义检查：限制 entry 数、单 entry 和总声明大小，以有界读取解析
唯一且一致的 `.dist-info/METADATA` 和 `WHEEL`；`METADATA` 上限为 4 MiB，`WHEEL` 上限为 64 KiB，均读取 `limit + 1` 判断越界。
检查拒绝重复、加密、路径不安全、非普通成员或歧义 dist-info，并一次提取 Name、Version、Tags 和 Extras 供后续版本、文件名和
extras 校验复用，不重复打开 wheel。无效 source 或 wheel 分别以
`dependency_source_package_invalid`、`dependency_nexus_wheel_invalid` 永久失败，不重试 Builder Job。

Wheelhouse 只包含 `wheels/*.whl`，不内嵌 manifest、逐文件 size/hash 或文件列表。DB 只保存 `schema_version`、`build_key` 和
`nexus_wheel_path` 等小元数据；可复用现有 `manifest_json` 列承载这些数据，不新增 migration。Builder 上传时只通过受控 header
提交安全的 `nexus_wheel_path`，Application Gateway 从 build-scoped ticket 和 DB 取得可信 build key，校验整个正文
size/sha256 后原子链接 Artifact，不解压 wheelhouse。Controller 只有在 Job 成功且 output 已 staged 时才把 build 标记为
`ready`。

Builder 的 Python index 由部署侧 `dependencyBuild.builder.indexUrl` 统一配置，默认
`https://mirrors.tencent.com/pypi/simple/`，并作为 `PIP_INDEX_URL`、`UV_INDEX_URL` 注入 Job。该值必须是 HTTP(S) URL；
submission 不能覆盖，也不支持 extra index 混源。

single/live 的 execution allocation 和 batch 的 parent/child allocation 都绑定同一个 build。Build 未 ready 时 allocation 保持
`requested`，不会创建 Manager 资源；build 终态失败则 execution fail close。Builder Job 使用独立 CPU、memory、workspace、
deadline 和 retry 配置，不占 Manager slot，也不进入 Scheduler。

### Manager Venv 和 Runtime Worker Venv

两类 venv 都复用现有 `nexus.runtime.server.venv.manager.VenvManager`、`VenvSpec` 和 `ArtifactSpec`，但它们是不同位置、
不同生命周期的 venv：

- Manager venv：dynamic Pod bootstrap 创建，用提交 wheel 启动对应版本的 InstanceManager。
- runtime worker venv：InstanceManager 执行 `runtime_configs.<name>.worker_venv` 时，在 runtime sandbox 内创建。

Dynamic source-package 启动流程：

1. Reconciler 只在 `DependencyBuild=ready` 后把 wheelhouse ref 和小元数据注入 allocation 启动快照。
2. bootstrap 使用 allocation-scoped ticket 下载 `source.tar.zst` 和 `wheelhouse.tar.zst`，不再下载 lockfile 或联网解析依赖。
3. submission 创建时已将 Environment、base image、extras 和三个输入 ArtifactRef 绑定进 build key；bootstrap 校验启动快照中的
   build key 与 wheelhouse 小元数据一致。HTTP client 对下载正文做流式 size/sha256 校验，不重复扫描已验证的本地文件。
4. source package 按上述规则安全解包；wheelhouse 只允许 `wheels/` 下的普通 `.whl`，并校验安全路径、重复成员、成员数和总大小，
   不逐个计算 wheel hash。
5. VenvManager 要求 DB 指定的 `nexus_wheel_path` 存在，并以流式 hash 与原始 `nexus_wheel` ArtifactRef 做一次内容绑定；Manager
   不再重复执行 wheel 语义检查。
6. VenvManager 使用 `--no-index --find-links <wheelhouse>` 创建 Manager venv；缺 wheel 立即失败，不回退网络或基础 venv symlink。
7. workspace 作为 cwd，并放在 `PYTHONPATH` 首位，随后 `exec` 启动该 venv 中的 InstanceManager。

`config_only` 直接使用基础镜像中的 Nexus，不创建 Manager venv。Python 版本、基础依赖、index/CA/proxy 由 Environment
管理员配置。Runtime `worker_venv` 继续复用同一个 VenvManager，但属于 runtime sandbox 内的另一个生命周期；cloud submit
不把它暴露为通用依赖入口，也不复用 Manager wheelhouse 的状态目录。

`image` 同样不创建 Manager venv。bootstrap 校验 `/workspace` 存在，以其作为 cwd 和 `PYTHONPATH` 首项，并直接使用镜像内的
`/opt/nexus-venv/bin/python` 启动 InstanceManager。Single/Live 的 imports 必须能从 `/workspace` 或该 venv 导入；Batch remote
generator 的安全相对路径以 `/workspace` 为根。Batch local generation 仍在客户端生成并上传 task list，不要求镜像包含 generator。

Source-package 模式当前只面向可信内部用户，因为提交 wheel 会在 Manager Pod 内执行。非可信代码隔离需要未来改为稳定
Manager 加独立 worker，不属于本期；image 模式具有相同信任级别。

## Artifact

### ArtifactStore

MVP 实现两个 backend：

```text
local
generic_repository
```

`local` 只用于本地开发和隔离测试环境，使用 Application Gateway 本地目录，不提供跨 Pod HA 或持久化承诺；生产 Environment
必须配置 `generic_repository`。

Artifact kind 只有：

```text
source_package
wheel
nexus_wheel
uv_lock
wheelhouse
task_list
run_result
```

Dataset 由 Dataset Service 提供，不是 Artifact kind。

Generic Repository backend 使用：

```text
PUT/GET https://mirrors.tencent.com/repository/generic/{repo}/{path}/{filename}
```

object key 由 `artifact_id` 和 kind 生成，用户不能提供任意远端路径。Repository username/token 只从平台密钥配置读取。

Control Gateway 创建 artifact metadata 和 scoped upload/read credential。CLI、bootstrap 和 Manager 均通过 Application Gateway
上传或下载正文；它们不直接访问 backend。Application Gateway 校验权限和整个 Artifact 的 size/sha256 后读写 ArtifactStore，
再以 CAS 将 artifact 标记为 `ready`；它不承担 source、wheel 或 wheelhouse 的语义校验。正文先落临时 spool 并完成流式校验，
再进入进程内上传协调器：input 与 `run_result` 分别限流并共享总并发，input 保留固定容量；每个 Gateway 默认
total/input/result 为 12/4/8。backend 的 408/425/429/5xx 在同一 upload lease 内最多指数退避重试 8 次，重试耗尽后才记录
upload 失败。Manager 下载和结果上传对临时网络错误使用有界指数退避、full jitter，并尊重 Gateway 的 `Retry-After`；下载的全部
attempt、backoff 和 Retry-After 同时受 provisioning、build 或 generation 操作的绝对 deadline 限制。

Application Gateway 在同一 Pod 的 spool 下按 `sha256 + size_bytes` 缓存不可变输入 `source_package`、`nexus_wheel`、`uv_lock`、
`wheelhouse` 和 `task_list`。每个请求都先完成 ticket、owner/Environment 和 ArtifactRef scope 校验，再查缓存；同内容并发 miss
只执行一次 backend GET，流式发送期间条目保持 pinned，上传和缓存按同一逻辑容量做 LRU 淘汰。缓存只在单个 Gateway 进程及其
本地 volume 内复用，不要求副本间一致或共享；`run_result` 始终直读 backend，不进入缓存。

完整性校验按信任边界各执行一次：producer 在生成时计算 identity；Gateway 接收请求并写 spool 时流式校验；`ArtifactStore.put`
信任已验证 spool，只做廉价 size 检查，Generic Repository PUT 后保留 HEAD size 作为远端提交确认；`ArtifactStore.get` 在写本地
spool 或首次填充输入缓存时对 backend 字节流校验预期 size/sha256；Gateway 直接发送已验证内容；HTTP client 再对收到的正文
流式校验。local backend 遇到已存在对象时仍校验 hash 以识别幂等冲突。Builder 不再二次校验下载文件，maintenance recovery
复用 `ArtifactStore.get` 的校验结果。

Gateway 因此不保留独立的 archive validation executor、等待队列或 archive validation deadline；archive 的路径、类型和解压
资源限制只在实际使用内容的 Builder/bootstrap 安全解包器中执行。Artifact ticket 的 owner、Environment、assignment、attempt
scope、fencing 和 DB ref 校验保持不变。

Task list 仍校验 record version、item count、manifest 与 ArtifactRef 绑定，run result 仍校验小型 JSON schema；Builder 的
source/wheel 版本、extras、`uv export`、`pip wheel` 和离线安装/import/version 验证也保持不变。此次调整不改变 Artifact kind、
submission payload、`ArtifactRef`、backend object model 或数据库 schema。

凭证拆分为：

- CLI、bootstrap、InstanceManager、ArtifactTraceSink/Source：只调用 Application Gateway，不持有 backend credential。
- Application Gateway：持有 Generic Repository reader/writer credential，并提供 local backend 的唯一远端读写入口。
- Reconciler/maintenance：持有清理 credential。

CLI upload ticket 短期有效。Builder 使用 build-scoped credential，只能读取该 build 的三个输入并写对应 wheelhouse；
Dynamic bootstrap 使用 Reconciler 创建并挂载的 allocation-scoped read credential，只允许读取该 allocation 引用的
source/wheelhouse/task-list；credential 在 allocation 生命周期内可轮转，StatefulSet 替代 Pod 总能挂载当前
版本，并在 allocation released 时撤销。Manager 的 internal read 还要按 DB ref 校验 execution/batch。Credential 不写入
execution spec 或 task env。

ArtifactRef 对用户只暴露 `artifact_id`、`kind`、`sha256` 和 `size_bytes`，不暴露 backend object key 或凭证。

### ArtifactTraceSink / ArtifactTraceSource

`ArtifactTraceSink` 接入标准 run trace sink 管线，但只提取最终 `run.output`、terminal status 和 error，生成不可变
`run_result`。它不保存完整 span、event 或 log。

`ArtifactTraceSource` 只提供 `read_result()`，不提供 event replay。Live SSE 在 Manager 存活期间继续读取 Local Run Trace；
dynamic Manager 释放后，closed live events 返回明确的 `trace_events_not_retained`。

成功 execution 必须在 `run_result` ready 后才能转为 `succeeded`。Failed/killed/lost/expired 可以只保存 DB 中的 status/error，
不强制要求 result artifact。Batch UI 按需读取 child result，parent finalize 不下载全部 child 输出。
临时上传错误使用同一 producer key 普通重试；不可恢复错误收敛为 `failed/ResultPersistenceFailed`，execution TTL 到期仍为
`expired`。成功结果持久化默认允许在 execution deadline 内重试 30 分钟；超过独立 publish deadline 后才映射为
`ResultPersistenceFailed`，避免大规模结果突发被短任务 timeout 误杀。

### TDSQLTraceSink / TDSQLTraceSource

完整 raw `SpanEventRecord` backend 命名为 `nexus.runner.trace.tdsql`，公开类型为 `TDSQLTraceSink`、`TDSQLTraceSource` 和
`TDSQLTraceWriter`；不保留 `cloud` 包、类名、环境变量或日志前缀。single/live 继续写共享表
`trajectory_span_events`；batch 及其全部 child/retry 写入 owner principal 在该 Trace 数据库中的唯一物理表：

```text
trajectory_span_events_pr_<principal_hash>
```

`principal_hash` 固定为
`sha256(b"nexus-tdsql-principal-table:v1\0" + owner_principal_id.encode("utf-8")).hexdigest()[:32]`；最终表名为 58 个 ASCII
字符，不包含 StaffId、用户名、token、Environment 或其他可识别明文。同一 principal 在同一个 Trace 数据库只使用一张 batch 表；
不同 backend 指向不同数据库时各自维护一张。表名必须由受控 helper 生成并通过严格格式校验，不能接受 submission 或 trace
metadata 提供的任意 identifier。

orchestration Manager 在 materialize 和提交 child 前，通过 Run Trace backend registry 的可选 scope preparation 能力传入可信
owner scope 并幂等预建 principal 表。该写操作由选中的 TDSQL sink 执行，source 保持只读；Local 和 Artifact backend 不受影响。
并发首批 batch 可以同时执行 `CREATE TABLE IF NOT EXISTS`。预建失败使 parent fail close，child 不建表，也不能静默写入共享表。

持久化 execution 的 `owner_principal_id` 和 `parent_execution_id` 是权威来源；admin 代提交仍按 owner 而不是 actor 路由。child spec
中的 parent 副本只用于一致性校验。新 metadata scope 固定为 `batch_principal_v1`，只携带受控 `principal_scope_id`、
`parent_execution_id`、run 和 run span identity，不暴露原始 principal。single/live 继续使用 `shared` scope。

Principal batch 表删除 `trajectory_group_id`，每条记录增加 `parent_execution_id`。它只是 batch 的逻辑过滤键，不创建 MySQL 原生
partition。表使用 child `run_id` 作为 TDSQL `shardkey`，唯一键为 `(run_id, run_span_id, record_id)`，并建立
`(parent_execution_id, time_unix_nano, run_id, run_span_id, record_id)` 索引。上述 identity 列使用与系统 ID 上限一致的有界 ASCII
类型。同一 batch 的 child 和重试按 parent 查询，但仍分散到不同 run shard，避免万级 batch 形成单分片热点；批量读取按同一复合
顺序使用稳定 cursor，不使用大 OFFSET。

Writer 继续按数据库配置共享一个队列和连接，不按 principal 或表创建 writer；写请求携带受控 principal table key，并按表和 SQL
合并 `executemany`。交错 principal 写入按表独立提交，单表失败只标记对应 run。Batch 终态不删除 principal 表；retention 按
`parent_execution_id` 分批删除过期记录，只有 principal 注销且表内没有保留数据时才允许 maintenance 删除空表。表数量、建表失败
和数据库表容量纳入系统监控。

### TDSQL 只读访问（MVP）

启用 TDSQL Sink/Source 的 Environment 可以向用户提供自助只读账号。每个 principal 使用独立数据库用户名和密码，并拥有自己的
batch 物理表；但首期账号权限仍统一为独立 Trace 数据库全部表只读，因此 principal 表只是存储边界，不是安全隔离边界。账号也能
读取共享 single/live 表和其他 principal 表。Trace 数据库必须与 Distributed 控制数据库使用不同存储目标；write endpoint 和
schema 都相同时禁止开启用户只读访问，相同 schema 名位于不同数据库实例时不误判为同一目标。

账号按 `(owner_principal_id, backend_name)` 唯一，用户名由 principal 和 backend 的稳定 hash 派生，不包含 StaffId、submit token
或其他可识别信息。只有对使用该 backend 的 Environment 持有有效 Grant 的用户才能创建或轮转自己的账号。首次获取凭据时，
Control Gateway 通过独立 maintenance credential 幂等创建数据库用户、设置随机密码并执行：

```sql
GRANT SELECT ON `<trace_database>`.* TO '<readonly_user>'@'%';
```

数据库级 Grant 自动覆盖后续创建的 principal 表，因此 batch 执行链路不创建用户、不生成密码，也不逐表 Grant。账号可以先于或
晚于 principal 表创建，两者生命周期独立。轮转只更新该账号密码并保持现有 Grant；管理员修复操作重新校准只读 Grant。管理员
可以禁用、恢复、强制失效和修复账号，不能查询已有密码；管理员 reset 成功时只能在当前响应中查看一次新密码。
principal 被禁用或失去最后一个使用该 backend 的有效 Environment Grant 时，Control Gateway 必须先禁用数据库账号再完成访问
回收。禁用 Environment 也按同一规则处理其全部 Grant：先锁定所有没有其他 enabled Environment 的账号，再提交 Environment
状态；TDSQL 不可用时 Environment/Grant 保持 enabled。跨多个账号的数据库 DDL 无法形成原子事务，部分成功时允许已锁账号保持
disabled，但不能先撤销 Nexus 访问；管理员修复后重试禁用。Restore 只解锁现有账号且不返回密码；需要更换或遗失密码时，由用户
或管理员显式 reset 后一次性返回新密码。

Nexus 只保存 `owner_principal_id`、backend、用户名、状态、credential version 和创建/轮转/禁用时间，不保存密码明文、hash 或
可逆密文。随机密码只存在于创建或轮转请求内存，并在数据库用户和 Grant 全部成功后返回一次；响应设置
`Cache-Control: no-store`，前端不写 URL、日志、localStorage 或持久状态。响应丢失或用户遗失密码时再次轮转。账号操作按
principal/backend 串行化且可重试，部分失败记录明确状态，不能返回尚未生效的密码。

账号行不存在时页面状态为 `not_created`；持久状态只保留 `ready`、`disabled` 和 `error`。Create、reset、disable、restore 和
reconcile 先在控制库中通过 `operation_id + operation_expires_at` 获取带 fencing 的短 lease，再执行 TDSQL DDL，最后以同一个
operation ID 提交状态。并发操作返回 conflict，过期 lease 可由后续请求接管；旧请求即使晚返回也不能覆盖新状态。TDSQL DDL 成功
前不返回密码，控制库终态提交失败时不声称操作成功，后续 reconcile/reset 负责收敛。

### 生命周期

Artifact ref 生命周期跟随 execution/batch：

- execution 存在时不按 TTL 删除 ready artifact。
- 普通用户没有 execution DELETE。
- 未来 admin execution purge 同时删除 execution、ledger、refs 和无引用对象。
- 本期只自动清理 incomplete upload 和 ready-but-unreferenced orphan。

因此 MVP 存储会持续增长到 admin purge 上线，这是显式接受的限制。

## Submit API 和 CLI

三个公开接口：

```text
POST /v1/submissions/single
POST /v1/submissions/live
POST /v1/submissions/batch
```

现有通用 `POST /v1/executions` 暂时保留，但只允许 internal/admin 使用。Gateway 根据 submission type 生成内部
`mode/spec_type/instance_mode/workload/placement`，用户不能设置 priority、target Manager、内部 driver config 或任意
spec type。

### Dynamic single/live resources

```json
{
  "resources": {
    "managers": {
      "count": 1,
      "cpu_cores": 8,
      "memory_gib": 16,
      "slots": 1
    }
  }
}
```

Gateway 强制 `count=1`、`slots=1`。CLI 只暴露：

```text
--manager-cpu-cores
--manager-memory-gib
```

### Dynamic batch resources

```json
{
  "resources": {
    "batch_manager": {
      "cpu_cores": 4,
      "memory_gib": 8
    },
    "managers": {
      "count": 25,
      "cpu_cores": 1,
      "memory_gib": 1,
      "slots": 20
    }
  },
  "batch": {
    "max_inflight": 500,
    "retry_failed": 0,
    "fail_fast": false
  }
}
```

`batch_manager` 的 count 和 slots 固定为 1。所有资源一起计入上限：

```text
manager_count = 1 + managers.count
cpu           = batch_manager.cpu + managers.count * managers.cpu
memory        = batch_manager.memory + managers.count * managers.memory
slots         = 1 + managers.count * managers.slots
```

新请求应显式提供 `max_inflight`。`managers.count` 和 `managers.slots` 都省略时，以 20 slots 推导 count；只提供其中一个时，以
`ceil(max_inflight / provided)` 推导另一个；同时提供时保留原值，并要求 `max_inflight <= count * slots`。为兼容
submission v1，完整 Manager 容量省略 `max_inflight` 时以容量作为在途上限；CLI 三项都省略时使用 1 个 Manager、20 slots。
`submit_batch_size` 是内部参数，默认 `min(50, max_inflight)`，不暴露给用户。

Dynamic batch CLI 使用：

```text
--batch-manager-cpu-cores
--batch-manager-memory-gib
--manager-count
--manager-cpu-cores
--manager-memory-gib
--manager-slots
--max-inflight
```

### Predefined submit

Predefined 只选择现有 pool：

```text
--environment
--pool
--config-only
```

它拒绝 CPU/memory/count/slots 和 source-package 参数。Dynamic 拒绝用户指定 pool；服务端为 allocation 生成独占 pool id。
Batch 的用户 pool 表示 child execution pool，parent orchestration pool 由 Environment 配置。

Predefined batch 不推导 Manager；显式 `max_inflight` 不能超过 Environment 单 batch 上限、可用 slots 和该 principal Grant
并发额度的有效较小值。submission v1 省略时使用该有效较小值。

Predefined 不预留 Manager/CPU/memory，因为物理资源已由管理员创建；它只记并发使用量：single/live 预留 1 个
`execution` slot，batch 预留 1 个 active batch、1 个 `orchestration` slot 和 `max_inflight` 个 `execution` slots。
这些 slots 在 top-level admission 时一次性预留，创建 child 时不重复计数，在 top-level execution 终态并释放逻辑
allocation 后归还。

### Timeout

Single/live 可提交：

```text
provisioning_timeout_seconds
queue_ttl_seconds
execution_ttl_seconds
stop_timeout_seconds
```

Batch 另外支持：

```text
generation.timeout_seconds
batch.child_queue_ttl_seconds
batch.timeout_per_run
```

省略时使用 Environment 默认值；显式 `null` 不能绕过默认值。值必须为正数并受 Environment/Grant 最大值约束。
最大值只使用 `max_<field>` canonical 字段，不接受别名。
Live 可以不设 execution TTL。Batch top-level execution TTL 覆盖整个 parent 生命周期；child task 同时存在
`runner.timeout` 和 `batch.timeout_per_run` 时取较短者。

`queue_ttl` 从 allocation ready、execution 第一次可调度时开始，不从 submission 创建时开始。`provisioning_timeout`
从 allocation 创建时开始。Execution TTL 到期必须收敛为 `expired`，不能被 stop 流程错误写成 `killed`。

### 返回和等待

Gateway 在 top-level execution、allocation、资源预留和 idempotency 信息持久化后立即返回 `202 Accepted`，不等待
Kubernetes/TKEx 或 Manager heartbeat：

```json
{
  "execution_id": "exec-123",
  "batch_id": "batch-123",
  "status": "queued",
  "phase": "waiting_resources",
  "reused": false,
  "control_url": "...",
  "result_url": "...",
  "task_web_ui": "...",
  "environment_web_ui": "...",
  "tracedb_web_ui": "...",
  "events_url": "...",
  "command_url": "..."
}
```

`control_url` 和 `result_url` 对 single/live/batch 都存在；live 额外返回 `events_url`、`command_url`，batch 额外返回
`batch_url`。`task_web_ui` 按类型指向 execution 或 batch 详情页，`environment_web_ui` 和 `tracedb_web_ui` 指向用户的
Environment 页面。API 和 Web URL 分别使用部署侧 canonical public base URL，不从反向代理的请求 Host 推断。
这些字段都指向已注册路由，不返回抽象的 Application 根 URL。资源未就绪时，路由返回明确的状态错误。

CLI 行为：

- `nexus submit single` 默认立即返回，可选 `--wait`。
- `nexus submit batch` 默认立即返回，可选 `--wait`。
- `nexus submit live` 立即返回 execution、result、events 和 command URL，不提供 `--wait`。
- `--wait` 的 Ctrl-C 只停止本地等待，不停止远端任务；远端停止必须显式调用 stop。
- `nexus run live` 保留；`nexus live-server` 和 `nexus live-runner` 删除。

## Batch Generation

无论 generation 在本地还是云端运行，cloud batch 的 dataset 输入都只接受 Dataset Service URI；本地 dataset path 仍可用于
本地 `nexus run`，不能直接提交到云端。Dataset alias/latest 在生成 task list 时解析成不可变版本并写入 manifest。
真实 Dataset 所需的 Wedata key 由用户通过可选的 `submission.data_credentials.wedata` 提交；CLI 只接受
`--wedata-key-file`，不接受命令行明文。Control Gateway 在创建 top-level execution 的事务中把凭据从公开 submission spec
移入 `distributed_execution_credentials`，公开 API 只返回 credential kind。Batch generation 和远端 `task_files` 签名按
execution scope 读取该凭据；batch child 复用 parent credential，不复制到 task list、Manager 全局环境变量或共享文件。

Batch 不是通过新的 run kind 识别 compose。CLI 传 `--generator` 时使用 generator；否则按顺序解析 `--data`、`--use`、
`--set`、`--set-json` 等 compose parts，并在 submission 中保存结构化 ordered parts。两者最终都输出同一 task-list 格式。

支持矩阵：

| Generation | Predefined config-only | Dynamic config-only | Dynamic source-package |
| --- | --- | --- | --- |
| 本地 generator | 支持 | 支持 | 支持 |
| 本地 compose | 支持 | 支持 | 支持 |
| 云端 generator 文件 | 不支持 | 不支持 | 支持 |
| 云端 compose + cloud dataset | 支持 | 支持 | 支持 |

Config-only 的云端 compose 依赖组件已安装在镜像中。本期不在 submit 前做 compatibility preflight；缺少组件时
execution 明确失败。

### Task-list schema

每行是一个 versioned record：

```json
{"version": 1, "item_id": "item-000000000001", "run_input": {"task_name": "...", "task_config": {}}}
{"version": 2, "item_id": "item-000000000001", "run_input": {"task_name": "...", "task_config": {}}, "metadata": {}}
```

- `run_input` 只描述 child `nexus_single` 的业务配置，不包含 owner、placement、allocation、resource 或内部 driver 字段。
- v1 不接受 item metadata；v2 接受最大 64 KiB 的 JSON object item metadata，并将其纳入 canonical line、Artifact sha256、
  generation 非确定性检查和 ledger。两个版本都不支持 batch metadata 或 conflict groups，遇到非空值时 fail close。
- Materializer 优先使用 source 中的稳定 item id；没有时按 canonical ordinal 生成固定宽度 id。
- `(task_list, item_id)` 必须唯一；重复或空 id 使 generation 失败。
- Manifest 保存 schema version、item count、size、sha256、generation source 和已解析的 immutable dataset versions。
- Parent 在合并服务端控制的 owner/code/timeout/placement 后 canonicalize 完整 child spec，并计算 `spec_hash` 写入 ledger；
  不信任 JSONL 提供的 hash。

### 本地生成

1. CLI 在本地调用现有 generator/compose loader。
2. generation DB 写入 `generation_complete` 后导出规范化 `tasks.jsonl`。
3. CLI 上传 immutable `task_list` Artifact。
4. Submit 创建 batch parent，parent 从 Artifact 开始物化。

### 云端生成

1. Parent allocation ready 后，`nexus_batch` 在 source workspace 内启动 generator/compose。
2. 输出先写 allocation `emptyDir` 中的 `tasks.jsonl.tmp`。
3. 生成过程中强制 item count、单行 bytes 和总 bytes；完成后计算 sha256，fsync 并原子 rename 为 `tasks.jsonl`。
4. Parent 在当前 assignment/attempt fencing 下先把 manifest、sha256、size 和 item count 暂存到 DB，状态为 `publishing`；同一
   generation epoch 重试得到不同 manifest 时以 `batch_generation_nondeterministic` 失败。
5. Parent 经 Application Gateway 上传 immutable `task_list`。Application Gateway 在同一 DB transaction 中将 Artifact 标记
   `ready`、写 batch/allocation ArtifactRef，并把 staged manifest 与 ArtifactRef 链接到 batch。
6. Parent 分块写 ledger 和 byte cursor；全部 item 写完后才把 generation 标记为 `complete`。

JSONL 使用 canonical UTF-8、LF、无压缩格式，ledger 保存 byte offset/length。Artifact ready 前 Pod 丢失可以重跑
generator，但必须与已 staged manifest 完全一致。通用 Artifact maintenance 不会独立把未链接的 `task_list` 标记 ready；
上传完成后的幂等 Application Gateway 重试负责原子链接。Artifact ready 后只从 ArtifactStore 恢复，不再重跑 generator。
Generation timeout 时终止整个 subprocess tree。Dynamic workload 的 `emptyDir.sizeLimit` 和 predefined Manager 的 cache volume
上限负责运行期磁盘硬边界；Manager root filesystem 只读，受限 cache 卷同时挂载为 cache root 和 `/tmp`。生成结束后还要
校验本 batch 的实际目录大小。磁盘、workspace、单行或总输出大小超限时均不上传 task-list，也不写 generation complete 或
ready pointer。每个 batch assignment 退出时删除自己的本地目录；ready Artifact 和 DB ledger 是后续恢复事实源。

## Allocation 和调度

### Allocation 状态

Dynamic allocation：

```text
requested -> provisioning -> ready -> draining -> deleting -> released
```

失败和取消也通过 `deleting -> released` 收敛。Predefined 使用轻量状态：

```text
requested -> ready -> released
```

Allocation 与 execution/Manager 状态独立：

- single/live：一个 execution allocation。
- batch：一个 parent orchestration allocation 加一个 child execution allocation。
- allocation 由 top-level execution 持有，batch child 不单独申请资源。
- 用户 stop top-level execution，不直接删除 allocation。
- 只有管理员可以对 stuck allocation 执行带审计的 force cleanup。

绑定 `DependencyBuild` 的 source-package allocation 在进入 provider 状态机前先等待 build `ready`，因此完整顺序是：

```text
dependency requested/building -> dependency ready -> allocation provisioning -> allocation ready
```

build 失败不会申请 Stateful workload，也不会预留额外一份 compute quota；已绑定的所有 allocation 读取同一个 build 终态。
Control API 在 allocation 中返回 dependency build 的 id、status、attempt 和时间；管理员额外看到 provider Job 和错误信息。

Dynamic resource 只在 provider 已确认删除、allocation 进入 `released` 后释放额度。Provisioning 期间收到 stop，也必须删除
已创建的 provider 对象。Predefined release 只释放逻辑引用，不删除现有 pool。
Provisioning timeout 或不可恢复错误将 top-level execution 标记为 `failed/EnvironmentProvisioningFailed`，随后进入相同清理流程。

Environment 和 Grant 创建时在同一事务预建零值 resource usage 行。Reserve/release 先锁 Allocation，并根据
`resources_reserved_at/resources_released_at` 直接结束幂等重放；需要变更额度时，再用一次按主键排序的
`SELECT ... FOR UPDATE` 锁住 Environment/Grant usage。热路径随后只读 Environment/Grant 配置，不加配置行写锁。
管理员修改配置时按 usage、配置行的顺序串行；资源上限不得降到当前 usage 以下，但允许在仍有 usage 时禁用
Environment 或 Grant。MVP 保留每个 `(environment_id, principal_id)` 的精确串行额度边界，不增加跨 Allocation delta
聚合状态机。

### Provider Workload

Kubernetes allocation 创建 `apps/v1 StatefulSet`，TKEx 创建语义一致的 StatefulSetPlus；不创建 Service。每个 Pod 运行
一个 InstanceManager。Stateful workload 只用稳定 ordinal 生成 Manager id，Gateway 和 Scheduler 不使用集群 DNS。
升级前已存在且仍依赖 headless Service 的 allocation 保留原 Service，不原地修改不可变的 `serviceName`；allocation 释放时再
随旧 workload 删除。新 allocation 只走 PodIP。
Dynamic Manager 不创建 PVC，Source workspace、Manager venv/cache、generation 临时 JSONL 和 Local Run Trace 位于
挂载到 `/data` 的 `emptyDir`，并按 Environment 的 `max_workspace_bytes` 设置 `sizeLimit`。Predefined Manager 由 chart
为每个 Pod 创建独立的 RWO PVC 并挂载到 `/data`，使预建环境及缓存可跨 Pod 重启保留；关闭 persistence 时才退回受
`cacheSizeLimit` 限制的 `emptyDir`。BatchControl 的临时目录固定放在 `/data` 下的 `batch-control/`，assignment 退出即回收。
两类容器的 root filesystem 都只读，`/tmp` 使用独立、受限的 `emptyDir`。

私有 workload/builder 镜像的 pull Secret 名由部署侧统一配置，并注入 Manager 和 builder Pod；同名 Secret 必须预先存在于
每个目标 cluster 的 workload namespace。Config-only/source-package 使用部署侧 workload digest；image submission 只提交同一
workload repository 下的版本，不能传完整 image ref 或 registry credential。

Dependency builder 是短生命周期 Job，不是 InstanceManager，也不注册 heartbeat。Job 使用 builder image、独立受限
`emptyDir` 和 build-scoped Secret，禁用 ServiceAccount token、Service links、提权与可写 root filesystem。Kubernetes 与 TKEx
client 都提供 Job/Secret 的 get/create/delete；Controller 按 DB attempt 生成稳定资源名并在重试或终态后清理旧资源。删除 Job
和 StatefulSet 时显式使用 background propagation，级联回收其 Pod；集群验收等待全部 Nexus-managed Pod 消失。

Dynamic Manager id 由 `allocation_id + pod ordinal` 稳定生成；predefined Manager id 使用 StatefulSet Pod 名。进程每次启动生成
新 `session_id`。两类 Manager 都通过 Downward API 读取 PodIP，并在 internal HTTP 监听成功后注册：

```text
environment_id
pool_id
allocation_id
workload
allocation_spec_hash
endpoint_url = http://<pod_ip>:<manager_port>
capacity_total
supported spec/mode
capability snapshot
metadata_json(endpoint_kind, provider_config_ref, cluster_name, namespace, pod_name, pod_uid)
```

`endpoint_url` 只能由受控 workload 的 PodIP 或管理员启动参数生成，用户 submission 不能覆盖。Control/Application Gateway、Scheduler
所在网络必须能路由到所有 Manager Pod CIDR，并在防火墙或 NetworkPolicy 中放行 Manager 端口；本期不增加 CLB、Relay 或 mesh。
Capability snapshot 和 `metadata_json` 只在 `register_session` 上报一次。注册快照保存在 `instance_managers`；
heartbeat 只条件更新 `instance_manager_liveness` 的 session、容量和时间，不锁注册快照，也不携带 `local_instances`。
`local_instances` 由低频 snapshot（默认 15 秒）单独更新。Pod 重建后保持 Manager id，使用新 session 原子刷新两张表；
旧 session 或已 fenced session 的 heartbeat 条件更新失败后立即退出。

默认时序为 heartbeat 5 秒、healthy 15 秒、自我 fencing 30 秒、dead 45 秒，必须严格递增。数据库连续不可用达到
self-fence timeout 时，Manager 停止接收新 assignment、停止本地任务并退出进程，由 workload controller 重启；
Reconciler 在 dead TTL 后按现有 retry/lost 规则收敛。
新 capability 字段本期只供管理员查看，不做 admission preflight；已有 supported spec/workload 检查继续强制。

workload 访问 Artifact API 使用普通部署配置 `application_internal_base_url`；同集群配置为 Application Gateway Service DNS，
跨集群配置为已有内网 CLB 的稳定域名。Reconciler 将同一个解析结果注入 dynamic Manager 和 dependency builder，submission 与
Environment 不能覆盖。`application_public_base_url` 仅用于浏览器、CLI 和客户端上传，两者语义独立。
Service 的 LoadBalancer annotation 和流量策略由 Helm values 管理，EnvironmentController 不负责创建、发现或回收 CLB。

### Ready 条件

Provider replicas ready 不是 allocation ready。Reconciler 必须观察到全部期望 Manager 完成 heartbeat，并匹配
Environment、pool、allocation、workload、spec hash 和 session 后，才能把 allocation 标记为 ready。

Batch parent/child allocation 可以并行 provisioning：

- parent allocation ready 后可以开始 remote generation。
- child allocation 全部 ready 后，parent 才能创建 child execution。

运行中 Pod 丢失时 allocation 主状态保持 ready，Control API 返回派生字段 `degraded=true`，页面显示 degraded；allocator 创建
替代 Pod，Scheduler 按 lost/retry 规则处理受影响 assignment。

### Scheduler

Scheduler 只调度普通 execution，不理解 batch 编排。Dynamic execution 必须精确匹配 `allocation_id`；Predefined execution
匹配 Environment/pool。只有 ready allocation 可被 claim。

Scheduler 使用 `shard_leases` 中的 `execution-scheduler` group；owner id 包含进程启动 session，旧进程不能复用新进程
lease。lease 由独立 heartbeat 协程使用短事务续租，assignment 事务阻塞时仍持续续约。`shard_count` 只允许增加；
`ownership_limit=0` 表示单副本最多持有全部 shard，水平扩展时应显式设置每副本上限。
首次发现时使用与真实 shard group 隔离的内部 discovery guard 补齐完整 shard 拓扑；后续只续租自己的行，并通过
`FOR UPDATE SKIP LOCKED` 抢占过期 shard。内部 guard 不计入对外状态和容量统计。

每个 assignment 批次只读一次
ready allocation 类型快照，唯一认领仍由 execution 和 Manager 的 `FOR UPDATE SKIP LOCKED` 保证；不得为每个 child 锁同一个
allocation 行。正常 allocation 只在 top-level 终态后离开 ready，此时已无可调度 execution；管理员强制清理与调度的竞态由
Manager session、assignment fencing 和 lost 收敛处理。

批次内一次性读取 demand 和 slot reservation，再用一次 `FOR UPDATE SKIP LOCKED` 批量锁定候选 Manager，并在锁后用一次
聚合查询复核 reservation。MySQL 连接统一使用 `READ COMMITTED`，保证取得 Manager 行锁后的聚合看到此前已提交的
reservation，不能使用旧快照超卖 slot。每个 Scheduler 副本的 Manager 锁预算为
`min(manager_lock_limit, ceil(candidate_managers * owned_shards / shard_count))`，避免少量 Manager 行被多个副本同时争抢；
锁竞争失败的候选不会在同一批次按 child 重试。assignment control events 使用已锁 execution 的 event sequence 批量写入；
数据库往返次数不能随 child 数线性增长。事务提交前按本批实际修改的 shard 对 lease 行执行最终
`FOR UPDATE SKIP LOCKED` fence；lease 行被 heartbeat 短暂占用时整批回滚并稍后重试，只有 owner 或过期状态不匹配时
才判定失去 ownership。

Scheduler scan/max assignments 是内部配置。500-slot DevCloud 和 TKEx 验收使用以下默认值：

```text
scheduler_scan_limit=250
scheduler_max_assignments=250
scheduler_manager_lock_limit=64
scheduler_manager_lock_contention_backoff=0.25s
scheduler_poll_interval=0.1s
batch_lifecycle_poll_interval=0.5s
batch_lifecycle_page_size=500
batch_submit_batch_size=2000
```

生产值由压测确定，不作为用户 submission 字段。
候选 Manager 仍有容量但 `SKIP LOCKED` 未取得行锁时，execution 记录 `manager_lock_contended`，按上述基准加
`1x-3x` 确定性抖动后重试；没有可运行 Manager 或 slot 确实耗尽时仍使用 `no_capacity_backoff=3s`。两类退避不能合并，
避免多 Scheduler 同时热重试，也避免一次瞬时锁冲突让短任务等待完整无容量退避。
Batch parent 本地填窗间隔默认 `0.1s`。中央 lifecycle 同步使用 Reconciler 内的独立 worker，默认每 `0.5s` 处理最多 500 个 item，
控制状态同步事务的锁范围和单次耗时。每次创建 child 的批量大小默认 2000，服务端按
`min(batch_submit_batch_size, max_inflight)` 固化到 parent spec，用较大的创建批次快速填满并发窗口。两者是独立边界，且都不由
用户提交。

### Reconciler

Reconciler 使用同一 `shard_leases` 表中的独立 `execution-reconciler` group。execution 修复、超时和 cleanup 只扫描当前
副本持有的 shard，并在每个短事务提交前执行最终 lease fence；全局 quota repair 只由 shard 0 owner 执行。BatchRun 固化
parent execution 的 `shard_id`，batch lifecycle worker 也只 claim 当前副本持有的 shard。

core loop 轮询 manager dead/session mismatch、pending/starting timeout、queue/TTL、active repair、stopping 和 cleanup；
每种操作每 tick 默认最多 2 个 batch，整个 core tick 默认 2 秒预算，每个 batch 独立提交。Environment allocation、
DependencyBuild、Artifact maintenance 和 Batch lifecycle 使用同一 Reconciler 进程内的固定 async worker（默认各 2 个），
依靠各自 operation lease/fencing 抢占，不新增进程、服务或消息队列。lease heartbeat 与这些 worker 独立。

## Batch Parent 和 Ledger

`nexus_batch` Driver 不直接访问 DB。InstanceManager 注入 `BatchControlClient`，由 Manager 侧 repository/service 执行
materialize 和 create child；中央 Reconciler 统一执行 child 状态同步、逻辑重试、fail-fast、停止补偿和 finalize。两者不能同时
扫描 child 状态，避免重复事务争抢 batch ledger。

`BatchControlClient.create_children()` 使用内部服务身份，不携带或复用用户凭据。每个 child 强制继承 parent 的
`owner_principal_id`、Environment、child pool/allocation 和 code/task-list refs，并记录 `created_by=system:batch-parent`；
parent spec 不能覆盖这些字段。创建前必须校验当前 parent assignment/attempt fencing，并以 CAS 确认 batch
`control_state=active`。

主流程：

1. 获取或生成 immutable task-list Artifact。
2. 每次物化 500-1000 个 item，一次批量查询已有 item，再在事务内写 ledger 和 cursor。
3. child allocation ready 后，按 `max_inflight` 填充 active window。初始窗口达到 `min(total_items, max_inflight)` 前，
   Reconciler 不锁 batch/parent 做 child 状态同步，避免短任务完成过快或同步事务竞争导致窗口永远填不满；stop/TTL 仍优先处理。
4. 每次最多创建 2000 个 `nexus_single` child execution。
5. Reconciler 使用 `SKIP LOCKED` 分页锁定 active batch item，再无锁读取对应 child execution 快照；只处理发生变化的 item，
   不占用 Scheduler 认领 execution 所需的行锁。根据 `retry_failed` 将 item 转回 pending，Manager 后续窗口填充创建新的
   logical attempt。
6. 所有 item 终态后由 Reconciler 计算 immutable summary；成功 parent 先通过 Application Gateway 发布 `run_result`，确认 Artifact ready
   且已链接 `result_artifact_id` 后再写 `succeeded`、释放 assignment。临时错误以同一 producer key 在后续 reconcile 重试，
   永久错误收敛为 `failed/ResultPersistenceFailed`。发布失败和终态写入都校验发起发布时的 assignment/attempt；旧 attempt 不能
   终结新 attempt。已经通过 fencing 链接的 ready Artifact 在 parent 重派后仍可完成 finalize。

Ledger 编排内存复杂度为 `O(materialize_chunk + max_inflight)`，不能加载完整 task list、完整 ledger 或全部 child result；
generator 为检测重复 ID 保留受 `max_items` 限制的 item-id set。

Ledger：

```text
distributed_batch_runs
  batch_id, parent_execution_id, generation_status, task_list_artifact_id
  generation_epoch, task_list_manifest_json, task_list_sha256
  total/materialized/submitted/active/terminal counters
  control_state, stop_reason, stop_requested_at, stop_cursor
  materialize_cursor_bytes, summary_version, last_progress_at

distributed_batch_items
  (batch_id, item_id) PK
  task_list_offset, task_list_length, spec_hash, status
  current_attempt_no, current_execution_id, status_version

distributed_batch_item_attempts
  (batch_id, item_id, attempt_no) PK
  execution_id UNIQUE, status, result_artifact_id, error_code, timestamps
```

页面和 parent 使用稳定 `item_id` cursor，page size 最大 500；不使用 offset 分页。运行期间使用汇总 counter，不持续执行
全量 `COUNT/GROUP BY`。Finalization 允许执行一次带索引的精确汇总并修复 counter 漂移。

Parent assignment/attempt 通过 fencing 写 ledger。旧 parent、重复 poll 和 callback 重放只能复用已存在的 child，不能覆盖
新状态。Parent 重启后从 task-list Artifact、materialize cursor 和 active ledger 恢复。

### Retry 和终态

基础设施重试和任务逻辑重试分离：

- single、batch parent、batch child 和尚未进入 running 的 live 默认 `max_attempts=2`，即首次执行加一次 infra retry。
- infra retry 只处理 Manager lost、start failed、未 ack 或启动超时。
- attempt 在 Scheduler 创建 assignment 时计数，未 ack 的 assignment 同样消耗 attempt 额度。
- 同一 execution 的 `run_id=execution_id`，infra retry 间保持稳定。
- task-list 的 `item_id` 只标识稳定 batch item；materialize 不把它写入 `run_input.run_id`。创建 child execution 后由
  distributed ledger/Driver 将 child `execution_id` 写为 `run_id`。
- batch logical retry 使用 `(item_id, attempt_no)` 标识；每次 logical retry 创建新的 child execution，因此获得新的
  `run_id`。同一 child 的 infra retry 继续复用该 `run_id`。
- task 代码是 at-least-once；可用 `ExecutionContext.run_id` 为外部副作用做幂等。
- assignment fencing 只接受当前 attempt 的状态和结果，不承诺任务代码 exactly-once。
- task failed 不消耗 infra retry；batch 通过 `retry_failed` 创建新 logical attempt。
- `killed` 不重试；`failed/expired` 按 batch logical retry 配置处理；infra attempts 耗尽后保留 `lost`。
- single、batch child 和 live 只允许启动阶段 infra retry；进入 running 后 Manager 丢失即收敛为 `lost`，batch 再按 logical retry
  策略决定是否新建 child。
- batch parent 是唯一允许在 running 后 infra retry 的执行类型。Task-list Artifact、generation epoch、materialize cursor 和
  child ledger 都已持久化，新 attempt 复用这些事实继续编排，不重跑 generator、不重复创建 child。

Stop 收敛：

1. 用户 stop、parent TTL 或 fail-fast 先将 batch CAS 为 `control_state=stopping` 并持久化 `stop_reason`；从此禁止继续
   materialize 或创建 child。
2. Parent Driver 分页对所有非终态 child，包括 queued/assigned/starting/running，调用现有 execution stop 路径。Stop request
   全部持久化后 Driver 即可退出，不等待 child 真正结束；parent execution 保持 `stopping`。
3. Parent 在 fan-out 中途退出或 Manager 丢失时，Reconciler 调用同一幂等 batch lifecycle 方法，从 `stop_cursor` 继续补发。
4. 用户 stop 的 active child 收敛为 `killed`，parent TTL 的 child 收敛为 `expired`；fail-fast 停止的 active child 为
   `killed`。尚未创建 execution 的 item 不再创建 child，分别计入 `cancelled` 或 `expired` summary counter；不为停止而
   强制物化剩余 task-list。
5. 全部已创建 child 终态后，Reconciler 写最终 summary，将 parent 收敛为对应终态，再释放 parent/child allocation。

为兼容现有 InstanceManager 的 stop 路径，`ExecutionDriver` 增加预先声明的 `IMMEDIATE|DEFERRED` stop disposition。
Single/live 使用 `IMMEDIATE`，保持 Driver 停止后由 Manager 写 `killed` 的现有行为；`nexus_batch` 使用 `DEFERRED`。
Manager 即使在 fan-out 中途达到 `stop_timeout`，也只完成 stop command、释放 assignment 和本地 slot，保持 parent execution 为
`stopping`，不写终态；Reconciler 从 `stop_cursor` 继续。Parent assignment fencing 随释放失效，最终 summary 和 parent 终态只由
Reconciler 写入。通用 `stopping` timeout reconciler 跳过 deferred batch parent，child 仍使用现有 stop timeout 收敛。

`fail_fast=true` 在某个 item 耗尽 logical retry 后使用同一 stop 流程，停止创建新 child 并停止仍在运行的 child。

Parent 终态：

```text
全部 item succeeded                  -> succeeded
存在最终 failed/lost/expired         -> failed
fail_fast 停止剩余 child             -> failed
用户 stop                            -> killed
parent execution TTL                 -> expired
provisioning/generation/编排错误       -> failed
```

无论 parent succeeded 或 failed，都保存完整 batch summary。

## 幂等和一致性

各层独立使用唯一键、request hash 和 CAS，不使用跨 DB、Kubernetes/TKEx、ArtifactStore 的分布式事务。

Top-level submit 唯一键：

```text
(owner_principal_id, environment_id, submit_kind, idempotency_key)
```

CLI 每次命令生成随机 key，并在该命令的 HTTP retry 中复用；用户可显式传 `--idempotency-key`。同 key、同 request hash
返回原 execution 并设置 `reused=true`；不同 hash 返回 409。

Gateway 对包含 Environment、code refs、resources、timeouts、generation 和 batch 参数的 canonical submission envelope 计算
`submission_sha256`，并与 top-level execution 一起持久化。`generation_epoch` 在 remote generation 首次开始时分配并写入
`distributed_batch_runs`；parent infra retry 必须复用同一 epoch 和 task-list manifest。

其他 producer key：

```text
submit:{idempotency_key}:source
submit:{idempotency_key}:nexus-wheel
submit:{idempotency_key}:uv-lock
submit:{idempotency_key}:task-list
dependency-wheelhouse:{build_key}
batch-task-list:{batch_id}:{generation_epoch}
batch:{batch_id}:item:{item_id}:attempt:{attempt_no}
run-result:{execution_id}:{attempt_id}
```

同 producer key、同 sha/spec hash 复用原对象；不同 hash 是一致性错误。Provider workload 名称由 `allocation_id` 派生，
allocator create 重试遇到同 spec hash 的现有对象视为成功，hash 不同则 allocation 失败。

Artifact metadata 创建遇到 producer/idempotency 唯一键竞争时，使用新 DB session 重读并复用胜出的记录；正在接收正文的
`artifact_upload_busy` 是可重试错误，不得直接把成功 execution 收敛为 result persistence failure。

Stop、allocation delete、Artifact delete 都必须重复调用安全。任何 Kubernetes/TKEx/ArtifactStore 网络调用均在 DB 事务外执行。

## 持久化和数据库访问

### 状态快照

不新增通用 progress-event 表。MySQL 保存控制状态、ledger、计数和 ArtifactRef；ArtifactStore 保存大对象。

`distributed_environment_allocations` 的普通列至少保存：

```text
allocation_id, environment_id, pool_id, workload, status
desired_manager_count, ready_manager_count
dependency_build_id
config_version, provider_resource_ref, allocation_spec_hash
operation_owner, operation_lease_until, status_version
last_error_code, last_error_message, last_progress_at
created_at, ready_at, released_at, updated_at
```

`allocation_type`、`provider_type`、`provider_config_ref` 和 source hard limits 保存在 immutable `allocation_spec_json` 中；
后续 Environment 更新不改变在途资源。旧记录只有在 `config_version` 未变化时才能回退读取 Environment，否则 fail close。
Reconciler 只在 ready 数量、
provider 状态或错误变化时更新 allocation；普通 heartbeat 不刷新 `last_progress_at`。

Batch child 只更新自己的 execution。Batch Manager 分页获取状态变化，再在一个短事务内 CAS item/attempt、计算 counter delta
并更新 `distributed_batch_runs`。没有变化的 poll 不写 DB，因此 1000 个 child 不会同时争抢同一 summary row。

Result 顺序固定为：写 ArtifactStore、校验、Artifact CAS ready、execution 写 `result_artifact_id` 并 CAS succeeded。

### 最小新增表

```text
distributed_principals
distributed_provider_configs
distributed_environments
distributed_environment_grants
distributed_environment_pools
distributed_environment_allocations
distributed_environment_resource_usage
distributed_submit_tokens
distributed_dependency_builds
distributed_artifacts
distributed_artifact_refs
distributed_artifact_uploads
distributed_batch_runs
distributed_batch_items
distributed_batch_item_attempts
distributed_audit_events
distributed_trace_database_accounts
instance_manager_liveness
shard_leases
```

Existing execution、assignment 和 Manager 表增加 owner、Environment、pool、allocation、workload、`submission_sha256`、
spec hash、result Artifact、first schedulable time、stop reason 和 capability snapshot 等字段。

### 索引和批量边界

关键索引：

```text
executions              (owner_principal_id, created_at, execution_id)
executions              (allocation_id, status)
executions              (status, shard_id, queue_expires_at, execution_id)
executions              (status, shard_id, execution_expires_at, execution_id)
executions              (status, shard_id, stop_deadline_at, execution_id)
executions              (status, shard_id, cleanup_done_at, execution_id)
instance_managers       (allocation_id, last_heartbeat_at)
instance_managers       (environment_id, pool_id, workload, last_heartbeat_at)
instance_manager_liveness (last_heartbeat_at, manager_id)
batch_runs              (shard_id, control_state, last_reconciled_at, lifecycle_lease_expires_at)
allocations             (status, updated_at)
allocations             (environment_id, status)
allocations             (dependency_build_id)
dependency_builds       UNIQUE (owner_principal_id, environment_id, build_key)
dependency_builds       (status, operation_lease_until)
batch_items             (batch_id, status, item_id)
batch_items             (current_execution_id)
batch_item_attempts      UNIQUE (execution_id)
batch_item_attempts      (batch_id, status, item_id, attempt_no)
artifacts                UNIQUE (owner_principal_id, kind, producer_key)
artifacts                (status, updated_at)
trace_database_accounts  UNIQUE (owner_principal_id, backend_name)
trace_database_accounts  UNIQUE (backend_name, db_username)
TDSQL principal batch     UNIQUE (run_id, run_span_id, record_id)
TDSQL principal batch     (parent_execution_id, time_unix_nano, run_id, run_span_id, record_id)
```

调度、权限、分页和 GC 使用的字段必须是普通列，不能依赖 JSON 查询。控制数据库和 TDSQL principal 表都不使用 MySQL 原生
partition；batch 只通过普通列 `parent_execution_id` 过滤，先以 10 万 item 和百万级 Trace row 压测结果决定后续方案。

### Connection Pool

当前已经使用 SQLAlchemy `AsyncAdaptedQueuePool`。部署按进程角色显式覆盖现有 pool：

```text
Control Gateway        pool_size=10, max_overflow=10
Application Gateway    pool_size=5,  max_overflow=5
Scheduler              pool_size=2,  max_overflow=2
Reconciler             pool_size=4,  max_overflow=4
InstanceManager        pool_size=2,  max_overflow=0
```

Dynamic Manager Pod 必须注入 InstanceManager 限额。MySQL engine 固定使用 `READ COMMITTED`，一致性依靠显式行锁、CAS、
operation lease 和 commit fence。网络上传、provider API 和 generator subprocess 期间不能占用 DB transaction。

### 平台配置与密钥

所有 Distributed 进程使用同一个强类型 `DistributedConfig` 启动快照。配置来源按边界分为：

- Kubernetes 应用 Secret 只保存 database URL；内置 MySQL root password 使用独立基础设施 Secret。
- 普通平台配置保存在 `distributed_config_overrides`。
- internal、太湖、签名 keyring、Artifact backend 和远端 provider 等平台长期密钥保存在
  `distributed_config_secret_overrides`，并映射为不参与 repr/dump 的 `DistributedSecrets`。
- Run Trace backend 的普通参数保存在 `run_trace_sinks`，密码等构造参数通过 `secretConfig` 引用
  `DistributedSecrets.run_trace_credentials`；敏感参数出现在普通配置、缺失引用或 backend 构造失败时启动失败。
- `trace_database_access` 按 Run Trace backend 配置用户只读入口、独立 Trace schema、account provider 和 credential ref。
  `mysql` provider 直接执行账号 SQL；`tencent_cloud_dcdb` provider 使用部署侧 region、instance ID、API endpoint 和 account host
  调用腾讯云 API，创建主机读账号并只授予目标 database 的 `SELECT`，权限变更等待 DCDB Flow 完成。腾讯云 VPC 内且 CAM
  限制请求来源时配置 `dcdb.internal.tencentcloudapi.com`，其他部署使用公网 endpoint。MySQL 管理账号或腾讯云 SecretId/SecretKey
  只写入 `distributed_config_secret_overrides` 的 `control-gateway/*` scope，不写入 `*/*`，其余角色均不可读取。Control Gateway
  启动时校验 provider 与 credential 类型一致、read database 与 backend database 一致，且 Trace endpoint 不得指向控制数据库。
- ProviderConfig 继续使用业务表，只保存非敏感参数与 `credential_ref`；in-cluster ServiceAccount、imagePullSecret 和短期
  execution/allocation ticket 仍由 Kubernetes 管理。
- 工蜂、模型和 Wedata 凭据属于 submission，不进入平台配置或 Helm。

普通配置与密钥配置均按 `*/* -> role/* -> role/identity` 覆盖，未知字段、字段分类错误或类型错误会使进程启动失败。进程只在启动时
加载一次；修改后必须滚动重启对应角色。`nexus distributed config set|show|delete` 管理普通配置，
`secret-set|secret-list|secret-delete` 管理密钥；密钥写入只接受文件或 stdin，查询只返回字段名和版本。

生产 Helm 默认设置 `NEXUS_DIST_CONFIG_STORE_REQUIRED=1`：缺少 `*/*` 普通或密钥基线、或最终 `auth_mode=disabled` 时，长期服务在
监听前退出。`production_mode=true` 再启用 HTTPS、强 keyring、共享 Artifact backend 和 digest image 校验；DevCloud/local
无太湖验证可显式关闭启动闸门。该开关是数据库不可覆盖的 bootstrap-only 配置，Reconciler 将同一值注入 dynamic Manager。
`environment_enabled_providers` 是 allowlist，空列表表示禁止全部 dynamic provider，并在 ProviderConfig enable、Environment
enable/create 和 submission 三处重复校验。

Console 由 Control Gateway 提供，但浏览器直接访问 Application Gateway。Control Gateway 渲染 `/v1/console` 时，将启动时
加载的普通部署配置 `application_public_base_url` 写入页面配置，前端以它作为 Application Gateway 的默认且唯一生产地址；
不要求用户录入，也不从当前 Control 域名推导。`auth_mode=identity|required` 时忽略 `applicationBaseUrl` 查询参数，只有
`auth_mode=disabled` 的本地开发允许用该参数临时覆盖。

个人浏览器请求 Control API 时由 Control 公网域名注入太湖 JWE，请求 Application API、SSE 或 WebSocket 时由 Application 公网
域名重新注入 JWE。应用请求在两个入口分别由 SGW 校验调用方签名和订阅，并附加 Nexus 可验证的被调用方签名。两个 Gateway 使用
同一个太湖应用 Token 独立验证入口身份：个人按 `StaffId` 构造 principal，应用按 `x-rio-paasid` 查找 token binding，因此 owner
过滤和 Grant 校验保持一致。Gateway 不在公网链路间透传原始 JWE 或 x-rio 签名。`application_internal_base_url` 与公网地址完全
分离，只用于 Manager、batch Manager 和 builder 通过内网 CLB 访问 Application Gateway，并使用 internal credential 或短期
ticket。

runner-test 使用 `https://test.control.runner.nexus-agent.woa.com/v1/console` 和
`https://test.application.runner.nexus-agent.woa.com` 两个太湖入口。普通配置设置 `auth_mode=required`、
`application_public_base_url=https://test.application.runner.nexus-agent.woa.com`，并将 Control Origin 精确加入 `cors_origins`。
两个入口使用同一个太湖应用 Token；测试完成前不发布 runner-prod。个人 PAT 入口只用于 CLI 或浏览器个人身份，不能作为应用
x-rio 调用的替代；浏览器太湖入口必须允许无凭据的 CORS preflight，再在实际请求上注入身份。

## 部署和安全

- 只有 Control Gateway 和 Application Gateway 暴露用户入口；InstanceManager internal service 不创建公网 Ingress。
- 个人入口消费用户 Bearer PAT 并注入 `x-tai-identity`；应用入口由 SGW 校验调用方 x-rio 签名并添加被调用方签名。Gateway 只在
  JWE 或 SGW 后端签名验证成功后建立身份，缺失、混用、过期、签名失败、App 未绑定或用户授权失败时 fail close。
- 生产 CORS 只允许配置的 console origin。使用浏览器 identity 的 mutating API 必须校验可信 Origin 和 CSRF token；WebSocket
  GET 握手通过 `nexus-csrf.<token>` subprotocol 携带 CSRF，Gateway 验证后剥离该保留协议，只向 runtime 转发应用协议。CLI 的
  CLI 请求不依赖浏览器 CSRF。
- Gateway 代理 live command/runtime 时不转发任意用户 header，只保留固定 HTTP/trace allowlist 和受限 `HX-*`，再重建内部凭证。
- Reconciler ServiceAccount 只按已知资源名 `get/create/delete` StatefulSet 和 dependency builder Job，并以 `get/delete`
  在 allocation 释放时清理旧版本遗留的 headless Service，按需
  `get/create/patch/delete` allocation/build Secret，不授予 list/watch。
- internal auth、Generic Repository 和远端 provider 凭证来自平台密钥配置；Manager 只获得
  allocation-scoped Artifact credential。平台密钥不能进入 submission spec、Artifact 或日志。
- 生产模式下 identity trust、internal auth、ArtifactStore credential、provider config 或 workload image 缺失时组件 fail close。
- 旧 allocation snapshot 中的 `api_token_file`、`token_file` 或 `ca_file` 不做隐式 ref 转换；离线升级前必须清空非终态 dynamic
  allocation/dependency build，否则 Reconciler 明确拒绝旧 snapshot。
- workload/builder image 使用服务端快照并在生产校验 digest；bootstrap 对 source 和 wheelhouse 做 whole-Artifact 完整性校验及
  安全解包，并将选定 Nexus wheel 绑定到原始 ArtifactRef。用户不能覆盖 image、ServiceAccount、securityContext 或保留 label。
- Dynamic workload 禁用 ServiceAccount token 自动挂载和 Service link 环境变量；所需 Artifact 凭证只通过 allocation-scoped
  Secret 文件提供。
- role、Grant、default quota claim、Environment enable/disable、Trace 数据库账号生成/轮转/禁用/恢复、stop、command 和 force
  cleanup 写审计事件，审计不含密钥明文。

## API 和页面

Control Gateway 提供：

```text
Environment 管理、disable、Grant、catalog、default quota claim 和 my-environments
ProviderConfig 管理、enable/disable
principal role（`user`、`admin`）管理
Artifact upload metadata/status
typed single/live/batch submission
execution list/detail/stop
batch summary（默认只读持久化 counter/final summary；可显式请求 child 状态诊断聚合）、item cursor list、item/attempt detail
execution allocation detail
admin allocation list/detail/force-cleanup
admin dependency build 低基数状态汇总
用户自己的 Trace 数据库只读账号获取、轮转和状态查询
admin Trace 数据库账号列表、禁用、恢复、强制失效和 Grant 修复
```

用户凭据 API 复用 My Environments 的 owner 语义，密码只出现在 POST 成功响应：

```text
GET  /v1/my-environments
POST /v1/my-environments/{environment_id}/trace-database-credentials

GET  /v1/access/trace-database-accounts
POST /v1/access/trace-database-accounts/{principal_id}/{backend_name}/{disable|restore|reset|reconcile}
```

`GET /v1/my-environments` 在 Environment 列表外按 backend 去重返回 `trace_database_accounts`，每项包含 backend、read
host/port、database、username、status、`access_scope=database` 和确定性 `batch_table_name`；不返回密码。只有用户对
至少一个使用该 backend 的 Environment 持有有效 Grant 时才返回并允许操作。凭据 POST body 显式携带 `backend_name` 和
`operation=create|reset`，成功响应额外返回一次性 password。Admin 路由全部要求 `admin` role，普通用户不能传入 principal 覆盖
owner。所有凭据 POST 响应设置 `Cache-Control: no-store`。

Application Gateway 提供：

```text
Artifact content upload
scoped Artifact content download
run result read
active live SSE
live command
runtime proxy
```

Console 按工作流分为普通工作区和管理员区。普通工作区包含 Runs、Batches 和 My Environments；管理员区包含 Access、
Environments、Infrastructure 和 System，普通用户不渲染管理员入口。Infrastructure 统一承载 Allocations 和 Managers，避免将低频
控制面对象全部作为一级导航。

My Environments 显示可认领的 Environment default quota 和已有 Grant 快照，但不展示 system-default token id 或密钥材料。页面
顶部增加按 backend 去重的 `Trace Database Access` 表，字段固定为 Backend、Connection、Username、Batch table、Scope、Status
和 Action。Scope 首期明确显示 `All trace tables`；`not created` 提供 Create credentials，`ready` 提供 Reset password，disabled
账号只能由管理员恢复。Principal 表尚未由首个 batch 预建时仍展示确定性表名，不在页面刷新时跨库探测表是否存在。

创建或轮转成功后使用一次性凭据 modal 展示 host、port、database、username、password 和 batch table，每项提供复制图标；密码默认
隐藏，可显式显示。关闭 modal、切换页面或刷新时立即清除前端内存，不进入 URL、日志、localStorage、全局错误状态或后续 GET。

管理员 Environment 表单可启用或清空完整 default quota。Access 页面切换项固定为 Users、Grants 和 Database accounts；账号表展示
可完整复制的 Principal、Backend、Database、Username、Batch table、Status、Updated 和 Action。Action menu 提供 Disable、
Restore、Reset password 和 Reconcile，危险操作需要确认并写审计。列表和普通页面都不能只复制视觉截断后的 principal、用户名或
表名；管理员同样只能在 reset 成功后的 modal 中看到一次密码。

Runs 默认只展示 single、live 和 batch parent，batch child 只从 Batch Detail 进入。Runs、Allocations、Managers 和 batch items 都
使用服务端稳定 cursor 分页，热路径不计算精确总数。Execution Detail 的 allocation 只在 Overview 加载，events、result 和 batch
items 在对应视图打开后按需加载；请求切换时取消旧请求。

Batch 使用独立、可刷新和可分享的页面：

```text
/v1/console/batches/{parent_execution_id}
/v1/console/batches/{parent_execution_id}/items/{item_id}
/v1/console/executions/{child_execution_id}?from_batch={parent_execution_id}
```

Batch Detail 展示 parent phase、generation、完整 child 状态计数、进度、最后进展时间和 stop 操作。子任务列表默认每页 100 条，
支持按状态、item ID、child execution ID 和 error code 查询；item/attempt 在详情抽屉中展示，child 打开统一 Execution Detail，并
保留返回 parent 的上下文。未进入 Batch Detail 时不得预取 items。状态计数读取 Batch ledger 的持久化 counter/final summary，
诊断聚合只能由用户显式触发。

Environment 表单默认只展示名称、allocation/provider、Manager 数量、CPU、内存和 slots；其他资源限制与 timeout 作为 Advanced
overrides，空值表示继承部署或 Environment 默认值。详情页同时展示有效值及其来源。Grant 提供 Standard、Batch 和 Custom preset，
并为权限独立提供 All submissions 和 Custom preset；Token 提供 Submit Single、Submit Live、Submit Batch、Read-only 和
Operator scope preset。原始 submission JSON 仅保留为高级调试入口。

Managers 默认查询 healthy/current allocation，stale Manager 按 allocation/session 汇总后再显式展开；System 首屏只读取可缓存的
低基数健康摘要，dependency build、Capability snapshot、builder Job 和 provider 错误按诊断区懒加载。

页面展示的 phase 从已持久化状态推导，不增加 execution 状态机：

```text
waiting_resources
building_dependencies
dependency_build_failed
waiting_policy
generating
materializing
running_children
finalizing
uploading_result
releasing
```

## 可观测性

MVP 使用状态快照、管理 API 和结构化日志，不额外引入 Prometheus 框架。统一日志字段：

```text
execution_id
batch_id
allocation_id
environment_id
manager_id
assignment_id
artifact_id
```

不记录 token、完整 spec、用户代码和凭证明文。

`/v1/system/status` 增加低基数汇总：

- allocation 各状态数量、desired/ready 差值和 deletion backlog。
- dependency build 各状态数量。
- active batch、queued/running child 和最长无进展 batch。
- unhealthy Manager、最老 queued execution 和 scheduler backlog。
- pending/failed Artifact upload、checksum failure 和 orphan backlog。

告警覆盖 provisioning/generation 超时、Manager 长期不 ready、parent 阶段性无进展、allocation deleting 卡住、Artifact 校验失败
和 scheduler backlog 持续增长。监控标签不包含 execution_id/batch_id，避免高基数。

## Migration 和发布

沿用当前 Alembic 风格 migration。`0013` 及以前仍是 expand-only；本期控制面吞吐改造的 `0014` 至 `0018`
属于一次需要维护窗口的协调迁移：

- `0014` 将 `scheduler_leases` 直接改名为通用的 `shard_leases`。迁移清除旧 owner，并按旧 `shard_count` 为
  `execution-scheduler` / `execution-reconciler` 写入已过期占位 lease：新进程必须重新抢占，同时保留只增不减校验基线。
  旧控制面进程不能与该迁移并行。
- `0015` 新增并回填 `instance_manager_liveness`。新控制面优先读取新表；若旧 Manager 仍只更新
  `instance_managers.last_heartbeat_at`，则在未 fenced 时读取两者中较新的心跳，支持 workload 滚动升级。
- `0016` 回填 BatchRun shard，`0017` 增加 Reconciler shard-leading 索引；迁移完成后再启动新 Scheduler/Reconciler。
- `0018` 为缺少计数行的 Environment 和 Grant 回填零值 resource usage；已有计数保持不变。新控制面不再在 Allocation
  reserve/release 热路径执行 duplicate-key upsert。
- `0021` 新增 ProviderConfig registry；`0022` 为已有 Grant 回填完整 canonical scopes，并清理误写入的 `submit_all` role。
- `0023` 新增平台密钥配置表；`0024` 新增 execution-scoped 用户凭据表；`0025` 新增 system-default token/default quota；`0026`
  新增应用身份绑定；`0027` 新增 Trace 数据库只读账号元数据表，表中不保存密码。
- TDSQL principal batch 表不由 Alembic 管理，在首个 batch 的 scope preparation 中懒创建。新代码只接受 `shared` 和
  `batch_principal_v1` 两种 TDSQL storage scope，不提供其他 batch 表路由、数据搬迁或双写。
- 历史 execution 的 `owner_principal_id=NULL`，视为 legacy，只对管理员可见。
- 新 typed submit 必须写 owner、Environment 和 allocation。
- 旧 execution 的 `allocation_id=NULL`，继续走 legacy placement。
- 旧 Manager 缺少 workload 时只视为 `execution`，不能运行 batch parent。
- 不回填历史 owner quota；Environment/Grant resource usage 只补齐缺失的零值行，不重算或覆盖已有计数。

发布顺序：

1. 进入短维护窗口，暂停 submit 入口，确认没有非终态 dynamic allocation/dependency build，并停止全部旧 Gateway、Scheduler、
   Reconciler、predefined Manager 和 batch Manager。`0012` 会清空并改造 Manager 注册表，旧 Manager 不能在迁移期间继续
   heartbeat；迁移后由新版本 Manager 重新上报。
2. 执行 migration 到最新 head，再部署兼容 legacy/new execution 的 Gateway、Scheduler 和 Reconciler。
3. 验证 `/v1/health`、`/v1/system/status`、两个 shard lease group 和 Manager 心跳后恢复入口。
4. 仅使用 database URL bootstrap 启动 db-init；长期服务保持零副本，通过配置命令导入 `*/*` 普通配置和平台密钥，再滚动
   启动各角色。启动闸门会拒绝缺失基线或 `auth_mode=disabled` 的部署。
5. 滚动部署支持新 liveness 写入的 Manager、ArtifactStore 和 allocator。
6. 管理员依次创建 disabled ProviderConfig、Environment、Grant 和初始 token，验证 provider credential ref 后再 enabled。
7. 最后发布 CLI 和页面。

迁移到最新 head 后，不能单独回滚到仍查询旧表或读取 Helm provider registry 的 control image。优先向前修复；若必须回滚，
需要再次停止控制面，执行数据库 downgrade，并同步回滚、重启 workload Manager，不能只切镜像 tag。

Environment enabled/disabled 是功能开关，不再增加全局 cloud-submit flag。

## 需要修改的模块

- `nexus/distributed/config/*`：普通配置、`DistributedSecrets`、作用域存储、启动快照和角色 ConfigView；删除
  `artifact_validation_concurrency`、`artifact_validation_max_waiters`、`artifact_validation_admission_timeout`、
  `artifact_validation_deadline`，Helm 不新增替代配置。
- `nexus/distributed/persistence/*`：ORM、数据库引擎和 migration；领域 model 分别归入 `access`、`artifacts`、`environment`、
  `execution` 和 `instance_manager`。
- `nexus/distributed/repositories/*`：owner filter、Grant/resource reservation、allocation/DependencyBuild CAS、Artifact、batch
  ledger 和 cursor API。
- `nexus/distributed/environment/*`、`dependency_build/*`：独立 allocator/reconciler、renderer 和共享 provider client resolver；
  Environment limits 删除 `max_source_compression_ratio`。
- `nexus/distributed/control_plane/*`：Scheduler、Reconciler、shard lease 以及相互隔离的 core、Environment、DependencyBuild、
  Artifact maintenance 和 Batch lifecycle loop。
- `nexus/distributed/execution/executors/*`、`instance_manager/*`、`repositories/manager.py`：stop disposition、deferred
  assignment release、新注册字段、startup capability、wheelhouse-only dynamic bootstrap 和角色化 DB pool。
- `nexus/distributed/execution/drivers/*` 与 `batch/*`：single/live/batch 执行、parent 编排、恢复、fencing 和 BatchControl port/service。
- `nexus/distributed/artifacts/run_result.py` 和 `execution/trace_backends.py`：最终结果的
  ArtifactTraceSink/ArtifactTraceSource 和 Application Gateway client。
- `nexus/runner/trace/tdsql/*`、`execution/trace_backends.py`、`execution/drivers/nexus_batch.py`、
  `execution/drivers/nexus_single.py` 和 `instance_manager/*`：TDSQL raw trace backend、principal scope 预建、可信 owner 传递以及
  child parent metadata 校验。
- `nexus/distributed/artifacts/store.py` 提供 local/Generic Repository backend，并把下载流式 identity 校验纳入 `get` contract；
  Application Gateway 不保留独立 `artifacts/validation.py` 或 archive validation pool。
- `submission/source_package.py`、`dependency_build/wheelhouse.py`、`dependency_build/worker.py`：确定性 `source.tar.zst`、安全解包、
  `max_archive_bytes`、build key、一次有界 wheel 检查、无 manifest wheelhouse 和 builder Job 入口；
  `runtime/server/venv/manager.py` 增加严格离线 wheelhouse 安装及 Nexus wheel 内容绑定。
- `nexus/distributed/access/taihu.py`、`access/auth.py`、Trace 数据库账号 service/repository、`gateways/common/*`、
  `gateways/control/*`、
  `gateways/application/*`：太湖协议、auth/Origin/CSRF policy、typed submit、Artifact、batch、allocation、runtime proxy、
  result API、TDSQL 只读账号和共享错误/代理契约。TDSQL maintenance client 通过宽松 Protocol 隔离具体驱动。
- `nexus/cli/commands/*`：新增 `submit` 命令；移除 live-server/live-runner，保留 `run live`。
- `nexus/runner/batch/*`、`nexus/runner/compose/*`：共享 deterministic materialization 和 canonical JSONL 导出。
- `nexus/distributed/console/src/*`：用户、管理员、Batch Detail、Environment、Grant 和 Allocation 页面；My Environments 增加按
  backend 去重的 Trace Database Access 和一次性凭据 modal，管理员 Access 增加 Database accounts；Environment 表单删除 source
  compression ratio 字段。
- `deployment/distributed/*`：universal/workload/builder 镜像、Job/StatefulSet RBAC、NetworkPolicy、database bootstrap Secret 和
  dynamic workload 模板；普通业务配置与平台长期密钥不再通过 Helm 注入。
- `tests/integration/distributed/cloud_batch_*`、`tests/e2e/distributed/run_tkex_cloud_batch.py`：共享场景、API 验收、TKEx
  只读观测、分阶段容量保护和 JSON 报告。

## 验证

所有原定在 Kind 执行的容器集群验收统一改在 DevCloud 测试容器集群执行。测试场景、规模、故障注入和验收指标保持不变，
不能以 Kind 结果替代 DevCloud 验收证据。

### 单元和组件测试

- Environment/Grant/token owner、scope、资源上限和非 owner 404。
- Source tar.zst 安全、确定性 whole-archive hash、相对 symlink、nexus wheel/uv.lock/extras 和 config-only 拒绝规则；不保留 ZIP
  兼容测试。
- Image mode 的 version 语法、`latest`/完整 image ref 拒绝、固定 workload repository 解析、CLI 互斥参数、submission
  idempotency 和 predefined 拒绝；验证 image submission 不创建代码 Artifact、DependencyBuild 或 builder Job。
- DependencyBuild 去重、lease/fencing、attempt/retry/terminal failure、build-scoped ticket、Job renderer 和 staged-output-before-ready。
- Builder 的一次有界 wheel 元数据检查及永久错误分类；无 manifest wheelhouse 的路径、成员数、总大小限制，以及 VenvManager
  `--no-index --find-links`、Nexus wheel 内容绑定和无网络回退。
- Artifact upload/complete CAS、producer key、checksum、orphan cleanup、分类并发、backend 429 重试和 result-ready-before-success。
- Artifact upload、backend download 和 HTTP download 各信任边界只校验一次；覆盖 Generic Repository PUT 后 HEAD size、缓存复用
  hash 和 maintenance recovery，不保留 Gateway archive validation executor 测试。
- Artifact scoped read credential 的 owner/Environment/ref 校验、轮转撤销和 Generic Repository HTTP contract。
- Browser CORS/CSRF、token Environment mismatch 和跨 Environment ArtifactRef 拒绝。
- TDSQL 表名稳定且不包含 principal 明文；single/live 始终写共享表，同一 principal 的多个 batch/child/retry 写同一 principal 表，
  不同 principal 写不同表。覆盖并发首次建表、可信 owner/admin 代提交、parent filter/cursor、`run_id` shardkey、交错表批写和单表
  失败隔离，确认预建失败不会回退共享表。
- Trace 数据库账号按 principal/backend 唯一、非 owner 404、普通用户不能管理他人账号、管理员操作审计、密码不落库/日志/持久前端
  状态、响应 `no-store`、并发创建/轮转串行化和失败重试；真实 TDSQL 组件测试使用两个 principal 验证账号和 principal 表均不同、
  旧密码轮转后失效、两个账号首期都能 SELECT 全部 Trace 表但不能写入或读取控制 schema，并验证账号先于或晚于 principal 表创建
  都不需要逐表补 Grant。Frontend 测试覆盖 backend 去重、状态操作、完整值复制和一次性 modal 清理。
- Allocator observe/create/delete 幂等、operation lease、stop during provisioning 和 delete 后释放额度。
- Kubernetes/TKEx fake client，以及独立测试集群上的 TKEx E2E。
- Scheduler allocation/workload/spec hash 匹配和可配置 scan/assignment batch。
- task-list generation commit、chunk resume、cursor、重复 item/spec hash 冲突和 generator 非确定性。
- 10 万 item JSONL 使用 500-item chunk 物化，并以最大 500 的稳定 cursor 完整分页；测试不启动 10 万个 workload。
- Parent fencing、child idempotency、active-window refill、logical retry、fail_fast、deferred stop 和 final summary；验证 batch
  parent 在 child 全部终态前保持 `stopping`，single/live 仍按 immediate stop 收敛。
- at-least-once、live running 后 lost 不重试、TTL stop reason 和 parent 部分失败终态。
- Submit idempotency、202 immediate return、single/batch `--wait` 和 live URL 返回。
- Migration 可重复执行、legacy admin-only 和新旧 execution 并存。

### DevCloud source-package smoke

先在 DevCloud 测试容器集群运行一个独立的小规模 dynamic source-package batch：测试镜像不包含该测试 task/generator，
测试现场构造确定性 `source.tar.zst`、
当前 Nexus wheel 和 `uv.lock`。Source package 定义唯一的 `distributed_source_package_echo` task 和 remote generator，生成 20 个 item；
申请 1 个 batch Manager、2 个 execution Managers、每个 2 slots，`max_inflight=4`。

验收三个输入经 Application Gateway 上传、builder Job 生成并发布 wheelhouse、workload 离线创建 Manager venv、source workspace
优先 import、remote generator、child
`nexus_single` 和 run-result 全链路。Allocation ready 后删除一个 execution Manager Pod，替代 Pod 必须用同一 allocation-scoped
credential 重新下载 source/wheelhouse 并启动，逐个读取并校验全部 20 个输出都包含 source package 中的唯一 marker。该用例与 2000-task 常规 batch
测试分开。

### DevCloud image-mode smoke

从本次 DevCloud workload digest 构建同 repository 的唯一测试版本，只把测试 task 和 remote generator 放入 `/workspace`，不通过
source Artifact 提供。分别提交一个 dynamic single 和一个 20-item remote-generator batch，验证 CLI 没有上传 source、wheel 或
lockfile，DB 没有创建 DependencyBuild，single 以及 Batch parent/child 都使用解析后的同一 image version。逐项校验结果 marker、
Manager capability、task-list/run-result Artifact、allocation release 和 Stateful workload/Pod 清理；同时覆盖不存在版本导致的
provisioning failure。该 smoke 独立运行，不加入容量矩阵。

### DevCloud 常规 batch 验收

DevCloud 测试容器集群使用 Kubernetes allocator、config-only 测试镜像和隔离的测试 ArtifactStore。创建：

```text
1 个 batch Manager
25 个 execution InstanceManager
每个 execution Manager limit 为 1 CPU / 1 GiB
Kubernetes provider 的 manager_resource_request_ratio = 0.1
每个 execution Manager 20 slots
max_inflight = 500
2000 个 distributed_sleep_echo task
每个 task 使用 await asyncio.sleep(20)
```

常规 batch 基线单独运行，不注入故障。验收：

- 25 个 execution Manager 全部 heartbeat 后 child allocation 才 ready。
- 2000 个 item/child/result 不重不漏，最终 `2000 succeeded`。
- 每个 Manager 并发不超过 20，总 `running` child 不超过 500，`running` child 峰值至少达到 450；`active_items` 只作为
  提交窗口诊断，不能作为并发验收依据。
- batch 分多页读取完整，summary、ledger 和 execution 事实一致。
- Parent/child allocation 最终 released，Stateful workload 和 Manager session 完成清理。
- 记录 MySQL `Max_used_connections`、`Slow_queries`、scheduler 填满 slots 时间和总体耗时，并写 JSON 验收报告。

故障恢复拆成两个独立 DevCloud 用例：

1. 并发达到稳定值后删除一个 execution Manager Pod；最多 20 个 assignment lost/retry，替代 Pod 使用新 session，最终
   仍为 2000 succeeded 且无重复 child/result，恢复用例的 `running` child 峰值至少达到 400。
2. generation complete 且部分 child 已物化后删除 batch Manager Pod；新 parent 从 task-list Artifact 和 ledger cursor 恢复，
   不重跑 generator、不重复创建 child。

Application Gateway 临时错误、Scheduler/Reconciler 重启等使用独立组件测试，不叠加到 DevCloud batch 基线。该基线同时验证
500-result 突发不会突破 input/result/total 并发边界，backend 429 可在 upload lease 内恢复且不会产生失败 Artifact。

### TKEx E2E 和容量验收

TKEx 测试使用专用 dynamic Environment、测试太湖用户和 Environment Grant，控制面连接测试数据库，计算资源只创建在目标测试集群。
测试 workload 镜像包含 config-only 的 sleep fixture；source-package task/generator 只存在于提交的 `source.tar.zst`。所有 stage 串行，
上一 stage 的 allocation、Pod 和 workload 完全释放后才进入下一 stage。

Execution Manager 申请 `1 CPU / 1 GiB`；`manager-2000`、`sandbox-agent-10000` 和 `sandbox-agent-60000` 的单个
batch Manager 申请 `4 CPU / 4 GiB`，其余 stage 的 batch Manager 仍为 `1 CPU / 1 GiB`。Builder Job 独立使用
`2 CPU / 4 GiB`，并在 Manager allocation 前完成，不与大规模 Manager stage 同时运行。Environment/Grant 压测上限为
3001 Managers、3010000 millicores、3010 GiB、60001 slots、100000 items 和 1 个 active batch。slot 上限包含
orchestration Manager 固定占用的 1 slot，以及 execution Managers 提供的 60000 slots。实际矩阵为：

`manager-2000` 的 runner-dev 压测基线使用 16 个 Scheduler 副本，并设置
`scheduler_shard_ownership_limit=1`。提交前必须确认 16 个 shard 都有 active lease、`owner_count=16` 且每个 owner 只持有 1 个
shard。该 stage 还把内部 `batch_submit_batch_size` 设为 8000，使 7996 个初始 child 在一个事务中可见；中央
`batch_lifecycle_page_size` 仍为 500。以上都是大规模验收部署配置，不改变 Chart 的通用默认值。
`sandbox-agent-10000` 把内部 `batch_submit_batch_size` 设为 10000。

| Stage | Code | Items | Execution Managers | Slots/Manager | max_inflight | Task sleep | 最低 running child 峰值 |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| wheelhouse-smoke | source-package | 20 | 2 | 2 | 4 | 20s | 4 |
| sandbox-agent-smoke | config-only | 20 | 5 | 4 | 20 | 360s | 18 |
| wheelhouse-fanout | source-package | 499 | 499 | 1 | 499 | 20s | 450 |
| config-baseline | config-only | 2000 | 25 | 20 | 500 | 20s | 450 |
| source-heavy | source-package | 2000 | 25 | 20 | 500 | 20s | 450 |
| container-scale | config-only | 4000 | 199 | 10 | 1990 | 22s | 1800 |
| container-ceiling | config-only | 4000 | 499 | 4 | 1996 | 22s | 1800 |
| manager-2000，显式 opt-in | config-only | 16000 | 1999 | 4 | 7996 | 22s | 7200 |
| sandbox-agent-10000，显式 opt-in | config-only | 16000 | 1999 | 6 | 10100 | 1200s | 10000 |
| sandbox-agent-60000，显式 opt-in | config-only | 100000 | 3000 | 20 | 60000 | 3600s | 60000 |
| extreme，显式 opt-in | config-only | 20000 | 499 | 20 | 9000 | 20s | 8500 |

Config-only 压测使用 `tests/integration/distributed/Dockerfile.test-workload` 构建的专用测试镜像。镜像中的
`tests.integration.distributed.cloud_tasks` 提供 `sleep_echo` 和 `sandbox_agent_echo`，正式 Nexus workload wheel 不包含这些任务。

各 stage 的 `batch.timeout_per_run` 默认 120 秒；`manager-2000` 使用 2400 秒，因为该超时从 child 创建开始计时，必须覆盖
万级 child 的调度排队、启动延迟和结果持久化背压，而不仅是 22 秒任务执行时间。Manager 的成功结果 publish deadline 为
1800 秒，必须短于该 per-run timeout。sandbox smoke 使用 1200 秒，万级 sandbox stage 使用 3600 秒；
`sandbox-agent-60000` 使用 7200 秒 per-run timeout 和 3600 秒 stop timeout，其 generation/queue 和 parent TTL 同步放宽。

Sandbox stage 使用 Gongfeng prod-x3、Tlinux 4 minimal、`1 CPU / 1 GiB` runtime 和 GPT-5.5。每个 child 先取得唯一 runtime ID，
再要求 `main_agent` 通过 `swe_agent/bash` 写入并读回唯一 marker，随后按 profile 持有 runtime。Provider 使用 HTTP、关闭自动 WebSocket，
并在 30 秒窗口内随机启动。Gongfeng 与模型凭证由测试 submission 作为 execution-scoped 用户输入提供，不通过 Helm、
`release-secrets` 或 Manager 全局环境注入；报告必须脱敏。

表中规模按每个 Manager Pod 只有主容器计算。提交前必须完成只读容量、DB、镜像和 Secret 预检。目标集群可用 CPU/内存至少是
本次最大需求的 110%；先通过小 stage 读取实际 Pod spec。若每个 Manager Pod 实际有 `c` 个容器，则 execution Manager 数量
自动收敛到 `min(表中数量, floor(3001 / c) - 1)`，`max_inflight` 和最低峰值按可用 slots 等比例收敛，item 数不变。任何 stage
都不得依赖运行中熔断来修正已知的 sidecar 扩容。超过 100 个 Manager Pod 的 stage 还要求 execution Manager 使用
`NEXUS_DIST_DB_POOL_SIZE=1`、`NEXUS_DIST_DB_MAX_OVERFLOW=0`，batch Manager 使用 `4`、`2`。前者限制随 worker 数增长的常驻连接，
后者为 batch 提交、心跳、command 和 stop 保留并行连接。压测持续采集 `Threads_connected`、`Threads_running`、slow query
和连接上限。压测机无法直连 MySQL 时，显式使用 `--control-database-metrics` 从管理员接口 `/v1/system/status` 读取相同指标；
接口只返回方言和计数器，不返回 DSN、地址、用户名或凭证。超过 100 个 Manager Pod 的 stage 必须配置这一入口或受限权限的
`--database-url-file`，否则不允许执行。

三个 source stage 复用同一个 `source.tar.zst`、Nexus wheel 和 `uv.lock`，因此必须得到同一个 DependencyBuild 和 wheelhouse。
Source package 选择基础 workload 未安装的锁定依赖；每个结果返回依赖版本、`sys.prefix`、`nexus.__file__` 和唯一 marker，逐条证明
`--no-index --find-links` Manager venv、source import 和 result Artifact 全链路。TKEx 只读观测同时验证 StatefulSetPlus
ready、PodIP、`POD_IP` Downward API、无 allocation Service、无 OOM 和最终资源删除。

出现以下任一条件立即请求 stop，不继续放量：活动测试容器超过 2000、预留容量低于 10%、DB 连接超过上限 70%、ready gap
超过 2、generation complete 后超过 profile 的无进展阈值、OOM/restart 达到 5、结果上传失败率超过 1% 或 Gateway 持续 5xx。

每个 stage 最终必须满足 item、child execution、result Artifact 一一对应且全部 succeeded；分页结果、summary 和 execution 事实一致，
allocation released，StatefulSetPlus/Job/临时 Secret/Pod 清理完成。报告保存镜像、输入 Artifact、DependencyBuild、ready/fill/总耗时、
TKEx/DB 采样和停止原因，不保存 token 或数据库凭证。Manager/parent 删除恢复、stop、predefined config-only single/live/batch、
source-package 拒绝以及用户/管理员页面权限使用独立 E2E，不与容量 stage 叠加故障。

Sandbox stage 另按每个结果中的 `[sandbox_ready_at, sandbox_hold_finished_at)` 区间计算真实 ready 峰值；smoke 必须达到 20，
万级 stage 必须达到 10000，不能用 child `running` 峰值替代。结束后通过 Gongfeng runtime list 核对全部 runtime 为
`stopped/not_found`；超时先按已知 ID 兜底 stop，再判定失败。

普通 stage 的并发验收通过 `include_status_counts=true` 显式读取 batch summary 的
`child_execution_status_counts.running`。该诊断字段直接按当前 child execution 聚合，不使用 Reconciler 异步回填的 item ledger
状态；默认 summary 不执行该聚合，后者只用于最终一致性和分页查询，不能衡量 20 秒短任务的瞬时并发。

## 实施顺序

1. Schema、identity/Grant/token、ArtifactStore 和 owner filter。
2. Typed submit、config-only/source package/image、DependencyBuild/wheelhouse、EnvironmentController 和 Kubernetes/TKEx allocator。
3. Dynamic/predefined single/live、ArtifactTraceSink 和资源生命周期。
4. Task-list generation、batch ledger、`nexus_batch` parent 和 child 调度。
5. CLI、页面、migration rollout 和完整 DevCloud 验收。

每一步都保持 legacy execution 可运行；只有对应 Environment enabled 后，新 submission 才对用户开放。
