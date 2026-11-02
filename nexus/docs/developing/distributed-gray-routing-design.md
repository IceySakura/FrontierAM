# distributed 通用灰度路由组件与 artifact-service-store 灰度设计

状态：Proposal（设计评审）  
日期：2026-09-07  
范围：Nexus distributed；首个接入点为 `artifact-service-store`

## 1. 摘要

为 distributed 增加一个与业务无关的稳定加权路由器 `WeightedTrafficRouter`，由调用方提供稳定的业务 key，路由器根据 `seed + key` 的确定性 hash 选择 target。ArtifactStore 通过 `RoutedArtifactStore` 适配该能力：新对象按 object key 选择后端，已存在对象按持久化 URI 的 scheme 选择后端。

该设计的关键保证是：灰度只影响新对象的写入位置，不改变已有对象的位置；不随机切分同一个对象的读写，不双写，不在写失败时自动切到另一个后端。这样既能将部分新 artifact 流量切到 `artifact_service`，也能支持回退后继续读取和 GC 已经写入 artifact-service 的对象。

本设计只定义组件、配置、生命周期、验证和发布方式，不包含业务代码实现。

## 2. 背景与现状

当前 distributed 制品抽象位于 [`nexus/distributed/artifacts/store.py`](../../nexus/distributed/artifacts/store.py)，已有 `ArtifactStore` protocol 以及 `local`、`generic_repository` 等实现。`feat/artifact-service-store` 分支已经增加了 `ArtifactServiceStore` 和 `MultiProviderArtifactStore`：

- `ArtifactServiceStore` 以 `artifactsvc://` URI 标识对象；
- generic repository 以 `generic://` URI 标识对象；
- `MultiProviderArtifactStore` 新写入走 primary，存量对象的读、stat、delete 按 URI scheme 路由；
- `ArtifactRow.uri` 在实际上传前就会创建并持久化，上传阶段还会再次计算并校验 URI 一致性；
- Control Gateway、Application Gateway、Reconciler 都可能构造 ArtifactStore，分别承担 metadata、实际上传/下载、GC 等职责。

因此，“每个请求随机选择 backend”不适用于当前模型：同一个 artifact 可能在不同进程、不同请求中被随机选到不同 backend，最终会出现 URI 与实际对象位置不一致。

另外，distributed 配置是进程启动快照。灰度比例不是一个只重启单个 gateway 就能安全完成的局部参数，相关消费者必须使用同一套路由策略。

## 3. 目标与非目标

### 3.1 目标

1. 提供一个可被 distributed 其他模块复用的稳定加权路由能力。
2. 支持按百分比逐步把新流量切到任意 target。
3. 确保同一稳定 key 在同一策略下始终命中同一 target。
4. 对已持久化对象提供与灰度策略解耦的稳定读取和删除路径。
5. 支持灰度期间、回退期间和旧对象过期期间的平滑共存。
6. 提供启动校验、路由指标和可验证的上线/回退流程。
7. 保持现有单 backend 配置兼容，不要求一次性改造所有 distributed 组件。

### 3.2 非目标

MVP 不包含：

- 动态管理接口、实时热更新或控制台；
- 按用户、租户、环境的 allowlist；
- 双写、shadow write 或异步复制；
- backend 故障时的自动跨 backend failover；
- 全 distributed 共享的一套全局灰度比例；
- 复杂的一致性 hash ring、服务发现或跨语言策略同步协议。

## 4. 核心概念与不变量

### 4.1 概念

- **Policy**：一个组件的灰度策略，包含 `seed`、`version` 和 target 权重。
- **Target**：一个可被路由到的逻辑目标，例如 `generic_repository`、`artifact_service`。
- **Stable key**：由调用方提供、在同一业务对象生命周期内不变的 key。
- **Write target**：创建新对象时选中的 target。
- **Read target**：读取或删除已有对象时由持久化地址决定的 target。

### 4.2 必须保持的不变量

**I1. 确定性**

同一 `seed`、同一策略和同一 stable key 必须得到同一个 target。不能使用 Python 内置 `hash()`、进程随机数或请求随机数。

**I2. 地址权威**

对象一旦持久化了 URI，后续 `get/stat/delete` 以 URI 为准，不重新执行灰度选择。

**I3. 单写入目标**

一次 `put` 只调用一个 target，绝不双写。

**I4. 不做隐式写 fallback**

选中的 target 写失败时，只在该 target 上执行已有重试策略；重试耗尽后失败。不能把同一次写入改投另一个 target，否则可能造成 URI 和实际位置不一致。

**I5. 零权重仍可读**

权重为 `0` 表示不接收新写入，不表示可以从路由表移除。只要数据库中仍存在对应 URI，就必须保留该 target 的 scheme 路由。

**I6. route identity 可恢复**

负责生成 URI 的进程必须把最终 URI 持久化；负责实际上传的进程必须优先按这个已持久化 URI 绑定 backend，而不是用当前灰度策略重新选择。所有消费者仍必须认识存量 URI 对应的 scheme 和 target，但不要求它们在灰度比例变化时共享完全相同的权重快照。

## 5. 方案取舍

### 5.1 Pod/进程级灰度

让一部分 Pod 使用 artifact-service，其他 Pod 使用 generic。

优点是实现简单，缺点是实际流量比例取决于 Pod 数量和请求分布，无法保证对象创建和上传落到同一 backend，也无法自然支持多个调用方使用同一策略。不作为通用组件方案。

### 5.2 每请求随机灰度

每个请求按概率随机选择 backend。

优点是代码直观，缺点是同一个 artifact 的 `object_uri`、`put`、`get`、`delete` 可能分别命中不同 backend，尤其不适合当前 URI 先落库、后上传的流程。不采用。

### 5.3 稳定 key + 加权路由 + URI 路由存量对象

新对象使用确定性 hash 做加权选择，存量对象使用 URI scheme 直接路由。该方式在对象创建前决定归属，跨进程一致，并且灰度回退不会改变已有对象的地址。推荐采用。

## 6. 总体架构

```text
                         ┌────────────────────────┐
                         │ WeightedTrafficRouter  │
                         │ seed + stable_key      │
                         │ -> target              │
                         └───────────┬────────────┘
                                     │
                         new object  │ choose
                                     ▼
┌───────────────┐       ┌────────────────────────┐       ┌──────────────────┐
│ Control       │──────▶│ RoutedArtifactStore    │──────▶│ generic          │
│ object_uri()  │       │ object_uri()/put()     │       │ repository       │
└───────────────┘       │ get/stat/delete by URI │       ├──────────────────┤
                        └───────────┬────────────┘       │ artifact_service │
                                    │                    └──────────────────┘
                                    │ persisted URI scheme
                                    ▼
                         ┌────────────────────────┐
                         │ Application / Reconciler│
                         │ read, upload, GC        │
                         └────────────────────────┘
```

分层职责：

1. `WeightedTrafficRouter`：只负责稳定加权选择，不感知 ArtifactStore 和网络错误。
2. `RoutedArtifactStore`：将 router 适配到 ArtifactStore，处理新对象选择和已有 URI 路由。
3. 各 backend store：继续负责自身协议、认证、上传、下载和错误归一。
4. distributed 配置：提供策略快照和 target 到具体 backend 的映射。

## 7. 通用路由器设计

### 7.1 运行时接口

建议提供如下最小运行时模型：

```python
@dataclass(frozen=True, slots=True)
class TrafficTarget:
    name: str
    weight_bps: int


@dataclass(frozen=True, slots=True)
class TrafficPolicy:
    component: str
    seed: str
    version: int
    targets: tuple[TrafficTarget, ...]


class WeightedTrafficRouter:
    def choose(self, stable_key: str) -> str: ...
```

配置模型可以使用 Pydantic，运行时模型不应依赖配置来源。未来可以由 DB、环境变量或静态配置分别构造同一个 `TrafficPolicy`。

### 7.2 选择算法

MVP 使用固定的 SHA-256 bucket 算法：

```text
payload = UTF-8(seed + "\\0" + stable_key)
digest  = SHA256(payload)
bucket  = uint64(big-endian(digest[0:8])) % 10000

按 target 声明顺序做累计权重区间查找：
  [0, cumulative_weight_1)       -> target_1
  [cumulative_weight_1, ...)     -> target_2
```

约束：

- 权重单位为 basis point，`10000 = 100%`；
- 所有权重必须在 `0..10000`；
- 权重总和必须等于 `10000`；
- 至少一个 target 的权重大于 0；
- target name 不得重复；
- `seed` 不为空；
- stable key 为空时直接报配置/调用错误，不退化为随机路由。

算法名称和输入拼接格式应视为 `sha256-mod-10000-v1` 契约，避免未来修改 hash 算法后出现跨版本隐式重分配。`version` 用于观测和发布审计，不拼入 hash；改变权重时不应因为单纯改版本而让所有 key 无条件重排。

权重变化会使一部分“尚未持久化的新对象”的 target 变化，这是预期行为。已经持久化 URI 的对象不受影响。

### 7.3 路由器错误语义

路由器启动时完成全部静态校验。运行时只可能返回已配置的正权重 target；任何未知 target、非法策略或空 key 都应是明确错误，而不是静默使用默认 target。

路由器不负责：

- 调用 target；
- retry/backoff；
- 读取 URI；
- 网络健康检查；
- 根据错误重新选择 target。

## 8. RoutedArtifactStore 设计

### 8.1 构造

逻辑构造如下：

```python
RoutedArtifactStore(
    router=WeightedTrafficRouter(policy),
    stores_by_target={
        'generic_repository': generic_store,
        'artifact_service': artifact_service_store,
    },
)
```

所有配置中的 target 都要能解析为 store。包括权重为 0 的 target；这样回退后仍可读取和删除存量对象。

每个 store 必须声明唯一的 `uri_scheme`：

```text
generic_repository -> generic
artifact_service   -> artifactsvc
local              -> file
```

构造阶段建立两张表：

- `target -> store`：服务新对象的 route selection；
- `uri_scheme -> store`：服务已有对象的读取和删除。

如果两个 target 声明相同 URI scheme，启动失败。不能使用 target name 猜测 URI 所属，也不能让后注册的 backend 覆盖先注册的 backend。

### 8.2 新对象操作

`object_uri(object_key)`：

1. 调用 router 选择 target；
2. 调用选中 store 的 `object_uri(object_key)`；
3. 返回带有该 backend scheme 的 URI。

`put(..., object_key=..., stored_object=...)` 支持两种模式：

1. **Bound mode（Artifact 上传的强制模式）**：调用方传入已经从 `ArtifactRow.uri` 得到的 `StoredObject`。RoutedArtifactStore 解析 URI scheme，直接选择对应 backend；可以额外校验该 backend 对 `object_key` 计算出的 URI 与传入 URI 相等，但不得重新执行灰度策略。
2. **Unbound mode（没有预先持久化地址的通用调用）**：调用方不传 `stored_object`，RoutedArtifactStore 使用 router 根据 object key 选择 target，再调用该 target 的 `put`。

Artifact 的 metadata → upload 流程必须使用 Bound mode。Unbound mode 只用于调用方明确拥有“创建并选择新对象”的完整生命周期，不能用于已经存在 `ArtifactRow` 的上传会话。

该接口改动可以向现有 `ArtifactStore` protocol 增加可选的 `stored_object` 参数；非 routed 的单 backend 实现可以忽略它，RoutedArtifactStore 必须使用它进行 URI 绑定。这样配置在 Control Gateway 和 Application Gateway 之间变化时，上传仍然遵守已落库的对象位置。

### 8.3 已有对象操作

`get`、`stat` 和 `delete` 解析 `StoredObject.uri` 的 scheme，并通过 scheme map 找到 backend：

```text
artifactsvc://... -> ArtifactServiceStore
generic://...    -> GenericRepositoryArtifactStore
file://...       -> LocalArtifactStore
```

这些操作不重新调用 router。未知 scheme 返回 `stored_object_provider_mismatch`，不尝试其他 backend。

### 8.4 close 语义

`close()` 关闭所有已构造的唯一 backend，即使某个 backend 权重为 0。应尽量完成所有 close，再抛出第一个 close 错误，保持当前 `MultiProviderArtifactStore` 的资源释放语义。

## 9. ArtifactStore 配置

### 9.1 新配置

建议在 `DistributedConfig` 中增加 ArtifactStore 灰度专用策略：

```yaml
artifact_store_routing_enabled: true
artifact_store_routing_seed: artifact-store-v1
artifact_store_routing_version: 1
artifact_store_routes:
  - target: generic_repository
    weight_bps: 9500
  - target: artifact_service
    weight_bps: 500
```

已有 backend 连接配置继续保留，例如：

```yaml
artifact_store_provider: generic_repository
artifact_generic_base_url: ...
artifact_generic_repository: ...
artifact_service_base_url: ...
artifact_service_system_prefix: /system
artifact_service_namespace_prefix: /_nexus_dist
```

其中 `artifact_store_routes.target` 是逻辑 target 名，不直接承载 URL、凭据或 backend 细节。

### 9.2 兼容模式

当 `artifact_store_routing_enabled: false` 时保持当前单 provider 行为：

- `artifact_store_provider` 是新写入 primary；
- `artifact_store_read_fallback_providers` 仅用于按 URI scheme 读取存量对象；
- 不创建通用加权 router。

当灰度模式开启时，`artifact_store_routes` 成为新写入的唯一权威配置。为避免配置歧义，建议校验并拒绝同时指定会影响新写入的 legacy primary/fallback 组合；legacy 字段仍可作为各 backend 的连接参数保留。

一个可选的兼容实现是：工厂内部将旧配置规范化为同一个路由模型：

```text
artifact_store_provider               -> weight 10000
artifact_store_read_fallback_provider -> weight 0, read-only target
```

这样可以最终收敛到一套 `RoutedArtifactStore`，并保留旧配置的行为。

### 9.3 配置校验

校验分为三层：

1. **策略校验**：权重范围、总和、重复 target、空 seed、至少一个正权重。
2. **backend 校验**：每个 target 都能构造；生产环境禁止 `local` 作为新写入或存量 fallback；`artifact_service` 使用 HTTPS；generic repository 具备所需访问配置。
3. **服务作用域校验**：Control Gateway、Application Gateway、Reconciler 必须使用相同的 target name、URI scheme 映射和 backend 连接语义；灰度权重可以由 Control Gateway 的新对象创建路径独立更新。Application Gateway 和 Reconciler 必须保留所有仍有存量 URI 的 target。

配置错误在启动时失败，不允许进程启动后才第一次请求时发现。

## 10. Artifact 流程与一致性

### 10.1 新对象上传

以 run result 为例：

```text
1. 生成 object_key
2. Control Gateway: router.choose(object_key)
3. Control Gateway: selected_store.object_uri(object_key)
4. 将 ArtifactRow(uri=selected_uri, status=uploading) 持久化
5. Application Gateway: 读取 ArtifactRow.uri
6. Application Gateway: 以 Bound mode 按 URI scheme 选择 backend
7. 校验该 backend 的 object_uri(object_key) == ArtifactRow.uri
8. 在绑定的 backend 执行 put
9. 发布 ready
```

如果第 7 步失败，必须报告 `artifact_object_uri_mismatch`，不能为了让上传成功而改写数据库 URI 或切换 backend。

### 10.2 读取、删除和 GC

```text
ArtifactRow.uri
       ↓ parse scheme
artifactsvc:// -> artifact_service
generic://    -> generic_repository
       ↓
get/stat/delete
```

Reconciler 的 GC 与普通下载遵守同一条 URI 路由规则。新策略中的 zero-weight target 仍要构造，直到该 scheme 对应的存量对象全部过期并完成清理。

### 10.3 策略切换与 in-flight 上传

线上无法可靠排空 in-flight upload，因此不能把排空作为一致性前提。MVP 采用已持久化 URI 绑定方案：

- Control Gateway 创建 artifact 时按当时策略选择 target，并把最终 `ArtifactRow.uri` 持久化；
- Application Gateway 收到上传时读取该 URI，以 Bound mode 调用 `put`；
- `put` 按 URI scheme 路由，不按当前权重重新选择；
- Reconciler 的 stat/delete 同样按 URI scheme 路由；
- 灰度比例可以在不中断业务流量的情况下通过配置刷新变化，不影响已经进入 `uploading` 的对象。

因此，灰度比例变更不需要排空业务流量。仍然存在一个代码发布顺序要求：必须先把支持 Bound mode 的 Application Gateway 全量部署完成，再让 Control Gateway 开始产生新的灰度 URI。否则旧版 Application Gateway 仍可能按旧的 primary 逻辑处理新 URI。

如果更新的是 URI scheme、target name 或 backend 映射，而不是单纯调整权重，则必须先保证所有消费者都能识别新旧映射，再进行配置变更；这属于路由表兼容性门槛，而不是流量排空门槛。

## 11. 错误处理与回退

### 11.1 backend 错误

backend 自身继续将网络错误、超时、429、5xx 等归一为当前的 transient error，并使用已有 retry/backoff。router 不参与错误处理。

### 11.2 不自动跨 backend 回退

以下行为明确禁止：

- artifact-service `put` 失败后改写到 generic；
- artifact-service `get` 404 后尝试 generic；
- generic backend 暂时不可用时把新对象切到 artifact-service；
- 同一个对象先写一个 backend，再异步补写另一个 backend。

URI 是对象位置的权威记录，跨 backend fallback 会掩盖配置、权限或数据丢失问题。

### 11.3 人工回退

将新写入权重切为 0，保留新 backend 的 scheme 路由：

```yaml
artifact_store_routes:
  - target: generic_repository
    weight_bps: 10000
  - target: artifact_service
    weight_bps: 0
```

这样新对象回到 generic，已经写入 artifact-service 的对象仍可读取、删除和 GC。只有在 artifact-service 对应对象全部过期、确认无引用后，才允许从路由表移除它。

## 12. 可观测性

### 12.1 启动信息

每个相关进程启动时记录一条不含凭据的策略摘要：

```text
component=artifact_store
routing_enabled=true
policy_version=1
seed_fingerprint=<short hash>
targets=generic_repository:9500,artifact_service:500
policy_fingerprint=<hash of normalized policy>
```

日志不记录 token、用户名、完整 URL 中的敏感信息或 object key。

### 12.2 指标

建议增加：

```text
distributed_artifact_route_selected_total{
  target, kind, operation, policy_version
}

distributed_artifact_route_result_total{
  target, kind, operation, result, policy_version
}

distributed_artifact_route_latency_seconds{
  target, operation, policy_version
}
```

`result` 使用有限集合，例如 `success`、`not_found`、`transient_error`、`permanent_error`、`validation_error`。不要把 object key、artifact id、owner id 放入 label。

### 12.3 流量核对

新写入流量的主核对口径是 `distributed_artifacts.uri` 的 scheme 分布：

- 按 `kind` 统计 `artifactsvc://` 与 `generic://` 的对象数；
- 按 `size_bytes` 统计字节量，避免只看对象数；
- 将数据库分布与 route-selected 指标对比；
- 小样本不要求精确命中比例，使用足够样本观察长期分布。

关键告警：

- `artifact_object_uri_mismatch > 0`；
- 新 backend 的 transient/permanent error 或 p95 明显高于 old baseline；
- run result publish 未完成或 ready 延迟异常；
- URI scheme 无法路由；
- route-selected 与持久化 URI 分布出现持续偏差。

## 13. 测试设计

### 13.1 `WeightedTrafficRouter` 单元测试

覆盖：

1. 相同 seed + key 多次选择结果相同；
2. 两个独立 router 实例选择结果相同；
3. 0% / 100% 边界；
4. 95% / 5% 区间边界；
5. 权重总和错误、负数、超过 10000、重复 target、空 seed、全 0 均启动失败；
6. route 顺序确定，且使用明确的 SHA-256 算法而不是语言内置 hash；
7. 足够数量 key 的统计分布接近配置权重；
8. 策略 version 变化但 seed 和权重不变时选择保持不变。

统计测试只用于验证整体分布，不能替代边界和确定性测试。

### 13.2 `RoutedArtifactStore` 单元测试

覆盖：

1. `object_uri(key)` 与 `put(key)` 命中同一个 target；
2. Bound mode 根据已持久化 URI scheme 路由，即使当前权重已经改变；
3. Unbound mode 才根据 object key 和当前权重选择 target；
4. `get/stat/delete` 根据 URI scheme 路由，不重新计算权重；
5. 权重为 0 的 target 仍可读取和删除；
6. 旧 generic URI 在灰度和回退期间仍走 generic；
7. artifact-service URI 在回退到 0% 后仍走 artifact-service；
8. 未知 scheme 抛 `stored_object_provider_mismatch`，不尝试其他 backend；
9. 写入失败不调用第二个 backend；
10. 两个 target 复用同一 URI scheme 时构造失败；
11. 所有 backend 都被 close，且第一个 close 错误最终被抛出；
12. legacy `MultiProviderArtifactStore` 配置转换后的行为与现有测试一致。

### 13.3 Artifact 集成测试

使用 fake stores 验证完整的“建 metadata → 上传 → 读回 → 删除”闭环，并覆盖两种 URI scheme。再使用 artifact-service 分支已有的真实 client contract 测试验证：

- `/system` 路由请求不带身份头；
- `artifactsvc://` URI 可往返读写；
- sha256 和 size 校验仍生效；
- GC 可以删除系统 namespace 对象。

### 13.4 部署验证

在 dev/test 环境使用至少几千个真实对象或真实请求，核对：

- 预期 URI scheme 分布；
- 每个 target 的成功率、延迟和字节量；
- source package、wheel、task list、run result 都能完成闭环；
- 不出现 `artifact_object_uri_mismatch`；
- 回退到 0% 后新对象只进入旧 backend，旧对象仍可从新 backend 读回。

## 14. 发布与回退流程

### 阶段 A：代码和双 backend 就绪

先部署包含通用 router、Bound mode、RoutedArtifactStore 和两个 backend 的版本。此时策略设置为：

```text
old=10000, new=0
```

确认所有相关进程能构造两个 target，所有 Pod 都能按新旧 URI scheme 路由，并确认 Application Gateway 已经使用 Bound mode。此阶段不改变灰度比例。

### 阶段 B：小比例灰度

建议从 `1%` 或 `5%` 开始，具体选择取决于能够产生的对象量。先观察上传、读回、GC、run result publish 和 artifact-service 服务端指标。

### 阶段 C：逐步放量

建议采用：

```text
5% -> 10% -> 25% -> 50% -> 100%
```

每一步只需要确认所有消费者的 target/scheme 映射和 backend 连接配置兼容；权重策略由负责创建新 artifact 的 Control Gateway 生效。无需等待 in-flight upload 排空。

建议验收条件：

- `artifact_object_uri_mismatch = 0`；
- 新 backend 的错误率不高于 old baseline 加约定阈值；
- 主要操作 p95 没有持续显著劣化；
- 新旧 URI 对象均可读、可删，GC 无积压；
- 系统账号、权限和网络边界没有异常告警。

### 阶段 D：回退

出现错误率、延迟、服务端稳定性或数据闭环问题时：

1. 将新 backend 权重设置为 `0`；
2. 保留新 backend 在 scheme map 中；
3. 刷新负责创建新对象的配置；其他消费者至少保持新旧 scheme 映射可用；
4. 验证新写入只落旧 backend；
5. 验证已有 `artifactsvc://` 对象仍可读、可删；
6. 保留至少一个完整 artifact TTL 周期，或直到数据库确认没有对应存量对象，再移除新 backend。

## 15. 代码改动边界

实现阶段预计涉及：

```text
nexus/distributed/traffic/               通用策略模型与 router
nexus/distributed/artifacts/store.py     RoutedArtifactStore / 工厂接入
nexus/distributed/config/model.py        配置字段与校验
nexus/distributed/config/views.py        运行时配置视图
tests/unit/distributed/traffic/           router 单测
tests/unit/distributed/artifacts/         adapter 与兼容行为单测
docs/developing/                         灰度发布与回退 runbook
```

不修改各业务 route 的上传、下载、GC 业务语义；它们继续依赖 `ArtifactStore` protocol。Artifact-service backend 的认证、URI 解析、错误归一继续由其自身实现负责。

首个实现可以将当前 `MultiProviderArtifactStore` 的 scheme dispatch 逻辑抽取/扩展为 `RoutedArtifactStore`，避免形成两个语义相近但行为逐渐分叉的 wrapper。需要同步扩展 `ArtifactStore.put` 和 `ArtifactUploadCoordinator`，把已持久化的 `StoredObject` 传入 Bound mode。保留兼容别名或兼容工厂路径，待旧配置全部迁移后再清理。

## 16. 后续扩展方向

当其他 distributed 模块需要灰度时，只复用通用 router：

```python
target = router.choose(stable_key)
```

如果该模块的结果会被后续请求访问，必须像 ArtifactStore 一样持久化最终 target、URI 或其他可恢复的 route identity；不能在读取阶段重新按当前比例计算。

后续可单独评估：

- DB-backed 动态策略和无损版本切换；
- per-tenant/per-environment allowlist；
- backend 健康状态参与新对象 admission；
- shadow/replication 语义；
- 路由策略变更的 in-flight upload token。

这些增强都不应改变本设计的两个基础原则：新对象归属必须确定，存量对象地址必须权威。

## 17. 评审结论与实施门槛

本设计建议按以下顺序进入实现：

1. 先实现并测试通用 `WeightedTrafficRouter`；
2. 再将当前 multi-provider scheme routing 收敛到 `RoutedArtifactStore`；
3. 接入 ArtifactStore 配置和三类服务的一致性校验；
4. 完成 fake-store 闭环测试和真实 artifact-service contract 测试；
5. 最后按阶段 A～D 执行部署灰度。

在设计正式进入实现前，需要确认的硬门槛有两个：

- `ArtifactRow.uri` 能在 Application Gateway 上传路径中被可靠取出，并传入 Bound mode；
- 所有消费者都能识别新旧 URI scheme，并且 backend target 映射在发布期间保持兼容。

不再要求排空或冻结 in-flight `uploading` session；这正是 Bound mode 要解决的线上约束。
