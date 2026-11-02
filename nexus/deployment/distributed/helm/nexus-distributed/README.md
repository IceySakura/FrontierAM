# Nexus Distributed Helm Chart

该 chart 部署 Nexus distributed control plane、可选 predefined Manager 和本地测试依赖。Scheduler 通过 MySQL
heartbeat 发现 Manager；Reconciler 负责 dynamic environment、dependency build、Artifact maintenance 和 batch lifecycle。

## 配置边界

Helm 只管理 Kubernetes 拓扑和启动引导：

- Deployment、StatefulSet、Service、RBAC、NetworkPolicy、volume、端口、镜像和资源。
- 所有长期进程只挂载一个 Nexus 应用 Secret key：`database-url`。
- 普通平台配置保存在 `distributed_config_overrides`。
- internal、太湖、签名 keyring、Artifact backend 和远端 provider credential 保存在
  `distributed_config_secret_overrides`。
- ProviderConfig 的 endpoint、cluster、namespace 和 `credential_ref` 保存在独立业务表。
- Kubernetes ServiceAccount、imagePullSecret 和 execution-scoped ticket 仍由 Kubernetes 管理。
- 工蜂、模型和 Wedata 凭据由用户随任务提交，不进入 chart 或平台全局 Secret。

配置在进程启动时加载一次，修改数据库配置后需要滚动重启对应组件。

`configStore.required` 默认开启。长期服务启动时必须读到 `*/*` 普通配置和密钥配置，且最终 `auth_mode` 不能为 `disabled`；否则
在监听端口前退出。该值只来自部署 bootstrap，数据库不能覆盖，并会原样传给 dynamic Manager。`production_mode=true` 仍负责
更严格的生产校验。两个 local values 示例显式关闭该闸门。

## 镜像

- `universal`：Control/Application Gateway、Scheduler、Reconciler、db-init 和 fake runtime。
- `workload`：predefined/dynamic InstanceManager。
- `builder`：短生命周期 dependency builder Job。

chart 直接引用 `controlImage` 和 `workloadImage`。Dynamic workload 与 builder 的 digest-pinned image 通过数据库普通配置
`environment_base_image` 和 `environment_builder_image` 提供，submission 不能覆盖。

```bash
docker build -f deployment/distributed/Dockerfile.universal \
  -t ${REGISTRY}/nexus-distributed-universal:${TAG} .
docker build -f deployment/distributed/Dockerfile.workload \
  --build-arg NEXUS_UNIVERSAL_BASE_IMAGE=${REGISTRY}/nexus-distributed-universal:${TAG} \
  -t ${REGISTRY}/nexus-distributed-workload:${TAG} .
docker build -f deployment/distributed/Dockerfile.builder \
  --build-arg NEXUS_WORKLOAD_BASE_IMAGE=${REGISTRY}/nexus-distributed-workload:${TAG} \
  -t ${REGISTRY}/nexus-distributed-builder:${TAG} .
```

## Database Secret

生产环境应使用已有 Secret：

```bash
kubectl -n nexus-dist create secret generic nexus-dist-database \
  --from-literal=database-url="${NEXUS_DIST_DATABASE_URL}"
```

```yaml
secrets:
  create: false
  existingSecret: nexus-dist-database
  databaseUrlKey: database-url

mysql:
  enabled: false
```

默认本地 values 会创建 database Secret 和单独的 MySQL auth Secret。MySQL root password 不会混入 Nexus 应用 Secret；
正式环境不要把真实 database URL 或 MySQL password 写入 values，以免进入 Helm release 历史。

Dynamic Manager 所在的每个目标集群也必须预先存在 database URL Secret。Secret 名和 key 通过数据库普通配置
`environment_database_url_secret_name`、`environment_database_url_secret_key` 指定。

## 数据库配置

迁移完成后，使用管理命令写入配置。`expected-version=0` 表示创建；更新和删除必须传当前 version。

```bash
nexus distributed config set \
  --role '*' --identity '*' --expected-version 0 \
  --file /secure/distributed-config.json

nexus distributed config secret-set \
  --role '*' --identity '*' --expected-version 0 \
  --file /secure/distributed-secrets.json

nexus distributed config show --role '*' --identity '*'
nexus distributed config secret-list
```

`secret-set` 只接受文件或 stdin；`secret-list` 只显示 scope、version 和字段名。覆盖顺序为 global、role、role+identity，
普通配置和密钥分别做顶层覆盖。配置输入会拒绝未知字段、类型错误、普通表中的密钥字段以及密钥表中的普通字段。

最小认证配置示例：

```json
{
  "auth_mode": "required",
  "production_mode": true,
  "control_public_base_url": "https://control.runner.example",
  "console_public_base_url": "https://cloud.runner.example",
  "application_public_base_url": "https://application-mcp.runner.example",
  "console_application_base_url": "https://application.runner.example",
  "application_internal_base_url": "http://application-gateway.internal:8082",
  "cors_origins": ["https://control.runner.example"]
}
```

长期密钥文件使用 `DistributedSecrets` 的强类型结构，例如 internal token、太湖应用 Token、Artifact ticket keyring、
Artifact writer/maintenance credential、provider credential 和 `run_trace_credentials`。Run Trace backend 的
`run_trace_sinks.<name>.secretConfig` 将构造参数名映射到逻辑 credential ref，例如
`{"password":"runner-trajectory-db"}`；敏感字段不能出现在普通 `config`、`sinkConfig` 或 `sourceConfig` 中。密钥文件不得进入
Git、日志或命令行参数。

需要由用户选择的 backend 还应设置 `kind`、`displayName`、`description` 和 `userSelectable: true`，再由管理员在 Console 的
**Access → Trace backends** 授权给 principal。迁移期间可保留
`require_principal_run_trace_backend_grant: false`；所有 principal 配置好 grants/defaults 后，正式环境可切为 `true`。

Console 由 Control Gateway 的 `/v1/console` 提供。认证模式下，前端 Application Gateway 地址使用
`console_application_base_url`，未配置时兼容回退到 `application_public_base_url`；仅 `auth_mode=disabled` 的本地开发允许
`applicationBaseUrl` 查询参数覆盖。

旧 live API 仅在迁移期显式开启：

```yaml
liveCompatibility:
  enabled: true
```

入口为 `/compat/live/v1/runs`。除 `/compat/live/health` 外，它复用 Gateway 的个人 JWE 和应用 x-rio 鉴权，并使用 internal
credential 将已验证的 principal 或 `app_id` 委托给内部 Control/Application Gateway；不转发原始公网凭据，也不使用共享 submit
token。兼容请求按映射后的真实用户应用 Environment Grant、Quota、owner 隔离和审计。默认 placement 取
`instanceManager.environmentId`、`instanceManager.poolId` 和 labels。

## Provider

`dynamicEnvironment.enabledProviders` 只决定 chart 是否创建 Kubernetes provider RBAC；provider endpoint 和 credential 不通过
Helm env 或 volume 注入。

```yaml
dynamicEnvironment:
  enabledProviders: [kubernetes]
  namespace: nexus-workload
```

`auth_type=in_cluster` 使用 Reconciler 的标准 ServiceAccount token/CA。远端 Kubernetes 和 TKEx ProviderConfig 保存
`credential_ref`，Reconciler 从 `DistributedSecrets.provider_credentials` 解析强类型内存 credential。MVP 的一个 Reconciler
只管理一个 workload namespace。

`environment_enabled_providers` 为空时禁止全部 dynamic provider；系统在 ProviderConfig enable、Environment create/enable 和
submission 时都校验该 allowlist。

Reconciler 从自身 bootstrap 文件读取数据库 URL，并在创建或观察 dynamic Manager 前，自动在目标 namespace 创建或校准
`environment_database_url_secret_name` 指定的 Secret。私有镜像凭据仍由目标集群管理员创建 imagePullSecret；Reconciler 不复制
registry credential。

## 安装

```bash
helm upgrade --install nexus-dist deployment/distributed/helm/nexus-distributed \
  --namespace nexus-dist \
  --create-namespace \
  -f deployment/distributed/helm/nexus-distributed/examples/values-local-predefined.yaml \
  --wait --timeout 240s
```

`db-init` 会等待数据库可连接并执行迁移。生产首次安装时先以 Gateway、Scheduler、Reconciler 和 Manager 零副本完成迁移，
再从受控管理环境导入 `*/*` 普通配置和密钥，最后恢复目标副本并使用 `--wait`。升级前必须先在独立操作中将全部旧长期
进程缩到零副本；不能指望同一次 `helm upgrade` 的 replicas 变更先于 pre-upgrade db-init hook 生效。迁移和配置导入完成后，
再滚动恢复目标副本。

predefined Manager 通过 Downward API 读取 PodIP，internal server 就绪后注册 `endpoint_url`。跨集群运行需要 control plane
能够路由 Manager Pod CIDR 并放行 Manager 端口，不依赖 Headless Service 或跨集群 DNS。

Application Gateway 的 Service/CLB 属于部署拓扑：

```yaml
applicationGateway:
  service:
    type: LoadBalancer
    externalTrafficPolicy: Local
    annotations:
      service.cloud.tencent.com/loadbalance-id: <existing-clb-id>
```

公网和内网 URL 均写入数据库配置，不放在 values。Control 与 Application 公网入口分别接受太湖注入的个人
`x-tai-identity` JWE，或经 SGW 验证并附加被调用方签名的应用 x-rio 请求；两个 Gateway 使用同一个太湖应用 Token 独立验证。
内部 Manager、builder 和 batch 流量使用内网 URL 与 internal/ticket credential。合法个人身份首访只登记普通用户；首个管理员
必须在核对 `StaffId` 后写入数据库普通配置 `admin_users`，再滚动重启 Control Gateway，不按访问顺序自动授权。

## OpenTelemetry

OTEL auto-instrumentation 在 Python import 时启动，因此 endpoint 和开关仍是 process bootstrap 配置。可选 OTEL credential 使用
独立基础设施 Secret，不与 Nexus database Secret 合并：

```yaml
otel:
  enabled: true
  endpoint: http://otel-collector.observability.svc:4318
  protocol: http/protobuf
  tracesExporter: otlp_proto_http
  tracesSampler: always_on
  existingSecret: nexus-dist-otel
  resourceAttributesKey: otel-resource-attributes
  exporterHeadersKey: otel-exporter-headers
```

未配置 `existingSecret` 时不注入 Resource Attributes 或 exporter headers。修改 OTEL bootstrap 后同样需要滚动重启。
