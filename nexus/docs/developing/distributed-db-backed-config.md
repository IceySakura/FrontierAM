# Distributed 数据库配置

## 目标

Distributed 长期进程只通过 Kubernetes Secret 获得数据库 URL。普通平台配置和平台长期密钥存入数据库，进程在启动时加载
一次强类型快照；修改配置后滚动重启对应角色，不做运行时热更新。

Helm 只管理拓扑、镜像、资源、RBAC、Service 和数据库引导。Kubernetes ServiceAccount、imagePullSecret、内置 MySQL root
password 及 execution/allocation scoped 短期 ticket 仍由 Kubernetes 管理。

## 配置边界

配置分为三类：

| 类型 | 存储 | 内容 |
| --- | --- | --- |
| 普通平台配置 | `distributed_config_overrides` | 调度、并发、超时、镜像、URL、CORS、Environment 默认值等 |
| 平台长期密钥 | `distributed_config_secret_overrides` | internal、太湖、签名 keyring、Artifact、provider 和 Run Trace 凭据 |
| ProviderConfig | `distributed_provider_configs` | provider type、endpoint、cluster、namespace、逻辑 `credential_ref` 和非敏感参数 |

`DistributedSecrets` 是平台长期密钥的唯一强类型模型。密钥字段不参与配置 `repr`、dump、审计值或查询响应；管理接口只展示
字段名和版本。数据库表本期保存 JSON 明文，后续在存储层增加对称加密，不改变业务配置模型。

工蜂 Token、模型凭据和 Wedata key 属于 submission-scoped 用户输入，不进入平台配置。Wedata key 单独存入
`distributed_execution_credentials`，公开 submission 和 execution API 只显示 credential kind。

## 启动与优先级

长期进程按以下顺序启动：

1. 从 `NEXUS_DIST_DATABASE_URL`、数据库 URL 文件或显式 CLI 参数构造 bootstrap 配置。
2. 连接数据库并等待目标 schema revision。
3. 根据 CLI 子命令确定 `role` 和可选 `identity`。
4. 分别读取普通配置和密钥配置的作用域 overlay。
5. 构造并校验最终 `DistributedConfig`，再初始化日志、正式连接池和业务服务。

Helm 的长期服务默认注入 `NEXUS_DIST_CONFIG_STORE_REQUIRED=1`。此模式要求 `*/*` 普通配置和密钥配置两行均已存在，且最终
`auth_mode` 必须为 `identity` 或 `required`；否则进程在监听端口前退出。`production_mode=true` 继续执行 HTTPS、强 keyring、
共享 Artifact backend 和 digest image 等生产约束。仅本地示例显式关闭该闸门。该开关属于 bootstrap-only 配置，数据库不能
覆盖；Reconciler 将当前启动快照中的值原样注入 dynamic Manager，避免控制面与工作负载使用不同的启动策略。

普通配置的优先级从低到高为：代码默认值、环境变量 fallback、数据库全局、数据库角色、数据库实例、显式 CLI 参数。密钥配置
只按数据库全局、数据库角色、数据库实例和显式启动覆盖合并。overlay 仅做顶层字段替换，不深度合并嵌套字典。

`database_url` 和 `database_url_file` 是 bootstrap-only 字段，数据库禁止覆盖。其他长期密钥不能写入普通配置表；普通字段也不能
写入密钥表。未知字段、字段分类错误或类型错误均使进程启动失败。

数据库管理命令 `distributed db init|upgrade|shell` 不读取配置表；本地 `distributed dev` 可以继续使用环境变量和显式参数。

## 作用域

两个配置表均以 `(role, identity)` 为联合主键，`*` 表示通配：

1. `role=*`, `identity=*`
2. `role=<role>`, `identity=*`
3. `role=<role>`, `identity=<identity>`

CLI 映射如下：

| CLI 子命令 | role | identity |
| --- | --- | --- |
| `control-gateway` | `control-gateway` | 无 |
| `application-gateway` | `application-gateway` | 无 |
| `scheduler --owner-id X` | `scheduler` | `X` |
| `reconciler --owner-id X` | `reconciler` | `X` |
| `instance-manager --instance-manager-id X` | `instance-manager` | `X` |

每行包含 JSON payload、单调递增 `version`、创建时间和更新时间。创建使用 `expected-version=0`，更新和删除必须携带当前版本；
版本冲突不覆盖并发修改。审计只记录作用域、字段名和版本，不记录配置值。

## Provider 与凭据

Environment 的 `provider_config_ref` 必须引用 enabled ProviderConfig。ProviderConfig 只保存非敏感配置和逻辑
`credential_ref`；Reconciler 从 `DistributedSecrets.provider_credentials` 解析对应的强类型 Kubernetes/TKEx 凭据。

`environment_enabled_providers` 是部署能力 allowlist，空列表表示禁用全部 dynamic provider。ProviderConfig enable、Environment
创建或启用以及每次 submission 都重新校验该 allowlist，避免旧的 enabled 记录绕过部署能力变更。

`auth_type=in_cluster` 的 Kubernetes Provider 不需要 `credential_ref`，直接使用 Reconciler Pod 的 ServiceAccount token/CA。
远端 Provider 缺少凭据、provider type 不匹配或 namespace 越界时 fail close。allocation 和 dependency build 保存不含凭证明文的
ProviderConfig 快照。

Control API/Console 可以在创建或更新 ProviderConfig 时同时提交实际 credential。TKEx credential 只包含 token；远端 Kubernetes
credential 包含 bearer token 和 CA certificate。Gateway 将 credential 写入全局 `*/*` 密钥配置的
`provider_credentials[credential_ref]`，并与 ProviderConfig 变更在同一事务提交。更新时省略 credential 表示保留原值；查询响应
只返回 `credential_configured`，不返回、掩码回显或记录 credential。启用远端 ProviderConfig 前必须确认对应 credential 已配置且
provider type 一致。credential 变更继续遵循启动快照语义，管理员保存后滚动重启 Reconciler，新 allocation 才使用新值。

Reconciler 从自身数据库 URL bootstrap 文件读取当前值。在远端 Kubernetes/TKEx 创建或观察 dynamic Manager workload 前，按
`environment_database_url_secret_name` 和 `environment_database_url_secret_key` 在目标 namespace 幂等创建或校准数据库
Secret；内容 hash 不一致时只更新 Nexus 管理的 key 和标记。该共享 Secret 不随 allocation 删除，多个 allocation 和 dependency
build 不重复持有数据库凭据。`environment_image_pull_secrets` 仍引用目标集群预先存在的集群侧凭据，Reconciler 不跨集群复制
镜像仓库 credential。

## Run Trace 凭据

Run Trace backend 的普通构造参数保存在 `run_trace_sinks`，密码等字段通过 `secretConfig` 引用
`DistributedSecrets.run_trace_credentials`：

```json
{
  "run_trace_sinks": {
    "tdsql": {
      "sinkClass": "nexus.runner.trace.tdsql::TDSQLTraceSink",
      "sourceClass": "nexus.runner.trace.tdsql::TDSQLTraceSource",
      "config": {
        "host": "trajectory-db.internal",
        "port": 3306,
        "user": "nexus",
        "db": "runner"
      },
      "secretConfig": {
        "password": "runner-trajectory-db"
      }
    }
  }
}
```

对应密钥配置为：

```json
{
  "run_trace_credentials": {
    "runner-trajectory-db": "<secret>"
  }
}
```

启动时 registry 将引用解析到 Sink/Source 的同名构造参数。密码、token、secret、credential、API/access/private key 等敏感
字段只允许出现在 `secretConfig`，在普通 `config`、`sinkConfig` 或 `sourceConfig` 中出现即拒绝。未知引用或 backend 构造失败
同样使 InstanceManager 启动失败。TDSQL Trace 的密码字段不得出现在对象 `repr`。

TDSQL backend 对 single/live 使用共享 `trajectory_span_events` 表。Batch 在 orchestration Manager 提交任何 child 前，以
`parent_execution_id` 幂等预建一张 `trajectory_span_events_<parent_execution_id>` 逻辑表；预建失败时 parent 直接失败，
不得提交 child，也不能由 child 懒建表或回退共享表。表名只能由系统校验后的 execution ID 派生。

Batch 表由 parent 表达归属，表内以 child `run_id` 作为 TDSQL `shardkey`，以 `run_span_id` 区分 attempt。Writer 仍按数据库
配置共享，不为每个 parent 创建线程或连接池；队列按目标表和 SQL 分组批量写入。Sink metadata 保存 storage scope 和 parent ID，
Source 重新派生受控表名，不能直接执行 metadata 中提供的 SQL identifier。Batch 完成后不立即删除表，保留周期由独立维护策略
管理。

## Gateway 地址

Control Gateway 渲染 `/v1/console` 时，将数据库普通部署配置启动快照中的 `application_public_base_url` 写入页面配置；Console
默认且在鉴权模式下只使用该地址访问 Application Gateway，不要求用户录入，也不从 Control 域名推导。只有
`auth_mode=disabled` 的本地开发允许通过 URL 参数临时覆盖。

个人浏览器分别访问 Control 和 Application 公网域名，太湖在两个入口分别注入 `x-tai-identity`；应用调用通过 SGW 的 x-rio
鉴权。默认情况下，SGW 在两个入口增加被调用方签名，两个 Gateway 使用同一个太湖应用 Token 独立验签，再按
`x-rio-paasid` 查找管理员配置的 token binding。尚未配置被调用方签名的环境可在全局普通配置中显式设置
`tai_trust_x_rio_app_id=true`，临时信任 SGW 转发的单一非空 `x-rio-paasid`；只要出现任一被调用方签名头，仍必须完整验签，不能
降级。该开关默认关闭，修改后滚动重启两个 Gateway 生效。Gateway 不在公网链路间透传原始 JWE 或 x-rio 签名。
`application_internal_base_url` 与公网地址分离，只供 Manager、batch Manager 和 builder 通过内网 CLB 使用 internal credential
或短期 ticket，不得与公网地址混用。

合法太湖身份首次访问任一 Gateway 时会幂等登记为普通用户，不附带 Environment Grant 或管理员权限。若首次部署时尚不知道
首个管理员的 `StaffId`，先保持 `admin_users=[]`，由目标用户访问一次，再从 `distributed_principals` 中核对
`created_by='system:tai-identity'` 的 `subject` 和 `display_name`。随后将 `tai/default/<StaffId>` 写入普通平台配置的
`admin_users`，并滚动重启 Control Gateway；启动引导会显式提升该 principal。不得采用“第一个访问者自动成为管理员”。

## Console 使用指引与欢迎动画

普通配置 `cloud_console_guide_url` 控制顶部、Runs 空状态和欢迎提示中的指引链接，默认
`https://nexus.pages.woa.com/cloud/overview/`。只接受完整的 HTTP(S) URL；修改后滚动重启 Control Gateway，
刷新页面即可使用新链接，无需重建前端。

普通配置 `cloud_console_welcome_animation_enabled` 默认 `false`。管理员可在 Console 的 System 页面为
Control Gateway 对应 scope 设置 `true`，保存后滚动重启 Control Gateway，再刷新页面生效。
关闭时只保留「使用指引」链接，不播放动画，也不记录动画的访问时间。
开启后，同一浏览器、同一账号首次访问或至少 7 天未访问时，展示主题色烟雾光带与 `Nexus Cloud` 字样，
随后飞向「使用指引」。减少动态效果的系统设置会跳过动画，直接显示可关闭的指引提示。

## 管理与部署

```bash
nexus distributed config set --role '*' --identity '*' --expected-version 0 --file config.json
nexus distributed config secret-set --role '*' --identity '*' --expected-version 0 --file secrets.json
nexus distributed config show --role '*' --identity '*'
nexus distributed config secret-list
```

`secret-set` 只接受文件或 stdin；密钥不能通过命令行明文传入。部署顺序固定为：以长期服务零副本运行 migration，导入或更新
`*/*` 配置和密钥，再启动长期进程。生产入口开放前必须验证配置版本、组件 readiness、internal 鉴权和公网身份链路。

管理员也可以在 Console 的 System 页面编辑或删除配置 scope。页面通过 `PUT`/`DELETE` `/v1/system/config-overrides`
使用同一个 `expected_version` CAS 和审计路径；普通配置值可回显，Secret 只能写入或替换，
不会返回已有值。保存后需要滚动重启受影响的服务，配置不会热更新。

## 验证

测试至少覆盖：

- fresh schema 与升级迁移创建两个配置表；
- 全局、角色、实例作用域及显式覆盖的确定性优先级；
- bootstrap-only、未知字段和跨表字段拒绝；
- 配置 CAS、删除冲突和不含值的审计；
- secret 查询、`repr` 和错误日志不泄漏密钥；
- ProviderConfig/`credential_ref` 解析与 fail-close；
- Run Trace `secretConfig` 注入、缺失引用、明文冲突和密码脱敏；
- TDSQL batch parent 预建表、child 分片、重试隔离、动态表名校验和建表失败 fail-close；
- Console 公网地址、内部地址和鉴权模式覆盖规则；
- 太湖首访普通用户登记、并发幂等、无默认 Grant 以及已有管理员不被降级；
- DevCloud 无太湖 E2E，再在 runner-test 验证双太湖域名和用户隔离；不发布 runner-prod。
