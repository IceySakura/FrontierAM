# Label 服务部署

本目录集中维护 `label` 服务的镜像构建与 Kubernetes 发布资产。Docker 构建上下文仍需使用仓库根目录，以便安装 Nexus、`infra` 和服务代码。


## 构建镜像

```bash
make service-build SERVICE=label
```

等价命令：

```bash
docker build -f deployment/services/label/Dockerfile -t nexus-label:local .
```

## Helm 校验

```bash
make service-helm-lint SERVICE=label
make service-helm-template SERVICE=label
```

## 安装或升级

沙盒 provider 凭据（AGS/DevCloud/SmartRun）不再通过 values 注入，而是由
七彩石配置中心下发。Helm 只负责创建七彩石**鉴权**所需的 Secret
（`RAINBOW_USER_ID` / `RAINBOW_SECRET_KEY`），仅在 `rainbow.enabled=true`
时生成。

启用七彩石并提供鉴权凭据。敏感值应放在仓库外的独立 values 文件（下文以
`<secrets-values>.yaml` 表示，请替换为自己环境中的实际路径），不得提交仓库：

```yaml
rainbow:
  enabled: true
  appId: 你的七彩石AppId
  group: 你的配置分组
  envName: Dev
  userId: 你的RAINBOW_USER_ID
  secretKey: 你的RAINBOW_SECRET_KEY
```

本地 kind 只维护 `nexus-dev` 环境。部署时引用本地 dev values：

```bash
helm upgrade --install nexus-label \
  deployment/services/label/helm/nexus-label \
  --namespace nexus-dev \
  --create-namespace \
  -f deployment/services/label/helm/nexus-label/values.dev.yaml
```

`values.dev.yaml` 含本地七彩石鉴权信息，已被 Git 忽略，不得提交仓库。

不接入七彩石的临时环境（provider 凭据留空、服务降级到默认值）可直接安装：

```bash
helm upgrade --install nexus-label \
  deployment/services/label/helm/nexus-label
```

也可以执行：

```bash
make service-deploy SERVICE=label
```

## 开启 OpenTelemetry

默认关闭。需要上报追踪时通过 `otel.enabled` 开启并指定 OTLP endpoint：

```yaml
otel:
  enabled: true
  endpoint: http://otel-collector:4318
```

开启后会自动注入 `NEXUS_OTEL_ENABLED=true` 与 `OTEL_EXPORTER_OTLP_ENDPOINT`。其余 `OTEL_*` 参数可通过 `extraEnv` 补充。

生产环境应通过独立 values 文件或发布系统覆盖镜像地址、标签和 provider 凭据。敏感配置不得提交到本目录。