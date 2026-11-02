# Label Service

GUI 标注服务的首个 FastAPI 可运行骨架，通过 Uvicorn 启动。目前提供统一健康检查和服务信息接口，后续标注任务、人员分配、结果管理等业务能力在本目录内演进。

## 本地运行

在仓库根目录执行：

```bash
make service-sync
make service-run SERVICE=label
```

默认监听 `0.0.0.0:8080`，可通过以下环境变量配置：

- `NEXUS_LABEL_HOST`
- `NEXUS_LABEL_PORT`
- `NEXUS_LABEL_LOG_LEVEL`

接口：

- `GET /health/live`
- `GET /health/ready`
- `POST /api/v1/label/sandbox/acquire`
- `POST /api/v1/label/sandbox/release`

沙盒接口直接复用 `nexus.runtime.providers` 中的 AGS、DevCloud 和 SmartRun provider。provider 凭据与下游参数由**七彩石配置中心**下发（`provider` 配置分组），不再通过环境变量注入；拉取失败时降级到各字段默认值。七彩石里的结构对应 `ProviderConfig`：

- AGS：`ags.secretId`、`ags.secretKey`、`ags.toolId`（已有内置 waa 工具的 ToolId），以及可选的 `ags.region`、`ags.domain`
- DevCloud：`devcloud.appId`、`devcloud.appToken`、`devcloud.project`、`devcloud.buildId`、`devcloud.vmSeqId`，以及可选的 API 地址和 `devcloud.pipelineId`
- SmartRun：`smartrun.token`，以及可选的 `smartrun.apiBaseUrl`、`smartrun.duration`

七彩石鉴权本身所需的 `RAINBOW_*` 参数通过环境变量（部署时由 Helm Secret 注入）提供，不得提交到仓库。启动期参数（`NEXUS_LABEL_HOST` / `PORT` / `LOG_LEVEL` 等）仍走环境变量。

## 测试与镜像

```bash
make service-test SERVICE=label
make service-build SERVICE=label
```

Dockerfile 使用仓库根目录作为构建上下文，以便复用 `nexus-agent` 和 `services/packages/infra`。
