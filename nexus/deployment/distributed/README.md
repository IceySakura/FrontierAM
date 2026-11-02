# Nexus Distributed 部署

Distributed runner 的 Kubernetes 部署封装成 Helm chart：

```text
deployment/distributed/helm/nexus-distributed
```

chart 默认部署 `predefined` InstanceManager，也可启用 Kubernetes 或 TKEx dynamic Environment。Reconciler 内的
EnvironmentController 按 allocation 创建和回收 Manager；InstanceManager 通过 MySQL heartbeat 注册，Scheduler 只从 MySQL
事实源发现容量并分配 execution。

部署使用三个 Tlinux 4 / Python 3.13 镜像角色：

- `deployment/distributed/Dockerfile.universal`：只安装 `nexus-agent[distributed]`，用于 Gateway、Scheduler、Reconciler 和 db-init。
- `deployment/distributed/Dockerfile.workload`：从 universal 增加 runtime，用于 single/live/batch Manager。
- `deployment/distributed/Dockerfile.builder`：从 workload 增加 C/C++/Rust 和 wheel 构建工具，用于一次性 dependency builder Job。

控制面使用标准 asyncio 事件循环；`uvloop-snapshot` 仅保留在 `runtime` 依赖集，universal 镜像不安装它。
`orjson` 仅在 `cloud` 依赖集中提供导出解析加速；universal 镜像不安装它，TDSQL trace 使用已有的标准库 JSON 回退。

可选的 `Dockerfile.universal.threading-free` 使用 Python 3.14.8t，沿用 universal 的路径、CLI 入口及 distributed/otel 依赖集。
构建时校验 free-threaded 解释器及主要原生依赖导入后的 GIL 状态，不设置 `PYTHON_GIL=0` 强制绕过兼容性检查。
在仓库根目录构建：

```bash
docker build -f deployment/distributed/Dockerfile.universal.threading-free \
  --build-arg NEXUS_BUILD_REVISION="$(git rev-parse HEAD)" \
  -t nexus-distributed-universal-threading-free:local .
```

`NEXUS_PYTHON_VERSION` 可覆盖为其他 free-threaded 版本（例如 `3.14.8t`）；构建工具默认固定为 uv 0.13.1。
SQLAlchemy 最低版本为 2.0.55，直接启用其兼容 free-threaded Python 的 C 扩展。
缺少 free-threaded wheel 的依赖在独立构建阶段编译，复用 `rpm/builder.txt`，编译工具不进入最终镜像。
该变体需显式选择；构建检查覆盖主要依赖，新增原生扩展仍需验证是否会重新启用 GIL。

`Dockerfile.control` 仅保留为 universal 的兼容别名。三个镜像使用同一 release tag，生产部署分别以 digest 固定内容。
镜像内 Python 依赖从 `uv.lock` 冻结导出并校验 package hash，Tlinux 基础镜像和直接安装的 RPM 版本固定；三层镜像默认统一使用
腾讯 PyPI mirror。发布构建必须传入 `NEXUS_BUILD_REVISION`，并分别用已推送的 universal/workload digest 作为下一层 `FROM`。
Source-package submission 上传确定性 `source.tar.zst`、当前 Nexus wheel 和 `uv.lock`；builder 生成离线 wheelhouse，dynamic Manager 再用
同一套 `VenvManager` 创建独立 Manager venv，不在 workload 启动时联网安装依赖。Builder 的 pip/uv index 由 Helm 统一配置，
默认使用腾讯 PyPI mirror，submission 不能覆盖。

完整部署说明见：

```text
deployment/distributed/helm/nexus-distributed/README.md
```

平台指标、状态接口和智研接入见 [平台可观测性](observability.md)。
