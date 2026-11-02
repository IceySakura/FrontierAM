# Docker in Docker（DinD）

在 Nexus Gongfeng / gVisor 沙箱内启动独立的 dockerd，让任务可以直接使用 `docker build` / `docker run` / `docker compose`。外层沙箱仍由 gVisor 提供隔离。

```text
你的机器
└── gVisor 沙箱（外层容器）        ← 安全屏障
    └── 任务容器（含 dockerd）     ← 内层 Docker
        ├── 业务容器 A
        └── 业务容器 B
```

## 启用方式

在 `runtime_configs.<name>` 中设置 `enable_dind: true`：

```yaml
runtime_provider_name: gongfeng
runtime_provider_config:
  api_base_url: ${oc.env:GONGFENG_API_BASE_URL}
  token: ${oc.env:GONGFENG_TOKEN}
  runtime_image: ${oc.env:NEXUS_GONGFENG_RUNTIME_IMAGE,"current"}
  resource_spec:
    cpu: 4
    memory: 16

runtime_configs:
  main:
    image: mirrors.tencent.com/hunyuan_yanguan/ubuntu:24.04
    enable_dind: true        # 开启 DinD
```

完整示例见 [docs/examples/configs/dind_gongfeng.yaml](../../examples/configs/dind_gongfeng.yaml)。

沙箱启动后 dockerd 已就绪，任务里直接使用 `docker` 即可，无需自己安装 dockerd：

```bash
docker version
docker run --rm alpine echo ok
docker compose up -d
```

> `enable_dind` 默认 `false`，不开启的任务行为与普通沙箱一致。该能力仅支持 Gongfeng / gVisor 运行时，本地 Docker provider 会拒绝该参数。

## 内层容器出网

Gongfeng 集群默认不能直连公网，外层沙箱通过平台注入的 HTTP proxy、CA 证书和下载源环境变量出网。DinD 开启后，Nexus runtime 会通过 `dind-runc` 包装内层 runc，在内层容器创建时自动注入：

- `HTTP_PROXY` / `HTTPS_PROXY` / `NO_PROXY` 等 proxy 环境变量；
- `SSL_CERT_FILE` / `REQUESTS_CA_BUNDLE` / `NODE_EXTRA_CA_CERTS` 等 CA 环境变量和 CA bundle mount；
- `PIP_INDEX_URL` / `UV_INDEX_URL` 等下载源环境变量。

dockerd 进程启动时会把 `HTTP_PROXY` / `HTTPS_PROXY` / `NO_PROXY`（及小写）置为空字符串，`docker pull` / `push` 不走外层显式 HTTP 代理。官方镜像通过 `daemon.json` 的 `registry-mirrors` 拉取：`mirror.ccs.tencentyun.com`、`dockerhub.woa.com`、`csighub.woa.com`。内层 `docker run` / `compose` / BuildKit RUN 仍使用上面注入的快照。

该注入覆盖 `docker run`、`docker exec`、`docker compose` 和默认 BuildKit 的 `docker build` RUN 步骤。因此内层容器通常可以直接执行：

```bash
docker run --rm python:3.12-alpine python -c \
  "import urllib.request; print(urllib.request.urlopen('https://pypi.org').status)"

cat > Dockerfile <<'DOCKERFILE'
FROM python:3.12-alpine
RUN python -m pip install requests
DOCKERFILE
docker build -t example .
```

## docker compose 服务名访问与 no_proxy

`dind-runc` 会给内层容器注入 HTTP proxy。对于 `docker compose` 内部服务名访问，例如 `http://web/`、`http://db:8080/`，HTTP 客户端通常会先按主机名匹配 `no_proxy`，不会先解析成 Docker bridge IP。因此仅靠 Docker bridge CIDR 不足以让这类请求绕过 proxy；请求可能被发到外部代理，而代理无法解析 compose 服务名，导致失败。

如果 compose 服务之间需要通过 HTTP 服务名互访，请在对应 client 服务里显式设置服务名级别的 `no_proxy` / `NO_PROXY`：

```yaml
services:
  web:
    image: nginx:alpine
  client:
    image: buildpack-deps:bookworm-curl
    depends_on: [web]
    environment:
      - no_proxy=web
      - NO_PROXY=web
    command: bash -c 'curl -s -o /dev/null -w "%{http_code}" http://web/'
```

多个服务名用逗号分隔：

```yaml
environment:
  - no_proxy=web,api,db
  - NO_PROXY=web,api,db
```

这是 compose 文件和业务拓扑相关的信息，Nexus runtime 无法在框架层自动推断所有服务名。

## 已验证可用的功能

| 功能 | 状态 | 备注 |
| --- | --- | --- |
| `docker pull` / `run` / `exec` / `logs` / `inspect` | ✅ | 内层容器自动注入 proxy / CA / 下载源 env |
| `docker build`、BuildKit | ✅ | BuildKit RUN 容器也走 `dind-runc` |
| `docker buildx`（默认 driver, linux/amd64） | ✅ | 多架构未验证 |
| `docker compose up/down/ps/logs` | ✅ | 服务名 HTTP 互访需按上文设置 `no_proxy` |
| bind mount `-v /host:/c` | ✅ | |
| named volume、tmpfs 挂载 | ✅ | |
| `--privileged` / `--cap-add` | ✅ | 仅沙箱内有效 |
| `--memory` / `--cpus` 资源限制 | ✅ | |
| `docker save` / `load` / `history` | ✅ | |
| `docker network create` 自定义 bridge | ✅ | 容器间 IP 直连、service-name DNS 正常 |
| 容器间 service-name 通信 | ✅ | HTTP 客户端需注意 `no_proxy` |
| `--network=host` | ✅ | |
| `docker system prune` | ✅ | |
| 镜像拉取 | ✅ | dockerd 不走 HTTP 代理；官方镜像走 `registry-mirrors`：`mirror.ccs.tencentyun.com`、`dockerhub.woa.com`、`csighub.woa.com` |

## 已知限制

- `enable_dind` 与 transparent proxy sidecar 不是同一种内层出网机制，不应同时启用。
- 不读取 `HTTP_PROXY` / `HTTPS_PROXY` 的程序仍无法通过显式 proxy 方案出网。
- 不读取系统 CA 或 CA 环境变量的特殊程序可能仍需业务侧额外配置。
- compose 服务名 HTTP 互访需要在 compose 用例中显式设置服务名级别的 `no_proxy`。

## 快速自检

```bash
docker info                          # dockerd 是否就绪；Registry Mirrors 应含 tencentyun / dockerhub.woa.com / csighub.woa.com
docker run --rm alpine echo ok       # 内层容器能否拉起
docker run --rm python:3.12-alpine env | grep -iE 'PROXY|SSL_CERT_FILE|PIP_INDEX'
```
