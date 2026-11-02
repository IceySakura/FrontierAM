# Configure Runtime

Nexus 设计上支持 Linux 容器、安卓、IOS、Windows 虚拟机等不同的运行时环境，目前对 Linux 容器的支持最为完善，其余部分开发中。 本节主要介绍 Linux 容器相关的配置。

## Runtime 与 Runtime Provider

在 Nexus 中，**Runtime** 是指 Agent 执行工具调用的环境。 为了保证安全性和可复现性，工具调用（如执行 bash 命令、文件操作等）通常会在独立的沙箱环境中运行，而不是直接在主进程中执行。 Nexus 也支持在本地直接执行这些操作（对应 Local Runtime），但不推荐用于生产环境。

**Runtime Provider** 负责管理 Runtime 的生命周期，包括创建、配置、启动、停止和销毁 Runtime 实例。 不同的 Runtime Provider 提供不同的底层实现。

简单来说：Runtime Provider 是"管理者"，Runtime 是"执行环境"。 你需要在配置文件中指定使用哪种 Runtime Provider，以及如何配置它创建的 Runtime 实例。

## 配置文件中的 Runtime 相关设置

在配置文件中，Runtime 相关的设置包括以下几个部分：

```yaml
# 指定 Runtime Provider 类型
runtime_provider_name: "docker"  # 或 "gongfeng" / "taiji"（GPU 沙盒）

# Runtime Provider 的配置参数
runtime_provider_config:
  # 这里的配置取决于所选的 provider
  # 参考下文的 Docker 或 Gongfeng 配置示例
  {}

# Runtime 实例的额外配置
runtime_configs: {}
```

### 配置参数的作用

配置文件中的参数在运行时会按以下方式使用（`nexus/runner/single.py:202-207`）：

```python
# 1. 使用 runtime_provider_config 创建 Provider 实例
provider_cls = runtime_provider_registry.get(run_input.runtime_provider_name)
runtime_provider = provider_cls(**run_input.runtime_provider_config)

# 2. 使用 runtime_configs 预配置每个 Runtime 实例的参数
for runtime_name, runtime_config in run_input.runtime_configs.items():
    runtime_provider.configure(runtime_name, **runtime_config)
```

- **`runtime_provider_config`**：传递给 Provider 初始化方法（`__init__`），作为 Provider 级别的全局配置
- **`runtime_configs`**：调用 `configure()` 方法预设每个 Runtime 实例的参数，这些参数会在调用 `acquire()` 时与传入的参数合并（`nexus/runtime/base.py:140-149`）

**两阶段参数传递机制**：

```python
# 阶段 1：通过 configure() 预配置默认参数，configure 可以调用多次
provider.configure('main', image='python:3.11', cpu=2)

# 阶段 2：通过 acquire() 传入额外/覆盖参数
runtime = await provider.acquire('main', memory='4G')
# 最终参数：{image: 'python:3.11', cpu: 2, memory: '4G'}
```

完整的配置文件示例请参考 [Tutorial](../get-started/tutorial.md)。

针对 Linux 容器， Nexus 提供了两种 Runtime Provider，用于管理隔离的运行时环境：

- **Docker Runtime Provider**：在本地 Docker 环境中运行沙箱，适合本地开发调试
- **Gongfeng Runtime Provider**：基于云端服务，提供一套标准的生命周期管理 API，同时负责底层不同沙箱运行时的资源统一抽象、调度与可观测。 支持大规模批量启动任务，适合生产环境和大规模评测场景。 GPU 沙盒（`runtime_provider_name: taiji`）也走这一套生命周期，只是底层机器和卡调度到太极，见 [使用 GPU 沙盒](use-gpu.md)

## 使用 Docker Runtime Provider

Docker Runtime Provider 在本地 Docker 环境中运行沙箱，适合本地开发调试。

### 配置示例

```yaml
runtime_provider_name: "docker"
runtime_provider_config:
  # 可选：主容器镜像
  image: "mirrors.tencent.com/hunyuan_yanguan/sweb.eval.x86_64.django__django-12155"

  # 可选：Runtime 镜像（提供 /nix 环境，默认 nexus-runtime:<package version>）
  runtime_image: "nexus-runtime:latest"

  # 可选：强制重新构建镜像（默认 false）
  force_rebuild: false

  # 可选：自定义构建上下文目录
  build_context: "/path/to/context"

  # 可选：Runtime 服务端口（默认 8000）
  runtime_port: 8000

  # 可选：额外暴露的端口列表，用于容器内服务对外访问
  # 场景：WebArena 浏览器访问容器内 Web 服务、Jupyter Notebook 等
  exposed_ports: [8023, 8080, 8888]

  # 可选：API 请求超时时间（秒，默认 1800）
  request_timeout: 1800.0

  # 可选：Socket.IO 握手/HTTP polling 请求超时时间（秒，默认 60）
  socketio_request_timeout: 60.0

  # 可选：沙箱资源规格配置
  resource_spec:
    cpu: 2      # CPU核数，不设置时 docker run 不加 --cpus 限制
    memory: 8   # 内存大小(GB)，不设置时 docker run 不加 --memory 限制

  # 可选：容器工作目录
  working_dir: "/testbed"

runtime_configs:
  main:
    # 可选：是否创建 Worker Venv（默认 false）
    # 设置为 true 时，框架自动将当前项目打包上传到沙盒，
    # 在隔离的虚拟环境中执行用户代码
    worker_venv: false
```

!!! tip "Worker Venv"

    `worker_venv: true` 会在沙盒中创建隔离的 Python 虚拟环境并启动 Worker 子进程。 适用于需要在沙盒中执行自定义 Python 代码的场景（如 Agent 的 `runtime_setup`）。 详见 [Worker Venv 配置指南](worker-venv-config.md)。

### 配置项与代码参数的对应关系

配置文件中的参数会在运行时传递给 `DockerRuntimeProvider` 的初始化方法（`nexus/runtime/docker.py:433`）。 以下示例展示了配置项如何对应到代码参数：

```python
from nexus.runtime import DockerRuntimeProvider

# 配置文件中的 runtime_provider_config 会传递给 __init__ 方法
rp = DockerRuntimeProvider(
    image="mirrors.tencent.com/hunyuan_yanguan/sweb.eval.x86_64.django__django-12155",
    runtime_image="nexus-runtime:latest",
    force_rebuild=False,
    build_context="/path/to/context",
    runtime_port=8000,
    exposed_ports=[8023, 8080, 8888],
    request_timeout=1800.0,
    resource_spec={"cpu": 2, "memory": 8},
    working_dir="/testbed"
)

# 调用 acquire 创建 Runtime 实例
runtime = await rp.acquire(
    name='main',
    image='ubuntu:24.04'  # 可以覆盖 provider 级别的 image
)
```

完整的使用示例请参考 [Play with Docker Runtime](../play-with-docker-provider)。

### 构建 Runtime 镜像

首次运行时，框架会自动构建 Runtime 镜像。 你也可以手动构建镜像：

```bash
nexus build-runtime
```

可选参数：

- `--image, -i`：指定镜像名称/标签（默认: `nexus-runtime:<package version>`）
- `--force, -f`：强制重新构建
- `--context, -c`：自定义构建上下文目录
- `--push`：构建后推送镜像到公司软件源 `mirrors.tencent.com`

推送镜像示例：

```bash
export DOCKER_REGISTRY_USERNAME=<your-username>
export DOCKER_REGISTRY_PASSWORD=<your-password>
nexus build-runtime --push
```

!!! note "镜像更新说明"

    如果在构建好镜像之后，工具实现相关的代码有更新，需要手动执行 `nexus build-runtime -f` 重新构建镜像。 但如果工具代码更新后版本号也同步更新了，则无需手动重新构建，框架会根据包版本自动构建新的镜像。

## 使用 Gongfeng Runtime Provider

Gongfeng Runtime Provider 提供一套标准的生命周期管理 API，同时负责底层不同沙箱运行时的资源统一抽象、调度与可观测。

**混元沙盒门户平台**：<https://sbox.woa.com/?tab=resource>，可用于查看和管理沙箱资源配额、监控沙箱运行状态。 使用沙箱前，请先在该平台申请 token，token 下可挂载不同类型的资源配额（如 CPU、GPU 等）。

如果有任何沙盒相关的使用问题，可以咨询沙盒平台的 **AI 助手**：<https://sbox.woa.com/?tab=ai-assistant>

沙盒接口的错误码、归因取值与对应处置动作由沙盒平台定义，见 [沙盒错误码与错误处理](https://iwiki.woa.com/p/4041018198)。Nexus 把这些字段原样透出到异常上，不另立一套错误语义，见下文「[接口失败与错误码](#接口失败与错误码)」。

底层沙盒运行时资源主要有三个来源：

- 工蜂沙箱
- 算力平台沙箱
- 太极 GPU 机器（只换机器和卡；沙盒起来之后仍复用 Gongfeng 的 start/get/stop、runtime 与工具调用，见 [使用 GPU 沙盒](use-gpu.md)）

### 开始前：配额与网络（首次接入必看）

首次在 nexus 上接入 Gongfeng 沙盒，按以下三步准备（带截图的图文步骤见 [iwiki 接入指引](https://iwiki.woa.com/p/4018105883)）：

1. **申请 token + 配额** ：在[沙盒门户](https://sbox.woa.com/?tab=resource) 申请 token，并在其名下申请并发配额（确认并发数与有效期）。 个人并发上限默认 1000，超出需走审批。
2. **确认网络可达** ：运行机器必须能连通 `hyrl-sandbox.prod.woa.com:8052` 。 当前仅支持 **idc 区**（推荐）与 **devcloud**（经企业 IT igate 网关，稳定性不保证，不建议生产正式场景；被 igate 拦截时可申请网络策略）；**办公网 OA 区暂不支持** 。
3. **声明 `purpose`**：Gongfeng / suanli 等非 Taiji provider 的 `purpose` 参与 token 配额桶校验，必须传 `prod` / `eval` / `test`。Nexus 不做必填强校验；未传则 Start 不带该字段。非法值初始化阶段报 `ValueError`。GPU 沙盒见 [使用 GPU 沙盒](use-gpu.md)。

启动后确认沙盒真的拉起来了：在[沙盒门户的 sandbox 列表](https://sbox.woa.com/?tab=sandbox) 按【用户】查询创建记录；或直接看 nexus 运行日志（见下文「怎么读运行日志」一节）。

### 配置示例

```yaml
runtime_provider_name: "gongfeng"
runtime_provider_config:
  # 必填：Gongfeng Runtime Provider 服务地址，非必要不要改动
  api_base_url: "http://hyrl-sandbox.prod.woa.com:8052"

  # 必填：认证 token（申请可参考：https://iwiki.woa.com/p/4018105883）
  token: "<your-token>"

  # 必填：主容器镜像（可在此处配置，也可通过 configure 或 acquire 传入，最终执行时必须提供）
  image: "mirrors.tencent.com/hunyuan_yanguan/sweb.eval.x86_64.django__django-12155"

  # 可选：Runtime 镜像（提供 /nix 环境）。推荐使用 "current"，会自动检测并传递本地 nexus 版本对应的镜像
  runtime_image: "current"

  # 可选：API 请求超时时间（秒，默认 1800）
  request_timeout: 1800.0

  # 可选：Socket.IO 握手/HTTP polling 请求超时时间（秒，默认 60）
  socketio_request_timeout: 60.0

  # 可选：资源调度策略 (gongfeng/suanli，通常是调试才需要指定。默认为空，表示自动选择最优调度策略)
  # 本示例下方启用了 cap_add / network_policy，两者非空时后端都会强制路由到 suanli，故这里写 suanli 保持一致
  runtime_provider: "suanli"

  # 可选：沙箱资源规格配置
  resource_spec:
    cpu: 2      # CPU核数，范围 0-32，默认 2（0 表示使用默认值，请按实际用量申请）
    memory: 8   # 内存大小(GB)，范围 0-64，默认 8（0 表示使用默认值，请按实际用量申请）
    # gpu: 1    # GPU卡数，默认为0，非必须不要指定（需先在混元沙盒门户申请 token 及对应资源配额：https://sbox.woa.com/?tab=resource）
    # gpu_card_type: "A100"  # 可选，期望的 GPU 卡型，仅在 gpu >= 1 时有意义（gpu=0 的容器沙盒填了会被算力平台拒绝）
                             # 取值为算力平台支持的卡型名，纯透传，不影响 provider 路由

  # 可选：额外暴露的端口列表
  # 用于容器内服务对外访问（如 WebArena 场景的浏览器访问容器内 GitLab/电商站点）
  # 返回的 port_mappings 会包含每个端口对应的外部访问 URL
  exposed_ports: [8023, 8080, 8888]

  # 可选：出网网络策略（默认不开启）。非空时后端强制路由到 suanli provider
  # （若同时指定了非 suanli 的 runtime_provider，会被后端静默覆盖为 suanli 并打 WARN 日志，不报错）
  # - default_action: 未命中任何 egress 规则时的默认动作（allow / deny，省略视为 allow）
  # - egress: 出站规则列表，按顺序匹配，首条命中即生效
  # - 启动路径不接受 proxy。代理路由规则须通过 update_runtime 下发
  # - target 支持：精确域名 example.com / 通配域名 *.example.com / IPv4 1.2.3.4
  #   / CIDR 10.0.0.0/8 / 带端口 api.example.com:443。不是 URL（勿带 scheme/path）；
  #   CIDR 不支持端口；暂不支持 IPv6
  # 完整说明见官方文档：https://iwiki.woa.com/p/4021452048
  network_policy:
    default_action: deny           # allow / deny
    egress:
      - action: allow              # allow / deny
        target: "*.tencent.com"

  # 可选：需要开启的透明代理协议列表（默认不开启）。取值 http / https / tcp，最多 3 个
  # 设置后后端在算力平台开启 NetworkSidecarEnabled
  need_transparent_proxy: ["http", "https"]

  # 可选：追加给沙盒主容器的 Linux capabilities 列表（默认不开启，仅 suanli 支持）
  # 非空时后端强制路由到 suanli provider（若同时指定了非 suanli 的 runtime_provider，会被后端静默覆盖为 suanli 并打 WARN 日志，不报错）
  # 取值为 capability 名，大写下划线风格，不含 CAP_ 前缀。示例：["NET_ADMIN", "SYS_PTRACE"]
  # 常见值：NET_ADMIN(网络配置)、SYS_PTRACE(调试/strace)、SYS_ADMIN(挂载等特权操作)、NET_RAW(原始套接字)
  # 错误示例：["CAP_NET_ADMIN"]（不能带 CAP_ 前缀）、["net_admin"]（不能小写）
  cap_add: ["NET_ADMIN"]

  # 可选：关闭 runtime 容器 ASLR（默认 false，仅 suanli / sandbox-sdk 支持）
  # 为 true 时后端强制路由到 suanli provider（若同时指定了非 suanli 的 runtime_provider，会被后端静默覆盖为 suanli 并打 WARN 日志，不报错）
  runtime_disable_aslr: false

  # 可选：沙盒生存时间（不指定时为 12h）
  # 取值为 Go duration 字符串，单位为 s / m / h。示例："30m"、"12h"、"1h30m"、"168h"
  # 上限 168h（7 天），超过直接报错、不会静默截断
  # 错误示例："7d"（Go duration 没有天单位，7 天写成 "168h"）、"12"（缺单位）、"500ms"（亚秒级不受理）
  # 到期时沙盒被回收，nexus 侧由健康探活发现，报的是连续探活失败（数分钟后），不是"ttl 到期"；
  # 启动日志的 Sandbox 'x': creating (... ttl=...) 一行会回显本次生效值，便于对上号
  ttl: "12h"

  # 可选：容器工作目录
  working_dir: "/testbed"

  # 可选：Python 版本（默认自动匹配客户端版本）
  python_version: "3.12"

  # 可选：批次标识，用于按批次查询或停止同一批沙盒（如 sbox stop runtime --batch my-batch-123）。不传则不带该字段
  batch_id: "my-batch-123"

  # 必须传（配额桶），enum: prod / eval / test。Nexus 不强制校验，未传则 Start 不带该字段
  # - prod: Production tasks, typically launched via the Hunyuan platform
  # - eval: Evaluation tasks, typically launched via the evaluation platform
  # - test: Offline testing / experimentation / validation tasks
  purpose: "test"

  # 可选：被后端入队（status=queued）时客户端的最长轮询等待秒数。
  # 默认 0 = 无客户端截止，一直轮询到 scheduled 或后端放弃；> 0 则封顶等待时长。
  # 是否入队由服务端决定（token 端 quota_full_policy=queue 时配额满会入队，否则 reject 直接报错）。
  # 详见下文「被后端入队时的排队等待」一节。
  queue_wait_max: 0

  # 可选：启动失败且错误码可重试时，由 Nexus 换新 runtime_id 重建一次，默认 false。
  # 开启后不要再在上层（batch retry_failed、自写循环）重试同一失败，详见下文「启动失败自动重建」。
  start_retry: false

  # Docker-in-Docker（enable_dind）的配置见 docker-in-docker.md，此处不重复
runtime_configs: {}
```

### 配置项与代码参数的对应关系

配置文件中的参数会在运行时传递给 `GongfengRuntimeProvider` 的初始化方法（`nexus/runtime/gongfeng.py:78`）。 以下示例展示了配置项如何对应到代码参数：

```python
from nexus.runtime import GongfengRuntimeProvider

# 配置文件中的 runtime_provider_config 会传递给 __init__ 方法
rp = GongfengRuntimeProvider(
    api_base_url="http://hyrl-sandbox.prod.woa.com:8052",
    token="<your-token>",
    image="mirrors.tencent.com/hunyuan_yanguan/sweb.eval.x86_64.django__django-12155",
    runtime_image="current",  # 自动使用本地 nexus 版本对应的镜像
    request_timeout=1800.0,
    # 本示例启用了 network_policy / cap_add，两者非空都会强制路由 suanli，故这里写 suanli 保持一致（默认为空则自动选择）
    runtime_provider="suanli",
    resource_spec={"cpu": 2, "memory": 8},  # gpu 默认为0，非必须不要指定；GPU 场景可加 "gpu_card_type": "A100"
    exposed_ports=[8023, 8080, 8888],
    network_policy={  # 可选：出网网络策略，非空时强制路由 suanli（若传了非 suanli 的 runtime_provider 会被静默覆盖为 suanli，不报错）
        "default_action": "deny",  # 未命中任何 egress 规则时的默认动作；省略视为 allow
        # target 支持 域名/*.通配/IPv4/CIDR/带端口（详见 https://iwiki.woa.com/p/4021452048）
        # 启动路径不接受 proxy，须在沙盒运行后通过 update_runtime 下发
        "egress": [{"action": "allow", "target": "*.tencent.com"}],
    },
    need_transparent_proxy=["http", "https"],  # 可选：透明代理协议，最多 3 个
    cap_add=["NET_ADMIN"],  # 可选：追加 Linux capabilities，非空时强制路由 suanli（若传了非 suanli 的 runtime_provider 会被静默覆盖为 suanli，不报错）
    runtime_disable_aslr=False,  # 可选：关闭 runtime 容器 ASLR，true 时强制路由 suanli
    ttl="12h",  # 可选：沙盒生存时间，Go duration（"7d" 不合法，7 天写 "168h"），上限 168h，不指定为 12h
    working_dir="/testbed",
    python_version="3.12",
    batch_id="my-batch-123",  # 可选，不传则不带该字段；需要按批次查询或停止时显式传入
    purpose="test",  # 必须传（配额桶），enum: prod / eval / test；Nexus 不强制校验，未传则 Start 不带该字段
    queue_wait_max=0,  # 可选，入队时客户端最长轮询等待秒数（0=无截止，一直等到 scheduled）
    start_retry=False,  # 可选，启动失败且错误码可重试时换新 runtime_id 重建一次；开启后上层不要再重试
)

# 调用 acquire 创建 Runtime 实例
runtime = await rp.acquire(
    name='main',
    image='mirrors.tencent.com/hunyuan_yanguan/tbench2__mailman__ruha8w',
    # queue_wait_max=300,  # 可选：单次 acquire 临时覆盖 provider 默认值
    # ttl="30m",  # 可选：单次 acquire 临时覆盖 provider 默认值；传 "" 表示退回默认生命周期
)
```

完整的使用示例请参考 [Play with Gongfeng Runtime](../play-with-gongfeng-provider)。

### 运行期更新网络策略

沙盒启动后可通过 `update_runtime` 更新出网策略。按配置面替换：未传入的配置面保持不变；传入的配置面整体替换，不是合并或追加。当前唯一可更新的配置面是 `network_policy`。

```python
await rp.update_runtime(
    'main',  # acquire / configure 时使用的 runtime 名称
    network_policy={
        'default_action': 'deny',                                   # 所有规则均未命中时的默认动作，省略视为 allow
        'egress': [{'action': 'allow', 'target': '*.pypi.org'}],    # 访问控制，先于代理路由求值
        'proxy': [                                                  # 代理路由，仅本方法支持；创建时传入会报错
            {'action': 'direct', 'target': '10.0.0.0/8'},           # direct 直连，不得携带 to
            {'action': 'proxy', 'target': '0.0.0.0/0',              # proxy 经上游转发，to 至少指定一项
             'to': {'http': 'http://1.2.3.4:3180', 'https': 'https://1.2.3.4:3444',
                    'default': 'socks5://1.2.3.4:3280'}},
        ],
    },
)
```

- `egress` 与 `proxy` 均按列表顺序求值，首条命中即返回。具体规则置于列表前部，通配规则置于末尾。访问控制先于代理路由求值：`egress` 命中 `deny` 时立即拒绝，不再匹配 `proxy`
- `target` 格式与创建路径相同：精确域名、`*.` 前缀通配域名、IPv4、CIDR；前三类可附端口。上游地址格式为 `scheme://IPv4:port`，`scheme` 取值 `tcp` / `http` / `https` / `socks5`
- 恢复为允许全部出网时传入 `{'default_action': 'allow'}`。传入 `{}` 视为未指定任何配置面，接口返回参数错误
- 适用条件：算力平台沙盒（`suanli` / `suanli.pool.*`），且创建时指定了 `need_transparent_proxy`。工蜂、太极沙盒，以及未启用透明代理的沙盒返回 `260400` + `Runtime.NetworkPolicyNotSupported`，须重建沙盒并指定 `need_transparent_proxy`
- 策略即时生效，但不持久化。沙盒内代理组件重启后，生效策略回退为创建时下发的配置，须再次调用本方法

字段与错误码见[沙盒 v3 接口文档](https://iwiki.woa.com/p/4021798953)的「运行期更新沙盒配置」。

### 被后端入队时的排队等待

客户端是**响应式**的：只要 `StartRuntime` 返回 `status=queued`（url 为空），`acquire` 就会自动轮询 `GetRuntime` 等待沙盒被调度——客户端不关心后端为什么入队，是否入队完全由服务端决定。

**后端何时返回 queued：**

**当前（已上线）**：只有 token 配额排队会触发。在 [混元沙盒门户](https://sbox.woa.com/?tab=resource) 上把 token 的 `quota_full_policy` 设为 `queue`（默认 `reject`）后，配额满时 `StartRuntime` 不再立刻报错，而是把请求落到后端 `runtime_queue` 表里排队，由 `sandbox-offline` worker 在其它沙盒释放配额后按 FIFO 顺序调度，**单条记录最长保留 48h**（超时会被自动清理并标记 `remark=queue_timeout`）。若 token 仍是 `reject`，配额满会直接返回错误，`acquire` 照旧抛 `RuntimeError`。

> **未来（暂未上线）**：后端高负载限流也会以同样方式返回 queued（`url=""`）——届时即便 token 是 `reject`，系统在资源紧张时也可能把低优先级请求透明入队等待。客户端无需为此改动：判定只看 `url` 是否为空，这条新触发源上线后会被现有轮询逻辑自动接住。

**客户端如何等待（`queue_wait_max`）：**

拿到 queued 响应后，`acquire` 用 `~15s ±25% jitter` 的间隔轮询 `GetRuntime`，直到 `status=scheduled` 拿到 url。`queue_wait_max=0`（默认）表示**无客户端截止时间**，一直轮询到调度成功或后端放弃；设为 `> 0` 则封顶等待时长，耗尽后主动取消队列条目并抛 `RuntimeQueueWaitTimeoutError`。

```python
rp = GongfengRuntimeProvider(
    api_base_url="http://hyrl-sandbox.prod.woa.com:8052",
    token="<queue-policy-token>",
    image="...",
    queue_wait_max=600,  # 客户端最多等 10 分钟
)

# acquire 期间如果配额满，会自动阻塞等待最多 600s
runtime = await rp.acquire(name='worker-1')

# 也可以按单次任务覆盖：
runtime = await rp.acquire(name='worker-2', queue_wait_max=3600)  # 这一次愿意等 1 小时
```

#### 两种排队专用异常

排队等待引入了 2 个专用异常，方便上层差异化处理（重试 / 降级 / 直接报错）。接口本身返回错误码时抛的是 `SandboxApiError`，见「[接口失败与错误码](#接口失败与错误码)」。

| 异常 | 触发条件 | 典型处理 |
|---|---|---|
| `RuntimeQueueWaitTimeoutError` | 设置了正的 `queue_wait_max` 且耗尽时沙盒仍排队中（客户端已经主动 cancel 队列条目，不会泄漏） | 放弃本次任务或调大 `queue_wait_max` |
| `RuntimeQueueFailedError` | 后端 worker 把队列条目标记为 `queue_timeout`（等满 48h 未调度）或 `queue_give_up`（worker 起 Pod 连续失败）后告知客户端 | 检查 quota / 镜像 / `runtime_provider` 配置，通常需要人工介入 |

异常导入示例：

```python
from nexus.runtime.providers.gongfeng import (
    RuntimeQueueWaitTimeoutError,
    RuntimeQueueFailedError,
)

try:
    runtime = await rp.acquire(name='worker-1')
except RuntimeQueueWaitTimeoutError:
    logger.warning('queued too long, skipping this task')
except RuntimeQueueFailedError as e:
    logger.error('backend gave up on the queued runtime: %s', e)
```

#### 注意事项

- **`queue_wait_max` 与 `request_timeout` 是两件不同的事**：`request_timeout`（默认 1800s）控制单次 HTTP RPC 的超时，`queue_wait_max` 控制 polling 累计的墙钟时间，两者正交独立
- **`socketio_request_timeout` 是 Socket.IO 握手请求超时**：默认 60s，主要影响 `/socket.io/?transport=polling` 的 Engine.IO 连接请求，不改变 `invoke` 的 HTTP RPC 超时
- **轮询间隔写死 15s**（带 ±25% jitter）：这是实现细节，不通过配置暴露——估算上能在用户感知 ≤ 30s 与后端管理面压力之间取得平衡，且常见的 worker 调度延迟也在两个 poll 周期内
- **`acquire` 期间 release/cancel 是安全的**：polling 中收到 `asyncio.CancelledError` 时客户端会 best-effort 调一次 `/runtime/stop` 取消队列条目，不会泄漏后端 `runtime_queue` 表行
- **`acquire` 阻塞期间也算这个名字占用**：和已就绪的 runtime 一样，`release(name)` 同样可以 cancel 还在排队的请求（后端会把队列条目状态置为 `cancel`）

### 接口失败与错误码

沙盒接口（start / get / stop / list）返回非 `0` 的 `code` 时，provider 抛 `SandboxApiError`。它继承 `RuntimeError`，既有 `except RuntimeError` 的调用点不受影响。

错误码、归因取值与处置动作以沙盒平台的 [沙盒错误码与错误处理](https://iwiki.woa.com/p/4041018198) 为准。Nexus 不另立错误语义，只把契约字段挂到异常上，并按 `code` 附一句下一步该做什么：

| 属性 | 含义 |
|---|---|
| `code` | 沙盒错误码，决定处置动作（改请求 / 退避重试 / 停止重试） |
| `reason` | 归因，`<域>.<原因>` 两段式，如 `Image.NotFound`。后端无法归类时为空 |
| `msg` | 后端的可读说明，已做敏感值脱敏 |
| `trace_id` | 报障标识，可能为空 |
| `exit_code` | 容器退出码，仅沙盒容器确实终止时非零（`137` 内存超限或被强杀，`143` 收到终止信号） |

异常消息把这些拼成一行，便于直接贴进报障：

```text
Gongfeng API error (code=260503, reason=Image.PullError): failed to create sandbox: ... | next step: transient platform-side condition — back off and retry the same request | trace_id=abc123
```

按属性分支处理，不要解析消息文本：

```python
from nexus.runtime.providers.gongfeng import SandboxApiError

try:
    runtime = await rp.acquire(name='worker-1')
except SandboxApiError as e:
    if e.code == 260429:  # 配额或频率受限
        logger.warning('quota or rate limited: reason=%s', e.reason)
    else:
        logger.error('sandbox api failed: code=%s reason=%s trace_id=%s', e.code, e.reason, e.trace_id)
    raise
```

#### 启动失败自动重建

`start_retry` 默认关闭：`acquire()` 只发一次创建请求，失败即抛出 `SandboxApiError`，是否重试由调用方决定。每次 `acquire()` 都使用新的 `runtime_id`，上层重试不会与失败的那次冲突。

开启后，创建失败且错误码属于契约规定的可重试范围时，Nexus 先停止失败的沙盒，等待 30～45 秒，再用新的 `runtime_id` 重建一次：

| code | reason | 是否重建 |
|---|---|---|
| `260503` | 任意 | 是 |
| `260504` | 任意 | 是 |
| `260429` | `Quota.RateLimited` | 是 |
| 其余 | 任意 | 否，直接抛出 |

契约规定同一失败至多重建一次。开启 `start_retry` 后，**不要再在上层重试同一失败**，包括 batch 的 `retry_failed` 和包在 `acquire()` 外的循环。否则每层的重试次数相乘，平台限流期间会放大创建压力。重建后仍失败时，异常消息末尾会追加以下提示，并打一条 WARNING 日志：

```text
... | nexus already recreated this sandbox once (start_retry); do not retry it at a higher layer
```

可以在 provider 级统一开启，也可以按沙盒覆盖。`runtime_configs` 的键是沙盒名，即任务代码里 `ctx.get_runtime(name)` 使用的名字：

```yaml
runtime_provider_config:
  start_retry: false       # provider 级默认值，对所有沙盒生效
runtime_configs:
  main:                    # 对应 ctx.get_runtime('main')
    start_retry: true      # 只对 main 这个沙盒开启
```

运行中的沙盒死亡由 watchdog 探测，不走异常而是把 runtime 标记为不可用，在场任务快速失败。WARNING 日志形如：

```text
Sandbox 'worker-1' (nexus-xxx) DIED: Process.OOMKilled (status=502, exit_code=137); in-flight tasks will fail fast | next step: raise the memory spec or lower the sandbox memory usage, then rebuild | trace_id=abc123
```

判死只认「确实回不来了」的归因：容器被 OOM 终止、主进程已退出、沙盒记录已停止。端口尚未监听、建连超时这类不确定的失败不立即判死，需连续多次探测失败才判，避免代理抖动误杀在跑的任务。

### 怎么读运行日志

provider 日志分两层，默认只输出 INFO ：

- **INFO = 每个沙盒一条人话生命周期主线** ，统一 `Sandbox '<name>':` 前缀，grep 名字即可看全程：`creating`（提交创建，带 image / 资源规格）→ `QUEUED` + `still queued for resources, Ns so far (not stuck)`（排队中，每 60s 一条心跳，明示「正常排队、非卡住」）→ `started after queuing Ns` / `READY at <url>`（已启动并运行）→ `stopping` → `released` 。 异常分支：`QUEUE FAILED`（排队被后端放弃）、`DIED`（运行中的沙盒被判死，watchdog 探测到，WARNING 级别，消息带归因与下一步，见 [接口失败与错误码](#接口失败与错误码)）。
- **DEBUG = 原始 API 报文** ：`/runtime/start`、`/stop`、`/get` 的完整请求与响应 body 。 排查「后端到底返回了什么」时再把日志级别调到 DEBUG 。 默认不输出——高频的 watchdog 巡检（每 ~120s/沙盒）与排队轮询（每 ~15s）会刷屏，且原始 payload 里可能含 `environment` 的密钥，不宜进 INFO 。

> **`status=scheduled` 不是「排队等待」** ：这是后端 API 的状态字段，表示沙盒**已被调度并启动、正在运行**（排队等待的状态是 `queued`）。 该词只出现在 DEBUG 的原始报文里；INFO 主线一律用 `started` / `running` / `READY` 表述，避免误读。

**watchdog 健康巡检约定** ：沙盒就绪后会启动一个后台 watchdog，每 ~120s 探一次状态，只在**探测到故障**时发声（`DIED`），健康巡检静默——所以 acquire 后会先打一条 `health watchdog started (...)` 把这个契约讲明。

### Runtime 远程镜像更新

!!! note "适用范围"

    本节内容仅适用于使用 Gongfeng Runtime Provider 的场景。 如果你使用的是 Docker Runtime Provider，镜像会在运行时自动构建。

根据不同的开发阶段，有两种更新 Runtime 远程镜像的方式：

#### 阶段一：开发调试阶段（Feature 分支）

当你在 feature 分支上开发，需要更新 Runtime 镜像进行调试时：

**方案1：通过流水线构建（推荐）**

直接执行流水线：https://zhiyan.woa.com/qci/18172/pipeline/173384/envType/prod#/pipeline/detail/11550690/build/current

注意：必须指定你的开发分支，流水线会根据分支名称自动生成镜像 tag（分支名中的 `/` 会替换为 `-`）

**方案2：本地构建并推送**

```bash
nexus build-runtime --image nexus-runtime:$IMAGE_TAG --push
```

注意：

- 执行前需要到 https://mirrors.tencent.com/#/private/docker 获取访问 token
- 添加 `DOCKER_REGISTRY_USERNAME` 和 `DOCKER_REGISTRY_PASSWORD` 到环境变量中，否则无法上传

**更新配置**

镜像推送完成后，在配置文件（如 `example.yaml`）中配置你刚刚构建的镜像：

```yaml
runtime_provider_config:
  runtime_image: "mirrors.tencent.com/hunyuan_yanguan/nexus-runtime:$IMAGE_TAG"
```

!!! tip "查看已有镜像"

    `IMAGE_TAG` 为你的开发分支名，如 `feat-task-swebench`。 可以到 https://mirrors.tencent.com/#/private/docker/detail?project_name=hunyuan_yanguan&repo_name=nexus-runtime 查看已有镜像。

#### 阶段二：主干合并阶段（Main 分支）

!!! warning "1.0 版本策略变更"

    从 1.0 开始，Runtime 镜像 **不再维护 `latest` 标签**。 详见 [Runtime 镜像版本策略](../../explanation/runtime-image-versioning.md)。

当你的 **MR 合并到 main 分支** 且 **更新了版本号** 后，流水线会自动触发构建并推送带版本号的镜像（如 `nexus-runtime:1.0.0`）。 同版本号不会重复推送——如果有 Runtime 侧内容更新，需要升级版本号。

对于测试需求，推荐以下方式：

- **Worker Venv**：设置 `worker_venv: true`，业务代码通过 artifact 动态注入沙盒，无需更新镜像
- **COS 中转**：将 artifact 上传到 COS，沙盒从 COS 下载
- **直连上传**：通过 Worker Venv 直接上传到沙盒

非 main 分支的镜像不保证超大规模并发稳定性。

## 下一步

- **[Play with Docker Runtime](play-with-docker-provider.ipynb)**：Docker Runtime Provider 完整使用示例
- **[Play with Gongfeng Runtime](play-with-gongfeng-provider.ipynb)**：Gongfeng Runtime Provider 完整使用示例
- **[使用 GPU 沙盒](use-gpu.md)**：底层用太极的机器和 GPU 卡，启动后仍走 Gongfeng 链路
- **[与运行环境交互](interact-with-runtime.md)**：了解如何通过 Runtime 对象与运行环境进行交互
