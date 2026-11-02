# 使用 GPU 沙盒

GPU 沙盒**不是另一套沙盒协议**。Nexus 侧仍走原来的 Gongfeng 链路：`TaijiRuntimeProvider` 是 `GongfengRuntimeProvider` 的薄封装，`StartRuntime` / `GetRuntime` / `StopRuntime`、排队、健康探活、`invoke` / `run_command`、watchdog、runtime URL 和 executor 全部复用。task / agent / 工具调用不用改。

换的只是**底层机器和 GPU 卡**：调度到太极资源池，而不是工蜂或算力平台。配置项也沿用 Gongfeng 那一套（`runtime_provider_config` 字段名、语义与 [Gongfeng Runtime Provider](configure-runtime.md#使用-gongfeng-runtime-provider) 相同）。把已有 Gongfeng YAML 的 `runtime_provider_name` 换成 `taiji`，补上 GPU 卡数和卡型即可。`gongfeng.py` 里一部分能力太极机器暂未支持，原样带着会在 Nexus 侧直接拒绝，见下文「暂未支持的 Gongfeng 配置项」。

不要和 LLM 客户端里的 `openai_clients.name: taiji` 搞混。

```
Nexus task / agent
    │  acquire / invoke / stop   ← 原来的 Gongfeng 链路
    ▼
TaijiRuntimeProvider  （GongfengRuntimeProvider 的薄封装）
    │  sandbox v3 API（与 Gongfeng 相同）
    ▼
混元沙盒后端
    ├─ 调度：runtime_provider=taiji → 太极机器 + GPU 卡
    └─ 起来之后：runtime 进程、/health、工具调用 ← 原来的 Gongfeng 链路
```

GPU 卡数和卡型配额在 **太极平台应用组** 上，不走混元沙盒门户的 token GPU 配额。Nexus 只把请求调度到该应用组；组里没有对应卡，Start 会在太极侧失败或排队。

启动成功后，仍可在混元沙盒工作台的 [sandbox 列表](https://hy-sandbox-9tuxesn3.app.with.woa.com/?tab=sandbox) 里看到这条沙盒（按用户查）。工作台只是观测面，不在这里申请 token，也不在这里扣 GPU 配额。

## 开始前：配额与网络

1. **GPU 配额**：在太极平台给目标 **应用组** 配好 GPU 卡（卡型、卡数）。这不是沙盒门户 token 下的并发配额。
2. **网络**：运行机器必须能连通 prod 沙盒 API `http://hyrl-sandbox.prod.woa.com:8052`。当前仅支持 **idc 区**（推荐）与 **devcloud**（经 igate，稳定性不保证）；**办公网 OA 区暂不支持**。
3. **`purpose`**：每次启动必须传 `prod` / `eval` / `test` 之一，否则初始化阶段即报 `ValueError`。
4. **卡型必填**：`resource_spec.gpu_card_type` 必须是太极支持的卡型枚举，缺了或写错会在 Nexus 侧直接 `ValueError`，不会发到后端。

## 配置示例

Gongfeng 的 `runtime_provider_config` 可以原样复用（`api_base_url`、`token`、`image`、`runtime_image`、`resource_spec`、`purpose`、超时和排队等）。只需：

- `runtime_provider_name` 改成 `taiji`
- `resource_spec` 里补上 `gpu` 和 `gpu_card_type`
- 去掉太极暂未支持的项（见下一节），也不要再写 `runtime_provider:`（由 provider 钉死成 `taiji`）

```yaml
task_name: echo
task_config:
  message: "Hello, Taiji!"
  run_oracle: true

runtime_provider_name: taiji
runtime_provider_config:
  # 与 Gongfeng 相同：prod 沙盒 API
  api_base_url: "http://hyrl-sandbox.prod.woa.com:8052"

  # 与 Gongfeng 相同：调 start/get/stop 的认证。GPU 卡配额在太极应用组，不在这个 token 上
  token: "<your-token>"

  # 与 Gongfeng 相同：主容器镜像（也可在 configure / acquire 时传入）
  image: "mirrors.tencent.com/hunyuan_yanguan/sweb.eval.x86_64.django__asgiref-471"

  # 与 Gongfeng 相同：Runtime 镜像。推荐 "current"，与本地 nexus 包版本对齐
  runtime_image: "current"

  # 与 Gongfeng 相同：单次 Start/Get/Stop HTTP 超时（秒，默认 1800）
  request_timeout: 1800.0

  # 字段名与 Gongfeng 相同。GPU 沙盒必填 gpu + gpu_card_type（不要写成 gpuNum / gpuType）
  resource_spec:
    gpu: 1
    gpu_card_type: "H20"

  purpose: "prod"

runtime_configs:
  main:
    worker_venv: false
```

跑起来：

```bash
uv run nexus run-single your-gpu.yaml -o result.json
```

`echo` + `run_oracle: true` 不调 LLM，只验证沙盒能否拉起并在容器里写文件。任务结束后仍走原来的 Gongfeng `StopRuntime` 把沙盒停掉。

### 配置项与代码参数的对应关系

`runtime_provider_config` 会传给 `TaijiRuntimeProvider.__init__`，字段与 `GongfengRuntimeProvider` / `gongfeng.py` 同一套。除调度目标和 GPU 约束外，语义不变。

```python
from nexus.runtime.providers.taiji import TaijiRuntimeProvider

rp = TaijiRuntimeProvider(
    api_base_url='http://hyrl-sandbox.prod.woa.com:8052',
    token='<your-token>',
    image='mirrors.tencent.com/hunyuan_yanguan/sweb.eval.x86_64.django__asgiref-471',
    runtime_image='current',
    purpose='prod',
    resource_spec={'gpu': 1, 'gpu_card_type': 'H20'},
    request_timeout=1800.0,
)

runtime = await rp.acquire(name='main')
# 此后 invoke / run_command / release 与原来的 Gongfeng 沙盒相同
await rp.release('main')
```

`acquire()` 返回的仍是 Gongfeng 那套 `Runtime` 对象，连的是沙盒里的 nexus-runtime，不是太极控制台。

## GPU 规格

| 字段 | 必填 | 说明 |
|---|---|---|
| `resource_spec.gpu` | 是 | 卡数。用 sandbox 的 `gpu`，不要写 `gpuNum`。具体上下限由沙盒后端校验 |
| `resource_spec.gpu_card_type` | 是 | 卡型。用 sandbox 的 `gpu_card_type`，不要写 `gpuType` |

`gpu_card_type` 允许值（与太极 `gpuType` 枚举对齐，但 **YAML / Python 里仍写 sandbox 字段名**）：

`P40`、`M40`、`V100`、`EF_T10`、`T4`、`P4`、`ASCEND910B`、`A100`、`A100PRO`、`A800`、`H800`、`H20`、`L20`

以当前太极应用组实际可调度的卡型为准。

可选的 `cpu` / `memory` 与 Gongfeng 一样按核数、GB 理解；不填则由后端按 GPU 规格补默认值。

## 暂未支持的 Gongfeng 配置项

下面这些是 `gongfeng.py` / Gongfeng YAML 里已有的配置，太极机器暂未实现。`TaijiRuntimeProvider` 会在发请求前拒绝，避免请求已经落到太极却被静默丢掉：

| 参数 | 说明 |
|---|---|
| `enable_dind` | Docker-in-Docker |
| `cap_add` | 额外 Linux capability |
| `ttl` | 沙盒 TTL（Go duration） |
| `need_transparent_proxy` | 透明代理 |
| `network_policy` | 出网策略 |
| `runtime_provider` | 由本 provider 钉死为 `taiji`，再写 `gongfeng` / `suanli` 会报错 |

其余 Gongfeng 配置（`api_base_url`、`token`、`image`、`runtime_image`、`python_version`、`resource_spec`、`exposed_ports`、`working_dir`、`purpose`、`batch_id`、`request_timeout`、`queue_wait_max`、`watchdog_poll_interval`、`auto_connect_ws`、`socketio_request_timeout`、`rpc_protocol`、`print_port_mappings`、`startup_jitter_max` 等）都可以复用。

空值（`cap_add: []`、`ttl: ""`、`enable_dind: false`）视为未开启，不会拒绝。需要上述暂未支持的能力时，改回 `runtime_provider_name: gongfeng`。

排队（`queue_wait_max`）、watchdog、`request_timeout` 与 Gongfeng 相同，见 [被后端入队时的排队等待](configure-runtime.md#被后端入队时的排队等待) 和 [怎么读运行日志](configure-runtime.md#怎么读运行日志)。

## 怎么读运行日志

日志 logger 仍是 `nexus.runtime.providers.gongfeng`，因为 GPU provider 复用原来的 Gongfeng 客户端。看 `Sandbox '<name>':` 这一条主线即可：

- `Created runtime provider: taiji`：选中了 GPU 调度
- `creating (... resource_spec={'gpu': 1, 'gpu_card_type': 'H20'}, ...)`：Start 已提交，底层去太极抢卡
- `READY at http://<runtime_id>.19527.hyrl-sandbox.prod.woa.com:...`：沙盒起来了，之后的 invoke 走原来的 Gongfeng runtime 链路
- `stopping` / `released`：同样走 `StopRuntime`

DEBUG 下能看到 Start payload 里带 `runtime_provider: taiji`。不要只凭 logger 名字以为还在走工蜂机器。

## 下一步

- **[配置 Runtime · Gongfeng](configure-runtime.md#使用-gongfeng-runtime-provider)**：可复用的配置项、排队、日志、镜像更新（GPU 沙盒启动后共用同一条链路）
- **[Play with Gongfeng Runtime](play-with-gongfeng-provider.ipynb)**：同一套 `acquire` / executor 的交互演练；把 provider 换成 `TaijiRuntimeProvider` 并补上 `gpu_card_type` 即可打 GPU
- **[与运行环境交互](interact-with-runtime.md)**：Runtime 对象用法
