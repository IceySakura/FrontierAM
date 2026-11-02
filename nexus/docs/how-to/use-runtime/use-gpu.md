# 使用 GPU 沙盒

GPU 沙盒沿用既有沙盒协议。Nexus 侧复用 Gongfeng 链路：`TaijiRuntimeProvider` 是 `GongfengRuntimeProvider` 的薄封装，`StartRuntime` / `GetRuntime` / `StopRuntime`、排队、健康探活、`invoke` / `run_command`、watchdog、runtime URL 与 executor 均复用。task / agent / 工具调用无需修改。

差异仅在于底层机器与 GPU 卡：调度到太极资源池，而非工蜂或算力平台。配置项沿用 Gongfeng（`runtime_provider_config` 字段名与语义见 [Gongfeng Runtime Provider](configure-runtime.md#使用-gongfeng-runtime-provider)）。将已有 Gongfeng YAML 的 `runtime_provider_name` 改为 `taiji`，并配置 GPU 卡数、卡型和应用组。

请勿与 LLM 客户端中的 `openai_clients.name: taiji` 混淆。

```
Nexus task / agent
    │  acquire / invoke / stop   ← Gongfeng 链路
    ▼
TaijiRuntimeProvider  （GongfengRuntimeProvider 的薄封装）
    │  sandbox v3 API（与 Gongfeng 相同）
    ▼
混元沙盒后端
    ├─ 调度：runtime_provider=taiji → 太极机器 + GPU 卡
    └─ 启动后：runtime 进程、/health、工具调用 ← Gongfeng 链路
```

GPU 卡数与卡型配额归属太极应用组，不占用沙盒门户 token 的 GPU 配额。应用组无对应卡型或额度不足时，Start 在太极侧失败或排队。CPU / 内存超出卡数对应份额的请求由沙盒后端在入口拒绝（见 [CPU / 内存上限按卡数比例折算](#cpu--内存上限按卡数比例折算)），不进入太极排队。配额与运行中实例见 [沙盒平台 GPU 页](https://sbox.woa.com/?res_tab=gpu)；按用户检索单条实例仍使用 [sandbox 列表](https://sbox.woa.com/?tab=sandbox)，该页不扣减 GPU 配额。

`token` 必填：在 [混元沙盒门户](https://sbox.woa.com/?tab=resource) 申请，用于调用 start/get/stop。Nexus 仅校验 token 可解析为非空，不校验 token 名下配额。

## 开始前：配额与网络

1. **申请 token**（必填）：[混元沙盒门户](https://sbox.woa.com/?tab=resource)。图文步骤见 [iwiki 接入指引](https://iwiki.woa.com/p/4018105883)。
2. **应用组与 GPU 配额**（必填）：从 [沙盒平台 GPU 页](https://sbox.woa.com/?res_tab=gpu) 读取应用组名称与地域短码，分别写入 `taiji_app_group` / `taiji_app_group_location`。缺少任一字段时，Nexus 抛出 `ValueError`。
3. **网络**：运行机器须能访问 prod 沙盒 API `http://hyrl-sandbox.prod.woa.com:8052`。当前仅支持 **idc 区**（推荐）与 **devcloud**（经 igate，稳定性不保证）；**办公网 OA 区暂不支持**。
4. **卡型必填**：`resource_spec.gpu_card_type` 缺失时，Nexus 抛出 `ValueError`；取值非法由沙盒后端在 `StartRuntime` 入口拒绝，并在错误信息中给出当前可用卡型。
5. **`purpose` 推荐填写**：`prod` / `eval` / `test`。Taiji 侧用于审计，不参与 token 配额校验。未填写时 Start 请求不包含该字段。

| 沙盒平台 GPU 页 | YAML / Python 字段 | 取值 |
|---|---|---|
| 应用组名称 | `taiji_app_group` | 应用组 id，例如 `TaiJi_HYAide_T3_Nexus_Sandbox`；使用英文标识，勿使用中文展示名 |
| 地域 | `taiji_app_group_location` | 地域短码，例如 `gy`；勿使用中文地名 |

## 配置示例

Gongfeng 的 `runtime_provider_config` 字段可复用。另需配置：

- `runtime_provider_name` 设为 `taiji`
- `resource_spec` 配置 `gpu` 与 `gpu_card_type`
- 配置必填的 `taiji_app_group` / `taiji_app_group_location`
- 请勿填写 `runtime_provider:`（本 provider 将其固定为 `taiji`）

```yaml
task_name: echo
task_config:
  message: "Hello, Taiji!"
  run_oracle: true

runtime_provider_name: taiji
runtime_provider_config:
  # 与 Gongfeng 相同：prod 沙盒 API
  api_base_url: "http://hyrl-sandbox.prod.woa.com:8052"

  # 必填 https://sbox.woa.com/?tab=resource
  token: "<your-token>"

  # 与 Gongfeng 相同：主容器镜像（也可在 configure / acquire 时传入）
  image: "mirrors.tencent.com/hunyuan_yanguan/sweb.eval.x86_64.django__asgiref-471"

  # 与 Gongfeng 相同：Runtime 镜像。推荐 "current"，与本地 nexus 包版本对齐
  runtime_image: "current"

  # 与 Gongfeng 相同：单次 Start/Get/Stop HTTP 超时（秒，默认 1800）
  request_timeout: 1800.0

  # 字段名与 Gongfeng 相同。GPU 沙盒必填 gpu + gpu_card_type（请勿使用 gpuNum / gpuType）
  resource_spec:
    gpu: 1
    gpu_card_type: "H20"

  # 推荐。Taiji 用于审计，不参与 token 配额校验
  purpose: "test"

  # 必填。取值见 https://sbox.woa.com/?res_tab=gpu
  taiji_app_group: "TaiJi_HYAide_T3_Nexus_Sandbox"
  taiji_app_group_location: "gy"

runtime_configs:
  main:
    worker_venv: false
```

启动：

```bash
uv run nexus run-single your-gpu.yaml -o result.json
```

`echo` + `run_oracle: true` 不调用 LLM，仅验证沙盒能否启动并在容器内写入文件。任务结束后通过 Gongfeng `StopRuntime` 停止沙盒。

### 配置项与代码参数的对应关系

`runtime_provider_config` 传入 `TaijiRuntimeProvider.__init__`，字段与 `GongfengRuntimeProvider` / `gongfeng.py` 相同。除调度目标、GPU 卡型约束、ttl 上限和应用组字段外，语义不变。

```python
from nexus.runtime.providers.taiji import TaijiRuntimeProvider

rp = TaijiRuntimeProvider(
    api_base_url='http://hyrl-sandbox.prod.woa.com:8052',
    token='<your-token>',
    image='mirrors.tencent.com/hunyuan_yanguan/sweb.eval.x86_64.django__asgiref-471',
    runtime_image='current',
    resource_spec={'gpu': 1, 'gpu_card_type': 'H20'},
    taiji_app_group='TaiJi_HYAide_T3_Nexus_Sandbox',
    taiji_app_group_location='gy',
    request_timeout=1800.0,
)

runtime = await rp.acquire(name='main')
# 此后 invoke / run_command / release 与 Gongfeng 沙盒相同
await rp.release('main')
```

`acquire()` 返回 Gongfeng `Runtime` 对象，连接对象为沙盒内 nexus-runtime，而非太极控制台。

## GPU 规格

| 字段 | 必填 | 说明 |
|---|---|---|
| `resource_spec.gpu` | 是 | 卡数。使用 sandbox 字段 `gpu`，请勿使用 `gpuNum`。取值 1 至该卡型的整机卡数（H20 为 8），并决定 `cpu` / `memory` 的上限 |
| `resource_spec.gpu_card_type` | 是 | 卡型。使用 sandbox 字段 `gpu_card_type`，请勿使用 `gpuType` |

可用卡型由沙盒平台配置（七彩石 `taiji.gpu_machine_specs`）决定，当前唯一可用值为 **`H20`**（YAML / Python 仍使用 sandbox 字段名）。Nexus 只校验该字段非空并原样透传；填写其它卡型会被沙盒后端在 `StartRuntime` 入口拒绝，错误信息中给出当前可用卡型。

### CPU / 内存上限按卡数比例折算

太极平台不允许单任务的 CPU / 内存超出其 GPU 卡数对应的整机份额：申请半台机器的卡，就只能申请半台机器的 CPU 和内存。整机资源先扣除平台预留，余量再按卡数比例切分：

```
上限 =（整机资源 − 平台预留）× 申请卡数 ÷ 整机卡数
```

结果向下取整。沙盒后端在 `StartRuntime` 入口按同一公式校验，超额请求直接返回 `InvalidParam` 并在错误信息中给出卡型、卡数与实际上限，不会静默降配后提交给太极。

举例：H20 整机为 8 卡 / 100 核 / 1000 GB，平台预留 8 核 + 16 GB，各卡数下的上限：

| `gpu` | `cpu` 上限（核） | `memory` 上限（GB） |
|---|---|-----------------|
| 1 | 11 | 123             |
| 2 | 23 | 246             |
| 4 | 46 | 492             |
| 8 | 92 | 984             |

`gpu` 的上限即整机卡数，H20 为 8。

yaml配置中，`cpu` / `memory` 字段值的语义与 Gongfeng 相同，单位为核数、GB。

GPU 资源的 CPU 和内存的资源配比是硬限制，无法调整。若任务需更大的CPU和内存资源，只能申请对应配比的多张GPU卡。


#### 当前机型配额：
- H20 ：整机为 8 卡 / 100 核 / 1000 GB，平台预留 8 核 + 16 GB



## 与 Gongfeng 相同的能力字段

Gongfeng YAML 中的能力字段，Taiji 同样接收并转发给沙盒后端。显式 `runtime_provider=taiji` 时保持太极调度，不会调度至 suanli：

| 字段 | 行为 |
|---|---|
| `enable_dind` / `cap_add` / `need_transparent_proxy` / `network_policy` | 与 Gongfeng 同字段；后端映射至太极 `sandboxConfig`。DinD 与透明代理不可同时启用 |
| `runtime_disable_aslr` | 同字段；后端映射至太极 annotation |
| `ttl` | 同 Go duration 写法（如 `"30m"` / `"12h"` / `"48h"`，请勿使用 `"7d"`）。未传时默认 48h，上限 48h（Gongfeng / suanli 未传为 12h、上限 168h）。超出上限时 Nexus 抛出错误 |
| `token` | **必填**。申请入口与 CPU 沙盒相同。Nexus 不校验 token 名下配额 |
| `taiji_app_group` + `taiji_app_group_location` | 仅 Taiji，**必填**。取值见 [沙盒平台 GPU 页](https://sbox.woa.com/?res_tab=gpu) |
| `runtime_provider` | 本 provider 将其固定为 `taiji`；若填写 `gongfeng` / `suanli` 将报错 |

其余 Gongfeng 配置（`api_base_url`、`token`、`image`、`runtime_image`、`python_version`、`resource_spec`、`exposed_ports`、`working_dir`、`purpose`、`batch_id`、`request_timeout`、`queue_wait_max`、`watchdog_poll_interval`、`auto_connect_ws`、`socketio_request_timeout`、`rpc_protocol`、`print_port_mappings`、`startup_jitter_max` 等）均可复用。

空值（`cap_add: []`、`ttl: ""`、`enable_dind: false`）视为未开启，不写入 Start 请求。

排队（`queue_wait_max`）、watchdog、`request_timeout` 与 Gongfeng 相同，见 [被后端入队时的排队等待](configure-runtime.md#被后端入队时的排队等待) 和 [运行日志](configure-runtime.md#怎么读运行日志)。

## 运行日志

日志 logger 仍为 `nexus.runtime.providers.gongfeng`，因 GPU provider 复用 Gongfeng 客户端。关注 `Sandbox '<name>':` 主线：

- `Created runtime provider: taiji`：已选择 GPU 调度
- `creating (... resource_spec={'gpu': 1, 'gpu_card_type': 'H20'}, ...)`：Start 已提交，向太极申请 GPU
- `READY at http://<runtime_id>.19527.hyrl-sandbox.prod.woa.com:...`：沙盒启动完成，后续 invoke 走 Gongfeng runtime 链路
- `stopping` / `released`：通过 `StopRuntime` 停止

DEBUG 下可见 Start payload 含 `runtime_provider: taiji`、`taiji_app_group`、`taiji_app_group_location`。请勿仅依据 logger 名称判断调度目标。

## 下一步

- **[配置 Runtime · Gongfeng](configure-runtime.md#使用-gongfeng-runtime-provider)**：可复用的配置项、排队、日志、镜像更新（GPU 沙盒启动后共用同一条链路）
- **[Play with Gongfeng Runtime](play-with-gongfeng-provider.ipynb)**：同一组 `acquire` / executor 的交互演练；将 provider 换为 `TaijiRuntimeProvider`，并配置 `gpu_card_type` 与应用组
- **[与运行环境交互](interact-with-runtime.md)**：Runtime 对象用法
