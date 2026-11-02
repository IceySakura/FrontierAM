# 使用 AGRT Runtime

## 概述

AGRT 是 CAPD 自建的 GPU 沙盒平台，支持 Windows 和 Linux 实例。Windows 模板内置：

- **WAA HTTP server**（`:5000`，Windows Agent Arena 协议）：截图、鼠标键盘、命令执行、文件操作、无障碍树
- **xdesk 推流**：metarender WebRTC 实时画面，支持远程操控

Linux 的 `linux-gpu-osworld` 模板在 `:5000` 提供 OSWorld HTTP server。

上层通过 **AGRT REST API** 管理实例生命周期（创建/查询/续期/销毁），通过**端口 ingress 泛域名**访问实例内服务，无需关心 VM 的实际 IP。

## 前提条件

找平台管理员申请 API Key，保存在本地环境变量 `AGRT_API_KEY` 中。所有控制面 API 用
`Authorization: Bearer <API_KEY>` 认证，实例访问使用单独签发的 Token。

从 DevCloud 访问 IDC 部署时，需要先通过 iGate 开通到平台入口的访问权限；HTTP
`acl_denied` 或 TLS `access denied` 发生在业务鉴权之前，不能据此判断 API Key 有效性。

## 在 Nexus 中使用

Provider 注册名为 `agrt`；无需安装 AGS 云控制面 SDK，也不需要在 Windows VM 中部署
Nexus worker。native 模式下，Windows 复用 WAA adapter，Linux 复用 OSWorld adapter。

```yaml
runtime_provider_name: agrt
runtime_provider_config:
  api_key: ${oc.env:AGRT_API_KEY}
  template: ${oc.env:AGRT_TEMPLATE,win-hunyuan-v1}
  interface_mode: nexus
  timeout: 30m
runtime_configs:
  main: {}
```

完整示例见 [agrt_windows_gpu.yaml](../../examples/configs/agrt_windows_gpu.yaml)。在已经设置
`AGRT_API_KEY` 的 shell 中运行：

```bash
uv run python docs/examples/scripts/test_agrt_windows_gpu.py
```

脚本通过注册表加载 provider，检查 Windows 标识、命令执行、`nvidia-smi`、UTF-8 文件读写、
GUI 截图、续期和 Token 刷新。截图、文件和 `result.json` 写入 gitignored 的
`outputs/agrt_windows_gpu/`；无论检查成功或失败，都会尝试释放实例并查询最终停止状态。
这是不接 LLM 的 provider 验证。GPU 检查确认设备和驱动可见，不代表已完成 CUDA 计算验证。
各检查独立记录错误；即使命令或文件检查失败，也会继续尝试截图和 GPU 诊断。
如果 `nvidia-smi` 失败，脚本会做有限重试并记录 `Win32_VideoController` 信息。
GPU 项以 NVIDIA 驱动可查询为验收条件；如果镜像只暴露虚拟显示适配器（例如
`TENCENT GPU controller`），`nvidia-smi` 失败不等同于物理 GPU 故障，需要平台确认
图形加速或 CUDA 的实际暴露方式。脚本仍会将这一项记为未通过。

Windows 镜像中的 `/setup/create_file` 可能使用系统默认编码。需要 UTF-8 文件时，
应显式指定编码（示例脚本通过 PowerShell 写入），不能将 `/file` 返回的任意字节直接当作 UTF-8。

`runtime_configs.<name>` 可覆盖 `api_key`、`template`、`timeout`、`token_ttl_seconds`、
`os`、`interface_mode`、`server_port`、`request_timeout` 和 `resource_spec`。
`domain`、`instance_ready_timeout`、`service_ready_timeout`、`poll_interval` 是 provider 级配置。
`resource_spec` 只保存在 runtime 元数据中；实际 GPU/CPU/内存由平台模板决定。

`interface_mode: raw` 返回原始 WAA runtime；`nexus` 返回标准 `run_command` / `gui_*`
executor 接口。可通过 `require_raw_waa_runtime(runtime)` 获取原始文件等接口；并非所有
Nexus executor 都由 WAA 实现。API 参见 [WAA Runtime](use-waa-runtime.md)。

如果实例已经运行 Nexus Runtime Server（例如通过 bootstrap 启动），使用
`launch_mode: nexus_server` 和 `runtime_port` 连接它：

```yaml
runtime_provider_config:
  api_key: ${oc.env:AGRT_API_KEY}
  launch_mode: nexus_server
  runtime_port: 18888
```

`runtime_port` 默认是 8000，应设为实例内 Nexus server 的实际监听端口。
此模式通过 `/health` 探活，返回支持 RPC、Socket.IO 和普通 `claude_code` agent 的
`RemoteRuntime`；它不会安装或启动 server。`launch_mode: native` 为默认值，连接 WAA / OSWorld
的 `server_port`（默认 5000）。`interface_mode: nexus` 仅控制内置 server 的接口适配，
不代表切换到 Nexus server；`nexus_server` 模式不支持 `interface_mode: raw`。
`launch_mode`、`runtime_port` 均可在 `runtime_configs.<name>` 中覆盖。

### Linux GPU 实例

显式指定 `os: linux`，无需根据模板名称推断系统。`os` 默认为 `windows`，可在 runtime 参数中覆盖。
架构固定为 `x86_64`，无需配置。`os` 选择客户端 adapter 和 bundle 地址，
实际实例系统仍由 `template` 决定。

```yaml
runtime_provider_name: agrt
runtime_provider_config:
  api_key: ${oc.env:AGRT_API_KEY}
  template: linux-gpu-osworld
  os: linux
  launch_mode: native
  interface_mode: nexus
runtime_configs:
  main: {}
```

Linux native 模式通过 `/screenshot` 探活；`interface_mode: nexus` 返回 `AGSOSWorldRuntime`，
`raw` 返回 `AGSRawOSWorldRuntime`，与 Windows 的 WAA 两种接口对应。
Linux 已安装 Nexus Runtime Server 时，同样可设置 `launch_mode: nexus_server`、
`runtime_port: 18888` 和 `worker_venv: true`。

实例续期和访问 Token 刷新均需调用方主动触发：`await provider.renew('main', '30m')`、
`await provider.refresh_token('main')`。结束后 `await provider.release('main')`。
对于 provider 创建的实例，如果 stop 失败，会抛出错误并保留实例 ID，供再次 `release` 或
`release_all` 重试。启动失败或取消也会尝试 stop，未清理成功时不能以同一 runtime 名再次创建实例。

## 复用实例与账号池

通过 `runtime_credential.instance_id` 连接已有的运行中实例，无需配置 `template`：

```yaml
runtime_provider_name: agrt
runtime_provider_config:
  api_key: ${oc.env:AGRT_API_KEY}
runtime_configs:
  main:
    runtime_credential:
      instance_id: ${oc.env:AGRT_INSTANCE_ID}
```

也可以直接调用 `await provider.acquire('main', runtime_credential={'instance_id': instance_id})`。
顶层 `instance_id` 不受支持。Provider 会查询实例状态、签发访问 Token 并连接 WAA；
`release`、`release_all` 和连接失败的清理只移除本地引用，不停止该实例。
复用时 `timeout` 不会重设实例到期时间，续期仍需显式调用 `renew`。

使用账号池时，将账号的 credentials 保存为 `{"instance_id": "<实例 ID>"}`，并将上面的
`runtime_credential` 配置替换为：

```yaml
runtime_configs:
  main:
    runtime_pool:
      name: agrt-windows
```

ExecutionContext 从池中租用账号，并将 credentials 作为 `runtime_credential` 传给 AGRT。
实例不存在或状态不是 `RUNNING` 时，provider 抛出 `InvalidPersistentRuntimeError`，
由 ExecutionContext 标记账号失效；鉴权失败、网络错误或服务探活超时不会据此标记账号失效。
任务结束时归还账号，实例继续保留。上述固定实例配置无需账号池，也可单独测试。

## 平台接口参考

以下模板和服务说明来自随代码提供的平台文档；当前可用模板应以平台管理员确认为准。

当前可用工具（tool）：

| tool_name | 内容 |
|-----------|------|
| `win-hunyuan-v1` | Win10 + RTX3080 + WAA + xdesk |

> 注意：tool 与镜像版本一一对应（revision 不可变）。新镜像发布后会有新 tool 名，过期 tool 建实例会永久 PENDING（放置失败无重试），下单前确认当前 tool 名。

## 快速上手

```bash
API=https://agrt.nano.woa.com
AUTH="Authorization: Bearer ${AGRT_API_KEY}"

# 1. 创建实例（timeout 必填建议值见"生命周期"节）
curl -X POST $API/v1/instances -H "$AUTH" -H "Content-Type: application/json" \
  -d '{"template":"win-hunyuan-v1","timeout":"6h"}'
# → {"instance_id":"inst-01m2...","status":"PENDING",...}

# 2. 轮询直到 RUNNING（通常 2~5 分钟）
curl $API/v1/instances/inst-01m2... -H "$AUTH"

# 3. 获取实例访问凭证（绑定该实例，TTL 内有效）
curl -X POST $API/v1/instances/inst-01m2.../token -H "$AUTH" \
  -H "Content-Type: application/json" -d '{"ttl_seconds":3600}'
# → {"token":"tok_..."}

# 4. 访问实例内 WAA（端口 5000）
curl "https://5000-inst-01m2....agrt.nano.woa.com/probe?access_token=tok_..."
# → {"message": "Service is operational", "status": "Probe successful"}
```

## 实例访问规则（端口 ingress）

URL 规则：`https://{port}-{instance_id}.agrt.nano.woa.com/{path}`

- 子域名第一段是**实例内目标端口**，第二段是实例 ID
- token 两种传法：query `?access_token=tok_...` 或 header `X-Access-Token: tok_...`（**推荐 header**：query 里的 token 会进网关日志，虽然网关已做 REDACTED 脱敏）
- 链路：泛域名 DNS → 网关 nginx（TLS 终结）→ agrt-api（校验 token + 实例状态）→ 节点数据面代理 → VM 内服务

```bash
# 任意端口同理，比如实例内的其它服务：
curl -H "X-Access-Token: tok_..." "https://<port>-inst-01m2....agrt.nano.woa.com/<path>"
```

## 可用接口（WAA :5000）

实例内 WAA server 的原始 HTTP 端点（均为 `5000-inst-<id>` 子域名下的路径）：

| 端点 | 方法 | 说明 |
|------|------|------|
| `/probe` | GET | 探活 |
| `/platform` | GET | 平台标识（`Windows`） |
| `/screenshot` | GET | 截屏，返回 PNG |
| `/screen_size` | POST | 屏幕分辨率 |
| `/execute` | POST | 执行命令（`subprocess.run`，可选 shell） |
| `/execute_windows` | POST | 执行 Python 代码块（含 computer 对象） |
| `/update_computer` | POST | 同步元素坐标/截图给 server |
| `/obs_winagent` | GET | 完整观测（窗口列表、剪贴板等） |
| `/desktop_path` `/documents_path` | POST | 取桌面/文档目录路径 |
| `/list_directory` `/file` `/file_exists` `/folder_exists` | POST | 文件操作 |
| `/setup/launch` `/setup/execute` `/setup/create_file` `/setup/create_folder` | POST | 启动程序/建文件等 setup 动作 |
| `/start_recording` `/end_recording` | POST | 屏幕录制 |
| `/terminal` | GET | 活动终端输出 |
| `/cursor_position` `/window_size` `/wallpaper` 等 | GET/POST | 其它辅助端点（共 34 个，以 server 源码 `main.py` 为准） |
| `/shutdown` | POST | 关闭 server |

GUI 输入（鼠标移动/点击/键盘）通过 `/execute` 或 `/execute_windows` 里跑 pyautogui 实现。注意：本版 WAA server **没有直接的无障碍树（a11y tree）端点**（uiautomation 能力在代码里但未暴露 HTTP 路由）；窗口标题/窗口列表/剪贴板等复合观测走 `/obs_winagent`。如需 a11y 接口要改 server 加路由。

## 实时画面（xdesk 拉流）

实例同时提供 metarender WebRTC 桌面流（可看可控）：

```
webrtc://wbrtc.liveplay.myqcloud.com/metarender?verify_stream=0&instance_id=<裸实例ID>
```

裸实例 ID = `inst-` 前缀去掉后的部分。拉流需先经 WB 架平 `alloc_instance` 锁定实例（参考 webrtc-viewer 网关的实现）。

## 生命周期与超时

- `timeout` 创建时指定，**范围 60s ~ 168h**（硬上限，超范围 API 直接拒绝）；不传默认 12h
- **固定到期**：`expires_at = 创建时刻 + timeout`，使用中**不会**自动顺延；到期平台强制回收（销毁 VM + 清空全部磁盘状态，guest 内不要存长期数据）
- **续期**：`PATCH /v1/instances/{id}` 带 `{"timeout":"6h"}`，`expires_at` 重置为"现在 + timeout"。需要长时间使用时，调用方应周期性 PATCH 保活（建议间隔 ≤ TTL/2，失败重试）
- 主动销毁：`POST /v1/instances/{id}/stop`（overlay 全部清理）

## 状态语义与排障

| 状态/现象 | 含义 | 处理 |
|-----------|------|------|
| `PENDING`（短时间） | 放置中/VM 启动中/readiness 探测中 | 等待，通常 2~5 分钟 |
| `PENDING` 持续不动 | 放置失败（tool 的 revision 与节点不匹配、无空闲 GPU）且**不会重试** | stop 掉换当前有效 tool 重建 |
| `RUNNING` | 就绪（3389 探测通过），WAA/xdesk 已自启 | 正常使用 |
| 拉流黑屏/无画面 | 控制台会话未登录或 xdesk 未就绪 | 等 1~2 分钟；持续则反馈平台 |
| 托盘"无网络"角标 | 镜像已修复（WinINet 失效代理已去除）；若再现说明节点出网故障 | 反馈平台 |

## 安全注意

- API Key 可创建/管理实例，instance token 可**完全操控**对应实例（WAA `/execute` = 远程命令执行），都不要外发
- token 建议最小 TTL（默认 3600s 可调）
- 实例到期即销毁，任何需要保留的产物及时通过 WAA 文件接口取出

## 启动环境与 worker venv

```yaml
runtime_provider_name: agrt
runtime_provider_config:
  api_key: ${oc.env:AGRT_API_KEY}
  template: win-gpu-base
  launch_mode: nexus_server
  runtime_port: 18888
  environment:
    REGION: cn-north
runtime_configs:
  main:
    worker_venv: true
```

`worker_venv` 配置在 `runtime_configs.<name>`，只接受 `true` 或 `false`，不支持
`WorkerVenvConfig` 字典或指定 Python 版本。开启时沿用 `RemoteRuntime` 的标准 worker venv 创建流程，不额外查询或覆盖 Python 版本。
`ExecutionContext` 上传 host wheel、创建 worker 并等待就绪，provider 不重复安装。
AGRT 必须使用 `launch_mode: nexus_server`；WAA / OSWorld native 模式不支持 worker venv。

`environment` 支持 provider 默认值与 runtime 参数按键合并，runtime 同名键优先，
最终通过创建接口的 `env` 传递，仅作用于新建实例，复用账号池实例时不能修改其启动环境。
Provider 不选择运行包版本，也不自动注入 manifest 地址；模板须预装并配置好 Nexus Runtime Server 的启动方式。

Windows 未开启长路径支持时，建议使用短缓存根目录（如 `-StateRoot C:\nbs`），
避免依赖文件路径超过系统限制而导致 worker venv 创建失败。
选择固定版本前需先发布对应 manifest 和压缩包；worker venv 更新的是 worker，不会替换基础 server。
