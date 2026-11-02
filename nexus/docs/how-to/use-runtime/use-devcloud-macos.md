# 使用 DevCloud macOS

## 概述

`DevCloudRuntimeProvider` 把 DevCloud macOS 沙箱接入 Nexus。 Provider 通过 DevCloud OpenAPI 创建 macOS 沙箱，通过 `/health` 等待沙箱内 OSWorld HTTP server ready，然后返回一个 `DevCloudRuntime`。 `DevCloudRuntime` 把 OSWorld 的 `/execute`、`/screenshot`、`/screen_size`、`/file`、`/setup/upload` 等接口适配成 Nexus 的 command、GUI 和 file executor。

本文介绍 `devcloud` provider 的配置字段、支持的 executor，以及如何运行端到端示例脚本。

## 前提条件

- DevCloud 分配的 `app_id` 和 `app_token`。
- Nexus host 能访问新版 DevCloud Sandbox API（创建、查询和销毁使用同一入口）。
- Nexus host 能访问 macOS 沙箱的 `http://<Ip>:<server_port>`。
- macOS 沙箱内已启动 OSWorld HTTP server，默认端口为 `5000`。
- 不要把 `app_token` 写入仓库；推荐通过环境变量传入。

## 快速上手

```yaml
runtime_provider_name: devcloud
runtime_provider_config:
  app_id: "${oc.env:DEVCLOUD_APP_ID}"
  app_token: "${oc.env:DEVCLOUD_APP_TOKEN}"
  sandbox_api_base_url: "${oc.env:DEVCLOUD_SANDBOX_API_BASE_URL,http://21.6.221.82:18800}"
  lease_time: 300
  server_port: 5000
  request_timeout: 1800
  instance_ready_timeout: 300
```

也可以在 Python 中直接创建 provider：

```python
from nexus.runtime.providers.devcloud.provider import DevCloudRuntimeProvider

provider = DevCloudRuntimeProvider(
    app_id='90002',
    app_token='...',
    sandbox_api_base_url='http://21.6.221.82:18800',
)
runtime = await provider.acquire('main')
result = await runtime.run_command('uname -a && sw_vers || true')
await provider.release('main')
```

### Provider 配置字段

| 字段 | 默认值 | 说明 |
|------|--------|------|
| `app_id` | 无（创建时必填） | DevCloud 应用 ID，作为 `APPID` header 发送。 |
| `app_token` | 无（创建时必填） | DevCloud 共享密钥，只用于本地计算签名，不发送、不打印。 |
| `sandbox_api_base_url` | 必填 | 新版 Sandbox API 统一入口，用于创建、查询、销毁。 |
| `lease_time` | `300` | 沙箱租约秒数，传给 `leaseTime`。 |
| `server_port` | `5000` | OSWorld HTTP server 端口。 |
| `request_timeout` | `1800` | runtime 请求超时和每个异步任务的轮询总预算（秒）。 |
| `control_request_timeout` | `30` | 单次管理 API 请求超时（秒）。 |
| `release_timeout` | `120` | 整次释放（含查询）的最大等待时间（秒）。 |
| `proxy_url` | 空 | 管理 API、执行/截图/文件传输和 native bridge 使用的显式 HTTP 代理。 |
| `execute_mode` | `sync` | `sync` 保持原同步执行；`async` 显式开启后台执行与结果轮询。 |
| `instance_ready_timeout` | `300` | 拿到 IP 后等待 `/health` ready 的最大秒数。 |

### 新版协议与迁移

协议以 [DevCloud Sandbox API 文档](https://iwiki.woa.com/p/4024218319) 为准：

1. `POST /mac/sandboxCreate` 只提交 `leaseTime`，记录返回的 `taskId`。
2. `POST /mac/sandboxTaskResult` 查询创建结果；兼容文档中的 `Status`、`Ip`、`Id` 字段大小写，成功后检查 `/health`。
3. `POST /mac/sandboxDelete` 只提交创建结果中的实例 `id`，再通过 `sandboxTaskResult` 确认删除。

`lease_time` 大于 0 时最少 300 秒；0 或负数关闭租约，需要调用方负责清理。
鉴权保持 `APPID/TIMESTP/RANDOM/ENCKEY`，token 只用于本地签名。
旧 `control_api_base_url`、`project`、`pipeline_id`、`build_id`、`vm_seq_id`、`model` 配置仍可读取，
但会警告字段已废弃，且不发送给服务端；应从配置中删除。旧查询、删除端点不再调用。
这是 `devcloud` 的统一协议升级，使用旧服务端的环境需要先切换到新版 Sandbox API。

普通查询每 2 秒轮询；只读查询遇到连接中断/超时会有限退避重试，创建和删除提交不会盲目重放。
创建中断后若已有 `taskId`，先查出精确实例 ID 再清理。释放有总预算，接受删除但未确认成功、
提交超时等情况会明确记为未确认；不能把 `release()` 返回等同于服务端已经完成销毁。
复用 `runtime_credential.instance_ip` 的外部实例只断开引用，不负责销毁。

默认 `runtime_configs.<name>.runtime_kind` 为 `devcloud`（也接受 `osworld`），仍使用原执行客户端。
macOSWorld 使用相同的 provider，显式设置 `runtime_kind: macosworld_devcloud` 选择专属执行/bridge
适配，详见 [macOSWorld 配置](use-macosworld-devcloud.md)。代理和异步执行是两种 runtime 均可选择的公共能力，
不依赖 macOSWorld 注册名。未指定时保持原同步执行、超时和截图/错误处理策略。

### 可选代理与异步执行

`proxy_url` 按 provider/runtime/bridge 实例传递，不修改进程环境变量或全局 HTTP session。
显式代理覆盖管理接口、执行、截图、文件上传下载、bridge 上传/健康检查及 WebSocket。
未指定时不增加代理参数，保留原连接方式（包括原 bridge 上传/健康检查对环境代理的支持）。
WebSocket 的 HTTP 代理握手兼容仅在显式指定代理时开启，原默认 polling/升级方式和路径前缀保持不变。

`execute_mode: async` 与代理相互独立。它复用 DevCloud 的单份后台 worker，通过现有 `/execute`
提交唯一 job，再经 `/file` 轮询原始执行结果。退出码和 stdout/stderr 保留；响应丢失时只查询，
不重复提交有副作用命令。普通命令指定 timeout 时，结果等待预算为 `min(request_timeout, timeout + 30)` 秒。
取消或轮询超时不等于远端进程已停止，应继续按原生命周期释放自有实例。

公共 `RemoteRuntime` 已有 `/api/v1/invoke` 异步协议，但要求远端部署 Nexus Server；当前 DevCloud 的
OSWorld 服务没有该协议。这里将已有 DevCloud worker/轮询实现迁入公共 provider 复用，不另外复制一套
Nexus invoke，也不改变 OSWorld 的执行用户和命令语义。若将来服务端支持 Nexus invoke，再统一传输实现。

## 可用 executor

| Executor | 后端 OSWorld 接口 | 说明 |
|----------|-------------------|------|
| `run_command` | `POST /execute` | 运行 shell 命令，支持 `cwd`、`stdin`、`extra_env`、`timeout`。 |
| `gui_screenshot` | `GET /screenshot` | 截图，返回 base64 PNG。 |
| `gui_get_screen_size` | `POST /screen_size` | 获取屏幕尺寸。 |
| `gui_click` | `POST /execute` | 通过 pyautogui 点击坐标。 |
| `gui_double_click` | `POST /execute` | 通过 Quartz 发送带系统点击计数的双击事件。 |
| `gui_long_click` | `POST /execute` | 通过 pyautogui 长按。 |
| `gui_drag` | `POST /execute` | 通过 pyautogui 拖拽。 |
| `gui_type` | `POST /execute` | ASCII 使用 PyAutoGUI，非 ASCII 文本使用 Quartz Unicode 事件。 |
| `gui_press_key` | `POST /execute` | 通过 pyautogui 按键。 |
| `gui_right_click` | `POST /execute` | 右键点击。 |
| `gui_move_to`、`gui_mouse_down`、`gui_mouse_up` | `POST /execute` | 移动鼠标、按下和释放鼠标按钮。 |
| `gui_drag_to` | `POST /execute` | 从当前鼠标位置拖动到目标位置，时长单位为毫秒。 |
| `gui_scroll` | `POST /execute` | 在当前鼠标位置按滚轮刻度滚动，`dx` 正值向右，`dy` 正值向上。 |
| `gui_key_down`、`gui_key_up`、`gui_hotkey` | `POST /execute` | 持续按键、释放按键和组合键。 |
| `gui_pyautogui` | `POST /execute` | 执行受限 PyAutoGUI 脚本，使用请求中的 `timeout`。 |
| `gui_get_device_info` | `POST /execute` | 获取设备名称、系统版本和桌面尺寸。 |
| `gui_start` | `POST /execute` | 检查已有 macOS 桌面是否可访问。 |
| `gui_start_recording`、`gui_stop_recording` | `POST /execute` | 通过设备侧 ffmpeg 开始和停止录屏。 |
| `gui_launch_app` | `POST /setup/launch` | 通过 `open <app>` 启动 macOS App 路径。 |
| `gui_close_app` | `POST /setup/close_window` | 关闭窗口。 |
| `download_file` | `POST /file` | 从沙箱下载文件到 Nexus host。 |
| `upload_file` | `POST /setup/upload` | 从 Nexus host 上传文件到沙箱。 |

首版不支持 `_invoke()`、`call_tool()`、browser executor、jupyter executor 和 terminal session executor。

## GUI 行为与依赖

GUI 操作在 OSWorld 控制器的 Python 环境中执行，需要 PyAutoGUI 和 PyObjC。该环境可能不同于 `run_command` 登录 shell 中找到的 `python3`。录屏还需要设备侧 `ffmpeg`、`ffprobe` 和相应的 macOS 屏幕录制权限。

- 普通 GUI 操作沿用 HTTP 层的 `request_timeout`，不额外设置固定 30 秒超时；`gui_pyautogui` 另使用请求中的 `timeout`。客户端等待超时不保证远端 Python 已停止。
- `gui_start` 只检查现有桌面，不接受 Linux 桌面配置；脚本中的 `user`、`home_dir` 不会切换 macOS 登录用户。
- `gui_pyautogui` 使用受限代码校验，不支持任意导入或子进程调用，并保留上游 PyAutoGUI 的字符输入和双击限制。
- `gui_reset_apps`、`gui_computer_13` 暂不支持。客户端不记录已启动应用，请用 `gui_close_app` 显式关闭应用。
- 参数校验和控制器调用异常沿原调用链传播；远端执行返回的非零退出码会转成失败响应。

另一种 macOS 接入方式见 [使用 SmartRun macOS](use-smartrun-macos.md)。

## 示例脚本

仓库提供端到端示例脚本。`--app-id` 和 `--app-token` 必填；其他参数都有默认值，可按需覆盖。

完整运行命令：

```bash
cd nexus
NEXUS_AUTO_INSTALL=1 uv run python docs/examples/scripts/test_devcloud_macos.py \
  --app-id <your_app_id> \
  --app-token <your_app_token> \
  --sandbox-api-base-url http://21.6.221.82:18800 \
  --lease-time 900 \
  --server-port 5000 \
  --output-dir docs/examples/scripts/output
```

也可以只传凭据，其他参数使用默认值：

```bash
cd nexus
NEXUS_AUTO_INSTALL=1 uv run python docs/examples/scripts/test_devcloud_macos.py \
  --app-id <your_app_id> \
  --app-token <your_app_token>
```

或者通过环境变量提供凭据：

```bash
cd nexus
DEVCLOUD_APP_ID=<your_app_id> \
DEVCLOUD_APP_TOKEN=<your_app_token> \
NEXUS_AUTO_INSTALL=1 uv run python docs/examples/scripts/test_devcloud_macos.py
```

脚本参数：

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `--app-id` | `DEVCLOUD_APP_ID` | DevCloud 应用 ID，必填。 |
| `--app-token` | `DEVCLOUD_APP_TOKEN` | DevCloud appToken，必填；只用于本地签名，不会发送到服务端。 |
| `--sandbox-api-base-url` | `http://21.6.221.82:18800` | Sandbox API 统一入口。 |
| `--lease-time` | `300` | 沙箱租约秒数；完整命令示例里用 `900` 方便调试。 |
| `--server-port` | `5000` | OSWorld HTTP server 端口。 |
| `--output-dir` | `docs/examples/scripts/output` | 截图输出目录。 |

脚本会创建沙箱并执行以下端到端检查，最后释放沙箱：

1. 运行 `uname -a && sw_vers || true`，确认 command executor 可用。
2. 调用 `gui_get_screen_size` 获取屏幕尺寸。
3. 打开 TextEdit 临时文件 `/tmp/nexus-devcloud-gui-demo.txt`，执行可观测 GUI 动作：
   - `gui_click` 聚焦编辑区；
   - `gui_type` 输入 `Nexus DevCloud GUI smoke test`；
   - `gui_press_key('enter')` 换行；
   - 再次 `gui_type` 输入 `Visible click type enter drag demo`；
   - `gui_double_click`、`gui_long_click` 和 `gui_drag` 在文本区域产生可见选择效果。
4. 保存 TextEdit 相关截图：
   - `devcloud_macos_textedit_typed.png`：应能看到两行输入文本；
   - `devcloud_macos_textedit_drag_selection.png`：应能看到第二行文本被拖拽选中。
5. 打开并激活 System Settings，保存 `devcloud_macos_after_launch.png`；截图中应能看到“系统设置”窗口。
6. 关闭 System Settings，保存 `devcloud_macos_after_close.png` 和最终截图 `devcloud_macos_screenshot.png`。
7. 上传并下载 `/tmp/hello-devcloud.txt`，确认 file executor round-trip 成功。

如果本地环境变量 `DEVCLOUD_SERVER_PORT` 仍设置为旧端口，请显式改成 `5000` 或在命令前覆盖：

```bash
DEVCLOUD_SERVER_PORT=5000 NEXUS_AUTO_INSTALL=1 uv run python docs/examples/scripts/test_devcloud_macos.py
```
