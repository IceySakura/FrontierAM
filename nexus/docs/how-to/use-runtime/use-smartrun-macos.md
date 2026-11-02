# 使用 SmartRun macOS

## 概述

`SmartRunRuntimeProvider` 通过 SmartRun MCP 租用 macOS 设备，并将设备操作适配为 Nexus 的 GUI 和 command executor。设置 `platform='macos'` 后，可通过 `model='Virtual Mac'` 选择虚拟 Mac，或通过 `model='Macmini9,1'` 选择 Mac mini。MCP 响应支持 JSON 和 SSE 格式。

本文介绍 macOS 的配置方式、可用 executor、设备依赖和示例脚本。移动设备用法见 [SmartRun iOS](use-smartrun-ios.md) 和 [SmartRun Android](use-smartrun-android.md)。

## 前提条件

- 已申请 SmartRun API token，并通过运行 Nexus 的主机环境变量 `SMARTRUN_TOKEN` 提供。
- 有可分配的 macOS 设备，且桌面会话和 GUI 自动化所需的系统权限已就绪。
- 录屏需要设备侧 `ffmpeg`、`ffprobe` 在 `PATH` 中可用，并授予屏幕录制权限。
- 扩展 GUI 操作需要设备侧 `python3` 安装 PyAutoGUI 和 PyObjC。Provider 固定使用 `python3`，不自动安装依赖。

依赖必须安装在 Mac 设备上，而不是运行 Nexus 的主机上。可先在设备中检查：

```bash
python3 -c 'import pyautogui, Quartz; print(pyautogui.size())'
```

此前 Python 3.9 环境中，PyAutoGUI 与 PyObjC 10.3.2 的组合在独立 venv 中通过测试。对于相同 Python 版本，可将依赖安装到设备默认 `python3` 对应的环境，再执行上面的检查：

```bash
python3 -m pip install pyautogui \
  pyobjc-core==10.3.2 pyobjc-framework-Cocoa==10.3.2 pyobjc-framework-Quartz==10.3.2
```

其他 Python 版本需选择兼容的 PyObjC 版本。已有 venv 的测试结果不代表默认 `python3` 已具备依赖。

## 快速上手

`acquire()` 租用设备并返回已连接的 `SmartRunRuntime`，`release()` 或 `release_all()` 归还设备。

使用 Virtual Mac：

```yaml
runtime_provider_name: smartrun
runtime_provider_config:
  token: "${SMARTRUN_TOKEN}"
  platform: macos
  model: "Virtual Mac"
  duration: 1800

runtime_configs:
  main:
    purpose: "macOS GUI 示例"
```

使用 Mac mini 真机：

```yaml
runtime_provider_name: smartrun
runtime_provider_config:
  token: "${SMARTRUN_TOKEN}"
  platform: macos
  model: "Macmini9,1"
  duration: 1800

runtime_configs:
  main:
    purpose: "Mac mini 真机 GUI 示例"
```

上述配置按型号分配设备。如需固定设备，可在 `runtime_configs.main` 中增加 `device_id: "${SMARTRUN_DEVICE}"`，并将环境变量设为目标设备 ID。

也可以在 Python 中直接创建 provider：

```python
from nexus.runtime.providers.smartrun.provider import SmartRunRuntimeProvider

provider = SmartRunRuntimeProvider(platform='macos', model='Virtual Mac')
try:
    runtime = await provider.acquire('main', purpose='macOS GUI 示例')
    result = await runtime.gui_get_screen_size()
    screenshot = await runtime.gui_screenshot()
finally:
    await provider.release_all()
```

### Provider 配置字段

| 字段 | 默认值 | 说明 |
|------|--------|------|
| `token` | `SMARTRUN_TOKEN` | SmartRun API token，也可显式传入。 |
| `api_base_url` | `https://smartrun.woa.com` | SmartRun 服务地址。 |
| `platform` | `None` | 本文使用 `macos`。 |
| `brand` | `None` | 品牌过滤条件。 |
| `model` | `None` | 型号过滤条件，例如 `Virtual Mac`、`Macmini9,1`。 |
| `duration` | `1800` | 设备租约时长，单位为秒。 |
| `request_timeout` | `60.0` | Provider 请求的 HTTP 超时，单位为秒；不是 GUI shell 命令的执行超时。 |
| `startup_jitter_max` | `0.0` | 申请设备前随机等待的上限，单位为秒。 |

`acquire()` 支持通过 `device_id` 指定设备，也支持用 `platform`、`brand`、`model`、`duration`、`purpose` 覆盖默认值。固定使用同一设备仍需正常申请和归还租约。

## 可用 executor

以下接口通过 `SmartRunRuntime` 的同名方法或 `call_executor()` 调用。

| Executor | 说明 |
|----------|------|
| `gui_start_recording`、`gui_stop_recording` | 通过 MCP `shell` 运行设备侧录屏 worker，使用 FFmpeg 录制桌面并生成 MP4。 |
| `run_command` | 通过 MCP `shell` 执行命令，支持 `command`、`timeout`、`cwd`、`extra_env`；暂不支持 `stdin`。 |
| `gui_start` | 检查已有 macOS 桌面是否可访问。 |
| `gui_screenshot` | 截图，返回 base64 编码 JPG。 |
| `gui_get_screen_size` | 获取屏幕尺寸。 |
| `gui_get_device_info` | 获取设备名称、系统版本和桌面尺寸。 |
| `gui_click` | 单击，支持按钮和点击次数；省略坐标时使用当前鼠标位置。 |
| `gui_double_click`、`gui_right_click`、`gui_long_click` | 双击、右键和长按；长按时长单位为毫秒。 |
| `gui_move_to`、`gui_mouse_down`、`gui_mouse_up` | 移动鼠标、按下和释放鼠标按钮。 |
| `gui_drag`、`gui_drag_to` | 在指定坐标间拖动，或从当前位置拖动到目标位置；时长单位为毫秒。 |
| `gui_scroll` | 在当前鼠标位置滚动；使用滚轮刻度，`dx` 正值向右，`dy` 正值向上。 |
| `gui_type` | 通过原生文本输入接口输入文本，调用前需聚焦目标输入区。 |
| `gui_press_key`、`gui_key_down`、`gui_key_up`、`gui_hotkey` | 按键、持续按下、释放和组合键；支持 `cmd` 等别名。 |
| `gui_pyautogui` | 执行受限的 PyAutoGUI 脚本。 |
| `gui_launch_app`、`gui_close_app` | 通过应用标识启动或关闭应用，例如 `com.apple.calculator`。 |

显式坐标的普通左键点击、双击、右键、长按、拖动、文本输入、截图和应用启停使用原生 MCP 工具。其他扩展操作，以及省略坐标、多次点击等情况，通过 MCP `shell` 调用设备侧 `python3` 执行。

暂不支持 `gui_reset_apps` 和 `gui_computer_13`。客户端不记录已启动应用，需要关闭应用时显式调用 `gui_close_app`。

## 行为说明

- `gui_start` 不启动 Linux 桌面或配置 VNC，不接受 Linux 桌面配置参数。
- 普通 GUI shell 操作不额外指定固定执行超时，使用 SmartRun 服务端默认值；`gui_pyautogui` 将请求中的 `timeout` 传给 shell。
- `gui_pyautogui` 只接受受限脚本，不支持任意导入或子进程调用。`user`、`home_dir` 参数不会切换 macOS 登录用户。
- `Home`、`Back` 保留移动端操作语义，在 macOS 上不支持。
- 显式 PyAutoGUI 脚本保留上游库自身的字符输入和双击限制；常规文本输入、双击优先使用对应 GUI executor。
- 系统权限弹窗和未激活的目标窗口可能影响操作，应结合截图确认实际效果。缺少设备依赖时会报告错误。
- 释放后重新申请同一设备不保证清除文件或关闭应用。测试结束需显式清理临时状态。

与 DevCloud 的区别见 [使用 DevCloud macOS](use-devcloud-macos.md)：DevCloud 通过 OSWorld 控制器执行 GUI 操作，支持开始和停止录屏；非 ASCII 文本通过 Quartz Unicode 事件输入。

## 录屏

SmartRun 的 shell 位于 SSH 会话中，录屏 worker 会通过临时 LaunchAgent 将 FFmpeg 放到当前用户的桌面会话运行，停止后移除该任务。FFmpeg 需已获得 macOS 屏幕录制权限。

每个 runtime 同时只能有一个录制任务，重复开始会返回失败。通过 `gui_stop_recording()` 停止并收集成片信息；未开始或已停止时再次调用会返回失败。

```python
started = await runtime.gui_start_recording(output_path='/tmp/smartrun-demo.mp4', fps=10)
if not started.success:
    raise RuntimeError(started.error)
try:
    await runtime.gui_move_to(300, 300)
    await runtime.gui_move_to(1000, 700)
finally:
    stopped = await runtime.gui_stop_recording()
if not stopped.success:
    raise RuntimeError(stopped.error)
print(stopped.output_path, stopped.duration_seconds, stopped.size_bytes)
```

`output_path` 是设备内的绝对 MP4 路径，省略时自动生成；已有文件不会被覆盖。默认帧率为 10，始终录制鼠标指针。`max_duration_seconds` 默认 3600 秒，到时自动停止，仍需调用 `gui_stop_recording()` 收集结果。停止后返回路径、时长、大小、宽高和帧率，不会自动把视频下载到主机。请在释放设备前停止录屏并取回需要的文件。

## Native bridge

`codex_native_bridge` 和 `claude_code_native_bridge` 支持 SmartRun 的 arm64 macOS 设备，例如 `Virtual Mac`。
分别参见 [Codex 示例](../../examples/configs/codex_native_bridge_smartrun_macos.yaml) 和
[Claude Code 示例](../../examples/configs/claude_code_native_bridge_smartrun_macos.yaml)。

桥接使用 SmartRun HTTP 端口代理，不需要手动建立 SSH 隧道。首次运行会在设备的
`$HOME/.local/share/nexus-native-bridge` 下安装独立 Python 3.11，保留 GUI 自动化使用的系统 Python。
设备需能下载 Python 和对应 agent 安装包；主机需能下载 macOS arm64 bridge 依赖。

## 示例脚本

```bash
SMARTRUN_TOKEN=<your_token> uv run python docs/examples/scripts/test_smartrun_macos.py \
  --model 'Virtual Mac'
```

### 脚本参数

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `--device-id` | `SMARTRUN_DEVICE` | 指定设备；未设置时自动分配。 |
| `--model` | — | 型号过滤条件。 |
| `--api-base-url` | `https://smartrun.woa.com` | SmartRun 服务地址。 |
| `--duration` | `600` | 设备租约时长，单位为秒。 |
| `--output-dir` | `docs/examples/scripts/output` | 截图和测试结果输出目录。 |

脚本使用 `SmartRunRuntime` 操作 TextEdit，检查文本输入、保存和文件回读，每步保存截图并生成 `report.html`、`result.json`，最后归还设备并检查租约状态。该示例验证文本编辑流程，不覆盖上表所有 executor。
