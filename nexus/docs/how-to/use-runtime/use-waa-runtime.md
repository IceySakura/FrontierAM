# 使用 WAA Runtime

本节介绍 WAA（Windows Agent Arena）Runtime 的使用。WAA Runtime 提供对 Windows 虚拟机的访问，支持截屏、无障碍树获取、代码执行等操作，用于 GUI 自动化 agent 和评测。

## 概述

WAA Runtime 有两种形态：

| 类型 | 类名 | 说明 |
|------|------|------|
| Raw | `AGSRawWindowsAgentArenaRuntime` | 直接暴露 WAA server 的原始 HTTP API |
| Nexus | `AGSWindowsAgentArenaRuntime` | 实现标准 Executor 接口（gui_click、gui_screenshot 等） |

两者可互转：`runtime.to_raw()` / `runtime.to_nexus()`。

## Provider 配置

### AGS Provider（推荐）

通过 AGS 云沙箱管理 Windows VM 实例的生命周期。关于 AGS Provider 的完整设计和配置说明，参见 [AGS Runtime Provider 设计](../../explanation/ags-runtime-provider-design.md)。

```yaml
runtime_provider_name: ags
runtime_provider_config:
  secret_id: "<your-secret-id>"
  secret_key: "<your-secret-key>"
  region: <region> # eg: ap-nanjing
  interface_mode: "raw"
  request_timeout: 180.0
  instance_ready_timeout: 300.0

runtime_configs:
  main:
    tool_id: <tool-id>  # eg: sdt-gprl2m4v
    runtime_kind: waa
    launch_mode: native
    interface_mode: raw
    provision_mode: control_plane
```


## 端口说明

| 端口 | 默认值 | 用途 |
|------|--------|------|
| `server_port` | 5000 | WAA HTTP API（截屏、执行、a11y 等） |
| `chromium_port` | 9222 | Chrome DevTools Protocol |
| `vnc_port` | 8006 | VNC Web 客户端（浏览器查看桌面） |

对应 property：`runtime.server_url`、`runtime.chromium_url`、`runtime.vnc_url`

## Raw Runtime API

Raw runtime（`AGSRawWindowsAgentArenaRuntime`）暴露以下方法：

### 观察

```python
# 截屏（返回 PNG bytes）
screenshot_bytes = await runtime.screenshot()

# 无障碍树（返回 {"AT": "<xml_string>"}）
a11y_data = await runtime.accessibility_tree(backend='uia')

# 完整观察（窗口标题、窗口列表、剪贴板等）
obs = await runtime.obs_winagent()

# 屏幕尺寸
size = await runtime.screen_size()  # → [width, height]
```

### 执行

```python
# 执行命令（shell=False，通过 subprocess.run 直接执行）
result = await runtime.execute(['python', '-c', 'print("hello")'], shell=False)

# 使用 cmd.exe 执行
result = await runtime.execute(['cmd.exe', '/c', 'dir C:\\'], shell=False)

# 使用 PowerShell 执行
result = await runtime.execute(['powershell.exe', '-Command', 'Get-Process'], shell=False)

# 执行 Python 代码块（通过 WAA computer API 环境，代码中可使用 computer 对象）
result = await runtime.execute_windows('computer.mouse.single_click()')

# 更新 computer 对象状态（同步元素坐标给 server，使 computer.mouse.move_id() 等能定位元素）
await runtime.update_computer(
    rects=[[x1, y1, x2, y2], ...],
    window_rect=[0, 0, width, height],
    screenshot=base64_png_string,
    scale=(1.0, 1.0),
    clipboard_content="...",
    swap_ctrl_alt=False,
)
```

`execute` 底层调用 WAA server 的 `/execute` 端点，使用 `subprocess.run(command, shell=shell)`。`shell=False` 时 command 为列表（第一个元素是可执行文件），可以自由选择 `cmd.exe`、`powershell.exe` 或任何可执行程序。

### 文件操作

```python
# 下载文件
content = await runtime.get_file(r'C:\Users\Desktop\test.txt')

# 列出目录
tree = await runtime.list_directory(r'C:\Users\Desktop')

# 获取桌面路径
info = await runtime.desktop_path()  # → {"desktop_path": "C:\\Users\\...\\Desktop"}
```

### 窗口管理

```python
# 激活窗口
await runtime.setup_activate_window('Notepad', strict=False, by_class=False)

# 关闭窗口
await runtime.setup_close_window('Notepad', strict=False, by_class=False)

# 关闭所有应用
await runtime.setup_close_all()
```

### Setup

```python
# 启动程序
await runtime.setup_launch('notepad.exe')

# 打开文件
await runtime.setup_open(r'C:\Users\Desktop\test.txt')

# 下载文件到 VM
await runtime.setup_download('https://example.com/file.zip', r'C:\Users\Desktop\file.zip')
```

## Nexus Runtime API

Nexus runtime（`AGSWindowsAgentArenaRuntime`）实现标准 Executor 接口：

```python
# GUI 操作（通过 call_executor）
await runtime.call_executor('gui_click', {'x': 100, 'y': 200})
await runtime.call_executor('gui_double_click', {'x': 100, 'y': 200})
await runtime.call_executor('gui_type', {'text': 'hello'})
await runtime.call_executor('gui_press_key', {'key': 'enter'})
await runtime.call_executor('gui_screenshot', {})

# 命令执行（默认通过 cmd.exe，shell=True）
await runtime.call_executor('run_command', {'command': 'dir C:\\'})

# 使用 PowerShell 执行
await runtime.call_executor('run_command', {'command': 'powershell -Command "Get-Process"'})

# 指定工作目录和环境变量
await runtime.call_executor('run_command', {
    'command': 'python main.py',
    'cwd': r'C:\Users\Desktop\project',
    'extra_env': {'DEBUG': '1'},
})
```

### run_command 说明

`run_command` 底层通过 WAA server 的 `/execute` 端点执行，使用 `shell=True`（即 `subprocess.run(command, shell=True)`）。在 Windows 上：

- **默认 shell 是 cmd.exe**：直接写 cmd 命令即可，如 `dir`、`type`、`set`
- **使用 PowerShell**：在 command 前加 `powershell -Command "..."` 或 `pwsh -c "..."`
- **支持 cwd**：通过 `cd /d {cwd} &&` 前缀实现
- **支持 extra_env**：通过 `set KEY=VALUE &&` 前缀实现

## 互转

```python
from nexus.runtime.providers.ags.waa import require_raw_waa_runtime

# 自动转换：接受任一类型，返回 Raw runtime
raw_runtime = require_raw_waa_runtime(runtime)

# 手动转换
raw = nexus_runtime.to_raw()
nexus = raw_runtime.to_nexus()
```

## 在代码中获取 Runtime

```python
from nexus.core import require_context
from nexus.runtime.providers.ags.waa import require_raw_waa_runtime

ctx = require_context()
runtime = require_raw_waa_runtime(await ctx.get_runtime())

# 现在可以使用 runtime 的所有方法
screenshot = await runtime.screenshot()
```
