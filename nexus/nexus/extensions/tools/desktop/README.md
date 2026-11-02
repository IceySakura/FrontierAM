# Desktop

`desktop` 提供 Nexus 在 Linux GUI runtime 中执行通用桌面任务所需的工具集合。它适合运行需要启动桌面 session、打开应用、操作鼠标键盘、截图、执行 shell 命令的 Agent 任务。

`desktop` 可以与 `identity` task 一起使用：task 负责启动 Agent，Agent 根据提示词调用桌面工具完成目标。

## 工具

- `desktop/start`：启动 GUI stack 和可选 Ubuntu GNOME session。
- `desktop/launch_app`：启动桌面应用或 GUI 命令。
- `desktop/close_app`：关闭已启动或匹配名称的桌面应用。
- `desktop/screenshot`：获取当前桌面截图。
- `desktop/screen_size`：读取当前桌面分辨率。
- `desktop/computer_13`：执行 OSWorld `computer_13` 风格的鼠标和键盘动作。
- `desktop/pyautogui`：直接执行 PyAutoGUI 控制代码。
- `desktop/bash`：在桌面 session 用户环境中执行 shell 命令。

## 运行环境

支持的 runtime provider：

- `docker`
- `gongfeng`

两类 provider 使用同一套 desktop tools。差异只在 runtime 启动、镜像、资源配置和端口映射。

## 本地 Docker

先准备 runtime 镜像和 Ubuntu 桌面会话镜像。会话镜像可以通过模块命令构建：

```bash
uv run nexus module_commands build-ubuntu-session-image \
  --image nexus-ubuntu-session:latest
```

运行 desktop GUI 任务配置：

```bash
uv run nexus run-single docs/examples/configs/desktop.yaml \
  --output /tmp/desktop.result.json
```

Docker 配置会暴露 `5900` 和 `6080`。打开 `print_port_mappings: true` 后，runtime 会在 `stderr` 打印 `/debug/gui`、noVNC 和 VNC 入口。

## 工蜂

设置工蜂 token，并通过环境变量指定 runtime 镜像和桌面镜像：

```bash
export GONGFENG_TOKEN=<token>
export NEXUS_GONGFENG_RUNTIME_IMAGE=<runtime-image>
export NEXUS_GONGFENG_DESKTOP_IMAGE=<desktop-image>
```

工蜂运行使用相同的 `desktop` 工具集合。运行配置中将 `runtime_provider_name` 设置为 `gongfeng`，并在 `runtime_provider_config` 和 `runtime_configs` 中配置对应 runtime 镜像和桌面镜像即可。

## 结果检查

一次成功运行应满足：

- 顶层 `success=true`
- `agent_output.exit_status=finished`
- trajectory 中包含实际 desktop tool 调用
- trajectory 不以 `tool_error` 结束
