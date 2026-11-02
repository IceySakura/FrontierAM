# 使用 AGS OSWorld

## 概述

AGS OSWorld 把腾讯云 AGS 的桌面环境沙盒（兼容 OSWorld 协议）接进 Nexus，让 agent 在一个完整的 Linux 桌面里做 GUI 操作和 Shell 命令——鼠标键盘、截图、读 accessibility tree 都有。 本文介绍它的接入凭证（template / control_plane）、两套接口（`nexus` 的 `run_command` + `gui_*`、`raw` 的原始 OSWorld HTTP 接口）的配置方式和可用工具。 接口取舍的设计背景见 [AGS Runtime Provider 设计](../../explanation/ags-runtime-provider-design.md)。

## OSWorld评测与底层runtime的区别

当前OSWorld-V1六模式使用AGS控制面、`interface_mode: nexus`与COS数据链路。
正式评测见[OSWorld原生运行指南](../osworld-runbook.md)。本页的raw接口、
identity示例用于底层能力调试，不等于完成benchmark setup/evaluator。

## 前提条件与快速上手

六模式准备 `AGS_SECRET_ID`、`AGS_SECRET_KEY`、`AGS_OSWORLD_TOOL_ID`。
选择已配置OSWorld服务的Tool；CC/Codex还要求沙箱LLM bridge端口9281可访问。
以下仅为runtime配置片段，不是完整RunInput：

```yaml
runtime_provider_name: ags
runtime_provider_config:
  secret_id: ${oc.env:AGS_SECRET_ID}
  secret_key: ${oc.env:AGS_SECRET_KEY}
  region: ${oc.env:AGS_REGION,ap-guangzhou}
  domain: ${oc.env:AGS_DOMAIN,ap-guangzhou.tencentags.com}
  interface_mode: nexus
  request_timeout: 1800
runtime_configs:
  main:
    runtime_kind: osworld
    provision_mode: control_plane
    tool_id: ${oc.env:AGS_OSWORLD_TOOL_ID}
    skip_tool_lookup: true
    timeout: 15300
```

AGS provider也支持template模式，但它使用template标识与AGS API key，
不能把template与ToolId视为同一字段，也不能假设任意模板都具备CC/Codex所需端口。
六模式默认示例不使用template。

## 镜像服务、CLI与端口

CLI与桌面服务版本以选定镜像为准，不能假设所有镜像都预装匹配版本。
原生agent按其安装/版本策略检查CLI。以下是常用服务端口，不代表任意Tool都已暴露：

| 端口 | 用途 |
|------|------|
| `9281` | 原生CC/Codex的沙箱LLM bridge，由Nexus启动 |
| `5000` | OSWorld server，Nexus 的 `run_command`、`gui_*` 和 raw OSWorld API 都走这里 |
| `9222` | Chrome CDP proxy，任务启动 Chrome 后可用 |
| `5910` | noVNC |
| `8080` | VLC HTTP / 其他镜像内 HTTP 服务 |

沙箱LLM bridge由原生CC/Codex harness启动，模型请求回传宿主client。镜像与网络配置须允许
9281端口访问。鉴权头由AGS handle提供，不能用模型API key代替AGS access token。
CLI执行采用后台进程与短请求轮询，进程退出、超时和清理由harness负责。

## 正式OSWorld-V1：复用原生agent

正式评测使用 `os_world` task，加 `claude_code_native_bridge` 或
`codex_native_bridge` 的 `execution_profile: osworld`。task负责COS数据、setup与评分，
profile负责CLI执行策略。

完成模型、COS与AGS配置后：

```bash
nexus run-single nexus/extensions/tasks/os_world/configs/osworld_claude_cc.yaml \
  -o /tmp/osworld-cc.json
nexus run-single nexus/extensions/tasks/os_world/configs/osworld_gpt_codex.yaml \
  -o /tmp/osworld-codex.json
```

两条命令属于不同模型服务配置，执行前分别准备对应client凭据。
单题选题和六模式batch命令见[原生运行指南](../osworld-runbook.md)。
模型请求使用9281上的native bridge回传host client。手工进程启动不替代原生内核的
退出码、超时和进程回收策略。

## 进阶runtime配置（非benchmark默认入口）

### 使用 raw interface（兼容原版 OSWorld agent / evaluator）

如果你的代码基于原版 OSWorld HTTP 协议（如 accessibility tree、文件操作、`run_python` 等），使用 `interface_mode: raw`，返回 `AGSRawOSWorldRuntime`，最大程度保留 OSWorld 原生 HTTP 接口。

```yaml
runtime_provider_name: ags
runtime_provider_config:
  api_key: ${oc.env:AGS_API_KEY}
  domain: "ap-guangzhou.tencentags.com"
  # raw interface：保留完整的 OSWorld 原生 HTTP 接口
  interface_mode: raw
  request_timeout: 1800.0

runtime_configs:
  main:
    runtime_kind: osworld
    template: "<your-osworld-template>"
```

拿到 `AGSRawOSWorldRuntime` 实例后，可通过 `screenshot()`、`accessibility_tree()`、`execute()`、`run_python()`、`run_bash_script()`、`setup_open_file()` 等接口直接调用完整的 OSWorld 原生 HTTP 接口。 具体用法见 [AGS Runtime Provider 设计](../../explanation/ags-runtime-provider-design.md)。

### 使用 control_plane 模式（内置 osworld 工具）

除了 template 模式，OSWorld 也支持走腾讯云控制面（control_plane）：用 `secret_id` / `secret_key` 认证，工具 ID 填到 `tool_id`，provider 按 ID 定位并复用该工具。

```yaml
runtime_provider_name: ags
runtime_provider_config:
  secret_id: ${oc.env:AGS_SECRET_ID}
  secret_key: ${oc.env:AGS_SECRET_KEY}
  region: ap-guangzhou
  interface_mode: nexus

runtime_configs:
  main:
    runtime_kind: osworld
    provision_mode: control_plane
    tool_type: osworld
    # AGS控制面已有OSWorld工具的ToolId
    tool_id: "<your-osworld-tool-id>"
```

control_plane 与 template 拿到实例后，后续的 `run_command` / `gui_*` executor 用法完全一致。

## 可用工具（nexus interface）

| Executor | 说明 |
|----------|------|
| `run_command` | 执行 Shell 命令，返回 stdout / stderr / exit_code |
| `gui_click` | 鼠标点击（x, y） |
| `gui_double_click` | 鼠标双击 |
| `gui_long_click` | 长按（duration_ms） |
| `gui_drag` | 拖拽（start_x, start_y, end_x, end_y） |
| `gui_type` | 输入文本 |
| `gui_press_key` | 按键（如 `enter`、`ctrl+c`） |
| `gui_screenshot` | 截图，返回 base64 编码图片 |
| `gui_get_screen_size` | 获取屏幕分辨率 |
| `gui_launch_app` | 启动应用 |
| `gui_close_app` | 关闭应用 |

## 示例脚本与运行效果

仓库提供了端到端示例脚本 [`docs/examples/scripts/test_ags_osworld.py`](https://git.woa.com/) ，模拟一个真实的桌面任务：用 shell 创建并写入一个文本文件、用 GUI 打开它、做鼠标点击和键盘输入、再用 shell 读回内容验证。 演示 `run_command` + `gui_*`（nexus）和 OSWorld 原生 HTTP 接口（raw）怎么配合。

### nexus interface

```bash
uv run python docs/examples/scripts/test_ags_osworld.py \
  --api-key demo-ags-key \
  --template <your-osworld-template> \
  --domain ap-guangzhou.tencentags.com \
  --interface nexus
```

脚本用 `run_command` 写文件、`gui_launch_app` 打开、`gui_click` + `gui_type` 操作、再用 shell 读回。 运行截图：

<figure markdown="span">
  ![GUI 打开文件](/assets/images/use-ags-osworld/nexus-1-file-opened.png){ width="480" }
  <figcaption>① gui_launch_app 在默认 GUI 应用里打开了 shell 刚写入的文件</figcaption>
</figure>

<figure markdown="span">
  ![键盘输入后](/assets/images/use-ags-osworld/nexus-2-after-type.png){ width="480" }
  <figcaption>② gui_click 聚焦窗口、gui_type 输入文字并回车后的桌面</figcaption>
</figure>

### raw interface

```bash
uv run python docs/examples/scripts/test_ags_osworld.py \
  --api-key demo-ags-key \
  --template <your-osworld-template> \
  --domain ap-guangzhou.tencentags.com \
  --interface raw
```

raw 模式直接调 OSWorld 原生 HTTP 接口：`run_bash_script` 造文件、`run_python` 读回、`setup_open_file` 打开、`screenshot()` 截图、`accessibility_tree()` 读观测。 运行截图：

<figure markdown="span">
  ![raw 打开文件](/assets/images/use-ags-osworld/raw-1-opened.png){ width="480" }
  <figcaption>① setup_open_file 打开文件后，screenshot() 抓到的桌面</figcaption>
</figure>

### control_plane 模式

脚本也支持 `--provision control_plane`，用腾讯云控制面复用一个已有内置 osworld 工具（只需 secret + `--tool-id`，不需 template）。 任务流程与上面完全一致：

```bash
uv run python docs/examples/scripts/test_ags_osworld.py \
  --provision control_plane \
  --secret-id <SecretId> --secret-key <SecretKey> --tool-id <ToolId> \
  --region ap-guangzhou --interface nexus
```

provider 按 `--tool-id` 定位已有工具并启动实例，运行结束只停止实例、不删除工具。
