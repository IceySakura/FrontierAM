# AGS Runtime Provider 设计

`AGSRuntimeProvider` 是当前 Nexus 里最“多形态”的 provider。它既能像 `docker` / `gongfeng` 一样承载一个完整的 Nexus runtime server，也能直接把 AGS 原生 `browser`、`code-interpreter`、OSWorld-compatible 服务暴露给上层。

通用的 `ExecutionContext -> RuntimeProvider -> Runtime` 分层先看 [Runtime Provider 与 Runtime](./runtime-system.md)。这篇只讲 AGS 特有的三件事：实例怎么拿、原生服务怎么转换、以及 `interface_mode=raw` 如何尽量保留原始能力。

## 三个正交维度

`AGSRuntimeProvider` 内部有三个正交维度：

| 维度 | 取值 | 决定什么 |
| --- | --- | --- |
| `provision_mode` | `control_plane` / `template` | 实例怎么拿到 |
| `launch_mode` | `nexus_server` / `native` | 沙箱里实际跑什么 |
| `interface_mode` | `nexus` / `raw` | provider 向上层返回什么接口表面 |

这三个维度的职责不要混在一起：

- `provision_mode` 解决的是“谁来创建实例、谁来定义环境”。
- `launch_mode` 解决的是“沙箱内部跑的是 Nexus runtime server，还是原生服务”。
- `interface_mode` 解决的是“上层拿到的是 Nexus 风格接口，还是尽量保真的原始接口”。

当前实现里的几个关键规则是：

- 如果显式传了 `provision_mode`，就按显式值走；否则同时存在 `template + api_key` 时默认走 `template`，其余情况默认走 `control_plane`。
- `runtime_kind=browser` 和 `runtime_kind=osworld` 默认走 `launch_mode=native`。
- `tool_type=custom` 时，如果 `interface_mode=raw`，默认走 `launch_mode=native`；否则默认走 `launch_mode=nexus_server`。
- `interface_mode=raw` 只允许和 `launch_mode=native` 组合；`nexus_server + raw` 没有单独意义，也不支持。

## control_plane 与 template

### `control_plane`

`control_plane` 走腾讯云 AGS 控制面 API。对自定义环境来说，这条路径控制力最高，因为你可以定义：

- `Image`
- `Command`
- `Args`
- `Ports`
- `Probe`
- `Resources`
- `RoleArn`

典型 API 链路是：

1. `DescribeSandboxToolList`
2. `CreateSandboxTool`
3. `StartSandboxInstance`
4. `AcquireSandboxInstanceToken`

它适合：

- 长期维护的正式 runtime provider
- 需要自己定义镜像和启动命令的 custom runtime
- 想把 AGS 当成 Nexus 标准 runtime 承载层

### `template`

`template` 走 E2B 兼容模板模式：

1. `Sandbox.create(template=..., api_key=..., domain=...)`
2. 从返回对象里提取 `sandbox_id`
3. 提取 `_envd_access_token` 作为 API 认证凭据

它适合：

- 你手里已经有现成模板
- 目标是快速接通，而不是完全掌控环境定义
- 想直接复用 AGS 已有 browser / OSWorld / 预制 custom 模板

### 统一的实例 handle

无论实例来自 `control_plane` 还是 `template`，provider 使用同一种 `AGSInstanceHandle`：

```python
AGSInstanceHandle(
    instance_id=...,
    access_token=...,
    domain=...,
    tool_id=...,
    tool_name=...,
    tool_type=...,
)
```

后续所有数据面访问都通过这个 handle 统一生成：

- URL：`https://{port}-{instance_id}.{domain}`
- Header：`X-Access-Token: <token>`

数据面访问使用上述端口子域名和认证头；控制面凭据负责实例申请，实例access token负责数据面认证。

也就是说，`provision_mode` 只影响“实例怎么来”，不影响后面 runtime 适配层的抽象。

## 返回路径矩阵

`control_plane` 和 `template` 一旦拿到 `AGSInstanceHandle`，后面的 runtime 创建逻辑就是共用的。当前实现支持的返回路径如下：

| `launch_mode` | `interface_mode` | `runtime_kind` | 返回的 Runtime | 主要能力边界 |
| --- | --- | --- | --- | --- |
| `nexus_server` | `nexus` | `code` / `custom` | `RemoteRuntime` | 完整 Nexus runtime server 语义 |
| `native` | `nexus` | `browser` | `AGSBrowserRuntime` | `browser_*` 子集 |
| `native` | `nexus` | `code` | `AGSCodeInterpreterRuntime` | `run_command` |
| `native` | `nexus` | `osworld` | `AGSOSWorldRuntime` | `run_command` + `gui_*` 子集 |
| `native` | `raw` | `browser` | `AGSRawBrowserRuntime` | 原始 CDP / Playwright 接入 |
| `native` | `raw` | `code` | `AGSRawCodeInterpreterRuntime` | 原始 `/execute` |
| `native` | `raw` | `osworld` | `AGSRawOSWorldRuntime` | 原始 OSWorld HTTP 表面 |
| `native` | `raw` | `custom` | `AGSRawRuntime` | 任意 custom HTTP 服务 |

如果你把这个矩阵压成一句话，就是：

- `nexus_server + nexus`：把 AGS 当成 Nexus runtime server 的承载层。
- `native + nexus`：把 AGS 原生服务裁成 Nexus 可消费的专用子集。
- `native + raw`：尽量保留原始服务表面，只做 URL / token / transport 包装。

## Nexus-style 适配做了什么转换

`interface_mode=nexus` 的价值是“让现有 Nexus agent / task / tool 代码几乎不用改就能消费 AGS 服务”。代价是它一定会有不同程度的能力裁剪。当前几条路径的转换关系如下：

| Runtime | API 转换链路 | 保留的能力 | 能力损耗 |
| --- | --- | --- | --- |
| `RemoteRuntime` | AGS 实例 -> 透传 `X-Access-Token` -> Nexus runtime server RPC | `invoke` / `call_tool` / 全部 Nexus executors | 几乎没有语义损耗 |
| `AGSBrowserRuntime` | `browser_*` -> `AsyncBrowserSession` -> Playwright -> CDP -> AGS browser | `browser_*` 子集 | 丢失原始 CDP 与完整 Playwright 对象模型 |
| `AGSCodeInterpreterRuntime` | `run_command` -> 生成 Python wrapper -> native `/execute` -> tagged JSON -> `RunCommandResponse` | `run_command` | 丢失原始 `/execute`、文件/会话/事件等原语 |
| `AGSOSWorldRuntime` | `run_command` / `gui_*` -> OSWorld HTTP endpoint 或 pyautogui controller code | `run_command` + `gui_*` 子集 | 丢失原始 OSWorld HTTP contract 和额外端点 |

换句话说，如果问题是“这套转换会不会减弱原始 API 的能力”，答案是：

- `custom + nexus_server`：基本不会。这里 provider 只是做生命周期管理、健康检查和 token 透传。
- `browser + nexus`：会，中等程度。保留的是 Nexus 已定义的 browser executor 子集。
- `code-interpreter + nexus`：会，而且最明显。它主动把原始能力收窄成 `run_command`。
- `osworld + nexus`：会，中等程度。它把 OSWorld 裁成了一个桌面 runtime，而不是完整 OSWorld 协议。
- `raw`：几乎不会做语义裁剪，主要只统一了 URL、header 和一层很薄的 transport 包装。

## Raw 形态怎么暴露

`RuntimeProvider.acquire()` 的返回契约已经固定是 `Runtime`。所以 `interface_mode=raw` 不能直接返回某个 SDK client 或普通 dataclass，而是返回一组新的 `Runtime` 子类：

- `AGSRawRuntime`
- `AGSRawBrowserRuntime`
- `AGSRawCodeInterpreterRuntime`
- `AGSRawOSWorldRuntime`

### `AGSRawRuntime`

它是通用基类，负责三件事：

- 根据 `instance_id + domain + port` 生成 URL
- 自动带上 `X-Access-Token`
- 提供最薄的一层 HTTP transport helper

通用 helper 包括：

- `url(path='', port=None)`
- `request()`
- `request_json()`
- `request_bytes()`
- `get_metadata()`

此外它还暴露两个很薄的 raw executor，方便某些只能经由 `call_executor()` 访问 runtime 的调用路径：

- `raw_get_metadata`
- `raw_request`

这两个 executor 的作用只是兜底兼容，不是重新定义原始协议。

### `AGSRawBrowserRuntime`

它是给原始 browser 能力保真使用的，核心 helper 有：

- `cdp_url`
- `json_version()`
- `json_list()`
- `connect_playwright()`

因此如果目标是“让外部继续用 CDP 或 Playwright 对象模型”，应该走 raw browser，而不是 `AGSBrowserRuntime`。

### `AGSRawCodeInterpreterRuntime`

它直接保留 code-interpreter 的 `/execute` 表面，核心 helper 有：

- `execute(code, language='python')`
- `execute_raw(payload)`

因此如果上层想直接发送 Python 代码而不是被压缩成 `run_command`，应该走 raw code，而不是 `AGSCodeInterpreterRuntime`。

### `AGSRawOSWorldRuntime`

它保留的是 OSWorld-compatible server 的原始 HTTP 表面，核心 helper 有：

- `screenshot()`
- `accessibility_tree()`
- `terminal_output()`
- `get_file(file_path)`
- `screen_size()`
- `execute(command, shell=False)`
- `run_python(code)`
- `run_bash_script(script, timeout=30, working_dir=None)`
- `setup_open_file(path)`
- `setup_close_window(window_name)`

如果集成方需要原版OSWorld HTTP方法，可以选择 `AGSRawOSWorldRuntime`。
但本仓库当前OSWorld-V1六模式使用 `AGSOSWorldRuntime`（`interface_mode: nexus`）：
Nexus task adapter负责COS资源、setup与evaluator所需的适配。
“raw保留原始接口”不意味着当前benchmark应该改用raw，二者不要混淆。

## 推荐选型

- 需要完整 Nexus runtime 语义、最少改现有 Task / Agent / Tool 时，优先 `control_plane + nexus_server + nexus`。
- 需要快速把 AGS 原生 browser / code-interpreter / OSWorld 接进现有 Nexus 工具体系时，优先 `native + nexus`。
- 需要兼容原版 agent / evaluator，或者想尽量不损失原始协议能力时，优先 `native + raw`。
- 需要承载任意 custom native service 时，优先 `control_plane + native + raw`，并显式配置 `Image`、`Command`、`Ports`、`Probe`。

如果只看一条经验规则：

- 面向 Nexus 现有抽象，选 `nexus`。
- 面向原始协议保真，选 `raw`。

## 示例配置

### 1. 在 AGS 里跑完整 Nexus runtime server

```yaml
runtime_provider_name: ags
runtime_provider_config:
  secret_id: ${AGS_SECRET_ID}
  secret_key: ${AGS_SECRET_KEY}
  role_arn: ${AGS_ROLE_ARN}
  image: ${RUNTIME_IMAGE}
  interface_mode: nexus
  rpc_protocol: http

runtime_configs:
  main:
    launch_mode: nexus_server
    runtime_kind: code
```

这条路径返回 `HttpRuntime` 或 `SocketIoRuntime`，语义最接近 `docker` / `gongfeng`。

### 2. 用模板直接拿原始 OSWorld 接口

```yaml
runtime_provider_name: ags
runtime_provider_config:
  api_key: ${AGS_API_KEY}
  template: ${OSWORLD_TEMPLATE}
  interface_mode: raw

runtime_configs:
  main:
    launch_mode: native
    runtime_kind: osworld
```

这条路径返回 `AGSRawOSWorldRuntime`，适合接原版 OSWorld agent / evaluator。

### 3. 把 AGS 原生 browser 裁成 Nexus browser executor

```yaml
runtime_provider_name: ags
runtime_provider_config:
  api_key: ${AGS_API_KEY}
  template: ${BROWSER_TEMPLATE}
  interface_mode: nexus

runtime_configs:
  main:
    launch_mode: native
    runtime_kind: browser
```

这条路径返回 `AGSBrowserRuntime`，适合直接给现有 Nexus browser tool 使用。
