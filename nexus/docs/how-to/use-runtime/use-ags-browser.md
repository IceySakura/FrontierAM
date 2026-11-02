# 使用 AGS Browser

## 概述

AGS Browser 把腾讯云 AGS 的原生浏览器沙盒（Chromium + CDP）接进 Nexus，让 agent 用一组 `browser_*` executor 访问网页、点击、填表、截图。 本文介绍它的两种接入凭证（template / control_plane）、两套接口（`nexus` / `raw`）的配置方式和可用工具。 接口取舍的设计背景见 [AGS Runtime Provider 设计](../../explanation/ags-runtime-provider-design.md)。

## 前提条件

先在 AGS 平台创建好 browser 类型的工具，拿到对应的工具 ID。 两种模式都用这个 ID，区别只在认证方式：

- **template 模式** （推荐快速上手）：用腾讯云 AGS API Key 认证，工具 ID 填到 `template`。
- **control_plane 模式** ：用腾讯云 `secret_id` / `secret_key` 认证，工具 ID 填到 `tool_id`。

## 快速上手

以下配置使用 **template 模式 + nexus interface** ，返回 `AGSBrowserRuntime`，agent 直接使用 `browser_*` executor。

```yaml
runtime_provider_name: ags
runtime_provider_config:
  # 腾讯云 AGS API Key
  api_key: "${AGS_API_KEY}"
  # AGS 服务域名
  domain: "ap-guangzhou.tencentags.com"
  # 使用 nexus interface：返回标准 browser_* executor 接口
  interface_mode: nexus
  # 请求超时时间（秒）
  request_timeout: 1800.0

runtime_configs:
  main:
    # 声明使用 browser 类型的原生服务
    runtime_kind: browser
    # AGS 上的 browser 模板名称
    template: "<your-browser-template>"
```

## 进阶配置

### 使用 raw interface（保留完整 CDP 能力）

如果你的 task 代码需要直接操作 Playwright 对象模型或访问原始 CDP，使用 `interface_mode: raw`，返回 `AGSRawBrowserRuntime`。

```yaml
runtime_provider_name: ags
runtime_provider_config:
  api_key: "${AGS_API_KEY}"
  domain: "ap-guangzhou.tencentags.com"
  # raw interface：保留原始 CDP 能力
  interface_mode: raw
  request_timeout: 1800.0

runtime_configs:
  main:
    runtime_kind: browser
    template: "<your-browser-template>"
```

拿到 `AGSRawBrowserRuntime` 实例后，可通过 `connect_playwright()`、`cdp_url` 等接口直接操作浏览器。 具体用法见 [AGS Runtime Provider 设计](../../explanation/ags-runtime-provider-design.md)。

### 使用 control_plane 模式（内置 browser 工具）

除了 template 模式，还可以走腾讯云控制面（control_plane）：用 `secret_id` / `secret_key` 认证，工具 ID 填到 `tool_id`，provider 按 ID 定位并复用该工具。

```yaml
runtime_provider_name: ags
runtime_provider_config:
  secret_id: "${AGS_SECRET_ID}"
  secret_key: "${AGS_SECRET_KEY}"
  region: ap-guangzhou
  interface_mode: nexus

runtime_configs:
  main:
    runtime_kind: browser
    tool_type: browser
    # AGS 平台上 browser 工具的 ID（与 template 模式同一个 ID）
    tool_id: "<your-browser-tool-id>"
```

control_plane 与 template 拿到实例后，后续的 `browser_*` executor 用法完全一致。

## 可用工具（nexus interface）

`interface_mode: nexus` 提供与标准 Runtime 一致的 `browser_*` executor（`create_browser_session`、`browser_goto`、`browser_click`、`browser_get_observation`、`browser_screenshot` 等），通过 `call_executor()` 调用，也可绑定到 agent 的工具集。 完整清单和用法见 [Browser Executor](browser-executor.ipynb)。

大多数 executor 需要在请求中携带 `session_id`（由 `create_browser_session` 返回）。

## 示例脚本与运行效果

仓库提供了一个端到端示例脚本 [`docs/examples/scripts/test_ags_browser.py`](https://git.woa.com/) ，模拟一个真实的网页浏览任务：打开百度首页、在搜索框输入关键词回车、读取结果页可观测信息、滚动翻页、再开新标签页。 nexus 和 raw 两种 interface 都覆盖。

### nexus interface

```bash
uv run python docs/examples/scripts/test_ags_browser.py \
  --api-key demo-ags-key \
  --template <your-browser-template> \
  --domain ap-guangzhou.tencentags.com \
  --interface nexus \
  --query 腾讯云
```

脚本依次打开首页、用 `browser_fill` + `browser_press_key` 在搜索框输入并回车、用 `browser_get_observation` 读取 accessibility tree、滚动、再用 `browser_new_tab` 开新标签。 运行截图：

<figure markdown="span">
  ![打开百度首页](/assets/images/use-ags-browser/nexus-1-home.png){ width="320" }
  <figcaption>① 打开百度首页，首页搜索框稳定可见</figcaption>
</figure>

<figure markdown="span">
  ![搜索结果页](/assets/images/use-ags-browser/nexus-2-results.png){ width="320" }
  <figcaption>② 输入「腾讯云」回车后的搜索结果页</figcaption>
</figure>

<figure markdown="span">
  ![滚动后](/assets/images/use-ags-browser/nexus-3-scrolled.png){ width="320" }
  <figcaption>③ 向下滚动 1200px 后的结果列表</figcaption>
</figure>

### raw interface

```bash
uv run python docs/examples/scripts/test_ags_browser.py \
  --api-key demo-ags-key \
  --template <your-browser-template> \
  --domain ap-guangzhou.tencentags.com \
  --interface raw \
  --query 腾讯云
```

raw 模式通过 `connect_playwright()` 拿到 Playwright Browser 对象，直接用选择器操作页面、读取第一条结果标题、统计链接数。 运行截图：

<figure markdown="span">
  ![raw 首页](/assets/images/use-ags-browser/raw-1-home.png){ width="320" }
  <figcaption>① Playwright 打开的百度首页</figcaption>
</figure>

<figure markdown="span">
  ![raw 结果页](/assets/images/use-ags-browser/raw-2-results.png){ width="320" }
  <figcaption>② 搜索后的结果页，脚本读到第一条结果标题</figcaption>
</figure>

### control_plane 模式

脚本也支持 `--provision control_plane`，用腾讯云控制面复用一个已有内置 browser 工具（只需 secret + `--tool-id`，不需 template）。 任务流程与上面完全一致：

```bash
uv run python docs/examples/scripts/test_ags_browser.py \
  --provision control_plane \
  --secret-id <SecretId> --secret-key <SecretKey> --tool-id <ToolId> \
  --region ap-guangzhou --interface nexus
```

provider 按 `--tool-id` 定位已有工具并启动实例，运行结束只停止实例、不删除工具。
