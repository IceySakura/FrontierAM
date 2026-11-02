# 使用 AGR Mobile

## 概述

AGR Mobile 让 agent 操控安卓设备。 它底层是 E2B 兼容沙盒 + Appium WebDriver：沙盒里跑着一个安卓环境，Nexus 通过 Appium 把点击、滑动、截图、应用安装/启动这些操作暴露成 `gui_*` executor。 本文介绍 `agr_mobile` provider 的配置字段和可用工具。

> 批量调度（`nexus run-batch`、generator 脚本、output 目录结构）见 [使用 AGR Mobile（批量调度指南）](../use-agr-mobile/index.md)。

## 前提条件

- 腾讯云 AGS API Key（即 AGS 平台的 key，AGR Mobile 底层走 E2B 兼容沙盒，但用的是同一个 key），在 YAML 里通过 `${AGS_API_KEY}` 引用。
- AGS / E2B 兼容服务的域名（如 `ap-guangzhou.tencentags.com`）。
- 提前在 AGS 平台创建好安卓类型的沙盒模板，或持有已创建的 `sandbox_id`。

## 快速上手

以下配置使用 **template 模式** ，每次 run 创建一个新的安卓沙盒。 适合批量训练场景，run 结束后自动释放沙盒。

```yaml
runtime_provider_name: agr_mobile
runtime_provider_config:
  # 腾讯云 AGS API Key
  api_key: "${AGS_API_KEY}"
  # E2B 服务域名
  domain: "ap-guangzhou.tencentags.com"
  # 安卓沙盒模板名称
  sandbox_template: "<your-android-template>"
  # 模板创建超时（秒）
  template_timeout: 1800
  # Appium 就绪超时（秒）
  ready_timeout: 120
  request_timeout: 1800.0

# 单条运行时，runtime_configs 填空即可
runtime_configs:
  android: {}
```

运行（单条验证）：

```bash
uv run nexus run --config your_config.yaml
```

## 进阶配置

### 复用已有沙盒

如果你已经有一个正在运行的安卓沙盒，可以通过 `sandbox_id` 直接连接，run 结束后只断连不释放沙盒，方便调试。

```yaml
runtime_provider_name: agr_mobile
runtime_provider_config:
  api_key: "${AGS_API_KEY}"
  domain: "ap-guangzhou.tencentags.com"
  ready_timeout: 120
  request_timeout: 1800.0

runtime_configs:
  android:
    # 指定已有沙盒 ID，不会新建也不会释放
    sandbox_id: "<your-existing-sandbox-id>"
```

### 批量训练（使用 run-batch）

批量场景下，`runtime_configs.android` 由 generator 脚本按 `batch.mode` 自动注入，不需要在 YAML 里手写 `sandbox_id`。 详见 [使用 AGR Mobile（批量调度指南）](../use-agr-mobile/index.md)。

```bash
# 批量创建沙盒（template_create 模式）
uv run nexus run-batch \
  -g docs/how-to/use-agr-mobile/agr_mobile_batch_generator.py \
  -d /tmp/agr_mobile_batch.db \
  --max-concurrent 5

# 单条验证
uv run nexus run-batch \
  -g docs/how-to/use-agr-mobile/agr_mobile_batch_generator.py \
  -d /tmp/agr_mobile_single.db \
  --max-concurrent 1
```

## 可用工具

以下 executor 通过 `call_executor()` 调用，也可绑定到 agent 的工具集。

### GUI 操作

| Executor | 参数 | 说明 |
|----------|------|------|
| `gui_click` | `x`, `y` | 点击坐标 |
| `gui_double_click` | `x`, `y` | 双击坐标 |
| `gui_long_click` | `x`, `y`, `duration` | 长按（毫秒） |
| `gui_drag` | `x1`, `y1`, `x2`, `y2` | 滑动手势 |
| `gui_type` | `text` | 输入文本 |
| `gui_press_key` | `key` | 按键，支持 `Back`、`Home`、`Menu`、`Enter`、`Delete`、`Tab`、`Escape` |
| `gui_screenshot` | — | 截图，返回 base64 编码 PNG |
| `gui_get_screen_size` | — | 获取屏幕分辨率 |
| `gui_get_device_info` | — | 获取设备能力信息和当前状态 |

### 应用管理

| Executor | 参数 | 说明 |
|----------|------|------|
| `gui_launch_app` | `app` | 启动 App（包名） |
| `gui_close_app` | `app` | 关闭 App |
| `gui_reset_apps` | — | 强制停止所有用户 App（保留系统 App） |

## 示例脚本与运行效果

仓库提供了端到端示例脚本 [`docs/examples/scripts/test_agr_mobile.py`](https://git.woa.com/) ，模拟一个真实的安卓操作任务：连接设备、回到桌面、启动设置 App、上滑浏览列表、点击进入子页面、按返回键退回。 用 `--app` 指定要启动的包名（默认 `com.android.settings`）。

```bash
uv run python docs/examples/scripts/test_agr_mobile.py \
  --api-key demo-ags-key \
  --template <your-android-template> \
  --domain ap-guangzhou.tencentags.com \
  --app com.android.settings
```

脚本用 `gui_press_key`（Home/Back）、`gui_launch_app`、`gui_drag`（滑动）、`gui_click` 串起整个流程，每步用 `gui_get_device_info` 确认前台应用变化，并截图。 运行截图：

<figure markdown="span">
  ![回到桌面](/assets/images/use-agr-mobile/1-home.png){ width="240" }
  <figcaption>① 按 Home 回到桌面（NexusLauncher）</figcaption>
</figure>

<figure markdown="span">
  ![启动设置 App](/assets/images/use-agr-mobile/2-app-opened.png){ width="240" }
  <figcaption>② gui_launch_app 启动后，前台变为 com.android.settings</figcaption>
</figure>

<figure markdown="span">
  ![上滑浏览](/assets/images/use-agr-mobile/3-scrolled.png){ width="240" }
  <figcaption>③ gui_drag 上滑浏览设置列表</figcaption>
</figure>

<figure markdown="span">
  ![进入子页面](/assets/images/use-agr-mobile/4-detail.png){ width="240" }
  <figcaption>④ gui_click 点击列表项，activity 变为 .SubSettings</figcaption>
</figure>

<figure markdown="span">
  ![按返回键](/assets/images/use-agr-mobile/5-back.png){ width="240" }
  <figcaption>⑤ gui_press_key Back 退回上一级</figcaption>
</figure>
