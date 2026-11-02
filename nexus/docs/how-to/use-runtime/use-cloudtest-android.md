# 使用 CloudTest Android

## 概述

CloudTest Android 让 agent 操控云测平台上的已有安卓设备。Nexus 通过 `cloudtestapi` 连接设备，把点击、滑动、截图、按键和应用管理暴露成 `gui_*` executor。本文介绍 `android_cloudtest` provider 的配置字段和可用工具。

`CloudTestAndroidProvider` 按设备 ID 连接，不创建新设备。运行结束时只断开本地连接，不销毁或归还云端设备；设备的申请和占用管理需要在云测平台完成。

## 前提条件

- 可访问云测 API 的网络环境，默认地址为 `http://api.cloudtest.woa.com/`。
- CloudTest 用户 ID 和用户密钥，以及有权限操作的 Android 设备 ID。
- 使用公有云时，准备对应的项目名。
- 安装 provider 的 host 依赖：

```bash
uv run nexus modules install android_cloudtest
```

## 快速上手

在任务配置中加入以下 runtime 配置，连接已有设备。`device_id` 必须是整数；将示例占位值替换为实际设备 ID。

```yaml
runtime_provider_name: android_cloudtest
runtime_provider_config:
  host: "http://api.cloudtest.woa.com/"
  user_id: "${oc.env:CLOUDTEST_USER_ID}"
  user_key: "${oc.env:CLOUDTEST_USER_KEY}"

runtime_configs:
  android:
    device_id: 12345  # 替换为实际设备 ID
```

以上只是 runtime 部分，需要与任务、agent 和工具配置组合使用。若只验证设备连接和 GUI 操作，可直接运行下文的示例脚本。

也可以在 Python 中直接使用 provider：

```python
import asyncio
import os

from nexus.runtime.providers.android_cloudtest.provider import CloudTestAndroidProvider


async def main():
    provider = CloudTestAndroidProvider(
        user_id=os.environ['CLOUDTEST_USER_ID'],
        user_key=os.environ['CLOUDTEST_USER_KEY'],
    )
    try:
        runtime = await provider.acquire('android', device_id=int(os.environ['CLOUDTEST_DEVICE_ID']))
        result = await runtime.call_executor('gui_press_key', {'key': 'Home'})
        if not result.get('success'):
            raise RuntimeError(f'Home 操作失败: {result}')
    finally:
        await provider.release_all()


asyncio.run(main())
```

## 进阶配置

### 公有云项目

通过 `is_public_cloud` 启用公有云模式，`project` 会传给云测 SDK：

```yaml
runtime_provider_name: android_cloudtest
runtime_provider_config:
  host: "http://api.cloudtest.woa.com/"
  user_id: "${oc.env:CLOUDTEST_USER_ID}"
  user_key: "${oc.env:CLOUDTEST_USER_KEY}"
  is_public_cloud: true
  project: "${oc.env:CLOUDTEST_PROJECT}"

runtime_configs:
  android:
    device_id: 12345  # 替换为实际设备 ID
```

### 配置字段

| 字段 | 默认值 | 说明 |
|------|--------|------|
| `host` | `http://api.cloudtest.woa.com/` | 云测 API 地址 |
| `user_id` | — | CloudTest 用户 ID |
| `user_key` | — | CloudTest 用户密钥 |
| `is_public_cloud` | `false` | 是否使用公有云 |
| `project` | — | 公有云项目名，仅公有云模式传给 SDK |
| `device_id` | — | 必填整数，放在 `runtime_configs.<name>` 中或传给 `acquire()` |

`host`、`user_id`、`user_key`、`is_public_cloud` 和 `project` 也可以按设备配置在 `runtime_configs.<name>` 中，覆盖 provider 级别的值。

## 可用工具

以下 executor 通过 `call_executor()` 调用，也可绑定到 agent 的工具集。调用后应检查返回结果的 `success` 字段。

### GUI 操作

| Executor | 参数 | 说明 |
|----------|------|------|
| `gui_click` | `x`, `y` | 点击坐标 |
| `gui_double_click` | `x`, `y` | 双击坐标 |
| `gui_long_click` | `x`, `y`, `duration` | 长按，时长单位毫秒，默认 1000 |
| `gui_drag` | `x1`, `y1`, `x2`, `y2`, `duration` | 滑动，时长单位毫秒，默认 300 |
| `gui_type` | `text` | 向当前输入框输入文本 |
| `gui_press_key` | `key` | 支持 `Back`、`Home`、`Menu`、`Enter`、`Delete`、`Tab`、`Escape`、`Space`、`VolumeUp`、`VolumeDown`、`Power` |
| `gui_screenshot` | — | 截图，返回 base64 编码 PNG |
| `gui_get_screen_size` | — | 获取连接时读取的屏幕分辨率 |

坐标使用屏幕像素值。runtime 根据连接时读取的屏幕尺寸，转换为云测 SDK 使用的 0–1000 范围；操作前应确认屏幕尺寸有效。

### 应用管理

| Executor | 参数 | 说明 |
|----------|------|------|
| `gui_launch_app` | `app` | 启动 App，推荐传 Android 包名，如 `com.android.settings` |
| `gui_close_app` | `app` | 通过 `am force-stop` 关闭 App |

当前 provider 不支持 `gui_get_device_info`、`gui_reset_apps`、通用命令执行、文件操作、`invoke()` 或 `call_tool()`。`gui_close_app` 当前不检查底层 shell 返回状态；需要确认实际页面时，可结合截图验证。

## 示例脚本与运行效果

仓库提供了端到端示例脚本 [`test_cloudtest_android.py`](../../examples/scripts/test_cloudtest_android.py)，模拟一个安卓操作任务：连接设备、读取屏幕尺寸、回到桌面、启动系统设置、上滑浏览列表、点击列表项、按返回键退回，最后关闭设置并返回桌面。

在仓库根目录设置环境变量后执行，将占位值替换为实际配置：

```bash
export CLOUDTEST_USER_ID='<your-user-id>'
export CLOUDTEST_USER_KEY='<your-user-key>'
export CLOUDTEST_DEVICE_ID='12345'  # 替换为实际设备 ID

# 公有云设备按需设置：
# export CLOUDTEST_IS_PUBLIC_CLOUD=true
# export CLOUDTEST_PROJECT='<your-project>'

uv run python docs/examples/scripts/test_cloudtest_android.py
```

必填环境变量为 `CLOUDTEST_USER_ID`、`CLOUDTEST_USER_KEY` 和 `CLOUDTEST_DEVICE_ID`。可选变量为 `CLOUDTEST_HOST`（默认 `http://api.cloudtest.woa.com/`）、`CLOUDTEST_IS_PUBLIC_CLOUD`（默认 `false`）和 `CLOUDTEST_PROJECT`（公有云项目名）。

也可以通过 CLI 参数传入配置，**CLI 参数优先于对应环境变量**：

```bash
uv run python docs/examples/scripts/test_cloudtest_android.py \
  --user-id '<your-user-id>' \
  --user-key '<your-user-key>' \
  --device-id 12345 \
  --host http://api.cloudtest.woa.com/ \
  --no-is-public-cloud
```

将 `12345` 替换为实际设备 ID。公有云使用 `--is-public-cloud --project '<your-project>'`；`--no-is-public-cloud` 可覆盖环境变量中启用公有云的设置。完整参数见 `--help`。

脚本逐步检查 executor 返回值，保存 6 张截图到 `docs/examples/scripts/output/cloudtest_android/<timestamp>/`。点击位置按屏幕比例计算，实际进入的页面随设备和设置列表位置变化，不依赖固定的菜单名称。发生异常时也会尝试关闭设置、返回桌面并断开本地连接。

一次真实设备验证完成了全部 7 步，读取到的分辨率为 1080 × 2392。以下为该次运行截图：

<figure markdown="span">
  ![启动系统设置](/assets/images/use-cloudtest-android/1-settings.png){ width="240" }
  <figcaption>① gui_launch_app 启动系统设置</figcaption>
</figure>

<figure markdown="span">
  ![点击进入子页面](/assets/images/use-cloudtest-android/2-clicked.png){ width="240" }
  <figcaption>② gui_drag 后点击列表项，本次进入“蓝心智能”页面</figcaption>
</figure>

<figure markdown="span">
  ![返回桌面](/assets/images/use-cloudtest-android/3-done.png){ width="240" }
  <figcaption>③ 关闭设置并按 Home 返回桌面</figcaption>
</figure>
