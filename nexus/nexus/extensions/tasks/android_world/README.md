# Android World

`android_world` 提供 Nexus 对 Android World benchmark 任务的运行支持。它套用与 OSWorld 相同的实现方式：task 层负责编排，GUI 决策交给通用 `main_agent`，动作通过 `mobile/computer` 工具转发到 runtime。 与 OSWorld 不同的是，任务生命周期（初始化 / 打分 / 清理）由 Android World 后端的 REST 接口提供，封装在 `android_world` runtime provider 里。

按 Android World 流程完成：

1. 通过 runtime 初始化任务（reset + suite reinitialize + initialize + 取 goal）。
2. 运行 Agent 完成手机操作（每步截图观测）。
3. 调用 runtime 的原生 scorer 打分。
4. 清理任务环境并返回 score。

`android_world` 可以单独 import 使用。它会注册 Android World Agent 运行需要的 `mobile/*` 动作工具。

## 工具和任务

- `android_world`：初始化 Android World 任务，运行 Agent，再调用 runtime 原生 scorer 打分。
- `mobile/*`：与 HyData mobile 动作集对齐的移动端动作工具（见下方动作空间）。

## 运行环境

支持的 runtime provider：

- `android_world`：通过 REST 连接 Android World 后端设备，`gui_*` 动作被翻译成 `POST /execute_action`，任务生命周期走 `/reset`、`/suite/reinitialize`、`/task/initialize`、`/task/goal`、`/task/score`、`/task/tear_down`。

坐标使用 HyData 规范的**归一化 `[0, 1]`**（屏幕宽高的分数）；各 `mobile/*` 工具内部读取任务发布的真实屏幕尺寸，把坐标缩放成真实像素后再发给后端。

## 坐标与动作空间

与 HyData `schemas/tools_schema.json` 的 mobile 动作集对齐，暴露后端可执行的 11 个工具：
`mobile/click`、`mobile/type`、`mobile/press`、`mobile/wait`、`mobile/answer`、`mobile/terminate`、
`mobile/swipe`、`mobile/long_press`、`mobile/double_click`、`mobile/button_press`（`back` / `home` / `enter`）、
`mobile/open_app`。 坐标参数（`point` / `from` / `to`）为归一化 `[0, 1]`。 其中 `answer` / `terminate` 会置
`termination_requested=True` 结束 agent 循环，并把答案写入 `context_updates`，供 task 回填到输出的 `agent_answer`。

因 Android World 后端能力所限，HyData mobile 中的 `take_screenshot`（每步观测 hook 自动完成）、`call_user`、
`mobile.drag`（后端无原生 drag）、`mobile.close_app`（后端仅能回主页）、以及 `button_press` 的 `menu` 键未纳入。

## 配置

`docs/examples/configs/android_world.yaml` 使用 `android_world` provider。 运行前设置后端 host、api_key 和设备 id：

```bash
export ANDROID_WORLD_HOST=http://api.cloudtest.woa.com
export ANDROID_WORLD_API_KEY=<token>
export ANDROID_WORLD_DEVICE_ID=<device-id>

uv run nexus run-single docs/examples/configs/android_world.yaml \
  --output /tmp/android_world.result.json
```

## 任务输入

```yaml
task_config:
  task_type: ContactsAddContact   # Android World 任务类型
  seed: 0                          # 任务初始化随机种子
```

## 结果检查

一次成功运行应满足：

- 顶层 `success=true`
- `task_output.completed=true`
- `task_output.score` 由后端 scorer 返回（成功任务通常为 `1.0`）
- trajectory 不以 `tool_error` 结束
