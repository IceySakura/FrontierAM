# workbuddy_followup_and_steer

WorkBuddy 后端（`--backend workbuddy`，`wb --acp`，ACP v2）专属 case：主 Run 运行中追问（`session/fork` 深度追问）、运行中改需求（steer）、ACP 原生产物展示与打开产物。 用户由 scripted user 扮演，审查只看结构。

## 1. 数据

已冻结：[`data/workbuddy_followup_and_steer.json`](data/workbuddy_followup_and_steer.json)

scripted user 共 5 轮：

| 阶段 | 用户说的话 | 作用 |
|---|---|---|
| 开场 | 写一个俄罗斯方块。先检查工作目录，再按 HTML、CSS、JavaScript 分文件实现开始/暂停、下一块预览、暂存、幽灵落点、等级加速和键盘/触屏操作；最后逐项检查操作并修正问题。完成后通过原生产物展示能力将 index.html 作为主产物展示并打开预览，不要只在回复里写文件路径。 | 交出工作（主 Run），完成后展示主产物 |
| 权限请求到达 | 都允许 | 允许 WorkBuddy 请求的操作 |
| 立即 | 问问它准备怎样处理旋转踢墙，为什么选这个方案？让它用两句话解释，不只是汇报进度。 | 运行中追问：走 `session/fork`，不取消主 Run |
| 追问回答送达 | 现在就改掉正在做的计分规则，不要等这版做完：加一个计分板，消一行固定加 100 分。 | 运行中改需求：`TaskUpdate` 插入生成中的回合，同一个 Run 接着做 |
| 完成通知送达 | 好的，打开看看 | 打开产物（小脑 `open_artifact`，中脑 `present_files`） |

| 项 | 值 |
|---|---|
| `user_place` | `{"country": "中国", "province": "上海", "city": "上海"}` |
| `user_now` | `2026-10-14T15:00:00+08:00` |
| `working_dir` / `skip_permissions` / `permission_policy`（`work_agent_configs.workbuddy`） | `/workspace` / `false` / `defer` |
| `hold_next_user_until_permission` | `false` |
| `max_user_turns` / `task_delivery_timeout` / `default_budget.timeout_s` | 10 / 1800 / 2400 |

## 2. 运行

见 [`README.md`](README.md)「运行」，`--case workbuddy_followup_and_steer`，且必须 `--backend workbuddy`。 镜像预装 WorkBuddy 时无需提供包：

```bash
python -m nexus.extensions.projects.hy_live.example --case workbuddy_followup_and_steer \
  --backend workbuddy --runtime-image <runtime-image>
```

本 case 只在 `--backend workbuddy` 下可跑（`cases.py` 的 `backends`）。 镜像未含本次 WorkBuddy 改动时加 `--worker-venv`，见 [`../docs/impl/runtime_image.md`](../docs/impl/runtime_image.md)。

主任务分文件实现并检查多项操作，给运行中追问和改需求留出执行窗口。 追问要求后端简短解释技术方案，而不是查询已有进度；改需求明确要求立即替换当前规则。 「顺便加」这类追加需求同样走 `TaskUpdate`，但追加时若这一轮已经结束就插不进去，不能用于验收 steer。

## 3. 审查清单

措辞、产物细节、时间戳不算偏差；结构、顺序、字段不对才算。 在结构检查之外，本 case 额外检查三条（`cases.py` 的 `case_checks`）：

- `question_answered_during_run`：追问的回答在主 Run 仍在运行时送达（fork 生效，而不是等主 Run 结束）。
- `steer_replaced_running_run`：改需求的 `TaskUpdate` 返回 `steered=true`（打进了正在执行的一轮），这个 Task 最终 `done`。
- `acp_artifact_presented_and_opened`：完成的 Task 有展示 `index.html` 的 ACP `display` 事件，中脑 `present_files` 返回同名主产物的 `opened`、对应 artifact_id 和 URL。 只有扫描目录登记文件，不能通过这一项。

ACP 执行证据：

1. runtime 日志有实际 `session/fork` 的执行记录，追问不取消主 Run。
2. 后脑发送 `_codebuddy.ai/command`，action 为 `presentFiles` 或 `openResultView`，`focusFile` 或 `targetFile` 指向 `index.html`。 无需两种 action 都调用；另一种形式由单测覆盖。
3. harness 登记该产物并写入“展示产物”的 `display` 事件，最后用户打开同一页面。 若预览同时发出 `openBrowser`，核对其 URL 与“打开网页”事件；不强制要求本地页面预览走浏览器 URL。

中脑（`midbrain.jsonl`）：

1. 首轮 `TaskCreate` 交出俄罗斯方块；追问用 `question`（中脑 `TaskGet`，不够时 `ask_backend`）；改需求用 `TaskUpdate`，description 只写改动；最后一轮按快照的 artifacts 调 `present_files`。
2. `answer_permission` 的 `decision` 为 `allow_all`，`permission_id` 取自快照。
3. 改需求的 `TaskUpdate` 回 `accepted`、`steered=true`，同一个 Task 接续并最终 `completed`。
4. 追问的回答（`answer` 事件）在主 Run 结束前写进账本。

`result.json`：`task_output.tasks` 的俄罗斯方块 Task 为 `done`，产物里有游戏页面（以及加分后的计分板版本）。

FAIL 典型：追问没有在主 Run 运行中得到回答（退化为排队到回合结束）；改需求变成 interrupt 后重开一轮（`TaskUpdate` 的 `steered` 为 false 或后端日志里有 interrupt）；steer 后 Task 一直 `running` 没有完成；产物只有回合末目录扫描记录而没有 ACP 展示事件；`present_files` 找不到产物或打开了其他文件。
