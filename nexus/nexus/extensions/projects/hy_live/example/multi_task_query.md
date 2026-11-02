# multi_task_query

一通对话先后交出四项工作（贪吃蛇游戏、快速排序、扫雷、俄罗斯方块），中间几次问进度，每项工作各自要权限。 用户由 LLM user 扮演，审查只看结构。

## 1. 数据

已冻结：[`data/multi_task_query.json`](data/multi_task_query.json)

LLM user（`client_name=user_simulator`，`temperature=0`）读取剧本卡（`user_simulator.scenario`），按 `required_nodes` 的顺序推进：开场由 `initial_prompt` 固定，之后每句话都是用户模型的结构化决策，程序校验节点顺序和时机：

| 阶段 | 用户应说的话 |
|---|---|
| 开场 | 写一个贪吃蛇游戏（`initial_prompt`） |
| 贪吃蛇第一条权限请求到达 | 先问进度，再答应以后都允许 |
| 答完贪吃蛇的权限 | 再写一个快速排序 |
| 贪吃蛇做好 | 问还有什么任务在做 |
| 快速排序要权限 | 只允许这一次 |
| 快速排序做好 | 问哪些做完了，再提扫雷 |
| 扫雷要权限 | 每次都只允许这一次 |
| 答完扫雷的权限 | 再写一个俄罗斯方块 |
| 俄罗斯方块要权限 | 答应以后都允许，再问游戏的情况 |
| 四项都做好 | 道谢后结束（`ending.normal`） |

权限按工作来答（`event_plan.permission`）：贪吃蛇游戏、俄罗斯方块答应以后都允许，快速排序、扫雷每次只允许这一次；后者也写进 `constraints`，用户不会因为麻烦把它们升级成以后都允许。 后端要几次权限由后台 agent 决定，LLM user 有几条答几条；几条权限请求连着到达时，它一句答一条。 快速排序、扫雷也可能一次权限都不要；WorkBuddy 常为它们连续要十几次权限，每次都要单独答，所以用户发言上限放到 60 轮。 没有要说的话时用户选择等待，静默期间约每 30 秒有一次决策机会，不消耗发言轮次；某个节点的时机已经错过时，用户可以显式跳过它。

| 项 | 值 |
|---|---|
| `user_place` | `{"country": "中国", "province": "上海", "city": "上海"}` |
| `user_now` | `2026-10-09T14:35:00+08:00` |
| `web_search_provider` | `yuanbao`（本 case 不应触发搜索） |
| `working_dir` / `skip_permissions` / `permission_policy` | `/workspace` / `false` / `defer` |
| `hold_next_user_until_permission` | `false` |
| `max_user_turns` / `task_delivery_timeout` / `default_budget.timeout_s` | 60 / 1800 / 2400 |

LLM user 的措辞、问进度的次数和时机不固定，通知与用户话的先后也不固定；这些都不算偏差。

## 2. 运行

见 [`README.md`](README.md)「运行」，`--case multi_task_query`。

## 3. 审查清单

措辞、产物细节、时间戳不算偏差；结构、顺序、字段不对才算。

小脑（`cerebellum.jsonl`）：

1. 系统提示末尾 `<用户物理信息>`：位置上海，时间 2026年10月9日14时 星期五。 tools 是 7 个小脑工具。
2. 用户按阶段说话：开场交出贪吃蛇，之后依次交出快速排序、扫雷、俄罗斯方块；至少三次问进度；每次答复权限都在对应的权限 `[通知]` 之后。
3. 四次 `delegate`：贪吃蛇、快速排序、扫雷、俄罗斯方块，各一件，按此顺序，不合并，不续办。
4. 每次问进度都调用 `question`，不凭对话记忆直接回答。
5. `answer_permission` 的 text 写明工作名和决定：贪吃蛇、俄罗斯方块是「以后都允许」，快速排序、扫雷是「允许这一次」；不含 id。 没有待答复的权限时不调用，口头说明。
6. 不调用 `web_search` / `cancel` / `open_artifact`。
7. `question` 回执是 `[通知]` + `status=ok` + 中脑的 say；完成 `[通知]` 是「有一项工作做完了。」/「工作：」/「结果：」，合并时以「下面几项工作都做完了。」开头逐项列出。
8. 进度回答里说出每项工作的名称和状态（在做 / 卡在权限 / 已完成）；最后一条完成通知之后有一句口语回复。

中脑（`midbrain.jsonl`）：

1. 四次 `TaskCreate`，各开一个 Task。
2. `answer_permission` 的 `decision` 在贪吃蛇、俄罗斯方块上为 `allow_all`，在快速排序、扫雷上为 `allow`，`permission_id` 取自快照。 例外：用户对俄罗斯方块说「全部允许」时扫雷恰好还有一条开着的请求，`allow_all` 一并覆盖它是可以的。
3. `question` 时直接 `reply`，或先 `TaskGet` / `TaskList` 再 `reply`；账本够答时不调用 `ask_backend`。 回执不得编造账本里没有的进度。
4. `TaskCreate` 返回 `accepted` + `task_id`；`answer_permission` 返回 `answered` + 对应 `decision`。
5. 每步以单独的 `reply` 结束；`delegate` / `answer_permission` 回 `accepted`，`question` 和工作结束回 `ok`；`reply.say` 不含 task_id、权限 id。

`result.json`：`task_output.tasks` 四项都是 `done`，各有送达小脑的 `task_delivery`。

FAIL 典型：`question` 被 `delegate` 出去；快速排序、扫雷被答成 `allow_all`，或贪吃蛇、俄罗斯方块只允许一次；某项工作被续办到别的 Task 上；某项工作的完成通知没有送达小脑。
