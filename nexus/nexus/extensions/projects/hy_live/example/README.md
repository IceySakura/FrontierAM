# hy-live 沙箱 Example

单测盖不住 Gongfeng + 后脑 ACP 这条真链路，本目录只收这类用例。 默认使用 WorkBuddy（ACP v2），加 `--backend codebuddy` 可切回 CodeBuddy（ACP v1）。 每条 case 固定三件事：

1. **数据**：`data/<case>.json` 已冻结，不要现场拼剧本。
2. **运行**：下面「运行」一节的命令，只改 `--case` 和 `--output`。
3. **审查**：CLI 先做一层通用结构校验；再按 case md 里的清单审本次产出的 `cerebellum.jsonl`、`midbrain.jsonl` 和 `result.json`。

链路与协议见 [`../docs/impl/agent_loop.md`](../docs/impl/agent_loop.md)。 审查时记住两点：role=user 且以 `[通知]` / `[失败]` 开头的消息是系统插入的，不是用户说的；中脑的决定（权限的 `allow` / `reject` / `allow_all`、新建用 `TaskCreate` 还是更新用 `TaskUpdate`）只出现在 `midbrain.jsonl`，小脑轨迹里只有自然语言。

## 用例

| case | 数据 | 审查清单 |
|---|---|---|
| `single_snake` | [`data/single_snake.json`](data/single_snake.json) | [`single_snake.md`](single_snake.md)：权限都允许 → 完成后「速度有点快」在同一 Task 上续办 |
| `mixed_task_and_cancel` | [`data/mixed_task_and_cancel.json`](data/mixed_task_and_cancel.json) | [`mixed_task_and_cancel.md`](mixed_task_and_cancel.md)：联网搜索 + 并行两项 + 取消贪吃蛇后继续 |
| `multi_task_query` | [`data/multi_task_query.json`](data/multi_task_query.json) | [`multi_task_query.md`](multi_task_query.md)：LLM user 先后交出四项工作，多次问进度，逐项权限 |
| `ask_user_question` | [`data/ask_user_question.json`](data/ask_user_question.json) | [`ask_user_question.md`](ask_user_question.md)：CodeBuddy 用 `AskUserQuestion` 问游戏类型 → 用户回答 → 同一回合接着做完 → 打开产物 |
| `ask_permission_reject` | [`data/ask_permission_reject.json`](data/ask_permission_reject.json) | [`ask_permission_reject.md`](ask_permission_reject.md)：权限通知后用户拒绝，中脑 `decision=reject` |
| `backend_web_research` | [`data/backend_web_research.json`](data/backend_web_research.json) | [`backend_web_research.md`](backend_web_research.md)：后脑搜索与抓取来源，调研 Typesafe 的 JEV 并生成报告 |
| `backend_web_fetch` | [`data/backend_web_fetch.json`](data/backend_web_fetch.json) | [`backend_web_fetch.md`](backend_web_fetch.md)：后脑原生 WebFetch 读取指定官方文档并生成摘要 |
| `workbuddy_followup_and_steer` | [`data/workbuddy_followup_and_steer.json`](data/workbuddy_followup_and_steer.json) | [`workbuddy_followup_and_steer.md`](workbuddy_followup_and_steer.md)：WorkBuddy 运行中追问、改需求与打开产物 |

各条 case 共用同一套约定：`user_place` 为上海、`user_now` 固定、后脑 `skip_permissions=false` + `permission_policy=defer`（WorkBuddy 另写 `workbuddy.permission_mode=default`，每个工具都要权，否则默认的 `bypassPermissions` 不会发出审查清单里的权限通知）、`task_delivery_timeout` 与 `default_budget.timeout_s` 都写明。 `ask_user_question` 需要 `client_questions`，CLI 默认打开；加 `--no-client-questions` 时 CLI 拒跑这条 case。

## 联网搜索

小脑的 `web_search` 由 task_config 的 `web_search_provider` 决定：`none`（默认）时工具结果是 `搜索失败：没有接上搜索`；`yuanbao` 时走 `web_search.py` 的元宝 PromptSearch，无需凭证。 `single_snake`、`mixed_task_and_cancel`、`multi_task_query` 在数据里设为 `yuanbao`。 CLI 的 `--web-search yuanbao|none` 覆盖 case 的设置。 运行 CLI 的机器连不上搜索服务时，工具结果是 `搜索失败：搜索失败：<错误>`（抛错和兜底各加一次前缀），这是可接受的结果。

CodeBuddy 自带的 `WebFetch` 默认可用；`WebSearch` 默认是占位 key，返回 401，要在数据的 `work_agent_configs.codebuddy.extra_env` 里配有效 key，见 [`../docs/impl/codebuddy_web_search.md`](../docs/impl/codebuddy_web_search.md)。

`backend_web_research` 专门验收后脑的原生 `WebSearch`／`WebFetch`，小脑搜索关闭，后脑必须搜索成功并读取至少两篇不同来源。 `backend_web_fetch` 单独读取指定 URL，不依赖搜索。 两条回归都检查实际原生工具响应与 Task `done`，使用 `CODEBUDDY_API_KEY` 环境变量；联网失败不算通过。

## 运行

服务地址、模型和镜像在 `harness.py` 里有默认值，只有凭证必须用环境变量提供：

| 环境变量 | 内容 |
|---|---|
| `LIVE_AGENT_POLICY_API_KEY` / `LIVE_AGENT_EXTERNAL_API_KEY` | 小脑 / 其余模型 client 的 key |
| `CODEBUDDY_API_KEY` | `backend_web_research` 的后脑搜索 key，支持 CodeBuddy 和 WorkBuddy |
| `GONGFENG_RUNTIME_TOKEN` | 沙箱 token |
| `GONGFENG_RUNTIME_PURPOSE` | token 限定用途时必填，例如 `prod`；未设置时不发送（同 `--sandbox-purpose`） |

要换掉默认值时，用下面的环境变量或同名 CLI 参数：

| 环境变量 | CLI 参数 | 内容 |
|---|---|---|
| `LIVE_AGENT_POLICY_BASE_URL` | `--api-base-url` | 所有模型 client 共用的 OpenAI 兼容地址 |
| `LIVE_AGENT_POLICY_MODEL` | `--model` | 模型名 |
| `GONGFENG_API_BASE_URL` | `--sandbox-api-url` | Gongfeng 沙箱 API |
| `NEXUS_RUNTIME_IMAGE` | `--runtime-image` | 默认已包含 WorkBuddy 预装与 runtime 修复；tag 见 [`../docs/impl/runtime_image.md`](../docs/impl/runtime_image.md) |
| `LIVE_AGENT_RUNTIME_IMAGE` | `--workspace-image` | 任务 / 工作区容器的业务镜像，必须与 runtime 镜像不同 |

凭证只放环境变量，不要写进命令行。

列出 case：

```bash
uv run python -m nexus.extensions.projects.hy_live.example --list
```

本机直接运行，当前默认镜像无需 worker venv；镜像版本与要求见 [`../docs/impl/runtime_image.md`](../docs/impl/runtime_image.md)：

```bash
uv run python -m nexus.extensions.projects.hy_live.example --case single_snake --output <输出目录>/single_snake/result.json
```

经 Ray 提交时，把上面的环境变量写进权限为 600 的 runtime-env 文件（`env_vars`），并用 `py_modules: [<nexus 仓库>/nexus]` 打包 Host 侧代码：

```bash
CASE=single_snake
ray job submit \
  --submission-id "hy-live-${CASE}-$(date +%Y%m%d-%H%M)" \
  --no-wait \
  --runtime-env <runtime_env.yaml> \
  -- python -m nexus.extensions.projects.hy_live.example \
     --case "$CASE" \
     --output "<输出目录>/${CASE}/result.json"
```

其他开关：

- `--backend workbuddy|codebuddy`，默认 WorkBuddy。 未传 `--workbuddy-package` 时使用镜像中预装的 WorkBuddy，不需要重复上传安装包。
- `--relay-updates`（默认打开）让 runtime 实时推送 ACP 事件，harness 的 Event 在 Run 进行中更新；镜像不支持时 CodeBuddy session 启动失败，加 `--no-relay-updates` 后 Event 只在每轮结束时更新。 见 [`../docs/impl/live_progress_source.md`](../docs/impl/live_progress_source.md)。
- `--client-questions`（默认打开）让 CodeBuddy 可以向用户提问；镜像不支持时加 `--no-client-questions`。
- `--pass-env NAME`（可重复）把 Host 环境变量交给后脑，数据里只留变量名，值从输出中脱敏。 例如 `--pass-env TMAP_WEBSERVICE_KEY` 给预装的腾讯地图 skill 注入 Key，见 [`../docs/impl/workbuddy_skills.md`](../docs/impl/workbuddy_skills.md)。
- 默认 **不开** `worker_venv`，沙箱 Python 用当前 runtime 镜像。 后续修改沙箱侧代码时可用 `--worker-venv` 临时验证，或重建 runtime 镜像。
- 每层采样参数：`--cerebellum-*` / `--midbrain-*`（`reasoning-effort`、`temperature`、`top-p`、`max-tokens`）；`--reasoning-effort` / `--temperature` 作用于后脑和 LLM user。

CLI 只复用 `runner → task.py`，沙箱生命周期由现有链路管理，不额外连接沙箱或下载文件。

## 产物

`--output` 所在目录里：

| 文件 | 内容 |
|---|---|
| `result.json` | runner 输出加 `example_validation`；字段见 [`../docs/impl/agent_loop.md`](../docs/impl/agent_loop.md) 第 6 节，CodeBuddy 的原始轨迹在 `task_output.tasks[].result.trajectory` |
| `cerebellum.jsonl` | 小脑的 tools + 完整消息日志 |
| `midbrain.jsonl` | 常驻中脑的完整对话：每一步的账本快照、账本工具调用和 reply |
| `dialogue.jsonl` | 小脑与中脑按时间合并的用户轮视图 |
| `policy_requests.jsonl` | 每次 main（小脑）请求的开始 / 结束记录，已脱敏 |

## 审查

`example_validation` 只做对所有 case 都成立的结构检查（`validate.py`）：

- 小脑、中脑轨迹都导出了，三个 jsonl 文件存在且每行是 JSON 对象；
- 除 `web_search` 外的小脑工具结果都是 `{"status": "submitted"}`；
- 系统插入的消息只有 `[通知]` / `[失败]` 两种，模板头完整、正文非空，不含 `permission_id`、`input_id`、`artifact_id` 或 `task_1` 这类任务 ID；
- 小脑回复里没有漏出 DSML 或 `<tool_calls>` 标记；
- 每条插入消息之后，小脑的下一条回复有话说给用户；
- 每个 Task 都到了终态，做完或失败的 Task 都有送达小脑的通知。

结构过了不等于轨迹对， **仍要按 case md 的清单审一遍** 。 `policy_messages` 是小脑最后一次请求的视图，审完整过程看 `cerebellum.jsonl`；需要对照某次请求时，看同目录 `policy_requests.jsonl` 里的那一次调用。
