# Hy-live 配置参考

本文列出 hy-live Host 侧当前链路（小脑 → `hylive` core（中脑、账本、回合调度）→ `AcpSessionExecutor` → CodeBuddy / WorkBuddy Project Session）读取的全部配置项。 默认值以代码为准，改代码时同步更新本表。

## 1. 在哪里配置

| 位置 | 内容 |
|---|---|
| `task_config`（→ `LiveAgentTaskInput` / `LiveAgentAgentInput`） | episode、小脑和中脑相关字段，见 §2 |
| `task_config.default_budget`（→ `WorkAgentBudget`） | 后台执行预算，见 §3 |
| `task_config.work_agent_configs.codebuddy` | CodeBuddy 启动配置，见 §4 |
| `task_config.work_agent_configs.workbuddy` | WorkBuddy ACP v2 启动配置，见 §4.1 |
| `task_config.user_simulator` | 模拟用户，见 §5 |
| 代码常数 | 不可配置，见 §6 |

`example/prepare_scripted_user_data.py::build_scripted_user_row` 给出的基础行（example 在它上面叠加各 case）不写 `default_budget`、`max_user_turns`、`task_delivery_timeout`，这些字段都走代码默认值；各 case 只写自己需要的差异。 给基础行加新的覆盖前，先确认确实需要偏离代码默认值。

## 2. Episode、小脑与中脑（`LiveAgentAgentInput`）

| 配置项 | 默认值 | 作用 |
|---|---|---|
| `max_user_turns` | `100`（≥1） | 一个 episode 最多接收的真实用户轮次；静默等待和系统插入的消息不计入 |
| `task_delivery_timeout` | `1800` 秒 | 从 episode 开始算的总截止；到期还有活没完记 `episode_termination=TASK_TIMEOUT`。 沙箱在 episode 开始前就已拉起（`LiveExecution.warm_up`），启动时间不计入 |
| `hold_next_user_until_permission` | `false` | 小脑 delegate 后先等权限通知，再放行下一条真实用户消息，用于写死剧本的用户 |
| `summarize_outcomes` | `false` | 一项工作做完或失败时，默认不调用中脑，小脑听到的是后端自己的结果或失败原因。`true` 时先让中脑按「工作结束」写总结，再把总结交给小脑。造数时写在 `task_config` 上，和 `user_simulator` 同一层 |
| `policy_client_name` | `main` | 可训练小脑使用的 OpenAI client，也是轨迹采集（`MainTrajectoryProcessor`）的过滤键 |
| `external_client_name` | `external` | 后脑默认使用的 LLM client |
| `midbrain_client_name` | `null`，即与 `external_client_name` 共用 | 中脑使用的 LLM client |
| `user_place` | `null`，即上海 | 写进小脑 system prompt 的用户位置（`prompts.cerebellum_instructions`）。 字典如 `{"country": "中国", "city": "北京"}`，空值字段省略；也可以直接写字符串如 `"北京朝阳区"`，原样写入 |
| `user_now` | `null`，即当前上海时间 | 写进小脑 system prompt 的当前时间，按小时取整；造数据时固定它以复现。 task 配置里要写 ISO 时间并带时区，如 `2026-10-19T10:00:00+08:00` |
| `web_search_provider` | `none` | 小脑 `web_search` 的后端：`none` 时返回「没有接上搜索」；`yuanbao` 走 `web_search.py` 的元宝 PromptSearch。 example CLI 的 `--web-search` 可覆盖 |
| `web_search_backend` | `null`，不序列化 | 程序注入的搜索后端；非空时优先于 `web_search_provider` |
| `default_budget` | 见 §3 | 每轮后脑执行的预算 |
| `backend_adapter` | `workbuddy` | 默认 WorkBuddy；显式 `codebuddy` 切回 CodeBuddy |
| `work_agent_configs` | `{}` | 读取 `backend_adapter` 对应的一项，见 §4 |
| `cache_task_id` | `{midbrain: true, backend: true, user_simulator: true, cerebellum: false}` | 哪些对话带自己的网关 `cache_task_id`，见 §2.1 |

### 2.1 `cache_task_id`

网关（aihub）把 `cache_task_id` 相同的请求路由到同一节点，同一条对话不断增长的前缀才能命中 prefix cache。 打开的每条对话在 episode 内各自生成一个随机 id（`cache_task_id.py::conversation_client`），复制一份同类 client 带上它；复制出的 client 保留原 `_client_key`，轨迹采集和 trace 归属不变。

| 键 | 默认 | 粒度 |
|---|---|---|
| `midbrain` | 开 | 中脑一个 id；与后脑共用 `external` client 时也是独立的 id |
| `backend` | 开 | 每个 Project Session 一个 id，复制出的 client 以 `<llm_client_name>@<channel>` 注册进 `ctx.openai_clients`；WorkBuddy 追问 fork 沿用来源 Session 的 id |
| `user_simulator` | 开 | LLM 模拟用户一个 id；scripted 用户不发模型请求 |
| `cerebellum` | 关 | 小脑一个 id。 HyTuner 下小脑是可训练 policy，不走网关，所以默认关 |

只有读 `client_args.cache_task_id` 的 client（`openai`、`azure_openai`、`infer_proxy`、`litellm`、`openai_responses`）能带 id；其他 client 在默认打开的三项上照常使用并打 warning，打开 `cerebellum` 时直接报错。

## 3. 后台执行预算（`WorkAgentBudget`，`task_config.default_budget`）

| 字段 | 代码默认值 | 作用 |
|---|---|---|
| `timeout_s` | `3600` 秒 | 每个 CodeBuddy 轮次（一个 Run 或一次追问）的墙钟上限，等用户的时间也计入、episode 冻结的时间不计，超出报 `WORK_AGENT_TIMEOUT`。 一轮的正常限时是不计等用户时间的 2000 秒工作预算，它只是兜底，见 [`timeout_and_cancel.md`](timeout_and_cancel.md) 第 0 节 |

## 4. CodeBuddy 配置（`work_agent_configs.codebuddy`）

| 字段 | 默认值 | 作用 |
|---|---|---|
| `working_dir` | `/workspace`（backend 默认） | 各 Task 工作区根目录；brain 为每个 Task 使用 `<working_dir>/<task_key>`（默认 `/workspace/<task_key>`），产物在该 Task 目录下登记。 `task_key` 是 Task 内部的 12 位 uuid，跨会话唯一，不给模型看；中脑用的 `task_1` 只在一通对话内唯一，沙箱复用时会撞名 |
| `llm_client_name` | `external_client_name` | 沙箱内 CodeBuddy 经 LLM channel 使用的 Host client |
| `version` | `2.97.2`（`DEFAULT_CODEBUDDY_VERSION`） | 安装的 CodeBuddy 版本 |
| `npm_registry` | `null`，使用安装器默认镜像 | npm 安装源 |
| `model_id` | `nexus-proxy-model` | 写入 `models.json` 的模型名，代理会转发所有请求 |
| `skip_permissions` | `true` | CodeBuddy：为真时以 `--dangerously-skip-permissions` 启动，不产生权限请求。WorkBuddy 的权限由 `workbuddy.permission_mode` 决定，只有它显式写成 `null` 时才按这个参数推导，见 §4.1 |
| `permission_policy` | `defer`（`skip_permissions: false` 时 backend 也默认 `defer`） | `defer` 把 ACP 权限请求写进 harness，再以 `[通知]` 转给小脑；`allow` / `reject` 直接应答 |
| `relay_updates` | `False`；`docs/examples/configs/hy_live.yaml` 与 example CLI 打开 | 让 runtime 在一轮运行期间实时推送 ACP 事件；镜像不支持该 key 时 session 启动失败，见 [`live_progress_source.md`](live_progress_source.md) |
| `client_questions` | hy-live backend 在 `permission_policy: defer` 时默认 `true`；example CLI 总是显式写出 | 让 CodeBuddy 向用户提问（`AskUserQuestion`、form elicitation），问题落成 Run 上的补充问题；镜像不支持时须显式写 `false`，见 [`question_channel.md`](question_channel.md) |
| `extra_env` / `extra_cli_args` | 空 | 追加环境变量和 CLI 参数 |
| `config_file_patches` / `pre_start_scripts` | 空 | 启动前修改 JSON 配置、执行脚本 |
| `close_on_finish` | backend 固定为 `false` | 轮次结束后保留 ACP 进程，供同一 Task 的下一轮 Run 复用 |

### 4.1 WorkBuddy 配置（`work_agent_configs.workbuddy`）

公共启动参数沿用 §4，backend 自动设置 `acp_cli='workbuddy'`。 CodeBuddy 的 `version`、`npm_registry` 不控制 WorkBuddy，版本和安装选项放在嵌套的 `workbuddy` 对象里。

只读取所选后端的配置，不自动把 `work_agent_configs.codebuddy` 当作 WorkBuddy 配置。 旧数据切到 WorkBuddy 时也要将对应配置改为 `work_agent_configs.workbuddy`；example CLI 会在构造运行配置时完成转换。

| 字段 | 默认值 | 作用 |
|---|---|---|
| `workbuddy.version` | `5.11.0` | 预装或上传安装包的版本 |
| `workbuddy.cli_path` | `null` | 指定沙箱内 CLI 入口；未指定时优先查找 `/nix/runtime/workbuddy`，见 [runtime_image.md](runtime_image.md) |
| `workbuddy.package_path` | `null` | Host 安装包路径，非空时上传；预装镜像无需填写 |
| `workbuddy.package_sha256` | `null` | 上传包的预期 SHA-256 |
| `workbuddy.preset` / `mode` / `effort` | `null` | WorkBuddy CLI 启动选项 |
| `workbuddy.features` | `{"ecosystem": true, "tencentDocsEngine": true}` | 装载插件、技能和连接器，并启动本地腾讯文档引擎，内置的 `workbuddy-pptx` 等 office skill 靠它编辑文档，与 WorkBuddy Desktop 一致；文档引擎让每个会话冷启动多约 12 秒。`null` 或 `{}` 关闭。写了别的开关时要带上这两个，整份对象会替换默认值。镜像要求见 [runtime_image.md](runtime_image.md) |
| `workbuddy.fork_mode` | `quick` | 运行中追问分支的 mode：`quick` 纯对话不带工具，`ask` 只带只读工具；`null` 沿用主会话 mode |
| `workbuddy.permission_mode` | `bypassPermissions` | 会话创建和 fork 后经 ACP `session/set_config_option` 设置，不受 `skip_permissions` 影响。可选 `default`（每个工具都问）、`acceptEdits`（自动放行文件编辑）、`bypassPermissions`（不问，但 `rm -rf`、`sudo` 等危险命令仍要审批，审批按 `permission_policy` 处理）、`fullAccess`（全部放行）。数据里写在 `task_config.work_agent_configs.workbuddy.workbuddy.permission_mode`。显式写 `null` 时按 `skip_permissions` 推导：为真用 `fullAccess`，否则用 `default`。plan 模式没有这一项，不设置 |

运行中追问附带当前目标和近期执行记录，并说明主任务仍在执行、只需直接回答，避免无工具的分支把自身限制当成任务问题转述给用户。 主回合与追问的用户请求按 operation 分别清理。 fork 的模型请求共用主会话 LLM channel，外部模型轨迹不按 fork 独立采集。

## 5. 模拟用户（`user_simulator`）

### Scripted

| 字段 | 默认值 | 作用 |
|---|---|---|
| `turns` | 必填 | 按顺序发出的用户消息。 每项可以是文本、单个 content part（`{'type': 'audio', 'audio': {...}}`），或 `{'content': ..., 'gate': 'immediate' \| 'after_delivery' \| 'after_progress'}`；`after_delivery` / `after_progress` 在 `episode.py` 里都先等下一条通知送达小脑；若没有进行中的 Run 且 LiveApp 队列为空，则不再空等 |
| `turn_delay_s` | `0` 秒 | 第一条之后每条消息前的等待 |

### LLM（剧本卡）

| 字段 | 默认值 | 作用 |
|---|---|---|
| `client_name` | `user_simulator` | 独立于 policy 的冻结 client |
| `scenario` | 必填 | 剧本卡（`simulation_scenario.ScenarioCard`）：`meta`、`topic`、`user`、`goal`、`context`、`required_nodes`、`optional_insertions`、`event_plan`、`constraints`、`ending` |
| `initial_prompt` | `null`，可由顶层 `prompt` 填充 | 固定开场，作为开场节点交付，不调用用户模型；style A 且还有材料未给时开场节点保持未完成 |
| `system_prompt` | `null` | 为空时由 `simulator.md` 按剧本渲染；显式给出则整段替换 |
| `decision_interval_s` | `30` 秒 | 静默时的决策周期；有新的小脑发言立即重新决策 |
| `progress_interval_s` | `90` 秒 | 两次进度追问的最小间隔，由程序校验 |
| `max_elapsed_s` | `3600` 秒 | 模拟器运行预算，用完即结束用户 |
| `max_turns` | `40` | 用户最多发言数，与 `max_user_turns` 取较小者生效 |
| `decision_retries` | `2` | 决策不合法时带错误原因重试的次数；仍不合法抛 `SimulatorDecisionError` |
| `request_args` | `{}` | 透传给用户模型的请求参数 |

用户模型每次返回一个决策 JSON（`speak` / `wait` / `done`），程序只把 `message` 发给小脑。 决策要等 `episode.py` 实际交付后才推进节点、材料和进度追问时间；通知先到时 `is_message_current` 作废该决策，`done` 也要被接受才记为结束。 结局与节点进度写入任务输出 `metadata.user_simulation`。 协议细节见 `simulator.md` 与 `simulation_state.py`。

## 6. 代码常数

| 常数 | 值 | 位置 | 作用 |
|---|---|---|---|
| 小脑工具 | 7 个：`delegate` / `cancel` / `question` / `open_artifact` / `answer_permission` / `answer_task_input` / `web_search` | `cerebellum_contract.CEREBELLUM_TOOLS` | 除 `web_search` 外都只收自然语言 `text`，结果是 `{"status": "submitted"}` |
| 中脑工具 | `TaskCreate` / `TaskGet` / `TaskList` / `TaskUpdate` / `TaskStop` / `present_files` / `ask_backend` / `answer_permission` / `answer_task_input` / `reply` | `midbrain_contract.MIDBRAIN_TOOLS` | 按 `task_id` 操作，不出现 Run 路径；每一步以单独的 `reply` 结束，`status` 只有 `accepted` / `ok` / `ask` / `failed` |
| 小脑每次回应的模型调用上限 | 6 | `Cerebellum(max_steps=6)` | task 配置不能改 |
| 中脑每一步的模型调用上限 | 8 | `Midbrain(max_steps=8)` | task 配置不能改 |
| 通知优先级 | `failed` / `permission` / `input` 为 `high`，`completed` / `answer` / `cancelled` 为 `normal` | `hylive.harness.NOTICE_PRIORITIES` | core 一次只讲最高优先级的那批后台事实 |
