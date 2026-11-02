# Codex model catalog

每个文件是一份 codex 的 `model_catalog_json`。`HITL_CODEX_MODEL_CATALOG` 按文件名（不带
`.json`）选一份，generator 读进来放进 `worker_config`，actor 写到 `CODEX_HOME/model_catalog.json`。

catalog 条目决定 codex 给这个 slug 组什么形状的请求。要新增一份，取自 main 的
`docs/examples/configs/swepro-codex-gpt56/model_catalog.json`，不自行导出。

| 文件 | 来源 | `use_responses_lite` |
| --- | --- | --- |
| `gpt-5.6-sol.json` | main `ce6b1895ff` | false |
| `gpt-5.6-sol-lite.json` | main `b752844816` | true |
| `gpt-6-sol.json` | `mugengliu/swe-scaffold-gpt` 的 `swepro-codex-gpt6sol` | false |
| `gpt-6.1-sol.json` | `mugengliu/swe-codex-gpt6-gpt61` 的 `swe-codex-gpt61` | false |
| `gpt-6.1-sol-nc.json` | 同上，`-no-commentary` 那份 | false ← 备查，不作默认 |

两份 gpt-5.6 的差别不止 lite：

| 字段 | `gpt-5.6-sol` | `gpt-5.6-sol-lite` |
| --- | --- | --- |
| `use_responses_lite` | false | true |
| `base_instructions` | 无 | 17730 字符，codex 自己的 system prompt |
| `context_window` | 1000000 | 272000 |
| `max_context_window` | 1000000 | 872000 |
| `auto_compact_token_limit` | 900000 | null |
| `default_reasoning_level` | max | low |
| `supports_search_tool` | false | true |
| `supports_image_detail_original` | false | true |

## worker 收到的 system prompt 来自哪个字段

两个字段都能是它，内容在 `gpt-5.6-sol-lite.json` 里逐字节相同（都是 17730 字符）：

| | `gpt-5.6-sol` | `gpt-5.6-sol-lite` |
| --- | --- | --- |
| `base_instructions` | 无 | 17730 |
| `model_messages.instructions_template` | 17730 | 17730 |

各有一轮实测：0921b（lite）的 `trajectories[1].messages[0].content` 与那份的
`base_instructions` 相等；0923e（非 lite，没有 `base_instructions`）的同一位置是
17730 字符、开头 `You are Codex, an agent based on GPT-5`，只能来自
`instructions_template`。

两个都不带时 codex 用自己内置的那份 —— 这是给新模型换提示词的办法，见下。

## `gpt-6-sol.json`

原样取自官方示例
`docs/examples/configs/swepro-codex-gpt6sol/model_catalog.json`（分支
`mugengliu/swe-scaffold-gpt`，稍后合 main），只重排成 `indent=2`。与
`gpt-5.6-sol.json` 只差三个字段：

| 字段 | `gpt-5.6-sol` | `gpt-6-sol` |
| --- | --- | --- |
| `slug` | `gpt-5.6-sol` | `gpt-6-sol` |
| `display_name` | GPT-5.6-Sol | GPT-6-Sol |
| `description` | Latest frontier agentic coding model. | Workhorse model for coding and everyday work. |

其余 42 个字段逐字节相同，`minimal_client_version` 仍是 `0.144.0`，`model_messages`
（含首行 `an agent based on GPT-5` 的 17730 字符提示词）原样保留。

**Codex CLI 仍钉 0.154.0。** 官方示例明说：不要为了 0.156.1 自带的 picker 升级，而是
注入这份单条 catalog 让 0.154 认这个 slug；也不要把它与 bundled catalog 合并 —— 0 条
或 2 条命中都会报 `Expected exactly one model catalog entry for gpt-6-sol`。

不要自行删字段。试过删 `model_messages` 想让 codex 回落到内置提示词：整块删掉后
`20260924/0924f` 起得来，但只摘掉其中的 `instructions_template` 会留下一个有结构而无
提示词的对象（其余键除 `token_budget` 外本来全是 `null`），codex `exit_code=1` / 0 输出
/ 0.2 秒退出 —— `20260924/0924d` 三条全部 `worker_session_lost`，日志只说
`turn 1 reported no thread id`，看不出与 catalog 有关。

`supported_reasoning_levels` 与端点实际接受的不一致，它是 codex 侧白名单：

| effort | `gpt-5.6-sol` 端点 | `gpt-6-sol` 端点 | `gpt-6.1-sol` 端点 |
| --- | --- | --- | --- |
| `none` | 接受 | 接受 | 接受 |
| `minimal` | 400 | 接受 | 400 |
| `low` / `medium` / `high` / `xhigh` / `max` | 接受 | 接受 | 接受 |
| `ultra` | 接受 | 400 | 400 |

gpt-5.6 与 gpt-6 在 `minimal` 与 `ultra` 上正好相反；gpt-6.1 两个都拒。三份 catalog 里
写的都是 `low/medium/high/xhigh/max/ultra`，所以这个字段不能当端点白名单用。

gpt-6.1 的两条报错各报了一份不同的列表：`minimal` 那条说
`'gpt-6.1-sol-2026-09-29' 支持 low/medium/high/xhigh/max`，`ultra` 那条说
`none/minimal/low/medium/high/xhigh/max` —— 后者是协议层的枚举，前者才是这个模型的。

## `gpt-6.1-sol.json` 与 `gpt-6.1-sol-nc.json`

两份都原样取自官方示例
`docs/examples/configs/swe-codex-gpt61/`（分支 `mugengliu/swe-codex-gpt6-gpt61`），
只重排成 `indent=2`。`gpt-6.1-sol.json` 是官方那份，`-nc` 是同一个包里
`-no-commentary` 的那份。

**上游那个包默认走 `-nc`，我们默认走官方那份** —— 两边相反，跨仓库看时容易记反。上游
`generator.py` 里 `_NO_COMMENTARY = True` 就是它的默认，它的 README 也这么写（`True`（默认）
→ no-commentary）。

**官方那份是要求 worker 发 commentary，不是要求不发。** 判据是 `## Intermediate commentary`
整节，正文写「As you work, you use the `commentary` channel to share concise, meaningful
updates」与「If the user's request requires calling tools, start with a message in the
`commentary` channel」——是要求发。要认一份 prompt 属于哪边，看 `-nc` 的签名句
`Do not send any message in the commentary channel` 在不在（`-nc` 里它是唯一一次提
commentary）。这份的源头 `gpt-6-sol.json` 与更早的 `gpt-5.6-sol.json` 都是 17730 字符、
带这一节、不带签名句，所以自拼版从来没有要求过不发 commentary。

已跑的每一轮 worker prompt 都对得上：`0922f` 起的各轮是 17730 + 有这一节、无签名句，
`20261002` 那三轮同样是 17730（自拼版），`20261003` 是 21769（官方）。

**上一版的 `gpt-6.1-sol.json` 是我们自己拼的**，由 `gpt-6-sol.json` 改 `slug` 与
`display_name` 而来，因此带的是 gpt-5.6 的 17730 字符提示词。`20261002` 那三轮跑的是
它：`trajectories` 的 `messages[0].content` 是 17730 字符、开头
`an agent based on GPT-5`。

与官方那份的差别（左为我们拼的，右为官方）：

| 字段 | 自拼 | 官方 |
| --- | --- | --- |
| `model_messages.instructions_template` | 17730（gpt-5.6 的） | 21769 |
| `model_messages` 的键 | 9 个，6 个是 `null` | 12 个，`approvals` / `auto_review` / `collaboration_modes` / `confirmation_policies` / `guardian_v2` / `multi_agent` / `persistent_instructions` 都有内容 |
| `experimental_supported_tools` | `[]` | `["send_user_message_async", "clock"]` |
| `minimal_client_version` | 0.144.0 | 0.153.0 |
| `shell_type` | `unified_exec` | `shell_command`（0.160 归一成 `unified_exec`，见下） |
| `multi_agent_reasoning_effort` | 无 | `xhigh` |
| `include_apps_usage_instructions` / `include_plugin_usage_instructions` | true | false |
| `service_tiers` / `available_in_plans` / `priority` / `availability_nux` / `description` | 旧值 | 新值 |

### 两份 gpt-6.1 的差别只有一个字段

`models[0].model_messages.instructions_template`，其余逐字节相同。官方那份 21769 字符、
提 `commentary` 12 次；`-nc` 20576 字符、提 1 次（就是那条禁令本身）。7 处改动：

| # | 官方 | `-nc` |
| --- | --- | --- |
| 1 | 拒绝权限的说明放在 commentary 与 final 两处末尾 | 只放 `final` 频道末尾 |
| 2 | 「In progress updates, focus on…」 | 「When you report progress in the `final` channel, focus on…」 |
| 3 | 两个频道：commentary 发进展，final 收尾 | 只留 final 收尾那句，加「Do not send any message in the `commentary` channel.」 |
| 4 | 中途收到用户消息「answer briefly in commentary, then resume」 | 删掉这一句，steering 与「只有明确取消才放弃」保留 |
| 5 | 压缩后「不要重复已发过的 commentary 更新」 | 「Do not restart from scratch or redo completed work.」 |
| 6 | 整节 `## Intermediate commentary`（每 60 秒一条、工具前先发一条、不在其中提问或给终稿、终稿要自洽因为 commentary 会被折叠） | 整节删掉，只留「The final answer must always be fully self-contained.」 |
| 7 | 首次用 skill 要在 commentary 告知 | 删掉，紧随的 skill 权限段不动 |

**commentary 在 HITL 这条路上不是 UI 频道，是 `<worker_progress>` 的内容。** wire 上没有
`channel` 字段，一条 commentary 就是普通的 assistant `content`，经
`backend.steps_of` 变成一条 text step。`20261002` 那轮的 463 个 `<worker_progress>` 块里
发言占 27% 字符（平均 1373 字符、5.3 条 `[used X]`），只有 1 块没有任何发言。关掉
commentary 会让 user simulator 失去那 27%，只剩工具列表与 `<final_response>`。

**默认用官方那份，`-nc` 不作默认。** 在别处用 `-nc` 跑过的观察是：关掉 commentary 后
GPT 大量轨迹不再发 `content`。那正是上面那 27% 的来源，而 HITL 的 user simulator 是靠它
判断 worker 在做什么的——SWE 单轮任务只看最终 patch，少了这些无妨；这条路少了它，user
那侧每轮只剩一串 `[used X]`。所以 `-nc` 留在这里备查，要用它得先想清楚 user 侧拿什么
判断。

### 0.160 实测：`shell_type` 不起作用

`shell_command` 与 `unified_exec` 两种取值下，到达端点的请求体除每次运行自带的随机
id（`prompt_cache_key`、`session_id`、`msg_*`）外逐字节相同，`exec_command` 与
`write_stdin` 的 schema 也一样。`codex debug models` 两种都回显 `unified_exec`，
官方示例写 `shell_command` 在 0.160 上是无差别的选择。

### 0.160 实测：CLI 版本不改提示词

同一份 catalog 在 0.154.0 与 0.160.0 上发出的 `instructions` 逐字节相同（21769），工具
数组同为 10 个同名工具。提示词来自 catalog 的 `instructions_template`，不来自 CLI。两版
wire 的唯一差异在 `update_goal` 的描述（0.160 多一个 `paused` 状态）与
`<skills_instructions>` 的措辞，都与我们的任务无关。

因此**换这份 catalog 不要求升 CLI**，0.154.0 就能跑。

### 两个提问工具都关掉了

官方那份带 `experimental_supported_tools: ["send_user_message_async", "clock"]`，于是
wire 上多出 `request_user_input_async` 与 `clock`（共 10 个工具，自拼那份是 8 个）。

两个提问工具在 `--dangerously-bypass-approvals-and-sandbox` 下都答不了，`HITL_CODEX_USER_INPUT_TOOLS`
默认把它们关掉。失败方式与关法见 `docs/design_codex.md` §4.3；与本目录有关的那一半是：
**我们加载这份 catalog 时去掉 `send_user_message_async` 那一项**，`clock` 保留（这个字段
按项生效）。这是我们对官方文件唯一的内容改动，另一处是 `multi_agent_version` 被
`HITL_CODEX_MULTI_AGENT` 改成 `disabled`。

catalog 带着那一项就等于启用 —— `features.send_message_to_user_async` 两个方向都改不了它
（2×2 实测，见 design 文档），所以这一项只能从 catalog 去掉。

`20261002` 与 `20261003` 两轮共 6 条 worker 轨迹里两者调用 **0 次**（出现全是工具声明本身）。

`features.X` 的值是 bool 不是 table：写 `features.goals.enabled=false` 会报
`Error loading config.toml: invalid type: map, expected a boolean in features`。

### catalog 的 `default_reasoning_level` 只是兜底

`HITL_WORKER_REASONING_EFFORT` 经 `config.toml` 的 `model_reasoning_effort` 下发，与
catalog 的 `default_reasoning_level` 同时存在时实测如下：

| catalog | 旋钮 | wire 上的 `reasoning` |
| --- | --- | --- |
| `max` | `xhigh` | `{"effort":"xhigh"}` |
| 删掉 | `xhigh` | `{"effort":"xhigh"}` |
| `max` | 不设 | `{"effort":"max"}` |
| 删掉 | 不设 | `{}` |

旋钮始终赢，catalog 的值只在旋钮空着时说话。所以两份文件里这个字段写什么，对设了旋钮的
run 没有影响；删掉它则会让漏设旋钮的 run 不发 `reasoning`，端点用自己的默认而不报错。

`supported_reasoning_levels` 是必填：删掉 codex 拒绝启动，报
`failed to parse model_catalog_json path ...: missing field supported_reasoning_levels`。
`context_window` / `max_context_window` / `auto_compact_token_limit` 删掉 wire 不变，但
压缩触发点由 catalog 还是 `config.toml` 的 `model_context_window` 决定**未实测**——
`debug models` 只回显 catalog 自身，要判得跑到 90 万 token。

只翻 lite 不换文件：`HITL_CODEX_RESPONSES_LITE` 覆盖选中 catalog 的每个条目。
