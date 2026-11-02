# Codex 接入

HITL 的第三个 worker harness。本文记录 codex 的外部约定、我们的实现选择、参数从环境变量到
`codex exec` 的完整通路，以及尚未定论的条目。

Codex 与 Claude Code 的差别不只在 CLI：它走 Responses 协议、自带一份 model catalog、
按 provider 名字决定请求形状。这些约定多数不在文档里，是读源码和实测得到的，所以本文逐条
记录来源。

---

## 1. 目录与文件

| 路径 | 内容 |
| --- | --- |
| `actor/print_codex/runtime.py` | 沙箱内：装 CLI、起 proxy、写 `config.toml`、跑 `codex exec` |
| `actor/print_codex/manager.py` | 宿主侧：session 生命周期 |
| `actor/print_codex/backend.py` | 轨迹解析，复用 `agents/codex/json_events.parse_codex_jsonl` |
| `actor/print_codex/spec.py` | harness 声明的路径与前缀 |
| `utils/generator_util/codex.py` | 读环境变量，产出 `worker_config` |

对 `agents/codex/` 的依赖：`installer.install_codex_cli`、`json_events.parse_codex_jsonl`、
`runtime_setup._write_model_catalog_json`。都是正常复用，不是分叉。

---

## 2. CODEX_HOME 与 per-session 的那一个值

`CODEX_HOME` 固定为 `$HOME/.codex`，不带 session id，和一个人自己机器上的 `~/.codex`
一样：一份配置供所有会话。

动态目录试过并且出过事：曾经用 `~/.codex-sessions/<session_id>/`，而
`place_user_memory` 把 AGENTS.md 写到 `spec.MEMORY_PATH`（字面量 `.codex/AGENTS.md`），
两边路径对不上，worker 从来没读到过 AGENTS.md，没有任何报错。

这个目录下的东西按性质分：

| | 来源 | 性质 |
| --- | --- | --- |
| `AGENTS.md` | `place_user_memory`，`ensure_env` 一容器一次 | 全容器同一份 |
| `model_catalog.json` | `worker_config['model_catalog_json']` | 全容器同一份 |
| codex 持久化的 thread | codex 自己写，`resume <id>` 按 id 读回 | 按 id 索引，共存不冲突 |
| `config.toml` | `_write_config` | 除 `base_url` 外全容器同一份 |

**唯一 per-session 的是 proxy 端口**：每个 session 起自己的 `ResponsesProxy`，端口各不
相同。它由 `run_turn` 每轮以 `-c model_providers.proxy.base_url=...` 下发，不靠共享的
`config.toml`——那份文件里的 `base_url` 是最后一个开的 session 写的，只作兜底。

所以一个容器可以开多个 session。`config.toml` 被后开的 session 重写时，变的只有那个会被
命令行覆盖掉的值。

### AGENTS.md 的两层发现规则

codex 读两层，各自独立：

- **全局层**：只看 `$CODEX_HOME`，先找 `AGENTS.override.md`，没有再找 `AGENTS.md`，取第一个非空的。
- **项目层**：从 git 根向 cwd 逐级找 `AGENTS.md`，按 root→cwd 顺序拼接。

我们用的是全局层。

---

## 3. config.toml

`runtime.py` 的 `_write_config` 逐行拼出来。顶层键必须写在第一个 table header 之前，否则
会被归进那个 table。

| 键 | 来源 | 说明 |
| --- | --- | --- |
| `model` | `config['codex_model']` | codex 自己显示与记账用的 slug。真实模型由 proxy 每次调用时替换 |
| `model_catalog_json` | §4 | 指向写在 `CODEX_HOME` 下的 catalog 文件 |
| `model_context_window` | `config['model_context_window']` | |
| `model_auto_compact_token_limit` | `config['auto_compact_token_limit']` | codex 在 `min(limit, window × 90%)` 处压缩，所以两个都要写 |
| `web_search` | 硬写 `"disabled"` | proxy 不提供搜索后端 |
| `model_provider` | 硬写 `"proxy"` | 只是 `[model_providers.*]` 的表名 |
| `[features] remote_compaction_v2` | 硬写 `false` | 远程压缩会 POST 到 `/responses/compact`，proxy 不提供 |
| `[tools.experimental_request_user_input] enabled` | `config['user_input_tools']` | 假值时才写。见 §4.3 |
| `[model_providers.proxy] name` | `config['use_azure_provider']` | 见下 |
| `base_url` | 本 session 的 proxy 端口 | `http://127.0.0.1:<port>/v1` |
| `wire_api` | 硬写 `"responses"` | |

`config.toml` 里的内容到此为止。要改别的键走 `HITL_CODEX_CONFIG_OVERRIDES`，它转成
`codex exec -c key.path=value`——key 自带层级，顶层键和 table 字段都能到位。曾经有一个
`HITL_CODEX_EXTRA_CONFIG_TOML` 把原始 TOML 追加在文件末尾，那里已经在
`[model_providers.proxy]` 之内，写一个顶层键会变成 provider 的字段，不报错也不生效。
| 末尾 | `config['extra_config_toml']` | 原样追加 |

### provider 名字为什么是个开关

codex 内部有个 `is_azure_responses_provider()`，按 provider 的 name 或 base_url 判断。
我们的 base_url 是 `127.0.0.1`，没有 azure 标记，**只能靠 name**。

判为真时 codex 会：发 `store=true`、重放上游原始的 reasoning item id、0.136 以后走原生远程
compaction。这三件事都发生在请求到达我们 proxy 之前，**proxy 改不了**。

我们的 proxy 无状态，所以默认 `use_azure_provider=false`。

---

## 4. model catalog

codex 自带一份 bundled catalog，按 slug 决定这个模型的请求形状。

catalog 走 codex 官方的 `model_catalog_json` 配置字段：generator 把一份完整 catalog 放进
`worker_config['model_catalog_json']`，actor 用 `agents/codex/runtime_setup._write_model_catalog_json`
写到 `CODEX_HOME/model_catalog.json`，再在 `config.toml` 里指过去。

### 四份 catalog

文件在 `hitl/assets/codex/`，`HITL_CODEX_MODEL_CATALOG` 按文件名选。都取自 main 或官方示例
分支的权威条目，不另行导出。字段差异见那个目录的 `README.md`。

| 文件 | 来源 | `use_responses_lite` |
| --- | --- | --- |
| `gpt-5.6-sol` | `ce6b1895ff` | false ← 默认用它 |
| `gpt-5.6-sol-lite` | `b752844816` | true |
| `gpt-6-sol` | `mugengliu/swe-scaffold-gpt` 的 `swepro-codex-gpt6sol` | false |
| `gpt-6.1-sol` | `mugengliu/swe-codex-gpt6-gpt61` 的 `swe-codex-gpt61` | false |
| `gpt-6.1-sol-nc` | 同上，`-no-commentary` 那份 | false |

两份 gpt-6.1 只差 `model_messages.instructions_template` 一个字段：官方那份 21769 字符、提
`commentary` 12 次，`-nc` 20576 字符、提 1 次（禁令本身）。**commentary 在 HITL 这条路上不是
UI 频道**，wire 上没有 `channel` 字段，一条 commentary 就是普通的 assistant `content`，经
`backend.steps_of` 成为 `<worker_progress>` 里的一条发言。`20261002` 那轮 463 个
`<worker_progress>` 块里发言占 27% 字符，只有 1 块没有任何发言——关掉 commentary 会让 user
simulator 失去那一部分，只剩工具列表与 `<final_response>`。

`ce6b1895ff` 的 MR 描述写的是「One-line change，`use_responses_lite` true → false」。
文件是单行 JSON，行数上成立，内容上变了 8 个字段、42KB → 24KB：除 lite 外还删掉了
`base_instructions`，`context_window` 272000→1000000、`default_reasoning_level` low→max、
`supports_search_tool` 与 `supports_image_detail_original` 都转 false。

`base_instructions` 带的时候就是 worker 的 system prompt：0921b 那轮轨迹
`trajectories[1].messages[0].content` 与 `gpt-5.6-sol-lite.json` 的这个字段逐字节相等
（17730 字符）。`gpt-5.6-sol.json` 不带，codex 用自己内置的那份。

### use_responses_lite

跟随所选文件。`HITL_CODEX_RESPONSES_LITE` 设了才覆盖每个条目，用来只翻这一项而不换文件。

两种取值在 wire 上的差别**尚未实测**。早先代码注释里写过「lite 请求不带顶层 tools 数组，
Responses→Chat proxy 看不到工具」——这句没有可追溯的来源，不作为依据。要判需要抓两种配置下
到达 proxy 的实际 payload，比对顶层 `tools`、并行调用形状、上下文字段。

### 4.1 multi_agent_version

`HITL_CODEX_MULTI_AGENT` 覆盖每个条目，取 `disabled` / `v1` / `v2`，默认 `disabled`。
`assets/codex/` 四份文件自带的都是 `v2`。

codex 0.154 上的实测（`codex debug prompt-input` 渲染的 developer 消息）：

| 值 | prompt |
| --- | --- |
| `v2` | 带 `multi_agent.usage_hint` 与 `multi_agent.mode_instructions` 两条 developer 消息，共 15147 字节 |
| `v1` | 两条都没有，11760 字节 |
| `disabled` | 两条都没有，11761 字节 |

0.160 上抓到达端点的实际请求体，`gpt-6.1-sol.json`，三个值的工具与 developer 消息：

| 值 | 工具 | developer 消息 |
| --- | --- | --- |
| `disabled` | 10 个，无 agent 工具 | 1 条（skills），3230 字符 |
| `v1` | 11 个，多 `multi_agent_v1{close_agent,resume_agent,send_input,spawn_agent,wait_agent}` | 1 条，3230 字符 |
| `v2` | 11 个，多 `collaboration{followup_task,interrupt_agent,list_agents,send_message,spawn_agent,wait_agent}` | 3 条，5930 字符 |

`v1` 与 `disabled` 的 developer 消息逐字相同，工具差一个命名空间——这补上了之前只凭
`debug prompt-input` 判不出的那一条。`v2` 比 `disabled` 多 2700 字符，每轮都在 prompt 里。

**非法值被静默吞掉**：`codex debug models` 回显解析后的 catalog，写 `zzz_bogus` 回显是 `null`，
三个合法值原样回显。所以 generator 先挡一道，不让打错的值退化成 codex 自己的默认。

至于跑过的 run 用的是哪套协议：0921b 之前的轨迹 prompt 里是 `to=functions.collaboration.spawn_agent`、
`fork_turns`、`followup_task`、`list_agents`，没有 `multi_agent_v1.*` / `send_input` / `resume_agent`，
即 V2。

### 4.2 catalog 决定提示词，CLI 版本不决定

同一份 catalog 在 0.154.0 与 0.160.0 上发出的 `instructions` 逐字节相同（`gpt-6.1-sol.json`
是 21769 字符），工具数组同为 10 个同名工具。worker 的 system prompt 来自 catalog 的
`model_messages.instructions_template`，所以**换 catalog 不要求升 CLI**。

两版 wire 的唯一差异在 `update_goal` 的描述（0.160 多一个 `paused` 状态）与
`<skills_instructions>` 的措辞；`features.goals=false` 能把那三个 goal 工具一并去掉。

`shell_type` 在 0.160 上不起作用：`shell_command` 与 `unified_exec` 两种取值的请求体除每次
运行自带的随机 id 外逐字节相同，`debug models` 两种都回显 `unified_exec`。

### 4.3 worker 的两个提问工具都关掉了

两个都答不了，`HITL_CODEX_USER_INPUT_TOOLS` 默认把它们关掉（设 1 可以要回来）。失败方式不同：

| 工具 | codex 在 bypass 模式下回给 worker | 对 run 的影响 |
| --- | --- | --- |
| `request_user_input` | `request_user_input is unavailable in Default mode` | 看到拒绝后继续干活，浪费一次调用 |
| `request_user_input_async` | `{"accepted":true}`，并把问题当一条 `agent_message` 发出 | 问题变成 `<worker_progress>` 里的发言，没有 `<worker_question>` 标记，worker 以为问过了而永远等不到答案 |

**一个旋钮，但要在两个地方关** —— 两者各只认自己那一处：

| 工具 | 关它的地方 | 实现 |
| --- | --- | --- |
| `request_user_input_async` | catalog 的 `experimental_supported_tools` 去掉 `send_user_message_async` | generator 的 `_without_async_user_input` |
| `request_user_input` | `config.toml` 的 `[tools.experimental_request_user_input] enabled = false` | generator 放 `user_input_tools`，actor 的 `_write_config` 写那个 table |

只做一半就漏一个：只写 config 键时 `request_user_input_async` 还在（9 个工具），只删 catalog
项时 `request_user_input` 还在（9 个），两个都做才是 8 个。

**catalog 里带着它就等于启用**，`features` flag 两个方向都改不了：

| catalog 带 `send_user_message_async` | `features.send_message_to_user_async` | `request_user_input_async` | 工具数 |
| --- | --- | --- | --- |
| 有 | 默认（false） | 在 | 10 |
| 有 | 显式 false | 在 | 10 |
| 有 | 显式 true | 在 | 11 |
| 无 | 显式 true | 不在 | 10 |
| 无 | 默认 | 不在 | 9 |

那个 flag 管的是**另一个**工具：开它时下发一个叫 `send_message_to_user_async` 的工具（名字与
catalog 那一项相近但词序不同），两行都 +1。

`experimental_supported_tools` 按项生效，不是整体开关：只留 `['clock']` 时 async 那个消失而
`clock` 还在，所以不必清空这个字段。

那个 config 键的值是 struct 不是 bool：写 `tools.experimental_request_user_input=false` 报
`invalid type: boolean false, expected struct ExperimentalRequestUserInput`，字段名是 `enabled`。
**打错的键会被报出来**，顶层和 table 里都报：codex 发一条 `item.completed` 的 `error`，说
`` `tools.zzz_bogus` is ignored ``，不是静默吞。`-c` 与写进 `config.toml` 两种下发方式等效
（都实测过），写进文件是因为这是全 session 共享的一个值，与 `[features]` 同一个做法。

生效后的 wire：8 个工具（`exec_command` / `write_stdin` / `clock{curr_time,sleep}` /
`apply_patch` / `view_image` / 三个 goal），提示词仍是 21769 字符，无 ignored-key 警告。

试过但不改工具数组的：`features.default_mode_request_user_input=false`（默认就是 false）、
`features.request_user_input=false`（未知 flag）、`tools.request_user_input=false`、
`approval_policy="never"`、`model_messages.tools=[]`。同一机制下 `features.goals=false` 去掉 3
个、`features.view_image=false` 去掉 1 个。

`20261002` 与 `20261003` 两轮（两个工具都在）共 6 条 worker 轨迹里，两者调用 **0 次** —— 关掉
它们是去掉一个失败模式，不是修一个正在发生的故障。

### 历史：运行时改写

曾经在 `agents/codex/runtime_setup.py` 里实现过一套运行时改写（`_catalog_rewrite_kind`、
`_patch_catalog`、`_maybe_write_classic_catalog`，约 161 行）：起一个 codex 子进程跑
`debug models --bundled` 导出 catalog，按一张「模型 → 最低 codex 版本」的硬编码表决定要不要
改，再盖上 `tool_mode` / `use_responses_lite` / `apply_patch_tool_type` 三个字段。

放弃的原因：那张表是把上游 catalog 的演进史抄进我们的代码，codex 每发一版都可能失配，而失配
没有任何信号——返回 `None` 就是「不改写」，worker 静默失去工具。换成显式 catalog 文件后，
要什么写什么，slug 对不上时 codex 直接报错。

---

## 5. 参数通路

```
run 脚本的环境变量
  → utils/generator_util/codex.py     读 env，产出 worker_config
  → manager                            持有 worker_config
  → actor/print_codex/runtime.py       只读 worker_config 的键
```

**环境变量只在 generator 层读。** actor 层只读语义明确的配置项，它唯一碰的环境变量是
`HOME`——那是环境本身，不是配置。

### generator 读的变量

| 变量 | 默认 | 落到 |
| --- | --- | --- |
| `HITL_WORKER_VERSION` | `0.154.0` | `version` |
| `HITL_WORKER_TURN_TIMEOUT` | 7200 | `timeout` |
| `HITL_WORKER_CONTEXT_WINDOW` | — | `model_context_window` + `auto_compact_token_limit` |
| `HITL_CODEX_MODEL` | — | `codex_model`，catalog 里必须有这个 slug |
| `HITL_CODEX_MODEL_CATALOG` | — | `model_catalog_json`，见 §4 |
| `HITL_CODEX_RESPONSES_LITE` | 跟随 catalog | 覆盖 catalog 里那个字段 |
| `HITL_CODEX_MULTI_AGENT` | `disabled` | 覆盖 catalog 的 `multi_agent_version`，见 §4.1 |
| `HITL_CODEX_USER_INPUT_TOOLS` | `false` | 真值保留两个提问工具，见 §4.3 |
| `HITL_CODEX_AZURE_PROVIDER` | `false` | `use_azure_provider` |
| `HITL_CODEX_WRAP_PLAINTEXT_REASONING` | `false` | 传给 proxy 的 converter |
| `HITL_CODEX_CONFIG_OVERRIDES` | — | `extra_cli_args`，转成 `-c key.path=value` |
| `HITL_WORKER_NPM_REGISTRY` | — | `npm_registry` |

前三个是 harness 无关的：一个 run 只装一个 harness，按 harness 分别命名会导致设了不匹配的那个
而另一个被静默忽略。

### client 侧

worker client 走 `openai_responses`（只有 `gpt_responses` 这一个 profile 用它）。
`generator_util/clients.py` 的 `_with_responses_worker_args` 按 `client_name` 挡住其它 client，
所以 user 那侧的 litellm client 不受影响：

| 字段 | 层 | 值 |
| --- | --- | --- |
| `capture_raw_io` | `client_args` | `true`，`HITL_WORKER_CAPTURE_RAW_IO` 可关。原始 Responses 包进 `provider_call`，对密文时要看 |
| `prompt_cache_key` | `request_args` | 每 run 一个，网关按它缓存前缀 |
| `stream` | `request_args` | `false`。client 拒绝流式调用 |
| `max_output_tokens` | profile | 128000 |
| `max_retries` | `client_args` | 10（`builders.py` 的共享默认，client 自己的默认是 5） |

---

## 6. reasoning 的两个开关

`wrap_plaintext_reasoning_content` 与 `use_azure_provider` **不是一一对应**，管的是两件事：

- `wrap_plaintext_reasoning_content`：上游只给了 plaintext、没给 id 时，要不要**编造**一个 `rs_*`。
  编造出来的 id 会被 codex 在下一轮重放，无状态网关答 `rs_ not found`。GPT 走这条路要关。
  HY3/GLM 的 plaintext CoT 靠它才能重放，那些任务要开。
- `use_azure_provider`：上游给了**真** id 时，重放要不要带回去。

两者都可以取真值，不冲突。

---

## 7. 一轮怎么渲染给 user

`backend.steps_of` 把 `item.completed` 事件转成 `text` / `tool_use` 两种 step，
`manager.report` 交给 `rendering.report_from_steps`——那条链与两个 Claude Code 后端共享，
`<worker_progress>` 的形态因此一致：按时序，连续工具调用合成一块，每行一个 `[used X]`。

**工具名报 codex 自己的**，不映射成 Claude Code 的。曾经把 `command_execution` 报成 `Bash`、
`file_change` 报成 `Edit`，那是在告诉 user simulator worker 用了它根本没有的工具。

分层：`actor/rendering.py` 只有通用接口，不认识任何 harness 的工具名——`read_steps` 收
`{type, name, detail}`，`detail_from(arguments, fields)` 按给定字段名取第一个非空值。每个
harness 的词汇在自己目录下的 `rendering.py`：codex 的在 `print_codex/rendering.py`
（`TOOL_ITEM_TYPES` + `item_detail`），Claude Code 的在 `print_claude_code/rendering.py`
（`TOOL_DETAIL_FIELDS` + `steps_with_detail`，PTY 后端也读它）。

`item_type` 的取值来自 `agents/codex/json_events` 的 docstring，那份是照 codex
`rust-v0.146.1` 源码（`codex-rs/exec/src/exec_events.rs`）写的。

**0.160 把判别键从 `item_type` 改成 `type`。** 实测该版本的 `item.completed` 发的是
`{"id":"item_0","type":"command_execution","command":...}`，没有 `item_type` 键。我们两处读它
的地方都已有回落——`json_events._item_type` 与 `print_codex/backend.py` 都写
`item.get('item_type') or item.get('type')`——所以 0.160 照样解析得出。payload 的字段名没变：
`command_execution` 带 `command` 与 `exit_code`、`file_change` 带 `changes[].path`、
`agent_message` 带 `text`。`resume <thread_id>` 在 0.160 上回同一个 thread id。

| item_type | 报作 | detail 取自 |
| --- | --- | --- |
| `agent_message` | speech | `text` |
| `command_execution` | `command_execution` | `command` |
| `file_change` | `file_change` | `changes[].path`，逗号连接 |
| `web_search` | `web_search` | `query` |
| `collab_tool_call` | `collab_tool_call` | `prompt` |
| `mcp_tool_call` | `mcp_tool_call` | 无——schema 没给字段名 |
| `todo_list` | `todo_list` | 同上 |
| `reasoning` | 不收 | |

`reasoning` 不进 steps。turn 级失败由 `_failure_message` 走 `<note>` 单独报，不混进工具列表。

`<final_response>` 取最后一条 `agent_message` 的**全文**，没有长度上限；PTY CC 那侧的
`output` 是 350/150 摘要。这个差异的实际影响没量过。

---

## 8. 已知待办

- **`use_responses_lite` 的 A/B 尚未做**（§4）
- **`parallel_tool_calls` 谁说了算，没定**。`openai_responses.py` 组装请求是
  `{**config.request_args, **per_call}`，codex 自己发的值在后面，所以写在 `request_args` 里
  的设不生效——main 的 generator 也写在那里，它的生产同样跑着 codex 的值。我们现在不设这个字段。
  要真正压住得用 `enforced_request_args`（`openai_responses.py:1134`，在合并之后应用），但
  「该不该压」本身还没有依据。
- **`max_output_tokens` / `max_retries` 的取值没有依据**：128000 抄的是 main 的 generator，
  `max_retries` 保持 `builders.py` 的 10。两个数各自为什么是那个值，两边都不知道。
- **`print_codex/runtime.py` 与官方 `agents/codex/runtime_setup.py` 没有逐项对照过**。
  我们的 runtime 是重新实现的，config.toml 的键已比过（§3），但官方那侧还有哪些参数我们没有、
  漏掉的是不是有意的取舍，尚未核。
