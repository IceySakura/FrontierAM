# 使用

本文介绍在一个已有任务上跑起 HITL 的 user simulator。

HITL 提供 `hitl/agent_with_user` 作为通用的带 user simulator 的实现，允许接入已有的 nexus task，支持直接将原本的单轮任务多轮化，也支持原生多轮的任务。
---

## 1. 运行一个已有 recipe

HITL 目前提供了接入 `swe` 、`terminal`和 `frontend` 三类任务作为通用 agent_fn的使用示例，每个接入提供了generator与启动脚本示例。generator 在其文件头声明了其支持的所有环节变量配置方式，可参考其说明以及接入 task 本身的特点进行配置。

## 2. 接入另一个 task

接入新 task 时，主要需要重新配置以下项目以运行 user simulator：

- `agent_config` user 侧，`AgentWithUserInput` 声明的字段，见下。
- `agent_config['worker_config']` worker 配置，见 [design_actor.md](design_actor.md) §8。
- `openai_clients` user 与 worker 格子的 client，命名需要对应 `agent_config` 。
- `processors` 用于轨迹、用量与 langfuse 追踪 ，见 [design_label.md](design_label.md) §5。

`agent_config` 常用的字段：

```python
agent_config = {
    'user_client_name': 'human',           # 两侧各用哪个命名 client
    'worker_client_name': 'worker',
    'worker_backend': 'print_claude_code', # 或 pty_claude_code
    'worker_config': worker_config,        # 见下
    'persona': 'default',
    'max_turns': 120,                      # user 的轮次上限
    'requirements': [...],                 # 多轮需求，见 §5
    'skill_dirs': [...],                   # user 自己的 skill 目录，见 §3
    'mcp_servers': [...],                  # 挂给 user 的 MCP server，见 §3
    'extra_tools': ['str_replace_editor'], # persona 之外再给的工具
    'context_window': 200000,              # user 模型的上下文大小，0 则不压缩
    'compact_pct': 70,                     # 历史占到这个比例时压缩
    'reminder_heavy_every': 8,             # 完整提醒的间隔轮数
    'reminder_light_every': 3,             # 一句话提醒的间隔轮数
}
```

`working_dir` 是继承来的必填字段，task 自己调 `run_agent` 时传过就不必再给。

`worker_config` 原样交给 worker manager，由后端各自读取：
```python
worker_config = {
    'version': '2.1.117',                  # 装哪个版本的 Claude Code
    'timeout': 1800,                       # 单轮的时间上限
    'extra_cli_args': [...],               # 工具白名单与黑名单走这里
    'extra_env': {...},                    # 注入沙箱的环境变量
    'mcp_servers': [...],                  # 挂给 worker 的，见 design_actor.md §8
    'user_memory': '...',                  # user 写的工作偏好
    'worker_skills': [...],                # 给 worker 装的 skill 目录，见 design_actor.md §8
    'cc_model': 'claude-opus-4-8',         # Claude Code 对自己显示的模型标识
    'auto_compact_window': 240000,         # harness 自身的自动压缩窗口
    # 以下几项由 manager 转交 host 上的 llm handler 与 proxy，取值形状由
    # HITL_WORKER_CLIENT_PROFILE 选定的 profile 决定，见 design.md §4
    'thinking_preset': 'bedrock-claude',   # proxy 怎么编码 thinking 块
    'no_thinking_request_args': {...},     # 非 thinking 路径的请求参数
    'haiku_model': '...',                  # harness 内部小模型调用走哪个
}
```

### client 与 profile

user 与 worker 各用一个命名 client，通过 `user_client_name` 与 `worker_client_name` 进行区分。

client 由 `HITL_HUMAN_*` 与 `HITL_WORKER_*` 两组环境变量声明，各自的 `_MODEL`、`_BASE_URL`、`_API_KEY` 未给时回落到 `OPENAI_*`。只填 `OPENAI_*` 时两侧用同一个模型。

client 需要使用 `HITL_WORKER_CLIENT_PROFILE` 和 `HITL_HUMAN_CLIENT_PROFILE` 声明请求参数的形状，分别对应 worker 和 user，默认 `openai_basic`。可选的取值与各自适用的模型见 [design.md](design.md) §4。

### processors

三个 processor 都读 `openai_completion` span 并按 label 分组，缺一个就少一份记录，配置形式见 [design_label.md](design_label.md) §5。其中 `hitl_usage` 的 profile 名与 `HITL_WORKER_CLIENT_PROFILE` 对应，langfuse 需要额外 export 相关的凭据。

---

## 3. 配置 user simulator

通用 agent fn 提供三处可配置项：persona 决定 user 的行为模式与特点，加载 skill 扩展其能力，挂载 MCP server 提供额外的工具。

### persona

HITL 内置了以开发者为蓝本的默认 persona，同时支持以 markdown 文件定义 persona，文件名即 persona 名。它通过`HITL_USER_PERSONA_DIRS`声明所处的位置后即可按名称取用。

```markdown
---
tools: [send_message, accept, bash, read]
recalls:
  - "Reviewing: run it once and take it, without going through the change."
reads_memory: false
---

You are someone who ...

### Asking

...
```



更多细节写见 [design_usersim.md](design_usersim.md)

### skill

user 与 worker 两侧都支持挂载 skill。通过 `HITL_USER_SKILL_DIRS` 为 user 声明skill的父目录，`HITL_WORKER_SKILL_DIRS` 为 worker 声明。


### MCP server

`HITL_USER_MCP_SERVERS` 指向一个 JSON 文件，里面是要挂给 user 的 server 列表。一条声明给出 `name` 与连接方式，stdio 的写 `command` 与 `args`，HTTP 的写 `url`：

```json
[
  {"name": "search", "command": "python", "args": ["-m", "my.search_server"]},
  {"name": "docs", "url": "http://127.0.0.1:8080/mcp"}
]
```

连接默认从沙箱内经代理发起，需要由 host 直连时写 `"runtime_name": null`。worker 那一侧用 `HITL_WORKER_MCP_SERVERS`，形式见 [design_actor.md](design_actor.md) §8。

---

## 4. 创建一个新的 recipe

为了接入新的 recipe，可参考已有接入方式配置 user sim 的agent fn，并根据 task 的特点调整 skill、工具和 persona。若要开发高度自定义的逻辑，参考 develop 文档。

---

## 5. 声明多轮需求

单轮的 task 默认 将 `user_prompt` 作为 prompt传给 user simulator，由他决定工作进程。对于多轮的数据集（例如 HITL 原生的swe chain 任务），可以传入代替的 `requirments` 数组，由 user simulator 主动推进。

---

## 6. 筛出真正跑成的实例

`runs.status` 记的是这条 run 有没有把异常抛到 runner，不是任务有没有做成。nexus 要写
`status='failed'` 就得让异常一路抛出去，那样 `task_output` / `agent_output` 都不会落盘，
失败原因也就没了。HITL 选择把 worker 侧的失败收在 agent 里、正常返回结果，所以
session 丢失、user 侧模型限流上抛这些实例在 `runs.status` 里都是 `success`。这是权衡，
不是缺陷 —— 代价是筛结果不能只看 `runs.status`。

判据用 `agent_output.success`：

```sql
SELECT COUNT(*) FROM runs
WHERE agent_output IS NOT NULL AND json_extract(agent_output,'$.success') = 1;
```

一轮 250 条的 opus run 实测，三类结局分得干净，没有错位：

| `run_meta.status` | `agent_output.success` | `agent_output.session_lost` | 条数 |
| --- | --- | --- | --- |
| completed | 1 | 0 | 134 |
| session_lost | 0 | 1 | 21 |
| error | 0 | 0 | 13 |

`agent_output.session_lost` 把两类失败分开：PTY/会话被拆掉，还是 agent 内部报错。

**`task_output.completed` 不能用。** 同一批数据里 21 条 session_lost 有 20 条
`completed=1`，而一条真跑完 6/6 的反而是 0。

`agent_output IS NOT NULL` 这个条件是必须的：`pending` / `running` 的实例没有
`agent_output`，`json_extract` 返回 NULL。

### resume 补跑

`nexus run-batch --resume` 只跳 `status='success'`（`runner/batch/runner.py:567`），
所以上面那些实例会被当成已完成跳过。先把它们改回 `failed`：

```sql
UPDATE runs SET status='failed'
WHERE status='success'
  AND agent_output IS NOT NULL
  AND json_extract(agent_output,'$.success') != 1;
```

之后 resume 会连 `pending` / `running` 一起重跑。db 走本地化时改的是
`$HITL_DB_WORK_DIR/<expname>/<expname>.db`，不是 ceph 上那份。
