# HyCode 单轮 prompt 改为 stdin 传入

日期：2026-10-19
状态：已实现，已在 gongfeng sandbox 完成冒烟和请求等价性验证（见第 6.2、6.3 节）

## 1. 摘要

HyCode 单轮启动时把完整任务 prompt 作为命令行位置参数传给 `opencc-bin`。Agent 在 sandbox 里用
`pkill -f <名字>` 清理进程时，匹配的是整条命令行；只要 `<名字>` 在 prompt 里出现过，就会把 HyCode
自己（以及外层包装进程）一起杀掉，rollout 直接中断。

本方案沿用 `claude_code` 单轮在 !849 中的做法：启动 CLI 时命令行不带 prompt，进程起来后通过 stdin 管道写入
prompt 并关闭。模型收到的第一条 user 消息内容不变，所有进程的命令行里不再出现 prompt。

`hycode_dynamicagent` 单轮有同样的问题（`-p <prompt>`），一并修复。改动集中在两个 agent 的 `runtime_setup.py`
和 `installed_common` 中一个共用的 stdin 写入函数，RunInput、数据配置和 harness 不需要改。

## 2. 背景

### 2.1 现象

pb、nl2product 这类任务的 prompt 很长，里面列了大量进程名和文件路径。Agent 到后期想停掉某个进程时，
即使已经写得很具体（例如 `pkill -f nl2product_worker.py`，而不是 `pkill python`），也会命中 HyCode
进程，因为它的命令行里带着 prompt 原文：

```text
PID  CMD
100  opencc-bin --dangerously-skip-permissions --print ... -- 你是一个……请修复 nl2product_worker.py，修改后重启 nl2product_worker……
230  python /workspace/nl2product_worker.py --port 8080
```

`pkill -f nl2product_worker.py` 会同时杀掉 100 和 230。PID 100 就是 agent 本身。

目前 RL 侧的处理是对这类样本给 0 分，希望模型学会只杀具体进程。但上面的情况和模型杀得是否具体无关：
只要目标名字在 prompt 里出现过就躲不开，这个惩罚信号是有偏的。根因在 Nexus 的启动方式。

`pkill` / `pgrep` 不带 `-f` 时只匹配进程名（`/proc/<pid>/comm`），带 `-f` 时匹配完整命令行
（`/proc/<pid>/cmdline`）。Agent 为了区分 `python a.py` 和 `python b.py`，通常会用 `-f`。

### 2.2 当前代码

`nexus/extensions/agents/hycode/runtime_setup.py`：

```python
cmd: list[str] = [str(binary), '--dangerously-skip-permissions']
if extra_cli_args:
    cmd.extend(extra_cli_args)
cmd.extend(['--', task])
```

`_run_hycode_streaming` 启动子进程时使用 `stdin=subprocess.DEVNULL`。

`extra_cli_args` 来自 `hycode/generator.py` 和 `tasks/artifact/agent_setup/harness/hycode.py`，已经包含
`--print --input-format text --output-format stream-json`。

RunInput 无法绕开这个问题：`HyCodeInput.extra_cli_args` 的语义是“追加在 prompt 之前的参数”，
`-- task` 是硬编码的，没有字段可以去掉它或改成 stdin。

### 2.3 三种执行模式下 prompt 的暴露面

| 模式 | 包装方式 | 命令行里带 prompt 的进程 |
|---|---|---|
| 本地 | 直接启动 | `opencc-bin` |
| Direct（root worker 切换用户） | `su <user> -s /bin/sh -c "cd … && exec env … opencc-bin … -- '<prompt>'"`（`installed_common/execution.py`） | 常驻的 `su` 父进程和 `opencc-bin`。`su` 被命中后会把 SIGTERM 转发给子进程 |
| Compose | `docker compose … exec -T … bash -c '…exec "$@"' … opencc-bin … -- '<prompt>'`（`installed_common/compose.py`） | 宿主机上的 `docker compose` 进程，以及容器里的 `opencc-bin` |

另外，`runtime_setup.py` 中的 `logger.info('Running HyCode: %s', …)` 会把整段 prompt 打进日志。

### 2.4 其他 agent 的情况

- 多轮路径（`installed_common/multiturn.py` 的 `build_stream_json_command`）使用 `--input-format stream-json`，
  prompt 以 JSONL 写入 stdin，命令行不带 prompt，没有这个问题。
- `claude_code` 单轮已在 !849 改为 stdin 传入（`_build_single_run_command` 与 `_run_claude_streaming(stdin_input=...)`），
  起因同样是 agent 的 `pkill -f` 杀掉了 claude 父进程。
- `hycode_dynamicagent` 单轮（`setup_and_run_hycode_dynamicagent`）以 `-p <prompt>` 传参，同样受影响，本次一并修复。
  只有 `hycode_dynamicagent_multiturn` 走 stream-json。
- `opencc`、`workbuddy`、`pi_agent`、`handy`、`cline` 单轮也把 prompt 放在命令行。它们是不同的 CLI，是否支持从 stdin
  读 prompt 需要逐个确认，不在本次范围内。

## 3. 方案调研

### 3.1 Claude Code CLI 接收用户 prompt 的方式

依据 [Claude Code CLI reference](https://code.claude.com/docs/en/cli-reference)：

- 用户 prompt 只有两种来源：命令行位置参数，或 stdin。
  - `--input-format text`：读取 stdin 全部内容作为 prompt（文档示例 `cat file | claude -p "query"`）；
  - `--input-format stream-json`：stdin 逐行写入 JSON 格式的 user 消息，支持多轮。
- 带 `-file` 的 prompt 参数只有 `--system-prompt-file`、`--append-system-prompt-file`、
  `--append-subagent-system-prompt-file`，作用对象都是 system prompt。**没有从文件读取用户 prompt 的参数。**

### 3.2 Claude Agent SDK 的做法

SDK 启动 CLI 子进程，通过 stdin 以 stream-json 格式写入消息，命令行不带 prompt。SDK 没有比 CLI 多出的
prompt 输入入口。

### 3.3 仓库内先例

`claude_code` 单轮（!849）已经在线上用 `--print` + stdin 管道传 prompt，证明上游 Claude Code CLI 在这种方式下
行为正确。`opencc-bin` 是它的 fork，仍需第 6.3 节验证。

### 3.4 不采用的方案

- **prompt 写入 `--append-system-prompt-file`，user 消息只写一句引导语**：任务内容从 user 消息挪到了
  system prompt，和训练、推理时的数据分布不一致。
- **user 消息写“请读取 /tmp/task.md”**：agent 多一步读文件的工具调用，任务语义和轨迹都会改变。
- **匿名临时文件作为 stdin（`tempfile.TemporaryFile()`，写好后 `seek(0)` 传给子进程）**：实现最简单，没有并发写入、
  死锁和断管问题。但它依赖 worker 临时目录可写，prompt 会落盘，运行期间可从 `/proc/<pid>/fd/0` 读到完整 prompt，
  并且和 `claude_code` 的管道写法不一致。选择管道。

## 4. 设计

### 4.1 改后的进程命令行

本地模式：

```text
# 现在
opencc-bin --dangerously-skip-permissions --print --input-format text --output-format stream-json \
  --verbose --no-session-persistence --disallowedTools WebSearch WebFetch --max-turns 250 -- '<prompt>'
# 改后
opencc-bin --dangerously-skip-permissions --print --input-format text --output-format stream-json \
  --verbose --no-session-persistence --disallowedTools WebSearch WebFetch --max-turns 250
```

Direct 和 Compose 模式下，外层包装命令同样不再带 prompt。`su` 和 `docker compose exec -T` 都会把 stdin
透传给内层进程。

### 4.2 prompt 写入方式

`setup_and_run_hycode` 不再执行 `cmd.extend(['--', task])`，改为 `_run_hycode_streaming(..., stdin_input=task)`。
`setup_and_run_hycode_dynamicagent` 把 `cmd.extend(['-p', task])` 改为只追加 `-p`，并同样传 `stdin_input=task`。

两个 `_run_*_streaming` 的处理一致：

1. 传了 `stdin_input` 时以 `stdin=subprocess.PIPE` 启动子进程，否则仍为 `DEVNULL`；
2. 先启动 stdout、stderr 读取任务，再写 stdin，避免 prompt 超过管道缓冲区（通常 64KB）时双向写满导致死锁；
3. stdin 写入由 `installed_common.write_and_close_stdin` 完成：原样写入 UTF-8 编码的 prompt，不追加换行，
   `drain()` 后关闭 stdin；子进程启动即失败时出现的 `BrokenPipeError` / `ConnectionResetError` 直接忽略，失败原因由
   退出码和 stderr 体现；
4. 写入任务和 stdout、stderr 读取任务放在同一个 `gather` 里，受同一个超时控制（HyCode 下包括可暂停的
   `registered_command_timeout`），超时时一起取消。这一点与 `claude_code` 不同：它在超时范围外 `drain()`，子进程卡住
   不读 stdin 且 prompt 超过管道缓冲区时会一直挂起。

CLI 在 `--print --input-format text` 且没有位置参数时读到 EOF 后开始执行，把 stdin 全部内容作为第一条
user 消息，和位置参数的效果相同。

### 4.3 约束

- 命令行不能再出现任何位置参数。位置参数和 stdin 同时存在时，CLI 会把两者拼接成一条 prompt。
  自定义 `extra_cli_args` 不得包含位置参数。
- HyCode 的 `extra_cli_args` 必须包含 `--print`（`hycode_dynamicagent` 由 runtime 追加 `-p`），且输入格式不能是
  `stream-json`。`--input-format text` 是 CLI 默认值，写不写都可以。现有 generator 和 artifact harness 已满足。
- 以前 prompt 前的 `--` 也会结束 `--disallowedTools` 这类可变长参数。现在命令行末尾没有位置参数，不再需要它。

### 4.4 附带收益

- Linux 单个命令行参数上限为 128KB（`MAX_ARG_STRLEN`）。现在超长 prompt 会因 `E2BIG` 启动失败，改为 stdin
  后没有这个限制。
- `Running HyCode` 日志不再输出整段 prompt。

### 4.5 不在本次范围内

Claude Code 的 Bash 工具会把命令包在 `bash -c '…'` 中执行。Agent 运行 `pkill -f foo` 时，这层 shell 的命令行
同样包含 `foo`，会被自己的 `pkill` 杀掉，表现为本次工具调用返回 143；agent 进程不受影响。这是 `pkill -f`
的通用行为，与 Nexus 无关。

## 5. 兼容性与迁移

- `setup_and_run_hycode` 通过 `call_runtime_agent` 在 runtime 内执行，改动属于 runtime 侧代码。开启
  `runtime_configs.<name>.worker_venv: true` 时随 wheel 立即生效；其他场景需要更新 runtime wheel 或镜像。
- 宿主侧传给 runtime 的参数仍是 `task=...`，接口不变。宿主与 runtime 版本不一致时，旧 runtime 继续使用位置
  参数，不会报错，可以逐步替换。
- `HyCodeInput`、RunInput、parquet、`hycode/generator.py`、`hycode_dynamicagent/generator.py`、artifact harness
  都不需要修改。`HyCodeInput.extra_cli_args` 的 docstring 已同步更新，不再描述为“prompt 之前的参数”。
- 默认不加开关，回滚方式为 revert。若线上要求开关，可在 `HyCodeInput` 增加 `prompt_via_stdin: bool = True`。

## 6. 验证计划

### 6.1 单元测试（已完成）

`tests/unit/extensions/agents/test_hycode.py` 的 `TestHyCodePromptViaStdin`：

- 用真实子进程（Python 脚本）记录自身 `sys.argv` 和读到的 stdin：stdin 与 prompt 逐字节相等，argv 不含 prompt。
  prompt 覆盖中文、换行、引号、`$HOME`，长度超过 300KB；
- 子进程不读 stdin 直接退出时不抛异常，返回它自己的退出码和 stderr；
- 子进程一直不读 stdin 时，`drain()` 受超时约束，按超时返回；
- `setup_and_run_hycode` 传给 `_run_hycode_streaming` 的命令里没有 prompt 和 `--`，prompt 通过 `stdin_input` 传入。

`tests/unit/extensions/agents/test_hycode_dynamicagent_runtime.py` 对 `hycode_dynamicagent` 覆盖逐字节写入，以及命令行
以 `-p` 结尾、不含 prompt。

传给 `transport.prepare` 的命令已不含 prompt，`su` 和 `docker compose exec -T` 不会再加入 prompt，因此不再为三种
模式单独测 `prepared.command`。stdin 能否穿过这两层包装由第 6.2 节实测。

### 6.2 真实 sandbox 冒烟

使用 `worker_venv: true` 运行 1 条 pb 或 nl2product 任务：

- 运行中检查 `cat /proc/<opencc-bin pid>/cmdline | tr '\0' ' '`，确认不含 prompt；
- 构造复现任务：prompt 中提到 `foo_worker`，要求 agent 启动 `foo_worker` 后执行 `pkill -f foo_worker`。
  修复前 agent 被杀，修复后应正常跑完；
- 至少跑一条 Compose 模式任务。这是唯一一处 stdin 要经过 API 转发的路径（宿主 `docker compose exec -T` 转发进容器），
  要确认 prompt 和 EOF 都能到达容器内的 `opencc-bin`。较新的 Claude Code 在 stdin 不是 TTY、且约 3 秒内没有收到数据时
  会不再等待、直接继续，需要确认 Compose 启动延迟不会触发它（表现为 CLI 报没有 prompt，或 stderr 出现
  `no stdin data received` 一类警告）；
- `hycode_dynamicagent` 单轮跑一条任务，确认 wrapper 在 `-p` 之后追加的 `--agents` / `--plugin-dir` 仍然生效。

2026-10-20 实测结果（gongfeng、`ubuntu:24.04`、`worker_venv: true`、HyCode 0.1.10、`deepseek/deepseek-v4-pro`，`identity`
任务）。prompt 要求 agent 依次打印 `opencc-bin` 的 cmdline、启动 `exec -a foo_worker_demo sleep 3600`、执行
`pkill -f foo_worker_demo`、确认进程已停：

| 版本 | `opencc-bin` cmdline | `pkill -f foo_worker_demo` 之后 | 结果 |
|---|---|---|---|
| 修改前 | 末尾带 `-- '<完整 prompt>'`，`pgrep -af foo_worker_demo` 命中 `opencc-bin` 自己 | `opencc-bin` 被杀 | 退出码 143，`result_count=0` |
| 修改后 | 只到 `--max-turns 20`，不含 prompt | 只杀掉目标进程，工具调用自身返回 144（第 4.5 节） | 退出码 0，`subtype=success`，5 轮完成 |

- Compose：本地用 `docker compose exec -T agent /bin/sh -c 'printf "%s\n" "$$" > "$1"; shift; exec "$@"' …`（与
  `compose.py` 相同的包装）调用 `_run_hycode_streaming`，容器内 `cat` 读到 300,079 字节，md5 一致，并收到 EOF 正常退出。
- Direct：`su nobody -s /bin/sh -c 'cd … && exec env … <cmd>'` 包装下同样逐字节一致。
- `hycode_dynamicagent`（0.2.6）：默认 harness 没有 Bash 工具，无法复现 `pkill` 场景。修改后 CLI 正常读到 prompt 并作答，
  wrapper 注入的 harness 照常加载。

本地边界验证：用真实的 `opencc-bin` 0.1.10 加一个记录请求的假 Anthropic 服务，对比位置参数和 stdin 两种方式下模型收到的
prompt：

- 两种方式完全一致：首尾空白和换行、CRLF、引号与 shell 元字符、emoji 和 CJK 扩展字符、以 `-` 开头或等于 `--help` 的
  prompt。空 prompt 都报 `Input must be provided…`；只有空白的 prompt 都不发请求、直接退出 0（CLI 既有行为）。
- 以前无法启动、现在正常：含 NUL 字符（以前 `ValueError: embedded null byte`），超过 128KB（以前 `E2BIG`）。
- 编码：prompt 按命令行参数相同的规则（UTF-8 + `surrogateescape`）在启动子进程前编码，`\udcff` 这类字符得到与以前相同的
  字节；无法编码的孤立代理字符（如 `\ud800`）和以前一样在启动前抛 `UnicodeEncodeError`，不会变成空 stdin。
- CLI 读 stdin 的方式（`getInputPrompt` / `peekForStdinData`）：只在收到第一个字节前有 3 秒超时，超时后报
  `Input must be provided…` 并退出 1，不会带着空 prompt 继续跑；收到数据后一直读到 EOF，中途停顿 6 秒也不会截断。
  Nexus 在启动子进程后立即写入，不受这个超时影响。位置参数和 stdin 同时存在时，CLI 会用 `\n` 把两者拼起来。
- Compose：真实 `opencc-bin` 经 `docker compose exec -T` 和 `compose.py` 同样的包装运行，300KB prompt 完整到达，
  没有出现 3 秒超时警告。
- `hycode_dynamicagent` 的 node wrapper 原样转发 stdin，300KB 和以 `--agent` 开头的 prompt 都完整到达。

### 6.3 请求等价性（合入前必须通过）

同一任务、同一模型、temperature 0，分别用修改前后的代码各运行一次，从 gateway / trace 中取第一次 LLM 请求，
对比 `system` 与 `messages[0]` 是否逐字节一致。

这一步同时确认 `opencc-bin`（Claude Code 的 fork）在 `--input-format text` 下确实从 stdin 读取 prompt，且不会
对内容做 trim 之类的改动。

2026-10-20 实测：用 `json_line` processor 记录修改前后各一次 `--max-turns 1` 运行的第一次 LLM 请求。

- HyCode：`system`（26,577 字符）和第一条 user 消息（1,290 字符）逐字节一致，`model` 和其余请求参数也一致。
  prompt 末尾的换行保留，CLI 没有 trim stdin。
- `hycode_dynamicagent`：`system` 一致；user 消息的 3 个 block 除 system-reminder 中的 `Current time` 时间戳外完全一致。

### 6.4 上线后观测

在同一批 pb / nl2product 任务上对比修改前后：

- `opencc-bin` 被信号终止的比例（退出码 `-15` / `143` / `137` 且 stdout 无 `result` 事件）应明显下降。
  统计时以 `opencc-bin` 自身退出码为准，不计入工具调用内部的 143；
- RL 中因误杀判 0 分的样本比例应同步下降。之后评估是否撤掉该惩罚，或只保留给 `pkill python`、`killall node`
  这类大范围清理；
- 去除上述样本后，平均轮数和得分分布应基本不变。若整体出现偏移，回到 6.3 排查。
