# Session manager

SessionManager 将agent harness 包装成可轮驱动的命名 session，并支持以统一的接口进行管理和运行。

---

## 1. Session 的概念

Session 用于标识一个沙箱内的连续 Harness 进程，由 Manager 统一创建和管理。其公共字段如下：

- `name` 调用方在 `start` 时提供的名称。
- `session_id` harness 侧的唯一标识，由 manager 生成。
- `backend` 这个 session 由哪个 manager 驱动。
- `metadata` 调用方自行使用的元信息。

不同 Manager 可以通过创建子类的方法在Session内记录额外的数据与行为。


## 2. 基本接口说明

SessionManager 的抽象基类有以下接口：

- `setup()` 初始化 Session Manager 准备后端共享的资源，每个实例只应该调用一次。
- `start(name=, metadata=)` 启动一个 session，返回 `Session`。
- `send(session, message)` 向指定 Session 发送一轮指令，直到其运行结束，返回 `TurnResult`。
- `run_turns(turns)` 向多个 session 各投一条消息，按完成顺序产出 `BatchTurnResult`。
- `report(result, detail=)` 将 `TurnResult` 渲染为字符串。
- `traces(session)` 导出特定 session 的轨迹
- `session_of(session_id)` 按 harness 侧的 id 反查 session
- `close(session)` 关闭一个 session
- `shutdown()` 关闭 Manager 并释放所有共享资源。

Session Manager 需要声明基本类属性：

- `backend` Manager 的唯一标识符，如 `pty_claude_code` 与 `print_claude_code`。
- `product` Harness 的产品名称，用于呈现给外部使用者，如 `Claude Code`。
- `memory_path` 这个 harness 需要的 ‘AGENTS.md/Claude.md' 等偏好文件的路径。
- `skills_path` 这个 harness 读 skill 的目录，见 §8。
- `command_prefix` 与 `builtin_commands` 这个 harness 支持的 slash command，例如 Claude Code 的'/verify'（内置 Skill）。

---

## 3. send() 语义

一次 `send` 表示向 harness 发送一条指令，并等待跑到它认为这一轮结束，最后返回一个 `TurnResult`。向一个 session 的重复 send 即代表进行多轮交互。

`TurnResult` 是 Manager 的标准化返回，包含以下字段：

- `kind` 本轮结束的状态，用于判断下一轮的语义。
  * `completed` 本轮正常完成。
  * `needs_input` 本轮阻塞在提问上，下一次 `send` 将作为该提问的答案投递（例如触发 Claude 的 AskUserQuestion。
  * `terminal` Session 因为内部错误崩溃，无法承接下一轮。
  * `refused` 消息因为某种原因被拒绝，并且 session 状态未变，可以重发新的消息（例如，发送了不支持的slash command）。
- `output` harness 本轮输出的文本。
- `user_input_request` worker 的待答提问，没有提问时为 `None`。
- `session_id` 这一轮跑在哪个 harness session 上。
- `raw` 后端自己的原始材料，由 `report` 读取并渲染。
- `metadata` 这一轮的附加信息，如轮次序号与 session 名字。

**send 的特殊返回状态**

`metadata` 里包含一个额外 `status`字段，用于标识 session 的两种特殊状态：
- `runtime_unreachable` 进因为沙箱的问题，调用无法返回。
- `answer_missed` 答案送到时 worker 已经因为超时自行回答并继续，对应的output会被特殊提醒。

---

## 4. Session 与 label

每个 session 在 `start` 时会按分配到名称自动生成 label，从而在 HITL 轨迹、用量与 langfuse 三个 tracker中实现隔离。label 的取值形式与传递路径见 [design_label.md](design_label.md)。

---

## 5. 结果渲染

各后端根据自己拿到的信息粒度，将 worker 当轮动结果渲染为 xml 文本块，起主要包括以下部分：

- `<worker_progress>` 本轮的推理过程和工具调用，会截断过长的工具调用参数与推理过程。
- `<final_response>` worker 的最后一轮回答。
- `<worker_question>` worker 如果提问，代替 final response 出现。
- `<note>` 当 worker 当轮未正常结束时用于包裹说明。

---

## 6. 并发控制

HITL 支持多个 session 并发运行。`run_turns(turns)` 接受一批 `(session, message)`，按完成顺序产出 `BatchTurnResult`，它在 `TurnResult` 基础上拥有额外的字段：

- `session` 本轮运行的 session 。
- `inflight`批次内仍在执行的 session 名字。

`BatchTurnResult` 以 async generator 的形式产出，允许在某个turn完成后立刻发送下一条新的指令。


```python
async for turn in manager.run_turns([(a, msg_a), (b, msg_b)]):
    print(manager.report(turn.result))
    if turn.kind == 'completed':
        await manager.send(turn.session, next_message)
```

并发要求同一连接上的多个 proxy 共用一个 LLM bridge。每个 session 有自己的 proxy，而 proxy 共用运行时那一条 Socket.IO 连接；bridge 构造时注册 `llm_response`，而 python-socketio 按 `handlers[namespace][event]` 存放处理器，是赋值而非追加。因此每 proxy 各建一个 bridge 时只有最后创建的能收到响应，其余收不到 ack、每 5 秒重发直到超时，且 host 对每次重发都重新调用模型。`anthropic_proxy.server.bridge_for` 按连接返回同一个 bridge，消除重复注册；bridge 不含按调用方区分的状态，label 也仍按 session 隔离（它包裹每个 proxy 各自的 bridge 引用）。

---

## 7. 实现新的后端

为了实现新的后端，需要创建实现 §2 的接口与类属性的 Manager 类型并注册，然后在工厂函数内导入。

```python
@worker_manager_registry.register('my_backend')
class MyManager(SessionManager):
    backend = 'my_backend'          # 非空，注册时校验
    product = 'My Agent'            # 非空，注册时校验
    memory_path = '.myagent/AGENTS.md'
    skills_path = '.myagent/skills'
    command_prefix = '/'

    def __init__(self, *, llm_handler, ...):   # 接受注入，注册时校验
        ...
```

session 内部的 handler 需要从 `actor.resources.acquire_llm_handler` 获得保证单例，否则会导致 llm_reqiest 的广播事件被服务两次。`memory_path` 与 `skills_path` 声明 harnesss 默认读取偏好文件与 skill 目录的位置，由 manager 负责将 配置中 host skill 与 memory 传入沙箱。

**新后端若要支持并发 session**，其沙箱侧 proxy 必须复用连接上已有的 bridge（见 §6）。`AnthropicProxy` 系（含 `PtyAnthropicProxy`）已经如此；`OpenAIProxy`、`ResponsesProxy`、`GeminiProxy` 仍各自创建 bridge，基于它们的后端只能单 session 运行，或先照同样方式改造其 proxy。改造 `OpenAIProxy` 时注意它在 `start()` 而非应用工厂里创建 bridge，且其 `stop()` 会清空 `self.bridge`——共用后不可在停止路径上清理 bridge 本身，那会影响同一连接上仍在运行的其他 session。

---

## 8. Skill、slash command 与 MCP server

HITL 的 Session Manager 支持沙箱内 worker 的 Skill、slash command 与 MCP server，相关配置统一在 Manager 构造时通过配置文件传入。

### Skill

`worker_config` 用以下两个字段声明这次 run 给 worker 装哪些 skill：

- `worker_skills` 本地 skill 目录的路径列表。
- `worker_commands` 这些 skill 的名字与一句说明。

`utils.generator_util` 提供 helper `with_worker_skills(worker_config, skill_dirs)`，通过声明本地的 skill 目录可将其自动传入 worker。skill 会作为 worker 支持的 command 被调用，可以据此将其暴露给 user 模型；同时 worker 也可以主动调用 skill。

### slash command

Manager 用以下类属性声明它的 harness 接受哪种命令形式：

- `command_prefix` 一条消息的首个 token ，将以它开头的字符按命令进行解析和处理，对 Claude Code 的来说就是 `/`。
- `builtin_commands` harness 自带的命令，例如 Claude Code 的 `simplify`。取值以实测为准：harness 不带的名字会被它本地应答 `Unknown command` 并结束该轮，表现为一轮什么都没做却正常完成。

Manager 最终提供 `commands` ，包含所有内置命令与可能安装的skill，用于向 user 模型提供调用 worker的对应能力的上下文。当输入一prefix 开头但不是合法命令时，send会被立刻拒绝，避免在沙箱里产生未定义行为。

### MCP server

**mcp server** 只经过了基本测试，依旧需要更多的验证才可稳定使用

`worker_config` 允许向 worker 声明可挂载的 MCP server，通过 `mcp_servers`字段按 `claude mcp add` 的形式给出，包含`name`、`command` 与 `args`。


---

## 9. 当前支持

当前 HITL 支持以下后端

### print_claude_code

基于 `claude --print` 按 session id resume 上一轮的对话的 claude code 。worker 的提问以答案文件的形式交回，下一轮 resume 时由 Claude Code 读取。已在 HITL v1 上验证，需要确定性的 run 优先使用。


### pty_claude_code

基于常驻的交互式 PTY claude code运行，整个 session 共用一个常驻进程。它驱动真实的 TUI，因而覆盖了 Claude Code 的交互专属分支，worker 也能在一轮中途通过 hook 向 user 提问、等待答复。依旧需要进一步验证在大并发和超长任务下的稳定性。


### print_codex

基于 `codex exec`，每轮起一个进程，按 thread id 续上一轮。走 Responses 协议而非 Anthropic，自带一份 model catalog，并按 provider 名字决定请求形状，与两个 Claude Code 后端的约定差别较大，单独记在 [design_codex.md](design_codex.md)。`CODEX_HOME` 是全容器共享的一份，每轮以 `-c` 下发本 session 的 proxy 端口。


