# 已知问题

本文记录 HITL 开发中遇到的缺陷，包括已解决和未解决。

---

## 1. 已解决

### AskUserQuestion 的迟到答案拆掉整个 PTY session

worker 通过 hook 提问时可能因为 user 回答慢导致 hook 超时，当等真正的答案送到时，沙箱里的请求文件已经被清理，导致被 pty 认定为致命错误关闭session。目前 nexus 主分支已经上线修复，提供可配置1的超时并避免 hook 超时导致的崩溃。

### PTY 因为管道写满而卡死

PTY 每 15 秒把一份终端快照写到 worker 的 stderr，而 Runtime Server 启动 worker 时把 stdout 和 stderr 重定向到管道，却从不读取。大约一小时后管道被填满，worker 的写操作阻塞，Socket.IO 心跳发不出去，被判定为断连。已在 nexus主分支上线修复，默认关闭周期性快照。

### chat tracker 导出时递归溢出

chat tracker 用递归的方式深度优先遍历消息树，长 running 任务会因为 Python 的递归上限失败。该问题已在 新版 nexus 被修复

### 一条 sio 连接上多个 proxy 抢注 `llm_response`

测试多 claude session 在 host 并发运行时，发现最后一个除外的 session 都无法正常运行。原因在于每个 session 有自己的 proxy，每个 proxy 建自己的 bridge，bridge 在构造时都注册同一个事件名 `llm_response`。所有 proxy 用同一个 `sio`，而 python-socketio 的 `on` 是往字典里赋值，后注册的覆盖掉前面的。只有最后一个 bridge 的 handler 还活着，其他 bridge 的响应到它这里找不到对应的 request_id 被丢掉，那些 bridge 收不到 ack，卡在 5 秒的 ack 超时上每 5 秒重发一次直到 deadline。host 每次收到重发都重新跑一遍模型：`_emit_final` 成功之后会把去重记录删掉。

改法是一条 sio 连接只用一个 bridge，由 `bridge_for` 按连接取，首次使用时才建。单个 proxy 的行为不变。以 3 与 5 个 session 的沙箱 run 验证：重发 0 次（修复前 600+），每个 session 的轨迹只带自己的 label，互相不串。之后又用 swe（print 后端）与 terminal（pty 后端）各 3 条实例回归，两侧都是一条连接挂 3 个 proxy，重发 0 次、label 隔离正常、分数与修复前的基线一致。

### langfuse 上传量按轮次平方增长

上游 tracker 每次调用模型都记录一整份当前的消息列表，所以 N 次调用一共上传 N+(N-1)+…+1 条消息。有一个 run 产出了 1222 MB 轨迹，其中真正的内容只有 14.78 MB。HITL 通过Blake2b 哈代替完整消息列表进行标识，降低占用。

### print_codex 把超时的一轮当成功，整条链空转到停

`run_command` 在 turn timeout 杀掉 codex 进程后不留 exit code，`run_turn` 旧写法把它读成 0（日志原文 `treating as success (0)`），而 `manager._turn_kind` 只看 thread_id 有没有，于是一轮 7200 秒的空转被判成 `completed`，manager 接着发下一轮，再卡 7200 秒。

0921b 那轮实测：一个 worker 从 12:23 到 19:28 循环了四轮，八小时里 `output` 只有 101 字符，并发槽一直占着。三个 gpt56 run 的 `emit dropped` 分别 46897 / 4657 / 15434 次，单个 request 最高重发 5085 次。

改法：`run_turn` 多报一个 `timed_out`（exit code 记 124，`timeout(1)` 的约定），`interpret_turn` 透传，`_turn_kind` 据此判 `terminal`。pty_claude_code 的 `session_died` 本来就是这个语义，两个 backend 现在一致：撞上之后整条链以 `session_lost` 收尾，sandbox 立刻释放，损失从"一个 run 空烧八小时"降到"一条实例 + 一轮"。commit `863c5f0720`，附一个对拍过的回归测试（退回旧写法时报 `assert 'terminal' == 'completed'`）。

只治后果。让它卡住的那一层见下节。

---

## 2. 待解决

### socket.io 重连失败后留下半死的 client，codex 侧高发

**表现。** host 往 sandbox 发 `llm_response` 时抛 `BadNamespaceError: / is not a connected namespace.`，此后这条 session 的每次 emit 都失败。LLM 调用其实已经成功（日志里伴随 `replaying completed result`），卡的是把结果送回 sandbox 这一步。worker 每 5 秒重发一次，host 每次都重放那个已完成的结果，再 emit 失败，不收敛。

**机制。** 四步，每步都有日志佐证：

1. socket 被 transport 层撕掉（`socketio: disconnected … reason=transport`），`_ws_connected` 翻 False。
2. 下次 `invoke()` 触发 `_try_auto_connect_sio`（`runtime/client/remote.py:194`）。这里的 `sio.connect()` **没传 `wait_timeout`**，用库默认 1 秒，而另一条路 `_ensure_sio_connected` 传的是 `initial_connect_ns_wait_timeout=3.0`。
3. 高负载下 1 秒等不到 namespace 握手 → 库内 `asyncio.TimeoutError` 被 `except: pass` 吞掉 → `set(self.namespaces) != set(self.connection_namespaces)` → `await self.disconnect()` 清空 `namespaces` → 抛 `ConnectionError: One or more namespaces failed to connect: `（冒号后为空，`failed_namespaces` 也是空，意思是既没成功也没失败）。这次 `disconnect()` 因为 `self.connected` 已是 False，**不触发 disconnect 事件**。
4. host 侧 `remote.py:600` 的兜底只打一行 `auto-connect failed`，**既没清 `self._sio` 也没翻 `_ws_connected`**，还把 `_sio_auto_connect_disabled` 置 True（这个 client 以后不再尝试重连）。于是 `emit` 处的前置检查 `if not self._ws_connected` 放行，库内的 `namespaces` 检查必然抛错 —— 两个"是否连接"的真相源不同步。

**为什么 codex 高发。** 四步里只有第 1 步两个 backend 不同：

| run | transport 断连 | 占比 | auto-connect 失败 | emit dropped |
| --- | --- | --- | --- | --- |
| codex 80 并发 | 73 | 15% | 51 | 47079 |
| codex 120 并发 | 1775 | 85% | 189 | 4657 |
| CC 60 并发 | 7 | 2% | 0 | 0 |
| CC 80 并发 | 2 | 1% | 0 | 0 |

怀疑是 codex 的请求在 wire 上重得多：`preserve_reasoning` 把缓存的 `reasoning_items`（内含 `encrypted_content`）逐条塞回 assistant 消息，**codex 侧 42345 次、单次最多注入 444 条，CC 侧 0 次** —— Responses 协议要求每轮重放全部加密推理块，anthropic 协议没有这一环。`encrypted_content` 是 base64 blob，不计入 token 却实实在在占 websocket 帧，而且随轮次单调增长。

已排除的两个解释：
- **payload token 数**：`peak_input` 中位数 CC（160k/169k）比 codex（147k/149k）还大。
- **请求频率**：峰值小时 CC 11061 次，codex 5728 次，CC 是两倍。

**未证实的部分。** `websocket_extra_options={'max_msg_size': 100 * 1024 * 1024}` 已经把单帧上限提到 100 MiB，所以"单帧超限"这条路应该已堵上；`reason=transport` 更可能是大帧传输耗时长，被中间某一跳（sandbox 网关 / LB）的空闲或单请求超时掐掉。要确认得量"transport 断连时那条连接正在传的帧有多大"，日志里没记帧大小，需要埋点。

**可能的方向。** 都在 nexus 公共代码里，需要单独讨论：
- `_try_auto_connect_sio` 传 `wait_timeout`，与 `_ensure_sio_connected` 统一，并把默认值提上去（治第 3 步的发生率）。
- `remote.py:600` 那个 `except` 像 `_force_reset_sio_client` 一样把 `_sio = None; _ws_connected = False`，让下次取到干净的新 client（治第 4 步的半死状态）。
- 裁剪 codex 的 reasoning 重放（治第 1 步的根因）。裁错会让 codex 收到 `rs_ not found`，得先看清 `_reasoning_items_cache` 的攒法和协议约束。

**波及面。** `request_timeout=None` 全仓只有两处：`agents/codex/runtime_setup.py` 和 `actor/print_codex/runtime.py`，其余七个 agent（codebuddy / gemini_cli / opencode / cline / openclaw / dsh / claude_code）都传了那一轮的 timeout，撞上之后会在 N 秒后抛 TimeoutError 收手。但它们都是单轮 agent，撞上照样丢掉这条实例的产出，只是不会循环；`codebuddy` / `opencode` / `cline` 还有 `return exit_code or 0` 把超时读成成功，结果里会留一条没产出的"成功"记录。只有 print_codex 同时满足"多轮 + 超时当成功"，才放大成八小时空烧。

### 每个 span 都记完整消息列表，跑得久的任务磁盘占用异常大

nexus 的 trace 在每个 span 上都记录这一轮完整的输入消息，磁盘占用按轮次平方上涨，在 long running 任务下会导致大量磁盘空间的浪费。由于修改触及 nexus 核心模块，可能需要更自己的讨论与决策。

### worker 常用的命令沙箱里没有，manager 没有装它们的途径

gpt-5.6 习惯先试 `rg`。0921b 那批 codex 轨迹里 `rg: command not found` 在缺失命令中排第二（3 条实例 2 次，仅次于 `cargo` 3 次），CC + gpt-5.6 那轮也有。撞上之后两种反应都出现过：一条 fallback 到 `grep -nE` 继续干活，另一条两个 `rg` 都失败后没有换命令重试，直接去写报告，带着搜不到的结论下判断。

`bwrap` / `bubblewrap` 在这批数据里零次出现，没有它缺失的证据。

想做的形状是 manager 提供一个「要哪些系统包」的参数，自己探测 `apt-get` / `yum` / `dnf` 装，调用方只列包名。CC 侧已有 `pre_start_scripts` 能跑任意命令（`print_claude_code/runtime.py:133`，pty 经 `runtime_adapter` 转发），codex 侧没有；但直接用它要调用方自己写包管理器命令，而镜像有 apt 系也有 yum 系。落点是三个 backend 的 `setup_env`——那里已经是一容器一次、带 signature 防重配。

装失败时不应让 session 启动失败：worker 本来能用 `grep` 凑合，这是增强而非依赖。默认不装任何包，装什么由 run 脚本决定。

镜像里 `yum` 能不能用、有没有源，还没验证过——这批镜像是各项目自己的 `t3build_repolaunch_midtrain.*`。要动手前先在一个沙箱里试一次。

另一个要先想清的是评测环境的一致性：镜像里有什么工具是数据集决定的，给所有实例预装 ripgrep 等于替 worker 补了它原本没有的能力，拿这批轨迹做 SFT 时模型学到的是「`rg` 总是可用」。




